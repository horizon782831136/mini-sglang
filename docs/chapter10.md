# 第 10 章：端到端性能分析与扩展方向

> **本章目标**：带领读者从整体视角审视 mini-sglang 的性能瓶颈，掌握实用的分析工具，并在九章学习的基础上，看清系统下一步可以走向何方。

---

## 10.1 背景问题

经过前九章的学习，我们已经理解了 mini-sglang 的全貌：分页 KV 缓存、连续批处理、Radix 前缀复用、Tensor Parallelism、CUDA Graph 加速……这些技术各自解决了推理系统的某一个局部问题。

但是，当我们将这一切拼装在一起，真正跑起来之后，却发现了一个令人沮丧的规律：**单独优化每个模块，并不等于端到端的最优**。

这是深度学习系统优化中永恒的核心矛盾：

- GPU 算力在等 CPU 组装 batch、上传数据；
- CPU 在等 GPU 完成上一个 batch 的采样，才能处理响应；
- 通信带宽（NCCL all_reduce）在 forward 最深处阻塞所有 TP 进程；
- 多路并发请求的延迟分布呈长尾，P99 远高于 P50。

本章的核心任务，就是教你**定位这些瓶颈在哪里、如何量化、如何修复或绕过**，并在此基础上展望四个有价值的扩展方向。

---

## 10.2 核心概念讲解

### 10.2.1 Overlap Scheduling：让 CPU 和 GPU 真正并行

想象一条流水线工厂：GPU 是机器，CPU 是工人。如果工人每次都要等机器干完，才能准备下一批原料，机器就会空转等待；反之亦然。

mini-sglang 的 **Overlap Scheduling** 设计，正是流水线思想的直接体现。

```
时间轴：
             批次 N                批次 N+1
GPU  |████████ forward ████████|████████ forward ████████|
CPU  |←调度/收结果→|←调度/收结果→|
```

**普通模式**（`MINISGL_DISABLE_OVERLAP_SCHEDULING=1`）：CPU 调度 → GPU forward → CPU 收结果，串行执行，GPU 有空闲。

**Overlap 模式**：当 GPU 在执行批次 N 的 forward 时，CPU 同时处理批次 N-1 的采样结果、更新 token pool、调度批次 N+1。这样 GPU 几乎不需要等待 CPU。

关键在于 mini-sglang 使用了**两个 CUDA Stream**：

- `engine.stream`：专供 GPU forward 计算。
- `scheduler.stream`（默认当前流）：供 CPU 侧的 metadata 操作（pin_memory 拷贝、位置索引构造等）使用。

两条流可以并发，但需要在关键点手动同步，防止数据竞争。

### 10.2.2 NCCL 通信原语在 TP 中的位置

Tensor Parallelism 的本质是把权重矩阵按列或按行切分，分发到多张 GPU。每张卡只持有部分权重，前向传播需要在特定时刻聚合各卡的局部结果。

两种通信原语对应两种切分策略：

| 通信原语 | 触发时机 | 作用 |
|---|---|---|
| `all_reduce` (SUM) | MLP/Attention 的行并行输出 | 各卡局部和 → 全局和，结果广播回所有卡 |
| `all_gather` | Embedding 层、词表并行输出 | 各卡持有不同行 → 拼成完整张量 |

mini-sglang 在 `distributed/impl.py` 中提供了两套实现：基于 `torch.distributed` 的同步实现（`TorchDistributedImpl`）和基于自定义 PyNCCL 的异步实现（`PyNCCLDistributedImpl`）。后者绕过了 PyTorch 对 NCCL 流的控制，可以和 CUDA Graph 更好地配合，同时支持通过 `MINISGL_PYNCCL_MAX_BUFFER_SIZE` 控制通信缓冲区大小。

### 10.2.3 离线吞吐 vs 在线延迟：两个不同的度量世界

这两个指标经常被混淆，但它们衡量的根本上是不同的目标：

- **离线吞吐（Throughput）**：一次性提交所有请求，等全部完成，用总 token 数除以总时间。衡量的是系统的峰值产出能力，对应 `benchmark/offline/bench.py`。
- **在线延迟（Latency）**：请求以泊松流或固定速率到达，测量每个请求的 TTFT（首 token 时间）和 TPOT（每 token 时间）。对应 `benchmark/online/bench_simple.py`，通过 OpenAI 兼容 API 发送并发请求，收集每个 token 的时间戳（`tics`）。

两者之间存在本质的**吞吐-延迟权衡**：增大 batch size 能提升吞吐，但会增加等待时间，导致延迟上升。系统调优时必须根据业务场景选择合适的目标指标。

---

## 10.3 核心代码导读

### 10.3.1 Overlap Scheduling 的双流实现

文件：`python/minisgl/scheduler/scheduler.py`

```python
# 第 52-55 行：两条 CUDA Stream 的初始化
self.stream = torch.cuda.Stream(device=self.device)        # scheduler 流
self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)  # engine 流上下文
torch.cuda.set_stream(self.stream)                          # 当前流设为 scheduler 流
```

`run_forever` 方法（第 121-131 行）是整个系统的主循环入口。当 `MINISGL_DISABLE_OVERLAP_SCHEDULING` 为假时，进入 `overlap_loop`：

```python
def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
    # 第 101-103 行：在 engine 流上启动本批次的 GPU forward
    with self.engine_stream_ctx:
        self.engine.stream.wait_stream(self.stream)      # 等 scheduler 流准备完毕
        ongoing_data = (forward_input, self._forward(forward_input))

    # 第 105 行：在 scheduler 流上处理上一批次的 CPU 结果（与本批次 forward 并发）
    self._process_last_data(last_data)
    return ongoing_data
```

`engine.stream.wait_stream(self.stream)` 是关键同步点——它确保 CPU 在 scheduler 流上构造好 batch metadata（索引、位置等）后，GPU 才开始消费这些数据。但这条等待本身是**非阻塞的**：它只是在 GPU 端插入一个依赖事件，CPU 立刻继续执行 `_process_last_data`。

注意 `_forward` 方法（第 230-231 行）中还有一个可选的额外同步：

```python
if ENV.OVERLAP_EXTRA_SYNC:   # MINISGL_OVERLAP_EXTRA_SYNC=1
    self.stream.synchronize()
```

这个开关用于修复某些驱动版本下流间数据竞争的问题（见 issue #58），代价是牺牲部分重叠效果。

### 10.3.2 PyNCCL 初始化与 all_reduce/all_gather

文件：`python/minisgl/kernel/pynccl.py`（第 45-78 行）

PyNCCL 初始化时，rank 0 生成唯一的 NCCL communicator ID（`create_nccl_uid`），然后通过 CPU 侧的 `torch.distributed.broadcast_object_list` 广播给其他 rank。这是一个"用 CPU 通信引导 GPU 通信"的典型模式。

文件：`python/minisgl/distributed/impl.py`（第 44-60 行）

```python
class PyNCCLDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")   # 原地 all_reduce，结果覆盖 x
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        output_shape = list(x.shape)
        output_shape[0] *= world_size    # 沿 batch 维拼接
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result
```

`DistributedCommunicator.plugins` 是一个插件列表（第 64 行），始终调用 `plugins[-1]`。当 `enable_pynccl_distributed` 被调用时，PyNCCL 实现被追加到列表末尾，自动生效，无需修改模型代码。

### 10.3.3 Triton Fused MoE Kernel

文件：`python/minisgl/kernel/moe_impl.py` + `python/minisgl/kernel/triton/fused_moe.py`

Fused MoE Kernel 的核心思想是：将专家路由（token 到 expert 的分配）和专家 GEMM 融合进单个 Triton kernel，避免中间结果落回显存。

`fused_moe_kernel_triton` 函数（`moe_impl.py` 第 6-62 行）接受已排序的 token ID 和 expert ID，通过动态 grid 调度：

```python
grid = lambda META: (
    triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
    * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
)
```

每个 tile 处理 `BLOCK_SIZE_M` 个 token 与 `BLOCK_SIZE_N` 个输出维度的乘积，K 维的对齐情况（`even_Ks`）影响是否需要边界检查，从而控制内层循环的效率。

`moe_sum_reduce_triton` 函数（第 65-98 行）则是一个独立的规约 kernel，将 top-k 条路由的输出按权重加和，回到 `[token_num, hidden_dim]` 的形状。两阶段分开是为了让 tile 形状可以独立调优。

### 10.3.4 采样器与 NVTX 标注

文件：`python/minisgl/engine/sample.py`（第 70-75 行）

```python
@nvtx_annotate("Sampler")
def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
    with torch.cuda.nvtx.range("Sampler"):
        if args.temperatures is None:   # greedy：直接 argmax
            return torch.argmax(logits, dim=-1)
        return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
```

`nvtx_annotate` 装饰器（来自 `minisgl.utils`）在 Nsight Systems 中会显示为彩色区间。`make_device_tensor`（第 20-21 行）使用 `pin_memory=True` + `non_blocking=True` 的组合，让采样参数的 CPU→GPU 拷贝在后台流上异步完成，不阻塞主流。

---

## 10.4 设计决策

### 决策一：为什么选择 PyNCCL 而非直接用 torch.distributed？

`torch.distributed` 的 NCCL 后端会将通信操作提交到 PyTorch 内部管理的 NCCL 流，该流与用户自定义的 CUDA Stream 的同步关系不透明。当配合 CUDA Graph 使用时，Graph capture 阶段需要精确控制哪些 CUDA 操作被记录，PyTorch NCCL 流的隐式行为容易导致 Graph replay 时的通信错位。

自定义 PyNCCL 通过 TVM FFI 直接封装底层 NCCL C API，通信操作在调用方指定的 CUDA Stream 上执行，行为完全可预测。代价是需要维护额外的 C++/CUDA 代码（`kernel/csrc/`）和 TVM FFI 绑定。

**替代方案**：vLLM 早期版本使用 `torch.distributed` + 显式流同步来规避这一问题，但在 CUDA Graph 下需要特殊处理。

### 决策二：Overlap Scheduling 为什么不默认开启旧版同步？

`MINISGL_OVERLAP_EXTRA_SYNC` 开关（`env.py` 第 70 行，默认 False）的存在说明：流间重叠本质上是一种"乐观并发"——在绝大多数硬件和驱动版本下是安全的，但某些场景下存在竞争条件。设计选择是**默认走高性能路径，提供逃生通道**，而不是为了安全牺牲所有场景的性能。

### 决策三：离线 bench 为什么要先 warm up flashinfer？

`bench.py` 第 32 行：`llm.generate(["Benchmark: "], ...)` 是一个预热步骤。flashinfer 的 JIT 编译、CUDA Graph capture、以及 PyNCCL 初始化都发生在第一次推理时。若不预热，计时结果会包含大量一次性开销，导致吞吐数据失真。这是一个被许多论文忽视但至关重要的测量细节。

---

## 10.5 动手实验

### 实验一：测量 Overlap Scheduling 的实际收益

**目的**：量化 Overlap Scheduling 对吞吐的提升幅度。

```bash
# 基准测试：禁用 Overlap Scheduling
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl.benchmark.offline.bench

# 对照测试：启用 Overlap Scheduling（默认）
python -m minisgl.benchmark.offline.bench
```

观察两次输出的 `Throughput` 数值差异。在 decode-bound 场景（长输出）下，收益通常在 5%~15% 之间；在 prefill-bound 场景（长输入短输出）下，收益较小。

**进阶**：修改 `bench.py` 中 `max_input_len` 和 `max_output_len` 的比例，观察 Overlap 收益如何随 prefill/decode 比例变化。

### 实验二：用 torch.profiler 定位前向传播瓶颈

**目的**：找出 forward 中哪个算子最耗时。

```python
import torch
from minisgl.llm import LLM
from minisgl.core import SamplingParams

llm = LLM("Qwen/Qwen3-0.6B")
# 先预热
llm.generate(["warm up"], SamplingParams(max_tokens=10))

# 开启 profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=False,
) as prof:
    llm.generate(
        ["分析一下大模型推理系统的设计要点。"] * 8,
        SamplingParams(temperature=0.6, max_tokens=128),
    )

# 输出耗时最长的 CUDA kernel
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
# 导出为 Chrome Trace 格式供可视化
prof.export_chrome_trace("/tmp/mini_sglang_trace.json")
```

在 `chrome://tracing` 或 Perfetto UI 中打开 `/tmp/mini_sglang_trace.json`，可以直观看到 attention kernel、MoE kernel、all_reduce 通信各占多少比例。

### 实验三：用 NVTX 在 Nsight Systems 中追踪调度延迟

**目的**：观察 CPU 调度开销与 GPU forward 的重叠情况。

安装 Nsight Systems 后，通过以下命令采集 profile：

```bash
nsys profile --trace=cuda,nvtx,osrt \
  -o /tmp/mini_sglang_profile \
  python -m minisgl.benchmark.offline.bench
```

`sample.py` 中的 `@nvtx_annotate("Sampler")` 装饰器以及各处 `torch.cuda.nvtx.range()` 调用，会在时间轴上以彩色区间显示。重点观察：

1. `Sampler` 区间是否与下一批次的 forward 有时间重叠（表明 Overlap Scheduling 生效）。
2. `all_reduce` 通信在 forward 时间轴中占据的比例（表明 TP 通信开销）。

### 实验四（进阶）：测量 PyNCCL vs torch.distributed 的通信延迟

将 `engine.py` 中 `_init_communication`（第 112 行）的逻辑修改为强制使用 `torch.distributed` NCCL 后端（设置 `use_pynccl=False`），在双卡 TP 配置下重新运行基准测试，对比通信延迟和总体吞吐。注意：此实验需要 2 张 GPU。

---

## 10.6 可能的扩展方向

### 方向一：添加 Beam Search 采样策略

**现状**：`sample.py` 支持 greedy、top-k、top-p 三种策略，均为单路采样。

**扩展思路**：Beam Search 需要为每个请求维护 B 条候选序列（beam），每步从每条 beam 的 top-B logits 中选取。这要求：
1. `Req` 数据结构增加 `beam_width` 字段和多条 KV 序列的引用。
2. `BatchSamplingArgs` 增加 beam scores 张量。
3. `sample_impl` 中添加 `beam_search_from_probs`，使用 flashinfer 或自定义 Triton kernel 实现。
4. Scheduler 在 decode 阶段需要处理"一个逻辑请求 → B 个物理序列"的映射关系，可以参考 `DecodeManager` 的接口进行扩展。

**难点**：KV Cache 在 beam 之间的共享（beam 收束时释放）和扩展（beam 扩张时复制）会显著增加 Cache Manager 的复杂度。

### 方向二：支持 Speculative Decoding

**现状**：每个 decode step 运行一次完整的大模型前向，硬件利用率偏低（decode 阶段 GPU 算力严重不足）。

**扩展思路**：草稿-验证（Draft-Verify）架构：
1. 引入一个小型草稿模型（Draft Model），每步并行生成 K 个 token 候选。
2. 大模型一次前向验证所有 K 个 token（prefill 语义）。
3. 根据接受率（Acceptance Rate）剪枝，接受的 token 批量写入 token pool。

关键修改点：`Scheduler._schedule_next_batch` 需要区分"验证批次"和"普通批次"，`Engine.forward_batch` 需要支持可变长的 speculative token 序列。

### 方向三：支持 MLA（Multi-head Latent Attention）

**现状**：KV Cache 按 `[num_kv_heads, head_dim]` 格式存储，每层完整保存 K 和 V。

**扩展思路**：MLA（来自 DeepSeek-V2）将 KV 压缩为低秩潜变量 `c_kv`，只缓存压缩向量而非完整 K/V，可将 KV Cache 大小压缩 5~10 倍。实现需要：
1. 在 `kvcache/` 中添加 `MLAKVCachePool`，存储形状改为 `[num_pages, page_size, kv_lora_rank]`。
2. 在 `attention/` 中添加 `MLAAttnBackend`，在 attention 计算时动态解压 KV。
3. 在 `models/` 中支持 DeepSeek-V2/V3 的权重格式和 RoPE 变体（`yarn`、`longrope`）。

### 方向四：添加 LoRA 推理支持

**现状**：模型权重加载后固定，不支持多 LoRA adapter 切换。

**扩展思路**：S-LoRA 架构可以在单次 forward 中对不同请求应用不同的 LoRA adapter：
1. 在 `models/` 的线性层（`nn.Linear`）封装一个 `LoRALinear`，持有多个 adapter 的 A/B 矩阵（按 adapter ID 索引）。
2. `Req` 增加 `lora_id` 字段，`BatchSamplingArgs` 传递 per-request 的 adapter 索引。
3. 使用 Triton 实现 Batched LoRA GEMM：在一次 kernel 调用中，按 token-adapter 映射批量计算 `x @ A[i] @ B[i]`，避免循环调用和同步。
4. LoRA 权重可以在 CPU 上预取、换入换出（参考 `layers/embedding.py` 中的权重加载模式）。

---

## 10.7 小结

### 本章要点回顾

1. **Overlap Scheduling** 通过双 CUDA Stream 将 CPU 调度开销隐藏在 GPU forward 计算时间之内，是提升 decode 吞吐的关键技术。核心代码在 `scheduler/scheduler.py` 的 `overlap_loop` 方法，由 `MINISGL_DISABLE_OVERLAP_SCHEDULING` 环境变量控制开关。

2. **PyNCCL vs torch.distributed**：自定义 PyNCCL 实现提供了对 NCCL 通信流的精确控制，是 CUDA Graph 与 TP 通信兼容的基础。`all_reduce` 用于行并行 GEMM 的输出聚合，`all_gather` 用于词表/Embedding 的分片汇集。

3. **性能分析工具链**：`torch.profiler` → Chrome Trace / Perfetto 定位 GPU kernel 耗时；`nvtx_annotate` + Nsight Systems 追踪 CPU/GPU 协作时序；`MINISGL_OVERLAP_EXTRA_SYNC` 环境变量用于排查流间竞争问题。

4. **基准测试方法**：离线吞吐用 `benchmark/offline/bench.py`，必须预热后再计时；在线延迟用 `benchmark/online/bench_simple.py`，通过 OpenAI API 并发发送，统计 P50/P90/P99 延迟。两者度量不同维度，不能互相替代。

5. **Fused MoE Kernel**：Triton 实现的 `fused_moe_kernel` 将专家路由与 GEMM 融合为单次 kernel，`moe_sum_reduce_kernel` 在第二阶段完成 top-k 加权聚合。两个 kernel 的 tile 参数（`BLOCK_SIZE_M/N/K`）可独立调优。

### 课程总结

至此，我们完成了 mini-sglang 全部十章的学习旅程。

回顾这段路程：我们从最基础的 **KV Cache 分页管理**（第 1-2 章）出发，建立了"内存是第一约束"的认知；通过 **Radix Tree 前缀复用**（第 3 章）和 **Continuous Batching**（第 4 章），理解了调度如何在吞吐与公平性之间取得平衡；**Tensor Parallelism**（第 5-6 章）让我们看到分布式推理的通信代价与工程权衡；**Attention 算子优化**（第 7 章）和 **CUDA Graph**（第 8 章）深入到 GPU 计算的微架构层面；**MoE 路由与 Fused Kernel**（第 9 章）展示了稀疏模型的特殊挑战；而本章则将所有模块串联，教你如何测量、分析，并看到前方更远的路。

mini-sglang 的设计哲学始终是：**保持清晰，追求极致**。每一个抽象都有其必要性，每一个性能优化都有可量化的代价与收益。这种思维方式，比任何具体的代码技巧都更为重要。

希望这份课程讲义，能成为你继续探索大规模推理系统的起点。

---

*文件索引（本章涉及的关键源文件）：*
- `python/minisgl/scheduler/scheduler.py` — Overlap Scheduling 主循环
- `python/minisgl/distributed/impl.py` — all_reduce / all_gather 实现
- `python/minisgl/kernel/pynccl.py` — PyNCCL 初始化
- `python/minisgl/kernel/moe_impl.py` — Fused MoE Kernel 调用
- `python/minisgl/kernel/triton/fused_moe.py` — Triton kernel 实现
- `python/minisgl/engine/sample.py` — 采样器与 NVTX 标注
- `python/minisgl/env.py` — 全局环境变量（`DISABLE_OVERLAP_SCHEDULING`、`OVERLAP_EXTRA_SYNC`、`PYNCCL_MAX_BUFFER_SIZE`）
- `benchmark/offline/bench.py` — 离线吞吐基准测试
- `benchmark/offline/bench_wildchat.py` — WildChat 真实场景基准测试
- `benchmark/online/bench_simple.py` — 在线延迟基准测试
