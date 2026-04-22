# 第 7 章：注意力后端——FlashAttention 背后的工程选择

## 7.1 背景问题：标准 Attention 的内存困境

在深度学习课程里，Scaled Dot-Product Attention 的公式看起来简洁优雅：

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V
```

但当你真正在 GPU 上执行这个计算时，会遇到一个严重的工程问题：**中间注意力矩阵 `Q·Kᵀ` 的显存占用是序列长度的平方级别**。

以一个具体例子说明：假设批次中有一条序列长度为 4096 的请求，使用 32 个注意力头，数据类型为 float16（2 字节）：

- 注意力矩阵尺寸：`[32, 4096, 4096]`
- 显存占用：`32 × 4096 × 4096 × 2 ≈ 1GB`

这仅仅是**一个 Transformer 层**在**一条请求**上的中间结果。现代 LLM 动辄 32~96 层，批次中又有几十条请求并发处理，中间矩阵的显存占用会完全压垮 GPU。

更深的问题在于**内存带宽**。GPU 计算核心（CUDA Core / Tensor Core）的算力增长远快于 HBM 显存带宽。读写这些中间矩阵成为了整个前向传播中最大的瓶颈——不是算力不够，而是数据搬运跟不上。

这就是本章要解决的核心矛盾：**如何在不显式实例化 O(n²) 中间矩阵的前提下，正确计算 Attention？** 答案是 FlashAttention，而 mini-sglang 在此基础上又进一步针对推理场景做了分阶段后端优化。

---

## 7.2 核心概念讲解

### 7.2.1 FlashAttention：分块计算，在线 softmax

FlashAttention 的核心思想是**分块（tiling）计算**，并利用 GPU 的片上 SRAM（即 shared memory，远快于 HBM）完成计算，尽量减少对 HBM 的读写次数。

把 Q、K、V 矩阵切成若干小块，逐块加载到 SRAM 上做局部矩阵乘法。挑战在于 softmax 需要看到整行才能归一化——这里引入了**在线 softmax（online softmax）** 技巧：

维护两个统计量 `m`（当前块的最大值）和 `l`（当前块的归一化因子），每处理一个新块时按如下公式滚动更新：

```
m_new = max(m_old, m_block)
l_new = exp(m_old - m_new) · l_old + sum(exp(scores_block - m_new))
O_new = (exp(m_old - m_new) · l_old · O_old + exp(scores_block - m_new) · V_block) / l_new
```

遍历完所有 K/V 块后，得到的 `O_new` 与全矩阵一次性 softmax 的结果完全等价，但全程**不需要将完整的 n×n 矩阵写回 HBM**，显存占用从 O(n²) 降为 O(n)。

FlashAttention v2、v3 在此基础上继续优化了线程分工、寄存器使用和流水线，Hopper 架构（H100）上的 FA3 还利用了 Tensor Memory Accelerator（TMA）进一步提升带宽利用率。

### 7.2.2 Prefill 与 Decode 的本质差异

理解为什么需要两个后端，首先要理解推理的两个阶段具有**截然不同的计算形态**：

| 特征 | Prefill（预填充）| Decode（解码）|
|------|-----------------|--------------|
| Q 形状 | `[总 token 数, H, d]`，序列长 | `[batch_size, H, d]`，每序列 1 个 token |
| KV 访问 | 同批次内整个 prompt | 每序列独立查询历史 KV cache |
| 计算强度 | 高（矩阵×矩阵）| 低（矩阵×向量）|
| 内存访问模式 | 顺序、规则 | 随机、分散（分页 KV）|

Prefill 是**计算密集型（compute-bound）**，一次处理完整的 prompt，Q 和 K 都有完整序列长度，矩阵乘法的 arithmetic intensity 很高，FlashAttention（FA）的分块策略完全契合。

Decode 是**内存访问密集型（memory-bound）**，每步只生成一个 token，Q 的长度为 1，但需要从 KV cache 中读取整条序列的历史 K 和 V。此时矩阵规模极小，FlashAttention 反而会有较大的 overhead，而 FlashInfer（FI）为这种**短 Q + 分页 KV** 的场景做了专门的 CUDA kernel 优化。

### 7.2.3 分页 KV Cache 与索引结构

mini-sglang 将 KV Cache 组织为固定大小的**物理页（page）**，每条请求通过一张**页表（page_table）** 管理其持有的物理页号。这类似于操作系统的虚拟内存管理。

注意力后端计算时，需要通过页表将逻辑序列位置映射到物理存储位置，因此传入 kernel 的不再是连续内存指针，而是一组索引（indices/page_table）。两个后端在这一点上有不同的实现方式，这正是后续代码导读的重点。

---

## 7.3 核心代码导读

### 7.3.1 接口协议：`BaseAttnBackend`

**文件**：`python/minisgl/attention/base.py`

```python
# base.py 第 18-34 行
class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(self, q, k, v, layer_id, batch) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...
```

每个注意力后端必须实现这 5 个方法：

- `prepare_metadata`：在调度器准备好一个 batch 后调用，计算并缓存当前 batch 的元数据（序列长度、累积偏移量、页表等），避免在 `forward` 中重复计算；
- `forward`：实际执行注意力计算的入口，接收 Q/K/V 张量和 layer_id，返回注意力输出；
- `init_capture_graph` / `prepare_for_capture` / `prepare_for_replay`：支持 CUDA Graph 捕获与回放（第 9 章详述）。

注意 `forward` 签名中的 `layer_id`：每一层的 KV Cache 是独立存储的，后端需要通过 layer_id 取出正确的缓存分区。

### 7.3.2 模型层如何调用后端

**文件**：`python/minisgl/layers/attention.py`，第 56 行

```python
o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
```

这一行是整个模型与注意力后端之间的唯一接触点。`AttentionLayer` 不知道底层用的是 FA 还是 FlashInfer，它只调用 `ctx.attn_backend.forward`，后端由 `Context` 对象在引擎初始化时注入。这是依赖注入（Dependency Injection）的典型应用。

### 7.3.3 FlashAttention 后端：`fa.py`

**文件**：`python/minisgl/attention/fa.py`

`FlashAttentionBackend` 的 `prepare_metadata` 方法（第 67-105 行）做了三件重要的事：

**1. 计算累积序列长度（cu_seqlens）**

```python
# fa.py 第 81-90 行
cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

if max_seqlen_q == 1:  # decode
    cu_seqlens_q = torch.arange(0, padded_size + 1, ...)
elif all(l == 0 for l in cached_lens):  # pure prefill, no cache hit
    cu_seqlens_q = cu_seqlens_k
else:  # extend prefill with partial cache hit
    cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
```

`cu_seqlens` 是 FlashAttention 用于处理**变长序列批次（variable-length batch）** 的关键数据结构。例如批次中有序列长度为 `[10, 5, 8]` 的三条请求，对应的 `cu_seqlens_k` 就是 `[0, 10, 15, 23]`，kernel 通过相邻元素之差恢复各序列的边界。

代码中三种情况的区分处理是 extend prefill 语义的体现：Q 只包含尚未计算过的新 token（`seqlens_q = extend_len`），K 包含整条序列（`seqlens_k = device_len`）。

**2. 页表适配（page table remapping）**

```python
# fa.py 第 92-97 行
new_page_table = torch.stack(
    [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
)
if self.page_size > 1:
    new_page_table.div_(self.page_size, rounding_mode="floor")
```

全局页表以 `page_size=1`（token 粒度）存储物理位置，但 FA kernel 期望的页表以**页粒度**为单位。这里用步长采样（`:: self.page_size`）配合整除，将 token 级偏移量转换为页号，完成格式适配。

**3. 硬件版本选择**

```python
# fa.py 第 46 行
self.version = 4 if is_sm100_supported() else 3
```

sm100 对应 Blackwell 架构（B200/B100），使用 FA4；其余使用 FA3。版本通过 `ver` 参数透传给底层的 `sgl_kernel.flash_attn.flash_attn_with_kvcache`。

### 7.3.4 FlashInfer 后端：`fi.py`

**文件**：`python/minisgl/attention/fi.py`

FlashInfer 后端在初始化时分别创建了 prefill 和 decode 两个 wrapper（第 93-103 行）：

```python
self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
    self.float_workspace_buffer, kv_layout="NHD", backend="fa2"
)
self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
    self.float_workspace_buffer, use_tensor_cores=self.use_tensor_cores, ...
)
```

两者共用同一块 128MB 的 float workspace buffer，通过共享 `int_workspace_buffer` 进一步减少显存占用（第 106-107 行）。

**FlashInfer 的 plan/run 两阶段 API** 是其设计的独特之处：

```python
# fi.py 第 122-161 行（_initialize_metadata_once）
metadata.wrapper.plan(
    indptr=metadata.cu_seqlens_k_cpu,    # 在 CPU 上做计划
    indices=metadata.indices,
    ...
    non_blocking=True,
)
# ...
return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)  # 在 GPU 上执行
```

`plan` 阶段在 CPU 上预先分析 batch 的结构（序列长度分布、KV 块索引），生成 GPU kernel 的调度方案，写入 workspace buffer；`run` 阶段直接按计划执行，zero overhead。`non_blocking=True` 使 CPU 侧的 plan 与 GPU 计算异步重叠，进一步提高吞吐量。

**Tensor Core 自适应策略**（第 231-237 行）：

```python
@cached_property
def use_tensor_cores(self) -> bool:
    GQA = self.config.num_qo_heads // self.config.num_kv_heads
    return GQA >= 4
```

Grouped Query Attention（GQA）中，每个 KV head 对应多个 Q head。当 GQA 比例 ≥ 4 时（如 Llama-3 的 8:1 ratio），启用 Tensor Core 路径，利用矩阵运算的高并行性；比例较小时改用 CUDA Core 路径，避免额外的数据重排 overhead。

**FI 的索引格式与 FA 的差异**：

FA 使用二维页表 `[bs, max_pages_per_seq]`，FI 使用一维扁平索引 `indices = [所有序列的物理页号拼接]`，对应的 `cu_seqlens_k_cpu` 指明每条序列占用多少个页。FI 的这种格式对于序列长度差异悬殊的 batch 更节省内存。

### 7.3.5 HybridBackend：透明的阶段路由

**文件**：`python/minisgl/attention/base.py`，第 37-63 行

```python
class HybridBackend(BaseAttnBackend):
    def forward(self, q, k, v, layer_id, batch):
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch):
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)
```

`HybridBackend` 是一个极简的**策略模式（Strategy Pattern）** 实现。它本身不做任何计算，只根据 `batch.is_prefill` 标志将请求路由到对应的后端。对外暴露的接口与单一后端完全一致，调用方（`AttentionLayer`）无感知。

`init_capture_graph` 只初始化 decode 后端（第 56-57 行）：CUDA Graph 捕获只针对 decode 阶段，prefill 因为序列长度动态变化，无法静态捕获。

### 7.3.6 注册表模式与 `--attn` 参数

**文件**：`python/minisgl/attention/__init__.py` + `python/minisgl/utils/registry.py`

注册表 `SUPPORTED_ATTENTION_BACKENDS` 是一个泛型字典包装器：

```python
# __init__.py 第 19-40 行
SUPPORTED_ATTENTION_BACKENDS = Registry[BackendCreator]("Attention Backend")

@SUPPORTED_ATTENTION_BACKENDS.register("trtllm")
def create_trtllm_backend(config): ...

@SUPPORTED_ATTENTION_BACKENDS.register("fi")
def create_fi_backend(config): ...

@SUPPORTED_ATTENTION_BACKENDS.register("fa")
def create_fa_backend(config): ...
```

`Registry` 类（`utils/registry.py`）用 `@register(name)` 装饰器将工厂函数注册进字典，通过 `registry[name](config)` 取出并调用。要新增一个后端，只需在合适位置写一个带装饰器的工厂函数，无需修改引擎或模型代码。

`create_attention_backend` 函数（第 52-68 行）解析逗号分隔的字符串，自动构造 `HybridBackend`：

```python
# __init__.py 第 57-64 行
if "," in backend:
    p_backend, d_backend = backend.split(",", 1)
    if p_backend != d_backend:
        p_backend = create_attention_backend(p_backend, config)
        d_backend = create_attention_backend(d_backend, config)
        return HybridBackend(p_backend, d_backend)
```

用户通过命令行 `--attn fa,fi` 即可指定 Prefill 使用 FA、Decode 使用 FI 的混合后端，格式验证在 `validate_attn_backend` 中完成（第 43-48 行），服务启动前的 argparse 阶段就能给出友好的错误提示。

**Auto 模式的硬件感知选择**（`engine/engine.py` 第 224-226 行）：

```python
if config.attention_backend == "auto":
    backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
```

- **Blackwell（sm100，B200/B100）**：使用 `trtllm`，TensorRT-LLM 内核在 Blackwell 上有更好的优化；
- **Hopper（sm90，H100/H200）**：使用 `fa,fi` 混合，FA3 在 Hopper 上的 prefill 吞吐最优；
- **Ampere 及以下（sm80-，A100）**：使用 `fi`，FlashInfer 对 A100 兼容性更好。

---

## 7.4 设计决策

### 7.4.1 为什么 Prefill 用 FA，Decode 用 FlashInfer？

**FlashAttention 的强项**在于处理**长序列的 full attention**：分块计算减少 HBM 读写，在 H100 上 prefill 吞吐可达峰值理论值的 70% 以上。但 FA 的 kernel 对 `Q length = 1` 的 decode 场景并不特化，小批次下 SM 利用率低。

**FlashInfer 的强项**在于**灵活的批处理与 decode 特化**：它的 decode kernel 专门针对每序列生成 1 个 token 的情形设计，支持不规则批次（ragged batch）、GQA 快速路径、以及本文提到的 plan/run 两阶段 API，在 decode 阶段吞吐优于 FA。

两者互补，这是选择混合后端的根本原因。

### 7.4.2 FlashInfer 为何要求 `page_size=1`？

`fi.py` 第 66 行有断言：`assert self.page_size == 1`。

FlashInfer 的分页 KV API 将一个"页"定义为一个 token（page_size=1），其索引格式为**每个 token 独立一个物理 slot 的扁平列表**。这种格式下，每个 token 的物理位置都由 `indices` 数组直接给出，灵活性最高，但要求 KV Cache 以 token 为粒度分配页。若 page_size > 1，需要像 FA 的 `prepare_metadata` 那样先做步长采样转换，FI 暂不支持这个转换，因此直接做了约束。

### 7.4.3 为什么 metadata 在 forward 之外单独准备？

`prepare_metadata` 由调度器在 forward 前统一调用（`scheduler.py` 第 211 行），而不是在 `forward` 内部按需计算。原因有两点：

1. **CPU/GPU 流水线重叠**：metadata 构建涉及 CPU 上的张量运算和 `to(device, non_blocking=True)` 的异步拷贝，提前准备可以让数据在 forward 真正执行前就已经就绪；
2. **CUDA Graph 兼容性**：CUDA Graph 捕获要求 kernel 序列固定，`forward` 内只包含纯 GPU 操作。将 CPU 操作（plan、索引构建）移到 Graph 外，既满足捕获约束，又方便在 replay 时只更新变化的张量数据。

### 7.4.4 替代方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| 纯 FlashInfer | 统一 API，page_size 灵活性好 | Prefill 长序列吞吐不及 FA3 |
| 纯 FlashAttention | Prefill 性能最优，支持大 page_size | Decode 小批次效率低 |
| TensorRT-LLM | Blackwell 上深度优化 | page_size 需固定为 16/32/64，灵活性受限 |
| 混合 fa,fi | 各阶段最优，灵活 | 代码复杂度增加，两套 metadata 格式 |

---

## 7.5 动手实验

### 实验一：验证后端切换

确认三种后端均可正常加载，观察启动日志中的后端选择信息。

```bash
# 使用纯 FlashInfer 后端
python -m minisgl --model <your_model> --attn fi

# 使用混合后端（Prefill=FA, Decode=FI）
python -m minisgl --model <your_model> --attn fa,fi

# 查看 auto 模式自动选择的后端
python -m minisgl --model <your_model> --attn auto
```

预期：启动日志会打印 `Auto-selected attention backend: ...` 或 `Using hybrid attention backend: prefill=fa, decode=fi`。

### 实验二：验证注册表机制

在 Python 交互式环境中探查注册表内容：

```python
from minisgl.attention import SUPPORTED_ATTENTION_BACKENDS

# 查看所有已注册的后端名称
print(SUPPORTED_ATTENTION_BACKENDS.supported_names())
# 预期输出：['trtllm', 'fi', 'fa']

# 尝试访问不存在的后端，观察错误信息
try:
    SUPPORTED_ATTENTION_BACKENDS["nonexistent"]
except KeyError as e:
    print(e)
```

### 实验三：剖析 metadata 构建

编写一个小脚本，构造虚拟 batch，手动调用 `prepare_metadata` 并打印 `cu_seqlens` 的值，直观理解累积偏移量的语义：

```python
import torch

# 模拟三条请求，序列长度分别为 10、5、8
seqlens_k = [10, 5, 8]
cu_seqlens_k = torch.tensor([0] + seqlens_k).cumsum(dim=0)
print("cu_seqlens_k:", cu_seqlens_k)
# 输出：tensor([ 0, 10, 15, 23])

# 验证：第 i 条序列的长度
for i in range(len(seqlens_k)):
    length = (cu_seqlens_k[i+1] - cu_seqlens_k[i]).item()
    assert length == seqlens_k[i]
    print(f"序列 {i} 长度：{length}")
```

### 实验四（进阶）：自定义注意力后端

参照 `BaseAttnBackend` 接口，实现一个使用 PyTorch 原生 `scaled_dot_product_attention` 的参考后端，并通过注册表注入：

```python
# custom_backend.py
import torch
from minisgl.attention import SUPPORTED_ATTENTION_BACKENDS
from minisgl.attention.base import BaseAttnBackend, BaseAttnMetadata

@SUPPORTED_ATTENTION_BACKENDS.register("sdpa")
def create_sdpa_backend(config):
    return SDPABackend(config)

class SDPABackend(BaseAttnBackend):
    def forward(self, q, k, v, layer_id, batch):
        # 注意：这里需要从 KV cache 读取完整历史，仅作演示
        return torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=True
        ).squeeze(0)

    def prepare_metadata(self, batch): ...
    def init_capture_graph(self, max_seq_len, bs_list): ...
    def prepare_for_capture(self, batch): ...
    def prepare_for_replay(self, batch): ...
```

在启动时通过 `--attn sdpa` 使用该后端，验证注册表插件机制的可扩展性。

---

## 7.6 小结

本章从标准 Attention 的 O(n²) 内存问题出发，沿着问题→原理→代码→设计的脉络，梳理了 mini-sglang 注意力后端系统的全貌：

**核心要点回顾**：

1. **FlashAttention 的价值**：通过分块计算和在线 softmax，将中间矩阵的 HBM 读写量从 O(n²) 降至 O(n)，既节省显存又提升带宽效率。FA3 在 Hopper 架构上额外利用 TMA 硬件单元。

2. **Prefill vs. Decode 的异构性**：Prefill 是计算密集型（长序列矩阵乘），FA 最优；Decode 是内存访问密集型（单 token 查历史 KV），FlashInfer 的特化 kernel 更合适。`HybridBackend` 以最小代价融合两者。

3. **注册表模式**：`Registry` + `@register` 装饰器实现了开放-封闭原则，新增后端只需添加代码，无需修改现有逻辑。`create_attention_backend` 通过逗号语法自动构建混合后端。

4. **metadata 分离设计**：将 CPU 侧的索引构建（`prepare_metadata`）与 GPU 侧的计算（`forward`）解耦，既支持 CPU/GPU 异步流水，又为 CUDA Graph 捕获铺平了道路。

5. **硬件感知的 auto 选择**：auto 模式通过探测 SM 版本，在 Blackwell/Hopper/Ampere 三代架构上自动选择最优组合，用户无需了解底层细节。

**与后续章节的连接**：

- **第 8 章（KV Cache）**：注意力后端与 KV Cache 深度耦合，`store_kv`、页表格式、物理页分配都将在下一章详细展开；
- **第 9 章（CUDA Graph）**：`init_capture_graph` / `prepare_for_capture` / `prepare_for_replay` 三个接口在本章只是"冰山一角"，第 9 章将完整讲解 CUDA Graph 如何捕获 decode forward pass，以及为何只捕获 decode 而不捕获 prefill。
