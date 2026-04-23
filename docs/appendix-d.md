# 附录D：推理引擎（engine/）

本附录详细说明 `python/minisgl/engine/` 目录下四个核心文件的实现，包括引擎配置、主推理引擎、CUDA Graph 管理和采样器。

---

## D.1 `engine/config.py` — 引擎配置

### 文件职责

定义 `EngineConfig` 不可变数据类，集中管理推理引擎的所有启动参数，并通过 `cached_property` 懒加载模型配置。

### 公开类

#### `EngineConfig`

```python
@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None
```

**字段说明：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | `str` | — | HuggingFace 模型路径或本地目录 |
| `tp_info` | `DistributedInfo` | — | 张量并行信息（rank、world_size） |
| `dtype` | `torch.dtype` | — | 计算数据类型（如 `bfloat16`） |
| `max_running_req` | `int` | 256 | 同时运行的最大请求数，决定 page_table 行数 |
| `attention_backend` | `str` | `"auto"` | 注意力后端，`"auto"` 时按 GPU 架构自动选择 |
| `moe_backend` | `str` | `"auto"` | MoE 后端，`"auto"` 时对 MoE 模型选择 `"fused"` |
| `cuda_graph_bs` | `List[int] \| None` | `None` | 手动指定要捕获 CUDA Graph 的 batch size 列表 |
| `cuda_graph_max_bs` | `int \| None` | `None` | 自动生成 batch size 列表时的上限 |
| `page_size` | `int` | 1 | KV 缓存分页大小（TRTLLM 后端要求 16/32/64） |
| `memory_ratio` | `float` | 0.9 | 分配 KV 缓存时占用空闲显存的比例 |
| `distributed_timeout` | `float` | 60.0 | 分布式通信超时秒数 |
| `use_dummy_weight` | `bool` | `False` | 用随机权重替代真实权重（调试/测试用） |
| `use_pynccl` | `bool` | `True` | 是否使用 PyNCCL 而非原生 NCCL 做张量并行通信 |
| `max_seq_len_override` | `int \| None` | `None` | 覆盖模型默认的最大序列长度 |
| `num_page_override` | `int \| None` | `None` | 直接指定 KV 缓存页数，跳过自动显存估算 |

**属性方法：**

```python
@cached_property
def hf_config(self) -> AutoConfig:
    ...

@cached_property
def model_config(self) -> ModelConfig:
    ...

@property
def max_seq_len(self) -> int:
    ...

@property
def max_forward_len(self) -> int:
    ...

@property
def distributed_addr(self) -> str:
    ...
```

| 属性 | 返回值 | 说明 |
|------|--------|------|
| `hf_config` | HuggingFace `AutoConfig` | 懒加载并缓存 HF 配置对象 |
| `model_config` | `ModelConfig` | 从 HF 配置解析出的模型配置（层数、head数等） |
| `max_seq_len` | `int` | 有效最大序列长度，`max_seq_len_override` 优先 |
| `max_forward_len` | `int` | 当前等于 `max_seq_len`，为未来扩展预留 |
| `distributed_addr` | `str` | 分布式进程组初始化地址，固定为 `"tcp://127.0.0.1:2333"` |

### 关键实现细节

1. **不可变冻结**：使用 `frozen=True` 防止配置被意外修改。但 `_adjust_config()` 在引擎初始化时通过 `object.__setattr__()` 绕过冻结，完成自动后端选择的覆写，这是有意为之的设计。

2. **懒加载配置**：`hf_config` 和 `model_config` 使用 `@cached_property` 修饰，第一次访问时才从磁盘读取，避免不必要的 I/O 开销。

3. **`frozen=True` 与 `cached_property` 的兼容性**：`@cached_property` 会尝试向实例 `__dict__` 写入缓存值，而 `frozen=True` 会阻止这一写入。Python 3.8+ 的 `dataclasses` 模块通过为 `frozen` 数据类生成 `__setattr__` 钩子来实现冻结，但 `cached_property` 直接操作 `__dict__`，可以绕过该钩子。此处能正常工作是因为 Python 的 `__dict__` 访问在 `frozen` 数据类中并未被完全屏蔽。

---

## D.2 `engine/engine.py` — 主推理引擎

### 文件职责

定义 `Engine` 类，作为整个推理系统的核心协调器，负责完成从模型加载、KV 缓存分配到单步前向传播的全部工作。

### 公开类与函数

#### `ForwardOutput`

```python
class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor   # GPU 上的下一个 token id，shape: (batch_size,)
    next_tokens_cpu: torch.Tensor   # 异步复制到 CPU 的下一个 token id
    copy_done_event: torch.cuda.Event  # 标记 CPU 复制完成的 CUDA 事件
```

`ForwardOutput` 是 `forward_batch` 的返回值，封装了 GPU 端结果、CPU 端异步结果以及同步事件，调用方可以通过等待 `copy_done_event` 来安全读取 `next_tokens_cpu`。

#### `Engine.__init__`

```python
def __init__(self, config: EngineConfig) -> None:
```

**参数：** `config` — 完整的引擎配置对象。

**功能：** 按固定顺序完成以下六个阶段的初始化（详见下文"Engine 初始化流程"小节）。

#### `Engine.forward_batch`

```python
def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
```

**参数：**
- `batch: Batch` — 当前批次，包含请求列表、输入 token id、位置编码、输出位置等字段
- `args: BatchSamplingArgs` — 批次采样参数（temperature、top-k、top-p）

**返回值：** `ForwardOutput`

**功能：** 执行一步 decode/prefill 前向传播并采样下一个 token。

#### `Engine.shutdown`

```python
def shutdown(self) -> None:
```

**功能：** 安全关闭引擎，按顺序销毁 CUDA Graph、进程组和分布式资源。注意必须先销毁 CUDA Graph 再释放 NCCL 资源，否则会导致程序挂起。

#### 模块级辅助函数

```python
def _align_up_32(num: int) -> int:
```
将整数向上对齐到 32 的倍数，用于对齐 `max_seq_len` 以满足内存访问对齐要求。

```python
def _adjust_config(config: EngineConfig) -> None:
```
在引擎初始化前自动覆写若干配置字段：
- 按 GPU 架构自动选择 attention backend（SM100→`"trtllm"`，SM90→`"fa,fi"`，其他→`"fi"`）
- TRTLLM 后端要求 page_size 为 16/32/64，否则强制覆写为 64
- MoE 模型默认使用 `"fused"` moe_backend

### Engine 初始化流程

`Engine.__init__` 的完整初始化顺序如下：

```
1. 分布式初始化
   ├── set_tp_info(rank, size)
   ├── _adjust_config(config)        # 自动后端选择
   ├── torch.cuda.set_device(rank)
   ├── torch.manual_seed(42)         # 固定随机种子保证一致性
   └── _init_communication(config)   # 建立进程组

2. 内存基准线采集
   └── _sync_get_memory()            # 记录模型加载前的空闲显存

3. 模型初始化
   ├── create_model(model_config)    # 在 meta 设备上创建模型骨架
   └── load_state_dict(weights)      # 加载/生成权重

4. KV 缓存初始化
   ├── _determine_num_pages()        # 根据显存差量计算可分配页数
   └── create_kvcache_pool()         # 分配六维 KV 缓存 tensor

5. Page Table 初始化
   └── torch.zeros(max_running_req+1, aligned_max_seq_len)  # +1 for dummy req

6. Attention / MoE 后端初始化
   ├── create_attention_backend()
   └── create_moe_backend()          # 仅对 MoE 模型

7. 采样器初始化
   └── Sampler(device, vocab_size)

8. CUDA Graph 捕获
   ├── 创建 dummy_req（table_idx = max_running_req）
   ├── page_table[dummy_req.table_idx].fill_(num_tokens)  # 指向 dummy page
   └── GraphRunner(...)              # 执行 warmup + capture
```

**关键细节：**

- 模型先在 `meta` 设备上构建（`torch.device("meta")`），此时不分配实际显存，用于准确测量模型加载后的显存占用差。
- `_sync_get_memory()` 使用 `all_reduce(MIN)` 取各 TP rank 中最小的空闲显存，保证多卡场景下 KV 缓存分配的保守性。若最大值与最小值差距超过 2 GB，会抛出异常。
- `dummy_req` 的 `page_table` 行指向 `num_pages`（即刚好超出合法范围的 dummy page），确保 decode padding 期间的 attention 计算不会污染真实 KV 缓存。

### `forward_batch` 执行路径

```python
# engine.py, 行 193-208
def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
    assert torch.cuda.current_stream() == self.stream      # (1) 流一致性检查
    with self.ctx.forward_batch(batch):                    # (2) 设置全局批次上下文
        if self.graph_runner.can_use_cuda_graph(batch):    # (3) CUDA Graph 路径判断
            logits = self.graph_runner.replay(batch)
        else:
            logits = self.model.forward()                  # (4) 普通前向

    for req in batch.reqs:
        req.complete_one()                                 # (5) 更新请求状态

    next_tokens_gpu = self.sampler.sample(logits[:batch.size], args).to(torch.int32)
    next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)  # (6) 异步 D2H 复制
    copy_done_event = torch.cuda.Event()
    copy_done_event.record(self.stream)
    return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

执行路径的六个关键步骤：
1. 断言当前 CUDA 流与引擎专用流一致
2. 通过 `ctx.forward_batch(batch)` 上下文管理器将 batch 写入全局上下文，供模型各层无参数访问
3. 判断是否满足 CUDA Graph 回放条件（decode 阶段且 batch size ≤ max_graph_bs）
4. 根据条件选择 Graph replay 或普通 `model.forward()`
5. 对批次内每个请求调用 `complete_one()` 推进其内部状态
6. 采样后将结果异步复制到 CPU，记录完成事件供调用方等待

### `_init_communication` 通信初始化

```python
def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
```

根据 `use_pynccl` 配置选择两种通信模式：

- **PyNCCL 模式**（默认）：使用 `gloo` 后端建立进程组，再通过 `enable_pynccl_distributed()` 初始化 PyNCCL，`max_bytes` 按最大 forward 长度 × hidden_size × dtype_size 计算。
- **原生 NCCL 模式**：使用 `nccl` 后端建立主进程组，再额外建立 `gloo` CPU 进程组（用于 `all_reduce` 等 CPU 操作）。

### `_determine_num_pages` 页数估算

```python
def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
```

**核心计算逻辑（行 152-164）：**

```python
cache_per_page = (
    2                                                    # key + value
    * config.model_config.head_dim
    * div_even(config.model_config.num_kv_heads, config.tp_info.size)
    * config.page_size
    * self.dtype.itemsize
    * config.model_config.num_layers
)
model_memory = old_free_memory - new_free_memory         # 模型占用的显存
available_memory = int(config.memory_ratio * old_free_memory) - model_memory
num_pages = available_memory // cache_per_page
```

每页 KV 缓存的字节数 = `2（K+V）× head_dim × local_kv_heads × page_size × dtype_size × num_layers`。通过 `memory_ratio × 总空闲显存 - 模型显存` 得到可用于 KV 缓存的显存量，再除以每页大小得到页数。

---

## D.3 `engine/graph.py` — CUDA Graph 管理

### 文件职责

管理 CUDA Graph 的捕获与回放，通过预先录制计算图来消除每步 decode 的 Python/CUDA driver 开销，显著降低推理延迟。

### 公开类与函数

#### `GraphCaptureBuffer`

```python
@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor    # shape: (max_graph_bs,), dtype: int32
    out_loc:   torch.Tensor    # shape: (max_graph_bs,), dtype: int32
    positions: torch.Tensor    # shape: (max_graph_bs,), dtype: int32
    logits:    torch.Tensor    # shape: (max_graph_bs, vocab_size), dtype: float32
```

`GraphCaptureBuffer` 是 CUDA Graph 的核心数据结构。CUDA Graph 在捕获时绑定了固定的 GPU 内存地址，回放时只能通过**原地修改**这些缓冲区的内容来传入新数据，而不能传入不同地址的张量。`GraphCaptureBuffer` 正是为此目的预分配的一组固定 GPU 缓冲区。

**类方法：**

```python
@classmethod
def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
```

创建大小为 `bs × vocab_size` 的缓冲区，所有输入张量用零初始化，logits 用 `torch.empty`（未初始化）。

**实例方法：**

```python
def set_batch(self, batch: Batch) -> None:
```
将 `batch.input_ids`、`batch.out_loc`、`batch.positions` 替换为指向 `GraphCaptureBuffer` 对应切片的视图。在 CUDA Graph **捕获阶段**调用，使模型计算绑定到缓冲区地址。

```python
def copy_from(self, batch: Batch) -> None:
```
将实际批次数据**原地复制**到缓冲区。在 CUDA Graph **回放阶段**调用，写入新数据但保持地址不变，从而触发正确的计算。

#### `GraphRunner`

```python
class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
    ) -> None:
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `stream` | 引擎专用 CUDA 流，所有图的捕获与回放都在此流上执行 |
| `device` | 目标 GPU 设备 |
| `model` | 要捕获的 LLM 模型 |
| `attn_backend` | 注意力后端（需配合 CUDA Graph 初始化） |
| `cuda_graph_bs` | 手动指定 batch size 列表；为 None 时自动生成 |
| `cuda_graph_max_bs` | 自动生成模式的上限 |
| `free_memory` | 当前空闲显存（字节），用于自动确定 max_bs） |
| `max_seq_len` | 对齐后的最大序列长度，传入 attention backend |
| `vocab_size` | 词表大小，决定 logits 缓冲区大小 |
| `dummy_req` | 填充用虚请求，用于在 batch 不满时填充 |

**实例方法：**

```python
def can_use_cuda_graph(self, batch: Batch) -> bool:
```
判断条件：batch 必须处于 decode 阶段（`batch.is_decode == True`）且 `batch.size <= self.max_graph_bs`。

```python
def replay(self, batch: Batch) -> torch.Tensor:
```
执行 CUDA Graph 回放：
1. 将 batch 数据复制到 `GraphCaptureBuffer`（`buffer.copy_from(batch)`）
2. 从 `graph_map` 中取出对应 `padded_size` 的图
3. 调用 `attn_backend.prepare_for_replay(batch)` 更新 attention 后端元数据
4. `g.replay()` 回放计算图
5. 返回 `buffer.logits[:batch.size]`（截去 padding 部分）

```python
def pad_batch(self, batch: Batch) -> None:
```
将批次填充到下一个可用的图 batch size：在 `graph_bs_list` 中找第一个 `>= batch.size` 的值，用 `dummy_req` 补足差额。

```python
def destroy_cuda_graphs(self) -> None:
```
删除 `graph_map` 字典并调用 `gc.collect()`。**必须在释放 NCCL 资源之前调用**，否则可能导致程序挂起（因为 CUDA Graph 内部可能持有 NCCL communicator 的引用）。

### `_determine_cuda_graph_bs` 自动 batch size 生成

```python
def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
```

**生成策略：**
- 如果用户手动指定 `cuda_graph_bs`，直接返回
- 否则根据空闲显存自动决定 `cuda_graph_max_bs`：空闲 > 80 GiB（H200 级别）取 256，否则取 160
- 生成 `[1, 2, 4, 8, 16, 24, ..., cuda_graph_max_bs]`（即 1/2/4 加上 8 的倍数）
- `cuda_graph_max_bs < 1` 时返回空列表，禁用 CUDA Graph

### `_capture_graphs` 捕获两步流程

```python
def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
```

CUDA Graph 的捕获分为 **warmup（预热）** 和 **capture（录制）** 两个步骤，在同一个循环中按从大到小的 batch size 顺序执行：

```python
# graph.py, 行 128-144
pool = None
for bs in pbar:  # 按 batch size 从大到小遍历
    graph = torch.cuda.CUDAGraph()
    batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
    batch.padded_reqs = batch.reqs
    self.attn_backend.prepare_for_capture(batch)
    self.buffer.set_batch(batch)
    with get_global_ctx().forward_batch(batch):
        # 步骤1：warmup（正式录制前先运行一遍，预热 CUDA kernel）
        self.buffer.logits[:bs] = model.forward()
        # 步骤2：capture（录制计算图）
        with torch.cuda.graph(graph, pool=pool, stream=self.stream):
            self.buffer.logits[:bs] = model.forward()
    if pool is None:
        pool = graph.pool()   # 复用第一个图的内存池
    self.graph_map[bs] = graph
```

**两步流程的意义：**
1. **Warmup 步骤**（`torch.cuda.graph` 上下文之外的 `model.forward()`）：确保所有 CUDA kernel 已经被编译和缓存，避免首次录制时 JIT 编译的 kernel 被错误地录入图中。
2. **Capture 步骤**（`torch.cuda.graph` 上下文内的 `model.forward()`）：录制实际的 GPU 命令序列到 `CUDAGraph` 对象中。

**内存池复用：** 首个图（最大 bs）创建后，通过 `graph.pool()` 获取其内存池句柄，后续所有图共享该内存池（`pool=pool` 参数），减少碎片化显存占用。

**从大到小的顺序：** `sorted(self.graph_bs_list, reverse=True)` 按 batch size 降序捕获。由于较大 batch size 的图会分配更多内存，先捕获大图可以在内存池中预留足够空间，避免后续小图因内存不足失败。

**CUDA Graph 兼容性要求：** 以下操作不兼容 CUDA Graph，必须在 `prepare_for_capture` 和 `prepare_for_replay` 中提前处理：
- 动态形状操作（如 `torch.cat` 沿可变维度）
- CPU ↔ GPU 数据传输
- `torch.cuda.synchronize()`
- Python 控制流（if/for 的分支路径会被固化）

---

## D.4 `engine/sample.py` — 采样器

### 文件职责

封装 token 采样逻辑，支持贪心采样、温度采样、top-k 采样、top-p（nucleus）采样及其组合，底层调用 FlashInfer 的高性能 CUDA 采样 kernel。

### 公开类与函数

#### `BatchSamplingArgs`

```python
@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None  # shape: (batch_size,), dtype: float32
                                       # None 表示全批次贪心采样
    top_k: torch.Tensor | None = None  # shape: (batch_size,), dtype: int32
                                       # None 表示不限制 top-k
    top_p: torch.Tensor | None = None  # shape: (batch_size,), dtype: float32
                                       # None 表示不限制 top-p
```

`BatchSamplingArgs` 以逐元素张量的形式存储整个批次的采样超参数，允许批次内不同请求使用不同的采样策略。`temperatures=None` 是全批次贪心采样的特殊标记，可跳过 softmax 直接用 `argmax`。

#### `make_device_tensor`

```python
def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
```

**参数：**
- `data`：Python 列表
- `dtype`：目标数据类型
- `device`：目标设备

**返回值：** GPU 上的张量

**功能：** 将 Python 列表通过 pin memory 中间缓冲区**非阻塞**地传输到 GPU，避免 H2D 传输成为流水线瓶颈。

#### `sample_impl`

```python
def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
```

**参数：**
- `logits`：模型输出的 logit 值，shape `(batch_size, vocab_size)`，dtype `float32`
- `temperatures`：每个请求的温度值，shape `(batch_size,)`
- `top_k`：每个请求的 top-k 限制；`None` 表示不限制
- `top_p`：每个请求的 top-p 限制；`None` 表示不限制

**返回值：** 采样得到的 token id，shape `(batch_size,)`

**功能：** 根据 top-k / top-p 是否为 None，分四条路径调用 FlashInfer 采样 kernel：

| top_k | top_p | 调用函数 |
|-------|-------|---------|
| None | None | `sampling.sampling_from_probs(probs)` |
| 非 None | None | `sampling.top_k_sampling_from_probs(probs, top_k)` |
| None | 非 None | `sampling.top_p_sampling_from_probs(probs, top_p)` |
| 非 None | 非 None | `sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)` |

softmax 计算通过 `sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())` 完成，在 SM90（Hopper）架构上启用 PDL（Parallel Data Loading）优化。

#### `Sampler`

```python
@dataclass
class Sampler:
    device: torch.device
    vocab_size: int
```

**方法：**

```python
def prepare(self, batch: Batch) -> BatchSamplingArgs:
```

**参数：** `batch` — 当前批次

**返回值：** 为该批次准备好的 `BatchSamplingArgs`

**功能：** 从每个请求的 `sampling_params` 中提取并向量化采样参数。实现了以下逻辑：

1. **全批次贪心检测**：若所有请求都是贪心（`p.is_greedy`），返回 `BatchSamplingArgs(temperatures=None)` 直接走 `argmax` 路径，避免不必要的 GPU 内核调用。

2. **温度裁剪**：`temperature = max(0.0 if greedy else p.temperature, MIN_T)`，其中 `MIN_T = 1e-6`，防止除以零。

3. **top-k 处理**：`p.top_k < 1` 时视为"无限制"，映射为 `vocab_size`（即对全词表采样）；若所有请求的 top_k 均等于 `vocab_size`，则 `top_k` 字段保持 `None`。

4. **top-p 裁剪**：`top_p = min(max(p.top_p, MIN_P), 1.0)`，其中 `MIN_P = 1e-6`；若所有请求的 `top_p >= 1.0`，则 `top_p` 字段保持 `None`。

5. 非贪心参数通过 `make_device_tensor` 批量传输到 GPU。

```python
@nvtx_annotate("Sampler")
def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
```

**参数：**
- `logits`：已切片的 logit 张量，shape `(batch_size, vocab_size)`
- `args`：`prepare()` 返回的采样参数

**返回值：** 采样 token id，shape `(batch_size,)`

**功能：** 两条路径：
- `args.temperatures is None` → 贪心采样：`torch.argmax(logits, dim=-1)`
- 否则 → 调用 `sample_impl(logits.float(), ...)` 执行概率采样

`.float()` 确保 logits 为 float32，即使模型使用 bfloat16 推理也能保证采样精度。该方法使用 `@nvtx_annotate` 和 `torch.cuda.nvtx.range` 双重 NVTX 标记，方便在 Nsight Systems 中定位采样耗时。

### 各采样策略实现总结

```
输入 logits (batch_size, vocab_size)
         │
         ▼
temperatures is None ?
    ├─ 是 → argmax(logits, dim=-1)             [贪心采样]
    └─ 否 → softmax(logits, temperatures)
                │
                ▼
         top_k is None AND top_p is None ?
            ├─ 是 → sampling_from_probs(probs)  [纯温度采样]
            └─ 否 →
                top_p is None ?
                  ├─ 是 → top_k_sampling_from_probs(probs, top_k)
                  └─ 否 →
                      top_k is None ?
                        ├─ 是 → top_p_sampling_from_probs(probs, top_p)
                        └─ 否 → top_k_top_p_sampling_from_probs(probs, top_k, top_p)
```

---

## D.5 模块交互关系

```
EngineConfig
    │  配置注入
    ▼
Engine.__init__
    ├──► _adjust_config()          # 自动后端选择（覆写 EngineConfig）
    ├──► _init_communication()     # gloo/NCCL 进程组 + PyNCCL
    ├──► create_model()            # meta 设备骨架 → 加载真实权重
    ├──► _determine_num_pages()    # 显存估算
    ├──► create_kvcache_pool()     # MHAKVCache 六维 tensor
    ├──► create_attention_backend()
    ├──► Sampler()
    └──► GraphRunner()
            └──► _capture_graphs()  # warmup + capture（从大到小）

Engine.forward_batch(batch, args)
    ├──► ctx.forward_batch(batch)  # 设置全局 batch 上下文
    ├──► GraphRunner.can_use_cuda_graph()
    │       ├─ True  → GraphRunner.replay()     # 复制数据 → g.replay()
    │       └─ False → model.forward()
    ├──► req.complete_one()        # 更新每个请求状态
    └──► Sampler.sample()          # argmax 或 FlashInfer 概率采样
```
