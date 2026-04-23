# 附录 G：注意力后端与分布式模块详解

本附录对 mini-sglang 中 `attention/` 和 `distributed/` 两个核心子包进行全面代码解析，包含每个文件的职责说明、所有公开接口的签名与语义、关键实现细节，以及带行号的原始代码片段。

---

## 第一节：注意力后端（`python/minisgl/attention/`）

注意力后端子包由六个文件组成：

| 文件 | 职责 |
|------|------|
| `base.py` | 定义抽象基类 `BaseAttnBackend`、元数据基类 `BaseAttnMetadata`、组合后端 `HybridBackend` |
| `utils.py` | 定义 CUDA Graph 捕获共享数据结构 `BaseCaptureData` |
| `fa.py` | 基于 `sgl_kernel` 的 FlashAttention 后端 |
| `fi.py` | 基于 `flashinfer` 库的 FlashInfer 后端 |
| `trtllm.py` | 基于 TensorRT-LLM 内核的注意力后端 |
| `__init__.py` | 注册表、工厂函数、`auto` 模式后端选择 |

---

### G.1.1 `base.py` — 注意力后端抽象基类与 HybridBackend

#### 文件职责

定义所有注意力后端必须实现的抽象接口，以及将预填充（prefill）和解码（decode）分别路由到不同后端的 `HybridBackend` 组合类。

#### `BaseAttnMetadata`

```python
# base.py 第 12–15 行
@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...
```

**说明：** 所有注意力元数据的抽象基类，继承自 `ABC` 并使用 `@dataclass` 修饰。

**方法：**

- `get_last_indices(bs: int) -> torch.Tensor`
  - 参数：`bs` — 当前批次中真实请求数（不含填充）
  - 返回值：形状为 `(bs,)` 的整数张量，每个元素是第 `i` 个请求在扁平化 token 序列中的最后一个 token 的位置索引
  - 用途：采样阶段从注意力输出中提取每个序列的最后一个隐藏状态，用于后续的 logit 计算

#### `BaseAttnBackend`

```python
# base.py 第 18–34 行
class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...
```

五个方法的语义如下：

**1. `forward(q, k, v, layer_id, batch) -> torch.Tensor`**
- 参数：
  - `q`：查询张量，形状 `(total_tokens, num_qo_heads, head_dim)`
  - `k`：键张量，同 `q` 形状
  - `v`：值张量，同 `q` 形状
  - `layer_id`：当前 Transformer 层编号，用于索引 KV 缓存
  - `batch`：当前批次对象，含 `attn_metadata`、`out_loc`（写入位置）等信息
- 返回值：注意力输出张量，形状与 `q` 相同
- 语义：执行完整的注意力计算，包括将 `k/v` 写入 KV 缓存，然后执行分页注意力计算

**2. `prepare_metadata(batch: Batch) -> None`**
- 参数：`batch` — 当前批次
- 返回值：无（原地修改 `batch.attn_metadata`）
- 语义：在每次调度迭代开始时，根据批次中每个请求的长度信息，构建后端所需的元数据（如 `cu_seqlens`、`page_table` 等），并将其存储在 `batch.attn_metadata` 中

**3. `init_capture_graph(max_seq_len: int, bs_list: List[int]) -> None`**
- 参数：
  - `max_seq_len`：CUDA Graph 录制时支持的最大序列长度
  - `bs_list`：需要录制 CUDA Graph 的批次大小列表
- 返回值：无
- 语义：为 CUDA Graph 捕获预分配固定形状的张量缓冲区，只能调用一次（若重复调用则抛出断言错误）

**4. `prepare_for_capture(batch: Batch) -> None`**
- 参数：`batch` — 将被录制到 CUDA Graph 的批次
- 返回值：无
- 语义：在 CUDA Graph 录制前被调用，将 `batch.attn_metadata` 替换为指向静态缓冲区的元数据视图，确保录制过程中的张量地址不变

**5. `prepare_for_replay(batch: Batch) -> None`**
- 参数：`batch` — 包含真实运行时数据的批次
- 返回值：无
- 语义：在每次 CUDA Graph 回放前被调用，将真实的运行时数据（实际序列长度、页表等）复制到静态缓冲区，使 Graph 能够用真实数据执行

#### `HybridBackend`

```python
# base.py 第 37–63 行
class HybridBackend(BaseAttnBackend):
    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q, k, v, layer_id, batch
    ) -> torch.Tensor:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)
```

**策略路由逻辑：**

`HybridBackend` 是策略模式（Strategy Pattern）的典型应用。其核心逻辑是通过 `batch.is_prefill` 布尔标志在两个后端之间进行路由：

- `forward` 和 `prepare_metadata`：根据 `batch.is_prefill` 动态选择使用 `prefill_backend` 还是 `decode_backend`
- `init_capture_graph`、`prepare_for_capture`、`prepare_for_replay`：**仅委托给 `decode_backend`**，因为 CUDA Graph 只对解码阶段有意义（预填充阶段序列长度动态变化，无法用静态 Graph 录制）

这样的设计允许在 SM90（Hopper）GPU 上将预填充路由到 FlashAttention（`fa`），将解码路由到 FlashInfer（`fi`），充分利用不同内核在各阶段的性能优势。

---

### G.1.2 `utils.py` — 注意力工具函数与 CUDA Graph 捕获数据结构

#### 文件职责

定义 CUDA Graph 捕获阶段所需的静态缓冲区数据结构 `BaseCaptureData`，供所有后端共享使用。

#### `BaseCaptureData`

```python
# utils.py 第 7–23 行
@dataclass
class BaseCaptureData:
    seq_lens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor

    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            **kwargs,
        )
```

**字段说明：**

| 字段 | 形状 | 设备 | 用途 |
|------|------|------|------|
| `seq_lens` | `(max_bs,)` | GPU | 每个序列的实际 KV 长度（初始化为 1） |
| `positions` | `(max_bs,)` | GPU | 每个 token 的位置编码偏移（初始化为 0） |
| `cu_seqlens_k` | `(max_bs+1,)` | GPU | KV 序列的累积长度前缀和（初始化为 `[0,1,2,...,max_bs]`） |
| `cu_seqlens_q` | `(max_bs+1,)` | GPU | Query 序列的累积长度前缀和（解码时每序列长度为 1） |
| `page_table` | `(max_bs, max_seq_len)` | GPU | 分页 KV 缓存的页表，记录每个 token 位置对应的物理页索引 |

**工厂方法 `create`：**
- 参数：
  - `max_bs`：支持的最大批次大小
  - `max_seq_len`：每个序列支持的最大 KV 页数（对于 FA/TRTLLM 后端等于 `max_seq_len // page_size`）
  - `device`：目标 CUDA 设备
  - `**kwargs`：子类可通过 `cls(...)` 传递额外字段
- 返回值：初始化好的 `BaseCaptureData` 实例

**初始化约定：** `cu_seqlens_k` 和 `cu_seqlens_q` 初始化为 `[0, 1, 2, ..., max_bs]`，表示解码阶段每个序列贡献一个 token。`seq_lens` 初始化为全 1，`positions` 初始化为全 0，确保 CUDA Graph 录制时产生合法的（虽无意义的）计算输出。

---

### G.1.3 `fa.py` — FlashAttention 后端

#### 文件职责

封装 `sgl_kernel.flash_attn.flash_attn_with_kvcache` 内核，实现支持分页 KV 缓存和 CUDA Graph 的 FlashAttention 注意力后端。

#### `FAMetadata`

```python
# fa.py 第 23–33 行
@dataclass
class FAMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor   # GPU 上的 KV 累积序列长度，形状 (bs+1,)
    cu_seqlens_q: torch.Tensor   # GPU 上的 Q 累积序列长度，形状 (bs+1,)
    cache_seqlens: torch.Tensor  # GPU 上每个序列的 KV 长度，形状 (bs,)
    max_seqlen_k: int            # 批次中最长 KV 序列的长度
    max_seqlen_q: int            # 批次中最长 Query 序列的长度
    page_table: torch.Tensor     # 重新索引后的页表，形状 (bs, max_pages_per_seq)

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1
```

`get_last_indices` 利用 `cu_seqlens_q` 的差分性质：第 `i` 个序列的 token 在扁平 Q 张量中占据 `[cu_seqlens_q[i], cu_seqlens_q[i+1])` 范围，因此最后一个 token 的索引为 `cu_seqlens_q[i+1] - 1`。

#### `FlashAttentionBackend`

**构造函数：**

```python
# fa.py 第 37–46 行
def __init__(self, config: ModelConfig):
    ctx = get_global_ctx()
    self.config = config
    self.kvcache = ctx.kv_cache
    self.page_size = ctx.page_size
    self.capture: FACaptureData | None = None
    self.max_graph_bs = 0
    self.capture_bs: List[int] = []
    self.scale = config.head_dim**-0.5
    self.version = 4 if is_sm100_supported() else 3
```

`self.version` 的取值体现了硬件感知设计：在 SM100（Blackwell）架构上使用 FlashAttention 4，其余架构使用 FlashAttention 3。这一判断通过 `is_sm100_supported()` 查询 `torch.cuda.get_device_capability()` 实现。

**`prepare_metadata` 方法与 `cu_seqlens_q` 三分支逻辑：**

```python
# fa.py 第 67–105 行
def prepare_metadata(self, batch: Batch) -> None:
    reqs = batch.padded_reqs
    padded_size = len(reqs)
    seqlens_q = [req.extend_len for req in reqs]
    seqlens_k = [req.device_len for req in reqs]
    cached_lens = [req.cached_len for req in reqs]
    max_seqlen_k = max(seqlens_k)
    max_seqlen_q = max(seqlens_q)
    CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

    device = self.kvcache.device
    cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)
    cache_seqlens = cache_seqlens.to(device, non_blocking=True)
    cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
    cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

    if max_seqlen_q == 1:                          # 分支 1：纯解码
        cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
    elif all(l == 0 for l in cached_lens):          # 分支 2：无缓存命中的首次预填充
        cu_seqlens_q = cu_seqlens_k
    else:                                           # 分支 3：有部分缓存命中的扩展预填充
        cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)
```

**三分支的含义：**

| 分支 | 触发条件 | 逻辑 |
|------|----------|------|
| 分支 1（解码） | `max_seqlen_q == 1`，所有请求每步仅生成 1 个 token | `cu_seqlens_q = [0, 1, 2, ..., bs]`，即每个请求贡献 1 个 Q token，构造连续整数序列可完全在 GPU 上完成，避免 CPU-GPU 数据传输 |
| 分支 2（全新预填充） | 所有请求的 `cached_len == 0`，无任何 KV 缓存命中 | Q 的长度等于 K 的长度，因此 `cu_seqlens_q` 可以直接复用已计算的 `cu_seqlens_k`，节省一次 tensor 构造 |
| 分支 3（扩展预填充） | 有部分请求命中前缀缓存，`extend_len < device_len` | Q 的长度为 `extend_len`（新增 token 数），K 的长度为 `device_len`（含缓存 token），必须单独构造 `cu_seqlens_q` |

**页表转换逻辑：**

全局页表以 `page_size=1` 的粒度存储，而 FlashAttention 内核需要以实际 `page_size` 为粒度的页索引。因此需要进行转换：

```python
# fa.py 第 92–97 行
page_table = get_global_ctx().page_table
new_page_table = torch.stack(
    [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
)
if self.page_size > 1:
    new_page_table.div_(self.page_size, rounding_mode="floor")
```

以步长 `self.page_size` 切片（`:: self.page_size`）从全局页表中取出每个逻辑页的起始物理地址，再除以 `page_size` 转换为以页为单位的索引，使 FlashAttention 内核能正确寻址分页 KV 缓存。

**`forward` 方法：**

```python
# fa.py 第 48–65 行
def forward(self, q, k, v, layer_id, batch) -> torch.Tensor:
    metadata = batch.attn_metadata
    assert isinstance(metadata, FAMetadata)
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
    return _fa_sgl_impl(
        q=q,
        k_cache=self.kvcache.k_cache(layer_id),
        v_cache=self.kvcache.v_cache(layer_id),
        page_table=metadata.page_table,
        cache_seqlens=metadata.cache_seqlens,
        cu_seqlens_q=metadata.cu_seqlens_q,
        cu_seqlens_k=metadata.cu_seqlens_k,
        max_seqlen_q=metadata.max_seqlen_q,
        softmax_scale=self.scale,
        version=self.version,
    )
```

先调用 `kvcache.store_kv` 将新计算出的 K/V 写入缓存，再调用 `_fa_sgl_impl` 执行注意力计算（此时 KV 缓存已包含当前 token）。

**CUDA Graph 相关方法：**

`init_capture_graph`：预分配 `FACaptureData`，其 `page_table` 形状为 `(max_bs, max_seq_len // page_size)`，所有张量在 CUDA Graph 录制期间保持固定地址。

`prepare_for_capture`：创建指向静态缓冲区切片的 `FAMetadata`，`max_seqlen_q=1` 固定为解码模式。

`prepare_for_replay`：

```python
# fa.py 第 128–136 行
def prepare_for_replay(self, batch: Batch) -> None:
    metadata, bs = batch.attn_metadata, batch.padded_size
    assert isinstance(metadata, FAMetadata)
    assert self.capture is not None and bs in self.capture_bs
    table_len = metadata.page_table.size(1)
    self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
    self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
    self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)
```

通过 `copy_` 原地更新静态缓冲区的内容，CUDA Graph 在回放时读取到的是最新的运行时数据。注意 `cu_seqlens_q` 不需要更新，因为解码阶段其值始终为 `[0, 1, 2, ..., bs]`（`init_capture_graph` 时已初始化好且无需修改）。

#### `_fa_sgl_impl` 函数

```python
# fa.py 第 139–182 行
def _fa_sgl_impl(
    q, k_cache, v_cache, page_table, cache_seqlens,
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, softmax_scale,
    version, sm_margin=0, window_size=(-1, -1), softcap=0.0,
    num_splits=0, pack_gqa=None, causal=True,
) -> torch.Tensor:
```

该函数是对 `sgl_kernel.flash_attn.flash_attn_with_kvcache` 的薄封装：

- 延迟导入（`try/except ImportError`）以提供友好的安装提示
- 将 `cu_seqlens_k` 作为 `cu_seqlens_k_new` 参数传递，与 FlashAttention 内核接口保持一致
- `version=3/4` 直接透传，控制内核使用 FA3 还是 FA4 实现
- `window_size=(-1, -1)` 表示无限上下文窗口，`softcap=0.0` 表示不使用 logit cap，`causal=True` 表示因果注意力

---

### G.1.4 `fi.py` — FlashInfer 后端

#### 文件职责

封装 `flashinfer` 库的分页 KV 缓存注意力 API，实现预填充（`BatchPrefillWithPagedKVCacheWrapper`）和解码（`BatchDecodeWithPagedKVCacheWrapper`）的 plan/run 两阶段执行模式，并支持 CUDA Graph。

#### `FIMetadata`

```python
# fi.py 第 47–77 行
@dataclass
class FIMetadata(BaseAttnMetadata):
    cu_seqlens_q_cpu:   torch.Tensor  # CPU 上，形状 (bs+1,)
    cu_seqlens_k_cpu:   torch.Tensor  # CPU 上，形状 (bs+1,)
    cu_seqlens_q_gpu:   torch.Tensor  # GPU 上，形状 (bs+1,)
    indices:            torch.Tensor  # GPU 上，KV 缓存物理页索引，形状 (total_pages,)
    last_page_len_cpu:  torch.Tensor  # CPU 上，形状 (bs,)，全为 1
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          Literal[1]   # 当前仅支持 page_size=1
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor  # CPU 上，形状 (bs,)
    dtype:              torch.dtype
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized:        bool = False
```

与 FA/TRTLLM 后端不同，`FIMetadata` 同时维护 CPU 和 GPU 两份数据：`flashinfer` 的 `plan` 接口需要 CPU 张量（以 non-blocking 方式传输），而 `get_last_indices` 需要 GPU 张量。

`__post_init__` 中包含严格的断言检查，验证各张量确实在预期设备上。

`page_size` 固定为 `Literal[1]`，这是当前 FlashInfer 后端的约束（`__post_init__` 中强制断言）。

#### `FlashInferBackend`

**构造函数与工作区设计：**

```python
# fi.py 第 81–113 行
def __init__(self, config: ModelConfig) -> None:
    self.float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=self.device
    )
    self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
        self.float_workspace_buffer, kv_layout="NHD", backend="fa2",
    )
    self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
        self.float_workspace_buffer,
        use_tensor_cores=self.use_tensor_cores,
        kv_layout="NHD", backend="fa2",
    )
    # NOTE: 复用 int_workspace_buffer
    self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
    self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer
```

**关键设计：**
- 预填充和解码共用同一块 128 MB 的 `float_workspace_buffer`，节省显存
- 通过访问私有属性 `_int_workspace_buffer` 强制复用整数工作区，注释中标注为 `# NOTE: some hack`
- 两种 wrapper 均使用 `backend="fa2"`，而非 `fa3`，注释说明"flashinfer fa3 is slow, use fa2 instead"

**`use_tensor_cores` 属性：**

```python
# fi.py 第 231–237 行
@cached_property
def use_tensor_cores(self) -> bool:
    if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
        logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
        return overriden_value
    GQA = self.config.num_qo_heads // self.config.num_kv_heads
    return GQA >= 4
```

当 GQA（分组查询注意力）比例大于等于 4 时自动启用 Tensor Core 路径（如 Llama-3 的 GQA=8），否则使用标准路径。可通过环境变量覆盖。

**plan/run 两阶段 API：**

FlashInfer 的核心设计是将注意力计算分为两个阶段：

**阶段 1：plan（规划）**

`plan` 在 `_initialize_metadata_once` 中延迟调用，且通过 `initialized` 标志保证只执行一次：

```python
# fi.py 第 121–161 行
@staticmethod
def _initialize_metadata_once(metadata: FIMetadata) -> None:
    if metadata.initialized:
        return
    metadata.initialized = True
    if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
        metadata.wrapper.plan(
            indptr=metadata.cu_seqlens_k_cpu,
            indices=metadata.indices,
            last_page_len=metadata.last_page_len_cpu,
            num_qo_heads=metadata.num_qo_heads,
            num_kv_heads=metadata.num_kv_heads,
            head_dim=metadata.head_dim,
            page_size=metadata.page_size,
            pos_encoding_mode=metadata.pos_encoding_mode,
            seq_lens=metadata.seq_lens_cpu,
            data_type=metadata.dtype,
            q_data_type=metadata.dtype,
            kv_data_type=metadata.dtype,
            non_blocking=True,
        )
    else:  # BatchPrefillWithPagedKVCacheWrapper
        metadata.wrapper.plan(
            qo_indptr=metadata.cu_seqlens_q_cpu,
            paged_kv_indptr=metadata.cu_seqlens_k_cpu,
            paged_kv_indices=metadata.indices,
            paged_kv_last_page_len=metadata.last_page_len_cpu,
            ...
            non_blocking=True,
            causal=True,
        )
```

`plan` 接受 CPU 张量并使用 `non_blocking=True` 异步传输到 GPU，计算并缓存内核执行所需的中间数据结构（如分块策略、内存布局等）。

预填充和解码的 `plan` 接口参数略有不同：解码不需要 `qo_indptr`（因为每个序列只有 1 个 Q token）；预填充需要 `qo_indptr` 并声明 `causal=True`。

**阶段 2：run（执行）**

```python
# fi.py 第 171–183 行
def forward(self, q, k, v, layer_id, batch) -> torch.Tensor:
    def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:
        return cache.view(-1, 1, cache.shape[2], cache.shape[3])

    metadata = batch.attn_metadata
    assert isinstance(metadata, FIMetadata)
    self._initialize_metadata_once(metadata)
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
    kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
    kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))
    return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)
```

`_flatten_cache` 将形状为 `(num_pages, page_size, num_kv_heads, head_dim)` 的 KV 缓存视图转换为 `(-1, 1, num_kv_heads, head_dim)`（等效于 `page_size=1` 的布局），这是由于 FlashInfer 后端当前仅支持 `page_size=1`。

**`prepare_metadata` 中的 `indices` 构造：**

```python
# fi.py 第 205–210 行（部分）
indices=torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs]),
```

与 FA 后端不同，FlashInfer 使用的是**扁平化的一维 `indices`**（每个 token 对应一个物理页索引），而不是二维页表。因此通过 `torch.cat` 将所有请求的页索引拼接为一个长向量。

**`cu_seqlens_q` 三分支逻辑：** 与 FA 后端完全相同（见 G.1.3），不再赘述。

**CUDA Graph 支持：**

```python
# fi.py 第 239–259 行
def prepare_for_capture(self, batch: Batch) -> None:
    from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper
    bs = batch.size
    capture = self.capture
    self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
        self.float_workspace_buffer,
        kv_layout="NHD",
        use_tensor_cores=self.use_tensor_cores,
        indptr_buffer=capture.cu_seqlens_k[: bs + 1],
        indices_buffer=capture.indices,
        last_page_len_buffer=capture.one_tensor[:bs],
    )
    self.graph_wrappers[bs]._backend = "fa2"
    self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
    ...
```

FlashInfer 为 CUDA Graph 提供了专用的 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`，其特点是在构造时就绑定固定的缓冲区（`indptr_buffer`、`indices_buffer`、`last_page_len_buffer`），确保 Graph 录制期间内存地址不变。每个批次大小对应一个独立的 `graph_wrappers[bs]`。

`prepare_for_replay` 中不更新 `capture` 缓冲区，而是重新调用 `_initialize_metadata_once` 使 Graph wrapper 以最新的运行时数据重新 plan：

```python
# fi.py 第 261–266 行
def prepare_for_replay(self, batch: Batch) -> None:
    metadata, bs = batch.attn_metadata, batch.padded_size
    assert isinstance(metadata, FIMetadata) and not metadata.initialized
    assert self.capture is not None and bs in self.capture_bs
    metadata.wrapper = self.graph_wrappers[bs]
    self._initialize_metadata_once(metadata)
```

注意此处断言 `not metadata.initialized`：每次迭代会创建新的 `FIMetadata` 对象（`prepare_metadata` 中），其 `initialized` 默认为 `False`，确保每次回放都触发一次新的 `plan`。

**`_get_ones_cpu` 辅助方法：**

```python
# fi.py 第 163–169 行
def _get_ones_cpu(self, bs: int) -> torch.Tensor:
    if bs <= len(self.cached_ones_cpu):
        return self.cached_ones_cpu[:bs]
    next_len = _next_power_of_2(bs)
    self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
    return self.cached_ones_cpu[:bs]
```

`last_page_len` 对于 `page_size=1` 的情况始终为全 1，该方法缓存 pin_memory 的全 1 张量并按需扩容（扩容时对齐到 2 的幂次方）以减少内存分配开销。

---

### G.1.5 `trtllm.py` — TensorRT-LLM 后端

#### 文件职责

封装 `flashinfer.decode.trtllm_batch_decode_with_kv_cache` 和 `flashinfer.prefill.trtllm_batch_context_with_kv_cache`，实现使用 TensorRT-LLM 融合内核的注意力后端。

#### `TRTLLMMetadata`

```python
# trtllm.py 第 22–32 行
@dataclass
class TRTLLMMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int
    page_table: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1
```

结构与 `FAMetadata` 完全相同，同样通过 `cu_seqlens_q` 计算最后 token 索引。

#### `TensorRTLLMBackend`

**构造函数：**

```python
# trtllm.py 第 35–47 行
def __init__(self, config: ModelConfig):
    ctx = get_global_ctx()
    self.config = config
    self.kvcache = ctx.kv_cache
    self.page_size = ctx.page_size
    self.capture: TRTLLMCaptureData | None = None
    self.max_graph_bs = 0
    self.capture_bs: List[int] = []
    self.scale = config.head_dim**-0.5
    self.workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=self.kvcache.device
    )
```

与 FlashInfer 后端类似，预分配 128 MB 的工作区缓冲区，供 TRT-LLM 内核使用。

**`forward` 方法——预填充/解码分支：**

```python
# trtllm.py 第 49–89 行
def forward(self, q, k, v, layer_id, batch) -> torch.Tensor:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    from flashinfer.prefill import trtllm_batch_context_with_kv_cache

    metadata = batch.attn_metadata
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
    kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))

    if batch.is_prefill:
        return trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            block_tables=metadata.page_table,
            seq_lens=metadata.cache_seqlens,
            max_q_len=metadata.max_seqlen_q,
            max_kv_len=metadata.max_seqlen_k,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            cum_seq_lens_q=metadata.cu_seqlens_q,
            cum_seq_lens_kv=metadata.cu_seqlens_k,
            kv_layout="NHD",
            batch_size=batch.size,
            out_dtype=q.dtype,
        )
    else:
        return trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            block_tables=metadata.page_table,
            seq_lens=metadata.cache_seqlens,
            max_seq_len=metadata.max_seqlen_k,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            kv_layout="NHD",
            out_dtype=q.dtype,
        )
```

TRT-LLM 后端直接在 `forward` 中根据 `batch.is_prefill` 分支，而不像 `HybridBackend` 那样依赖外部路由。预填充调用 `trtllm_batch_context_with_kv_cache`（处理可变长度的上下文），解码调用 `trtllm_batch_decode_with_kv_cache`（每个序列 1 个 token）。

注意 TRT-LLM 内核的缩放设计：`bmm1_scale=self.scale`（即 `head_dim^{-0.5}`）用于 QK 乘法，`bmm2_scale=1.0` 用于 SV 乘法（softmax 后的注意力权重与 V 的乘法不需要额外缩放）。

**`prepare_metadata` 方法：** 与 FA 后端的实现完全相同，包含相同的三分支 `cu_seqlens_q` 逻辑和相同的页表转换方法，不再赘述。

**CUDA Graph 相关方法：** 与 FA 后端实现完全对称，`prepare_for_replay` 同样更新 `cu_seqlens_k`、`seq_lens` 和 `page_table` 三个字段。

**注意事项：** TRT-LLM 后端要求 `page_size` 必须为 16、32 或 64（在 `engine.py` 的 `_adjust_config` 中强制设置为 64），这与 FlashAttention/FlashInfer 后端可以使用更小 page_size 不同。

---

### G.1.6 `__init__.py` — 注册表与工厂函数

#### 文件职责

维护注意力后端注册表，提供 `create_attention_backend` 工厂函数（含混合后端逻辑），并定义 `validate_attn_backend` 用于启动参数校验。

#### `SUPPORTED_ATTENTION_BACKENDS` 注册表

```python
# __init__.py 第 15–19 行
class BackendCreator(Protocol):
    def __call__(self, config: ModelConfig) -> BaseAttnBackend: ...

SUPPORTED_ATTENTION_BACKENDS = Registry[BackendCreator]("Attention Backend")
```

`Registry` 是泛型键值存储，`BackendCreator` 是用 `Protocol` 定义的鸭子类型接口（可调用对象，接受 `ModelConfig` 返回 `BaseAttnBackend`）。

注册的后端：

```python
# __init__.py 第 22–40 行
@SUPPORTED_ATTENTION_BACKENDS.register("trtllm")
def create_trtllm_backend(config): return TensorRTLLMBackend(config)

@SUPPORTED_ATTENTION_BACKENDS.register("fi")
def create_fi_backend(config): return FlashInferBackend(config)

@SUPPORTED_ATTENTION_BACKENDS.register("fa")
def create_fa_backend(config): return FlashAttentionBackend(config)
```

所有注册均在模块加载时以装饰器形式完成，后端类的导入延迟到工厂函数执行时（避免未安装可选依赖导致的导入失败）。

#### `validate_attn_backend`

```python
# __init__.py 第 43–49 行
def validate_attn_backend(backend: str, allow_auto: bool = True):
    if backend != "auto":
        required_backends = backend.split(",") if "," in backend else [backend]
        SUPPORTED_ATTENTION_BACKENDS.assert_supported(required_backends)
    else:
        assert allow_auto, "auto is not allowed here"
    return backend
```

支持逗号分隔的混合后端名称校验。`allow_auto=True` 时允许 `"auto"` 字符串通过（由引擎在运行时解析）；当 `allow_auto=False`（如 `create_attention_backend` 内部调用时），`"auto"` 已被提前解析为具体名称。

#### `create_attention_backend` 工厂函数

```python
# __init__.py 第 52–68 行
def create_attention_backend(backend: str, config: ModelConfig) -> BaseAttnBackend:
    validate_attn_backend(backend, allow_auto=False)
    if "," in backend:
        assert backend.count(",") == 1, "Only one comma is allowed in hybrid backend"
        p_backend, d_backend = backend.split(",", 1)
        if p_backend != d_backend:
            logger.info(f"Using hybrid attention backend: prefill={p_backend}, decode={d_backend}")
            p_backend = create_attention_backend(p_backend, config)
            d_backend = create_attention_backend(d_backend, config)
            return HybridBackend(p_backend, d_backend)
        backend = p_backend  # 两者相同，回退到单一后端
        logger.warning(f"P/D attention backends are the same: {backend}, using single backend.")

    return SUPPORTED_ATTENTION_BACKENDS[backend](config)
```

逻辑：
1. 包含逗号：解析为 `prefill_backend,decode_backend` 格式
2. 若两者不同：递归创建各自的后端实例，包装为 `HybridBackend` 返回
3. 若两者相同：发出警告，回退为单一后端（避免不必要的双份初始化开销）
4. 不含逗号：直接从注册表查找并调用对应工厂函数

#### `auto` 模式的硬件感知选择

`auto` 关键字在 `engine.py` 的 `_adjust_config` 函数中被解析（调用 `validate_attn_backend` 和 `create_attention_backend` 之前）：

```python
# engine/engine.py 第 224–227 行
if config.attention_backend == "auto":
    backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
    override("attention_backend", backend)
    logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")
```

**选择策略：**

| GPU 架构 | `auto` 解析结果 | 理由 |
|----------|----------------|------|
| SM100+（Blackwell，如 B200） | `"trtllm"` | TRT-LLM 针对 Blackwell 架构深度优化，提供最优性能 |
| SM90+（Hopper，如 H100/H200） | `"fa,fi"`（混合） | FA3 在 Hopper 预填充性能卓越；FlashInfer 在解码阶段提供更好的张量核利用率 |
| SM90 以下（Ampere 等） | `"fi"` | FlashInfer 的通用性和兼容性最好，作为默认回退 |

此外，当后端包含 `"trtllm"` 且 `page_size` 不在 `{16, 32, 64}` 时，`page_size` 被强制设置为 64（TRT-LLM 内核对页大小有约束）。

---

## 第二节：分布式模块（`python/minisgl/distributed/`）

分布式子包由三个文件组成：

| 文件 | 职责 |
|------|------|
| `info.py` | 定义张量并行信息单例 `DistributedInfo` |
| `impl.py` | 定义通信后端抽象与具体实现（Torch 原生 / PyNCCL），以及 `DistributedCommunicator` 门面类 |
| `__init__.py` | 对外导出接口 |

分布式通信底层的 CUDA 实现位于 `python/minisgl/kernel/pynccl.py` 和 `python/minisgl/kernel/csrc/src/pynccl.cu`。

---

### G.2.1 `distributed/info.py` — 分布式信息单例

#### 文件职责

以进程内全局单例的形式存储当前进程的张量并行（Tensor Parallelism）信息，供整个系统的任意模块通过 `get_tp_info()` 访问。

#### `DistributedInfo`

```python
# info.py 第 6–15 行
@dataclass(frozen=True)
class DistributedInfo:
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0
```

**字段：**
- `rank`：当前进程在张量并行组中的编号（0 到 `size-1`）
- `size`：张量并行组的总进程数（`tp_size`）

**设计要点：**
- `frozen=True`：不可变数据类，创建后无法修改，避免意外改写
- `__post_init__` 包含合法性断言：`0 <= rank < size`
- `is_primary()`：判断当前进程是否为主进程（rank 0），用于控制只在主进程执行的操作（如日志输出、checkpoint 保存等）

#### 单例管理函数

```python
# info.py 第 18–38 行
_TP_INFO: DistributedInfo | None = None

def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)

def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO

def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO
```

**单例设计细节：**

- 模块级变量 `_TP_INFO` 以下划线前缀表示私有，不直接导出
- `set_tp_info`：带幂等性保护，第二次调用直接抛出 `RuntimeError`，防止多次初始化导致状态不一致
- `get_tp_info`：强制要求已初始化，否则抛出 `RuntimeError`，使调用方在未初始化时立刻得到明确错误而不是 `NoneType` 错误
- `try_get_tp_info`：宽松版本，返回 `None` 而不报错，供不确定初始化状态的代码使用（如条件判断）

**单进程的特殊情况：** 当 `size=1` 时，`rank=0`，`is_primary()` 始终返回 `True`，整个分布式层的通信操作均为空操作（见 `impl.py` 中的 `tp_size == 1` 短路逻辑）。

---

### G.2.2 `distributed/impl.py` — 通信原语实现

#### 文件职责

定义通信后端抽象 `DistributedImpl`、两种具体实现（PyTorch 原生的 `TorchDistributedImpl` 和自定义的 `PyNCCLDistributedImpl`），以及门面类 `DistributedCommunicator`（对外统一的通信接口）和 `enable_pynccl_distributed` 启用函数。

#### `DistributedImpl`（抽象基类）

```python
# impl.py 第 16–22 行
@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...
```

定义两个集合通信原语的接口。两个方法的约定均为原地操作（`all_reduce`）或返回新张量（`all_gather`）。

#### `TorchDistributedImpl`

```python
# impl.py 第 25–41 行
@dataclass
class TorchDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out
```

基于 `torch.distributed` 的参考实现，在 `tp_size == 1` 时短路返回，避免不必要的分布式通信开销。`all_gather` 仅沿第 0 维拼接，假设分片均沿 `dim=0` 进行（对应列并行线性层的输出特征分片）。

#### `PyNCCLDistributedImpl`

```python
# impl.py 第 44–60 行
@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info
        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result
```

委托给 `PyNCCLCommunicator` 对象（由 C++/CUDA 层实现）。`all_reduce` 直接原地操作；`all_gather` 先分配输出张量（使用 `x.new_empty` 保持设备和数据类型一致），再调用底层的 `comm.all_gather`。

#### `DistributedCommunicator`（门面类）

```python
# impl.py 第 63–70 行
class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)
```

**设计模式：插件栈。** `plugins` 是类变量（所有实例共享），初始化为只含 `TorchDistributedImpl()` 的列表。每次调用 `all_reduce`/`all_gather` 时使用栈顶（`plugins[-1]`）的实现。

调用 `enable_pynccl_distributed` 时，`PyNCCLDistributedImpl` 被追加到栈顶，此后所有通信操作自动切换到 PyNCCL 路径。调用 `destroy_distributed` 时，整个 `plugins` 列表被清空。

这种设计的优点是：
1. 默认使用 PyTorch 原生通信（无需任何额外安装）
2. PyNCCL 启用后自动覆盖，对调用方透明
3. 不需要条件判断，完全通过多态实现

#### `enable_pynccl_distributed`

```python
# impl.py 第 73–90 行
def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """Enable PyNCCL-based distributed communication for tensor parallelism."""
    if tp_info.size == 1:
        return
    from minisgl.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )
    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))
```

- 若 `tp_size == 1`（单卡），直接返回，不初始化 NCCL（节省开销）
- 调用 `init_pynccl` 初始化 NCCL 通信器（见 G.2.4）
- 将 `PyNCCLDistributedImpl` 追加到插件栈，覆盖默认的 Torch 实现

#### `destroy_distributed`

```python
# impl.py 第 93–97 行
def destroy_distributed() -> None:
    """Destroy all the distributed communication plugins."""
    DistributedCommunicator.plugins = []
```

清空插件列表，触发 Python GC 对 `PyNCCLDistributedImpl` 及其持有的 `PyNCCLCommunicator` 对象的析构，从而释放 NCCL 通信器和显存（通过 C++ 析构函数调用 `ncclCommDestroy` 和 `ncclMemFree`）。

---

### G.2.3 `kernel/pynccl.py` — PyNCCL Python 层封装

#### 文件职责

通过 TVM FFI 桥接机制加载编译好的 NCCL 动态库，并在 Python 层暴露 `PyNCCLCommunicator` 类型和 `init_pynccl` 工厂函数。

#### `PyNCCLCommunicator` 类型定义

```python
# kernel/pynccl.py 第 16–23 行（TYPE_CHECKING 分支）
class PyNCCLCommunicator:
    @abstractmethod
    def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
    @abstractmethod
    def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
    @abstractmethod
    def get_buffer(self) -> int: ...
```

此类型定义仅在类型检查（`TYPE_CHECKING`）时可见，运行时 `PyNCCLCommunicator = Any`，实际对象为 TVM FFI 注册的 C++ 对象（`NCCLWrapper`）。

#### 模块加载机制

```python
# kernel/pynccl.py 第 28–30 行
@functools.cache
def _load_nccl_module() -> Module:
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-lnccl"])
```

`load_aot` 编译并加载 `pynccl.cu`，链接 `-lnccl`（系统安装的 NCCL 库），返回 TVM FFI 模块对象。`@functools.cache` 确保编译只发生一次。

```python
# kernel/pynccl.py 第 33–42 行
@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("minisgl.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl
```

通过 `@tvm_ffi.register_object` 将 Python 类映射到 C++ 中注册的 `"minisgl.NCCLWrapper"` 对象类型，使 C++ 对象可以直接被 Python 代码持有和调用。

#### `init_pynccl` 函数

```python
# kernel/pynccl.py 第 45–78 行
def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(id_list, src=0, group=tp_cpu_group)
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(id_list, src=0, group=tp_cpu_group)

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)
```

**NCCL 初始化流程：**

1. Rank 0 调用 `module.create_nccl_uid()` 生成唯一的 `ncclUniqueId`（128 字节的随机标识符）
2. 通过 `torch.distributed.broadcast_object_list` 将该 ID 广播到所有 rank（使用 CPU 进程组而非 GPU 通信器，这是鸡生蛋问题的标准解法：GPU 通信器尚未建立，只能先用 CPU 通信）
3. 所有 rank 使用相同的 `nccl_id` 实例化 `NCCLWrapper`，内部调用 `ncclCommInitRank` 建立 GPU 通信器
4. `max_size_bytes` 受环境变量 `PYNCCL_MAX_BUFFER_SIZE` 上限约束，控制对称显存分配大小

---

### G.2.4 `kernel/csrc/src/pynccl.cu` — NCCL C++ 实现

#### 文件职责

通过 TVM FFI 对象系统实现 `NCCLWrapper`，封装 NCCL 集合通信原语，提供带对称显存优化的 `all_reduce` 和直接输出的 `all_gather`。

#### `NCCLWrapper` 构造函数

```cpp
// pynccl.cu 第 72–91 行
NCCLWrapper(int rank, int world_size, const size_t max_bytes, NCCLIDList uid)
    : m_rank(rank), m_world_size(world_size), m_max_bytes(max_bytes) {
    ncclUniqueId id = get_uid(uid);
    ncclComm_t comm;
    NCCL_CHECK(::ncclCommInitRank(&comm, m_world_size, id, m_rank));
    m_comm = {comm, template_fn<::ncclCommDestroy>};

    void *buf;
    NCCL_CHECK(::ncclMemAlloc(&buf, max_bytes));
    m_sym_mem = {buf, template_fn<::ncclMemFree>};

    ncclWindow_t win;
    NCCL_CHECK(::ncclCommWindowRegister(comm, buf, max_bytes, &win,
                                        NCCL_WIN_COLL_SYMMETRIC));
    m_win = {win, [comm = m_comm](ncclWindow_t w) {
                return NCCL_CHECK(::ncclCommWindowDeregister(comm.get(), w));
            }};
}
```

**初始化步骤：**
1. `ncclCommInitRank`：创建 NCCL 通信器（进程间同步点）
2. `ncclMemAlloc`：分配 `max_bytes` 字节的**对称显存**（symmetric memory），这是 NCCL 2.x 引入的特性，所有 rank 的该内存块具有相同的虚拟地址，支持跨 GPU 直接访问
3. `ncclCommWindowRegister`：将对称显存注册为 NCCL 窗口，标记为 `NCCL_WIN_COLL_SYMMETRIC`，允许内核进行无需显式 GPU-GPU 拷贝的集合操作

所有资源（通信器、显存、窗口）均使用 `shared_ptr` 管理，析构时自动调用对应的 NCCL 释放函数。

#### `all_reduce` 实现

```cpp
// pynccl.cu 第 93–134 行
auto all_reduce(tvm::ffi::TensorView t, std::string op) const -> void {
    const auto size_bytes = size_dim * (t.dtype().bits / 8);
    const auto reduce_op = kNCCLReduceOPMap.at(op);
    const auto stream = LaunchKernel::resolve_device(t.device());

    if (size_bytes <= m_max_bytes) {  // 使用内部对称显存缓冲区
        const auto buf_ptr = m_sym_mem.get();
        const auto need_memcpy = (buf_ptr != data_ptr);
        if (need_memcpy) {
            CUDA_CHECK(::cudaMemcpyAsync(buf_ptr, data_ptr, size_bytes,
                                         ::cudaMemcpyDeviceToDevice, stream));
        }
        NCCL_CHECK(::ncclAllReduce(buf_ptr, buf_ptr, size_dim, dtype,
                                    reduce_op, m_comm.get(), stream));
        if (need_memcpy) {
            CUDA_CHECK(::cudaMemcpyAsync(data_ptr, buf_ptr, size_bytes,
                                         ::cudaMemcpyDeviceToDevice, stream));
        }
    } else {  // 数据超出缓冲区大小，直接使用原始指针
        NCCL_CHECK(::ncclAllReduce(data_ptr, data_ptr, size_dim, dtype,
                                    reduce_op, m_comm.get(), stream));
    }
}
```

**对称显存优化路径（`size_bytes <= m_max_bytes`）：**

当数据量较小（如张量并行的 AllReduce 通常只涉及隐藏层激活值的一部分）时，先将数据拷贝到对称显存缓冲区，在该缓冲区上执行 AllReduce，再拷回原始张量。使用对称显存的优势是 NCCL 可以利用 NVLink 的 peer-to-peer 访问特性进行低延迟通信，而不需要通过主机内存中转。

**回退路径：** 当数据量超出预分配缓冲区时，直接对原始张量指针执行 AllReduce，与标准 NCCL 用法相同。

**数据类型支持：** 仅支持 `float16` 和 `bfloat16`（`kNCCLDtypeMap` 中只有这两种映射），与 LLM 推理的主流数值精度一致。

#### `all_gather` 实现

```cpp
// pynccl.cu 第 136–161 行
auto all_gather(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) const -> void {
    RuntimeCheck(dst.size(0) == src.size(0) * m_world_size, "Destination tensor has incorrect size");
    const auto size_dim = static_cast<size_t>(src.shape().Product());
    // do not use internal buffer for all_gather, directly gather to output tensor
    NCCL_CHECK(::ncclAllGather(
        src_ptr, dst_ptr, size_dim, dtype, m_comm.get(), stream));
}
```

`all_gather` **不使用**对称显存缓冲区，直接将各 rank 的 `src` 汇聚到 `dst`。注释明确说明理由："do not use internal buffer for all_gather, directly gather to output tensor"。这是因为 AllGather 的输出大小是输入的 `world_size` 倍，通常超出缓冲区大小，且 AllGather 不涉及跨 rank 的归约操作，不需要对称显存的访问特性。

#### 为何绕过 PyTorch 内部 NCCL

mini-sglang 选择直接调用 NCCL C API 而非使用 PyTorch 的 `torch.distributed` NCCL 后端，原因如下：

1. **避免 GIL 干扰：** PyTorch 的 NCCL 操作通过 Python 层调度，在多线程场景下受 GIL 影响；直接 C++ 调用完全绕过 GIL
2. **控制流灵活性：** 直接 API 允许精确控制 CUDA stream（通过 `LaunchKernel::resolve_device` 获取当前设备流），而 `torch.distributed` 使用内部维护的 NCCL stream，可能与推理的主计算流产生同步开销
3. **对称显存访问：** PyTorch 的 `dist.all_reduce` 不支持显式使用 `ncclMemAlloc` 分配的对称显存；自定义实现可以充分利用 NCCL 2.x 的对称显存特性降低延迟
4. **与 CUDA Graph 兼容：** CUDA Graph 要求所有操作（包括通信）都在录制期间记录到 stream 中；PyTorch 的某些分布式操作会触发 CPU 同步，与 CUDA Graph 不兼容；直接 NCCL 调用则完全异步，可以安全地录制到 Graph 中
5. **最小化依赖：** 仅依赖 `libnccl.so`（通过 `extra_ldflags=["-lnccl"]` 动态链接），不依赖 PyTorch 内部的分布式基础设施，易于独立测试和部署

#### TVM FFI 注册

```cpp
// pynccl.cu 第 177–186 行
TVM_FFI_STATIC_INIT_BLOCK() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<NCCLWrapper>()
        .def(refl::init<int, int, size_t, NCCLIDList>(), "__init__")
        .def("all_reduce", &NCCLWrapper::all_reduce)
        .def("all_gather", &NCCLWrapper::all_gather)
        .def("get_buffer", &NCCLWrapper::get_buffer);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(create_nccl_uid, &create_uid);
```

通过 TVM FFI 反射机制将 `NCCLWrapper` 的构造函数和三个方法导出为可从 Python 调用的 FFI 函数。`create_nccl_uid` 作为独立函数导出，供 Python 的 `init_pynccl` 在 Rank 0 调用以生成唯一 ID。

---

## 总结

### 注意力后端架构图

```
create_attention_backend("auto" → 经 _adjust_config 解析)
         │
         ├── SM100: "trtllm"  ──→ TensorRTLLMBackend
         │
         ├── SM90:  "fa,fi"   ──→ HybridBackend
         │                           ├── prefill: FlashAttentionBackend (FA3/FA4)
         │                           └── decode:  FlashInferBackend
         │
         └── 其他:  "fi"      ──→ FlashInferBackend
```

### 注意力后端接口语义对照

| 方法 | 调用时机 | 主要职责 |
|------|---------|---------|
| `prepare_metadata` | 每次调度迭代开始 | 根据批次信息构建后端专属元数据 |
| `forward` | 每个 Transformer 层 | 写 KV 缓存 + 执行分页注意力 |
| `init_capture_graph` | 引擎启动时（一次） | 为 CUDA Graph 预分配固定缓冲区 |
| `prepare_for_capture` | 每个 bs 的 Graph 录制前 | 将 batch 元数据指向静态缓冲区 |
| `prepare_for_replay` | 每次 Graph 回放前 | 将运行时数据复制到静态缓冲区 |

### 分布式通信架构

```
DistributedCommunicator.all_reduce/all_gather
         │
         └── plugins[-1]（栈顶）
                ├── 初始：TorchDistributedImpl（torch.distributed）
                └── enable_pynccl_distributed 后：PyNCCLDistributedImpl
                         └── PyNCCLCommunicator（TVM FFI → NCCLWrapper C++）
                                  ├── all_reduce：对称显存优化路径 / 直接路径
                                  └── all_gather：直接输出路径
```
