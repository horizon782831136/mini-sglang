# 附录 B：核心数据结构与消息系统

本附录对 mini-sglang 项目中核心数据结构模块（`core.py`）以及全部消息子系统（`message/` 包）进行系统性的代码级说明。内容包含每个公开类与函数的完整签名、字段含义、实现细节、模块依赖关系，以及带文件路径和行号的代码片段。

---

## B.1 总览：消息流转链路

在深入各模块之前，先建立对整个系统消息流的直觉。mini-sglang 采用多进程架构，前端（API Server）、Tokenizer Worker 和后端（Scheduler/Engine）三者通过 ZMQ 队列异步通信。消息流如下：

```
用户 HTTP 请求
    │
    ▼
[前端进程] FrontendManager
    │  发送 TokenizeMsg （含原始文本 + SamplingParams）
    │  ──────────────────────────────────────────▶  ZMQ (tokenizer_addr)
    │
    │                              [Tokenizer Worker]
    │                               接收 TokenizeMsg
    │                               调用 tokenizer.encode()
    │                               构造 UserMsg（含 input_ids tensor）
    │                               ──────────────────────────────▶  ZMQ (backend_addr)
    │
    │                                              [Scheduler 进程]
    │                                               接收 UserMsg
    │                                               调度 Prefill/Decode
    │                                               产出 next_token
    │                                               构造 DetokenizeMsg
    │                               ◀──────────────────────────────  ZMQ (detokenizer_addr)
    │
    │                              [Tokenizer Worker]
    │                               接收 DetokenizeMsg
    │                               调用 tokenizer.decode()
    │                               构造 UserReply（含增量文本）
    │  ◀──────────────────────────────────────────  ZMQ (frontend_addr)
    │
    ▼
流式/非流式 HTTP 响应返回给用户
```

消息类型的完整分层结构：

| 消息分层 | 基类 | 具体类型 |
|---------|------|---------|
| Tokenizer 入口消息 | `BaseTokenizerMsg` | `TokenizeMsg`, `DetokenizeMsg`, `AbortMsg`, `BatchTokenizerMsg` |
| 后端（Scheduler）消息 | `BaseBackendMsg` | `UserMsg`, `ExitMsg`, `AbortBackendMsg`, `BatchBackendMsg` |
| 前端（API Server）消息 | `BaseFrontendMsg` | `UserReply`, `BatchFrontendMsg` |

---

## B.2 `python/minisgl/core.py` — 核心数据类

### 文件职责

定义系统中最基础的共享数据结构：采样参数（`SamplingParams`）、推理请求（`Req`）、批次（`Batch`）以及全局推理上下文（`Context`）。这些类型是 Scheduler、Engine、KV Cache 等所有后端模块之间传递信息的"通用语言"。

### 依赖关系

```
core.py
  ├── 被 message/backend.py 引用（SamplingParams）
  ├── 被 message/tokenizer.py 引用（SamplingParams）
  ├── 被 scheduler/* 引用（Req, Batch）
  ├── 被 engine/* 引用（Batch, Context）
  ├── 被 attention/* 引用（Batch, BaseAttnMetadata）
  └── 被 kvcache/* 引用（BaseCacheHandle, BaseKVCachePool）
```

---

### B.2.1 `SamplingParams`

**文件**：`python/minisgl/core.py`，第 16–25 行

```python
# python/minisgl/core.py  L16-25
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
```

**功能说明**

`SamplingParams` 是一个纯数据类，封装了推理时控制 token 采样行为的所有超参数。它在前端构造（`api_server.py`），随消息链路一路传递至 Scheduler 并最终传入 `Engine.sampler`。

**字段详解**

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `temperature` | `float` | `0.0` | 采样温度。`<= 0.0` 表示贪心解码，不对 logits 做缩放；大于 0 时 logits 除以该值后 softmax，值越大输出越随机。 |
| `top_k` | `int` | `-1` | Top-K 采样：只从概率最高的 K 个 token 中采样。`-1` 表示不限制（使用全词表）；`1` 等价于贪心。 |
| `top_p` | `float` | `1.0` | Top-P（nucleus）采样：从累积概率达到 `top_p` 的最小集合中采样。`1.0` 表示不限制。 |
| `ignore_eos` | `bool` | `False` | 是否忽略 EOS token。设为 `True` 时，即使模型输出 EOS 也不会停止，直到达到 `max_tokens`。用于基准测试等场景。 |
| `max_tokens` | `int` | `1024` | 最大输出 token 数。Scheduler 会在超出限制时强制结束该请求。 |

**`is_greedy` 属性**

```python
@property
def is_greedy(self) -> bool:
    return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
```

判断当前配置是否等价于贪心解码。条件为：（温度为 0 或 top_k 为 1）且 top_p 为 1.0。Engine 的采样器会利用这个属性走不同的代码路径（贪心路径直接取 argmax，避免随机采样开销）。

**设计决策**

- 使用 `@dataclass` 而非普通类，配合 `__init__`、`__repr__` 的自动生成，方便调试和序列化（`serialize_type` 函数通过 `__dict__` 遍历字段）。
- 字段默认值覆盖了最常见的贪心场景，外部只需覆盖需要修改的参数，符合"最小惊讶"原则。

---

### B.2.2 `Req`

**文件**：`python/minisgl/core.py`，第 28–68 行

```python
# python/minisgl/core.py  L28-68
@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle
```

**功能说明**

`Req` 是系统中单个推理请求在 Scheduler 内部的完整状态表示。它记录了请求在整个生命周期（从 Prefill 到最后一个 Decode 步骤）中的 token 序列、缓存命中状态和资源句柄。

**`eq=False` 的含义**

`@dataclass(eq=False)` 禁用自动生成的 `__eq__` 和 `__hash__`，使得 `Req` 实例默认使用 Python 的对象标识（`id()`）进行比较。这是关键设计：Scheduler 的 `finished_reqs: Set[Req]` 依赖对象标识来判断是否"同一个请求对象"，防止 Overlap Scheduling 场景下的重复释放（见 `scheduler.py` 第 159 行）。

**字段详解**

| 字段 | 类型 | 含义 |
|------|------|------|
| `input_ids` | `torch.Tensor`（CPU，1D int32） | 当前已知的完整 token 序列。Prefill 时为原始输入，每次 Decode 后通过 `append_host` 追加新 token。 |
| `table_idx` | `int` | 该请求在全局 `page_table` 中的行索引（即"槽位"编号）。每个槽位对应一组分配给该请求的 KV Cache page 指针。 |
| `cached_len` | `int` | 当前已完成注意力计算并写入 KV Cache 的 token 数量。`input_ids[:cached_len]` 部分已在 cache 中，下次计算时可直接跳过。 |
| `output_len` | `int` | 允许输出的最大 token 数（来源于 `SamplingParams.max_tokens`，但可能被 Scheduler 截断）。 |
| `uid` | `int` | 全局唯一请求 ID，由 `FrontendManager.new_user()` 分配，贯穿整个消息链路，用于跨进程关联请求与响应。 |
| `sampling_params` | `SamplingParams` | 该请求的采样超参数。 |
| `cache_handle` | `BaseCacheHandle` | KV Cache 前缀匹配的句柄，记录了从 Prefix Cache 命中的起始位置和对应 page 索引。 |

**`__post_init__` 约束**

```python
# python/minisgl/core.py  L38-42
def __post_init__(self) -> None:
    assert self.input_ids.is_cpu
    self.device_len = len(self.input_ids)
    self.max_device_len = len(self.input_ids) + self.output_len
    assert 0 <= self.cached_len < self.device_len <= self.max_device_len
```

`__post_init__` 在 `__init__` 之后自动调用，完成以下工作：
- 断言 `input_ids` 在 CPU（Scheduler 在 CPU 侧管理 token 序列，GPU 侧通过 `token_pool` 另行维护）。
- 设置两个"运行时字段"：
  - `device_len`：当前序列总长度（初始等于 `input_ids` 长度）。
  - `max_device_len`：允许的最大序列长度（`input_len + output_len`）。
- 断言 `cached_len < device_len <= max_device_len`，防止非法状态。

**状态机与关键属性**

`Req` 的生命周期可以用以下状态机描述：

```
[创建]  cached_len < device_len  (有待计算的 token → Prefill 阶段)
   │
   │  complete_one() 被调用
   ▼
[Decode 进行中]  cached_len == device_len，remain_len > 0
   │
   │  append_host(next_token) → device_len++
   │  complete_one() → cached_len++ (追上 device_len)
   │  ... 反复 ...
   │
   ▼
[完成]  remain_len == 0  或  next_token == eos_token_id
```

| 属性/方法 | 计算公式 | 含义 |
|----------|---------|------|
| `remain_len` | `max_device_len - device_len` | 剩余可生成 token 数。为 0 时请求应结束。 |
| `extend_len` | `device_len - cached_len` | 本次 Prefill/Decode 步骤需要实际计算的 token 数（未命中 cache 的部分）。 |
| `can_decode` | `remain_len > 0` | 是否还有剩余输出配额，可继续 Decode。 |
| `complete_one()` | `cached_len = device_len; device_len += 1` | 表示刚刚完成了当前 token 的 Attention 计算，`cached_len` 追上 `device_len`，同时为下一个待生成的 token 占位（`device_len += 1`）。 |
| `append_host(next_token)` | `input_ids = cat([input_ids, next_token])` | 将 Scheduler 采样得到的新 token 追加到 CPU 侧 `input_ids` 中，保持序列完整性。 |

**`complete_one` 与 `append_host` 的协作**

```python
# python/minisgl/core.py  L52-57
def complete_one(self) -> None:
    self.cached_len = self.device_len
    self.device_len += 1

def append_host(self, next_token: torch.Tensor) -> None:
    self.input_ids = torch.cat([self.input_ids, next_token])
```

在 Scheduler 的 `_process_last_data` 中（`scheduler.py` 第 151 行），每个 Decode 步骤结束后先调用 `append_host` 将新 token 追加到 CPU 序列，再由 Decode Manager 内部调用 `complete_one` 更新缓存指针。两者配合完成一个完整的 token 生成周期。

**`__repr__`**

```python
# python/minisgl/core.py  L63-68
def __repr__(self) -> str:
    return (
        f"{type(self)}(table_idx={self.table_idx}, "
        f"cached_len={self.cached_len}, device_len={self.device_len}, "
        f"max_device_len={self.max_device_len})"
    )
```

调试友好的字符串表示，只显示最关键的长度信息，不打印完整 token 序列（避免日志过长）。

---

### B.2.3 `Batch`

**文件**：`python/minisgl/core.py`，第 71–98 行

```python
# python/minisgl/core.py  L71-98
@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)
```

**功能说明**

`Batch` 是 Scheduler 向 Engine 提交一次 forward pass 所需信息的容器。它将多个 `Req` 打包，并由 Scheduler 在 `_prepare_batch` 方法中填充所有 `field(init=False)` 字段，再传入 `engine.forward_batch()`。

**字段详解**

构造参数（由调用者传入）：

| 字段 | 类型 | 含义 |
|------|------|------|
| `reqs` | `List[Req]` | 参与本批次的真实请求列表（不含 padding 请求）。 |
| `phase` | `Literal["prefill", "decode"]` | 批次类型。Prefill 批次包含至少一个需要做 Attention 的完整/分块输入；Decode 批次每个请求只计算新生成的 1 个 token。 |

由 Scheduler 填充（`field(init=False)`）：

| 字段 | 类型 | 填充位置 | 含义 |
|------|------|---------|------|
| `input_ids` | `torch.Tensor`（GPU） | `scheduler._forward()` | 从 `token_pool` 中按 `input_tuple` 索引取出的 GPU token id 张量，shape 为 `(total_extend_tokens,)`。 |
| `positions` | `torch.Tensor`（GPU，int32） | `scheduler._prepare_batch()` | 每个 token 在序列中的绝对位置编号，用于 RoPE 等位置编码。shape 同 `input_ids`。 |
| `out_loc` | `torch.Tensor`（GPU） | `scheduler._prepare_batch()` | 每个 token 计算完成后 KV 应写入的 page 槽位索引，由 `page_table[input_mapping]` 计算。 |
| `padded_reqs` | `List[Req]` | `engine.graph_runner.pad_batch()` | 含 padding 请求的列表。CUDA Graph Capture 要求固定 batch size，padding 请求用来凑齐。 |
| `attn_metadata` | `BaseAttnMetadata` | `attn_backend.prepare_metadata()` | 注意力后端所需的元数据（如 FlashAttention 的 cu_seqlens、block table 等），由各 backend 子类具体定义。 |

**属性**

| 属性 | 计算 | 含义 |
|------|------|------|
| `is_prefill` | `phase == "prefill"` | 是否为 Prefill 批次。 |
| `is_decode` | `phase == "decode"` | 是否为 Decode 批次。 |
| `size` | `len(reqs)` | 真实请求数（不含 padding）。 |
| `padded_size` | `len(padded_reqs)` | 含 padding 的请求数，对应 CUDA Graph 的实际 batch size。 |

**设计决策**

- `phase` 字段是字符串字面量类型而非枚举，保证了序列化透明性，同时 `Literal` 注解提供了静态类型检查。
- `field(init=False)` 的字段采用"延迟填充"模式：`Batch` 对象先用最少信息构造，各模块按需填充自己负责的字段，形成流水线式的准备过程。这避免了构造时的循环依赖。
- Scheduler 的 `_prepare_batch`（`scheduler.py` 第 204–217 行）是填充这些字段的主要场所，执行顺序为：pad → allocate paged KV → make positions → make input/write tuples → prepare attn metadata。

---

### B.2.4 `Context`

**文件**：`python/minisgl/core.py`，第 100–136 行

```python
# python/minisgl/core.py  L100-136
@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    _batch: Batch | None = field(default=None, init=False)
```

**功能说明**

`Context` 是 Engine 内部的全局状态容器，以单例模式通过 `set_global_ctx` / `get_global_ctx` 暴露给所有层（Layer）。它持有所有 forward pass 所需的共享资源引用（KV cache、Attention backend 等），并通过上下文管理器安全地绑定当前正在处理的 `Batch`。

**字段详解**

| 字段 | 类型 | 含义 |
|------|------|------|
| `page_size` | `int` | KV Cache 的分页大小（每页包含的 token 数）。用于对齐地址计算。 |
| `page_table` | `torch.Tensor` | 全局 page 映射表，`shape = (max_running_req, max_seq_len)`。每行对应一个请求槽位，每列存储该位置对应的 KV Cache 物理 page 索引。注释指出该表在逻辑上始终以 `page_size=1` 视角存储（token 级别），由上层模块负责将 page 级地址转换为 token 级地址。 |
| `attn_backend` | `BaseAttnBackend` | 注意力计算后端（FlashAttention / FlashInfer / TRT-LLM 等），在 Engine 初始化时填充。 |
| `moe_backend` | `BaseMoeBackend` | MoE（Mixture of Experts）路由后端，用于 Qwen3-MoE 等模型。 |
| `kv_cache` | `BaseKVCachePool` | KV Cache 物理存储池，提供 `store_kv`、`k_cache`、`v_cache` 接口。 |
| `_batch` | `Batch \| None` | 当前正在处理的 batch（私有字段）。通过 `forward_batch` 上下文管理器管理生命周期。 |

**`batch` 属性与 `forward_batch` 上下文管理器**

```python
# python/minisgl/core.py  L111-122
@property
def batch(self) -> Batch:
    assert self._batch is not None, "No active batch in context"
    return self._batch

@contextmanager
def forward_batch(self, batch: Batch):
    assert self._batch is None, "Nested forward_batch is not allowed"
    try:
        self._batch = batch
        yield
    finally:
        self._batch = None
```

`forward_batch` 是 `Context` 最关键的接口。它在 Engine 执行 `forward_batch` 时将 `Batch` 绑定到全局上下文，使得深层的 Attention Layer 可以通过 `get_global_ctx().batch` 访问当前批次信息（如 `attn_metadata`、`out_loc`），无需逐层传递。`finally` 块保证即使 forward 过程抛出异常，`_batch` 也会被重置，防止状态泄漏。断言"不允许嵌套调用"则防止了递归 forward 等异常场景。

**全局单例管理**

```python
# python/minisgl/core.py  L125-136
_GLOBAL_CTX: Context | None = None

def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx

def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
```

`_GLOBAL_CTX` 是模块级私有变量，仅允许设置一次（再次调用 `set_global_ctx` 会触发断言），保证了单 GPU 进程中有且只有一个推理上下文。`get_global_ctx` 对应获取接口，调用者无需了解 `Engine` 的内部结构，降低了各层与引擎的耦合度。

---

## B.3 `python/minisgl/message/__init__.py` — 消息包导出

**文件**：`python/minisgl/message/__init__.py`，第 1–19 行

```python
# python/minisgl/message/__init__.py  L1-19
from .backend import AbortBackendMsg, BaseBackendMsg, BatchBackendMsg, ExitMsg, UserMsg
from .frontend import BaseFrontendMsg, BatchFrontendMsg, UserReply
from .tokenizer import AbortMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg, TokenizeMsg

__all__ = [
    "AbortMsg",
    "AbortBackendMsg",
    "BaseBackendMsg",
    "BatchBackendMsg",
    "ExitMsg",
    "UserMsg",
    "BaseTokenizerMsg",
    "BatchTokenizerMsg",
    "DetokenizeMsg",
    "TokenizeMsg",
    "BaseFrontendMsg",
    "BatchFrontendMsg",
    "UserReply",
]
```

**文件职责**

作为 `message` 包的公开接口，将三个子模块的所有消息类型统一导出，使调用方可以通过 `from minisgl.message import UserMsg, TokenizeMsg` 等简洁语句导入任意消息类型，无需了解内部子模块划分。

**设计说明**

`__all__` 列表明确限定了包的公开 API，防止外部代码意外依赖内部实现细节。三个子模块的职责边界清晰：
- `tokenizer.py`：定义在 API Server ↔ Tokenizer 之间流动的消息；
- `backend.py`：定义在 Tokenizer ↔ Scheduler 之间流动的消息；
- `frontend.py`：定义在 Tokenizer → API Server 之间的回复消息。

---

## B.4 `python/minisgl/message/frontend.py` — 前端消息类型

**文件**：`python/minisgl/message/frontend.py`，第 1–30 行

### 文件职责

定义从 Tokenizer Worker 发回 API Server（前端）的消息类型，以及通用的批量消息容器。这些消息承载已解码为文本的增量输出，供前端封装为 HTTP 流式响应或完整响应。

### 依赖关系

```
frontend.py
  ├── 依赖 message/utils.py（serialize_type, deserialize_type）
  └── 被 tokenizer/server.py 生产
      被 server/api_server.py 消费
```

---

### B.4.1 `BaseFrontendMsg`

```python
# python/minisgl/message/frontend.py  L9-17
@dataclass
class BaseFrontendMsg:
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)
```

**功能说明**

所有前端消息的抽象基类，提供序列化（`encoder`）和反序列化（`decoder`）静态方法。`encoder` 和 `decoder` 均以静态方法形式提供，方便作为 ZMQ 队列的 `encoder=` / `decoder=` 参数传入，无需实例化。

**`decoder` 的 `globals()` 参数**

`deserialize_type(globals(), json)` 将当前模块的全局命名空间作为类型注册表传入。`deserialize_type` 会读取 JSON 中的 `__type__` 字段（如 `"UserReply"`），再从 `globals()` 中查找同名类并实例化。因此，`frontend.py` 中定义的所有消息类型自动可被反序列化，无需额外注册。

---

### B.4.2 `BatchFrontendMsg`

```python
# python/minisgl/message/frontend.py  L20-22
@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    data: List[BaseFrontendMsg]
```

**功能说明**

批量前端消息容器，将多个 `BaseFrontendMsg`（通常为多个 `UserReply`）合并为一条 ZMQ 消息发送，减少进程间通信次数。

**字段详解**

| 字段 | 类型 | 含义 |
|------|------|------|
| `data` | `List[BaseFrontendMsg]` | 打包的消息列表。通常包含多个 `UserReply` 实例，对应同一个 Detokenize 批次的多个请求结果。 |

**使用模式**

在 `tokenizer/server.py` 第 83–85 行，当 Detokenize 结果超过 1 条时打包为 `BatchFrontendMsg` 发送；API Server 收到后在 `_unwrap_msg` 函数（`api_server.py` 第 43–51 行）中展开，逐一分发给对应请求的等待 event。

---

### B.4.3 `UserReply`

```python
# python/minisgl/message/frontend.py  L25-29
@dataclass
class UserReply(BaseFrontendMsg):
    uid: int
    incremental_output: str
    finished: bool
```

**功能说明**

从 Tokenizer Worker 发回 API Server 的单条用户响应，携带增量解码文本。API Server 的 `FrontendManager` 根据 `uid` 路由到对应请求的异步 event，实现流式输出。

**字段详解**

| 字段 | 类型 | 含义 |
|------|------|------|
| `uid` | `int` | 请求唯一 ID，与 `TokenizeMsg.uid` / `DetokenizeMsg.uid` 一致，贯穿整个请求生命周期。 |
| `incremental_output` | `str` | 自上次回复以来新生成的文本增量。注意是**增量**而非全量，调用方需自行累积。由 `DetokenizeManager` 的 `find_printable_text` 逻辑计算，处理了 Unicode surrogate 和单词边界问题。 |
| `finished` | `bool` | 是否为该请求的最后一条回复。API Server 收到 `finished=True` 后关闭对应的 SSE 流并清理资源。 |

**与流式响应的关联**

`api_server.py` 的 `stream_generate` 方法（第 153–159 行）通过 `wait_for_ack` 逐个 yield `UserReply`，将 `incremental_output` 包装为 SSE 格式推送给客户端：

```python
# python/minisgl/server/api_server.py  L153-159
async def stream_generate(self, uid: int):
    async for ack in self.wait_for_ack(uid):
        chunk = {"text": ack.incremental_output}
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
        if ack.finished:
            break
    yield b"data: [DONE]\n\n"
```

---

## B.5 `python/minisgl/message/backend.py` — 后端消息类型

**文件**：`python/minisgl/message/backend.py`，第 1–41 行

### 文件职责

定义从 Tokenizer Worker 发往 Scheduler（后端）以及 Scheduler 控制指令的消息类型。这些消息承载 token ID 张量和采样参数，是 Tokenizer ↔ Scheduler 通信的载体。

### 依赖关系

```
backend.py
  ├── 依赖 core.py（SamplingParams）
  ├── 依赖 message/utils.py（serialize_type, deserialize_type）
  ├── 被 tokenizer/server.py 生产（UserMsg, AbortBackendMsg, BatchBackendMsg）
  └── 被 scheduler/scheduler.py 消费（_process_one_msg）
      被 scheduler/io.py 路由（BaseBackendMsg.decoder）
```

---

### B.5.1 `BaseBackendMsg`

```python
# python/minisgl/message/backend.py  L12-19
@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)
```

**功能说明**

所有后端消息的抽象基类。与 `BaseFrontendMsg` 的设计略有不同：`encoder` 是**实例方法**而非静态方法，因为序列化时需要访问 `self` 的实际类型。`decoder` 仍为静态方法，用于从字节流恢复对象。

**与 `BaseFrontendMsg` 的编码器差异**

`BaseFrontendMsg.encoder` 是 `@staticmethod`，接收 `msg` 参数；`BaseBackendMsg.encoder` 是实例方法（`self`）。这是因为 ZMQ Push Queue 的 `encoder` 参数在不同地方的调用约定不同——在 `tokenizer/server.py` 中，后端消息通过 `send_backend.put(batch_output)` 发送，ZmqPushQueue 内部调用 `encoder(batch_output)` 即 `batch_output.encoder()`（实例方法调用）。

---

### B.5.2 `BatchBackendMsg`

```python
# python/minisgl/message/backend.py  L22-24
@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]
```

**功能说明**

批量后端消息容器，将多个 `UserMsg` 或 `AbortBackendMsg` 合并为一条 ZMQ 消息传输。Scheduler 的 `_process_one_msg` 方法在接收到 `BatchBackendMsg` 时会递归展开：

```python
# python/minisgl/scheduler/scheduler.py  L170-172
def _process_one_msg(self, msg: BaseBackendMsg) -> None:
    if isinstance(msg, BatchBackendMsg):
        for msg in msg.data:
            self._process_one_msg(msg)
```

**字段详解**

| 字段 | 类型 | 含义 |
|------|------|------|
| `data` | `List[BaseBackendMsg]` | 打包的消息列表，可以包含 `UserMsg`、`AbortBackendMsg` 或嵌套的 `BatchBackendMsg`（理论上，但实际未出现嵌套）。 |

---

### B.5.3 `ExitMsg`

```python
# python/minisgl/message/backend.py  L27-29
@dataclass
class ExitMsg(BaseBackendMsg):
    pass
```

**功能说明**

控制消息，通知 Scheduler 进程优雅退出。Scheduler 收到该消息后触发 `KeyboardInterrupt`（`scheduler.py` 第 173–174 行），使 `run_forever` 的无限循环退出并执行 `shutdown()` 清理资源。

**设计说明**

`ExitMsg` 没有字段，是一个纯信令消息。在测试和服务停止场景下（如 `llm.py` 的 LLM 类的离线模式），前端或测试框架通过发送 `ExitMsg` 通知 Scheduler 停止，避免了使用操作系统信号（`SIGTERM`）的复杂性，且可以在 ZMQ 消息队列中排队，保证在已有消息处理完毕后才退出。

---

### B.5.4 `UserMsg`

```python
# python/minisgl/message/backend.py  L32-37
@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
```

**功能说明**

从 Tokenizer Worker 发往 Scheduler 的用户请求消息，携带 tokenize 后的 token ID 张量和采样参数。这是 Scheduler 创建 `Req` 对象的原始数据来源。

**字段详解**

| 字段 | 类型 | 含义 |
|------|------|------|
| `uid` | `int` | 请求唯一 ID，与 `TokenizeMsg.uid` 一一对应，由 `FrontendManager` 分配。 |
| `input_ids` | `torch.Tensor`（CPU，1D，int32） | tokenize 后的输入 token ID 序列，shape 为 `(seq_len,)`。类型为 int32 以节省内存（int64 的一半）。注释标注了 tensor 必须在 CPU，这是因为 ZMQ 序列化通过 `numpy().tobytes()` 完成，GPU tensor 无法直接序列化。 |
| `sampling_params` | `SamplingParams` | 从 `TokenizeMsg` 原样透传过来的采样超参数。 |

**Scheduler 中的处理**

`scheduler.py` 第 175–189 行展示了 Scheduler 对 `UserMsg` 的处理：检查序列长度是否超过 `max_seq_len`，必要时截断 `max_tokens`，然后调用 `prefill_manager.add_one_req(msg)` 将其加入 Prefill 等待队列。

---

### B.5.5 `AbortBackendMsg`

```python
# python/minisgl/message/backend.py  L40-41
@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int
```

**功能说明**

请求取消消息，通知 Scheduler 中止指定 `uid` 的请求并释放其占用的资源（KV Cache pages、page table slot 等）。

**字段详解**

| 字段 | 类型 | 含义 |
|------|------|------|
| `uid` | `int` | 要取消的请求 ID。 |

**生产路径**

在 `tokenizer/server.py` 第 102–108 行，当 Tokenizer Worker 收到前端发来的 `AbortMsg`（`BaseTokenizerMsg`）时，将其转换为 `AbortBackendMsg` 发往 Scheduler。此外，前端的 `FrontendManager.abort_user` 方法（`api_server.py` 第 213–220 行）在检测到客户端断开连接时，向 Tokenizer 发送 `AbortMsg`，最终触发这条链路。

---

## B.6 `python/minisgl/message/tokenizer.py` — Tokenizer 消息类型

**文件**：`python/minisgl/message/tokenizer.py`，第 1–43 行

### 文件职责

定义进入 Tokenizer Worker 的所有消息类型，包括文本 tokenize 请求（`TokenizeMsg`）、token detokenize 请求（`DetokenizeMsg`）和取消请求（`AbortMsg`）。这些消息由 API Server 发送，是 Tokenizer Worker 的输入接口。

### 依赖关系

```
tokenizer.py
  ├── 依赖 core.py（SamplingParams）
  ├── 依赖 message/utils.py（serialize_type, deserialize_type）
  ├── 被 server/api_server.py 生产（TokenizeMsg, AbortMsg）
  ├── 被 scheduler/io.py 生产（DetokenizeMsg 经由 BaseTokenizerMsg.encoder）
  └── 被 tokenizer/server.py 消费（全部类型）
```

---

### B.6.1 `BaseTokenizerMsg`

```python
# python/minisgl/message/tokenizer.py  L11-19
@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)
```

**功能说明**

所有 Tokenizer 输入消息的基类，提供与 `BaseFrontendMsg` 相同结构的 `encoder`/`decoder` 静态方法对。`decoder` 使用 `tokenizer.py` 模块的 `globals()`，因此 `DetokenizeMsg`、`TokenizeMsg`、`AbortMsg`、`BatchTokenizerMsg` 均可被正确反序列化。

---

### B.6.2 `BatchTokenizerMsg`

```python
# python/minisgl/message/tokenizer.py  L22-24
@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]
```

**功能说明**

批量 Tokenizer 消息容器。值得注意的是，此类在消息流中扮演双向角色：

1. **入口方向**（API Server → Tokenizer）：多个 `TokenizeMsg` 可打包为 `BatchTokenizerMsg` 一次发送。
2. **来自 Scheduler 方向**（Scheduler → Tokenizer）：多个 `DetokenizeMsg` 同样被包装为 `BatchTokenizerMsg` 发送（见 `scheduler/io.py` 第 130 行）。

`tokenizer/server.py` 的 `_unwrap_msg` 函数（第 24–27 行）负责透明展开批量消息：

```python
# python/minisgl/tokenizer/server.py  L24-27
def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]
```

---

### B.6.3 `DetokenizeMsg`

```python
# python/minisgl/message/tokenizer.py  L27-31
@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool
```

**功能说明**

从 Scheduler 发往 Tokenizer Worker 的 Detokenize 请求，携带一个新生成的 token ID 和完成标志。Tokenizer Worker 收到后，将 `next_token` 追加到该请求的解码历史并调用 `DetokenizeManager.detokenize` 产出文本。

**字段详解**

| 字段 | 类型 | 含义 |
|------|------|------|
| `uid` | `int` | 请求 ID，对应原始 `TokenizeMsg.uid`。 |
| `next_token` | `int` | Scheduler 本步骤采样得到的 token ID（Python int，非 tensor）。注意是标量而非张量，减少序列化开销。 |
| `finished` | `bool` | 是否为最后一个 token。Scheduler 在 `_process_last_data`（`scheduler.py` 第 153–156 行）中根据 `not req.can_decode` 以及是否命中 EOS 计算此标志。 |

**关键设计**：`finished=True` 且 `next_token == eos_token_id` 时，`DetokenizeManager` 不将该 token 追加到解码历史（见 `detokenize.py` 第 83–84 行），避免将 EOS 特殊符号输出给用户。

---

### B.6.4 `TokenizeMsg`

```python
# python/minisgl/message/tokenizer.py  L34-38
@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams
```

**功能说明**

从 API Server 发往 Tokenizer Worker 的 Tokenize 请求，携带原始文本（字符串或消息列表）和采样参数。这是整个请求生命周期的起点消息。

**字段详解**

| 字段 | 类型 | 含义 |
|------|------|------|
| `uid` | `int` | 由 `FrontendManager.new_user()` 分配的全局唯一 ID，在整个消息链路中不变。 |
| `text` | `str \| List[Dict[str, str]]` | 输入文本，有两种形式：（1）纯字符串 `str`，对应 `/generate` 接口直接传入原始 prompt；（2）消息列表 `List[Dict[str, str]]`，每个元素为 `{"role": "user/system/assistant", "content": "..."}` 形式，对应 `/v1/chat/completions` 接口的多轮对话。 |
| `sampling_params` | `SamplingParams` | 由 API Server 从 HTTP 请求参数构造，跟随 `TokenizeMsg` 进入系统。 |

**Tokenizer Worker 中的处理**

`tokenizer/tokenize.py` 第 17–31 行展示了两种 `text` 类型的处理逻辑：

```python
# python/minisgl/tokenizer/tokenize.py  L17-31
for msg in msgs:
    if isinstance(msg.text, list):
        prompt = self.tokenizer.apply_chat_template(
            msg.text,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert isinstance(prompt, str)
    else:
        prompt = msg.text
    input_ids: torch.Tensor = (
        self.tokenizer.encode(prompt, return_tensors="pt")
    )
    results.append(input_ids.view(-1).to(torch.int32))
```

列表形式的 `text` 先通过 `apply_chat_template` 转为字符串，然后统一调用 `tokenizer.encode`，最终输出 1D int32 tensor。

---

### B.6.5 `AbortMsg`

```python
# python/minisgl/message/tokenizer.py  L41-43
@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
```

**功能说明**

从 API Server 发往 Tokenizer Worker 的请求取消通知。Tokenizer Worker 收到后，将其转换为 `AbortBackendMsg` 继续转发给 Scheduler（`tokenizer/server.py` 第 102–108 行）。

**字段详解**

| 字段 | 类型 | 含义 |
|------|------|------|
| `uid` | `int` | 要取消的请求 ID。 |

**取消链路**

```
客户端断开连接
  → FrontendManager.stream_with_cancellation 捕获 CancelledError
  → asyncio.create_task(abort_user(uid))
  → send_one(AbortMsg(uid=uid))    [前端 → Tokenizer]
  → tokenizer_worker 转换为 AbortBackendMsg
  → send_backend.put(AbortBackendMsg)  [Tokenizer → Scheduler]
  → scheduler._process_one_msg(AbortBackendMsg)
  → prefill/decode_manager.abort_req(uid)
  → _free_req_resources(req)
```

---

## B.7 `python/minisgl/message/utils.py` — 消息序列化工具

**文件**：`python/minisgl/message/utils.py`，第 1–69 行

### 文件职责

提供消息的序列化（Python 对象 → `Dict`）和反序列化（`Dict` → Python 对象）功能，支持嵌套数据类、基本类型和 `torch.Tensor`（1D）的透明处理。这是整个消息系统能够跨进程传输的基础设施。

### 依赖关系

```
utils.py
  ├── 依赖 numpy（tensor 的字节序列化）
  ├── 依赖 torch（Tensor 类型处理）
  ├── 被 frontend.py 调用（serialize_type, deserialize_type）
  ├── 被 backend.py 调用
  └── 被 tokenizer.py 调用
```

---

### B.7.1 `_serialize_any`（内部辅助函数）

```python
# python/minisgl/message/utils.py  L9-17
def _serialize_any(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    else:
        return serialize_type(value)
```

**功能说明**

递归处理任意值的序列化分派函数。根据值的类型选择不同策略：
- `dict`：递归序列化所有键值对，保持字典结构。
- `list` / `tuple`：递归序列化每个元素，`type(value)(...)` 保持原始容器类型（`list` → `list`，`tuple` → `tuple`）。
- 基本类型（`int`, `float`, `str`, `None`, `bool`, `bytes`）：直接返回，无需处理。
- 其他对象（如嵌套的 dataclass、`torch.Tensor`、`SamplingParams`）：调用 `serialize_type` 走结构化序列化路径。

---

### B.7.2 `serialize_type`

```python
# python/minisgl/message/utils.py  L20-35
def serialize_type(self) -> Dict:
    serialized = {}

    if isinstance(self, torch.Tensor):
        assert self.dim() == 1, "we can only serialize 1D tensor for now"
        serialized["__type__"] = "Tensor"
        serialized["buffer"] = self.numpy().tobytes()
        serialized["dtype"] = str(self.dtype)
        return serialized

    # normal type
    serialized["__type__"] = self.__class__.__name__
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized
```

**函数签名**

```python
def serialize_type(self: Any) -> Dict
```

**参数**

- `self`：要序列化的对象。尽管参数名为 `self`，但这是普通函数而非方法，命名来源于它既被作为实例方法又被作为工具函数调用的双重角色。

**返回值**

返回 `Dict`，包含 `"__type__"` 键作为类型标识符，其余键为对象字段的序列化值。

**两种序列化路径**

**路径 1：`torch.Tensor`**

```python
# python/minisgl/message/utils.py  L24-29
if isinstance(self, torch.Tensor):
    assert self.dim() == 1, "we can only serialize 1D tensor for now"
    serialized["__type__"] = "Tensor"
    serialized["buffer"] = self.numpy().tobytes()
    serialized["dtype"] = str(self.dtype)
    return serialized
```

- 限制：只支持 1D tensor（当前系统中所有需要序列化的 tensor 均为 1D，如 `UserMsg.input_ids`）。
- `self.numpy()`：CPU tensor 转 numpy array（需要 tensor 在 CPU 且不需要梯度）。
- `.tobytes()`：将 numpy array 序列化为字节串，保留原始二进制表示，零拷贝高效。
- `str(self.dtype)`：记录 dtype 字符串（如 `"torch.int32"`），反序列化时还原。

**路径 2：普通 dataclass 对象**

```python
# python/minisgl/message/utils.py  L32-35
serialized["__type__"] = self.__class__.__name__
for k, v in self.__dict__.items():
    serialized[k] = _serialize_any(v)
return serialized
```

- `self.__class__.__name__`：类名作为类型标记（如 `"UserMsg"`、`"SamplingParams"`）。
- `self.__dict__`：dataclass 的所有字段均存储在 `__dict__` 中，无需手动枚举。
- 每个字段值通过 `_serialize_any` 递归处理，支持嵌套 dataclass（如 `UserMsg` 中的 `SamplingParams`）。

**序列化结果示例**

`UserMsg(uid=42, input_ids=tensor([1, 2, 3], dtype=torch.int32), sampling_params=SamplingParams())` 序列化后：

```json
{
  "__type__": "UserMsg",
  "uid": 42,
  "input_ids": {
    "__type__": "Tensor",
    "buffer": "<bytes>",
    "dtype": "torch.int32"
  },
  "sampling_params": {
    "__type__": "SamplingParams",
    "temperature": 0.0,
    "top_k": -1,
    "top_p": 1.0,
    "ignore_eos": false,
    "max_tokens": 1024
  }
}
```

---

### B.7.3 `_deserialize_any`（内部辅助函数）

```python
# python/minisgl/message/utils.py  L38-49
def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    if isinstance(data, dict):
        if "__type__" in data:
            return deserialize_type(cls_map, data)
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")
```

**功能说明**

`_serialize_any` 的逆操作，递归处理反序列化分派。与序列化不同，反序列化额外需要 `cls_map`（类型名 → 类的映射）来实例化对象。

- `dict` 且含 `"__type__"` 键：调用 `deserialize_type` 还原 Python 对象。
- `dict` 且无 `"__type__"` 键：视为普通字典，递归反序列化值。
- `list` / `tuple`：递归反序列化每个元素，保持容器类型。
- 基本类型：直接返回。
- 其他情况：抛出 `ValueError`（防御性编程，当前实现中不应触及）。

---

### B.7.4 `deserialize_type`

```python
# python/minisgl/message/utils.py  L52-69
def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    type_name = data["__type__"]
    # we can only serialize 1D tensor for now
    if type_name == "Tensor":
        buffer = data["buffer"]
        dtype_str = data["dtype"].replace("torch.", "")
        np_dtype = getattr(np, dtype_str)
        assert isinstance(buffer, bytes)
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)
        return torch.from_numpy(np_tensor.copy())

    cls = cls_map[type_name]
    kwargs = {}
    for k, v in data.items():
        if k == "__type__":
            continue
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)
```

**函数签名**

```python
def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any
```

**参数**

- `cls_map`：类型名到类对象的映射，通常传入调用模块的 `globals()`。
- `data`：包含 `"__type__"` 键的字典，来自 `serialize_type` 的输出。

**返回值**

还原后的 Python 对象，类型由 `data["__type__"]` 决定。

**两种反序列化路径**

**路径 1：`torch.Tensor`**

```python
# python/minisgl/message/utils.py  L55-61
if type_name == "Tensor":
    buffer = data["buffer"]
    dtype_str = data["dtype"].replace("torch.", "")
    np_dtype = getattr(np, dtype_str)
    assert isinstance(buffer, bytes)
    np_tensor = np.frombuffer(buffer, dtype=np_dtype)
    return torch.from_numpy(np_tensor.copy())
```

- `data["dtype"].replace("torch.", "")`：将 `"torch.int32"` 转为 `"int32"`，使 `getattr(np, "int32")` 可找到对应的 numpy dtype。
- `np.frombuffer(buffer, dtype=np_dtype)`：从字节串还原 numpy array，零拷贝（buffer 的内存被 numpy 直接引用）。
- `.copy()`：**关键细节**——`np.frombuffer` 返回的 array 是只读的（指向原始 bytes 缓冲区），`torch.from_numpy` 要求 array 可写，因此必须先 `.copy()` 创建一个可写副本。

**路径 2：普通 dataclass**

```python
# python/minisgl/message/utils.py  L63-69
cls = cls_map[type_name]
kwargs = {}
for k, v in data.items():
    if k == "__type__":
        continue
    kwargs[k] = _deserialize_any(cls_map, v)
return cls(**kwargs)
```

- 从 `cls_map`（即调用方的 `globals()`）查找类名对应的类对象。
- 跳过 `"__type__"` 元字段，将其余字段递归反序列化后作为关键字参数传入类构造函数。
- 依赖 dataclass 的 `__init__` 参数名与字段名一致，保证了反序列化的正确性。

**`cls_map` 的作用域限制**

每个消息模块（`frontend.py`、`backend.py`、`tokenizer.py`）都使用各自的 `globals()` 作为 `cls_map`。这意味着 `backend.py` 的 decoder 只能识别 `backend.py` 中定义的类型（如 `UserMsg`、`BatchBackendMsg`），但 `UserMsg` 中嵌套的 `SamplingParams` 来自 `core.py`。

这是通过"嵌套类型自包含序列化"解决的：`SamplingParams` 在序列化时会携带 `"__type__": "SamplingParams"`，而当 `_deserialize_any` 在 `backend.py` 的 `globals()` 中找不到该类型时，`deserialize_type` 会抛出 `KeyError`。

实际上，`backend.py` 通过 `from minisgl.core import SamplingParams` 将 `SamplingParams` 导入到了自己的模块命名空间，因此 `globals()` 中包含 `"SamplingParams"` 键，反序列化可以正确工作。这是一个隐式但重要的依赖：**凡是需要被 `deserialize_type` 处理的嵌套类型，必须被显式 import 到调用 `decoder` 的模块中**。

---

## B.8 消息序列化机制综合说明

### B.8.1 序列化协议设计原则

mini-sglang 的消息序列化协议遵循以下设计原则：

1. **自描述**：每个序列化对象都携带 `"__type__"` 字段，接收方无需提前知道消息类型即可反序列化。
2. **递归组合**：任意嵌套深度的对象树均可透明序列化，只要叶子节点是支持的基本类型或 tensor。
3. **零依赖**：不使用 `pickle`（不安全、版本敏感）、`protobuf`（需要 IDL）或 `msgpack`（不支持 tensor），而是通过自实现的轻量协议配合 ZMQ 的字节流传输。
4. **类型安全**：`cls_map` 的限定范围确保了只有预期的消息类型可以被实例化，防止反序列化注入攻击。

### B.8.2 Tensor 序列化的约束与取舍

当前实现仅支持 1D CPU tensor，这一约束来源于两个设计决策：

- **进程隔离**：Tokenizer Worker 和 API Server 运行在不同进程，GPU 内存无法跨进程共享（除非使用 CUDA IPC）。所有需要跨进程传输的 tensor 必须在 CPU。
- **简单高效**：1D tensor（token ID 序列）是消息系统唯一需要传输的 tensor 类型。`tobytes` + `frombuffer` 是零拷贝的高效实现，不引入额外的编解码开销。

若未来需要传输 2D 或更高维度 tensor，只需修改 `serialize_type` 中的断言并在序列化时增加 `shape` 字段即可。

### B.8.3 批量消息的优化策略

每条 ZMQ 消息都有固定的发送/接收开销。`Batch*Msg` 系列类（`BatchFrontendMsg`、`BatchBackendMsg`、`BatchTokenizerMsg`）通过批量打包，将多个逻辑消息合并为一次 ZMQ 传输：

```python
# python/minisgl/tokenizer/server.py  L87-101
if len(tokenize_msg) > 0:
    tensors = tokenize_manager.tokenize(tokenize_msg)
    batch_output = BatchBackendMsg(
        data=[
            UserMsg(uid=msg.uid, input_ids=t, sampling_params=msg.sampling_params)
            for msg, t in zip(tokenize_msg, tensors, strict=True)
        ]
    )
    if len(batch_output.data) == 1:
        batch_output = batch_output.data[0]  # 单条时直接发裸消息，避免多余包装
    send_backend.put(batch_output)
```

当批次只有 1 条消息时，直接发送裸消息而不包装为 `BatchBackendMsg`，减少一层序列化开销。

---

## B.9 完整消息流转代码路径索引

以下表格汇总了消息从生产到消费的完整代码路径，方便追踪：

### TokenizeMsg 链路

| 步骤 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 构造 | `server/api_server.py` | 244–252 | HTTP 请求 → `TokenizeMsg(uid, text, sampling_params)` |
| 发送 | `server/api_server.py` | 131–133 | `FrontendManager.send_one` → `ZmqAsyncPushQueue.put` |
| 接收 | `tokenizer/server.py` | 45, 61 | `ZmqPullQueue.get` → `BatchTokenizerMsg.decoder` |
| 分类 | `tokenizer/server.py` | 68 | `isinstance(m, TokenizeMsg)` 分流 |
| 处理 | `tokenizer/tokenize.py` | 14–32 | `TokenizeManager.tokenize` → `List[Tensor]` |
| 产出 | `tokenizer/server.py` | 89–96 | 构造 `UserMsg` 放入 `BatchBackendMsg` |

### UserMsg 链路

| 步骤 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 发送 | `tokenizer/server.py` | 100–101 | `ZmqPushQueue.put(batch_output)` |
| 接收 | `scheduler/io.py` | 36–39 | `ZmqPullQueue` + `BaseBackendMsg.decoder` |
| 分发（多卡） | `scheduler/io.py` | 88–107 | rank0 广播给其他 rank |
| 处理 | `scheduler/scheduler.py` | 169–189 | `_process_one_msg` → `prefill_manager.add_one_req` |
| 创建 Req | `scheduler/prefill.py` | 123–124 | `PendingReq(uid, input_ids, sampling_params)` |

### DetokenizeMsg 链路

| 步骤 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 构造 | `scheduler/scheduler.py` | 156 | `DetokenizeMsg(uid, next_token, finished)` |
| 发送 | `scheduler/io.py` | 124–130 | `_reply_tokenizer_rank0` → `ZmqPushQueue` |
| 接收 | `tokenizer/server.py` | 61, 67 | 按类型分流为 `detokenize_msg` |
| 处理 | `tokenizer/detokenize.py` | 70–111 | `DetokenizeManager.detokenize` → `List[str]` |
| 产出 | `tokenizer/server.py` | 73–85 | 构造 `UserReply` 放入 `BatchFrontendMsg` |

### UserReply 链路

| 步骤 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 发送 | `tokenizer/server.py` | 85 | `ZmqPushQueue.put(batch_output)` → frontend_addr |
| 接收 | `server/api_server.py` | 119 | `FrontendManager.listen` → `ZmqAsyncPullQueue.get` |
| 路由 | `server/api_server.py` | 121–124 | 按 `uid` 追加到 `ack_map`，触发 `event_map` |
| 消费 | `server/api_server.py` | 135–148 | `wait_for_ack` → 逐个 yield `UserReply` |
| 响应 | `server/api_server.py` | 153–159 | `stream_generate` 或 `collect_full_output` |

---

*附录 B 完*
