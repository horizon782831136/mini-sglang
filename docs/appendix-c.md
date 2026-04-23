# 附录 C：调度器模块代码详解

mini-sglang 调度器（scheduler）是整个推理服务的核心控制单元，负责请求的接收、资源分配、批次组装、前向推理调度以及结果回传。本附录对 `python/minisgl/scheduler/` 目录下的 8 个源文件逐一进行详细说明，包括每个文件的职责、全部公开类/函数的接口与实现细节、关键算法分析以及模块间的依赖关系。

---

## 目录

1. [config.py — 调度器配置](#c1-configpy--调度器配置)
2. [scheduler.py — 主调度器](#c2-schedulerpy--主调度器)
3. [prefill.py — 预填充管理](#c3-prefillpy--预填充管理)
4. [decode.py — 解码管理](#c4-decodepy--解码管理)
5. [cache.py — 缓存管理](#c5-cachepy--缓存管理)
6. [table.py — 表管理](#c6-tablepy--表管理)
7. [io.py — I/O 通信](#c7-iopy--io-通信)
8. [utils.py — 工具函数](#c8-utilspy--工具函数)

---

## 模块总体架构

在深入各文件之前，先从宏观视角理解调度器的整体架构：

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            Scheduler 主调度器                              │
│                                                                            │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────────────────┐   │
│  │ SchedulerIO │   │  PrefillManager  │   │     DecodeManager        │   │
│  │   Mixin     │   │  ┌────────────┐  │   │  running_reqs: Set[Req]  │   │
│  │ zmq recv    │   │  │PrefillAdder│  │   │  filter_reqs()           │   │
│  │ zmq send    │   │  │token_budget│  │   │  schedule_next_batch()   │   │
│  │ tp_broadcast│   │  └────────────┘  │   └──────────────────────────┘   │
│  └─────────────┘   └──────────────────┘                                   │
│                              │                                             │
│              ┌───────────────┼───────────────┐                            │
│              ▼               ▼               ▼                            │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│   │ CacheManager │  │ TableManager │  │    Engine    │                    │
│   │ prefix_cache │  │ token_pool   │  │ (KV, attn,   │                   │
│   │ free_slots   │  │ page_table   │  │  model)      │                   │
│   └──────────────┘  └──────────────┘  └──────────────┘                   │
└──────────────────────────────────────────────────────────────────────────┘
```

**请求生命周期**：
1. 网络层（tokenizer 进程）通过 ZMQ 发送 `UserMsg` → `SchedulerIOMixin.receive_msg()`
2. `_process_one_msg()` 将消息转化为 `PendingReq`，加入 `PrefillManager.pending_list`
3. `PrefillManager.schedule_next_batch()` 按 token budget 约束分配资源，生成预填充 `Batch`
4. `_prepare_batch()` 分配页表、构建 GPU 索引张量
5. `Engine.forward_batch()` 执行前向计算
6. `DecodeManager` 将已完成预填充的请求纳入解码队列
7. `_process_last_data()` 处理采样结果，发送 `DetokenizeMsg`，释放已结束请求的资源

---

## C.1 `config.py` — 调度器配置

**文件路径**：`python/minisgl/scheduler/config.py`

**文件职责**：定义调度器进程所需的全部配置参数，包括调度策略参数、网络通信地址以及 IPC 端点，继承自 `EngineConfig`。

### C.1.1 辅助函数

#### `_get_pid_suffix() -> str`

```python
# python/minisgl/scheduler/config.py，第 8-11 行
def _get_pid_suffix() -> str:
    import os
    return f".pid={os.getpid()}"
```

- **参数**：无
- **返回值**：`str`，形如 `".pid=12345"` 的字符串
- **功能**：为 IPC 地址生成基于当前进程 PID 的唯一后缀，确保同一机器上同时运行的多个调度器实例不会共享 ZMQ 端点，避免消息混乱。

### C.1.2 `SchedulerConfig` 数据类

```python
# python/minisgl/scheduler/config.py，第 14-21 行
@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    _unique_suffix: str = field(default_factory=_get_pid_suffix)
```

`SchedulerConfig` 是一个不可变数据类（`frozen=True`），继承自 `minisgl.engine.EngineConfig`，扩展了调度器专属的参数。

#### 继承关系与父类关键字段

`EngineConfig`（`python/minisgl/engine/config.py`）提供以下关键字段：

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `model_path` | `str` | — | 模型权重路径或 HuggingFace 模型名 |
| `tp_info` | `DistributedInfo` | — | Tensor Parallelism 分布式信息 |
| `dtype` | `torch.dtype` | — | 模型推理精度 |
| `max_running_req` | `int` | 256 | 最大并发请求数（决定页表行数） |
| `page_size` | `int` | 1 | KV 缓存页大小（token 数） |
| `memory_ratio` | `float` | 0.9 | GPU 内存用于 KV 缓存的比例 |

#### `SchedulerConfig` 新增字段

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `max_extend_tokens` | `int` | 8192 | 单次调度最大 prefill token 预算（Chunked Prefill 的总上限） |
| `cache_type` | `str` | `"radix"` | 前缀缓存类型，可选 `"radix"` 或 `"naive"` |
| `offline_mode` | `bool` | `False` | 离线模式开关；为 `True` 时跳过 ZMQ 初始化，用于批处理/测试场景 |
| `_unique_suffix` | `str` | PID 后缀 | IPC 地址唯一性后缀，运行时自动生成 |

#### 属性方法详解

**`zmq_backend_addr -> str`**

```python
# python/minisgl/scheduler/config.py，第 23-25 行
@property
def zmq_backend_addr(self) -> str:
    return "ipc:///tmp/minisgl_0" + self._unique_suffix
```

- **返回值**：ZMQ IPC 地址，用于 tokenizer → scheduler 方向的消息通道（PUSH/PULL 模式）。
- **示例**：`"ipc:///tmp/minisgl_0.pid=12345"`

**`zmq_detokenizer_addr -> str`**

```python
# python/minisgl/scheduler/config.py，第 27-29 行
@property
def zmq_detokenizer_addr(self) -> str:
    return "ipc:///tmp/minisgl_1" + self._unique_suffix
```

- **返回值**：ZMQ IPC 地址，用于 scheduler → detokenizer 方向的结果回传通道（PUSH/PULL 模式）。

**`zmq_scheduler_broadcast_addr -> str`**

```python
# python/minisgl/scheduler/config.py，第 31-33 行
@property
def zmq_scheduler_broadcast_addr(self) -> str:
    return "ipc:///tmp/minisgl_2" + self._unique_suffix
```

- **返回值**：ZMQ IPC 地址，用于多 TP rank 场景下 rank 0 向其他 rank 广播请求消息（PUB/SUB 模式）。

**`max_forward_len -> int`**

```python
# python/minisgl/scheduler/config.py，第 35-37 行
@property
def max_forward_len(self) -> int:
    return self.max_extend_tokens
```

- **覆盖**父类同名属性。将单次前向计算的最大 token 数限制为 `max_extend_tokens`（默认 8192），而非模型的 `max_seq_len`。这是 Chunked Prefill 机制的配置入口——决定了每个调度周期最多处理多少个 prefill token。

**`backend_create_detokenizer_link -> bool`**

```python
# python/minisgl/scheduler/config.py，第 39-41 行
@property
def backend_create_detokenizer_link(self) -> bool:
    return True
```

- **返回值**：`True`，指示调度器（backend）侧主动 `bind` detokenizer 地址，而 detokenizer 侧 `connect`。这与标准 ZMQ 的 bind/connect 语义一致：服务端 bind，客户端 connect。

### C.1.3 三条 ZMQ 通道的角色

| 通道编号 | 地址属性 | 方向 | ZMQ 模式 | 用途 |
|---------|---------|------|---------|------|
| 0 | `zmq_backend_addr` | tokenizer → scheduler | PUSH/PULL | 传递用户请求 (`UserMsg`) 和控制消息 (`AbortMsg`, `ExitMsg`) |
| 1 | `zmq_detokenizer_addr` | scheduler → detokenizer | PUSH/PULL | 传递 token 采样结果 (`DetokenizeMsg`) |
| 2 | `zmq_scheduler_broadcast_addr` | rank 0 → rank 1..N | PUB/SUB | 多卡 TP 下广播请求给非主 rank |

### C.1.4 依赖关系

```
SchedulerConfig
    └── 继承 EngineConfig (minisgl.engine.config)
            ├── DistributedInfo (minisgl.distributed.info)
            └── ModelConfig (minisgl.models.config)  [cached_property]
```

`SchedulerConfig` 实例在 `Scheduler.__init__()` 中创建，并传递给所有子管理器（`CacheManager`、`TableManager`、`PrefillManager`、`DecodeManager`、`SchedulerIOMixin`）。

---

## C.2 `scheduler.py` — 主调度器

**文件路径**：`python/minisgl/scheduler/scheduler.py`

**文件职责**：实现调度器主体逻辑，包括 Overlap Scheduling 主循环、批次准备、前向推理触发、结果处理以及请求生命周期管理。

### C.2.1 类型定义

```python
# python/minisgl/scheduler/scheduler.py，第 31-42 行
Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]

class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or 0)

ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"
```

**`Indice2D`**：表示一对 GPU 索引张量的类型别名，用于在 token_pool 和 page_table 上进行二维索引操作。

**`ForwardInput`**（`NamedTuple`）：将一次前向计算所需的所有数据打包在一起。设计为 `NamedTuple` 而非普通 `dataclass` 的原因：Overlap Scheduling 场景下需要将前一轮的计算数据缓存至下一个调度周期处理，使用不可变命名元组可以避免意外修改（IMA，In-place Modification Attack）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `batch` | `Batch` | 本次参与计算的请求批次 |
| `sample_args` | `BatchSamplingArgs` | 采样超参（temperature、top_k、top_p 等） |
| `input_tuple` | `Indice2D` | `(token_mapping, positions)`：用于从 token_pool 中读取输入 token id |
| `write_tuple` | `Indice2D` | `(req_mapping, seq_lens)`：用于将采样到的新 token 写回 token_pool |

**`ForwardData`**：将 `ForwardInput` 与 `ForwardOutput` 配对的类型别名，是 Overlap Scheduling 中在调度周期之间传递"上一轮结果"的载体。

---

### C.2.2 `Scheduler` 类

```python
# python/minisgl/scheduler/scheduler.py，第 45 行
class Scheduler(SchedulerIOMixin):
```

`Scheduler` 继承 `SchedulerIOMixin`，后者提供所有 ZMQ I/O 方法。`Scheduler` 本体负责调度逻辑，混入的 I/O 代码保持分离。

#### `__init__(self, config: SchedulerConfig)`

```python
# python/minisgl/scheduler/scheduler.py，第 46-76 行
def __init__(self, config: SchedulerConfig):
    from minisgl.engine import Engine

    self.engine = Engine(config)

    # use another stream to overlap metadata processing with computation
    self.device = self.engine.device
    self.stream = torch.cuda.Stream(device=self.device)
    self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
    torch.cuda.set_stream(self.stream)

    # initialize other managers
    self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
    self.cache_manager = CacheManager(
        self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
    )
    self.decode_manager = DecodeManager(config.page_size)
    self.prefill_manager = PrefillManager(
        self.cache_manager, self.table_manager, self.decode_manager
    )
    ...
    super().__init__(config, self.engine.tp_cpu_group)
```

**初始化顺序及关键细节**：

1. **创建 `Engine`**：初始化 GPU、分布式通信、加载模型权重、分配 KV 缓存池、构建 CUDA Graph。此时 CUDA 当前流为 `engine.stream`。

2. **双流设计（Overlap Scheduling 核心）**：
   - `self.engine.stream`：引擎的主计算流（前向传播、矩阵乘法等在此流上执行）
   - `self.stream`：调度器的元数据处理流（CPU-GPU 内存拷贝、索引构建在此流上执行）
   - `torch.cuda.set_stream(self.stream)` 切换当前流至调度器流，后续默认操作在此流上进行
   - `self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)` 是一个上下文管理器，在 `_forward()` 中使用，临时切换到引擎流执行前向计算

3. **管理器初始化**：
   - `TableManager`：持有 `page_table` 的引用，并创建同形状的 `token_pool`
   - `CacheManager`：管理 KV 缓存页的分配、驱逐和前缀缓存
   - `DecodeManager`：维护正在解码的请求集合
   - `PrefillManager`：维护待预填充的请求队列，依赖前三个管理器

4. **便利别名**：`self.token_pool = self.table_manager.token_pool`，避免深层访问路径。

5. **prefill_budget**：从 `config.max_extend_tokens` 读取，控制每次调度最多处理的 prefill token 数量，是 Chunked Prefill 的硬上限。

---

#### `run_forever(self) -> NoReturn`

```python
# python/minisgl/scheduler/scheduler.py，第 120-131 行
@torch.inference_mode()
def run_forever(self) -> NoReturn:
    if ENV.DISABLE_OVERLAP_SCHEDULING:
        with self.engine_stream_ctx:
            self.engine.stream.wait_stream(self.stream)
            while True:
                self.normal_loop()
    else:
        assert torch.cuda.current_stream() == self.stream
        data = None
        while True:
            data = self.overlap_loop(data)
```

调度器的入口函数，整个 Python 进程在此无限循环中运行。

- **`@torch.inference_mode()`**：禁用梯度计算，减少内存占用，提升推理速度。
- **`ENV.DISABLE_OVERLAP_SCHEDULING`**：通过环境变量 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` 可关闭 Overlap Scheduling，退化为同步模式（用于调试）。
- **非重叠模式**：进入引擎流上下文（`with self.engine_stream_ctx`），并等待调度器流完成（`wait_stream`），之后在同一流中执行 `normal_loop()`。
- **重叠模式**：断言当前流为 `self.stream`（调度器流），然后循环调用 `overlap_loop()`，每轮将上一轮的 `ForwardData` 传入处理。

---

#### `overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None`

```python
# python/minisgl/scheduler/scheduler.py，第 83-106 行
def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
    blocking = not (
        last_data is not None
        or self.prefill_manager.runnable
        or self.decode_manager.runnable
    )
    for msg in self.receive_msg(blocking=blocking):
        self._process_one_msg(msg)

    forward_input = self._schedule_next_batch()
    ongoing_data = None
    if forward_input is not None:
        with self.engine_stream_ctx:  # run the batch in the engine's stream
            self.engine.stream.wait_stream(self.stream)
            ongoing_data = (forward_input, self._forward(forward_input))

    self._process_last_data(last_data)
    return ongoing_data
```

**这是 mini-sglang 调度器最核心的方法**，实现了 CPU 元数据处理与 GPU 计算的流水线重叠。

**重叠调度原理**：

```
调度周期 N:
  CPU 侧 (self.stream):    [接收消息] → [构建批次 N] → [处理批次 N-1 的结果]
  GPU 侧 (engine.stream):              → [计算批次 N]

调度周期 N+1:
  CPU 侧 (self.stream):    [接收消息] → [构建批次 N+1] → [处理批次 N 的结果]
  GPU 侧 (engine.stream):              → [计算批次 N+1]
```

关键点：
- **`self.engine.stream.wait_stream(self.stream)`**：确保在启动新批次前，调度器流（CPU→GPU 拷贝）已完成，防止数据竞争。
- **`with self.engine_stream_ctx`**：`_forward()` 在 `engine.stream` 上执行 GPU 计算，因此不阻塞 `self.stream` 上的 CPU 工作。
- **`self._process_last_data(last_data)`**：处理上一轮的 GPU 计算结果（等待 CPU 数据拷贝完成），与当前轮的 `_forward()` 并行进行。

**blocking 判断逻辑**：
- 当且仅当：上一轮无结果待处理（`last_data is None`）**且** 预填充和解码队列均为空时，才阻塞等待新消息到来
- 否则非阻塞轮询，以保证调度延迟最小化

---

#### `normal_loop(self) -> None`

```python
# python/minisgl/scheduler/scheduler.py，第 108-118 行
def normal_loop(self) -> None:
    blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
    for msg in self.receive_msg(blocking=blocking):
        self._process_one_msg(msg)

    forward_input = self._schedule_next_batch()
    ongoing_data = None
    if forward_input is not None:
        ongoing_data = (forward_input, self._forward(forward_input))

    self._process_last_data(ongoing_data)
```

`overlap_loop` 的同步版本，所有操作串行执行：接收消息 → 调度批次 → 前向计算 → 处理结果。用于调试或不支持双流重叠的场景。

---

#### `_schedule_next_batch(self) -> ForwardInput | None`

```python
# python/minisgl/scheduler/scheduler.py，第 219-225 行
def _schedule_next_batch(self) -> ForwardInput | None:
    # TODO: support other policies: e.g. DECODE first
    batch = (
        self.prefill_manager.schedule_next_batch(self.prefill_budget)
        or self.decode_manager.schedule_next_batch()
    )
    return self._prepare_batch(batch) if batch else None
```

**调度策略**：当前实现为 **Prefill First**（预填充优先），即优先处理待预填充的请求，只有在没有待预填充请求时才处理解码请求。

注释提示未来可支持 `DECODE first` 等其他策略。

---

#### `_prepare_batch(self, batch: Batch) -> ForwardInput`

```python
# python/minisgl/scheduler/scheduler.py，第 204-217 行
def _prepare_batch(self, batch: Batch) -> ForwardInput:
    self.engine.graph_runner.pad_batch(batch)
    self.cache_manager.allocate_paged(batch.reqs)
    batch.positions = _make_positions(batch, self.device)
    input_mapping = _make_input_tuple(batch, self.device)
    write_mapping = _make_write_tuple(batch, self.device)
    batch.out_loc = self.engine.page_table[input_mapping]
    self.engine.attn_backend.prepare_metadata(batch)
    return ForwardInput(
        batch=batch,
        sample_args=self.engine.sampler.prepare(batch),
        input_tuple=input_mapping,
        write_tuple=write_mapping,
    )
```

将逻辑批次转化为 GPU 可直接消费的数据结构。执行步骤：

1. **`pad_batch(batch)`**：为 CUDA Graph 固定 batch size 填充 dummy 请求
2. **`allocate_paged(batch.reqs)`**：为本轮 extend 的 token 分配 KV 缓存页
3. **`_make_positions()`**：构建每个 token 的序列位置张量（用于 RoPE 位置编码）
4. **`_make_input_tuple()`**：构建 `(table_idx, position)` 二维索引，用于从 `token_pool` 读取输入 token id
5. **`_make_write_tuple()`**：构建 `(table_idx, device_len)` 二维索引，用于将新生成的 token 写回 `token_pool`
6. **`batch.out_loc`**：通过页表映射将逻辑位置转换为 KV 缓存的物理存储地址
7. **`prepare_metadata(batch)`**：让 attention backend 构建注意力计算所需的元数据（如 flash-attention 的 `cu_seqlens`）
8. **`sampler.prepare(batch)`**：从请求的 `sampling_params` 中提取 temperature、top_k 等参数，打包为 `BatchSamplingArgs`

---

#### `_forward(self, forward_input: ForwardInput) -> ForwardOutput`

```python
# python/minisgl/scheduler/scheduler.py，第 227-235 行
def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
    batch, sample_args, input_mapping, output_mapping = forward_input
    batch.input_ids = self.token_pool[input_mapping]
    if ENV.OVERLAP_EXTRA_SYNC:
        self.stream.synchronize()
    forward_output = self.engine.forward_batch(batch, sample_args)
    self.token_pool[output_mapping] = forward_output.next_tokens_gpu
    self.decode_manager.filter_reqs(forward_input.batch.reqs)
    return forward_output
```

执行一次实际的前向推理：

1. **`batch.input_ids = self.token_pool[input_mapping]`**：从全局 `token_pool` 中读取本批次的输入 token（在 GPU 上通过二维索引完成）
2. **`ENV.OVERLAP_EXTRA_SYNC`**：Issue #58 的临时修复，在极少数情况下需要额外的流同步以避免数据竞争
3. **`engine.forward_batch()`**：触发模型前向计算，返回 `ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)`
4. **`token_pool[output_mapping] = ...`**：将 GPU 上新采样的 token 写回 `token_pool` 对应位置（通过 `write_tuple` 定位）
5. **`decode_manager.filter_reqs()`**：将本批次中 `can_decode=True` 的请求更新到解码集合（见 C.4 节）

---

#### `_process_last_data(self, last_data: ForwardData | None) -> None`

```python
# python/minisgl/scheduler/scheduler.py，第 138-167 行
def _process_last_data(self, last_data: ForwardData | None) -> None:
    if last_data is None:
        return

    batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
    copy_done.synchronize()
    reply: List[DetokenizeMsg] = []
    new_finished_reqs: Set[Req] = set()
    with self.cache_manager.lazy_free_region():
        for i, req in enumerate(batch.reqs):
            if isinstance(req, ChunkedReq):
                continue
            next_token = next_tokens_cpu[i]
            req.append_host(next_token.unsqueeze(0))
            next_token = int(next_token.item())
            finished = not req.can_decode
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id
            reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            if finished and req not in self.finished_reqs:
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)
                new_finished_reqs.add(req)
            elif batch.is_prefill:
                self.cache_manager.cache_req(req, finished=False)

    self.finished_reqs = new_finished_reqs
    self.send_result(reply)
```

处理上一轮前向计算的结果，是调度器中最复杂的方法之一。

**关键步骤**：

1. **`copy_done.synchronize()`**：等待 GPU→CPU 的 token 拷贝完成（该拷贝在 `engine.forward_batch()` 中以非阻塞方式启动）

2. **`lazy_free_region()`**：使用延迟释放上下文（见 C.5 节），将本循环内的所有页释放操作收集起来，在循环结束后批量执行 `torch.cat`，避免逐次拼接带来的内存碎片

3. **跳过 `ChunkedReq`**：分块预填充请求在本轮不产生采样 token，跳过采样结果处理

4. **EOS 检测**：如果 `ignore_eos=False` 且采样到 EOS token，则标记请求结束

5. **`req.append_host(next_token)`**：将新生成的 token 追加到请求的 `input_ids`（CPU 侧），保持 CPU 侧的 token 序列完整（用于后续缓存写入时的 token 匹配）

6. **已结束请求**：调用 `_free_req_resources()` 释放页表槽位和前缀缓存

7. **预填充结束的请求**：调用 `cache_req(finished=False)` 将本轮新扩展的前缀插入前缀缓存，为后续匹配提供加速

8. **`self.finished_reqs`**：防止重复释放的集合——Overlap Scheduling 下，相同请求在相邻两个周期内可能各处理一次，`finished_reqs` 追踪上一周期已释放的请求

---

#### `_process_one_msg(self, msg: BaseBackendMsg) -> None`

```python
# python/minisgl/scheduler/scheduler.py，第 169-198 行
def _process_one_msg(self, msg: BaseBackendMsg) -> None:
    if isinstance(msg, BatchBackendMsg):
        for msg in msg.data:
            self._process_one_msg(msg)
    elif isinstance(msg, ExitMsg):
        raise KeyboardInterrupt
    elif isinstance(msg, UserMsg):
        ...
        self.prefill_manager.add_one_req(msg)
    elif isinstance(msg, AbortBackendMsg):
        req_to_free = self.prefill_manager.abort_req(msg.uid)
        req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
        if req_to_free is not None:
            self._free_req_resources(req_to_free)
    else:
        raise NotImplementedError
```

消息分发器，处理四类消息：

| 消息类型 | 动作 |
|---------|------|
| `BatchBackendMsg` | 批量消息解包，递归处理每条子消息 |
| `ExitMsg` | 抛出 `KeyboardInterrupt`，触发优雅退出 |
| `UserMsg` | 校验序列长度，裁剪 `max_tokens`，加入预填充队列 |
| `AbortBackendMsg` | 从预填充或解码队列中移除指定请求，释放其资源 |

**UserMsg 处理中的长度校验**：

```python
# python/minisgl/scheduler/scheduler.py，第 177-188 行
input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
max_output_len = max_seq_len - input_len
if max_output_len <= 0:
    return logger.warning_rank0(...)  # 输入过长，静默丢弃
if msg.sampling_params.max_tokens > max_output_len:
    msg.sampling_params.max_tokens = max_output_len  # 裁剪输出长度
```

---

#### `_free_req_resources(self, req: Req) -> None`

```python
# python/minisgl/scheduler/scheduler.py，第 200-202 行
def _free_req_resources(self, req: Req) -> None:
    self.table_manager.free(req.table_idx)
    self.cache_manager.cache_req(req, finished=True)
```

- 将 `table_idx` 还回 `TableManager` 的空闲池
- 调用 `cache_req(finished=True)`：向前缀缓存插入已完成序列的缓存片段，并释放无法插入的尾部页

---

#### `shutdown(self) -> None`

```python
# python/minisgl/scheduler/scheduler.py，第 133-136 行
def shutdown(self) -> None:
    torch.cuda.synchronize(self.device)
    self.sync_all_ranks()
    self.engine.shutdown()
```

优雅关闭：同步 GPU 操作 → 所有 TP rank 同步 → 销毁 CUDA Graph 和分布式进程组。

---

#### `run_when_idle(self) -> None`

```python
# python/minisgl/scheduler/scheduler.py，第 78-81 行
def run_when_idle(self) -> None:
    logger.info_rank0("Scheduler is idle, waiting for new reqs...")
    self.cache_manager.check_integrity()
```

调度器空闲时（blocking=True 进入阻塞等待前）调用，执行 `CacheManager` 的一致性校验，属于后台健康检查。

---

### C.2.3 模块级辅助函数

#### `_make_positions(batch: Batch, device: torch.device) -> torch.Tensor`

```python
# python/minisgl/scheduler/scheduler.py，第 238-251 行
def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)
```

构建 positions 张量，形状 `[total_extend_tokens]`，每个元素是对应 token 在其所属序列中的位置索引（从 `cached_len` 到 `device_len-1`）。

- 使用 `pin_memory=True` 的 CPU 张量作为中间缓冲区，再通过 `non_blocking=True` 异步拷贝到 GPU，最小化 CPU-GPU 传输延迟。
- 遍历 `batch.padded_reqs`（包含 dummy 请求），确保 CUDA Graph batch size 对齐。

#### `_make_input_tuple(batch: Batch, device: torch.device) -> Indice2D`

```python
# python/minisgl/scheduler/scheduler.py，第 254-261 行
def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)
```

返回 `(table_idx_tensor, position_tensor)` 二元组，用于 `token_pool[table_idx, position]` 的二维索引，从而读取每个 extend token 的 token id。

- `table_idx_tensor`：每个 token 的所属请求行索引（`int64`，GPU 索引要求）
- `position_tensor`：复用已构建的 `batch.positions`，转为 `int64`

#### `_make_write_tuple(batch: Batch, device: torch.device) -> Indice2D`

```python
# python/minisgl/scheduler/scheduler.py，第 264-269 行
def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
```

返回 `(table_idx_tensor, write_pos_tensor)` 二元组，用于 `token_pool[table_idx, write_pos] = new_token` 将新生成的 token 写入 token_pool。

- `write_pos`：等于 `req.device_len`（采样前的序列末尾位置，即新 token 的写入位置）
- 对 `can_decode=False`（即 `ChunkedReq`）的请求，write_pos 为 `-1`，表示不写入（Python 风格的"无效索引"占位）
- 注意：此函数遍历 `batch.reqs`（实际请求），而不是 `batch.padded_reqs`（含 dummy 的填充请求）

---

### C.2.4 依赖关系图

```
Scheduler
    ├── 继承 SchedulerIOMixin (scheduler.io)
    ├── Engine (minisgl.engine)
    │       ├── page_table: torch.Tensor
    │       ├── graph_runner: GraphRunner
    │       ├── attn_backend: BaseAttnBackend
    │       └── sampler: Sampler
    ├── TableManager (scheduler.table)
    ├── CacheManager (scheduler.cache)
    │       └── BasePrefixCache (minisgl.kvcache)
    ├── DecodeManager (scheduler.decode)
    └── PrefillManager (scheduler.prefill)
            ├── PrefillAdder
            └── ChunkedReq
```

---

## C.3 `prefill.py` — 预填充管理

**文件路径**：`python/minisgl/scheduler/prefill.py`

**文件职责**：管理待预填充请求队列，实现带 token budget 约束的贪心调度算法（Chunked Prefill），并通过 `PrefillAdder` 控制每轮预填充的资源分配。

### C.3.1 `ChunkedReq` 类

```python
# python/minisgl/scheduler/prefill.py，第 23-29 行
class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager
```

`ChunkedReq` 继承自 `minisgl.core.Req`，表示一个**分块预填充中的请求**——即由于 token budget 不足而无法在本轮完成全部预填充的请求。

**核心设计**：

- **`can_decode` 返回 `False`**：这是 `ChunkedReq` 的核心约束。`DecodeManager.filter_reqs()` 依赖此属性判断请求是否可以进入解码阶段。分块请求尚未完成预填充，自然不能解码。
- **`append_host` 抛出异常**：分块请求不产生采样 token，若错误地尝试为其追加 token 则应报错。
- **与 `Req` 的区别**：`Req.can_decode = remain_len > 0`（取决于剩余长度），而 `ChunkedReq.can_decode = False`（强制禁止）。

`ChunkedReq` 与普通 `Req` 在 `_process_last_data()` 中的处理差异：

```python
# scheduler.py，第 148-149 行
if isinstance(req, ChunkedReq):
    continue  # 跳过 ChunkedReq，不发送采样结果
```

---

### C.3.2 `PrefillAdder` 数据类

```python
# python/minisgl/scheduler/prefill.py，第 32-37 行
@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager
```

`PrefillAdder` 是一次调度周期的**资源分配器**，持有当次批次组装过程中的可变状态：`token_budget`（剩余 token 预算）和 `reserved_size`（已为在途请求预留的 KV 缓存量）。

`PrefillAdder` 由 `PrefillManager.schedule_next_batch()` 创建，生命周期仅限于单次批次组装。

#### `_try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None`

```python
# python/minisgl/scheduler/prefill.py，第 39-63 行
def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
    if self.table_manager.available_size == 0:
        return None

    handle = self.cache_manager.match_req(req).cuda_handle
    cached_len = handle.cached_len
    extend_len = req.input_len - cached_len
    estimated_len = extend_len + req.output_len

    if estimated_len + self.reserved_size > self.cache_manager.available_size:
        return None
    self.cache_manager.lock(handle)
    if estimated_len + self.reserved_size > self.cache_manager.available_size:
        return self.cache_manager.unlock(handle)

    table_idx = self.table_manager.allocate()
    if cached_len > 0:
        device_ids = self.table_manager.token_pool[table_idx][:cached_len]
        page_entry = self.table_manager.page_table[table_idx][:cached_len]
        device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
        page_entry.copy_(handle.get_matched_indices())

    return handle, table_idx
```

**功能**：为全新请求（非分块续传）尝试分配 `table_idx` 和前缀缓存句柄。

**KV 缓存内存估算逻辑**：

```
estimated_len = (input_len - cached_len) + output_len
             = extend_len + output_len
```

即：本次需要计算的 token 数（extend_len）加上未来解码需要的 token 数（output_len）。这是一个**保守估计**，确保为整个请求的完整生命周期保留足够的 KV 缓存空间。

**双重检测（TOCTOU 防护）**：

```python
# 检测 1：快速路径（lock 之前）
if estimated_len + self.reserved_size > self.cache_manager.available_size:
    return None
self.cache_manager.lock(handle)
# 检测 2：慢速路径（lock 之后）
if estimated_len + self.reserved_size > self.cache_manager.available_size:
    return self.cache_manager.unlock(handle)
```

`lock()` 操作将前缀缓存中对应条目标记为"受保护"，从 `evictable_size` 转移到 `protected_size`，导致 `available_size` 减少。因此在 `lock` 前后都需检测，避免 TOCTOU（Time-of-Check-Time-of-Use）竞态。

**前缀命中时的初始化**：若 `cached_len > 0`，说明前缀缓存命中，需要将缓存的 KV 物理地址（`handle.get_matched_indices()`）和对应 token id 预写入 `token_pool` 和 `page_table`，以便 attention kernel 可以直接读取历史 KV。

---

#### `_add_one_req(self, pending_req, cache_handle, table_idx, cached_len) -> Req`

```python
# python/minisgl/scheduler/prefill.py，第 65-90 行
def _add_one_req(
    self,
    pending_req: PendingReq,
    cache_handle: BaseCacheHandle,
    table_idx: int,
    cached_len: int,
) -> Req:
    remain_len = pending_req.input_len - cached_len
    chunk_size = min(self.token_budget, remain_len)
    is_chunked = chunk_size < remain_len
    CLS = ChunkedReq if is_chunked else Req
    self.token_budget -= chunk_size
    self.reserved_size += remain_len + pending_req.output_len
    _slice = slice(cached_len, cached_len + chunk_size)
    device_ids = self.table_manager.token_pool[table_idx, _slice]
    device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
    return CLS(
        input_ids=pending_req.input_ids[: cached_len + chunk_size],
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=pending_req.output_len,
        uid=pending_req.uid,
        cache_handle=cache_handle,
        sampling_params=pending_req.sampling_params,
    )
```

**Chunked Prefill 的核心逻辑**：

1. `chunk_size = min(token_budget, remain_len)`：本轮实际处理的 token 数，受预算上限约束
2. `is_chunked = chunk_size < remain_len`：若预算不足以处理全部剩余 token，则为分块模式
3. **消耗预算**：`token_budget -= chunk_size`
4. **预留空间**：`reserved_size += remain_len + output_len`（注意：这里是 `remain_len`，非 `chunk_size`，是为整个请求的剩余生命周期预留）
5. **写入 token id**：将 `input_ids[cached_len : cached_len + chunk_size]` 通过 pin_memory + 非阻塞拷贝写入 `token_pool`
6. **构造请求对象**：`input_ids` 截取到 `cached_len + chunk_size`，这决定了 `device_len = cached_len + chunk_size`，也决定了 `extend_len = chunk_size`

---

#### `try_add_one(self, pending_req: PendingReq) -> Req | None`

```python
# python/minisgl/scheduler/prefill.py，第 92-113 行
def try_add_one(self, pending_req: PendingReq) -> Req | None:
    if self.token_budget <= 0:
        return None

    if chunked_req := pending_req.chunked_req:
        return self._add_one_req(
            pending_req=pending_req,
            cache_handle=chunked_req.cache_handle,
            table_idx=chunked_req.table_idx,
            cached_len=chunked_req.cached_len,
        )

    if resource := self._try_allocate_one(pending_req):
        cache_handle, table_idx = resource
        return self._add_one_req(
            pending_req=pending_req,
            cache_handle=cache_handle,
            table_idx=table_idx,
            cached_len=cache_handle.cached_len,
        )

    return None
```

**`try_add_one` 的两条路径**：

**路径 1 — 分块续传**：`pending_req.chunked_req` 非空，说明此请求在上一轮已开始预填充但未完成。此时复用上一轮的 `cache_handle` 和 `table_idx`，无需重新分配，并以 `chunked_req.cached_len` 为本轮的起始位置，实现无缝续传。

**路径 2 — 全新请求**：`pending_req.chunked_req` 为空，调用 `_try_allocate_one()` 申请新资源，若资源充足则创建新的 `Req` 或 `ChunkedReq`。

**预算检查**：在进入任何路径前，首先检查 `token_budget <= 0`，若预算耗尽立即返回 `None`。

---

### C.3.3 `PrefillManager` 数据类

```python
# python/minisgl/scheduler/prefill.py，第 116-122 行
@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)
```

`PrefillManager` 维护全局的待预填充请求队列 `pending_list`，并在每个调度周期内通过 `PrefillAdder` 完成贪心资源分配。

#### `add_one_req(self, req: UserMsg) -> None`

```python
# python/minisgl/scheduler/prefill.py，第 123-124 行
def add_one_req(self, req: UserMsg) -> None:
    self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))
```

将 `UserMsg`（网络消息）转化为 `PendingReq`（内部队列元素）并追加到队列末尾。转化过程保留 `uid`、`input_ids` 和 `sampling_params`，丢弃网络层信息。

---

#### `schedule_next_batch(self, prefill_budget: int) -> Batch | None`

```python
# python/minisgl/scheduler/prefill.py，第 126-151 行
def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
    if len(self.pending_list) == 0:
        return None

    adder = PrefillAdder(
        token_budget=prefill_budget,
        reserved_size=self.decode_manager.inflight_tokens,
        cache_manager=self.cache_manager,
        table_manager=self.table_manager,
    )
    reqs: List[Req] = []
    chunked_list: List[PendingReq] = []
    for pending_req in self.pending_list:
        if req := adder.try_add_one(pending_req):
            pending_req.chunked_req = None
            if isinstance(req, ChunkedReq):
                pending_req.chunked_req = req
                chunked_list.append(pending_req)
            reqs.append(req)
        else:
            break
    if len(reqs) == 0:
        return None
    self.pending_list = chunked_list + self.pending_list[len(reqs):]
    return Batch(reqs=reqs, phase="prefill")
```

**调度算法详解**：

1. **初始化 `PrefillAdder`**：
   - `token_budget = prefill_budget`（来自 `SchedulerConfig.max_extend_tokens`）
   - `reserved_size = decode_manager.inflight_tokens`：已在飞的解码请求所占用的未来 KV 缓存估算量（不能被 prefill 抢占）

2. **贪心遍历**：按 FIFO 顺序遍历 `pending_list`，对每个请求调用 `adder.try_add_one()`。一旦某个请求失败（资源或预算不足），立即 `break`（不继续尝试后续请求，保持队列公平性）。

3. **分块请求的队列重排**：完成的请求从 `pending_list` 中移除；分块请求保留并放在队首（`chunked_list + self.pending_list[len(reqs):]`），确保下一轮优先续传。

4. **`pending_req.chunked_req` 的双向引用**：
   - `PendingReq.chunked_req` 指向当前的 `ChunkedReq` 对象
   - 在 `try_add_one` 的续传路径中，通过此引用复用上一轮的 `table_idx` 和 `cache_handle`

---

#### `abort_req(self, uid: int) -> Req | None`

```python
# python/minisgl/scheduler/prefill.py，第 153-158 行
def abort_req(self, uid: int) -> Req | None:
    for i, req in enumerate(self.pending_list):
        if req.uid == uid:
            self.pending_list.pop(i)
            return req.chunked_req
    return None
```

从 `pending_list` 中移除指定 uid 的请求。若该请求已有 `chunked_req`（即已分配了 `table_idx` 和 KV 页），则返回该 `ChunkedReq` 供调用方释放其资源；否则返回 `None`（纯 pending 请求无需释放资源）。

---

#### `runnable` 属性

```python
# python/minisgl/scheduler/prefill.py，第 160-162 行
@property
def runnable(self) -> bool:
    return len(self.pending_list) > 0
```

调度器主循环通过此属性判断是否有待处理的预填充请求，决定是否阻塞等待新消息。

---

### C.3.4 依赖关系

```
PrefillManager
    ├── CacheManager (scheduler.cache)
    │       ├── match_req()        — 前缀缓存匹配
    │       ├── lock() / unlock()  — 句柄锁定保护
    │       └── allocate_paged()   — 页分配（由 Scheduler 调用）
    ├── TableManager (scheduler.table)
    │       ├── available_size     — 空闲槽位数
    │       ├── allocate()         — 分配 table_idx
    │       ├── token_pool         — 写入 extend token ids
    │       └── page_table         — 写入缓存页索引
    ├── DecodeManager (scheduler.decode)
    │       └── inflight_tokens    — 在途 decode 请求的预留空间
    └── PendingReq (scheduler.utils)
            └── ChunkedReq (scheduler.prefill)  [双向引用]
```

---

## C.4 `decode.py` — 解码管理

**文件路径**：`python/minisgl/scheduler/decode.py`

**文件职责**：维护正在解码阶段的请求集合，负责 decode batch 的组装、请求过滤（filter_reqs）、在途 token 数量的估算以及请求的添加/移除。

### C.4.1 `DecodeManager` 数据类

```python
# python/minisgl/scheduler/decode.py，第 9-12 行
@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)
```

`DecodeManager` 使用 `Set[Req]` 存储当前正在解码的请求集合（无序）。使用集合而非列表的原因：解码阶段的请求无序执行，集合提供 O(1) 的插入、删除和成员检测。

`Req` 通过 `@dataclass(eq=False)` 定义，使用默认的对象身份（`id()`）作为相等判断，因此可以安全放入集合。

---

#### `filter_reqs(self, reqs: Iterable[Req]) -> None`

```python
# python/minisgl/scheduler/decode.py，第 14-15 行
def filter_reqs(self, reqs: Iterable[Req]) -> None:
    self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}
```

**这是 DecodeManager 的核心方法**，实现了两项关键操作：

1. **新增请求**：将 `reqs`（本轮前向计算中参与的请求）合并入 `running_reqs`。对于刚完成 prefill 的普通 `Req`，`can_decode=True`（`remain_len > 0`），将进入解码集合。
2. **过滤请求**：筛选条件为 `req.can_decode`，即：
   - `ChunkedReq`：`can_decode = False`，不加入（预填充未完成）
   - 已用尽输出配额的请求：`remain_len = 0`，`can_decode = False`，被移除

**调用时机**：`_forward()` 执行完前向计算后立即调用，此时 `req.complete_one()` 已被 `engine.forward_batch()` 调用，`device_len` 已自增，`remain_len` 已减少。

**设计精妙之处**：`filter_reqs` 同时完成"添加新请求"和"移除已完成请求"两件事，在一次集合操作中完成批次状态同步，代码极为简洁。

---

#### `remove_req(self, req: Req) -> None`

```python
# python/minisgl/scheduler/decode.py，第 17-18 行
def remove_req(self, req: Req) -> None:
    self.running_reqs.discard(req)
```

显式从集合中移除指定请求。使用 `discard`（而非 `remove`）是因为请求可能已不在集合中（例如 Overlap Scheduling 下的重复处理），`discard` 在元素不存在时静默返回，避免 `KeyError`。

**调用场景**：在 `_process_last_data()` 中，检测到请求已完成（EOS 或 max_tokens 耗尽）时调用。

---

#### `abort_req(self, uid: int) -> Req | None`

```python
# python/minisgl/scheduler/decode.py，第 20-25 行
def abort_req(self, uid: int) -> Req | None:
    for req in self.running_reqs:
        if req.uid == uid:
            self.running_reqs.remove(req)
            return req
    return None
```

按 uid 中止指定请求，从集合中移除并返回该请求（供调用方释放资源）。若未找到则返回 `None`。

**注意**：此处使用 `remove` 而非 `discard`，因为已通过 `if req.uid == uid` 确认元素存在，不会触发 `KeyError`；但在集合迭代过程中修改集合是不安全的，这里通过 `return` 立即退出迭代避免了问题。

---

#### `inflight_tokens` 属性

```python
# python/minisgl/scheduler/decode.py，第 27-30 行
@property
def inflight_tokens(self) -> int:
    tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
    return sum(req.remain_len for req in self.running_reqs) + tokens_reserved
```

**功能**：计算所有正在解码的请求未来还需要的 KV 缓存总量（token 数），供 `PrefillAdder` 初始化 `reserved_size` 使用。

**计算公式**：

```
inflight_tokens = Σ(req.remain_len) + (page_size - 1) * num_running_reqs
```

- `req.remain_len = max_device_len - device_len`：每个请求还剩余的生成 token 数
- `(page_size - 1) * num_running_reqs`：页对齐的内存开销——由于 KV 缓存按页分配，每个请求可能最多浪费 `page_size - 1` 个 token 的空间（末页未满）

此估算确保 prefill 调度时不会侵占已在解码的请求所需的 KV 缓存页。

---

#### `schedule_next_batch(self) -> Batch | None`

```python
# python/minisgl/scheduler/decode.py，第 32-35 行
def schedule_next_batch(self) -> Batch | None:
    if not self.runnable:
        return None
    return Batch(reqs=list(self.running_reqs), phase="decode")
```

将 `running_reqs` 集合转为列表，创建 decode `Batch`。解码批次包含所有可解码请求，没有 token budget 约束（每个请求每轮只生成 1 个 token，总开销为 `num_running_reqs`）。

**注意**：每次调用 `list(self.running_reqs)` 时顺序不确定（集合无序），但这对解码阶段没有影响，因为每个请求独立生成 token。

---

#### `runnable` 属性

```python
# python/minisgl/scheduler/decode.py，第 37-39 行
@property
def runnable(self) -> bool:
    return len(self.running_reqs) > 0
```

调度器主循环通过此属性判断是否有待解码的请求。

---

### C.4.2 DecodeManager 与 PrefillManager 的协作

```
PrefillManager.schedule_next_batch()
    └── PrefillAdder(reserved_size=decode_manager.inflight_tokens)
            ↑ 防止 prefill 超量分配，保护 decode 请求的 KV 缓存

Scheduler._forward(forward_input)
    └── decode_manager.filter_reqs(forward_input.batch.reqs)
            ↑ 将 prefill 完成的请求转移至 decode 集合

Scheduler._process_last_data(last_data)
    └── decode_manager.remove_req(req)  # 当 req 完成时
            ↑ 移除已结束的 decode 请求
```

---

### C.4.3 依赖关系

```
DecodeManager
    └── minisgl.core.Req
            ├── can_decode   — 决定是否留在 running_reqs
            └── remain_len   — 用于 inflight_tokens 计算
```

`DecodeManager` 是调度器中最简单的管理器，无外部服务依赖，仅操作 `Req` 对象的属性。

---

## C.5 `cache.py` — 缓存管理

**文件路径**：`python/minisgl/scheduler/cache.py`

**文件职责**：管理 GPU KV 缓存的物理页分配、回收和驱逐，以及通过前缀缓存（Radix Cache 或 Naive Cache）实现跨请求的 KV 复用。

### C.5.1 `CacheManager` 类

```python
# python/minisgl/scheduler/cache.py，第 15-25 行
class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
```

**关键数据结构**：

- **`free_slots`**：GPU 端的 `int32` 一维张量，存储所有空闲页的**起始 token 偏移量**。初始化为 `[0, page_size, 2*page_size, ..., (num_pages-1)*page_size]`，即每个空闲页的首个 token 在 KV 缓存中的物理位置。这种设计将页的物理地址以"页首 token offset"的形式表示，简化了与 `page_table` 的映射。
- **`prefix_cache`**：`BasePrefixCache` 实现，支持 `"radix"`（基数树）或 `"naive"`（朴素）两种类型，管理已完成请求的 KV 数据复用。
- **`page_table`**：与 `Engine.page_table` 共享同一 `torch.Tensor` 引用，形状 `[max_running_req+1, aligned_max_seq_len]`，存储每个请求每个 token 位置对应的 KV 缓存物理地址。

---

#### `match_req(self, req: PendingReq) -> MatchResult`

```python
# python/minisgl/scheduler/cache.py，第 27-30 行
def match_req(self, req: PendingReq) -> MatchResult:
    input_len = req.input_len
    assert input_len > 0, "Input length must be greater than 0."
    return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])
```

在前缀缓存中查找当前请求的最长公共前缀。

**`input_ids[: input_len - 1]`**（不含最后一个 token）的原因：前缀缓存存储的是"已完成 attention 计算的 token 序列"。最后一个 token 在预填充时需要作为"extend token"参与本次前向计算，才能将其 KV 写入缓存，因此不应纳入前缀匹配范围。

返回的 `MatchResult.cuda_handle` 提供：
- `cached_len`：匹配的前缀长度
- `get_matched_indices()`：匹配前缀对应的 KV 物理地址张量

---

#### `available_size` 属性

```python
# python/minisgl/scheduler/cache.py，第 32-34 行
@property
def available_size(self) -> int:
    return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size
```

计算当前可用的总 token 容量：

```
available_size = evictable_size + free_pages * page_size
```

- `evictable_size`：前缀缓存中可以驱逐的 token 数（未被任何请求锁定的缓存条目）
- `len(self.free_slots) * page_size`：已知空闲页可提供的 token 数

**设计含义**：可驱逐的缓存页和空闲页都是"可以被新请求使用"的资源，两者之和即为系统当前的可用容量。

---

#### `lock(self, handle: BaseCacheHandle) -> None` / `unlock(self, handle: BaseCacheHandle) -> None`

```python
# python/minisgl/scheduler/cache.py，第 36-41 行
def lock(self, handle: BaseCacheHandle) -> None:
    self.prefix_cache.lock_handle(handle, unlock=False)

def unlock(self, handle: BaseCacheHandle) -> None:
    self.prefix_cache.lock_handle(handle, unlock=True)
```

包装前缀缓存的锁定/解锁操作。

- **锁定**：将 handle 对应的缓存条目从 `evictable_size` 移入 `protected_size`，防止在请求持有期间被驱逐
- **解锁**：将 handle 从 `protected_size` 移回 `evictable_size`，允许其被后续请求驱逐

**锁的生命周期**：
- 在 `PrefillAdder._try_allocate_one()` 中 `lock()`
- 在 `cache_req()` 中 `unlock()`（插入前缀缓存时解锁）

---

#### `allocate_paged(self, reqs: List[Req]) -> None`

```python
# python/minisgl/scheduler/cache.py，第 42-53 行
def allocate_paged(self, reqs: List[Req]) -> None:
    needed_pages = 0
    allocation_info: List[Tuple[int, int, int]] = []
    for req in reqs:
        first_page = div_ceil(req.cached_len, self.page_size)
        last_page = div_ceil(req.device_len, self.page_size)
        if last_page > first_page:
            needed_pages += last_page - first_page
            allocation_info.append((req.table_idx, first_page, last_page))
    if needed_pages > 0:
        allocated = self._page_to_token(self._allocate(needed_pages))
        _write_page_table(self.page_table, allocated, allocation_info, self.page_size)
```

为批次中的所有请求分配尚未分配的 KV 缓存页，并将物理地址写入 `page_table`。

**页范围计算**：

```
first_page = ceil(cached_len / page_size)   # 前缀缓存已覆盖的页边界
last_page  = ceil(device_len / page_size)   # 本轮 extend 结束后的页边界
需分配页数 = last_page - first_page
```

使用 `div_ceil`（向上取整）确保即使 `device_len` 不是 `page_size` 的整数倍，末尾的不完整页也会被分配。

**两阶段操作**：
1. 先扫描所有请求，汇总总需页数（避免多次分配）
2. 一次性 `_allocate(needed_pages)` 获取连续的页地址张量
3. 调用 `_write_page_table()` 将地址批量写入 GPU 端的页表

---

#### `cache_req(self, req: Req, *, finished: bool) -> None`

```python
# python/minisgl/scheduler/cache.py，第 55-79 行
def cache_req(self, req: Req, *, finished: bool) -> None:
    insert_ids = req.input_ids[: req.cached_len]
    page_indices = self.page_table[req.table_idx, : req.cached_len]
    old_handle = req.cache_handle
    cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
    # unlock until all operations on handle is done
    self.unlock(old_handle)
    # this part is already in the prefix cache, free it
    self._free(page_indices[old_handle.cached_len : cached_len])
    if finished:  # this tail part should be freed
        self._free(page_indices[new_handle.cached_len :])
    else:  # keep the tail part, update the handle
        req.cache_handle = new_handle
        self.lock(new_handle)
```

将请求已计算的 KV 数据插入前缀缓存，并处理相关页的释放。这是缓存管理中最复杂的方法。

**内存区域划分**（注释中详细说明）：

```
token 偏移:  0        old_handle.cached_len    cached_len    new_handle.cached_len    req.cached_len
             |<-- 前缀缓存已有 -->|<-- 本次新分配 -->|<-- 已被其他请求缓存 -->|<-- 尾部未完整页 -->|

其中：
  [0, old_handle.cached_len)        → 已在前缀缓存，对应物理页不变
  [old_handle.cached_len, cached_len) → 在此次 insert 之前已被其他请求插入缓存，本请求分配了重复页 → 需释放
  [cached_len, new_handle.cached_len) → 本次 insert_prefix 新插入的部分 → 归前缀缓存所有
  [new_handle.cached_len, req.cached_len) → 尾部（可能是不完整的最后一页）→ finished 时释放，否则保留
```

**操作流程**：

1. **插入前缀**：`prefix_cache.insert_prefix()` 返回 `(cached_len, new_handle)`，其中 `cached_len` 是插入前已存在于缓存中的长度
2. **解锁旧句柄**：`unlock(old_handle)` — 此前由 `PrefillAdder` 锁定，现在可以解锁
3. **释放重复页**：`[old_handle.cached_len, cached_len)` 部分已被其他请求插入缓存，本请求持有的物理页属于重复分配，需归还
4. **处理尾部**：
   - `finished=True`：请求已结束，尾部物理页释放
   - `finished=False`：请求仍在解码，尾部物理页保留，更新句柄并重新锁定

---

#### `check_integrity(self) -> None`

```python
# python/minisgl/scheduler/cache.py，第 81-91 行
def check_integrity(self) -> None:
    self.prefix_cache.check_integrity()
    cache_pages = self.prefix_cache.size_info.total_size // self.page_size
    if len(self.free_slots) + cache_pages != self.num_pages:
        raise RuntimeError(
            "CacheManager integrity check failed: ..."
        )
    if self.page_size > 1:
        assert torch.all(self.free_slots % self.page_size == 0)
```

在空闲时执行的完整性校验：
- 验证前缀缓存内部一致性
- 验证 `free_pages + cache_pages == num_pages`（没有内存泄漏）
- 验证所有空闲槽位按页对齐

---

#### `lazy_free_region(self)` 上下文管理器

```python
# python/minisgl/scheduler/cache.py，第 93-104 行
@contextmanager
def lazy_free_region(self):
    def lazy_free(indices: torch.Tensor) -> None:
        lazy_free_list.append(indices[:: self.page_size])

    lazy_free_list: List[torch.Tensor] = []
    try:
        self._free = lazy_free
        yield
    finally:
        del self._free
        self.free_slots = torch.cat([self.free_slots] + lazy_free_list)
```

**性能优化设计**：在 `_process_last_data()` 的循环中，可能有多个请求同时完成并释放页。若每次都调用 `torch.cat()` 拼接 `free_slots`，会造成大量小张量拼接，性能低下。

`lazy_free_region` 通过**猴子补丁**临时替换 `self._free` 方法：在上下文内，所有 `_free()` 调用都只是将待释放的页地址追加到 `lazy_free_list`；上下文退出时，一次性执行 `torch.cat` 合并所有待释放地址。

**`indices[:: self.page_size]`**：只取每 `page_size` 个 token 的第一个（页首 token），因为 `free_slots` 以页首 token 偏移为单位存储。

---

#### `_allocate(self, needed_pages: int) -> torch.Tensor`

```python
# python/minisgl/scheduler/cache.py，第 106-113 行
def _allocate(self, needed_pages: int) -> torch.Tensor:
    if needed_pages > (free_pages := len(self.free_slots)):
        evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
        self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
        assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
    allocated = self.free_slots[:needed_pages]
    self.free_slots = self.free_slots[needed_pages:]
    return allocated
```

分配指定数量的物理页：

1. 若 `free_slots` 不足，调用 `prefix_cache.evict()` 驱逐最少最近使用（LRU）的前缀缓存条目
2. 将驱逐得到的页地址合并入 `free_slots`
3. 从 `free_slots` 头部取出所需页数并返回

**注意**：`evict()` 返回的是以 token 为单位的物理地址，`[:: self.page_size]` 采样取页首地址。

---

#### `_free(self, indices: torch.Tensor) -> None`

```python
# python/minisgl/scheduler/cache.py，第 115-117 行
def _free(self, indices: torch.Tensor) -> None:
    if len(indices) > 0:
        self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])
```

将物理页地址归还到 `free_slots`。在 `lazy_free_region` 上下文内此方法被替换为延迟版本；上下文外则直接执行。

---

#### `_page_to_token(self, pages: torch.Tensor) -> torch.Tensor`

```python
# python/minisgl/scheduler/cache.py，第 119-124 行
def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
    if self.page_size == 1:
        return pages
    offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
    return (pages.unsqueeze(1) + offsets).flatten()
```

将页首地址列表展开为每个 token 的物理地址列表：

```
输入: [0, 16, 32]    (page_size=16 时的三个页首地址)
输出: [0,1,...,15, 16,17,...,31, 32,33,...,47]
```

用于 `allocate_paged()` 中将分配的页地址写入 `page_table`（`page_table` 以 token 为粒度存储地址）。

---

### C.5.2 `_write_page_table` 模块级函数

```python
# python/minisgl/scheduler/cache.py，第 127-146 行
def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated
```

**功能**：将 `allocate_paged()` 中分配的物理页地址批量写入 GPU 端的 `page_table`。

**工作流程**：

1. 在 CPU 侧构建 `table_idx_host` 和 `positions_host`（pin_memory 保证快速传输）
2. `table_idx_host[i]` = 第 i 个 token 属于哪个请求的行索引
3. `positions_host[i]` = 第 i 个 token 在序列中的位置（列索引）
4. 非阻塞异步拷贝到 GPU
5. 通过二维高级索引：`page_table[table_idxs, offsets] = allocated`，一次 kernel 完成所有写入

这种向量化写入方式比逐请求循环写入高效得多，特别是在大批次场景下。

---

### C.5.3 依赖关系

```
CacheManager
    ├── BasePrefixCache (minisgl.kvcache)
    │       ├── match_prefix()    — 前缀查找
    │       ├── insert_prefix()   — 前缀插入
    │       ├── evict()           — LRU 驱逐
    │       └── lock_handle()     — 锁定保护
    ├── page_table (Engine.page_table) — 共享引用
    └── div_ceil (minisgl.utils.misc)  — 页对齐计算
```

---

## C.6 `table.py` — 表管理

**文件路径**：`python/minisgl/scheduler/table.py`

**文件职责**：管理请求到行索引（`table_idx`）的分配与回收，并维护 `token_pool`（存储每个请求每个位置的 token id）与 `page_table`（存储 KV 缓存物理地址）这两张核心 GPU 张量。

### C.6.1 `TableManager` 类

```python
# python/minisgl/scheduler/table.py，第 4-11 行
class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)
```

**核心数据结构**：

**`page_table`**（共享引用，由 `Engine` 分配）：
- 形状：`[max_running_req + 1, aligned_max_seq_len]`，`dtype=torch.int32`，设备：GPU
- 含义：`page_table[table_idx, pos]` = token `pos` 对应的 KV 缓存物理地址（以 token offset 计，而非页号）
- 额外的 `+1` 行用于 dummy 请求（用于 CUDA Graph padding）

**`token_pool`**（由 `TableManager` 创建）：
- 形状：与 `page_table` 相同，`dtype=torch.int32`，设备：GPU
- 含义：`token_pool[table_idx, pos]` = 请求 `table_idx` 在位置 `pos` 的 token id
- 用途：`_forward()` 中通过 `batch.input_ids = token_pool[input_mapping]` 读取输入 token；`token_pool[output_mapping] = next_tokens_gpu` 写入新生成的 token
- 初始化为 0（避免 dummy 请求读取未初始化内存）

**`_free_slots`**：Python `list`，初始化为 `[0, 1, 2, ..., max_running_reqs-1]`，维护空闲行索引的池（栈式管理，`pop()` 分配，`append()` 回收）。使用 Python list 而非 GPU 张量，因为行索引管理是纯 CPU 操作，频率低，不需要 GPU 加速。

---

**两张张量的关系图**：

```
table_idx=0: [token_0, token_1, token_2, ..., token_N]   ← token_pool 行
             [  kv_0,   kv_1,   kv_2, ...,   kv_N ]   ← page_table 行
table_idx=1: ...
...

token_pool[i, j]: 请求 i 第 j 个位置的 token id（int32）
page_table[i, j]: 请求 i 第 j 个位置的 KV 缓存物理地址（int32，token offset 形式）
```

---

#### `available_size` 属性

```python
# python/minisgl/scheduler/table.py，第 13-15 行
@property
def available_size(self) -> int:
    return len(self._free_slots)
```

返回当前可分配的空闲 `table_idx` 数量，即可以新接入的请求数上限。`PrefillAdder._try_allocate_one()` 在分配前检查此值，确保不超过 `max_running_reqs`。

---

#### `allocate(self) -> int`

```python
# python/minisgl/scheduler/table.py，第 17-18 行
def allocate(self) -> int:
    return self._free_slots.pop()
```

从空闲池中弹出一个 `table_idx`，分配给新请求。使用 `list.pop()` 即 LIFO（后进先出）顺序，这对调度没有语义影响（任意空闲槽都等价），但 LIFO 有更好的缓存局部性（最近释放的行可能仍在 CPU 缓存中）。

**调用时机**：`PrefillAdder._try_allocate_one()` 确认资源充足后调用。

---

#### `free(self, slot: int) -> None`

```python
# python/minisgl/scheduler/table.py，第 20-21 行
def free(self, slot: int) -> None:
    self._free_slots.append(slot)
```

将 `table_idx` 归还到空闲池。`Scheduler._free_req_resources()` 在请求完成或中止时调用此方法。

**注意**：`free()` 不清零 `token_pool` 和 `page_table` 对应行——这些数据被后续分配的请求覆盖时自然失效，不需要显式清除（避免额外的 GPU 操作）。

---

### C.6.2 TableManager 与 page_table/token_pool 的数据流

```
PrefillAdder._try_allocate_one()
    └── TableManager.allocate()        → 获取 table_idx
         └── token_pool[table_idx, :cached_len] ← 前缀命中的 token ids
         └── page_table[table_idx, :cached_len] ← 前缀命中的 KV 物理地址

PrefillAdder._add_one_req()
    └── token_pool[table_idx, cached_len:cached_len+chunk_size]
              ← input_ids[cached_len:cached_len+chunk_size]  (非阻塞 CPU→GPU)

CacheManager.allocate_paged()
    └── page_table[table_idx, first_page*ps:last_page*ps]
              ← 新分配的 KV 物理地址 (非阻塞 CPU→GPU)

Scheduler._forward()
    └── batch.input_ids = token_pool[input_mapping]   (GPU 读)
    └── token_pool[write_mapping] = next_tokens_gpu   (GPU 写)

Scheduler._free_req_resources()
    └── TableManager.free(table_idx)  → 回收 table_idx
```

---

### C.6.3 依赖关系

```
TableManager
    └── page_table (Engine.page_table)  — 共享张量引用（由 Engine 分配，TableManager 持有）
```

`TableManager` 是调度器中最简单的有状态类之一，无复杂外部依赖，专注于 `table_idx` 的生命周期管理和两张核心张量的访问入口。

---

## C.7 `io.py` — I/O 通信

**文件路径**：`python/minisgl/scheduler/io.py`

**文件职责**：实现调度器与 tokenizer 进程之间的 ZMQ 消息收发，以及多 TP rank 场景下 rank 0 对其他 rank 的消息广播；通过 Mixin 模式与 `Scheduler` 组合。

### C.7.1 `SchedulerIOMixin` 类

```python
# python/minisgl/scheduler/io.py，第 15-25 行
class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles the communication between the scheduler and the tokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send results back to the tokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.
    """
```

`SchedulerIOMixin` 采用 **Mixin 设计模式**，将 I/O 逻辑从调度逻辑中解耦。`Scheduler` 通过继承获得 `receive_msg`、`send_result`、`sync_all_ranks` 三个公共接口，而无需关心其底层实现（单卡/多卡、在线/离线）。

---

#### `__init__(self, config: SchedulerConfig, tp_cpu_group: ProcessGroup)`

```python
# python/minisgl/scheduler/io.py，第 27-65 行
def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
    tp_info = config.tp_info
    self.tp_cpu_group: Final = tp_cpu_group
    if config.offline_mode:
        self.receive_msg = self.offline_receive_msg
        self.send_result = self.offline_send_result
        return  # early exit

    if tp_info.is_primary():
        self._recv_from_tokenizer: Final = ZmqPullQueue(
            config.zmq_backend_addr,
            create=True,
            decoder=BaseBackendMsg.decoder,
        )
        self._send_into_tokenizer: Final = ZmqPushQueue(
            config.zmq_detokenizer_addr,
            create=config.backend_create_detokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        )

    recv = self._recv_msg_single_rank
    send = self._reply_tokenizer_rank0
    if tp_info.size > 1:
        if tp_info.is_primary():
            recv = self._recv_msg_multi_rank0
            self._send_into_ranks: Final = ZmqPubQueue(
                config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
            )
        else:
            recv = self._recv_msg_multi_rank1
            send = self._reply_tokenizer_rank1
            self._recv_from_rank0: Final = ZmqSubQueue(
                config.zmq_scheduler_broadcast_addr,
                create=False,
                decoder=BaseBackendMsg.decoder,
            )

    self.receive_msg = recv
    self.send_result = send
```

`__init__` 实现了**策略模式**：根据配置（单卡/多卡、在线/离线）动态绑定 `receive_msg` 和 `send_result` 到对应的实现方法。

**四种场景的分配**：

| 场景 | `receive_msg` | `send_result` |
|------|-------------|---------------|
| 离线模式 | `offline_receive_msg` | `offline_send_result` |
| 单卡在线 | `_recv_msg_single_rank` | `_reply_tokenizer_rank0` |
| 多卡 rank 0 | `_recv_msg_multi_rank0` | `_reply_tokenizer_rank0` |
| 多卡 rank N | `_recv_msg_multi_rank1` | `_reply_tokenizer_rank1` |

**ZMQ socket 初始化策略**：

- `rank 0` (`is_primary()`) 创建所有与 tokenizer 通信的 socket（bind）
- 其他 rank 通过 ZMQ PUB/SUB 订阅 rank 0 的广播
- 多卡模式下 `_send_into_ranks` 使用 `ZmqPubQueue`（PUB 模式，广播给所有订阅者）

---

#### `sync_all_ranks(self) -> None`

```python
# python/minisgl/scheduler/io.py，第 76-77 行
def sync_all_ranks(self) -> None:
    self.tp_cpu_group.barrier().wait()
```

通过 CPU 端的 `ProcessGroup`（gloo 后端）执行全同步屏障，确保所有 TP rank 在同一时间点就绪。用于 `Scheduler.shutdown()` 的安全关闭流程。

---

#### `_recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]`

```python
# python/minisgl/scheduler/io.py，第 79-86 行
def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
    pending_msgs: List[BaseBackendMsg] = []
    if blocking:
        self.run_when_idle()
        pending_msgs.append(self._recv_from_tokenizer.get())
    while not self._recv_from_tokenizer.empty():
        pending_msgs.append(self._recv_from_tokenizer.get())
    return pending_msgs
```

单卡模式的消息接收：

- **`blocking=True`**（调度器空闲时）：先调用 `run_when_idle()` 执行后台任务，然后阻塞等待第一条消息
- **`blocking=False`**（有任务待处理时）：非阻塞轮询，将队列中所有待处理消息一次性取出（批量处理）

使用 `empty()` + `get()` 的 drain 模式，确保不遗漏任何堆积的消息。

---

#### `_recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]`

```python
# python/minisgl/scheduler/io.py，第 88-107 行
def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
    pending_msgs: List[BaseBackendMsg] = []
    if blocking:
        self.run_when_idle()
        raw = self._recv_from_tokenizer.get_raw()
        self._send_into_ranks.put_raw(raw)
        pending_msgs.append(self._recv_from_tokenizer.decode(raw))

    pending_raw_msgs: List[bytes] = []
    while not self._recv_from_tokenizer.empty():
        pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

    # broadcast the number of raw messages to all ranks
    src_tensor = torch.tensor(len(pending_raw_msgs))
    self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

    for raw in pending_raw_msgs:
        self._send_into_ranks.put_raw(raw)
        pending_msgs.append(self._recv_from_tokenizer.decode(raw))
    return pending_msgs
```

多卡 rank 0 的消息接收与广播逻辑：

**关键设计**：rank 0 接收消息后，需确保其他 rank 收到完全相同的消息（保证所有 rank 的调度决策一致）。因此：

1. 使用 `get_raw()`（取原始 bytes）+ `put_raw(raw)` 将原始消息字节转发给 PUB socket
2. 本地再 `decode(raw)` 解析
3. **广播消息数量**：`tp_cpu_group.broadcast(src_tensor, root=0)`，让其他 rank 知道本轮应接收几条消息，避免其他 rank 无限等待或提前结束
4. 广播数量是必要的，因为非阻塞 drain 阶段的消息数量是动态的

---

#### `_recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]`

```python
# python/minisgl/scheduler/io.py，第 109-122 行
def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
    pending_msgs: List[BaseBackendMsg] = []
    if blocking:
        self.run_when_idle()
        pending_msgs.append(self._recv_from_rank0.get())

    # ensure all ranks have the same number of raw messages
    dst_tensor = torch.tensor(-1)
    self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
    dst_length = int(dst_tensor.item())

    for _ in range(dst_length):
        pending_msgs.append(self._recv_from_rank0.get())
    return pending_msgs
```

多卡非主 rank 的消息接收逻辑：

- **阻塞模式**：直接从 `_recv_from_rank0`（SUB socket）阻塞接收第一条消息
- **接收数量同步**：等待 rank 0 广播的消息数量（`dst_tensor`，初始为 -1，被 broadcast 覆盖）
- 按照 rank 0 广播的数量精确接收，确保与 rank 0 的状态严格同步

**注意**：`_recv_msg_multi_rank1` 中的 `blocking=True` 路径同样调用 `run_when_idle()`，对非主 rank 而言这是安全的（`run_when_idle` 只是日志和缓存检查）。

---

#### `_reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None`

```python
# python/minisgl/scheduler/io.py，第 124-130 行
def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
    num_reply = len(reply)
    logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
    if num_reply == 1:
        self._send_into_tokenizer.put(reply[0])
    elif num_reply > 1:
        self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))
```

向 tokenizer 发送采样结果：

- **单条消息**：直接发送 `DetokenizeMsg`，避免不必要的封装开销
- **多条消息**：打包为 `BatchTokenizerMsg`，一次 ZMQ send 传输，减少系统调用次数
- **零条消息**：不发送（空 batch，如全为 ChunkedReq 的批次）

---

#### `_reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None`

```python
# python/minisgl/scheduler/io.py，第 132-133 行
def _reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None:
    _ = reply  # do nothing for non-primary ranks
```

非主 rank 不向 tokenizer 发送结果（所有结果统一由 rank 0 发送），避免重复。

---

#### `offline_receive_msg` / `offline_send_result`（抽象接口）

```python
# python/minisgl/scheduler/io.py，第 70-74 行
def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
    raise NotImplementedError("should be implemented")

def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
    raise NotImplementedError("should be implemented")
```

离线模式的消息收发接口，由子类（如 `LLM` 类）覆盖实现，用于批量推理场景（不通过 ZMQ，直接在内存中传递数据）。

---

### C.7.2 ZMQ 通信架构总览

```
tokenizer 进程                    rank 0 调度器               rank 1..N 调度器
─────────────────────────────────────────────────────────────────────────────
    ZmqPushQueue  →  zmq_backend_addr  ←  ZmqPullQueue
                                                │
                          ZmqPubQueue  ──────────┼─────────────────────────→  ZmqSubQueue
                       (zmq_scheduler_broadcast_addr)
                                                │
    ZmqPullQueue  ←  zmq_detokenizer_addr  ←  ZmqPushQueue
                          (rank 0 only)
```

**消息编码**：所有 ZMQ 消息使用 `msgpack` 序列化，通过 `BaseBackendMsg.encoder/decoder` 和 `BaseTokenizerMsg.encoder/decoder` 进行类型安全的序列化/反序列化。

---

### C.7.3 依赖关系

```
SchedulerIOMixin
    ├── ZmqPullQueue / ZmqPushQueue  (minisgl.utils.mp)   — IPC PUSH/PULL
    ├── ZmqPubQueue / ZmqSubQueue    (minisgl.utils.mp)   — IPC PUB/SUB
    ├── BaseBackendMsg               (minisgl.message)    — 消息类型与编解码
    ├── BaseTokenizerMsg             (minisgl.message)    — 结果消息类型
    ├── torch.distributed.ProcessGroup                    — CPU 端 TP 同步
    └── SchedulerConfig              (scheduler.config)   — 地址与模式配置
```

---

## C.8 `utils.py` — 工具函数

**文件路径**：`python/minisgl/scheduler/utils.py`

**文件职责**：定义调度器内部使用的辅助数据类，包括 `PendingReq`（待处理请求的内存表示）和 `ScheduleResult`（调度结果的封装，当前仅用于类型标注）。

### C.8.1 `PendingReq` 数据类

```python
# python/minisgl/scheduler/utils.py，第 14-28 行
@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens
```

`PendingReq` 是请求在**预填充等待队列**中的内存表示，是 `UserMsg`（网络消息）和 `Req`（执行对象）之间的中间状态。

**字段说明**：

| 字段 | 类型 | 含义 |
|------|------|------|
| `uid` | `int` | 请求唯一标识符，贯穿请求全生命周期 |
| `input_ids` | `torch.Tensor` | CPU 端的完整输入 token 序列（`int32`，一维） |
| `sampling_params` | `SamplingParams` | 采样超参（temperature、top_k、top_p、max_tokens 等） |
| `chunked_req` | `ChunkedReq | None` | 若请求正在分块预填充中，指向当前的 `ChunkedReq` 对象；否则为 `None` |

**`input_len` 属性**：

```python
@property
def input_len(self) -> int:
    return len(self.input_ids)
```

返回输入序列的总长度。在 `PrefillAdder._try_allocate_one()` 中用于计算 `extend_len` 和 `estimated_len`。

**`output_len` 属性**：

```python
@property
def output_len(self) -> int:
    return self.sampling_params.max_tokens
```

返回请求期望生成的最大 token 数（用于 KV 缓存容量预估）。

---

#### `PendingReq` 与 `Req` 的关系

`PendingReq` 和 `Req` 携带信息的比较：

| 字段 | `PendingReq` | `Req` |
|------|-------------|-------|
| `uid` | ✓ | ✓ |
| `input_ids` | ✓（完整） | ✓（截取到 cached+chunk） |
| `sampling_params` | ✓ | ✓ |
| `table_idx` | ✗（由 TableManager 分配后存入 Req） | ✓ |
| `cached_len` | ✗（从前缀缓存 handle 获取） | ✓ |
| `output_len` | ✓（max_tokens） | ✓ |
| `cache_handle` | ✗ | ✓ |
| `chunked_req` | ✓（指向 ChunkedReq） | — |

关键区别：`PendingReq` 不持有 GPU 资源（无 `table_idx`、无 `cache_handle`），代表"等待中"的请求；`Req` 已拥有分配好的 GPU 资源，代表"正在执行"的请求。

---

#### `chunked_req` 字段的状态语义

`PendingReq.chunked_req` 编码了请求的分块状态：

```
None         → 请求尚未开始预填充，或已完成当前分块（等待分配新 table_idx）
ChunkedReq   → 请求正在分块预填充中，持有未完成的 GPU 资源
```

在 `PrefillManager.schedule_next_batch()` 中，每轮调度前先清空 `pending_req.chunked_req = None`（第 141 行），然后根据本轮是否继续分块来重新赋值（第 143 行）。

---

### C.8.2 `ScheduleResult` 数据类

```python
# python/minisgl/scheduler/utils.py，第 30-33 行
@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
```

`ScheduleResult` 是一个辅助数据类，用于封装调度结果，但在当前代码中未被主流程实际使用（可能是早期设计的遗留，或为未来扩展预留）。

| 字段 | 类型 | 含义 |
|------|------|------|
| `reqs` | `List[PendingReq]` | 本轮调度选中的请求列表 |
| `output_indices` | `List[torch.Tensor]` | 各请求的输出索引张量 |

---

### C.8.3 依赖关系

```
PendingReq
    ├── SamplingParams (minisgl.core)  — 采样参数
    └── ChunkedReq (scheduler.prefill) — 分块请求引用（循环依赖，TYPE_CHECKING 下解决）

ScheduleResult
    └── PendingReq
```

**循环依赖处理**：`utils.py` 中 `ChunkedReq` 的类型标注使用了 `TYPE_CHECKING` 导入保护，避免运行时的循环导入：

```python
# python/minisgl/scheduler/utils.py，第 8-11 行
if TYPE_CHECKING:
    from minisgl.core import SamplingParams
    from .prefill import ChunkedReq
```

---

## C.9 模块间协作总结

### C.9.1 完整请求生命周期

以一个完整请求的处理流程为例，展示所有模块的协作：

```
1. 接收阶段
   SchedulerIOMixin.receive_msg()
       └── ZmqPullQueue.get() → UserMsg
   Scheduler._process_one_msg(UserMsg)
       └── PrefillManager.add_one_req() → PendingReq 入队

2. 预填充调度阶段
   PrefillManager.schedule_next_batch(prefill_budget)
       └── PrefillAdder(token_budget, reserved_size=inflight_tokens)
               └── try_add_one(pending_req)
                       ├── CacheManager.match_req()         ← 前缀匹配
                       ├── CacheManager.lock(handle)        ← 锁定缓存句柄
                       ├── TableManager.allocate()          ← 分配 table_idx
                       ├── token_pool[idx, :cached_len] ← 前缀 token ids
                       ├── page_table[idx, :cached_len] ← 前缀 KV 地址
                       └── token_pool[idx, cached:cached+chunk] ← extend token ids

3. 批次准备阶段
   Scheduler._prepare_batch(batch)
       ├── GraphRunner.pad_batch(batch)         ← CUDA Graph padding
       ├── CacheManager.allocate_paged(reqs)    ← 分配新 KV 页，写入 page_table
       ├── _make_positions(batch, device)        ← 构建位置索引张量
       ├── _make_input_tuple(batch, device)      ← 构建 token 读取索引
       ├── _make_write_tuple(batch, device)      ← 构建 token 写入索引
       ├── page_table[input_mapping]             ← 生成 KV 物理地址 (out_loc)
       └── AttnBackend.prepare_metadata(batch)   ← 准备 attention 元数据

4. 前向计算阶段 (engine.stream)
   Scheduler._forward(forward_input)
       ├── token_pool[input_mapping]             ← 读取输入 token ids (GPU)
       ├── Engine.forward_batch(batch, args)
       │       ├── model.forward()               ← Transformer 计算
       │       ├── req.complete_one()            ← 更新 device_len, cached_len
       │       └── sampler.sample()              ← 采样生成 next_token
       ├── token_pool[output_mapping] = next_tokens_gpu  ← 写回新 token
       └── DecodeManager.filter_reqs(batch.reqs)
               └── 将 can_decode=True 的请求加入 running_reqs

5. 结果处理阶段 (overlap 与步骤 4 并行)
   Scheduler._process_last_data(last_data)
       ├── copy_done.synchronize()               ← 等待 GPU→CPU token 拷贝
       ├── with cache_manager.lazy_free_region():
       │       ├── req.append_host(next_token)   ← 追加 token 到 CPU 序列
       │       ├── 检测 EOS 或 max_tokens
       │       ├── DecodeManager.remove_req(req) ← 已结束请求
       │       ├── TableManager.free(table_idx)  ← 归还行索引
       │       ├── CacheManager.cache_req(finished=True/False) ← 缓存或释放
       │       └── reply.append(DetokenizeMsg)
       └── SchedulerIOMixin.send_result(reply)
               └── ZmqPushQueue.put(DetokenizeMsg)  ← 发送给 detokenizer

6. 解码阶段（循环执行直到请求结束）
   DecodeManager.schedule_next_batch()
       └── Batch(reqs=list(running_reqs), phase="decode")
   → 回到步骤 3（prepare_batch → forward → process_last_data）
```

---

### C.9.2 Overlap Scheduling 时序图

```
时间轴 →
                t=0              t=1              t=2              t=3
CPU (self.stream):
  接收消息 A   ─────▶
  调度批次 A   ──────────▶
                        准备批次 A (CPU侧) ────▶
  (等待 engine stream) ──┐
                         └─ wait_stream
                                          处理批次 A 结果 ──────────▶
                                          接收消息 B ────────▶
                                          调度批次 B ──────────────▶
                                                   准备批次 B ─────▶

GPU (engine.stream):
                             计算批次 A ──────────────────────▶
                                                         计算批次 B ──────▶

overlap 期间: CPU 处理批次 A 的结果（`_process_last_data`）与 GPU 计算批次 B 同时进行
```

---

### C.9.3 KV 缓存内存管理状态机

```
空闲页
  │  _allocate()
  ▼
已分配页（通过 page_table 映射给特定请求）
  │  cache_req(finished=False)：insert_prefix
  ▼
前缀缓存（evictable）
  │  lock_handle()
  ▼
前缀缓存（protected）
  │  unlock_handle()
  ▼
前缀缓存（evictable）
  │  evict() or free()
  ▼
空闲页

特殊情况：
- 请求结束：cache_req(finished=True) → 尾部页直接回到空闲页
- 缓存已满：evict() → 驱逐 evictable 页 → 转为空闲页 → 重新分配
```

---

### C.9.4 各文件职责速查

| 文件 | 核心类/函数 | 主要职责 |
|------|-----------|---------|
| `config.py` | `SchedulerConfig` | 配置参数，ZMQ 地址生成 |
| `scheduler.py` | `Scheduler` | 主循环、Overlap Scheduling、批次准备、结果处理 |
| `prefill.py` | `PrefillManager`, `PrefillAdder`, `ChunkedReq` | Chunked Prefill 调度与资源分配 |
| `decode.py` | `DecodeManager` | 解码请求集合管理，inflight token 估算 |
| `cache.py` | `CacheManager` | KV 页分配/回收/驱逐，前缀缓存插入 |
| `table.py` | `TableManager` | table_idx 生命周期，token_pool/page_table 访问 |
| `io.py` | `SchedulerIOMixin` | ZMQ 消息收发，TP 广播，离线模式接口 |
| `utils.py` | `PendingReq`, `ScheduleResult` | 请求队列元素定义 |

---