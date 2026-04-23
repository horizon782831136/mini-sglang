# 附录 A：入口与服务层代码详解

本附录对 mini-sglang 入口与服务层的全部源代码进行逐文件的深度解析，覆盖每个公开类与函数的签名、参数、返回值、实现细节，以及模块间的依赖关系。阅读本附录前，建议先掌握 ZMQ 进程间通信的基本概念和 Python `multiprocessing` 模块的工作方式。

---

## 目录

1. [`python/minisgl/__main__.py`](#1-pythonminisgл__main__py) — CLI 入口
2. [`python/minisgl/server/args.py`](#2-pythonminisglserverargspy) — 参数解析
3. [`python/minisgl/server/launch.py`](#3-pythonminisglserverlaunchpy) — 服务启动
4. [`python/minisgl/server/api_server.py`](#4-pythonminisglserverapiserverpy) — HTTP API 服务
5. [`python/minisgl/shell.py`](#5-pythonminisglshellpy) — 交互式 Shell
6. [`python/minisgl/env.py`](#6-pythonminisglenvpy) — 环境变量配置
7. [`python/minisgl/llm/llm.py`](#7-pythonminisglllmllmpy) — 高层 LLM 接口

---

## 1. `python/minisgl/__main__.py`

### 1.1 文件职责

作为 Python 包的可执行入口（`python -m minisgl`），将控制权直接委托给 `server.launch_server()`，以 **HTTP 服务模式** 启动系统。

### 1.2 代码全文（含行号）

```python
# python/minisgl/__main__.py
1: from .server import launch_server
2:
3: assert __name__ == "__main__"
4:
5: launch_server()
```

### 1.3 关键实现细节

- **断言保护**（第 3 行）：`assert __name__ == "__main__"` 确保该文件只能作为主程序执行，无法被 `import` 引入。这是一种防御性编程手段，防止意外导入时触发服务器启动。
- **模式区分**：不传入 `run_shell=True`，因此 `launch_server()` 默认以 HTTP 服务器模式运行，与 `shell.py` 形成功能对照。
- **最小化设计**：文件仅 5 行，遵循单一职责原则，将所有真实逻辑委托给 `server` 子包。

### 1.4 依赖关系

| 被依赖模块 | 引入内容 |
|---|---|
| `minisgl.server` | `launch_server` 函数 |

---

## 2. `python/minisgl/server/args.py`

### 2.1 文件职责

负责命令行参数解析，将用户传入的字符串参数转换并验证为类型安全的冻结数据类 `ServerArgs`，作为整个服务的唯一配置来源。

### 2.2 类：`ServerArgs`

#### 签名

```python
@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False
```

#### 继承链

`ServerArgs` → `SchedulerConfig` → `EngineConfig`

完整字段继承关系如下表：

| 字段 | 类型 | 默认值 | 来源层 | 说明 |
|---|---|---|---|---|
| `model_path` | `str` | 必填 | `EngineConfig` | 模型权重路径或 HuggingFace repo ID |
| `tp_info` | `DistributedInfo` | 必填 | `EngineConfig` | 张量并行分布式信息 |
| `dtype` | `torch.dtype` | 必填 | `EngineConfig` | 模型计算精度 |
| `max_running_req` | `int` | 256 | `EngineConfig` | 最大并发请求数 |
| `attention_backend` | `str` | `"auto"` | `EngineConfig` | 注意力计算后端 |
| `moe_backend` | `str` | `"auto"` | `EngineConfig` | MoE 路由后端 |
| `cuda_graph_max_bs` | `int \| None` | `None` | `EngineConfig` | CUDA Graph 最大批大小 |
| `page_size` | `int` | 1 | `EngineConfig` | KV 缓存页大小 |
| `memory_ratio` | `float` | 0.9 | `EngineConfig` | GPU 内存使用比例 |
| `use_dummy_weight` | `bool` | `False` | `EngineConfig` | 是否使用虚假权重（测试用） |
| `use_pynccl` | `bool` | `True` | `EngineConfig` | 是否启用 PyNCCL |
| `max_seq_len_override` | `int \| None` | `None` | `EngineConfig` | 覆盖最大序列长度 |
| `num_page_override` | `int \| None` | `None` | `EngineConfig` | 覆盖 KV 缓存页数 |
| `max_extend_tokens` | `int` | 8192 | `SchedulerConfig` | Chunk Prefill 最大 token 数 |
| `cache_type` | `str` | `"radix"` | `SchedulerConfig` | KV 缓存管理策略 |
| `offline_mode` | `bool` | `False` | `SchedulerConfig` | 离线批量推理模式 |
| `server_host` | `str` | `"127.0.0.1"` | `ServerArgs` | HTTP 服务绑定地址 |
| `server_port` | `int` | 1919 | `ServerArgs` | HTTP 服务端口 |
| `num_tokenizer` | `int` | 0 | `ServerArgs` | 独立 tokenizer 进程数量 |
| `silent_output` | `bool` | `False` | `ServerArgs` | Shell 模式下静默 INFO 日志 |

#### 属性说明

**`share_tokenizer`**（只读属性）

```python
# python/minisgl/server/args.py  行 21-23
@property
def share_tokenizer(self) -> bool:
    return self.num_tokenizer == 0
```

- **返回值**：`bool`
- **语义**：当 `num_tokenizer == 0` 时，tokenizer 和 detokenizer 合并为同一进程（共享模式）；否则为独立分离模式。
- **作用**：决定进程拓扑和 ZMQ 地址路由策略。

**`zmq_frontend_addr`**（只读属性）

```python
# python/minisgl/server/args.py  行 26-27
@property
def zmq_frontend_addr(self) -> str:
    return "ipc:///tmp/minisgl_3" + self._unique_suffix
```

- **返回值**：`str`，形如 `"ipc:///tmp/minisgl_3.pid=12345"`
- **语义**：API 前端（FastAPI）用于接收 detokenizer 回包的 ZMQ IPC 地址（PULL socket）。
- **`_unique_suffix`**：继承自 `SchedulerConfig`，以当前进程 PID 为后缀，保证多实例并发运行时地址不冲突。

**`zmq_tokenizer_addr`**（只读属性）

```python
# python/minisgl/server/args.py  行 29-35
@property
def zmq_tokenizer_addr(self) -> str:
    if self.share_tokenizer:
        return self.zmq_detokenizer_addr
    result = "ipc:///tmp/minisgl_4" + self._unique_suffix
    assert result != self.zmq_detokenizer_addr
    return result
```

- **返回值**：`str`
- **语义**：API 前端用于向 tokenizer 发送请求的 ZMQ IPC 地址（PUSH socket）。
  - 共享模式下与 detokenizer 地址（`minisgl_1`）相同；
  - 分离模式下使用独立的 `minisgl_4` 地址，并通过断言确认二者不冲突。

**`tokenizer_create_addr`**（只读属性）

```python
# python/minisgl/server/args.py  行 37-39
@property
def tokenizer_create_addr(self) -> bool:
    return self.share_tokenizer
```

- **返回值**：`bool`
- **语义**：决定 tokenizer/detokenizer 进程是否需要以 `bind`（创建）模式初始化其接收 socket。共享模式下，合并进程自己 bind；分离模式下由其他端 bind。

**`backend_create_detokenizer_link`**（只读属性）

```python
# python/minisgl/server/args.py  行 41-43
@property
def backend_create_detokenizer_link(self) -> bool:
    return not self.share_tokenizer
```

- **返回值**：`bool`
- **语义**：决定 backend（Scheduler）是否主动 bind detokenizer 链路。分离模式下 backend bind，共享模式下 tokenizer 自己 bind。

**`frontend_create_tokenizer_link`**（只读属性）

```python
# python/minisgl/server/args.py  行 45-47
@property
def frontend_create_tokenizer_link(self) -> bool:
    return not self.share_tokenizer
```

- **返回值**：`bool`
- **语义**：决定 API 前端是否主动 bind 向 tokenizer 发送消息的链路。

**`distributed_addr`**（只读属性）

```python
# python/minisgl/server/args.py  行 49-51
@property
def distributed_addr(self) -> str:
    return f"tcp://127.0.0.1:{self.server_port + 1}"
```

- **返回值**：`str`，形如 `"tcp://127.0.0.1:1920"`
- **语义**：覆盖父类 `EngineConfig` 中固定的 2333 端口，改为动态使用 `server_port + 1`，从而支持多服务器实例并行运行。

### 2.3 函数：`parse_args`

#### 签名

```python
def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
```

#### 参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `args` | `List[str]` | 命令行参数列表，通常为 `sys.argv[1:]` |
| `run_shell` | `bool` | 外部传入的 shell 模式标志，与 `--shell-mode` 参数逻辑或合并 |

#### 返回值

`Tuple[ServerArgs, bool]`：
- `ServerArgs`：已完成所有字段解析和转换的冻结配置对象。
- `bool`：最终的 `run_shell` 标志（命令行 `--shell-mode` 与入参 `run_shell` 的逻辑或）。

#### 支持的命令行参数

| CLI 参数 | 目标字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|---|
| `--model-path`, `--model` | `model_path` | `str` | 必填 | 模型路径 |
| `--dtype` | `dtype` | `str` | `"auto"` | 计算精度 |
| `--tensor-parallel-size`, `--tp-size` | `tp_info` | `int` | 1 | TP 并行度 |
| `--max-running-requests` | `max_running_req` | `int` | 256 | 最大并发请求 |
| `--max-seq-len-override` | `max_seq_len_override` | `int` | `None` | 最大序列长度覆盖 |
| `--memory-ratio` | `memory_ratio` | `float` | 0.9 | GPU 内存占用比例 |
| `--dummy-weight` | `use_dummy_weight` | `bool` | `False` | 虚假权重测试 |
| `--disable-pynccl` | `use_pynccl` | `bool` | `True` | 禁用 PyNCCL |
| `--host` | `server_host` | `str` | `"127.0.0.1"` | 服务监听地址 |
| `--port` | `server_port` | `int` | 1919 | 服务监听端口 |
| `--cuda-graph-max-bs`, `--graph` | `cuda_graph_max_bs` | `int` | `None` | CUDA Graph 最大批大小 |
| `--num-tokenizer` | `num_tokenizer` | `int` | 0 | tokenizer 进程数 |
| `--max-prefill-length`, `--max-extend-length` | `max_extend_tokens` | `int` | 8192 | Prefill 最大 token 数 |
| `--num-pages` | `num_page_override` | `int` | `None` | KV 缓存页数覆盖 |
| `--page-size` | `page_size` | `int` | 1 | KV 缓存页大小 |
| `--attention-backend`, `--attn` | `attention_backend` | `str` | `"auto"` | 注意力后端 |
| `--model-source` | — | `str` | `"huggingface"` | 模型下载来源（临时，不入 config） |
| `--cache-type` | `cache_type` | `str` | `"radix"` | KV 缓存管理策略 |
| `--moe-backend` | `moe_backend` | `str` | `"auto"` | MoE 后端 |
| `--shell-mode` | `run_shell` | `bool` | `False` | 交互式 Shell 模式 |

#### 关键实现细节

**Shell 模式自动调优**（第 231-234 行）：
```python
# python/minisgl/server/args.py  行 230-234
run_shell |= kwargs.pop("shell_mode")
if run_shell:
    kwargs["cuda_graph_max_bs"] = 1
    kwargs["max_running_req"] = 1
    kwargs["silent_output"] = True
```
Shell 模式下强制设置 `cuda_graph_max_bs=1`、`max_running_req=1`，避免 CUDA Graph 为大批大小预热浪费内存，同时静默 INFO 日志，保持终端交互的整洁。

**波浪号路径展开**（第 236-237 行）：
```python
# python/minisgl/server/args.py  行 236-237
if kwargs["model_path"].startswith("~"):
    kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])
```

**ModelScope 自动下载**（第 239-249 行）：
当 `--model-source modelscope` 且路径不是本地目录时，调用 `modelscope.snapshot_download` 下载模型，并在虚假权重模式下跳过实际权重文件（`*.bin`, `*.safetensors` 等）。`model_source` 字段在处理完成后从 kwargs 中删除，不写入 `ServerArgs`。

**dtype 解析**（第 251-261 行）：
```python
# python/minisgl/server/args.py  行 251-261
if (dtype_str := kwargs["dtype"]) == "auto":
    from minisgl.utils import cached_load_hf_config
    dtype_str = cached_load_hf_config(kwargs["model_path"]).dtype
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str
```
`"auto"` 模式会从 HuggingFace config.json 读取模型原生精度，实现精度自动推断。

**DistributedInfo 构建**（第 262-263 行）：
```python
# python/minisgl/server/args.py  行 262-263
kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
del kwargs["tensor_parallel_size"]
```
将整数 `tensor_parallel_size` 转换为 `DistributedInfo(rank=0, size=N)` 对象。此处 rank 始终为 0，因为主进程是 rank 0；其余进程的 `tp_info` 在 `launch.py` 中通过 `replace()` 重新赋值。

### 2.4 依赖关系

```
parse_args
├── minisgl.attention.validate_attn_backend   (验证 --attention-backend 参数)
├── minisgl.kvcache.SUPPORTED_CACHE_MANAGER   (获取合法 --cache-type 列表)
├── minisgl.moe.SUPPORTED_MOE_BACKENDS        (获取合法 --moe-backend 列表)
├── minisgl.distributed.DistributedInfo       (构建 tp_info)
├── minisgl.utils.cached_load_hf_config       (auto dtype 推断)
└── modelscope.snapshot_download              (可选，ModelScope 下载)

ServerArgs
└── SchedulerConfig
    └── EngineConfig
        └── minisgl.distributed.DistributedInfo
```

---

## 3. `python/minisgl/server/launch.py`

### 3.1 文件职责

负责系统的多进程启动编排：解析参数，为每个 TP rank 创建 Scheduler 子进程，创建 tokenizer/detokenizer 子进程，等待所有进程就绪后启动 API 服务器。

### 3.2 函数：`_run_scheduler`

#### 签名

```python
def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
```

#### 参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `args` | `ServerArgs` | 服务配置，每个 TP rank 传入不同的 `tp_info`（rank 不同） |
| `ack_queue` | `mp.Queue[str]` | 用于向主进程发送就绪信号的多进程队列 |

#### 返回值

无（`None`）；函数运行直到 `KeyboardInterrupt`。

#### 关键实现细节

```python
# python/minisgl/server/launch.py  行 16-37
def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            ...
            scheduler.shutdown()
```

- **`torch.inference_mode()`**：在整个调度循环外层启用推理模式，禁用自动梯度计算以节省内存和提升性能。
- **`scheduler.sync_all_ranks()`**：通过 CPU 进程组 barrier 确保所有 TP ranks 在发出就绪信号前完成初始化（包括权重加载、CUDA Graph 捕获等耗时操作）。
- **就绪信号**：只有 primary rank（rank 0）向队列发送 ack，其余 ranks 不发送，避免重复计数。
- **日志静默**：Shell 模式下禁用 INFO 级别日志，防止后台日志污染交互终端界面。
- **优雅退出**：捕获 `KeyboardInterrupt`，主动调用 `scheduler.shutdown()` 同步所有 rank 并清理 GPU 资源。

### 3.3 函数：`launch_server`

#### 签名

```python
def launch_server(run_shell: bool = False) -> None:
```

#### 参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `run_shell` | `bool` | Shell 模式标志，默认为 False（从 `shell.py` 调用时传 True） |

#### 返回值

无（`None`）；函数阻塞直到服务器退出。

#### 关键实现细节

函数内定义了一个闭包 `start_subprocess()`，该闭包捕获 `server_args`，并在 `run_api_server` 启动后被调用。这种设计将 **ZMQ socket 创建（前端）** 与 **子进程启动（后端）** 的时序解耦——确保前端 socket bind 在先，子进程 connect 在后，避免连接竞态。

**进程数量计算**（第 55-103 行）：

```python
# python/minisgl/server/launch.py  行 54-113

# 为每个 TP rank 创建独立的 Scheduler 进程
world_size = server_args.tp_info.size
for i in range(world_size):
    new_args = replace(server_args, tp_info=DistributedInfo(i, world_size))
    mp.Process(target=_run_scheduler, args=(new_args, ack_queue), ...).start()

# 始终创建 1 个 detokenizer 进程
mp.Process(target=tokenize_worker, kwargs={..., "tokenizer_id": num_tokenizers}, ...).start()

# 按需创建 num_tokenizer 个独立 tokenizer 进程
for i in range(num_tokenizers):
    mp.Process(target=tokenize_worker, kwargs={..., "tokenizer_id": i}, ...).start()
```

启动进程汇总：

| 进程名 | 数量 | 说明 |
|---|---|---|
| `minisgl-TP{i}-scheduler` | `world_size` 个 | 每个 TP rank 一个推理调度进程 |
| `minisgl-detokenizer-0` | 1 个 | 负责 token → 文本的 detokenize |
| `minisgl-tokenizer-{i}` | `num_tokenizer` 个 | 负责 文本 → token 的 tokenize（0 个时与 detokenizer 合并） |

**就绪等待逻辑**（第 105-111 行）：

```python
# python/minisgl/server/launch.py  行 109-111
# Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
for _ in range(num_tokenizers + 2):
    logger.info(ack_queue.get())
```

等待 `num_tokenizers + 2` 条 ack 消息：
- 1 条来自 primary scheduler（其余 ranks 不发）
- 1 条来自 detokenizer
- `num_tokenizers` 条来自各 tokenizer 进程

**多进程启动方式**（第 52 行）：
```python
mp.set_start_method("spawn", force=True)
```
强制使用 `spawn` 启动方式（而非 `fork`）。原因：`fork` 在持有 CUDA 上下文的进程中不安全，`spawn` 创建全新子进程，避免 GPU 状态污染。

**`replace()` 函数用途**：来自 `dataclasses.replace`，用于在冻结数据类（`frozen=True`）中创建字段不同的副本，此处为每个 rank 分配正确的 `tp_info`。

### 3.4 依赖关系

```
launch_server
├── parse_args                        (参数解析)
├── run_api_server                    (HTTP 服务器或 shell)
├── _run_scheduler                    (子进程入口)
│   └── minisgl.scheduler.Scheduler
└── minisgl.tokenizer.tokenize_worker (子进程入口)
```

---

## 4. `python/minisgl/server/api_server.py`

### 4.1 文件职责

实现基于 FastAPI + uvicorn 的 HTTP API 服务层，作为用户请求与后端推理系统之间的异步消息桥梁，通过 ZMQ 与 tokenizer 进程双向通信。

### 4.2 模块级变量与初始化

```python
# python/minisgl/server/api_server.py  行 32-34
logger = init_logger(__name__, "FrontendAPI")
_GLOBAL_STATE = None
```

`_GLOBAL_STATE` 是全局单例，在 `run_api_server()` 中初始化为 `FrontendManager` 实例，在 FastAPI lifespan 关闭时销毁。

### 4.3 函数：`get_global_state`

#### 签名

```python
def get_global_state() -> FrontendManager:
```

#### 说明

获取全局 `FrontendManager` 单例，若未初始化则抛出断言错误。所有请求处理函数通过此函数访问共享状态。

### 4.4 函数：`_unwrap_msg`

#### 签名

```python
def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
```

#### 参数与返回值

- `msg`：来自 detokenizer 的消息，可能是单个 `UserReply` 或批量 `BatchFrontendMsg`。
- 返回值：统一展开为 `List[UserReply]`。

#### 说明

消息拆包工具函数，将批量消息格式统一化为列表格式，供 `FrontendManager.listen()` 使用。

### 4.5 Pydantic 模型

#### `GenerateRequest`

```python
class GenerateRequest(BaseModel):
    prompt: str          # 输入提示文本
    max_tokens: int      # 最大生成 token 数
    ignore_eos: bool = False  # 是否忽略 EOS token
```

用于 `/generate` 端点的请求体，是简化版非 OpenAI 兼容接口。

#### `Message`

```python
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
```

OpenAI 兼容的对话消息结构，`role` 限定为三种合法角色。

#### `OpenAICompletionRequest`

```python
class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: str | None = None
    messages: List[Message] | None = None
    max_tokens: int = 16
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ignore_eos: bool = False
```

统一的 OpenAI 兼容请求模型，同时支持 `POST /v1/completions`（传 `prompt`）和 `POST /v1/chat/completions`（传 `messages`）。注意：`n`、`stop`、`presence_penalty`、`frequency_penalty` 字段当前为占位符，尚未在后端实现。

#### `ModelCard` / `ModelList`

```python
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)
```

用于 `GET /v1/models` 响应的 OpenAI 兼容模型列表格式。

### 4.6 类：`FrontendManager`

核心状态管理类，维护所有活跃用户请求的生命周期，通过事件驱动模型实现流式响应。

#### 签名

```python
@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
```

#### 字段说明

| 字段 | 说明 |
|---|---|
| `config` | 服务配置，用于获取模型路径等信息 |
| `send_tokenizer` | 向 tokenizer 发送请求的异步 ZMQ PUSH queue |
| `recv_tokenizer` | 接收 detokenizer 回包的异步 ZMQ PULL queue |
| `uid_counter` | 单调递增的用户请求 ID 计数器 |
| `initialized` | 是否已启动后台 listen 任务的标志位 |
| `ack_map` | `uid → 已接收但未消费的 UserReply 列表` 映射 |
| `event_map` | `uid → asyncio.Event` 映射，用于通知等待中的请求 |

#### 方法详解

**`new_user() -> int`**

```python
# python/minisgl/server/api_server.py  行 110-115
def new_user(self) -> int:
    uid = self.uid_counter
    self.uid_counter += 1
    self.ack_map[uid] = []
    self.event_map[uid] = asyncio.Event()
    return uid
```

分配新的请求 ID，并在 `ack_map` 和 `event_map` 中注册该请求。每次请求到达时调用，返回全局唯一的整数 uid。

**`listen() -> None`（异步协程）**

```python
# python/minisgl/server/api_server.py  行 117-124
async def listen(self):
    while True:
        msg = await self.recv_tokenizer.get()
        for msg in _unwrap_msg(msg):
            if msg.uid not in self.ack_map:
                continue
            self.ack_map[msg.uid].append(msg)
            self.event_map[msg.uid].set()
```

全局后台监听协程，持续从 ZMQ PULL socket 接收 detokenizer 推送的 `UserReply` 消息，按 uid 分发到对应的 `ack_map` 条目，并通过 `asyncio.Event.set()` 唤醒等待该 uid 的协程。

**`_create_listener_once() -> None`**

通过 `initialized` 标志确保 `listen()` 协程只被创建一次（懒初始化），在第一次调用 `send_one()` 时触发。

**`send_one(msg: BaseTokenizerMsg) -> None`（异步协程）**

启动 listener（若未启动），然后向 tokenizer 进程发送消息。

**`wait_for_ack(uid: int)` （异步生成器）**

```python
# python/minisgl/server/api_server.py  行 135-151
async def wait_for_ack(self, uid: int):
    event = self.event_map[uid]
    while True:
        await event.wait()
        event.clear()
        pending = self.ack_map[uid]
        self.ack_map[uid] = []
        ack = None
        for ack in pending:
            yield ack
        if ack and ack.finished:
            break
    del self.ack_map[uid]
    del self.event_map[uid]
```

核心的流式读取生成器。使用 `asyncio.Event` 实现无忙等待的异步等待：
1. 等待事件触发（`await event.wait()`）
2. 原子性地取出并清空当前积累的所有 ack
3. 逐条 yield 给调用方
4. 若最后一条 ack 的 `finished=True`，退出循环
5. 清理 uid 相关的所有状态

这种设计支持批量接收（单次唤醒处理多条 ack）和逐条流式输出的统一处理。

**`stream_generate(uid: int)` （异步生成器）**

```python
# python/minisgl/server/api_server.py  行 153-160
async def stream_generate(self, uid: int):
    async for ack in self.wait_for_ack(uid):
        chunk = {"text": ack.incremental_output}
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
        if ack.finished:
            break
    yield b"data: [DONE]\n\n"
```

生成 Server-Sent Events（SSE）格式的流式响应。每个增量输出被封装为 `{"text": "..."}` JSON，以 `data: ...\n\n` 格式发送，最终以 `data: [DONE]\n\n` 结束。用于 `/generate` 和 shell 模式。

**`stream_chat_completions(uid: int)` （异步生成器）**

生成 OpenAI Chat Completions 兼容的 SSE 格式流式响应，格式如下：
```json
{
  "id": "cmpl-{uid}",
  "object": "text_completion.chunk",
  "choices": [{"delta": {"role": "assistant", "content": "..."}, "index": 0, "finish_reason": null}]
}
```
第一个 chunk 包含 `"role": "assistant"`，后续 chunk 仅包含 `"content"`，最后发送 `finish_reason: "stop"` 的空 delta chunk，然后发送 `data: [DONE]`。

**`collect_full_output(uid: int) -> str`（异步方法）**

```python
# python/minisgl/server/api_server.py  行 192-199
async def collect_full_output(self, uid: int):
    full_text = ""
    async for ack in self.wait_for_ack(uid):
        full_text += ack.incremental_output
        if ack.finished:
            break
    return full_text
```

非流式模式下，收集所有增量输出并拼接为完整字符串返回。

**`stream_with_cancellation(generator, request, uid)` （异步生成器）**

包装任意流式生成器，在每个 chunk 发送后检测客户端是否已断开（`await request.is_disconnected()`）。若断开则触发 `asyncio.CancelledError`，并通过 `asyncio.create_task(self.abort_user(uid))` 异步发送终止消息给后端。

**`abort_user(uid: int) -> None`（异步方法）**

```python
# python/minisgl/server/api_server.py  行 213-220
async def abort_user(self, uid: int):
    await asyncio.sleep(0.1)
    if uid in self.ack_map:
        del self.ack_map[uid]
    if uid in self.event_map:
        del self.event_map[uid]
    logger.warning("Aborting request for user %s", uid)
    await self.send_one(AbortMsg(uid=uid))
```

延迟 0.1 秒（等待流式生成器确认取消），清理本地状态，然后向 tokenizer 发送 `AbortMsg`，最终通过 tokenizer 转发给 scheduler 中止推理。

**`shutdown() -> None`**

关闭 ZMQ send 和 recv socket，释放 ZMQ context。

### 4.7 FastAPI 生命周期管理

```python
# python/minisgl/server/api_server.py  行 227-233
@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()
```

使用 FastAPI 的 `lifespan` context manager，在服务器正常关闭时调用 `FrontendManager.shutdown()` 清理 ZMQ 资源。

### 4.8 FastAPI 路由端点

#### `POST /generate`

```python
@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
```

- **说明**：非 OpenAI 兼容的简化生成接口，仅支持字符串 prompt，非流式响应。
- **处理流程**：分配 uid → 发送 `TokenizeMsg` → 等待完整输出 → 返回 `{"text": "..."}`。

#### `GET/POST /v1`

```python
@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}
```

健康检查端点，支持所有 HTTP 方法，返回固定的状态响应。

#### `POST /v1/chat/completions`

```python
@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
```

- **说明**：OpenAI 兼容的 Chat Completions 接口，同时支持 `messages`（对话格式）和 `prompt`（纯文本）输入。
- **流式模式**（`stream=True`）：返回 `StreamingResponse`，包含断连检测。
- **非流式模式**（`stream=False`）：收集完整输出，返回 OpenAI 格式 JSON：
  ```json
  {
    "id": "cmpl-{uid}",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
  }
  ```
  注：`usage` 字段当前填充为 0，尚未实现 token 计数。

#### `GET /v1/models`

```python
@app.get("/v1/models")
async def available_models():
```

返回 OpenAI 兼容的模型列表，当前仅包含运行中的单一模型，`id` 和 `root` 均为 `config.model_path`。

### 4.9 函数：`shell_completion`（内部）

#### 签名

```python
async def shell_completion(req: OpenAICompletionRequest) -> StreamingResponse:
```

Shell 模式专用的补全函数，不通过 HTTP 路由，而是在同一进程内的 asyncio 循环中直接调用。只接受 `messages` 格式输入，返回使用 `stream_generate`（而非 `stream_chat_completions`）格式的 SSE 流，用于在 `shell()` 函数中逐字符打印。

### 4.10 函数：`async_input`

#### 签名

```python
async def async_input(prompt="") -> str:
```

使用 `loop.run_in_executor(None, ...)` 将阻塞式 `input()` 包装为异步非阻塞调用，供 Shell 模式使用（实际 Shell 模式改用 `prompt_toolkit`，此函数为备用实现）。

### 4.11 函数：`shell`（异步协程）

#### 签名

```python
async def shell() -> None:
```

#### 说明

交互式 Shell 的主循环协程，完整的对话功能实现（详见第 5 节）。

### 4.12 函数：`run_api_server`

#### 签名

```python
def run_api_server(
    config: ServerArgs,
    start_backend: Callable[[], None],
    run_shell: bool
) -> None:
```

#### 参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `config` | `ServerArgs` | 服务配置 |
| `start_backend` | `Callable[[], None]` | 启动后端子进程的回调函数 |
| `run_shell` | `bool` | 是否以 Shell 模式运行 |

#### 关键实现细节

```python
# python/minisgl/server/api_server.py  行 433-474
def run_api_server(config, start_backend, run_shell):
    global _GLOBAL_STATE
    # 1. 创建 ZMQ socket（必须在子进程启动前 bind）
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr, create=True, decoder=BaseFrontendMsg.decoder
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )
    # 2. 启动后端子进程（此时前端 socket 已就绪）
    start_backend()
    # 3. 启动 HTTP 服务器或 Shell
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
```

**Socket 创建时序**：先 bind/create ZMQ socket，再调用 `start_backend()` 启动子进程，确保子进程 connect 时前端已就绪。

**双模式入口**：
- HTTP 模式：`uvicorn.run(app, ...)` 启动事件循环并阻塞。
- Shell 模式：`asyncio.run(shell())` 运行交互式终端。

Shell 模式断言：`assert not config.use_dummy_weight`，禁止在虚假权重测试模式下使用 Shell。

### 4.13 依赖关系

```
api_server.py
├── fastapi, uvicorn, starlette    (HTTP 框架)
├── prompt_toolkit                  (Shell 交互界面)
├── minisgl.core.SamplingParams     (采样参数)
├── minisgl.env.ENV                 (Shell 模式默认参数)
├── minisgl.message.*               (消息类型)
│   ├── TokenizeMsg                 (发送给 tokenizer)
│   ├── AbortMsg                    (取消请求)
│   ├── BaseFrontendMsg             (接收自 detokenizer)
│   └── UserReply                   (增量输出回包)
├── minisgl.utils.ZmqAsyncPushQueue (发送 ZMQ 消息)
└── minisgl.utils.ZmqAsyncPullQueue (接收 ZMQ 消息)
```

---

## 5. `python/minisgl/shell.py`

### 5.1 文件职责

作为交互式 Shell 的独立 Python 包入口（`python -m minisgl.shell`），以 Shell 模式启动服务，供用户直接在终端与模型对话。

### 5.2 代码全文（含行号）

```python
# python/minisgl/shell.py
1: from .server import launch_server
2:
3: if __name__ == "__main__":
4:     launch_server(run_shell=True)
```

### 5.3 关键实现细节

- 与 `__main__.py` 的区别：传入 `run_shell=True`，触发 Shell 模式路径。
- **执行方式**：`python -m minisgl.shell --model /path/to/model`
- 使用 `if __name__ == "__main__"` 而非 `assert`，允许 `shell.py` 被其他模块 import（不会触发启动）。

### 5.4 Shell 主循环（位于 `api_server.py`）

Shell 的实际交互逻辑实现在 `api_server.py` 的 `shell()` 协程中（第 364-430 行），此处进行完整说明：

#### 功能特性

- 使用 `prompt_toolkit.PromptSession` 实现带命令自动补全的交互式输入，补全词典为 `["/exit", "/reset"]`。
- 维护多轮对话历史 `history: List[Tuple[str, str]]`（用户消息, 模型回复）。
- 每轮对话将历史消息拼接为 OpenAI messages 格式后发送，实现上下文保持。

#### 支持的命令

| 命令 | 说明 |
|---|---|
| `/exit` | 退出 Shell，触发资源清理 |
| `/reset` | 清空对话历史，开始新对话 |
| `Ctrl-D` | 触发 `EOFError`，等价于 `/exit` |

#### 流式输出渲染

```python
# python/minisgl/server/api_server.py  行 397-416
async for chunk in (await shell_completion(req)).body_iterator:
    msg = chunk.decode()
    for line in msg.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
            text = parsed.get("text", "")
        except json.JSONDecodeError:
            text = data
        cur_msg += text
        print(text, end="", flush=True)
```

逐行解析 SSE 格式数据，提取 `text` 字段并实时打印，实现逐字符流式显示效果。

#### 退出清理

```python
# python/minisgl/server/api_server.py  行 418-430
finally:
    print("Exiting shell...")
    await asyncio.sleep(0.1)
    get_global_state().shutdown()
    import psutil
    parent = psutil.Process()
    for child in parent.children(recursive=True):
        child.kill()
```

Shell 退出时：
1. 等待 0.1 秒确保正在进行的请求完成
2. 关闭 ZMQ socket
3. 使用 `psutil` 强制终止所有子进程（scheduler、tokenizer 等）

这种强制终止方式是必要的，因为子进程处于 `run_forever()` 无限循环中，无法通过正常的进程等待退出。

#### 采样参数来源

Shell 的默认采样参数来自环境变量（通过 `ENV` 单例）：
```python
max_tokens=ENV.SHELL_MAX_TOKENS.value    # 默认 2048
top_k=ENV.SHELL_TOP_K.value              # 默认 -1（禁用）
top_p=ENV.SHELL_TOP_P.value              # 默认 1.0
temperature=ENV.SHELL_TEMPERATURE.value  # 默认 0.6
```

### 5.5 依赖关系

```
shell.py
└── minisgl.server.launch_server  (传入 run_shell=True)

shell() 协程（在 api_server.py 中）
├── prompt_toolkit.PromptSession   (交互式输入)
├── prompt_toolkit.WordCompleter   (命令补全)
├── shell_completion()             (内部补全函数)
├── minisgl.env.ENV                (默认采样参数)
└── psutil                         (子进程管理)
```

---

## 6. `python/minisgl/env.py`

### 6.1 文件职责

提供全局环境变量配置系统，以单例模式集中管理所有 `MINISGL_*` 环境变量，并支持类型安全的自动解析。

### 6.2 类：`BaseEnv`

```python
class BaseEnv:
    def _init(self, name: str) -> None:
        raise NotImplementedError
```

所有环境变量类的抽象基类，定义了 `_init(name)` 接口用于从系统环境中读取并初始化值。

### 6.3 类：`EnvVar[T]`

#### 签名

```python
class EnvVar(BaseEnv, Generic[T]):
    def __init__(self, default_value: T, fn: Callable[[str], T]):
```

#### 参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `default_value` | `T` | 未设置环境变量时的默认值 |
| `fn` | `Callable[[str], T]` | 将字符串环境变量值转换为目标类型的解析函数 |

#### 关键实现细节

**`_init(name: str) -> None`**

```python
# python/minisgl/env.py  行 22-28
def _init(self, name: str) -> None:
    env_value = os.getenv(name)
    if env_value is not None:
        try:
            self.value = self.fn(env_value)
        except Exception:
            pass
```

从系统环境中读取名为 `name` 的变量，若存在则用 `fn` 解析；解析失败时静默忽略，保留默认值。这种设计保证了配置的健壮性：非法的环境变量值不会导致服务崩溃。

**`__bool__` / `__str__`**：使 `EnvVar` 可以直接用于布尔判断和字符串格式化，访问的是 `.value` 字段。

### 6.4 辅助解析函数

**`_TO_BOOL`**（第 37 行）：

```python
_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")
```

布尔值解析器，接受 `"1"`、`"true"`、`"yes"`（大小写不敏感）为真，其余为假。

**`_PARSE_MEM_BYTES`**（第 40-47 行）：

```python
# python/minisgl/env.py  行 40-47
def _PARSE_MEM_BYTES(mem: str) -> int:
    mem = mem.strip().upper()
    if not mem[-1].isalpha():
        return int(mem)
    if mem.endswith("B"):
        mem = mem[:-1]
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])
```

内存大小解析器，支持以下格式：
- 纯数字：直接解析为字节数（如 `"1073741824"`）
- 带单位后缀（大小写不敏感）：`K`/`KB`、`M`/`MB`、`G`/`GB`（如 `"1G"`、`"512MB"`）
- 支持浮点数（如 `"1.5G"` → 1610612736 字节）

### 6.5 便捷工厂函数（第 51-55 行）

```python
# python/minisgl/env.py  行 51-55
EnvInt    = partial(EnvVar[int],        fn=int)
EnvFloat  = partial(EnvVar[float],      fn=float)
EnvBool   = partial(EnvVar[bool],       fn=_TO_BOOL)
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem    = partial(EnvVar[int],        fn=_PARSE_MEM_BYTES)
```

使用 `functools.partial` 预绑定 `fn` 参数，简化类型化环境变量的声明。`EnvOption` 与 `EnvBool` 的区别在于默认值为 `None`，用于三态逻辑（未设置 / 显式 False / 显式 True）。

### 6.6 类：`EnvClassSingleton`

#### 签名

```python
class EnvClassSingleton:
    _instance: EnvClassSingleton | None = None
    # ... 环境变量声明 ...
```

#### 单例模式实现

```python
# python/minisgl/env.py  行 73-77
def __new__(cls):
    if cls._instance is None:
        cls._instance = super().__new__(cls)
    return cls._instance
```

经典的 Python 单例模式，确保全进程只有一个 `EnvClassSingleton` 实例，所有模块共享同一份环境变量配置。

#### 初始化逻辑

```python
# python/minisgl/env.py  行 79-85
def __init__(self) -> None:
    for attr_name in dir(self):
        if attr_name.startswith("_"):
            continue
        attr_value = getattr(self, attr_name)
        assert isinstance(attr_value, BaseEnv)
        attr_value._init(f"{MINISGL_ENV_PREFIX}{attr_name}")
```

通过反射遍历所有非私有属性，自动为每个 `BaseEnv` 实例调用 `_init()`，环境变量名为 `MINISGL_` 前缀加属性名（全大写），如 `MINISGL_SHELL_MAX_TOKENS`。

**注意**：断言 `assert isinstance(attr_value, BaseEnv)` 要求类中所有公开属性都必须是 `BaseEnv` 子类，这意味着不能在类中添加普通方法或非环境变量属性（否则会触发断言错误）。

#### 全部环境变量

| 属性名 | 环境变量 | 类型 | 默认值 | 说明 |
|---|---|---|---|---|
| `SHELL_MAX_TOKENS` | `MINISGL_SHELL_MAX_TOKENS` | `int` | 2048 | Shell 模式最大生成 token 数 |
| `SHELL_TOP_K` | `MINISGL_SHELL_TOP_K` | `int` | -1 | Shell 模式 top-k 采样值（-1 禁用） |
| `SHELL_TOP_P` | `MINISGL_SHELL_TOP_P` | `float` | 1.0 | Shell 模式 top-p 核采样阈值 |
| `SHELL_TEMPERATURE` | `MINISGL_SHELL_TEMPERATURE` | `float` | 0.6 | Shell 模式采样温度 |
| `FLASHINFER_USE_TENSOR_CORES` | `MINISGL_FLASHINFER_USE_TENSOR_CORES` | `bool \| None` | `None` | 是否在 FlashInfer 中使用 Tensor Cores |
| `DISABLE_OVERLAP_SCHEDULING` | `MINISGL_DISABLE_OVERLAP_SCHEDULING` | `bool` | `False` | 是否禁用重叠调度（调试用） |
| `OVERLAP_EXTRA_SYNC` | `MINISGL_OVERLAP_EXTRA_SYNC` | `bool` | `False` | 重叠调度中额外的 CUDA stream 同步 |
| `PYNCCL_MAX_BUFFER_SIZE` | `MINISGL_PYNCCL_MAX_BUFFER_SIZE` | `int` | 1 GiB | PyNCCL 通信缓冲区最大字节数 |

#### 全局单例

```python
# python/minisgl/env.py  行 88
ENV = EnvClassSingleton()
```

在模块导入时自动创建并初始化，所有模块通过 `from minisgl.env import ENV` 访问。

### 6.7 依赖关系

```
env.py
├── os         (os.getenv)
├── functools  (partial)
└── typing     (Generic, TypeVar, Callable)
```

被依赖关系：
```
ENV 被以下模块使用：
├── api_server.py  (SHELL_* 采样参数)
└── scheduler.py   (DISABLE_OVERLAP_SCHEDULING, OVERLAP_EXTRA_SYNC)
```

---

## 7. `python/minisgl/llm/llm.py`

### 7.1 文件职责

提供高层离线批量推理接口 `LLM`，继承自 `Scheduler`，通过重写消息收发接口将在线服务模式改造为同步批量推理模式，屏蔽底层 ZMQ 通信细节。

### 7.2 类：`RequestAllFinished`

```python
class RequestAllFinished(Exception):
    pass
```

内部哨兵异常，当所有请求均已完成且无新请求时从 `offline_receive_msg` 中抛出，用于终止 `run_forever()` 的无限循环。这是一种通过异常控制流实现"正常退出"的设计模式。

### 7.3 类：`RequestStatus`

```python
@dataclass
class RequestStatus:
    uid: int
    input_ids: List[int]
    output_ids: List[int]
```

记录单个请求的推理状态：
- `uid`：请求唯一标识
- `input_ids`：输入 token 列表（用于记录，便于调试）
- `output_ids`：已生成的输出 token 列表（追加写入）

### 7.4 类：`LLM`

#### 继承关系

```
LLM → Scheduler → SchedulerIOMixin + EngineConfig
```

`LLM` 继承 `Scheduler` 的全部推理调度能力，仅通过重写 `offline_receive_msg` 和 `offline_send_result` 两个接口方法改变数据输入/输出路径。

#### `__init__`

```python
# python/minisgl/llm/llm.py  行 29-39
def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
    config = SchedulerConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=dtype,
        offline_mode=True,
        **kwargs,
    )
    super().__init__(config)
    self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []
    self.status_map: Dict[int, RequestStatus] = {}
    self.counter = 0
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model_path` | `str` | 必填 | 模型权重路径 |
| `dtype` | `torch.dtype` | `torch.bfloat16` | 计算精度 |
| `**kwargs` | `Any` | — | 传递给 `SchedulerConfig` 的其他参数（如 `memory_ratio`、`page_size`等） |

**关键设计决策**：
- `tp_info=DistributedInfo(0, 1)`：强制单卡推理（TP size = 1），不支持张量并行。
- `offline_mode=True`：触发 `SchedulerIOMixin.__init__` 中的特殊路径，将 `receive_msg` 和 `send_result` 重定向为 `offline_receive_msg` 和 `offline_send_result`，跳过 ZMQ socket 初始化。

**实例变量**：
- `pending_requests`：待处理请求队列，每条为 `(prompt, sampling_params)` 二元组
- `status_map`：`uid → RequestStatus` 映射，跟踪每个请求的输出状态
- `counter`：全局 uid 计数器

#### `_tokenize_one`

```python
# python/minisgl/llm/llm.py  行 42-46
def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
    if isinstance(prompt, str):
        return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
    else:
        return torch.tensor(prompt, dtype=torch.int32, device="cpu")
```

#### 参数与返回值

- `prompt`：字符串或已编码的 token ID 列表
- 返回值：`torch.Tensor`，dtype 为 `torch.int32`，位于 CPU

统一将字符串和 token 列表转换为 int32 CPU tensor，作为推理引擎的输入格式。

#### `offline_receive_msg`

```python
# python/minisgl/llm/llm.py  行 48-69
def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
    if blocking and len(self.pending_requests) == 0:
        raise RequestAllFinished()
    results: List[BaseBackendMsg] = []
    added, sum_input_len = 0, 0
    for tokens_or_prompt, sampling_params in self.pending_requests:
        if sum_input_len >= self.prefill_budget:
            break
        input_ids = self._tokenize_one(tokens_or_prompt)
        sum_input_len += len(input_ids)
        uid, added = self.counter + added, added + 1
        results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))
        self.status_map[uid] = RequestStatus(...)
    self.counter += added
    self.pending_requests = self.pending_requests[added:]
    return results
```

#### 参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `blocking` | `bool` | 若为 True 且无待处理请求，抛出 `RequestAllFinished`；若为 False 则返回空列表 |

#### 返回值

`List[BaseBackendMsg]`：已 tokenize 的 `UserMsg` 列表，数量受 `prefill_budget`（`max_extend_tokens`）限制。

#### 关键实现细节

**Prefill 预算控制**：累积输入长度达到 `prefill_budget` 时停止添加新请求，实现 Chunk Prefill 分片——剩余请求在下一轮 `receive_msg` 调用时继续处理。

**停止条件**：`blocking=True` 时，若 `pending_requests` 为空，说明所有请求已分发完毕（且 scheduler 循环已处理完最后一批），抛出 `RequestAllFinished` 触发推理终止。`blocking=False` 时（scheduler 还有待处理批次时），返回空列表，让 scheduler 继续处理队列中的请求。

#### `offline_send_result`

```python
# python/minisgl/llm/llm.py  行 71-75
def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
    for msg in reply:
        status = self.status_map[msg.uid]
        if not (msg.finished and msg.next_token == self.eos_token_id):
            status.output_ids.append(msg.next_token)
```

#### 参数

`reply`：scheduler 产生的 `DetokenizeMsg` 列表，每条包含一个新生成的 token。

#### 关键实现细节

**EOS 过滤逻辑**：`not (msg.finished and msg.next_token == self.eos_token_id)` 表示：只有在请求结束**且**最后一个 token 是 EOS token 时，才跳过追加（避免 EOS 出现在输出文本中）。若请求中途结束（如达到 `max_tokens`）且最后 token 不是 EOS，则仍然追加该 token。

#### `generate`

```python
# python/minisgl/llm/llm.py  行 77-98
def generate(
    self,
    prompts: List[str] | List[List[int]],
    sampling_params: List[SamplingParams] | SamplingParams,
) -> List[Dict[str, str | List[int]]]:
```

#### 参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `prompts` | `List[str] \| List[List[int]]` | 输入提示列表，支持字符串或 token ID 列表 |
| `sampling_params` | `List[SamplingParams] \| SamplingParams` | 采样参数，可以是单个（广播到所有请求）或列表 |

#### 返回值

`List[Dict[str, str | List[int]]]`：每个元素对应一个输入的生成结果：
```python
{
    "text": "生成的文本字符串",
    "token_ids": [1, 2, 3, ...]  # 生成的 token ID 列表
}
```

#### 关键实现细节

```python
# python/minisgl/llm/llm.py  行 82-98
def generate(self, prompts, sampling_params):
    # 1. 重置状态
    self.pending_requests = []
    self.status_map = {}
    self.counter = 0
    # 2. 广播单个 sampling_params
    if isinstance(sampling_params, SamplingParams):
        sampling_params = [sampling_params] * len(prompts)
    # 3. 填充待处理队列
    for prompt, sp in zip(prompts, sampling_params):
        self.pending_requests.append((prompt, sp))
    # 4. 运行推理直到完成
    try:
        self.run_forever()
    except RequestAllFinished:
        pass
    # 5. 收集并解码结果
    results = []
    for i in range(len(prompts)):
        status = self.status_map[i]
        output_text = self.tokenizer.decode(status.output_ids)
        results.append({"text": output_text, "token_ids": status.output_ids})
    return results
```

**异常驱动的终止机制**：`run_forever()` 是设计为永不返回的无限循环（`NoReturn`），通过捕获 `RequestAllFinished` 异常实现正常终止。这是对现有 scheduler 架构的最小侵入性扩展。

**结果顺序保证**：以 `range(len(prompts))` 顺序（uid 0, 1, 2, ...）从 `status_map` 中取结果，保证输出顺序与输入顺序一致，符合批量推理接口的常规约定。

**状态重置**：每次 `generate()` 调用前清空 `pending_requests`、`status_map` 和 `counter`，使 `LLM` 实例可以被多次调用。

### 7.5 与在线服务模式的对比

| 维度 | 在线服务（`Scheduler` + HTTP） | 离线批量（`LLM`） |
|---|---|---|
| 消息来源 | ZMQ PULL socket（来自 tokenizer 进程） | `pending_requests` 内存队列 |
| 结果输出 | ZMQ PUSH socket（推给 detokenizer） | `status_map` 内存字典 |
| 并发度 | 多进程，最多 `max_running_req` 并发 | 单进程，所有请求在同一事件循环 |
| Tokenization | 独立 tokenizer 进程异步处理 | 主进程同步 tokenize |
| 终止条件 | `KeyboardInterrupt` | `RequestAllFinished` 异常 |

### 7.6 依赖关系

```
LLM
├── minisgl.scheduler.Scheduler      (继承，完整推理调度能力)
├── minisgl.scheduler.SchedulerConfig (配置类)
├── minisgl.distributed.DistributedInfo (单卡配置)
├── minisgl.core.SamplingParams       (采样参数)
└── minisgl.message.DetokenizeMsg     (输出消息格式)
    minisgl.message.UserMsg           (输入消息格式)
    minisgl.message.BaseBackendMsg    (消息基类)
```

---

## 附录：模块间通信架构全览

理解各模块的完整交互关系，有助于把握入口与服务层的整体设计。

### ZMQ 通信拓扑

```
用户
 │ HTTP
 ▼
[API Server / FastAPI]          ← run_api_server() 在主进程 asyncio 事件循环中运行
 │  PUSH  ──────────────────────────────────────────────────┐
 │  (zmq_tokenizer_addr: minisgl_4 或 minisgl_1)            │
 │                                                           ▼
 │                                              [Tokenizer Process(es)]
 │                                               ├── TokenizeManager
 │                                               └── DetokenizeManager
 │                                                    │
 │  PULL  ◄──────────────────────────── PUSH          │ PUSH
 │  (zmq_frontend_addr: minisgl_3)       │             │ (zmq_backend_addr: minisgl_0)
 │                                       │             ▼
 └───────────────────────────────────────┘   [Scheduler Process(es) × TP_SIZE]
                                              ├── TP rank 0  (primary)
                                              │    ├── recv: minisgl_0 (PULL)
                                              │    ├── send: minisgl_1 (PUSH → detokenizer)
                                              │    └── broadcast: minisgl_2 (PUB)
                                              └── TP rank 1..N
                                                   └── recv: minisgl_2 (SUB from rank 0)
```

### 进程启动时序

```
主进程 (run_api_server)
  │
  ├─[1] 创建 ZMQ PULL socket (minisgl_3)    ← FrontendManager 初始化
  ├─[2] 创建 ZMQ PUSH socket (minisgl_4/1)  ← FrontendManager 初始化
  │
  ├─[3] 调用 start_backend() ─────────────────────────────────┐
  │                                                            │
  │     ├─ spawn TP-0 Scheduler ──────────────────────────────┤
  │     ├─ spawn TP-1 Scheduler (若 tp_size > 1)             │
  │     ├─ spawn detokenizer                                  │
  │     └─ spawn tokenizer(s)                                 │
  │                                                            │
  ├─[4] 等待 ack_queue (num_tokenizers + 2 条)                │
  │     ◄── "Scheduler is ready"         (TP-0 发出)          │
  │     ◄── "Tokenize server 0 is ready" (detokenizer 发出)   │
  │     ◄── "Tokenize server i is ready" (tokenizer 发出)     │
  │                                                            │
  └─[5] 启动 uvicorn 或 asyncio.run(shell())
```

### 消息类型映射

| 消息类 | 方向 | 说明 |
|---|---|---|
| `TokenizeMsg` | API → Tokenizer | 文本 tokenization 请求 |
| `AbortMsg` | API → Tokenizer → Scheduler | 取消请求 |
| `UserMsg` | Tokenizer → Scheduler | 已 tokenize 的推理请求 |
| `AbortBackendMsg` | Tokenizer → Scheduler | 转发的取消请求 |
| `DetokenizeMsg` | Scheduler → Tokenizer | 单个新生成的 token |
| `UserReply` | Tokenizer → API | 增量文本回复 |
| `BatchTokenizerMsg` | 任意 → Tokenizer | 批量消息封装 |
| `BatchFrontendMsg` | Tokenizer → API | 批量回复封装 |
| `BatchBackendMsg` | Tokenizer → Scheduler | 批量请求封装 |
