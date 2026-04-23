# 附录 H：Tokenizer、工具函数与自定义算子模块代码详解

本附录对 mini-sglang 项目中三个支撑性目录——`tokenizer/`、`utils/`、`kernel/`——的全部源文件进行逐一解析。这三个目录不承载推理主逻辑，却是系统运行的基础设施：tokenizer 负责将用户请求转换为模型能理解的 token ID，并将输出 token 实时还原为文本；utils 提供跨进程通信、日志、GPU 探测等横切关注点；kernel 则封装了从向量化索引到分布式通信再到 MoE 矩阵乘法的所有自定义算子。

---

## 第一节 tokenizer/ — 分词子系统

tokenizer 子系统由三个文件构成，各自承担独立职责：`tokenize.py` 将原始文本或对话消息编码为 token ID；`detokenize.py` 将流式生成的 token ID 增量还原为可读字符串；`server.py` 是运行在独立进程中的 Tokenizer Worker，充当两者的调度中枢。

### H.1.1 `python/minisgl/tokenizer/tokenize.py` — 分词管理器

**文件职责**：将 `TokenizeMsg`（原始字符串或多轮对话列表）编码为 `int32` 类型的 token ID 张量。

#### 类与函数说明

##### `class TokenizeManager`

```python
class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None
    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]
```

**构造函数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `tokenizer` | `PreTrainedTokenizerBase` | 已加载的 HuggingFace 分词器实例 |

构造函数仅保存 `tokenizer` 引用，无其他初始化逻辑。

**`tokenize` 方法**

| 参数 | 类型 | 说明 |
|------|------|------|
| `msgs` | `List[TokenizeMsg]` | 待编码的消息列表 |
| 返回值 | `List[torch.Tensor]` | 每条消息对应一个一维 `int32` 张量，形状为 `(seq_len,)` |

#### Chat Template 处理逻辑（关键实现）

`TokenizeMsg.text` 字段有两种类型：
- **字符串**（`str`）：直接作为 prompt 送入 `tokenizer.encode`；
- **消息列表**（`list`）：代表多轮对话，需先通过 `apply_chat_template` 渲染成完整字符串，再编码。

```python
# tokenize.py，第 17–31 行
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

关键细节：
1. `tokenize=False`：让 `apply_chat_template` 仅返回渲染后的字符串，不自动编码，由后续 `encode` 统一处理。
2. `add_generation_prompt=True`：在对话末尾附加模型应答提示符（如 `<|im_start|>assistant\n`），驱动模型生成。
3. `enable_thinking=False`：关闭 Qwen3 系列模型的"思考模式"（Chain-of-Thought 前缀），保持输出简洁。
4. 最终张量通过 `.view(-1).to(torch.int32)` 展平并转为 32 位整型，与后端 CUDA kernel 的数据类型约定匹配。

当前实现以循环逐条处理（`# TODO: batch tokenization`），批量化编码留作后续优化。

---

### H.1.2 `python/minisgl/tokenizer/detokenize.py` — 反分词管理器

**文件职责**：在流式推理过程中，对每个请求维护一个解码状态机，将增量到来的 token ID 转换为可安全输出的增量字符串。

#### 辅助函数

##### `_is_chinese_char(cp: int) -> bool`

```python
def _is_chinese_char(cp: int) -> bool
```

判断 Unicode 码点 `cp` 是否属于 CJK（中日韩）统一表意文字区块。覆盖 8 个 Unicode 区间（U+4E00–U+9FFF、U+3400–U+4DBF、U+20000–U+2A6DF 等）。该函数用于流式输出时的边界判断：CJK 字符天然以单字为词，无需等待后续 token 即可安全输出。

##### `find_printable_text(text: str) -> str`

```python
def find_printable_text(text: str) -> str
```

从已解码字符串中截取可安全输出的最长前缀，遵循三级优先规则：
1. 若字符串以 `\n` 结尾，整体输出（换行符是强边界）；
2. 若最后一个字符是 CJK，整体输出；
3. 若倒数第二个字符是 CJK，输出除最后一个字符外的内容；
4. 否则，输出到最后一个空格（含）之前（避免截断拉丁文单词）。

借鉴自 HuggingFace Transformers `TextStreamer` 的同名实现。

#### 数据类

##### `@dataclass class DecodeStatus`

```python
@dataclass
class DecodeStatus:
    decoded_ids: List[int]   # 该请求已收到的全部 token ID
    decoded_str: str          # 已确认安全的累计输出字符串
    read_offset: int          # decoded_ids 中"已读指针"位置
    surr_offset: int          # decoded_ids 中"替补指针"位置
    sent_offset: int          # decoded_str 中已发送给调用方的字符数
```

三个整型偏移量共同构成状态机的核心：

| 指针 | 含义 | 前进时机 |
|------|------|----------|
| `surr_offset` | 上一轮批量解码的起点 | 确认新增文本无乱码（不含 `\uFFFD`）后，与 `read_offset` 对齐 |
| `read_offset` | 本轮批量解码的终点 | 每次追加新 token 后，更新为 `decoded_ids` 的当前长度 |
| `sent_offset` | 已输出给调用方的字符偏移 | 每次调用后，更新为 `len(output_str)` |

#### 类说明

##### `class DetokenizeManager`

```python
class DetokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None
    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]
```

**构造函数**：初始化 `decode_map: Dict[int, DecodeStatus]`（以请求 UID 为键），保存 tokenizer 及 EOS token ID。

**`detokenize` 方法**（行 70–111）

```python
def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `msgs` | `List[DetokenizeMsg]` | 每条消息包含 `uid`、`next_token`、`finished` 字段 |
| 返回值 | `List[str]` | 各请求本轮新增的增量字符串（流式 chunk） |

**三指针状态机工作流程**（关键实现细节）：

```python
# detokenize.py，第 83–109 行
# ① 追加新 token（非 EOS 结束时不追加）
if not (msg.finished and msg.next_token == self.eos_token_id):
    s.decoded_ids.append(msg.next_token)

# ② 构造两段 ID 范围
read_ids.append(s.decoded_ids[s.surr_offset :])      # "读窗口"：从 surr 到末尾
surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])  # "替补窗口"：surr 到 read

# ③ 批量解码（利用 batch_decode 减少 Python 调用开销）
read_texts = self.tokenizer.batch_decode(read_ids)
surr_texts = self.tokenizer.batch_decode(surr_ids)

# ④ 计算新增文本
new_text = read_str[len(surr_str) :]

# ⑤ 乱码检测
if len(new_text) > 0 and not new_text.endswith(""):
    # 无乱码：提交状态，前进 surr_offset
    output_str = s.decoded_str + new_text
    s.decoded_str = output_str
    s.surr_offset = s.read_offset
    s.read_offset = len(s.decoded_ids)
else:
    # 疑似乱码（未完整 UTF-8 序列）：用 find_printable_text 兜底
    new_text = find_printable_text(new_text)
    output_str = s.decoded_str + new_text

# ⑥ 提取增量，更新 sent_offset
incremental_output = output_str[s.sent_offset :]
s.sent_offset = len(output_str)

# ⑦ 请求结束时清理状态
if msg.finished:
    del self.decode_map[msg.uid]
```

**为什么需要"替补窗口"？**

某些多字节 token（如 UTF-8 中的汉字、Emoji）在被逐 token 解码时，若上一批次恰好切在字节边界中间，`batch_decode` 会产生 U+FFFD（`\uFFFD`，替换字符）。通过对比"包含新 token 的完整解码"（`read_texts`）与"不包含新 token 的旧解码"（`surr_texts`），取差值 `read_str[len(surr_str):]`，可以准确获取本 token 对应的新增字符，同时通过 `\uFFFD` 检测规避截断风险。`surr_offset` 只在确认无乱码后才前进，实现了"延迟提交"语义。

---

### H.1.3 `python/minisgl/tokenizer/server.py` — Tokenizer Worker 进程

**文件职责**：作为独立子进程运行，通过 ZMQ 队列接收来自 API Server 的消息，调用 `TokenizeManager` 或 `DetokenizeManager` 处理后，将结果分别路由到推理后端或前端响应队列。

#### 辅助函数

##### `_unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]`

```python
def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]
```

将 `BatchTokenizerMsg`（批次包装）拆解为单条消息列表；若输入本身是单条消息，则包装成单元素列表。用于统一后续的批处理逻辑。

#### 主函数

##### `tokenize_worker(...) -> None`

```python
@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `tokenizer_path` | `str` | 分词器模型路径（本地目录或 HuggingFace 模型 ID） |
| `addr` | `str` | 本 Worker 监听的 ZMQ 地址 |
| `create` | `bool` | 是否 `bind` 该地址（`True`）还是 `connect`（`False`） |
| `backend_addr` | `str` | 推理后端的 ZMQ PUSH 地址 |
| `frontend_addr` | `str` | 前端响应队列的 ZMQ PUSH 地址 |
| `local_bs` | `int` | 本地批大小上限，控制动态批次聚合的最大数量 |
| `tokenizer_id` | `int` | Worker 编号，用于日志标识，默认 -1 |
| `model_source` | `str` | 模型来源，当前仅支持 `"huggingface"` |
| `ack_queue` | `mp.Queue[str] \| None` | 启动完成确认队列，非 `None` 时在 ready 后写入确认消息 |

**工作循环（关键实现）**：

```python
# server.py，第 59–108 行
while True:
    # 1. 阻塞等待至少一条消息
    pending_msg = _unwrap_msg(recv_listener.get())
    # 2. 趁热打铁：在不阻塞的前提下尽量聚合消息，直到达到 local_bs
    while len(pending_msg) < local_bs and not recv_listener.empty():
        pending_msg.extend(_unwrap_msg(recv_listener.get()))

    # 3. 按消息类型分组
    detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
    tokenize_msg   = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
    abort_msg      = [m for m in pending_msg if isinstance(m, AbortMsg)]

    # 4. 分词结果 → 后端（UserMsg）
    if len(tokenize_msg) > 0:
        tensors = tokenize_manager.tokenize(tokenize_msg)
        batch_output = BatchBackendMsg(data=[
            UserMsg(uid=msg.uid, input_ids=t, sampling_params=msg.sampling_params)
            for msg, t in zip(tokenize_msg, tensors, strict=True)
        ])
        if len(batch_output.data) == 1:
            batch_output = batch_output.data[0]  # 单条时解包，避免批次开销
        send_backend.put(batch_output)

    # 5. 反分词结果 → 前端（UserReply）
    if len(detokenize_msg) > 0:
        replies = detokenize_manager.detokenize(detokenize_msg)
        batch_output = BatchFrontendMsg(data=[
            UserReply(uid=msg.uid, incremental_output=reply, finished=msg.finished)
            for msg, reply in zip(detokenize_msg, replies, strict=True)
        ])
        ...
        send_frontend.put(batch_output)

    # 6. 中止消息 → 后端（AbortBackendMsg）
    if len(abort_msg) > 0:
        ...
        send_backend.put(batch_output)
```

**动态批次聚合策略**：Worker 在取到第一条消息后，立即尝试从队列中继续消费消息（非阻塞 `recv_listener.empty()` 检查），直到积累到 `local_bs` 条为止。这在高并发下可以显著减少 `tokenize` / `batch_decode` 调用次数，摊薄 Python 开销。

**进程隔离价值**：`@torch.inference_mode()` 修饰整个函数，使整个进程的 PyTorch 运行在无梯度模式下。ZMQ 通道将 Tokenizer 与推理主进程完全解耦，二者之间无共享内存，避免了 Python GIL 竞争。

---

## 第二节 utils/ — 工具函数库

utils 目录提供系统级横切工具，涵盖跨进程消息队列、彩色日志、HuggingFace 模型下载、GPU 架构检测、数学工具函数以及 PyTorch 性能标注装饰器。

### H.2.1 `python/minisgl/utils/registry.py` — 通用注册表

**文件职责**：提供泛型注册表，支持以字符串名称注册任意 Python 对象（类、函数等），并在运行时按名称查找，同时提供参数校验辅助方法。

#### 类说明

##### `class Registry(Generic[T])`

```python
class Registry(Generic[T]):
    def __init__(self, type: str)
    def register(self, name: str) -> Callable[[T], None]
    def __getitem__(self, name: str) -> T
    def supported_names(self) -> List[str]
    def assert_supported(self, names: str | Iterable[str]) -> None
```

`Registry` 是一个泛型类，`T` 代表被注册对象的类型（如模型类、算子函数等）。

**`__init__(self, type: str)`**

| 参数 | 类型 | 说明 |
|------|------|------|
| `type` | `str` | 注册表的语义名称（如 `"model"`、`"attention"`），用于错误提示 |

内部维护 `_registry: dict`，键为注册名称，值为被注册对象。

**`register(self, name: str) -> Callable[[T], None]`**（装饰器工厂）

```python
@registry.register("llama")
class LlamaModel: ...
```

返回一个装饰器，将被装饰对象以 `name` 为键存入 `_registry`。若 `name` 已存在，抛出 `KeyError`，防止意外覆盖。**注意**：装饰器返回 `None`，被装饰的类/函数不会被替换，原始对象照常可用。

**`__getitem__(self, name: str) -> T`**

通过 `registry["llama"]` 语法按名称查找。未找到时抛出 `KeyError` 并说明类型信息。

**`supported_names(self) -> List[str]`**

返回当前已注册的全部名称列表，用于枚举可选项。

**`assert_supported(self, names: str | Iterable[str]) -> None`**

```python
registry.assert_supported(["llama", "qwen2"])
```

批量校验名称合法性。不合法时抛出 `argparse.ArgumentTypeError`（而非普通异常），便于在命令行参数解析阶段直接产生有用的错误消息，并自动列出所有支持的选项。

**使用模式示例**：

```python
# 定义注册表
MODEL_REGISTRY = Registry[Type[nn.Module]]("model")

# 注册（装饰器风格）
@MODEL_REGISTRY.register("llama3")
class LlamaForCausalLM(nn.Module): ...

# 查找并实例化
model_cls = MODEL_REGISTRY["llama3"]
model = model_cls(config)
```

---

### H.2.2 `python/minisgl/utils/mp.py` — ZMQ 队列封装

**文件职责**：对 ZeroMQ 的 PUSH/PULL/PUB/SUB 模式进行封装，提供类型安全、支持 msgpack 序列化的消息队列类，分同步和异步两个版本。

#### 设计思路

所有队列类均为泛型 `Generic[T]`，通过构造时传入的 `encoder`/`decoder` 回调实现消息的序列化/反序列化。序列化格式统一使用 `msgpack`（二进制紧凑格式，比 JSON 更高效）。

**`create` 参数语义**：`True` 表示该端调用 `socket.bind(addr)` 成为服务端（被动），`False` 表示调用 `socket.connect(addr)` 成为客户端（主动）。

#### 同步版本

##### `class ZmqPushQueue(Generic[T])`

```python
class ZmqPushQueue(Generic[T]):
    def __init__(self, addr: str, create: bool, encoder: Callable[[T], Dict])
    def put(self, obj: T) -> None
    def stop(self) -> None
```

使用 `zmq.PUSH` socket，调用 `put` 时同步发送一条消息。`encoder` 将对象转为可 msgpack 序列化的 `Dict`；`copy=False` 避免发送时的内存拷贝。

##### `class ZmqPullQueue(Generic[T])`

```python
class ZmqPullQueue(Generic[T]):
    def __init__(self, addr: str, create: bool, decoder: Callable[[Dict], T])
    def get(self) -> T
    def get_raw(self) -> bytes
    def decode(self, raw: bytes) -> T
    def empty(self) -> bool
    def stop(self) -> None
```

使用 `zmq.PULL` socket。

| 方法 | 说明 |
|------|------|
| `get()` | 阻塞接收并反序列化，返回类型为 `T` |
| `get_raw()` | 阻塞接收原始字节，不反序列化（配合 `decode` 手动处理）|
| `decode(raw)` | 对已有字节流进行反序列化 |
| `empty()` | 非阻塞轮询（`poll(timeout=0)`），检查队列是否为空 |

`empty()` 是 Tokenizer Worker 实现动态批次聚合的关键：Worker 在处理完当前批次前，通过该方法判断是否还有积压消息可以合并。

#### 异步版本

##### `class ZmqAsyncPushQueue(Generic[T])` 和 `class ZmqAsyncPullQueue(Generic[T])`

与同步版本结构相同，区别在于使用 `zmq.asyncio.Context()` 和 `asyncio` socket，`put`/`get` 方法均为 `async def`，可在 `asyncio` 事件循环中协程式调用。

```python
# 异步示例
async def send_request(queue: ZmqAsyncPushQueue, msg):
    await queue.put(msg)
```

#### PUB/SUB 模式

##### `class ZmqPubQueue(Generic[T])` 和 `class ZmqSubQueue(Generic[T])`

用于广播场景（如向所有 TP Worker 广播调度指令）。

`ZmqPubQueue` 提供 `put(obj)` 和 `put_raw(raw)` 两种发送接口，前者自动序列化，后者直接发送字节（转发已有消息时避免重复序列化）。`ZmqSubQueue` 在构造时通过 `setsockopt_string(zmq.SUBSCRIBE, "")` 订阅所有主题。

**六类队列汇总**：

| 类名 | ZMQ 模式 | 同步/异步 | 主要用途 |
|------|----------|-----------|---------|
| `ZmqPushQueue` | PUSH | 同步 | Tokenizer Worker 发送结果 |
| `ZmqPullQueue` | PULL | 同步 | Tokenizer Worker 接收请求 |
| `ZmqAsyncPushQueue` | PUSH | 异步 | API Server 发送请求 |
| `ZmqAsyncPullQueue` | PULL | 异步 | API Server 接收响应 |
| `ZmqPubQueue` | PUB | 同步 | TP 广播调度指令 |
| `ZmqSubQueue` | SUB | 同步 | TP Worker 接收广播 |

---

### H.2.3 `python/minisgl/utils/logger.py` — 日志初始化

**文件职责**：提供带 ANSI 颜色、时间戳格式化输出的日志初始化函数，并扩展标准 `Logger` 为分布式场景添加 `*_rank0` 方法。

#### 函数说明

##### `init_logger(name, suffix, *, strip_file, level, use_pid, use_tp_rank) -> Logger`

```python
def init_logger(
    name: str,
    suffix: str = "",
    *,
    strip_file: bool = True,
    level: str | None = None,
    use_pid: bool | None = None,
    use_tp_rank: bool | None = None,
) -> Logger
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | — | `logging.getLogger` 的 logger 名称（通常传 `__name__`） |
| `suffix` | `str` | `""` | 附加到时间戳中的标识字符串（如 `"tokenizer_0"`） |
| `strip_file` | `bool` | `True` | 若 `suffix` 是文件路径，只保留文件名部分 |
| `level` | `str \| None` | `None` | 日志级别，`None` 时读取 `LOG_LEVEL` 环境变量，默认 `INFO` |
| `use_pid` | `bool \| None` | `None` | 是否在时间戳中显示 PID，`None` 时读取 `LOG_PID` 环境变量 |
| `use_tp_rank` | `bool \| None` | `None` | 是否在时间戳中显示 TP rank，`None` 时自动检测 |

**时间戳格式**：`[YYYY-MM-DD|HH:MM:SS|suffix|pid=N|core|rank=R]`，各字段按实际启用情况拼接。

**颜色映射**：

| 级别 | ANSI 颜色 |
|------|-----------|
| DEBUG | 青色（`\033[36m`） |
| INFO | 绿色（`\033[32m`） |
| WARNING | 黄色（`\033[33m`） |
| ERROR | 红色（`\033[31m`） |
| CRITICAL | 洋红（`\033[35m`） |

**扩展方法**：函数在返回的 `logger` 上动态附加四个 `*_rank0` 方法（`info_rank0`、`debug_rank0`、`warning_rank0`、`critical_rank0`），通过 `functools.partial` 绑定内部 `_call_rank0` 函数。这些方法仅在当前进程是 TP 主进程（rank 0）时才实际输出日志，避免多卡训练/推理时日志重复。

**全局单例日志级别**：模块级变量 `_LOG_LEVEL` 在首次调用时初始化并缓存，后续调用跳过重复初始化，确保所有 logger 使用相同级别。

**TP Rank 懒加载**：`tp_info` 通过闭包变量缓存，第一次格式化日志时调用 `try_get_tp_info()` 获取，避免在 TP 初始化前过早调用。

---

### H.2.4 `python/minisgl/utils/hf.py` — HuggingFace 工具

**文件职责**：封装 HuggingFace Tokenizer、模型配置及权重的加载与下载操作。

#### 类与函数说明

##### `class DisabledTqdm(tqdm)`

继承自 `tqdm.asyncio.tqdm`，构造时强制传入 `disable=True`，屏蔽所有进度条输出。用于静默下载场景。

##### `load_tokenizer(model_path: str) -> PreTrainedTokenizerBase`

```python
def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase
```

对 `AutoTokenizer.from_pretrained` 的薄封装，供 `tokenize_worker` 调用。

##### `cached_load_hf_config(model_path: str) -> PretrainedConfig`

```python
@functools.cache
def _load_hf_config(model_path: str) -> Any

def cached_load_hf_config(model_path: str) -> PretrainedConfig
```

两层设计：`_load_hf_config` 使用 `@functools.cache` 将 `AutoConfig.from_pretrained` 的结果缓存（以 `model_path` 为键），避免重复 I/O；`cached_load_hf_config` 在此基础上通过 `type(config)(**config.to_dict())` 返回一个新的同类配置对象拷贝，防止调用方意外修改缓存中的原始对象。

##### `download_hf_weight(model_path: str) -> str`

```python
def download_hf_weight(model_path: str) -> str
```

| 情况 | 行为 |
|------|------|
| `model_path` 是已存在的本地目录 | 直接返回路径 |
| `model_path` 是 HuggingFace 模型 ID | 调用 `snapshot_download` 仅下载 `*.safetensors` 文件，返回本地缓存路径 |
| 两者均不满足 | 抛出 `ValueError` |

使用 `DisabledTqdm` 静默下载进度条，保持服务器环境下的日志整洁。

---

### H.2.5 `python/minisgl/utils/arch.py` — GPU 架构检测

**文件职责**：提供 CUDA 计算能力（Compute Capability）检测函数，用于在运行时按 GPU 架构选择最优 kernel 实现。

#### 函数说明

##### `_get_torch_cuda_version() -> Tuple[int, int] | None`

```python
@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None
```

通过 `torch.cuda.get_device_capability()` 返回当前 GPU 的计算能力版本元组（如 `(8, 0)` 代表 Ampere A100）。使用 `@functools.cache` 缓存结果，避免重复调用 CUDA API。若无 CUDA 可用则返回 `None`。

##### `is_arch_supported(major: int, minor: int = 0) -> bool`

```python
def is_arch_supported(major: int, minor: int = 0) -> bool
```

判断当前 GPU 是否满足最低计算能力要求，使用元组比较 `arch >= (major, minor)`。

##### `is_sm90_supported() -> bool`

```python
def is_sm90_supported() -> bool  # Hopper（H100）
```

##### `is_sm100_supported() -> bool`

```python
def is_sm100_supported() -> bool  # Blackwell（B200）
```

二者均为 `is_arch_supported` 的特化版，对应当前主流数据中心 GPU 架构。在加载 Flash Attention 3、CUTLASS GEMM 等需要特定 SM 版本的 kernel 前，应先调用这些函数进行门控。

---

### H.2.6 `python/minisgl/utils/misc.py` — 杂项工具

**文件职责**：提供整数对齐运算和条件执行装饰器等基础工具函数。

#### 函数与类说明

##### `call_if_main(name: str = "__main__", discard: bool | None = None)`

```python
def call_if_main(name: str = "__main__", discard: bool | None = None)
```

装饰器工厂，用于控制模块级脚本的条件执行。

| `name` 值 | `discard` 值 | 行为 |
|-----------|-------------|------|
| `"__main__"` | `None`（默认 `True`） | 立即调用被装饰函数，返回 `None`（用于"直接运行时执行，导入时跳过"） |
| `"__main__"` | `False` | 调用后仍返回原函数 |
| 其他 | `None`（默认 `False`） | 返回原函数（无操作） |
| 其他 | `True` | 返回 `lambda _: None`（丢弃函数） |

典型用途：替代 `if __name__ == "__main__": main()`，以装饰器形式表达。

##### `div_even(a: int, b: int) -> int`

整除并断言整除性，比 `a // b` 多一层运行时校验，用于内存布局计算等要求精确整除的场合。

##### `div_ceil(a: int, b: int) -> int`

向上取整除法：`(a + b - 1) // b`，等价于 `math.ceil(a / b)` 但避免浮点转换。

##### `align_ceil(a: int, b: int) -> int`

将 `a` 向上对齐到 `b` 的倍数：`div_ceil(a, b) * b`。常用于 CUDA kernel 中将序列长度、通道数对齐到 warp/tile 边界。

##### `align_down(a: int, b: int) -> int`

将 `a` 向下对齐到 `b` 的最大倍数：`(a // b) * b`。

##### `class Unset` / `UNSET`

```python
class Unset:
    pass

UNSET = Unset()
```

哨兵对象，区分"未传入参数"与 `None` 的语义。当函数默认值既可以是 `None` 又需要区分"未设置"状态时使用，避免歧义。

---

### H.2.7 `python/minisgl/utils/torch_utils.py` — PyTorch 工具

**文件职责**：提供 PyTorch 相关的上下文管理器和性能标注装饰器。

#### 函数与装饰器说明

##### `torch_dtype(dtype: torch.dtype)` — 上下文管理器

```python
@contextmanager
def torch_dtype(dtype: torch.dtype)
```

在 `with` 块内临时切换 PyTorch 默认张量数据类型，退出时恢复原始类型。

```python
# 使用示例
with torch_dtype(torch.bfloat16):
    model = LlamaModel(config)  # 所有默认类型参数均为 bfloat16
```

实现采用标准的"保存→设置→恢复"模式（`try/finally` 保证异常时也能恢复），避免全局状态污染。`import torch` 在函数体内延迟导入，减少模块加载时的 import 开销。

##### `nvtx_annotate(name: str, layer_id_field: str | None = None)` — NVTX 性能标注装饰器

```python
def nvtx_annotate(name: str, layer_id_field: str | None = None)
```

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | NVTX range 的名称，支持 `{}` 占位符（当 `layer_id_field` 非空时用于格式化） |
| `layer_id_field` | `str \| None` | `self` 上的属性名，用于将层编号嵌入标注名称中 |

**用途**：NVTX（NVIDIA Tools Extension）是 Nsight Systems 等 GPU 性能分析工具使用的标注 API。被装饰的方法在执行时会在 NVTX timeline 上产生一个命名区间，使分析工具能精确定位各层计算的时间和资源占用。

```python
# 使用示例
class TransformerLayer:
    layer_id = 3

    @nvtx_annotate("TransformerLayer[{}]", layer_id_field="layer_id")
    def forward(self, x):
        ...
# Nsight Systems 中将显示 "TransformerLayer[3]" 区间
```

**实现细节**：

```python
# torch_utils.py，第 26–37 行
def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)
        return wrapper

    return decorator
```

- 使用 `functools.wraps` 保留原函数的 `__name__`、`__doc__` 等元数据；
- `layer_id_field` 支持运行时动态获取层编号，生成如 `"Attention[7]"` 的唯一标注；
- `nvtx` 在装饰器工厂（外层函数）调用时立即导入，而非在每次 `forward` 调用时导入，减少热路径开销。

---

## 第三节 kernel/ — 自定义算子库

kernel 目录封装了 mini-sglang 的所有自定义 CUDA/Triton kernel，通过 TVM FFI（`tvm_ffi`）层暴露给 Python 层。算子分为两类：基于预编译（AOT，`load_aot`）的基础算子，以及基于即时编译（JIT，`load_jit`）的参数化 CUDA kernel。Triton kernel 则直接由 Python 调用 `@triton.jit` 编译。

### H.3.1 `python/minisgl/kernel/index.py` — 索引算子

**文件职责**：通过 JIT 编译的 CUDA kernel 执行嵌入表（Embedding Table）的高效查找（gather 操作），支持多分块并行以提升内存带宽利用率。

#### 常量与配置

```python
DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)
```

`KernelConfig` 是一个 NamedTuple，包含 CUDA kernel 的三个模板参数：线程数、最大占用率、是否启用 PDL（Programmatic Dependent Launch）。

#### 函数说明

##### `_jit_index_module(element_size, *, num_splits, config) -> Module`

```python
@functools.cache
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module
```

按 `(element_size, num_splits, config)` 三元组懒加载并缓存对应的 JIT 编译模块。`element_size` 是每行元素的字节数（即 `hidden_size * dtype_bytes`），作为模板参数传入 CUDA kernel，使编译器在编译时确定内存访问粒度。

##### `indexing(weights, indices, *, output, vocab_range) -> torch.Tensor`

```python
def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,
) -> torch.Tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `weights` | `Tensor[V, H]` | 嵌入权重矩阵，形状 `(vocab_size, hidden_dim)` |
| `indices` | `Tensor[N]` | 待查找的 token ID 列表，长度为 `N` |
| `output` | `Tensor[N, H] \| None` | 输出缓冲区，`None` 时自动分配 |
| `vocab_range` | `(start, length) \| None` | 分片词表范围，用于张量并行时按 TP rank 分割词表 |
| 返回值 | `Tensor[N, H]` | 查找结果，每行对应一个 token 的嵌入向量 |

**分块策略**（关键实现）：

```python
# index.py，第 41–48 行
element_size = weights.shape[1] * weights.element_size()
if element_size % 2048 == 0:
    num_splits = 4
elif element_size % 1024 == 0:
    num_splits = 2
else:
    num_splits = 1
```

根据每行字节数自动选择分块数，充分利用内存带宽：当隐藏维度较大（如 4096 维 × 2 字节 = 8192 字节，可整除 2048），使用 4 个并行流读取一行数据，减少 DRAM 访问延迟。

---

### H.3.2 `python/minisgl/kernel/store.py` — 缓存存储算子

**文件职责**：通过 JIT 编译的 CUDA kernel 将当前步骤计算出的 Key/Value 向量写入 KV Cache 的指定物理页（page）中。

#### 函数说明

##### `_jit_store_module(element_size, *, config) -> Module`

与 `index.py` 中的 `_jit_index_module` 类似，按 `element_size` 参数化并缓存 JIT 模块。

##### `store_cache(k_cache, v_cache, indices, k, v) -> None`

```python
def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `k_cache` | `Tensor[num_pages, page_size, ...]` | K Cache 物理存储，首维为 slot 数量 |
| `v_cache` | `Tensor[num_pages, page_size, ...]` | V Cache 物理存储，结构同 `k_cache` |
| `indices` | `Tensor[T]` | 当前批次各 token 对应的物理 slot 编号 |
| `k` | `Tensor[T, ...]` | 当前步骤计算的 Key 向量 |
| `v` | `Tensor[T, ...]` | 当前步骤计算的 Value 向量 |

函数先将 `k_cache`/`v_cache` 用 `.view(num_tokens, -1)` 展平到二维，再派发给 CUDA kernel 执行散射写入（scatter write），将 `k[i]`/`v[i]` 写入 `k_cache[indices[i]]`/`v_cache[indices[i]]` 对应行。这是 PagedAttention KV Cache 管理的核心写路径。

---

### H.3.3 `python/minisgl/kernel/radix.py` — 基数树算子

**文件职责**：通过 AOT 编译的 C++ 扩展，提供高效的 Tensor 比较操作，用于 Radix Tree（基数前缀树）的节点键值比较。

#### 函数说明

##### `_load_radix_module() -> Module`

```python
@functools.cache
def _load_radix_module() -> Module
```

AOT 加载 `radix.cpp`，结果被 `@functools.cache` 缓存，进程内只编译一次。

##### `fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int`

```python
def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x`, `y` | `Tensor[N]`（1D int, CPU） | 待比较的两个 token ID 序列 |
| 返回值 | `int` | 公共前缀的长度 |

在 Radix Tree 的 Prefix Caching 机制中，插入或查找一个序列时需要与树节点的 key 进行最长公共前缀匹配。该函数通过 C++ 扩展实现，比纯 Python 的逐元素比较快若干倍，对于缓存命中率分析的热路径至关重要。

---

### H.3.4 `python/minisgl/kernel/tensor.py` — Tensor 工具

**文件职责**：通过 AOT 编译的 C++ 扩展提供 Tensor 测试工具，主要供单元测试使用。

#### 函数说明

##### `test_tensor(x: torch.Tensor, y: torch.Tensor) -> int`

```python
def test_tensor(x: torch.Tensor, y: torch.Tensor) -> int
```

调用 `tensor.cpp` 中的 `test` 函数比较两个 Tensor，返回整型结果。具体语义由底层 C++ 实现决定，Python 层仅做薄封装。与 `radix.py` 同属"AOT + functools.cache"惯用模式。

---

### H.3.5 `python/minisgl/kernel/pynccl.py` — NCCL Python 绑定

**文件职责**：通过 AOT 编译的 CUDA 扩展（链接 libnccl）和 TVM FFI 对象系统，提供 NCCL AllReduce/AllGather 的 Python 接口，用于张量并行（Tensor Parallelism）的跨卡通信。

#### 类型存根（TYPE_CHECKING）

```python
class PyNCCLCommunicator:  # 仅在类型检查时可见
    def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
    def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
    def get_buffer(self) -> int: ...
```

实际运行时 `PyNCCLCommunicator = Any`，真实实现由 `PyNCCLImpl` 提供。

#### 函数说明

##### `_load_nccl_module() -> Module`

```python
@functools.cache
def _load_nccl_module() -> Module
```

AOT 加载 `pynccl.cu` 并链接 `-lnccl`，缓存结果。

##### `_get_pynccl_wrapper_cls()`

```python
@functools.cache
def _get_pynccl_wrapper_cls()
```

通过 `@tvm_ffi.register_object("minisgl.NCCLWrapper")` 注册一个 TVM FFI 托管对象类 `PyNCCLImpl`。TVM FFI 对象系统负责跨语言（Python ↔ C++）的对象生命周期管理和方法分发。

##### `init_pynccl(...) -> PyNCCLCommunicator`

```python
def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `tp_rank` | `int` | 当前进程在 TP 组内的 rank |
| `tp_size` | `int` | TP 组的总进程数 |
| `tp_cpu_group` | `ProcessGroup` | 用于广播 NCCL UID 的 CPU 通信组 |
| `max_size_bytes` | `int` | 通信缓冲区最大字节数（受 `ENV.PYNCCL_MAX_BUFFER_SIZE` 限制） |
| 返回值 | `PyNCCLCommunicator` | 初始化完成的 NCCL 通信器对象 |

**NCCL UID 广播流程**（关键实现）：

```python
# pynccl.py，第 59–78 行
if tp_rank == 0:
    id_list = [module.create_nccl_uid()]       # rank 0 生成唯一 UID
    torch.distributed.broadcast_object_list(
        id_list, src=0, group=tp_cpu_group,    # 通过 CPU 通信组广播到所有 TP 进程
    )
else:
    id_list = [None]
    torch.distributed.broadcast_object_list(
        id_list, src=0, group=tp_cpu_group,    # 其他 rank 等待接收
    )

nccl_id = id_list[0]
return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # 创建 C++ 侧 NCCL 通信器
```

NCCL 要求所有参与通信的进程使用相同的 UID（128 字节随机数）初始化通信器。mini-sglang 复用已有的 `torch.distributed` CPU 通信组来广播这个 UID，避免引入额外的同步机制。

---

### H.3.6 `python/minisgl/kernel/moe_impl.py` — MoE 算子实现

**文件职责**：提供基于 Triton 的混合专家（Mixture of Experts，MoE）前向计算接口，包括 token-expert 矩阵乘法和多专家输出的规约加法。

#### 函数说明

##### `fused_moe_kernel_triton(A, B, C, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type) -> None`

```python
def fused_moe_kernel_triton(
    A: torch.Tensor,              # 输入 token，形状 (M, K)
    B: torch.Tensor,              # 专家权重矩阵，形状 (E, N, K)
    C: torch.Tensor,              # 输出缓冲区，形状 (M, top_k, N)
    topk_weights: torch.Tensor,   # 路由权重，形状 (M, top_k)
    topk_ids: torch.Tensor,       # 各 token 对应专家 ID，形状 (M, top_k)
    sorted_token_ids: torch.Tensor,  # 按专家排序后的 token ID，形状 (EM,)
    expert_ids: torch.Tensor,        # 每个 block 对应的专家 ID
    num_tokens_post_padded: torch.Tensor,  # 填充后有效 token 数
    mul_routed_weight: bool,      # 是否将路由权重乘入输出
    top_k: int,                   # 每个 token 激活的专家数
    config: Dict[str, Any],       # Triton kernel 的分块配置参数
    compute_type: torch.dtype,    # 计算精度（bfloat16 或 float16）
) -> None
```

**grid 计算**：

```python
# moe_impl.py，第 28–31 行
grid = lambda META: (
    triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
    * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
)
```

Grid 维度为 `(M_blocks × N_blocks,)`，每个程序实例负责计算输出矩阵 `C` 的一个 `[BLOCK_SIZE_M, BLOCK_SIZE_N]` 分块。

**even_Ks 优化**：若 `K % BLOCK_SIZE_K == 0`，启用 `even_Ks=True`，使内层 K 循环中省略边界检查的 mask，提升编译后 kernel 的指令效率。

##### `moe_sum_reduce_triton(input: torch.Tensor, output: torch.Tensor) -> None`

```python
def moe_sum_reduce_triton(input: torch.Tensor, output: torch.Tensor) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `Tensor[M, top_k, H]` | 各 token 在各激活专家上的输出，须连续 |
| `output` | `Tensor[M, H]` | 各 token 所有激活专家输出的加权求和结果 |

固定超参数：`BLOCK_M=1`（每个程序实例处理 1 个 token）、`BLOCK_DIM=2048`（每次处理 2048 维）、`NUM_STAGE=1`、`num_warps=8`。Grid 为 `(M, H/2048)` 二维网格。

---

### H.3.7 `python/minisgl/kernel/utils.py` — 算子工具

**文件职责**：为 kernel 目录中所有算子提供统一的构建配置、参数转换和模块加载基础设施。

#### 常量

```python
KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]
DEFAULT_LDFLAGS = []
```

所有 C++/CUDA 编译均使用 C++20 标准和 `-O3` 优化，CUDA 额外启用 `--expt-relaxed-constexpr`（允许在 `constexpr` 函数中使用更多表达式，支持 C++20 的模板特性）。

#### 类说明

##### `class CppArgList(list[str])`

继承自 `list[str]`，覆盖 `__str__` 方法为 `", ".join(self)`，使 `str(args)` 直接生成 C++ 模板参数字符串，如 `"128, 1, false"`。

##### `class KernelConfig(NamedTuple)`

```python
class KernelConfig(NamedTuple):
    num_threads: int    # CUDA block 线程数
    max_occupancy: int  # 最大 SM 占用率
    use_pdl: bool       # 是否启用 PDL（Programmatic Dependent Launch）

    @property
    def template_args(self) -> str  # 生成 C++ 模板参数字符串
```

`template_args` 将三个字段格式化为 `"128,1,false"` 形式，直接嵌入 C++ 模板实例化语法。

#### 函数说明

##### `make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CppArgList`

```python
def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CppArgList
```

将 Python 的 `int`、`float`、`bool` 值转为 C++ 字面量字符串（`True` → `"true"`，`42` → `"42"`），组装成 `CppArgList`。

##### `load_aot(...) -> Module`

```python
def load_aot(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module
```

**AOT 模式**（Ahead-of-Time）：通过 `tvm_ffi.cpp.load` 加载预定义的 `.cpp`/`.cu` 文件。模块名由 `_make_name(*args)` 生成，格式为 `minisgl__arg1_arg2_...`，确保全局唯一。文件路径解析到 `KERNEL_PATH/src/` 下。

##### `load_jit(...) -> Module`

```python
def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    ...
) -> Module
```

**JIT 模式**（Just-in-Time）：通过 `tvm_ffi.cpp.load_inline` 在运行时内联编译。

与 AOT 的关键区别：
1. 文件路径解析到 `KERNEL_PATH/jit/` 下（与 `src/` 分开，JIT 文件通常是头文件形式）；
2. 支持 `cpp_wrappers` 和 `cuda_wrappers`：每个 wrapper 是一个 `(export_name, kernel_name)` 元组，通过 `_make_wrapper` 生成 `TVM_FFI_DLL_EXPORT_TYPED_FUNC(export_name, (kernel_name));` 宏调用，将 C++ 函数暴露为 TVM FFI 可调用接口；
3. 使用场景：参数化 CUDA kernel（如 `IndexKernel<128, 1, false>::run`）需要在知道 `element_size` 等参数后才能实例化，适合 JIT 编译。

**AOT vs JIT 对比**：

| 维度 | `load_aot` | `load_jit` |
|------|------------|------------|
| 编译时机 | 首次调用时（可被缓存） | 首次调用时（按参数编译） |
| 文件位置 | `csrc/src/` | `csrc/jit/` |
| 模板参数化 | 通过 wrapper 宏 | 通过 wrapper 宏 + 内联 |
| 典型用途 | `radix.cpp`、`pynccl.cu` | `index.cu`、`store.cu` |

---

### H.3.8 `python/minisgl/kernel/triton/fused_moe.py` — Triton 融合 MoE Kernel

**文件职责**：实现混合专家模型的核心计算：token-expert 分块矩阵乘法（`fused_moe_kernel`）和多专家输出规约（`moe_sum_reduce_kernel`）。

#### `moe_sum_reduce_kernel`

```python
@triton.jit
def moe_sum_reduce_kernel(
    input_ptr, input_stride_0, input_stride_1, input_stride_2,
    output_ptr, output_stride_0, output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
)
```

**功能**：将形状 `(M, top_k, H)` 的输入张量沿 `top_k` 维度求和，输出形状 `(M, H)`。

**分块策略**：
- Program ID 0 对应 token 分块（`BLOCK_M=1`，每个 program 处理 1 个 token）；
- Program ID 1 对应 hidden 维分块（`BLOCK_DIM=2048`）；
- 内层循环对 `topk_num` 个专家的输出累加到 `float32` accumulator，最后转换回输入数据类型写出。

```python
# fused_moe.py，第 36–47 行
for token_index in range(token_start, token_end):
    accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
    input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tmp = tl.load(input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0)
        accumulator += tmp
    tl.store(store_t_ptr, accumulator.to(input_ptr.dtype.element_ty), mask=offs_dim < dim_end)
```

#### `fused_moe_kernel`

```python
@triton.jit
def fused_moe_kernel(
    a_ptr, b_ptr, c_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
)
```

**输入矩阵语义**：

| 矩阵 | 形状 | 说明 |
|------|------|------|
| `A` | `(M_total, K)` | 所有请求的 token 向量（重复 top_k 次）|
| `B` | `(E, N, K)` | 全部 `E` 个专家的权重矩阵（转置存储） |
| `C` | `(M_total, top_k, N)` | 输出缓冲区 |
| `sorted_token_ids` | `(EM,)` | 按专家排序后的 token 索引（含填充） |
| `expert_ids` | `(EM/BLOCK_SIZE_M,)` | 每个 M-block 对应的专家编号 |

**分组排序（Grouped Ordering）**（关键实现）：

```python
# fused_moe.py，第 116–124 行
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

这是标准的 Triton GEMM 分组排序技巧：将 M 维的 `GROUP_SIZE_M` 个连续 block 与全部 N-block 组成一组，使同组内的程序实例共享 B 矩阵的 N-block 缓存，提升 L2 命中率。

**A 矩阵地址计算**（关键实现）：

```python
# fused_moe.py，第 142 行
a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
```

`sorted_token_ids` 中的 token 索引是"逻辑 token ID × top_k"，通过整除 `top_k` 还原到原始 token 位置，从 A 矩阵加载正确的行。

**B 矩阵专家选择**（关键实现）：

```python
# fused_moe.py，第 144–149 行
off_experts = tl.load(expert_ids_ptr + pid_m)   # 读取本 block 对应的专家编号
b_ptrs = (
    b_ptr
    + off_experts * stride_be                    # 跳到对应专家的权重页
    + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
)
```

每个 M-block 通过 `expert_ids` 查表确定本块所有 token 共同对应的专家，然后直接从 B 的对应专家位置加载 `[K, N]` 分块。由于 `sorted_token_ids` 已按专家排序，同一 M-block 内的所有 token 保证分配到同一个专家，B 矩阵访问无分支。

**内层 K 循环**（关键实现）：

```python
# fused_moe.py，第 158–181 行
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    if even_Ks:
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
    else:
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), ...)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, ...)
    accumulator += tl.dot(a, b)
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk

if MUL_ROUTED_WEIGHT:
    moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
    accumulator = accumulator * moe_weight[:, None]  # 乘以路由权重

accumulator = accumulator.to(compute_type)
```

- 在 `float32` 精度下累积，最后转换为 `bfloat16`/`float16`，保证数值稳定；
- `even_Ks` 编译时常量控制是否生成带 mask 的边界检查代码，`K % BLOCK_SIZE_K == 0` 时消除 mask 开销；
- 路由权重乘法（`MUL_ROUTED_WEIGHT`）在写出前融合进 accumulator，避免额外的全局内存读写。

**MoE 计算全流程总结**：

```
输入 tokens (M, K)
    ↓  Router：选出 top_k 个专家及权重
sorted_token_ids / expert_ids  ← 预处理排序（CPU）
    ↓  fused_moe_kernel：token × expert 矩阵乘法
C (M_padded, top_k, N)  ← 各专家输出（已乘路由权重）
    ↓  moe_sum_reduce_kernel：沿 top_k 维求和
输出 (M, N)
```

---

## 附录小结

本附录覆盖了 mini-sglang 三个支撑模块的全部 16 个源文件。核心设计模式总结如下：

| 模式 | 体现位置 | 要点 |
|------|----------|------|
| 三指针流式解码 | `DetokenizeManager` | `surr_offset`/`read_offset`/`sent_offset` 协同处理 UTF-8 字节边界 |
| 动态批次聚合 | `tokenize_worker` | `empty()` 非阻塞检查 + `local_bs` 上界控制 |
| 泛型注册表 | `Registry[T]` | 装饰器注册 + `ArgumentTypeError` 校验 |
| ZMQ PUSH/PULL | `ZmqPushQueue` / `ZmqPullQueue` | msgpack 序列化，同步/异步双版本 |
| NVTX 性能标注 | `nvtx_annotate` | 装饰器 + `layer_id_field` 动态格式化 |
| AOT/JIT 双模编译 | `load_aot` / `load_jit` | TVM FFI 统一接口，`functools.cache` 避免重复编译 |
| Triton 分块 GEMM | `fused_moe_kernel` | 分组排序提升 L2 复用，`even_Ks` 消除边界检查，路由权重融合 |
