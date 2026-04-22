# 第 3 章：API Server 与 OpenAI 兼容接口

## 3.1 背景问题

在深度学习研究阶段，我们习惯于这样调用模型：

```python
outputs = model.generate(input_ids, max_new_tokens=128)
```

这行代码简洁直接，但它有一个致命缺陷：**它是阻塞同步的**。在推理服务场景下，同时会有数十乃至数百个用户的请求并发到达。如果对每个请求都串行地等待 `model.generate()` 返回，那么第 100 个用户就要排队等前面 99 个人全部完成。即使单次推理只需 2 秒，第 100 位用户也要等待约 200 秒。

更棘手的问题来自 **流式输出（Streaming）**。大语言模型是逐 token 生成的，用户期望看到文字逐字出现的效果，而不是等待十几秒后一次性看到完整答案。`model.generate()` 必须等所有 token 生成完毕才返回，根本无法做到流式。

还有一个工程现实问题：**生态兼容性**。OpenAI 的 `/v1/chat/completions` 接口已经成为业界事实标准，绝大多数客户端工具（LangChain、OpenWebUI、各类 SDK）都内置了对这个接口的支持。如果自研推理服务不兼容这个格式，就意味着用户要重写所有接入代码。

本章的核心矛盾因此浮现：**如何在高并发场景下，同时支持流式输出和 OpenAI 兼容格式？** 答案是异步 HTTP 服务。

---

## 3.2 核心概念讲解

### 3.2.1 async/await：协程与事件循环

如果你写过 PyTorch 的训练代码，你的思维模型是**同步阻塞**的：调用 `forward()`，等待 GPU 计算完成，取结果。

`async/await` 引入了另一种思维模型：**协程（Coroutine）**。把每个请求想象成一根正在编织的毛线。同步代码只有一根针，一次只能编一根线；而 `asyncio` 的事件循环是一个熟练的织工，手里同时握着很多根针，哪根不需要等就先去推进它。

关键词是"不需要等"——即 I/O 等待。当一个协程在等待网络 I/O（接收请求、发送响应）或等待来自后端的推理结果时，它会主动让出控制权（`await`），事件循环趁机去推进其他协程。CPU 计算密集的部分（实际的模型推理）则运行在独立的后端进程中，不会阻塞事件循环。

这套机制使得 FastAPI + uvicorn 可以用**单线程**高效处理大量并发 HTTP 请求。

### 3.2.2 Server-Sent Events（SSE）：流式输出的 HTTP 协议

SSE 是 HTTP 协议的一种扩展模式：服务端把响应头设置为 `Content-Type: text/event-stream`，然后持续往连接里写数据，而不是一次性返回后关闭连接。每条消息格式为：

```
data: {"text": "Hello"}\n\n
data: {"text": " world"}\n\n
data: [DONE]\n\n
```

客户端收到 `[DONE]` 后知道流结束。这比 WebSocket 轻量很多，单方向推送场景首选 SSE。

### 3.2.3 OpenAI 接口格式的意义

OpenAI 的 Chat Completions 接口有固定的请求和响应格式。流式响应的每个 chunk 长这样：

```json
{
  "id": "cmpl-123",
  "object": "text_completion.chunk",
  "choices": [{"delta": {"content": "Hello"}, "index": 0, "finish_reason": null}]
}
```

非流式响应的完整格式为：

```json
{
  "id": "cmpl-123",
  "object": "chat.completion",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
}
```

兼容这个格式，意味着 mini-sglang 可以无缝替换 OpenAI 的 API 端点，接入任何已有的客户端生态。

### 3.2.4 进程间通信：ZMQ 管道

API Server（前端）运行在主进程的 asyncio 事件循环中，而模型推理运行在独立的 GPU 进程中。两者通过 ZeroMQ（ZMQ）的 IPC socket 通信。前端发送 `TokenizeMsg`（带着 uid 和采样参数），后端处理完每个 token 后回送 `UserReply`（带着 uid 和增量输出）。uid 是贯穿整个链路的请求标识符。

---

## 3.3 核心代码导读

核心文件：`python/minisgl/server/api_server.py`

### 3.3.1 FrontendManager：请求追踪的三件套

`FrontendManager`（第 100–224 行）是整个前端的大脑。它维护了三个核心数据结构：

```python
# api_server.py, 第 105–108 行
uid_counter: int = 0
ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
```

**uid_counter**：单调递增的请求 ID 生成器，每次 `new_user()` 调用后加 1。uid 是整个系统中追踪一次请求的唯一钥匙。

**ack_map**：`uid -> List[UserReply]` 的字典，缓存后端已经回传但协程还没来得及读取的 token。可以把它理解为每个请求的"收件箱"。

**event_map**：`uid -> asyncio.Event` 的字典。当后端送来新 token 时，对应的 `Event` 被 `set()`，唤醒正在 `await event.wait()` 的协程去消费收件箱。

`new_user()` 方法（第 110–115 行）展示了这三件套如何初始化：

```python
def new_user(self) -> int:
    uid = self.uid_counter
    self.uid_counter += 1
    self.ack_map[uid] = []
    self.event_map[uid] = asyncio.Event()
    return uid
```

### 3.3.2 listen()：单一后台任务收取所有回包

`listen()`（第 117–124 行）是一个永久运行的后台协程，负责从 ZMQ 管道读取后端回包，并按 uid 分拣到各自的 `ack_map` 里：

```python
async def listen(self):
    while True:
        msg = await self.recv_tokenizer.get()
        for msg in _unwrap_msg(msg):
            if msg.uid not in self.ack_map:
                continue
            self.ack_map[msg.uid].append(msg)
            self.event_map[msg.uid].set()
```

注意这里的设计：整个前端**只有一个** `listen()` 协程在读 ZMQ，然后把结果派发给各自等待的请求协程。这避免了多个协程同时竞争读取同一管道带来的复杂性。`_create_listener_once()`（第 126–129 行）用 `initialized` 标志保证这个后台任务只被创建一次。

### 3.3.3 wait_for_ack()：异步生成器驱动流水线

`wait_for_ack()`（第 135–151 行）是一个异步生成器（`async for` + `yield`），它把"等待 Event → 消费 ack_map → 清理"的循环封装成干净的迭代接口：

```python
async def wait_for_ack(self, uid: int):
    event = self.event_map[uid]
    while True:
        await event.wait()   # 挂起，等 listen() 唤醒
        event.clear()
        pending = self.ack_map[uid]
        self.ack_map[uid] = []
        ack = None
        for ack in pending:
            yield ack        # 逐个 yield 给调用方
        if ack and ack.finished:
            break
    del self.ack_map[uid]
    del self.event_map[uid]
```

`event.wait()` 是这里的关键：它不占用 CPU，只是让当前协程挂起，让出事件循环去处理其他请求，直到 `listen()` 调用 `event.set()` 将其唤醒。

### 3.3.4 流式输出：stream_chat_completions()

`stream_chat_completions()`（第 162–190 行）把后端的增量 token 包装成 OpenAI 格式的 SSE chunk：

```python
async def stream_chat_completions(self, uid: int):
    first_chunk = True
    async for ack in self.wait_for_ack(uid):
        delta = {}
        if first_chunk:
            delta["role"] = "assistant"
            first_chunk = False
        if ack.incremental_output:
            delta["content"] = ack.incremental_output
        chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
        if ack.finished:
            break
    # 最后发送带 finish_reason 的终止 chunk
    end_chunk = {..., "choices": [{"delta": {}, "finish_reason": "stop"}]}
    yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n".encode()
    yield b"data: [DONE]\n\n"
```

第一个 chunk 特殊处理，在 `delta` 中附带 `"role": "assistant"`，这是 OpenAI 流式协议的规范要求。最后额外发送一个 `finish_reason: "stop"` 的终止 chunk，再跟一个 `[DONE]` 标记。

### 3.3.5 非流式输出：collect_full_output()

非流式场景（第 192–199 行）更直接：消费所有 token，拼接成完整字符串，等 `finished=True` 后一次性返回：

```python
async def collect_full_output(self, uid: int):
    full_text = ""
    async for ack in self.wait_for_ack(uid):
        full_text += ack.incremental_output
        if ack.finished:
            break
    return full_text
```

### 3.3.6 客户端断连检测与 Abort

`stream_with_cancellation()`（第 201–211 行）在每次 yield 之前检查客户端是否已断开：

```python
async def stream_with_cancellation(self, generator, request: Request, uid: int):
    try:
        async for chunk in generator:
            if await request.is_disconnected():
                raise asyncio.CancelledError
            yield chunk
    except asyncio.CancelledError:
        asyncio.create_task(self.abort_user(uid))
        raise
```

断连后，`abort_user()`（第 213–220 行）会先清理内存中的 `ack_map`/`event_map` 条目，再向后端发送 `AbortMsg`，通知调度器停止对该 uid 的推理，节省 GPU 算力。

### 3.3.7 路由层：v1_completions()

`/v1/chat/completions` 路由（第 264–308 行）是 HTTP 请求的入口，逻辑清晰：

1. 解析请求，将 `messages` 列表或 `prompt` 字符串提取出来
2. 调用 `state.new_user()` 分配 uid
3. 通过 ZMQ 发送 `TokenizeMsg`
4. 根据 `req.stream` 决定返回 `StreamingResponse` 还是等待 `collect_full_output()` 后返回 `JSONResponse`

---

## 3.4 设计决策

### 为什么用"单 listen 任务 + Event 唤醒"而不是"每请求一个读协程"？

如果让每个请求各自阻塞在 `recv_tokenizer.get()` 上，ZMQ socket 就需要支持并发读，或者为每个请求开一个独立 socket，这会带来连接管理的复杂性和大量的 socket 资源消耗。

当前方案只有一根 ZMQ 管道，由 `listen()` 独占读取，然后用 Python 字典做内存内分发。分发是纯内存操作，几乎零开销。`asyncio.Event` 的 `set()`/`wait()` 也是无锁的协程原语，非常轻量。

### 为什么 wait_for_ack() 要在 event.wait() 后做 batch 消费？

两次 `await event.wait()` 之间，`listen()` 可能已经往 `ack_map[uid]` 放了多个 `UserReply`（后端 batch 处理多个 token 一起回传的场景）。用 `pending = self.ack_map[uid]; self.ack_map[uid] = []` 的 swap 模式，可以一次性消费所有积压的 token，减少协程切换次数，提升吞吐。

### 为什么 abort_user() 要先 sleep(0.1)？

```python
async def abort_user(self, uid: int):
    await asyncio.sleep(0.1)  # api_server.py, 第 214 行
    ...
```

`abort_user` 是通过 `asyncio.create_task()` 在后台调度的。在它真正执行时，`stream_with_cancellation` 可能还在异常处理栈中，仍有协程持有 `ack_map[uid]` 的引用。0.1 秒的小延迟是一个"软栅栏"，避免在清理映射表时发生竞争。这是异步代码中常见的"让出一拍"技巧。

### 替代方案：同步接口 + 线程池

Flask/Django + ThreadPoolExecutor 也能处理并发请求，每个请求占用一个线程，用阻塞 I/O 等待结果。这种方案更简单，但线程的内存开销是协程的数十倍（每线程约 8MB 栈），在高并发下资源消耗显著。协程的理论并发量远高于线程池方案。

---

## 3.5 动手实验

### 环境准备

启动 mini-sglang 服务器（使用虚拟权重快速测试）：

```bash
python -m minisgl.server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --dummy-weight \
    --port 8080
```

### 实验一：非流式请求

验证 OpenAI 兼容的非流式接口：

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "max_tokens": 32,
    "stream": false
  }' | python3 -m json.tool
```

预期返回 JSON，包含 `choices[0].message.content` 字段。

### 实验二：流式请求

使用 `-N`（关闭 curl 缓冲）观察 SSE 逐行输出：

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "Count from 1 to 5"}],
    "max_tokens": 64,
    "stream": true
  }'
```

你应该看到一系列 `data: {...}` 行逐行出现，最后以 `data: [DONE]` 结束。

### 实验三：查询模型列表

```bash
curl http://localhost:8080/v1/models | python3 -m json.tool
```

### 实验四：原始 generate 接口

```bash
curl -s http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 8}' | python3 -m json.tool
```

这个接口比 OpenAI 格式更简单，直接返回 `{"text": "..."}` 字符串。

### 进阶实验：模拟客户端断连

在一个终端启动流式请求，然后在 2 秒内用 `Ctrl+C` 中断：

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "Tell me a very long story"}], "max_tokens": 512, "stream": true}'
# 按 Ctrl+C 中断
```

观察服务器日志，应能看到类似 `Client disconnected for user X` 和 `Aborting request for user X` 的日志，证明 Abort 机制生效。

### 进阶实验：使用 OpenAI Python SDK 接入

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:8080/v1"
)

# 流式调用
stream = client.chat.completions.create(
    model="test",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=32,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

这证明了 mini-sglang 可以被任何兼容 OpenAI SDK 的工具直接使用。

---

## 3.6 小结

本章剖析了 mini-sglang API Server 的设计：

| 核心问题 | 解决方案 |
|---|---|
| 高并发 HTTP 请求处理 | FastAPI + asyncio 事件循环，单线程协程并发 |
| 流式输出 | SSE（Server-Sent Events）+ 异步生成器 |
| 生态兼容 | 实现 OpenAI `/v1/chat/completions` 格式 |
| 请求追踪 | uid + ack_map + event_map 三件套 |
| 客户端断连 | `is_disconnected()` 检测 + AbortMsg 传播 |

**核心数据流**总结：HTTP 请求进入 → `new_user()` 分配 uid → `TokenizeMsg` 经 ZMQ 发往 tokenizer → 后端推理逐 token 回送 `UserReply` → `listen()` 分拣到 `ack_map` → `Event.set()` 唤醒对应请求协程 → 流式或非流式组装响应 → 客户端收到结果。

**与后续章节的连接**：本章我们把前端当作"黑盒接收者"来看待后端的 `UserReply`。第 4 章将打开这个黑盒，深入 Tokenizer/Detokenizer 进程，理解文本如何被转化为 token ID，以及推理结果如何从 token ID 还原为可读文字。
