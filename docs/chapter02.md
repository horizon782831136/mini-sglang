# 第 2 章：多进程架构与进程间通信

## 2.1 背景问题：GIL 是推理服务的天花板

你刚刚写完一个单进程的 LLM 推理脚本：主循环调用 tokenizer 编码输入，把 token 序列送给模型，等模型跑完再 decode 输出文本。这套逻辑在 Jupyter Notebook 里运行良好，但当你把它包成 HTTP 服务对外开放时，立刻遭遇了瓶颈：

1. **GIL 序列化 CPU 工作**。Python 的全局解释器锁（GIL）确保同一时刻只有一个线程执行 Python 字节码。tokenizer 的编码/解码是纯 CPU 密集操作（正则匹配、BPE 合并），如果和 GPU 推理运行在同一个线程里，二者就会轮流排队，而不是并行。
2. **IO 等待拖慢 GPU 利用率**。FastAPI/uvicorn 在等待网络请求时，推理进程无事可做；而请求真正到来时，GPU 又要等 tokenizer 完成才能开始计算。
3. **多 GPU 需要多进程**。PyTorch 的张量并行（Tensor Parallelism）要求每个 GPU 对应一个独立的 Python 进程（因为 CUDA context 是进程级别的）。用线程做不到这一点。

**核心矛盾**：单进程既要响应 HTTP 请求（IO 密集），又要运行 tokenizer（CPU 密集），还要驱动 GPU 推理（GPU 密集）。这三件事的最优调度策略完全不同，必须拆开。

mini-sglang 的解法是：**把整个系统拆成四类独立进程**，用消息队列串联起来。本章重点讲清楚这个拆分是怎么做的，以及为什么选择 ZMQ 而非 `multiprocessing.Queue`。

---

## 2.2 核心概念：四类进程与消息流

### 2.2.1 进程职责拆分

```
┌─────────────────┐      TokenizeMsg      ┌──────────────────┐
│   API Server    │ ──────────────────►  │    Tokenizer     │
│  (FastAPI +     │                       │  (CPU: encode /  │
│   asyncio)      │ ◄──────────────────  │   decode text)   │
└─────────────────┘      UserReply        └────────┬─────────┘
                                                   │ UserMsg
                                                   ▼
                                          ┌──────────────────┐
                                          │    Scheduler     │
                                          │  (GPU: prefill + │
                                          │   decode loop)   │
                                          └────────┬─────────┘
                                                   │ DetokenizeMsg
                                                   ▼
                                          ┌──────────────────┐
                                          │  Detokenizer     │
                                          │  (CPU: decode    │
                                          │   tokens→text)   │
                                          └──────────────────┘
                                                   │ UserReply
                                                   └──────► API Server
```

| 进程 | 职责 | 关键约束 |
|------|------|----------|
| **API Server** | 接收 HTTP 请求，管理用户会话，流式回包 | asyncio 事件循环，不能阻塞 |
| **Tokenizer** | 文本 → token IDs（`TokenizeMsg` → `UserMsg`） | CPU 密集，可水平扩展 |
| **Scheduler** | 调度 prefill/decode，驱动 GPU 运算 | 每个 GPU 一个进程，持有 CUDA context |
| **Detokenizer** | token ID → 可打印文本（`DetokenizeMsg` → `UserReply`） | 维护每个请求的 decode 状态 |

这个设计有一个重要的合并：默认情况下（`num_tokenizer=0`），Tokenizer 和 Detokenizer 合并为同一个进程（`tokenize_worker`），因为当并发量不大时单进程即可满足需求。当 `--num-tokenizer N` 大于 0 时，tokenizer 和 detokenizer 才会分拆为不同进程。

### 2.2.2 类比：像 PyTorch DataLoader 的多进程预取

如果你用过 `DataLoader(num_workers=4)`，这个设计你一定不陌生：主进程（训练循环）不做数据增强，而是把这件事外包给 worker 进程，通过队列传递处理好的 batch。mini-sglang 做的是完全相同的事，只是"数据预处理"换成了"tokenize/detokenize"，"训练循环"换成了"GPU 推理调度"。

---

## 2.3 消息类型系统

消息系统定义在 `python/minisgl/message/` 目录下，共三个文件分别对应三条通道：

### 2.3.1 API Server → Tokenizer（`tokenizer.py`）

```python
# python/minisgl/message/tokenizer.py

@dataclass
class TokenizeMsg(BaseTokenizerMsg):   # 第 35 行
    uid: int                           # 请求唯一 ID，全局递增
    text: str | List[Dict[str, str]]   # 原始文本或 chat messages
    sampling_params: SamplingParams    # temperature, top_k, max_tokens 等

@dataclass
class DetokenizeMsg(BaseTokenizerMsg): # 第 28 行
    uid: int
    next_token: int                    # 本轮新生成的 token ID
    finished: bool                     # 是否已生成完毕
```

`TokenizeMsg` 由 API Server 发出，`DetokenizeMsg` 由 Scheduler 发出，二者走同一个 Tokenizer 进程的输入队列（`zmq_tokenizer_addr` / `zmq_detokenizer_addr`）。

### 2.3.2 Tokenizer → Scheduler（`backend.py`）

```python
# python/minisgl/message/backend.py

@dataclass
class UserMsg(BaseBackendMsg):         # 第 33 行
    uid: int
    input_ids: torch.Tensor            # CPU 1D int32 tensor，tokenize 结果
    sampling_params: SamplingParams
```

这条消息携带的是"可以直接喂给模型"的 token 序列，Scheduler 收到后放入 prefill 队列。

### 2.3.3 Detokenizer → API Server（`frontend.py`）

```python
# python/minisgl/message/frontend.py

@dataclass
class UserReply(BaseFrontendMsg):      # 第 26 行
    uid: int
    incremental_output: str            # 本轮新增的可打印文本
    finished: bool
```

`incremental_output` 是**增量**而非全量——用户每收到一条就能立刻追加显示，这是流式输出的核心。

### 2.3.4 完整链路一览

```
HTTP Request
    │
    ▼
API Server 创建 uid，构造 TokenizeMsg ──► Tokenizer 进程
                                              │ 调 HuggingFace tokenizer.encode()
                                              │ 构造 UserMsg(input_ids=...)
                                              ▼
                                         Scheduler 进程
                                              │ prefill → decode 循环
                                              │ 每生成一个 token 构造 DetokenizeMsg
                                              ▼
                                         Detokenizer 进程
                                              │ 维护 DecodeStatus，增量 decode
                                              │ 构造 UserReply(incremental_output=...)
                                              ▼
                                         API Server
                                              │ 按 uid 分发给对应的等待协程
                                              ▼
                                        SSE / JSON Response
```

---

## 2.4 ZMQ 消息队列：为什么不用 `multiprocessing.Queue`

### 2.4.1 `multiprocessing.Queue` 的局限

`multiprocessing.Queue` 内部依赖操作系统的管道（pipe）或共享内存，有以下约束：

- **只能在同一台机器内使用**，且通常需要在 `fork` 之前创建。
- **发送端和接收端的进程关系必须已知**：Queue 对象需要在父进程创建后传给子进程。
- **无法做 1-to-N 广播**（PUB/SUB 模式）：如果要把同一条消息发给多个 Scheduler（TP > 1），需要手动循环发送。
- **asyncio 集成麻烦**：`Queue.get()` 是阻塞调用，需要用 `run_in_executor` 包装才能在 asyncio 中使用，性能损耗明显。

### 2.4.2 ZMQ 的优势

ZMQ（ZeroMQ）是一个高性能异步消息库，其 socket 类似于"有缓冲区的管道"，但更灵活：

- **IPC（inter-process communication）地址**：形如 `ipc:///tmp/minisgl_3`，通过本机 Unix domain socket 通信，延迟极低。
- **解耦发送/接收端**：任何一方都可以先启动，ZMQ 会在连接建立后自动传递积压消息。
- **原生支持多种拓扑**：PUSH/PULL（管道）、PUB/SUB（广播）开箱即用。
- **asyncio 原生支持**：`zmq.asyncio` 模块提供协程版本的 `send`/`recv`，API Server 的异步循环可以直接 `await`。

### 2.4.3 mini-sglang 中的 ZMQ 封装

`python/minisgl/utils/mp.py` 提供了四个泛型包装类：

| 类名 | ZMQ Pattern | 用途 |
|------|-------------|------|
| `ZmqPushQueue[T]` | PUSH（同步） | Scheduler → Detokenizer 发 `DetokenizeMsg` |
| `ZmqPullQueue[T]` | PULL（同步） | Tokenizer 收消息，Scheduler 收 `UserMsg` |
| `ZmqAsyncPushQueue[T]` | PUSH（异步） | API Server 发 `TokenizeMsg` |
| `ZmqAsyncPullQueue[T]` | PULL（异步） | API Server 收 `UserReply` |
| `ZmqPubQueue[T]` | PUB | TP rank 0 向其他 rank 广播消息 |
| `ZmqSubQueue[T]` | SUB | TP rank 1+ 订阅来自 rank 0 的广播 |

每个类在构造时接受 `create: bool` 参数：`True` 表示该端 `bind`（创建 socket），`False` 表示 `connect`（连接到已有 socket）。这一设计让各进程的启动顺序无关紧要。

序列化采用 `msgpack`（而非 `pickle`）——msgpack 对二进制数据（如 `torch.Tensor` 的原始字节）处理更高效，且跨语言兼容。`torch.Tensor` 通过 `.numpy().tobytes()` 转换为字节串传输（见 `message/utils.py` 第 27-29 行）。

---

## 2.5 `launch_server()` 启动流程逐行解读

文件：`python/minisgl/server/launch.py`

```python
def launch_server(run_shell: bool = False) -> None:      # 第 40 行
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)

    def start_subprocess() -> None:
        mp.set_start_method("spawn", force=True)          # 第 52 行
        ...
```

**第 52 行** 强制使用 `spawn` 模式。这是关键决策：

- `fork` 会继承父进程的所有内存，包括 CUDA 状态——多 GPU 场景下这会导致 CUDA context 混乱。
- `spawn` 从零启动新 Python 解释器，干净但稍慢（约 1-2 秒），适合长期运行的服务进程。

```python
        world_size = server_args.tp_info.size
        ack_queue: mp.Queue[str] = mp.Queue()             # 第 57 行

        for i in range(world_size):
            new_args = replace(server_args, tp_info=DistributedInfo(i, world_size))
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                name=f"minisgl-TP{i}-scheduler",
            ).start()                                     # 第 64-69 行
```

**第 57 行** 创建一个普通的 `multiprocessing.Queue`，专门用来收集各子进程的"就绪确认"信号（ACK）。这是 `mp.Queue` 在这里的唯一用途——它不传递业务数据，只做简单的启动同步信号。这是一个很好的分工：ZMQ 负责高频业务通信，`mp.Queue` 负责一次性的启动握手。

```python
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "addr": server_args.zmq_detokenizer_addr,
                ...
                "tokenizer_id": num_tokenizers,           # detokenizer 的 id
            },
            name="minisgl-detokenizer-0",
        ).start()                                         # 第 73-87 行

        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={"addr": server_args.zmq_tokenizer_addr, ...},
                name=f"minisgl-tokenizer-{i}",
            ).start()                                     # 第 88-103 行
```

注意 tokenizer 和 detokenizer 调用的是**同一个函数** `tokenize_worker`，区别仅在于监听的 ZMQ 地址不同（`zmq_tokenizer_addr` vs `zmq_detokenizer_addr`）。Scheduler 会把 `DetokenizeMsg` 发到 detokenizer 地址，把包含 `TokenizeMsg` 转换后的 `UserMsg` 送到 backend 地址。

```python
        for _ in range(num_tokenizers + 2):              # 第 110-111 行
            logger.info(ack_queue.get())                 # 阻塞等待所有进程就绪
```

这里阻塞等待 `num_tokenizers + 2` 条 ACK 消息（1 个 Scheduler 主 rank + num_tokenizers 个 tokenizer + 1 个 detokenizer）。只有当所有进程都在 ZMQ 上 `bind` 完毕并发出 ACK，API Server 才开始监听端口，确保不丢失第一条请求。

```python
    run_api_server(server_args, start_subprocess, run_shell=run_shell)  # 第 113 行
```

`run_api_server` 在 `api_server.py` 中定义，它先初始化 `FrontendManager`（建立 ZMQ 连接），然后调用 `start_subprocess()` 回调启动所有子进程，最后才启动 uvicorn 监听 HTTP 端口。

---

## 2.6 设计决策

### 为什么 API Server 用异步，其他进程用同步？

API Server 的工作是**等待**——等 HTTP 请求、等 ZMQ 消息、等 SSE 写入完成。这些都是 IO 等待，正好适合 `asyncio` 的协程模型：一个线程可以同时"等"数千个请求。

Scheduler 的工作是**计算**——每一轮 loop 都要做内存分配、batch 准备、GPU forward。没有 IO 等待，同步阻塞调用反而更直接，不需要 `await` 带来的调度开销。

### 为什么 `UserMsg` 携带 `torch.Tensor` 而非 Python list？

Scheduler 拿到 `input_ids` 后要直接做 `tensor.to(device)`。如果用 list 传输，还需要在 Scheduler 进程里做一次 `torch.tensor(list)`，多一次内存拷贝。直接序列化为 `tobytes()` 然后 `frombuffer()` 重建，可以做到零额外拷贝（`torch.from_numpy`）。

### 替代方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| 单进程多线程 | 实现最简单 | GIL 限制 CPU 并行，无法多 GPU |
| `multiprocessing.Queue` | Python 标准库，无依赖 | 无广播、asyncio 集成差、需要 fork 前创建 |
| gRPC | 跨机器，类型安全 | 引入 proto 编译步骤，延迟高于 IPC |
| **ZMQ + msgpack（当前方案）** | 低延迟、灵活拓扑、asyncio 原生 | 多一个外部依赖 |
| Ray | 分布式调度全托管 | 重量级，调试困难，不适合教学 |

---

## 2.7 动手实验

### 实验一：观察进程树

启动服务后，运行以下命令观察进程层级：

```bash
# 在另一个终端启动服务（用 dummy weight 快速启动）
python -m minisgl --model Qwen/Qwen2.5-0.5B-Instruct --dummy-weight &

# 查看进程树
pstree -p $(pgrep -f "minisgl") 2>/dev/null || ps -ef | grep minisgl
```

你应该看到类似以下结构：
```
python (API Server, 主进程)
├── python (minisgl-TP0-scheduler)
├── python (minisgl-detokenizer-0)
└── python (minisgl-tokenizer-0)  # 仅在 --num-tokenizer > 0 时出现
```

试着用 `--num-tokenizer 2` 重启，观察进程数量变化。

### 实验二：追踪消息流

在 `python/minisgl/tokenizer/server.py` 第 61 行前后加入打印，观察消息批量合并行为：

```python
# server.py 第 61 行附近（tokenize_worker 函数内）
pending_msg = _unwrap_msg(recv_listener.get())
print(f"[DEBUG] 初始收到 {len(pending_msg)} 条消息")
while len(pending_msg) < local_bs and not recv_listener.empty():
    pending_msg.extend(_unwrap_msg(recv_listener.get()))
print(f"[DEBUG] 批量后共 {len(pending_msg)} 条消息")
```

然后用并发脚本发送多条请求：

```python
import asyncio, aiohttp

async def send_one(session, i):
    async with session.post("http://localhost:1919/generate",
                            json={"prompt": f"Hello {i}", "max_tokens": 5}) as r:
        return await r.json()

async def main():
    async with aiohttp.ClientSession() as s:
        tasks = [send_one(s, i) for i in range(10)]
        results = await asyncio.gather(*tasks)
    print(results)

asyncio.run(main())
```

观察服务端日志中批量大小的变化，理解 `local_bs` 参数的作用。

### 实验三（进阶）：测量 ZMQ vs mp.Queue 延迟

编写一个微基准测试，对比在同一台机器上：
- `multiprocessing.Queue` 的单次 `put/get` 延迟
- `ZmqPushQueue/ZmqPullQueue`（`ipc://` 地址）的单次 `put/get` 延迟

```python
import time, multiprocessing as mp
import zmq, msgpack

def bench_mpqueue(n=10000):
    q = mp.Queue()
    data = b"hello world" * 100
    t0 = time.perf_counter()
    for _ in range(n):
        q.put(data)
        q.get()
    return (time.perf_counter() - t0) / n * 1e6  # 微秒/次

def bench_zmq(n=10000):
    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.bind("ipc:///tmp/bench_push")
    pull = ctx.socket(zmq.PULL)
    pull.connect("ipc:///tmp/bench_push")
    data = msgpack.packb(b"hello world" * 100)
    t0 = time.perf_counter()
    for _ in range(n):
        push.send(data)
        pull.recv()
    return (time.perf_counter() - t0) / n * 1e6

print(f"mp.Queue:  {bench_mpqueue():.1f} μs/round-trip")
print(f"ZMQ IPC:   {bench_zmq():.1f} μs/round-trip")
```

参考结果：ZMQ IPC 通常比 `mp.Queue` 快 2-5 倍，且在高并发场景下差距更明显。

---

## 2.8 小结

本章要点：

1. **GIL + CUDA context + IO 阻塞**三重约束，共同决定了 LLM 推理服务必须采用多进程架构。

2. mini-sglang 将系统拆分为四类进程（API Server、Tokenizer、Scheduler、Detokenizer），每类进程只做一件事，职责清晰。

3. **消息类型体系**构成进程间的"API 契约"：`TokenizeMsg` → `UserMsg` → `DetokenizeMsg` → `UserReply` 是一条完整的请求生命周期链路。每条消息都携带 `uid`，使得无状态的队列传输中仍然能追踪到具体的用户请求。

4. **ZMQ 的核心优势**在于：解耦启动顺序、原生 asyncio 支持、PUB/SUB 广播满足多 GPU 场景，以及低延迟的 IPC 传输。

5. `launch_server()` 中用 `mp.Queue` 做一次性的启动 ACK，用 ZMQ 做持续的业务通信，这是"对的工具做对的事"的典型示范。

**与后续章节的连接**：Scheduler 收到 `UserMsg` 后，需要决定何时调度这条请求、分配多少 KV Cache 页面——这就是第 3 章（调度器与 KV Cache 管理）要解决的问题。而 Scheduler 中的 GPU forward 部分，则是第 4 章（推理引擎与注意力后端）的核心内容。
