# 第 1 章：从训练到推理——系统视角的转变

> **面向读者**：熟悉 PyTorch 训练流程、了解 Transformer 基本结构，但尚未接触推理系统工程的读者。

---

## 1.1 背景问题：训练的直觉为什么在推理时失效

你已经熟悉用 PyTorch 训练一个语言模型：准备好数据集，固定 batch size，循环调用 `loss.backward()` 和 `optimizer.step()`。整个过程是**批量的、离线的、静态的**——所有样本提前准备好，批次大小固定不变，每步执行完整的前向 + 反向传播。

现在来到推理场景。你的第一反应可能是：

```python
# 朴素的推理循环
for request in incoming_requests:
    output = model.generate(request.prompt, max_new_tokens=256)
    send_reply(output)
```

这段代码能跑，但在生产环境中会有严重的性能问题。为什么？因为训练时你可以假设"所有样本同时到达"，而推理时**请求是动态到来的**，而且生成过程有一个训练中不存在的特殊性：**自回归逐 token 生成**。

本章的核心矛盾可以用一句话概括：

> **训练追求最大化 GPU 利用率（大 batch 摊薄开销），推理则需要同时保证低延迟（单个请求快速完成）和高吞吐量（同时服务尽量多的请求）。这两个目标天然冲突。**

---

## 1.2 核心概念讲解

### 1.2.1 自回归生成的特殊性

训练时，模型对一个长度为 $L$ 的序列只做**一次前向传播**，损失是对所有位置的预测误差求和。推理时，生成长度为 $N$ 的输出需要做 $N$ 次前向传播，每次只生成**一个新 token**。

类比：训练就像老师批改整张试卷，推理就像在电话里一字一字地口述答案——每说一个字都要等对方确认再继续。

这带来一个关键问题：第 $t$ 步生成时，模型已经计算过位置 $1$ 到 $t-1$ 的注意力键值对（Key-Value，即 KV），如果每步都重新计算，代价是 $O(N^2)$；如果把之前的 KV 结果缓存起来（**KV Cache**），每步只需计算新 token 的 KV，代价降为 $O(N)$。

**KV Cache 是推理系统与训练系统最大的结构性差异之一。** 训练不需要这个概念——梯度反传时所有激活值已经保存在内存中；推理中 KV Cache 的分配、复用、回收是整个系统设计的核心。

### 1.2.2 Prefill 与 Decode：两种截然不同的计算模式

推理的一次完整服务流程分为两个阶段：

| 阶段 | 输入 | 计算特征 | 硬件瓶颈 |
|------|------|----------|----------|
| **Prefill（预填充）** | 用户完整的 prompt（数百到数千 token） | 对所有 prompt token 并行计算，类似训练的前向传播 | 计算密集（compute-bound） |
| **Decode（解码）** | 上一步生成的单个 token | 每步只处理 1 个新 token，读取巨量 KV Cache | 内存带宽密集（memory-bound） |

Prefill 阶段像训练——大量 token 并行计算，GPU 利用率高。Decode 阶段非常不同——每步只有一个 token，但需要从 GPU 内存读取整个模型和 KV Cache，计算量极低，GPU 大部分时间在等内存数据。这就是为什么"生成一个 token 的延迟"远比你直觉预期的长。

### 1.2.3 吞吐量与延迟的根本矛盾

为什么不能用大 batch 来提升吞吐量？

回到那段朴素代码——如果把 1000 个请求攒成一批再处理，GPU 利用率会很高，但每个请求都要等 999 个其他请求到齐才开始，**首 token 延迟（Time to First Token, TTFT）** 极高。

反过来，如果逐条处理，每个请求延迟最小，但 GPU 几乎一直在做 memory-bound 的 decode，吞吐量极低。

**解决方案是 continuous batching（连续批处理）**：不等待凑够固定 batch，而是把所有正在执行 decode 的请求聚合在一起，每个调度周期内同时推进它们的生成，同时允许新请求随时加入 prefill 阶段。这样 GPU 每一步都在处理尽量多的请求，同时不让任何单个请求等待时间过长。

Mini-SGLang 的调度器正是基于这种思想构建的——后续章节会深入分析。

### 1.2.4 KV Cache 的内存压力

KV Cache 占用的显存与批次大小、序列长度成正比。对于一个 7B 参数模型，典型配置下每个 token 的 KV Cache 约占 0.5MB（取决于层数、头数、精度）。若同时服务 100 个长度为 2048 的请求，KV Cache 约占 100GB，远超单卡显存。

这迫使推理系统必须像操作系统管理内存页一样管理 KV Cache：**分页（paging）**、**按需分配**、**前缀共享复用**。这些概念在训练中完全不存在。

---

## 1.3 核心代码导读

### 1.3.1 项目入口：`__main__.py`

```
python/minisgl/__main__.py
```

整个服务端的入口只有两行（第 1-5 行）：

```python
from .server import launch_server

assert __name__ == "__main__"
launch_server()
```

`python -m minisgl` 就等价于调用 `launch_server()`。这一行代码背后隐藏着一个多进程架构的启动序列。

### 1.3.2 核心数据抽象：`core.py`

文件路径：`python/minisgl/core.py`

推理系统的状态管理比训练复杂得多。Mini-SGLang 用四个核心数据类表示推理运行时的全部状态：

**`SamplingParams`（第 16-25 行）**：采样参数，封装 temperature、top_k、top_p 等。注意第 23-25 行的 `is_greedy` 属性——贪心解码是一条重要的快速路径，许多系统优化（如 CUDA Graph）只针对贪心解码生效。

**`Req`（第 29-68 行）**：单个推理请求的完整状态。关键字段含义：

```python
@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # 在 CPU 上保存的完整 token 序列（包含已生成的 token）
    table_idx: int            # 在 page_table 中的行索引，用于定位 KV Cache
    cached_len: int           # 已经写入 KV Cache 的 token 数
    device_len: int           # 当前序列总长度（= cached_len + 本轮新 token）
    max_device_len: int       # 最大序列长度（= input_len + max_output_tokens）
```

`cached_len` 和 `device_len` 的差值 `extend_len`（第 48-50 行）正是本轮 prefill 需要处理的 token 数——这是 chunked prefill 的关键度量。

**`Batch`（第 72-98 行）**：一批请求的集合，关键字段 `phase`（第 74 行）区分当前是 prefill 还是 decode 阶段。`attn_metadata`（第 81 行）由注意力后端填写，携带 FlashAttention 或 FlashInfer 所需的元数据，不同阶段的元数据格式完全不同。

**`Context`（第 101-122 行）**：全局推理上下文，持有模型、KV Cache 池、注意力后端等的引用。`forward_batch` 是一个上下文管理器（第 115-122 行），在前向传播期间将当前 `Batch` 注入全局状态，让模型层可以直接读取 KV Cache 地址，而无需层层传参。

第 125-136 行的全局 `_GLOBAL_CTX` 是一个单例——每个 GPU worker 进程只有一个推理上下文，这避免了层间传参的复杂性。

### 1.3.3 多进程启动：`server/launch.py`

文件路径：`python/minisgl/server/launch.py`

`launch_server()` 函数（第 40-113 行）揭示了整个系统的进程拓扑：

```python
# 启动 world_size 个 Scheduler 进程（每个 GPU 一个）
for i in range(world_size):
    mp.Process(
        target=_run_scheduler,
        args=(new_args, ack_queue),
        name=f"minisgl-TP{i}-scheduler",
    ).start()

# 启动 1 个 Detokenizer 进程
mp.Process(target=tokenize_worker, ..., name="minisgl-detokenizer-0").start()

# 启动若干 Tokenizer 进程
for i in range(num_tokenizers):
    mp.Process(target=tokenize_worker, ..., name=f"minisgl-tokenizer-{i}").start()
```

类比训练：如果说训练中 `DataLoader` 的多个 worker 负责数据预处理，那么推理中 Tokenizer worker 就类似于这个角色——异步地把文本转换成 token，不阻塞 GPU 计算。

注意第 52 行的 `mp.set_start_method("spawn", force=True)`——使用 `spawn` 而非 `fork` 是为了让每个子进程干净地初始化 CUDA 上下文，避免 fork + CUDA 的已知问题。

第 105-111 行使用 `ack_queue` 等待所有子进程就绪，确保在 API 服务开始接收请求前，所有后端进程已经完成初始化（包括模型权重加载）。

### 1.3.4 调度器主循环：`scheduler/scheduler.py`

文件路径：`python/minisgl/scheduler/scheduler.py`

`run_forever()` 方法（第 121-131 行）是推理系统的心跳：

```python
def run_forever(self) -> NoReturn:
    if ENV.DISABLE_OVERLAP_SCHEDULING:
        while True:
            self.normal_loop()     # 顺序执行：调度 -> 前向 -> 后处理
    else:
        data = None
        while True:
            data = self.overlap_loop(data)  # 重叠执行：上一批后处理 || 本批前向
```

`overlap_loop`（第 83-106 行）是性能关键路径：它让当前 batch 的 GPU 前向计算与上一 batch 的 CPU 后处理（token 采样、结果发送）**并行执行**，隐藏 CPU 开销，提升 GPU 利用率。这个技术类似于训练中的 gradient checkpointing——通过精心的流水线安排，避免 GPU 等待 CPU。

---

## 1.4 设计决策

### 1.4.1 为什么选择多进程而非多线程？

Python 有 GIL（全局解释器锁），多线程无法真正并行执行 CPU 密集任务。推理系统中 Tokenizer、Scheduler、API Server 各有大量 CPU 工作，必须用多进程才能真正并行。

代价是进程间通信（IPC）开销。Mini-SGLang 选择 ZMQ（ZeroMQ）作为 IPC 通道（见 `server/args.py` 中的 `zmq_*_addr` 属性），使用 IPC socket（`ipc://` 前缀）在同机进程间传递序列化消息，延迟约几十微秒，远低于网络 TCP。

### 1.4.2 为什么 `input_ids` 保存在 CPU 而非 GPU？

`Req.input_ids`（`core.py` 第 31 行）是 CPU tensor。这是一个有意识的设计：

- GPU 显存宝贵，应留给模型权重和 KV Cache
- 调度决策（选择哪些 token 做 prefill、管理 chunked prefill 进度）在 CPU 上完成，数据在 CPU 更自然
- token 序列最终通过 `token_pool`（一块 pinned memory 或 GPU tensor）批量上传，而非逐请求上传

### 1.4.3 分页 KV Cache vs. 连续 KV Cache

最朴素的实现是为每个请求预分配 `max_seq_len` 长度的连续 KV Cache。缺点显而易见：若 `max_seq_len = 4096`，即使请求只生成了 100 个 token，也要占用 4096 token 的显存，利用率极低。

分页方案（`page_size` 可配置，见 `server/args.py` 第 185 行）将显存切分为固定大小的页，按需分配，用完即回收。代价是需要维护 `page_table`（一个从请求到物理内存页的映射表），以及注意力计算时需要支持非连续内存访问（这正是 FlashInfer 等专用内核存在的原因）。

---

## 1.5 动手实验

### 实验一：环境搭建与第一次运行

**前提条件**：Linux 系统、NVIDIA GPU、CUDA Toolkit 已安装。

```bash
# 克隆仓库
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang

# 创建虚拟环境（推荐 Python 3.12）
uv venv --python=3.12
source .venv/bin/activate

# 安装（-e 表示可编辑模式，便于后续修改代码）
uv pip install -e .

# 启动服务（首次运行会从 HuggingFace 下载模型，约 1-2GB）
python -m minisgl --model "Qwen/Qwen3-0.6B"
```

服务启动后，你会看到类似输出：
```
[initializer] API server is ready to serve on 127.0.0.1:1919
```

**发送第一个请求**：

```bash
# 方式一：使用 curl（非流式）
curl -X POST http://127.0.0.1:1919/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 20}'

# 方式二：使用 OpenAI 兼容接口（流式）
curl -X POST http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "用一句话介绍巴黎"}],
    "max_tokens": 50,
    "stream": true
  }'
```

**交互式 Shell 模式**（无需单独发 HTTP 请求）：

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

### 实验二：观察进程拓扑

服务启动后，在另一个终端观察进程结构：

```bash
# 查看所有 minisgl 相关进程
ps aux | grep minisgl

# 预期输出示例（单卡场景）：
# minisgl-TP0-scheduler   <- GPU 调度器（含推理引擎）
# minisgl-detokenizer-0   <- 反分词进程
# python -m minisgl       <- 主进程（API Server）
```

这直观地展示了 1.3.3 节描述的多进程架构。

### 实验三：对比 overlap scheduling 的性能差异

Mini-SGLang 支持通过环境变量关闭 overlap scheduling，便于对比：

```bash
# 准备一个简单的基准脚本 bench_simple.py
cat > /tmp/bench_simple.py << 'EOF'
import time
import requests

prompts = ["Tell me about " + topic for topic in [
    "quantum computing", "deep learning", "the French Revolution",
    "climate change", "the Renaissance", "black holes",
    "machine translation", "protein folding", "ancient Rome", "jazz music"
] * 5]  # 50 个请求

start = time.time()
for p in prompts:
    requests.post("http://127.0.0.1:1919/generate",
                  json={"prompt": p, "max_tokens": 100})
elapsed = time.time() - start
print(f"Total: {elapsed:.2f}s, Throughput: {len(prompts)/elapsed:.1f} req/s")
EOF

# 启用 overlap scheduling（默认）
python -m minisgl --model "Qwen/Qwen3-0.6B" &
sleep 30 && python /tmp/bench_simple.py
pkill -f minisgl

# 关闭 overlap scheduling
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl --model "Qwen/Qwen3-0.6B" &
sleep 30 && python /tmp/bench_simple.py
pkill -f minisgl
```

### 实验四（进阶）：用 Python 直接观察 KV Cache 分配

阅读 `python/minisgl/scheduler/cache.py` 和 `python/minisgl/kvcache/` 下的代码，思考以下问题：

1. `page_size` 默认是多少？当一个请求生成 100 个 token 时，会分配几个 KV Cache 页？
2. 修改启动参数 `--page-size 1` 和 `--page-size 16`，观察内存占用变化（可用 `nvidia-smi` 监控）。
3. 阅读 `core.py` 中 `Req.extend_len` 的定义（第 48-50 行），它在 chunked prefill 中扮演什么角色？

---

## 1.6 小结

本章建立了从训练到推理的系统视角转变，核心要点如下：

| 维度 | 训练 | 推理 |
|------|------|------|
| 请求到达方式 | 批量、离线、静态 | 动态、在线、流式 |
| 计算单位 | 整个 batch 一次前向 + 反向 | 逐 token 自回归，prefill + N 步 decode |
| 内存管理重点 | 梯度、激活值（反传后即释放） | KV Cache（需在整个生成过程中保持） |
| 主要瓶颈 | 计算（FLOPS） | 吞吐与延迟的折中；KV Cache 的显存与带宽 |
| 批处理策略 | 固定 batch | Continuous batching（动态聚合） |

Mini-SGLang 的架构回应了上述所有挑战：
- **多进程架构**（API Server + Tokenizer + Scheduler）分离 CPU/GPU 工作，充分利用并行性
- **分页 KV Cache** 解决显存碎片和利用率问题
- **Prefill/Decode 两阶段调度** 针对两种不同计算特征分别优化
- **Overlap Scheduling** 隐藏 CPU 调度开销，提升 GPU 利用率

**与后续章节的连接**：

- **第 2 章** 深入 KV Cache 的分页实现：`kvcache/` 模块，理解 `page_table` 如何将逻辑 token 位置映射到物理显存
- **第 3 章** 探讨调度器细节：prefill 与 decode 如何交替执行，chunked prefill 如何限制单步延迟
- **第 4 章** 介绍前缀缓存（Radix Cache）：如何在不同请求间复用共同的 KV Cache 前缀，大幅降低重复 prompt 的计算开销
- **第 5 章** 分析注意力内核：为什么 decode 阶段需要 FlashInfer 而非普通 FlashAttention

---

*本章涉及的主要文件：*
- `python/minisgl/__main__.py` — 入口点
- `python/minisgl/core.py` — 核心数据类（`Req`、`Batch`、`Context`）
- `python/minisgl/server/launch.py` — 多进程启动逻辑
- `python/minisgl/server/args.py` — 服务配置参数
- `python/minisgl/scheduler/scheduler.py` — 调度器主循环
