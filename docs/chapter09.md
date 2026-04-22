# 第 9 章：CUDA Graph——消灭 kernel launch 开销

## 9.1 背景问题：推理时的 CPU 瓶颈

在深度学习推理中，我们习惯把 GPU 算力视为性能瓶颈。然而在 LLM 的 **decode 阶段**，存在一个反直觉的问题：GPU 算力已经足够，但 **CPU 反而成为瓶颈**。

理解这一点需要从 PyTorch 的执行模型说起。当你写下：

```python
y = model(x)
```

这行代码在 CPU 上发起了数十乃至数百次 CUDA kernel launch。每次 launch 都需要：

1. CPU 验证参数合法性
2. 将 kernel 描述符写入 CUDA driver 队列
3. GPU 从队列中取出任务开始执行

每次 kernel launch 的 CPU 开销大约在 **5~20 微秒**。对于一个 transformer 的 forward pass，即便模型很小，也涉及几十个矩阵乘法、激活函数、LayerNorm 等操作，合计 CPU launch 时间可能达到 **数百微秒甚至超过 1 毫秒**。

**Prefill 阶段**不受此影响：输入序列可能有数百到数千个 token，单次 kernel 的计算量（矩阵乘法是 O(n²) 的）远超 launch 开销，GPU 会持续保持忙碌。

**Decode 阶段**就不同了：每步只处理 1 个新 token（或极少的几个），矩阵乘法退化为矩阵-向量乘，计算量骤降，GPU 执行时间可能仅需 **几十微秒**，此时 CPU launch 开销与 GPU 执行时间相当甚至更长。极端情况下，GPU 50% 以上的时间处于空闲等待状态——这就是 **kernel launch overhead** 问题。

**本章的核心目标**：用 CUDA Graph 技术，把 decode 阶段的所有 kernel launch 从逐个发射变为"一次性重放"，彻底消除这一 CPU 瓶颈。

---

## 9.2 核心概念：CUDA Graph 是什么

### 9.2.1 从"乐谱"类比出发

CUDA Graph 的思想可以类比为**乐谱录制与演奏**：

- **普通执行**：指挥（CPU）每次演奏时，对每位乐手（GPU kernel）逐一发出指令，每发一次指令都有沟通成本。
- **CUDA Graph**：提前录制一段乐谱（capture），之后每次演奏只需指挥说一声"按乐谱来"（replay），所有乐手按照录好的顺序自动执行，省去了逐一指令的开销。

### 9.2.2 CUDA Graph 的两个阶段

**Capture 阶段**（录制）：

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=my_stream):
    output = model.forward()  # 这里的执行被"录制"下来
```

在 `with` 块内，GPU 的操作不会立即执行，而是被记录为一个有向无环图（DAG），每个节点是一个 kernel，每条边代表依赖关系。

**Replay 阶段**（重放）：

```python
graph.replay()  # 一次 CPU 调用，重放全部 kernel
```

replay 只向 CUDA driver 发出一次命令，CUDA driver 内部按照录制的 DAG 顺序依次调度所有 kernel，完全绕过了 Python 层面的逐次 launch。

### 9.2.3 静态形状的约束

CUDA Graph 最关键的限制是：**所有 tensor 的地址和形状必须在 capture 时固定，replay 时不得改变**。

这是因为 kernel 的 launch 参数（包括 grid size、block size、指针地址）都已经在 capture 时"写死"进了 DAG 节点。如果 replay 时输入 tensor 的地址变了，kernel 就会读错内存。

实际上，你可以改变 tensor 的**内容**（值），但不能改变 tensor 的**地址和 shape**。mini-sglang 利用这一点：capture 时分配好 buffer，replay 时先把新数据 copy 进 buffer，再触发 graph.replay()。

---

## 9.3 核心代码导读

### 9.3.1 GraphCaptureBuffer：固定地址的数据桥梁

文件：`python/minisgl/engine/graph.py`，第 21~47 行。

```python
@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
```

`GraphCaptureBuffer` 承担着 CUDA Graph 的"数据桥梁"职责。它在 capture 之前就分配好了固定大小（`max_graph_bs`）的 tensor，这些 tensor 的 GPU 地址在整个生命周期内不变。

`set_batch` 方法（第 36~40 行）的设计非常精妙：

```python
def set_batch(self, batch: Batch) -> None:
    _slice = slice(batch.padded_size)
    batch.input_ids = self.input_ids[_slice]   # 让 batch 的字段指向 buffer 的子 slice
    batch.out_loc = self.out_loc[_slice]
    batch.positions = self.positions[_slice]
```

它并不拷贝数据，而是**让 batch 的字段指向 buffer 的子 tensor**。这样 capture 阶段调用 `model.forward()` 时，模型读取的就是 buffer 里的数据，这些数据的 GPU 地址被记录进了 DAG。

replay 时，`copy_from` 方法（第 42~46 行）将真实 batch 数据写入 buffer，之后 `g.replay()` 触发，模型就自动读取到了新数据。

### 9.3.2 Capture 流程：_capture_graphs

文件：`python/minisgl/engine/graph.py`，第 105~144 行。

```python
def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
    ...
    pool = None
    for bs in pbar:
        graph = torch.cuda.CUDAGraph()
        batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
        batch.padded_reqs = batch.reqs
        self.attn_backend.prepare_for_capture(batch)
        self.buffer.set_batch(batch)
        with get_global_ctx().forward_batch(batch):
            self.buffer.logits[:bs] = model.forward()          # warmup run（第 139 行）
            with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                self.buffer.logits[:bs] = model.forward()      # capture run（第 141 行）
        if pool is None:
            pool = graph.pool()
        self.graph_map[bs] = graph
```

几个关键设计点：

**Warmup run（第 139 行）**：capture 前先做一次"热身"正向传播。这是必须的——PyTorch 的 cuBLAS、cuDNN 等算子在第一次运行时会做 workspace 分配和算法选择，这些操作不能在 capture 阶段发生（会导致 capture 失败）。热身 run 触发这些一次性初始化，之后 capture run 才能干净录制。

**Memory pool 共享（第 142~143 行）**：`pool = graph.pool()` 后，后续所有 graph 共用同一个 CUDA memory pool，显著减少显存占用。

**按 batch size 分别 capture（第 129 行）**：因为 CUDA Graph 要求静态形状，不同 batch size 对应不同的 kernel grid 参数，必须各自 capture。

### 9.3.3 Batch Size 列表的确定

文件：`python/minisgl/engine/graph.py`，第 49~67 行。

```python
return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
```

为什么要对 `[1, 2, 4, 8, 16, 24, ...]` 这些特殊的值分别 capture？

原因在于 replay 时的"向上取整"策略（第 160~166 行）：

```python
def pad_batch(self, batch: Batch) -> None:
    padded_size = (
        next(bs for bs in self.graph_bs_list if bs >= batch.size)
        ...
    )
    batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
```

实际的 batch size 可能是任意正整数，但只有 capture 过的 batch size 才能 replay。因此需要把实际 batch size 向上"对齐"到最近的已 capture size。小 batch（1/2/4）需要细粒度，大 batch 则每 8 个 token 一档，这是内存占用与灵活性之间的权衡。

### 9.3.4 Engine 中的调度逻辑

文件：`python/minisgl/engine/engine.py`，第 193~208 行。

```python
def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
    with self.ctx.forward_batch(batch):
        if self.graph_runner.can_use_cuda_graph(batch):
            logits = self.graph_runner.replay(batch)      # CUDA Graph 路径
        else:
            logits = self.model.forward()                  # 普通路径
```

`can_use_cuda_graph` 的判断条件（第 149~150 行）：

```python
def can_use_cuda_graph(self, batch: Batch) -> bool:
    return batch.is_decode and batch.size <= self.max_graph_bs
```

只有 decode 阶段且 batch size 不超过最大 capture size 时，才走 CUDA Graph 路径。Prefill 阶段始终走普通路径——原因在下一节解释。

---

## 9.4 设计决策与关键限制

### 9.4.1 为什么 Prefill 不能用 CUDA Graph

Prefill 阶段的 token 数量随请求长度变化，无法用静态形状覆盖。更根本的原因是：prefill 阶段的计算量（矩阵乘法，O(seq_len²) 复杂度）远大于 launch 开销，GPU 始终是满载的，CUDA Graph 带来的 CPU 节省对总吞吐的提升可以忽略不计。

### 9.4.2 Boolean Mask 问题：`cudaErrorStreamCaptureUnsupported`

这是 CUDA Graph 最容易踩到的坑之一。考虑下面的代码：

```python
# 问题代码（不能在 CUDA Graph capture 中使用）
mask = (x >= start) & (x < start + length)
y = self.weight[x[mask] - start]   # boolean mask 索引
```

在 CUDA Graph capture 阶段执行这段代码，会抛出：

```
RuntimeError: CUDA error: operation not permitted when stream is capturing
(cudaErrorStreamCaptureUnsupported)
```

原因：boolean mask 索引（`tensor[bool_mask]`）需要先执行一次 `nonzero` 操作来确定哪些元素满足条件，`nonzero` 的输出 shape 在 capture 时是未知的（依赖运行时数据），这违反了 CUDA Graph 的静态形状约束。CUDA driver 无法将一个形状动态变化的操作录制进 DAG。

### 9.4.3 修复方案：clamp + where

文件：`python/minisgl/layers/embedding.py`，第 38~40 行。

```python
# 修复后的代码（CUDA Graph 兼容）
mask = (x >= start) & (x < start + length)
clamped = torch.clamp(x - start, 0, length - 1)    # 形状静态，无条件执行
y = torch.where(
    mask.unsqueeze(-1),
    F.embedding(clamped, self.weight),              # 形状静态
    torch.zeros(...),
)
```

关键在于：

- `torch.clamp`：将超出范围的索引强制钳制到合法区间内，**不改变 tensor 形状**，因此可以安全地 capture。即使某些位置的 token id 不属于这个 TP 分片，clamp 后仍然能合法查表（只是查到了错误的值），但最终会被 `torch.where` 的 mask 清零。
- `torch.where`：根据 mask 选择正确值或零，**输入输出形状完全一致**，满足 CUDA Graph 的静态形状要求。

**对比总结**：

| 方式 | 形状是否静态 | 能否 capture | 说明 |
|------|------------|------------|------|
| `weight[x[mask] - start]` | 否（取决于 mask 中 True 的数量） | 不能 | 触发 `cudaErrorStreamCaptureUnsupported` |
| `clamp + where` | 是 | 能 | 计算量略增（多一次 clamp + 全量 embedding），但语义正确 |

这种"先无条件计算，再用 mask 选择"的模式，是 CUDA Graph 兼容代码的通用写法。

### 9.4.4 Dummy Request：让 padding 不影响正确性

Capture 时，多余的 padding 位置填充的是 `dummy_req`（engine.py 第 89~98 行）。Dummy request 指向 KV cache 中的一个"哑页"，所有注意力计算对哑页的访问不会影响真实请求的结果。这确保了 padding 后的 batch 在功能上与原 batch 等价，只是 logits 中多余的行会被丢弃（`replay` 第 158 行：`return self.buffer.logits[:batch.size]`）。

---

## 9.5 动手实验

### 实验一：验证 CUDA Graph 的加速效果

以下脚本对比 decode 阶段有无 CUDA Graph 时的延迟：

```python
import torch
import time

device = torch.device("cuda:0")
torch.cuda.set_device(device)

# 用一个简单的线性层模拟 decode 的矩阵-向量乘
hidden = 4096
vocab = 32000
linear = torch.nn.Linear(hidden, vocab, bias=False).to(device).to(torch.float16)

# 模拟 decode batch size = 4
x = torch.randn(4, hidden, device=device, dtype=torch.float16)
out_buf = torch.empty(4, vocab, device=device, dtype=torch.float16)

# --- 普通执行：测量 100 次的平均延迟 ---
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    out = linear(x)
torch.cuda.synchronize()
print(f"Normal: {(time.perf_counter() - t0) / 100 * 1000:.3f} ms/iter")

# --- CUDA Graph 执行 ---
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out_buf.copy_(linear(x))

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    graph.replay()
torch.cuda.synchronize()
print(f"CUDA Graph: {(time.perf_counter() - t0) / 100 * 1000:.3f} ms/iter")
```

**预期结果**：在小 batch size 下，CUDA Graph 版本通常快 30%~60%，在 batch size = 1 时差异最为显著。

### 实验二：复现 boolean mask 导致的 capture 失败

```python
import torch

device = torch.device("cuda:0")
torch.cuda.set_device(device)

weight = torch.randn(1000, 64, device=device)
x = torch.randint(0, 1000, (4,), device=device)
start, length = 0, 1000

graph = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(graph):
        mask = (x >= start) & (x < start + length)
        y = weight[x[mask] - start]   # 触发 nonzero，capture 失败
    print("Capture succeeded (unexpected)")
except RuntimeError as e:
    print(f"Capture failed as expected: {e}")
```

**预期输出**：

```
Capture failed as expected: CUDA error: operation not permitted when stream is capturing
```

然后将 boolean mask 改为 `clamp + where`，验证 capture 成功：

```python
graph2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph2):
    mask = (x >= start) & (x < start + length)
    clamped = torch.clamp(x - start, 0, length - 1)
    y = torch.where(
        mask.unsqueeze(-1),
        torch.nn.functional.embedding(clamped, weight),
        torch.zeros(x.shape[0], weight.shape[1], device=device),
    )
print("Fixed version captured successfully!")
graph2.replay()
print("Replay succeeded!")
```

### 实验三（进阶）：观察 Warmup Run 的必要性

修改 `graph.py` 的 `_capture_graphs` 方法，注释掉第 139 行的 warmup run，重新启动推理服务，观察 capture 是否报错或者首次 replay 是否产生异常结果。这个实验帮助你理解 PyTorch 算子内部的懒初始化机制。

### 实验四（进阶）：测量不同 batch size 下的加速比

修改实验一，在 batch size = 1, 2, 4, 8, 16, 32 下分别测量加速比，绘制加速比随 batch size 变化的曲线。**预期趋势**：batch size 越小，加速比越大；batch size 超过某个阈值（通常 32~64）后，GPU 计算时间主导，CUDA Graph 加速比趋近于 1。

---

## 9.6 小结

本章围绕 LLM decode 阶段的 CPU 瓶颈展开，核心要点如下：

1. **问题根源**：decode 阶段每步计算量小，GPU 执行时间短，CPU 的 kernel launch 开销成为主导。
2. **CUDA Graph 原理**：capture 阶段录制 kernel DAG，replay 阶段一次性重放，将数十次 launch 压缩为一次，消除 CPU 开销。
3. **静态形状约束**：CUDA Graph 要求 tensor 地址和形状在 capture 后保持不变。`GraphCaptureBuffer` 通过预分配固定 buffer 并在 replay 前 copy 数据来满足这一约束。
4. **Boolean mask 禁区**：凡是依赖运行时 tensor 内容决定输出 shape 的操作（如 `nonzero`、boolean 索引），都会触发 `cudaErrorStreamCaptureUnsupported`。正确的替代方案是 `clamp + where` 的组合。
5. **Prefill vs Decode**：prefill 计算量大，GPU 持续满载，CUDA Graph 收益不明显；decode 计算量小，CUDA Graph 效果显著。
6. **多 batch size capture**：由于形状静态，不同 batch size 需要分别 capture，实际 batch 向上对齐到最近的已 capture size。

**与后续章节的连接**：

CUDA Graph 解决了单机 decode 阶段的 CPU 开销问题。在多 GPU 张量并行（TP）场景下，每个 rank 需要独立执行 CUDA Graph replay，同时还要通过 NCCL/PyNCCL 完成 all-reduce 通信。第 10 章将介绍张量并行是如何与 CUDA Graph 配合工作的，以及为什么 `destroy_cuda_graphs`（`graph.py` 第 169 行）必须在释放 NCCL 资源之前调用——这背后涉及 CUDA stream 与通信库的生命周期管理。
