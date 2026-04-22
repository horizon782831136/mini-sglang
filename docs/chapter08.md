# 第 8 章：模型实现与张量并行——如何把一个大模型拆到多张 GPU 上

## 8.1 背景问题

假设你手头有一个 70B 参数的 LLM，每个参数以 bfloat16 存储，光是模型权重就需要约 140 GB 显存。单张 80 GB 的 A100 装不下，更别说还要留空间给 KV Cache 和激活值。

朴素的解法是"把模型切成几段，每张 GPU 负责几层"——这叫**流水线并行（Pipeline Parallelism）**。但流水线并行有气泡（bubble）问题，而且需要复杂的调度逻辑。

mini-sglang 采用的是另一种路线：**张量并行（Tensor Parallelism，TP）**。它不是按层切，而是把**每一层内部的权重矩阵**沿某个维度切开，让每张 GPU 只保存并计算一部分，然后在必要时通过一次集合通信（all-reduce 或 all-gather）把结果合并。

核心矛盾在于：**集合通信有代价，但必须在正确的位置插入才能保证数学等价**。本章的目标就是搞清楚：哪里切、怎么切、在哪里通信、为什么这样设计。

---

## 8.2 核心概念讲解

### 8.2.1 从一个 nn.Linear 说起

标准的全连接层：

```
y = x @ W^T + b
```

其中 `x` 的形状为 `[batch, in_features]`，`W` 的形状为 `[out_features, in_features]`。

如果有 N 张 GPU（tp_size = N），有两种基本的切法：

**切法一：列并行（Column Parallel）—— 切 output 维**

把 `W` 沿行方向（output 维）均匀切成 N 份，每张 GPU 保存 `W_i`（形状 `[out_features/N, in_features]`）。每张 GPU 独立计算 `y_i = x @ W_i^T`，得到完整输出的一段。这时 `x` 需要在所有 GPU 上保持相同的副本。

**切法二：行并行（Row Parallel）—— 切 input 维**

把 `W` 沿列方向（input 维）均匀切成 N 份，每张 GPU保存 `W_i`（形状 `[out_features, in_features/N]`）。相应地，输入 `x` 也被切成 `x_i`（形状 `[batch, in_features/N]`）。每张 GPU计算 `y_i = x_i @ W_i^T`，这是完整输出的一个**部分和**，需要 all-reduce 求和才能得到最终结果。

### 8.2.2 Transformer 中的两段式切法

Transformer 的 Attention 和 FFN 天然适合"列并行 → 行并行"的配对模式：

```
输入 x（所有 GPU 相同）
  ↓  列并行（每 GPU 算一部分 head/neuron）
中间激活（已经被切分，各 GPU 持有不同部分）
  ↓  行并行（各 GPU 算部分和）
  ↓  all-reduce（求和，恢复完整输出）
输出（所有 GPU 再次相同）
```

这个模式的关键性质：**两次矩阵乘之间不需要通信**，只在行并行的最后做一次 all-reduce。这极大地摊薄了通信开销。

### 8.2.3 GQA 与 QKV 合并投影

现代 LLM 普遍使用 **Grouped Query Attention（GQA）**：Q 头数（`num_qo_heads`）多于 KV 头数（`num_kv_heads`），比例为 `GQA_ratio = num_qo_heads / num_kv_heads`。

在张量并行下，GQA 的切分原则是：**以 KV head 为基本单元**。若 `num_kv_heads = 8`，tp_size = 4，则每张 GPU 分配 2 个 KV head，以及对应的 `2 * GQA_ratio` 个 Q head。

---

## 8.3 核心代码导读

### 8.3.1 分布式元信息：`DistributedInfo`

文件：`python/minisgl/distributed/info.py`

```python
# info.py 第 6-9 行
@dataclass(frozen=True)
class DistributedInfo:
    rank: int   # 当前 GPU 的编号（0 ~ size-1）
    size: int   # 总 GPU 数（即 tp_size）
```

`DistributedInfo` 是一个不可变的数据类，通过全局单例 `_TP_INFO` 在进程启动时设置一次，之后各层通过 `get_tp_info()` 读取。这个设计避免了把 `tp_rank` / `tp_size` 传递到每个构造函数参数里，让层的接口保持简洁。

### 8.3.2 列并行：`LinearColParallelMerged` 与 `LinearQKVMerged`

文件：`python/minisgl/layers/linear.py`

`LinearColParallelMerged` 用于 FFN 的 gate + up 合并投影（Gated MLP 里把 gate_proj 和 up_proj 合并为一个矩阵运算以提高效率）：

```python
# linear.py 第 56-68 行
class LinearColParallelMerged(_LinearTPImpl):
    def __init__(self, input_size, output_sizes, has_bias):
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)
```

关键点：
- `output_sizes` 是一个列表，例如 `[intermediate_size, intermediate_size]`，代表两个合并的投影
- 每个分量都被 `tp_size` 均分，本地权重形状为 `[tp_output_size, input_size]`
- **没有 all-reduce**：列并行的输出各 GPU 持有不同的 output channel，可以直接继续下一步运算

`LinearQKVMerged` 专门处理 Attention 的 QKV 合并投影，考虑了 GQA 的不对等性：

```python
# linear.py 第 71-88 行
class LinearQKVMerged(_LinearTPImpl):
    def __init__(self, hidden_size, head_dim, num_qo_heads, num_kv_heads, has_bias):
        tp_info = get_tp_info()
        GQA_ratio = div_even(num_qo_heads, num_kv_heads)
        local_num_kv = div_even(num_kv_heads, tp_info.size)
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)
```

`GQA_ratio + 2` 是因为每个 KV head 对应 `GQA_ratio` 个 Q head，再加上 1 个 K head 和 1 个 V head，共 `GQA_ratio + 2` 个 head 为一组同步切分。

### 8.3.3 行并行：`LinearOProj` 与 `LinearRowParallel`

文件：`python/minisgl/layers/linear.py`，第 91-127 行

`LinearOProj` 是 Attention 的 O 投影（output projection），`LinearRowParallel` 是 FFN 的 down_proj。两者结构一致，核心在 `forward`：

```python
# linear.py 第 102-106 行（LinearOProj.forward）
def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = F.linear(x, self.weight, self.bias)
    if self._tp_size > 1:
        y = self._comm.all_reduce(y)   # ← 这里是唯一的集合通信点
    return y
```

注意：
- 本地 input 形状是 `[batch, input_size/tp_size]`，对应 Attention 里各自算好的局部 head 输出
- `F.linear` 得到的是"部分和"（partial sum）
- `all_reduce` 做的是跨 GPU 的逐元素求和，还原出完整的 hidden_state
- 如果 `tp_size == 1`，完全跳过通信，无额外开销

### 8.3.4 词表并行：`VocabParallelEmbedding`

文件：`python/minisgl/layers/embedding.py`

词表（vocabulary）通常有 10 万量级的 token，每个 token 对应一行 embedding 向量。`VocabParallelEmbedding` 把词表按 GPU 数量均分：

```python
# embedding.py 第 25-28 行
self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
start_idx = self.num_embeddings_tp * tp_rank
finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
self.vocab_range = (start_idx, finish_idx - start_idx)
```

每张 GPU 只保存 `[num_embeddings/tp_size, embedding_dim]` 的权重切片。`vocab_range` 记录本 GPU 负责的 token ID 范围 `[start_idx, start_idx + length)`。

forward 里有一个值得细看的写法：

```python
# embedding.py 第 38-40 行
mask = (x >= start) & (x < start + length)
clamped = torch.clamp(x - start, 0, length - 1)
y = torch.where(
    mask.unsqueeze(-1),
    F.embedding(clamped, self.weight),
    torch.zeros(...),
)
```

**为什么不直接过滤掉不属于本 GPU 的 token ID？**

因为 `torch.index_select` / 条件索引会产生动态形状的张量，**无法被 CUDA Graph 捕获**。这里的写法刻意保持张量形状固定：
1. 用 `torch.clamp` 把所有 token ID 强制映射到本地词表范围内（越界 ID 被 clamp 到边界）
2. 对所有位置都执行 `F.embedding`（包括不属于本 GPU 的 ID，但它们被 clamp 后查到的是无意义值）
3. 用 `torch.where` 把不属于本 GPU 的位置清零

最后一行 `self._comm.all_reduce(y)` 把各 GPU 的局部 embedding 向量求和，因为只有一个 GPU 的对应位置是有效值（其余 GPU 该位置为零），all-reduce 求和等价于"广播正确的 embedding"。

### 8.3.5 `ParallelLMHead`：输出投影与权重共享

文件：`python/minisgl/layers/embedding.py`，第 54-119 行

`ParallelLMHead` 继承自 `VocabParallelEmbedding`，复用了词表切分逻辑。它的职责是把最后一层的 hidden_state 映射回词表分布（logits）：

```python
# embedding.py 第 107-119 行
logits = F.linear(x, module.weight, self.bias)
if self.tp_size == 1:
    return logits
output_tensor = self._comm.all_gather(logits)
# 多 batch 时需要重排维度顺序
output_tensor = output_tensor.view((self.tp_size,) + input_shape)
output_tensor = output_tensor.movedim(0, -1)
output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
return output_tensor[:, : self.num_embeddings]
```

这里用的是 **all-gather** 而非 all-reduce，原因是：LM Head 是列并行（每 GPU 算部分词表的 logits），各 GPU 的结果是互补的，需要拼接而不是求和。最后 `[:, :self.num_embeddings]` 截掉因 `div_ceil` 引入的 padding。

**`tie_word_embeddings`** 是一个常见优化：让 `lm_head` 的权重与 `embed_tokens` 共享同一张矩阵，减少参数量。实现方式见第 106 行：

```python
# embedding.py 第 106 行
module = self.tied_embedding or self
logits = F.linear(x, module.weight, self.bias)
```

当 `tie_word_embeddings=True` 时，`module.weight` 实际上指向 `embed_tokens.weight`，两者共享内存，不需要额外的权重加载。

### 8.3.6 Qwen3 完整前向路径

文件：`python/minisgl/models/qwen3.py`

```
Qwen3ForCausalLM.forward()
  └─ Qwen3Model.forward(input_ids)
       ├─ embed_tokens: VocabParallelEmbedding          # token id → hidden_state
       ├─ layers[0..N]:  Qwen3DecoderLayer.forward(x, residual)
       │    ├─ input_layernorm: RMSNormFused
       │    ├─ self_attn: RopeAttn.forward(x)
       │    │    ├─ qkv_proj: LinearQKVMerged            # 列并行，无通信
       │    │    ├─ attn: AttentionLayer (FlashAttn/Paged)
       │    │    └─ o_proj: LinearOProj                  # 行并行 + all-reduce
       │    ├─ post_attention_layernorm: RMSNormFused
       │    └─ mlp: GatedMLP.forward(x)
       │         ├─ gate_up_proj: LinearColParallelMerged  # 列并行，无通信
       │         ├─ act_fn: silu_and_mul
       │         └─ down_proj: LinearRowParallel           # 行并行 + all-reduce
       └─ norm: RMSNormFused
  └─ lm_head: ParallelLMHead                            # 列并行 + all-gather
```

每个 Decoder Layer 只发生**两次集合通信**：o_proj 后的 all-reduce 和 down_proj 后的 all-reduce。LM Head 发生一次 all-gather。对于 32 层的模型，tp_size=8 时，整个前向过程共 65 次 NCCL 操作，相比激活计算量而言开销可控。

---

## 8.4 设计决策

### 为什么用 `all_reduce` 而不是 `reduce_scatter + all_gather`（序列并行）？

序列并行（Sequence Parallelism）可以进一步把 LayerNorm 的激活值也切分，但需要在更多位置插入通信，调度更复杂。mini-sglang 优先追求实现简洁，采用标准 Megatron 风格的张量并行，仅在行并行的末尾做 all-reduce。

### `DistributedCommunicator` 的插件化设计

文件：`python/minisgl/distributed/impl.py`，第 63-70 行

```python
class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)
```

`plugins` 是一个类级别的列表，默认后端是 `TorchDistributedImpl`（使用 `torch.distributed`）。通过 `enable_pynccl_distributed()` 可以追加 `PyNCCLDistributedImpl`，实现 CUDA Graph 兼容的 NCCL 通信。新后端直接覆盖旧后端（取列表最后一个），切换过程不需要修改任何层代码。

### `torch.where` vs 条件索引

词表并行中用 `torch.where` 而非布尔索引，核心原因是 CUDA Graph 要求计算图的形状在捕获时就固定。`torch.where` 的输入输出形状完全确定，是 CUDA Graph 友好的操作；而 `x[mask]` 产生的输出长度取决于运行时的 mask 内容，无法被静态图捕获。

### `div_ceil` vs `div_even`

词表并行用 `div_ceil`（向上取整），线性层并行用 `div_even`（要求整除）。前者是因为词表大小不一定能被 tp_size 整除（如 vocab_size=151936, tp_size=8 就无法整除），padding 后截断处理；后者是因为权重矩阵的切分维度必须整除，否则会出现不等大的分片，逻辑更复杂，mini-sglang 选择在模型配置层面保证整除性。

---

## 8.5 动手实验

### 实验一：验证张量并行的数学等价性（基础）

在单机上用两个进程模拟 tp_size=2，手动验证列并行 + 行并行的结果与非并行结果一致。

```python
# 文件：experiments/ch08_tp_verify.py
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# 单 GPU 的参考计算
hidden = 64
inter = 128
x = torch.randn(4, hidden)

W_col = torch.randn(inter, hidden)     # gate_proj（列并行示意）
W_row = torch.randn(hidden, inter)     # down_proj（行并行示意）
ref = F.linear(F.silu(F.linear(x, W_col)), W_row)

# 模拟 tp_size=2：切分权重
W_col_0, W_col_1 = W_col.chunk(2, dim=0)   # 列并行：切 output dim
W_row_0, W_row_1 = W_row.chunk(2, dim=1)   # 行并行：切 input dim

# GPU 0 的计算
h0 = F.silu(F.linear(x, W_col_0))          # [4, inter/2]
y0 = F.linear(h0, W_row_0)                  # [4, hidden]，部分和

# GPU 1 的计算
h1 = F.silu(F.linear(x, W_col_1))          # [4, inter/2]
y1 = F.linear(h1, W_row_1)                  # [4, hidden]，部分和

# all-reduce（求和）
result = y0 + y1

print("最大误差:", (result - ref).abs().max().item())
# 期望输出：最大误差在 1e-5 量级（float32 精度范围内）
```

运行方式：`python experiments/ch08_tp_verify.py`

### 实验二：验证 VocabParallelEmbedding 的正确性（基础）

```python
# 文件：experiments/ch08_vocab_verify.py
import torch
import torch.nn.functional as F

vocab_size = 100
embed_dim = 16
tp_size = 4

# 完整 Embedding 表
full_weight = torch.randn(vocab_size, embed_dim)

# 模拟 4 张 GPU 各自持有的分片
chunk = (vocab_size + tp_size - 1) // tp_size   # div_ceil = 25

input_ids = torch.randint(0, vocab_size, (8,))
ref = F.embedding(input_ids, full_weight)

total = torch.zeros(8, embed_dim)
for rank in range(tp_size):
    start = rank * chunk
    length = min(chunk, vocab_size - start)
    weight_shard = full_weight[start:start+length]

    mask = (input_ids >= start) & (input_ids < start + length)
    clamped = torch.clamp(input_ids - start, 0, length - 1)
    local_out = torch.where(
        mask.unsqueeze(-1),
        F.embedding(clamped, weight_shard),
        torch.zeros(8, embed_dim),
    )
    total += local_out   # 模拟 all-reduce

print("最大误差:", (total - ref).abs().max().item())
```

### 实验三：观察通信开销与 tp_size 的关系（进阶）

修改 mini-sglang 的启动参数，分别以 tp_size=1、2、4 运行并发请求，记录 TTFT（首 token 延迟）和吞吐量：

```bash
# 启动服务（以 Qwen3-7B 为例）
python -m minisgl.server.api_server --model Qwen/Qwen3-7B --tp-size 1
python -m minisgl.server.api_server --model Qwen/Qwen3-7B --tp-size 2
python -m minisgl.server.api_server --model Qwen/Qwen3-7B --tp-size 4
```

分析预期结论：
- TTFT 随 tp_size 增大而减小（并行度更高，但通信量增加，边际收益递减）
- 对于小 batch，tp_size 增大可能带来负面效果（通信延迟 > 计算节省）
- 对于大 batch（吞吐量场景），tp_size 增大通常有收益

### 实验四：追踪 NCCL 调用次数（进阶）

在 `LinearOProj.forward` 和 `LinearRowParallel.forward` 里加计数器，统计每次 forward 调用的 all-reduce 次数，验证"每层 2 次通信"的结论：

```python
# 在 LinearOProj 的 forward 末尾临时加入：
import os
if os.environ.get("COUNT_NCCL"):
    LinearOProj._nccl_count = getattr(LinearOProj, "_nccl_count", 0) + 1
    print(f"[OProj] all-reduce #{LinearOProj._nccl_count}")
```

---

## 8.6 小结

本章从 `nn.Linear` 的两种切分方式（列并行和行并行）出发，逐层拆解了 mini-sglang 的张量并行实现：

| 组件 | 并行方式 | 通信操作 | 通信时机 |
|------|----------|----------|----------|
| `LinearColParallelMerged` / `LinearQKVMerged` | 列并行 | 无 | — |
| `LinearOProj` / `LinearRowParallel` | 行并行 | all-reduce | forward 末尾 |
| `VocabParallelEmbedding` | 词表切分 | all-reduce | forward 末尾 |
| `ParallelLMHead` | 列并行（词表） | all-gather | forward 末尾 |

几个核心设计要点：
1. **列并行不需要通信**，行并行需要 all-reduce，两者配对形成每层 2 次通信的节奏
2. **`torch.where` 替代条件索引**，是为了兼容 CUDA Graph 的静态形状要求
3. **`tie_word_embeddings`** 通过让 `lm_head` 持有 `embed_tokens` 的引用来实现零拷贝权重共享
4. **`DistributedCommunicator` 插件化**，使 torch.distributed 后端和 PyNCCL 后端可无缝切换

与后续章节的连接：
- 第 9 章将介绍 KV Cache 管理（PagedAttention），它与张量并行的交互在于 KV head 的切分需要与 Cache Block 的分配保持一致
- 第 10 章将介绍 CUDA Graph 的捕获与回放，届时会看到 `torch.where` 等 CUDA Graph 友好写法的全貌
