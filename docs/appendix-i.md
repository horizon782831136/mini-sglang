# 附录 I：Transformer 架构与注意力机制

> 本附录面向有 PyTorch 训练经验、但对 Transformer 底层细节或推理优化不够熟悉的读者。目标是从第一性原理出发，把"为什么这样设计"和"代码里实际在做什么"讲清楚。

---

## I.1 Transformer 整体架构回顾

### I.1.1 Encoder-Decoder 与 Decoder-only：为什么 LLM 只用 Decoder

原始 Transformer（Vaswani et al., 2017）是一个 Encoder-Decoder 结构，专为机器翻译设计：

```
输入序列 → [Encoder × N 层] → 上下文向量
                                    ↓
输出序列 → [Decoder × N 层] → 预测下一个 token
```

Encoder 对整个输入做**双向注意力**（每个位置可以看到前后所有位置），而 Decoder 做**因果（Causal）注意力**——每个位置只能看到它自己及之前的位置，从而保证自回归生成时不泄露未来信息。

LLM（如 GPT、LLaMA、Qwen）只保留 Decoder，原因有三：

1. **统一范式**：语言建模本质上是"给定前文，预测下一个 token"，天然是自回归的，不需要编码器提供独立的上下文向量。
2. **更易扩展**：Decoder-only 结构参数量集中，训练目标单纯（next-token prediction），规模扩展的行为更可预测。
3. **推理效率**：Encoder 的双向注意力每次生成都要重新跑全序列，而 Decoder 配合 KV Cache（见 I.3 节）可以复用历史计算结果。

### I.1.2 一个 Transformer Block 的完整组成

每个 Transformer Block 的计算流程如下：

```
输入 x
  │
  ▼
┌─────────────────────┐
│  Multi-Head Attention│  ← 自注意力（或跨注意力）
└──────────┬──────────┘
           │
     残差连接：x + Attention(x)
           │
  ▼
┌─────────────────────┐
│     Layer Norm       │  ← 归一化
└──────────┬──────────┘
           │
  ▼
┌─────────────────────┐
│   Feed-Forward Net  │  ← 两层线性 + 激活（SwiGLU 等）
└──────────┬──────────┘
           │
     残差连接：h + FFN(h)
           │
  ▼
┌─────────────────────┐
│     Layer Norm       │
└──────────┬──────────┘
           │
         输出
```

**残差连接**的作用：梯度可以不经过 Attention/FFN 直接回传，解决深层网络的梯度消失问题，同时让模型在初始化时近似于恒等映射，训练更稳定。

**Layer Norm** 的作用：对每个样本的特征维度做归一化，使激活值保持在合理范围内，加速收敛。现代 LLM 通常使用 **RMSNorm**（只做均方根归一化，不减均值），计算更简单且效果相当。

### I.1.3 位置编码：从 Sinusoidal 到 RoPE

Transformer 的注意力本身是**置换不变**的——打乱输入顺序，输出不变。因此必须显式注入位置信息。

**Sinusoidal 位置编码**（原始 Transformer）：

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

优点是可以外推到训练时未见过的长度（理论上），缺点是与输入做加法后**绝对位置**被编码进了向量，模型难以直接感知**相对位置**。

**可学习的绝对位置编码**（BERT、GPT-2）：用一个 `[max_len, d_model]` 的 Embedding 表，直接学习每个位置的向量。优点是灵活，缺点是长度固定，超出 `max_len` 就无法使用。这正是 LLM 推理中**不用可学习位置编码**的关键原因：LLM 需要处理任意长度的上下文，可学习编码无法外推。

**RoPE（Rotary Position Embedding）**（Su et al., 2021）：

RoPE 的核心思想是：不把位置向量加到特征向量上，而是在计算注意力分数时，**用旋转矩阵对 Q 和 K 施加位置相关的旋转变换**，使得 $q_m \cdot k_n$（位置 $m$ 的 Query 与位置 $n$ 的 Key 的内积）只依赖于相对位置 $m - n$。

对于二维情况，位置 $m$ 的旋转变换为：

$$R_m = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}$$

将 Q 和 K 的每对相邻维度 $(q_{2i}, q_{2i+1})$ 分别乘以 $R_{m, \theta_i}$（$\theta_i = base^{-2i/d}$），然后计算内积：

$$\tilde{q}_m^T \tilde{k}_n = q_m^T R_m^T R_n k_n = q_m^T R_{n-m} k_n$$

注意 $R_m^T R_n = R_{n-m}$——旋转矩阵的乘法自动编码了相对位置 $n-m$，不需要额外的相对位置表。这也意味着 RoPE 可以自然地外推到更长序列（配合适当的频率缩放）。

**为什么 LLM 推理需要 RoPE：**
- 推理时序列长度动态变化，RoPE 不依赖固定位置表；
- RoPE 编码的是相对位置，与 KV Cache 的使用方式完全契合（缓存的 K 已经包含了位置信息，无需重新注入）；
- Llama3、Qwen3 均使用 RoPE，并通过 `rope_scaling` 扩展上下文窗口。

### I.1.4 代码对照：`rotary.py` 中的 RoPE 实现

**文件：`python/minisgl/layers/rotary.py`**

```python
# 第 24 行：计算频率的倒数
inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

# 第 27-28 行：预计算所有位置的 cos/sin 缓存
t = torch.arange(max_position_embeddings, dtype=torch.float)
freqs = torch.einsum("i,j -> ij", t, inv_freq)  # shape: [max_pos, rotary_dim/2]

# 第 29-32 行：合并 cos 和 sin
cos = freqs.cos()
sin = freqs.sin()
self._cos_sin_cache = torch.cat((cos, sin), dim=-1)  # shape: [max_pos, rotary_dim]
```

这里 $\theta_i = base^{-2i/d}$（即 `inv_freq` 的第 $i$ 个元素），`freqs[pos, i] = pos * theta_i`，与 Sinusoidal 位置编码的频率选取方式相同，但用途完全不同——这里不是加法注入，而是在 Q/K 上做旋转。

**前向传播（第 45-51 行）** 调用了 FlashInfer 提供的 `apply_rope_with_cos_sin_cache_inplace`，直接原位修改 Q 和 K 的数值，避免创建中间张量。

**Llama3 的频率缩放（第 72-86 行）**：通过 `rope_scaling` 配置，对低频分量进行缩放（除以 `scaling_factor`），对高频分量保持不变，中间平滑过渡。这是 Llama3 能够扩展到 128K 上下文长度的关键技术之一。`ModelConfig.from_hf`（`python/minisgl/models/config.py` 第 65-70 行）负责从 HuggingFace 配置中读取 `rope_theta` 和 `rope_scaling` 并传入 `RotaryConfig`。

---

## I.2 传统 Scaled Dot-Product Attention

### I.2.1 Q、K、V 的来源与含义

给定输入序列 $X$，形状为 $(n, d_{model})$（$n$ 个 token，每个 $d_{model}$ 维），通过三个线性变换得到：

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

其中 $W_Q$ 和 $W_K$ 的形状为 $(d_{model}, d_k)$，$W_V$ 的形状为 $(d_{model}, d_v)$。

- **Q（Query，查询）**：当前 token 想要查询什么信息，可以理解为"我现在想问什么问题"。
- **K（Key，键）**：每个 token 向外暴露的"标签"或"索引"，可以理解为"我这个 token 有什么标签，方便别人找到我"。
- **V（Value，值）**：每个 token 真正携带的信息内容，可以理解为"如果有人找到了我，我给他的答案是什么"。

注意力的直觉是：当前 token（用 Q 表示）去"检索"所有历史 token（用 K 表示），检索相关性越高的 token，从其 V 中取用的信息越多。

### I.2.2 Scaled Dot-Product Attention 的完整计算步骤

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**步骤分解：**

1. **计算注意力分数矩阵**：$S = QK^T$（形状 $(n, n)$），$S_{ij}$ 表示位置 $i$ 的 Query 和位置 $j$ 的 Key 的相似度。
2. **缩放**：除以 $\sqrt{d_k}$。原因：当 $d_k$ 较大时，内积的方差会随 $d_k$ 线性增大，导致 softmax 梯度极小（"饱和"区域）。除以 $\sqrt{d_k}$ 将方差控制在 1 附近。
3. **Causal Mask**（自回归生成时必须）：将 $S$ 的上三角部分（未来位置）设为 $-\infty$，softmax 后这些位置的权重变为 0，保证位置 $i$ 只能关注位置 $\leq i$ 的 token。
4. **Softmax**：$A = \text{softmax}(S)$（形状 $(n, n)$），将每行归一化为概率分布（各行加和为 1）。
5. **加权求和**：$O = AV$（形状 $(n, d_v)$），对所有 Value 按注意力权重做加权平均，得到每个 token 的输出表示。

### I.2.3 Multi-Head Attention：为什么要多头

单头注意力的问题：**一组 Q、K、V 只能学习一种"检索模式"**。例如，一个 token 可能同时需要关注语法依赖关系、语义相关性、位置接近性等不同维度的信息，单头难以同时捕获。

Multi-Head Attention 的做法是将 $d_{model}$ 维的特征**分成 $h$ 个头**，每个头独立学习一种注意力模式：

$$\text{head}_i = \text{Attention}(Q W_{Q,i}, K W_{K,i}, V W_{V,i})$$

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

其中每个头的维度 $d_k = d_{model} / h$，总计算量与单头相当，但表达能力更强。实践中不同头确实学到了不同的模式：有的头关注局部依赖，有的头关注远程引用，有的头关注特殊语法结构。

### I.2.4 GQA（Grouped Query Attention）：为什么 LLM 要用 GQA

**MHA、MQA、GQA 的对比：**

```
MHA（Multi-Head Attention）：每个 Q 头都有独立的 K、V 头
  Q: [h1][h2][h3][h4][h5][h6][h7][h8]
  K: [h1][h2][h3][h4][h5][h6][h7][h8]
  V: [h1][h2][h3][h4][h5][h6][h7][h8]

MQA（Multi-Query Attention）：所有 Q 头共享同一组 K、V
  Q: [h1][h2][h3][h4][h5][h6][h7][h8]
  K:                [h]
  V:                [h]

GQA（Grouped Query Attention）：Q 头分组，每组共享一对 K、V
  Q: [h1][h2] | [h3][h4] | [h5][h6] | [h7][h8]
  K:    [g1]  |    [g2]  |    [g3]  |    [g4]
  V:    [g1]  |    [g2]  |    [g3]  |    [g4]
```

**GQA 如何减少 KV Cache 显存：**

KV Cache 的显存开销正比于 **KV 头数**。以 Llama3-70B 为例（32 个 Q 头，8 个 KV 头，head_dim=128，float16）：

- MHA：每层每 token 的 KV Cache = $2 \times 32 \times 128 \times 2$ 字节 = 16 KB
- GQA（8 KV 头）：每层每 token 的 KV Cache = $2 \times 8 \times 128 \times 2$ 字节 = 4 KB，**缩小 4 倍**

对于 32 层、序列长度 4096 的批次：
- MHA：$32 \times 4096 \times 16\,\text{KB} \approx 2\,\text{GB}$
- GQA：$32 \times 4096 \times 4\,\text{KB} \approx 512\,\text{MB}$

这意味着相同的显存可以容纳 4 倍多的并发请求，或支持 4 倍长的上下文。

在 `python/minisgl/models/config.py` 中，`ModelConfig` 同时记录了 `num_qo_heads`（Q 头数）和 `num_kv_heads`（KV 头数），`AttentionLayer`（`python/minisgl/layers/attention.py` 第 29 行）强制要求 `num_qo_heads % num_kv_heads == 0`，即合法的 GQA 分组。

### I.2.5 传统实现的显存瓶颈

传统注意力实现需要将完整的注意力矩阵 $A$（形状 $(n, n)$）**显式存储在显存（HBM）中**。

**以具体数字量化：**

- 序列长度：$n = 4096$
- 头数：$h = 32$
- 数据类型：float16（2 字节）

单次前向传播中，注意力矩阵的显存占用：

$$4096 \times 4096 \times 32 \times 2 = 4096^2 \times 64 \approx 1.07 \times 10^9\ \text{bytes} \approx 1.07\ \text{GB}$$

反向传播时还需保存这个矩阵用于梯度计算，总占用翻倍到约 **2 GB**，而且这只是注意力矩阵，尚未计算 Q、K、V 等中间激活。

对于 $n = 32768$（32K 上下文），注意力矩阵本身高达 **68 GB**，已超过单张 A100（80 GB）的显存容量。这就是为什么必须使用 FlashAttention（见 I.4 节）。

### I.2.6 代码对照：`attention.py` 和 `base.py`

**文件：`python/minisgl/layers/attention.py`，`AttentionLayer.forward`（第 47-57 行）**

```python
def forward(self, qkv: torch.Tensor) -> torch.Tensor:
    ctx = get_global_ctx()
    # 将 qkv 投影结果分拆为 q, k, v
    q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
    # 可选的 QK Norm（Qwen3 使用）
    if self.q_norm is not None:
        self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
    if self.k_norm is not None:
        self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
    # 施加 RoPE 旋转位置编码
    q, k = self.rotary.forward(ctx.batch.positions, q, k)
    # reshape 为 [num_tokens, num_heads, head_dim]
    q = q.view(-1, self.num_qo_heads, self.head_dim)
    # 调用注意力后端（FA 或 FlashInfer）
    o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
    return o.view(-1, self.qo_attn_dim)
```

关键设计：`AttentionLayer` 本身**不执行**注意力计算——它只完成 QKV 拆分、归一化、RoPE 变换，然后将工作委托给 `ctx.attn_backend`。这个后端在运行时可以是 FlashAttentionBackend、FlashInferBackend，或者 HybridBackend（见 I.6 节）。

**文件：`python/minisgl/attention/base.py`**

`BaseAttnBackend` 定义了所有后端必须实现的接口：
- `forward`：执行注意力计算（写入 KV Cache 并返回输出）
- `prepare_metadata`：在每个 batch 前准备调度元数据
- `init_capture_graph` / `prepare_for_capture` / `prepare_for_replay`：支持 CUDA Graph 捕获与重放

---

## I.3 KV Cache：自回归推理的关键优化

### I.3.1 没有 KV Cache 时的计算浪费

自回归生成的过程是：给定提示词（prompt），每次生成一个新 token，然后把生成的 token 拼接到序列末尾，再生成下一个。

**若没有 KV Cache**，生成第 $t$ 个 token 时，需要对长度为 $t$ 的完整序列重新计算所有层的 Q、K、V 矩阵，并做注意力计算。生成 $T$ 个 token 的总计算量为 $O(T^2)$，大量计算是重复的（前 $t-1$ 个 token 的 K、V 在每步都被重新算一遍）。

### I.3.2 KV Cache 的核心思想

注意到：对于位置 $i < t$ 的 token，其 Key 和 Value 向量**在生成过程中不会改变**（因为 Causal Mask 保证了 token $i$ 的表示只依赖于它之前的 token，而这些 token 在生成阶段保持不变）。

因此，可以在第一次计算时把所有 token 的 K、V 缓存下来，后续每步只计算**新 token**的 K、V，然后将其追加到缓存中。生成第 $t$ 步时：

```
新 token 的 Q:           [q_t]              shape: [1, h, d_k]
KV Cache 中的 K:    [k_1, k_2, ..., k_t]    shape: [t, h, d_k]
KV Cache 中的 V:    [v_1, v_2, ..., v_t]    shape: [t, h, d_v]

注意力计算：
  分数 = q_t · K^T / sqrt(d_k)              shape: [1, h, t]
  权重 = softmax(分数)                        shape: [1, h, t]
  输出 = 权重 · V                            shape: [1, h, d_v]
```

每步计算量从 $O(t^2)$ 降为 $O(t)$，总计算量从 $O(T^2)$ 降为 $O(T^2 / 2)$（渐进等价，但常数更小，且内存访问模式更友好）。

### I.3.3 Prefill vs Decode 阶段

LLM 推理分为两个截然不同的阶段：

**Prefill（预填充）阶段：**
- 输入：完整的提示词序列（可能有几百到几千个 token）
- 计算：一次性处理所有输入 token，计算并存储它们的 K、V
- 特点：矩阵乘法密集（compute-bound），GPU 计算单元是瓶颈
- 注意力形状：Q、K、V 的形状均为 $(n,\ h,\ d_k)$，$n$ 可能达数千

**Decode（解码）阶段：**
- 输入：每次只有 1 个新 token
- 计算：新 token 的 Q 与缓存中所有 K、V 做注意力
- 特点：内存带宽密集（memory-bound），主要瓶颈是从 HBM 读取 KV Cache
- 注意力形状：Q 的形状为 $(1,\ h,\ d_k)$，K/V 形状为 $(t,\ h,\ d_k)$（$t$ 为已缓存序列长度），矩阵乘退化为矩阵-向量乘

这两种阶段的计算特性完全不同，这正是 mini-sglang 使用不同注意力后端的根本原因（见 I.6 节）。

### I.3.4 KV Cache 的显存开销与分页管理

KV Cache 的显存占用为：

$$\text{Memory} = 2 \times N_{layers} \times H_{kv} \times d_{head} \times S_{max} \times B$$

其中 $N_{layers}$ 为层数，$H_{kv}$ 为 KV 头数，$d_{head}$ 为每头维度，$S_{max}$ 为最大序列长度，$B$ 为每元素字节数（float16 = 2）。

以 Llama3-8B（32 层，8 KV 头，head\_dim=128，float16，max\_seq\_len=8192）为例：

$$2 \times 32 \times 8 \times 128 \times 8192 \times 2 = 1{,}073{,}741{,}824 \approx 1\ \text{GB}$$

即单个请求占用约 1 GB 显存。

同时处理 100 个请求就需要 100 GB，远超单 GPU 显存。

此外，不同请求的序列长度各异，如果为每个请求预先分配固定大小的 KV Cache 空间，短请求会浪费大量显存。解决方案是**分页 KV Cache**（PagedAttention）：将 KV Cache 切分为固定大小的"页"（page），按需分配，类似操作系统的虚拟内存管理。这也是为什么 FlashInfer 后端需要分页注意力支持（见 I.5 节）。

---

## I.4 FlashAttention：IO 感知的注意力计算

### I.4.1 计算机存储层次与 IO 瓶颈

现代 GPU 的存储层次从快到慢：

```
寄存器（Registers）：~数 PB/s（per SM），极小容量
片上 SRAM（Shared Memory/L1 Cache）：~19 TB/s，A100 约 192 KB/SM
HBM（High Bandwidth Memory，显存）：~2 TB/s，A100 约 80 GB
```

SRAM 的带宽比 HBM **快约 10 倍**，但容量只有 HBM 的几十万分之一。GPU 的计算吞吐远高于 HBM 带宽，因此许多操作的瓶颈在于 **HBM 的读写次数**，而非实际浮点计算量。

**标准 Attention 的 IO 分析（前向传播）：**

对于序列长度 $n$，头维度 $d_k$，标准实现需要以下 HBM 操作：

| 操作 | 读 | 写 |
|------|----|----|
| $S = QK^T / \sqrt{d_k}$ | $Q, K$：$O(nd)$ | $S$：$O(n^2)$ |
| $A = \text{softmax}(S)$ | $S$：$O(n^2)$ | $A$：$O(n^2)$ |
| $O = AV$ | $A, V$：$O(n^2 + nd)$ | $O$：$O(nd)$ |

注意力矩阵 $S$ 和 $A$ 需要反复读写 HBM，总 IO 复杂度为 **$O(n^2)$**。当 $n = 4096$，每个矩阵约 128 MB（float16，32 头）；当 $n = 32768$，则高达 **8 GB**，IO 代价极高。

**反向传播的问题：** 标准实现需要保存 $A$（注意力权重矩阵）用于反向传播，这就是前文提到的 $O(n^2)$ 显存瓶颈。

### I.4.2 FlashAttention v1：分块计算 + Online Softmax

FlashAttention（Dao et al., 2022）的核心洞察：**$Q$, $K$, $V$ 可以分块加载到 SRAM 中计算，只要能正确地增量计算 softmax**。

**Online Softmax（增量 Softmax）推导：**

标准 softmax：$\text{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$

数值稳定版本：先减去最大值 $m = \max_j x_j$，再计算：$\text{softmax}(x)_i = e^{x_i - m} / \sum_j e^{x_j - m}$

问题：减最大值需要先**看完所有元素**才知道最大值是多少，无法分块。

**Online Softmax 的解决方案**：维护滚动最大值 $m$ 和滚动归一化因子 $\ell$，每处理一个新块时更新。

设已处理 $k$ 个元素时的统计量为 $(m^{(k)}, \ell^{(k)})$，新来一块元素 $x_{k+1}, \ldots, x_{k+B}$：

$$m^{(k+1)} = \max(m^{(k)}, \max_j x_{k+j})$$

$$\ell^{(k+1)} = e^{m^{(k)} - m^{(k+1)}} \cdot \ell^{(k)} + \sum_j e^{x_{k+j} - m^{(k+1)}}$$

注意 $e^{m^{(k)} - m^{(k+1)}}$ 是一个**修正因子**：当发现了更大的最大值 $m^{(k+1)}$，历史的归一化因子 $\ell^{(k)}$ 需要重新缩放。

最终 $\text{softmax}(x)_i = e^{x_i - m^{(\text{final})}} / \ell^{(\text{final})}$。

**分块（Tiling）计算流程：**

```
将 Q 按行分块（每块 Br 行），K、V 按列分块（每块 Bc 列）

外层循环（按 Q 块）：
  内层循环（按 K、V 块）：
    1. 从 HBM 加载 K_j, V_j 到 SRAM
    2. 计算 S_ij = Q_i K_j^T / sqrt(d_k)
    3. 更新 m_i, l_i（Online Softmax）
    4. 更新 O_i += softmax_corrected(S_ij) V_j

  写出 O_i 到 HBM
```

这样，注意力矩阵 $S$ 和 $A$ **永远不需要写回 HBM**，而是在 SRAM 内计算完就丢弃，IO 复杂度从 $O(n^2)$ 降为 $O(n)$（只读写 $Q$, $K$, $V$, $O$）。

**显存节省：** FlashAttention 不存储 $O(n^2)$ 的注意力矩阵，只存储每个 Q 块的 $m$（最大值）和 $\ell$（归一化因子），共 $O(n)$ 额外存储。

**反向传播的重计算（Recomputation）：**

既然前向传播不保存注意力矩阵，反向传播怎么办？FlashAttention 的做法是**用 $O(n)$ 的 SRAM 中间结果（即 $m$ 和 $\ell$）在反向传播时重新计算注意力矩阵**。虽然增加了计算量（约 1.5 倍 FLOPS），但大幅减少了 HBM IO 和峰值显存，总体上更快。

### I.4.3 FlashAttention v2 的改进

FA v2（Dao, 2023）在 v1 基础上做了以下优化：

1. **减少非矩阵乘法操作**：v1 的 rescale 操作（更新 $O$ 时乘以修正因子）引入了额外的逐元素乘法，v2 通过改变循环顺序和延迟归一化，将非 matmul 操作量减少约 2 倍。
2. **按 Query 而非 Key-Value 分块并行**：v1 的外层循环按 K、V 块并行（适合多线程），但会导致 Q 的更新需要跨线程归约；v2 改为按 Q 块并行，每个线程块独立处理一段 Q，输出可以直接写出，无需归约。这更好地适应了现代 GPU 的计算特性。
3. **更高的 GPU 利用率**：在 A100 上，FA v2 可达到约 72% 的理论 FP16 峰值算力，而 FA v1 约为 55%。

### I.4.4 FlashAttention v3/v4 简介

**FlashAttention v3**（针对 Hopper 架构，即 H100/H200）：
- 利用 Hopper 的 **Tensor Memory Accelerator（TMA）** 进行异步数据加载，将 GEMM 与数据搬运重叠执行（pipelining）。
- 利用 **WGMMA（Warpgroup Matrix Multiply Accumulate）** 指令，充分利用 Hopper 的第四代 Tensor Core。
- 在 H100 上实现约 **1.5-2 倍** 于 FA v2 的速度提升。

**FlashAttention v4**（针对 Blackwell 架构，即 B100/B200/GB200）：
- 完全基于 Blackwell 的 TMEM（Tensor Memory）和新型矩阵乘法指令重新设计。
- 进一步提升异步流水线效率。

在 mini-sglang 的 `fa.py`（第 46 行）中可以看到版本自动选择逻辑：

```python
self.version = 4 if is_sm100_supported() else 3
```

`sm100` 对应 Blackwell 架构（计算能力 10.0），在 Hopper 及以下使用 v3。

### I.4.5 代码对照：`fa.py` 中的 FlashAttention 调用

**文件：`python/minisgl/attention/fa.py`**

**`_fa_sgl_impl` 函数（第 139-182 行）** 封装了对 `sgl_kernel.flash_attn.flash_attn_with_kvcache` 的调用。关键参数说明：

**`cu_seqlens_q` 和 `cu_seqlens_k`（cumulative sequence lengths）：**

这两个参数是理解变长序列打包（sequence packing）的关键。

在批处理中，不同请求的序列长度不同。朴素做法是 padding 到最大长度，但这会浪费计算。更高效的做法是**把所有序列拼接成一个长序列**，然后用 `cu_seqlens` 告诉 FlashAttention 每个请求的边界在哪里。

以 3 个请求、Q 长度分别为 `[3, 1, 2]` 为例：

```
cu_seqlens_q = [0, 3, 4, 6]  # 前缀和，长度 = bs + 1
                ↑  ↑  ↑  ↑
                |  |  |  └── 所有序列结束（总长 6）
                |  |  └───── 第 2 个序列结束（偏移 4）
                |  └──────── 第 1 个序列结束（偏移 3）
                └─────────── 起始（偏移 0）
```

FlashAttention 通过这个偏移数组知道"Q 张量中位置 0-2 属于请求 0，位置 3 属于请求 1，位置 4-5 属于请求 2"，从而正确地对每个请求施加 Causal Mask（请求之间的 token 不互相关注）。

**Prefill 和 Decode 的不同路径（`prepare_metadata`，第 84-90 行）：**

```python
if max_seqlen_q == 1:
    # Decode：每个请求只有 1 个新 token，cu_seqlens_q = [0, 1, 2, ..., bs]
    cu_seqlens_q = torch.arange(0, padded_size + 1, ...)
elif all(l == 0 for l in cached_lens):
    # Prefill（无 cache 命中）：Q 和 K 长度相同，复用 cu_seqlens_k
    cu_seqlens_q = cu_seqlens_k
else:
    # Extend Prefill（部分 cache 命中）：Q 是 K 的子集
    cu_seqlens_q = torch.tensor([0] + seqlens_q, ...).cumsum_(dim=0)
```

三条路径对应推理的三种场景，正确处理了 KV Cache 部分命中的情况。

---

## I.5 FlashInfer：面向推理的分页注意力

### I.5.1 推理场景的特殊性

FlashAttention 假设 Q、K、V 都是**连续内存**中的张量，非常适合训练（每次处理固定批次）。但推理场景更复杂：

- KV Cache 是**分页存储**的，不同 token 的 K、V 可能散落在显存各处（类似操作系统的分页内存）。
- 每步 Decode 的序列长度不同，批次中不同请求的 KV Cache 大小各异。
- 需要高效支持**单 token Query**（Decode 阶段）的注意力计算。

FlashInfer（Ye et al., 2024）专门为这些推理场景设计，核心特性是：**原生支持分页 KV Cache 的注意力计算**，并针对 Decode 的矩阵-向量乘特性做了深度优化。

### I.5.2 Plan/Run 两阶段 API

FlashInfer 将注意力计算分为两个阶段：

**Plan（规划）阶段：**
- 接收序列长度、页表等元数据（全在 CPU 上）
- 计算 CUDA Kernel 的调度参数：每个 thread block 处理哪些 token，如何并行
- 将调度元数据上传到 GPU
- **不执行实际矩阵计算**，只做元数据准备

**Run（执行）阶段：**
- 接收实际的 Q 和 KV Cache 张量
- 根据 Plan 阶段准备好的调度元数据，直接执行注意力计算
- 返回输出张量

这种分离的好处：Plan 阶段的 CPU 开销可以与 GPU 计算**异步重叠**（`non_blocking=True`），不阻塞推理流水线。

### I.5.3 代码对照：`fi.py` 中的 FlashInfer 使用

**文件：`python/minisgl/attention/fi.py`**

**Plan 阶段（`_initialize_metadata_once`，第 122-161 行）：**

```python
@staticmethod
def _initialize_metadata_once(metadata: FIMetadata) -> None:
    if metadata.initialized:
        return  # 每个 batch 只 plan 一次
    metadata.initialized = True

    if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
        # Decode 路径
        metadata.wrapper.plan(
            indptr=metadata.cu_seqlens_k_cpu,   # 每个请求的 KV 偏移（CPU 上）
            indices=metadata.indices,            # 每个 token 对应的页号
            last_page_len=metadata.last_page_len_cpu,  # 最后一页的有效长度
            ...
        )
    else:
        # Prefill 路径
        metadata.wrapper.plan(
            qo_indptr=metadata.cu_seqlens_q_cpu,       # Q 的序列偏移
            paged_kv_indptr=metadata.cu_seqlens_k_cpu, # KV 的序列偏移
            paged_kv_indices=metadata.indices,          # 页号数组
            ...
            causal=True,  # 因果掩码
        )
```

**Run 阶段（`forward`，第 171-183 行）：**

```python
def forward(self, q, k, v, layer_id, batch):
    ...
    self._initialize_metadata_once(metadata)  # Plan（幂等，只运行一次）
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)  # 写入 KV Cache
    kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))
    return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)  # Run
```

**`indices` 参数的含义：**

`indices` 是一个扁平数组，记录每个 token（page_size=1 时一个 token 对应一页）在 KV Cache 存储中的物理位置索引。例如，如果请求 0 的 3 个 token 被分配到 KV Cache 的第 [2, 5, 7] 个槽位，请求 1 的 2 个 token 被分配到第 [0, 3] 个槽位，则：

```
cu_seqlens_k = [0, 3, 5]
indices       = [2, 5, 7, 0, 3]
```

FlashInfer 通过 `cu_seqlens_k` 和 `indices` 就能正确定位每个请求的 KV Cache，无需连续内存。

**`use_tensor_cores`（第 236-237 行）：**

```python
GQA = self.config.num_qo_heads // self.config.num_kv_heads
return GQA >= 4
```

当 GQA 比率较高（Q 头数是 KV 头数的 4 倍或以上）时，FlashInfer 的 Tensor Core 路径（专门优化的矩阵乘法）比标准路径更快。这是因为 GQA 使得 Q 的数据量相对于 KV 更大，更适合 Tensor Core 的矩阵乘法模式。

### I.5.4 FlashInfer vs FlashAttention 的对比

| 特性 | FlashAttention (FA) | FlashInfer (FI) |
|------|---------------------|-----------------|
| 主要场景 | Prefill（连续长序列） | Decode（分页、短序列多请求） |
| KV Cache 布局 | 连续内存 | 分页（Paged） |
| Query 长度 | 可以很长 | 通常为 1（Decode） |
| 计算特性 | Compute-bound | Memory-bound |
| 序列打包 | cu_seqlens 前缀和 | indptr + indices |
| Plan/Run 分离 | 无 | 有（CPU/GPU 异步） |

---

## I.6 Hybrid Backend：两种 Attention 的协同

### I.6.1 为什么要用两种后端

Prefill 和 Decode 阶段的计算特性决定了它们需要不同的优化策略：

**Prefill 使用 FlashAttention 的原因：**
- Prefill 的 Q 序列较长（等于输入长度），注意力是**全矩阵**计算，FlashAttention 的分块 tiling 策略在这里发挥最大优势。
- Prefill 时 KV Cache 尚未完全填充，内存访问模式较为规则，FA 的连续内存假设成立。
- 对于非常长的 Prefill 序列，FA 的 $O(n)$ IO 复杂度优势尤为突出。

**Decode 使用 FlashInfer 的原因：**
- Decode 时 Q 只有 1 个 token，注意力退化为**向量-矩阵乘**（GEMVs），FlashInfer 专门针对这种模式优化。
- KV Cache 是分页的，FlashInfer 直接支持 paged KV，无需额外的内存重排。
- FlashInfer 的 Plan/Run API 允许将调度开销与 GPU 计算重叠。

### I.6.2 HybridBackend 的路由逻辑

**文件：`python/minisgl/attention/base.py`，`HybridBackend`（第 37-63 行）**

`HybridBackend` 的实现极其简洁：

```python
class HybridBackend(BaseAttnBackend):
    def forward(self, q, k, v, layer_id, batch):
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch):
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)
```

路由判据只有一个：`batch.is_prefill`。每个 batch 要么是纯 Prefill（处理输入序列），要么是纯 Decode（生成一个新 token），不会混合。因此路由是确定性的 `if-else`，无需复杂逻辑。

### I.6.3 CUDA Graph 与注意力后端的兼容性

CUDA Graph 是一种将 GPU Kernel 序列"录制"为图结构，后续直接重放的优化技术，可以大幅减少 CPU 调度开销（从每次生成约 1-2 ms 降至微秒级）。

然而 CUDA Graph 要求所有 GPU 操作**完全静态**：相同的 Kernel，相同的内存地址（可以修改内存内容，但不能改变地址）。这与 Decode 阶段的需求吻合（batch size 固定，KV Cache 的指针固定），但与 Prefill 不兼容（序列长度每次不同）。

因此，在 `HybridBackend` 的 CUDA Graph 接口中（第 56-63 行）：

```python
def init_capture_graph(self, max_seq_len, bs_list):
    self.decode_backend.init_capture_graph(max_seq_len, bs_list)
    # prefill_backend 没有 CUDA Graph 支持

def prepare_for_capture(self, batch):
    self.decode_backend.prepare_for_capture(batch)

def prepare_for_replay(self, batch):
    self.decode_backend.prepare_for_replay(batch)
```

**CUDA Graph 只在 Decode 后端上启用**。`FlashInferBackend` 使用专门的 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`（`fi.py` 第 245 行），它将 `indptr_buffer`、`indices_buffer` 等参数绑定到**固定的预分配张量**，CUDA Graph 重放时直接更新这些张量的内容（通过 `prepare_for_replay`），而不改变张量的地址。

整个系统的设计链条是：**GQA 减少 KV Cache 显存 → 分页 KV Cache 支持更多并发 → FlashInfer 高效处理分页 Decode → CUDA Graph 消除 Decode 的 CPU 调度开销**，各个优化环环相扣，共同支撑了高效的 LLM 推理。

---

## 小结

本附录从 Transformer 的基础架构出发，沿着推理优化的需求链条逐层深入：

1. **RoPE** 解决了长上下文下的位置编码外推问题，是现代 LLM 可以处理超长序列的基础。
2. **GQA** 通过共享 KV 头将 KV Cache 显存缩减数倍，是大规模部署的关键。
3. **KV Cache** 将自回归推理的计算复杂度从 $O(T^2)$ 降至 $O(T)$，是 LLM 推理可行的前提。
4. **FlashAttention** 通过分块计算和 Online Softmax 将注意力计算的 HBM IO 从 $O(n^2)$ 降至 $O(n)$，解决了长序列的显存和速度瓶颈。
5. **FlashInfer** 专为分页 KV Cache 和 Decode 场景优化，补充了 FlashAttention 在推理场景的不足。
6. **HybridBackend** 将两者有机结合，配合 CUDA Graph，实现了从 Prefill 到 Decode 的全流程高效推理。

这些技术共同构成了 mini-sglang 推理引擎的注意力计算基础设施。
