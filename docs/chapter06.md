# 第 6 章：分页 KV 缓存——显存管理的操作系统思想

## 6.1 背景问题：KV Cache 的显存碎片化困境

在自回归生成中，每一步 decode 都需要访问当前序列所有历史 token 的 K/V 矩阵，因此将已计算的 KV 保存在显存中是提升吞吐的基本手段。然而，当系统需要同时服务多个请求时，朴素的 KV Cache 方案会暴露出一个致命问题：**显存碎片化**。

朴素方案的做法是：在请求进入时，按照该请求的最大可能长度（`max_seq_len`）预先分配一块连续的显存区域，用于存放整个生命周期内的所有 KV 向量。这个方案简单直接，但存在两个根本性缺陷：

1. **内部碎片**：绝大多数请求的实际生成长度远小于 `max_seq_len`，提前分配的显存有大量浪费。一个允许生成 4096 个 token 但实际只生成了 200 个 token 的请求，浪费了超过 95% 的预留显存。

2. **无法共享**：多个请求可能拥有完全相同的系统提示（system prompt）或对话历史前缀，但朴素方案为每个请求独立分配空间，导致相同的 KV 数据在显存中存在多份副本。

这两个问题叠加，在高并发服务场景下会急剧降低显存利用率，进而限制系统的最大吞吐量。

解决方案正是来自操作系统领域一个有四十年历史的经典思想：**分页（Paging）**。

---

## 6.2 核心概念：从虚拟内存分页到 KV Cache 分页

### 操作系统的分页回顾

操作系统将物理内存划分为固定大小的**页帧（Page Frame）**，将进程的虚拟地址空间划分为等大的**虚拟页（Virtual Page）**，通过**页表（Page Table）**维护从虚拟页到物理页帧的映射。进程看到的是一段连续的虚拟地址空间，而实际的物理内存可以是零散分布的。

这个机制带来了两个关键好处：
- **按需分配**：只有真正被访问的页才占用物理内存。
- **共享**：多个进程可以映射到同一个物理页帧（如共享库的代码段）。

### KV Cache 的分页类比

mini-sglang 将同样的思想搬到了 KV Cache 管理：

| 操作系统概念 | KV Cache 对应概念 |
|---|---|
| 物理内存页帧 | KV Cache 中的一个 Page（`page_size` 个连续 token 槽位）|
| 虚拟地址空间 | 某个请求的逻辑序列位置（0, 1, 2, ...）|
| 页表 | `page_table` 张量（逻辑位置 → 物理槽位编号）|
| 物理帧分配器 | `CacheManager.free_slots`（空闲页的列表）|
| 换出（Swap Out）| 驱逐（Evict）：将不活跃的 KV 页释放以腾出空间 |
| 共享内存映射 | 基数树前缀缓存：相同前缀的请求共享同一批物理 KV 页 |

这一类比不是表面上的相似，而是在系统设计层面的深度对应。理解了操作系统的虚拟内存，就理解了分页 KV Cache 的全部精髓。

---

## 6.3 核心代码导读

### 6.3.1 物理存储层：`MHAKVCache`

**文件**：`python/minisgl/kvcache/mha_pool.py`

`MHAKVCache` 是 KV Cache 的**物理存储**，可类比为操作系统中的"物理内存条"。

```python
# mha_pool.py，第 28-32 行
self._kv_buffer = torch.empty(
    (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
    device=device,
    dtype=dtype,
)
```

这个六维张量是整个系统 KV 存储的物理基底，各维度含义如下：

- **维度 0，大小 2**：K 和 V 两个缓冲，`_kv_buffer[0]` 为 K，`_kv_buffer[1]` 为 V。
- **维度 1，`num_layers`**：模型的层数，每层的 KV 独立存储。
- **维度 2，`num_pages`**：物理页的总数，是整个系统显存容量的体现。
- **维度 3，`page_size`**：每页能存放的 token 数，是分页粒度的核心参数。
- **维度 4，`local_kv_heads`**：KV Head 数量（张量并行已分片）。
- **维度 5，`head_dim`**：每个 Head 的维度。

`page_size` 是影响系统行为的关键旋钮：设为 1 则退化为逐 token 管理（最灵活，管理开销最大）；设为 16 或 32 则一次以多个 token 为单位分配和回收（开销低，但内部碎片增加）。

`store_kv` 方法（第 45-56 行）在写入时将缓冲区 reshape 成 `(num_pages * page_size, local_kv_heads, head_dim)` 的形式，使得页号和页内偏移合并为单一的**物理槽位编号（slot index）**，Attention 内核通过这个整数直接寻址。

### 6.3.2 逻辑到物理的映射层：`TableManager`

**文件**：`python/minisgl/scheduler/table.py`

```python
# table.py，第 5-11 行
class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)
```

`TableManager` 管理两个核心数据结构：

- **`token_pool`**：形状为 `[max_running_reqs, max_seq_len]` 的整数张量，存放每个活跃请求的 input token ID。行索引 `table_idx` 是请求在系统中的"逻辑行号"，是请求与系统其他组件交互的标识符。

- **`page_table`**：形状相同，但存放的是**物理槽位编号**而非 token ID。`page_table[table_idx, pos]` 的值 `s` 表示：第 `table_idx` 号请求在序列位置 `pos` 处的 KV 数据，存放在 `MHAKVCache._kv_buffer[:, :, s // page_size, s % page_size, :, :]` 中。

这两个张量共享相同的"坐标系"（行 = 请求，列 = 序列位置），但分别承载 token 语义（token ID）和内存语义（物理槽位），是整个系统逻辑地址空间的具象体现。

### 6.3.3 物理页分配与回收：`CacheManager`

**文件**：`python/minisgl/scheduler/cache.py`

`CacheManager` 是整个 KV Cache 的**操作系统内核**，负责物理页的分配、回收和前缀缓存管理。

**空闲页跟踪**（第 20 行）：

```python
self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
```

`free_slots` 是一个一维张量，存放所有空闲页的**起始槽位编号**（即每页首个 token 的槽位号）。当 `page_size=16` 时，初始内容形如 `[0, 16, 32, 48, ...]`。这种"只记录页起点"的设计巧妙地将页内连续性隐含其中，无需显式存储每页的所有槽位。

**分配流程**（第 106-113 行）：`_allocate` 方法先检查空闲页是否充足，若不足则驱逐前缀缓存中的页面。分配完成后，通过 `_page_to_token` 将页起点展开成该页所有 token 的槽位号，再由 `_write_page_table` 填写 `page_table` 张量。

**写回 page_table**（第 127-146 行）：

```python
# cache.py，第 144-146 行
table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
offsets = positions_host.to(page_table.device, non_blocking=True)
page_table[table_idxs, offsets] = allocated
```

这里使用了 GPU 张量的高级索引（fancy indexing）进行批量赋值，将一次调度中所有请求的新分配页一并写入 `page_table`，避免逐请求的循环。

### 6.3.4 前缀共享层：`RadixPrefixCache`

**文件**：`python/minisgl/kvcache/radix_cache.py`

这是整个系统中最精巧的数据结构。`RadixPrefixCache` 用一棵**基数树（Radix Tree）**来索引已计算的 KV 页面，使得具有公共前缀的请求能够共享这些页面的物理存储。

**基数树节点**（第 17-84 行）：每个 `RadixTreeNode` 存储一段连续 token 序列（`_key`）及其对应的物理槽位列表（`_value`）。树的根节点是空的起点，每条从根到叶的路径代表一个被缓存的完整前缀。

**键函数设计**（第 233-236 行）：

```python
def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
```

子节点的字典键（`children` 的 key）是一页中第一个 token ID（`page_size=1`）或前 `page_size` 个 token ID 的元组。这确保了树的分支精度与 `page_size` 对齐，避免在页中间切分导致的对齐问题。

**前缀匹配**（第 205-230 行）：`_tree_walk` 方法从根节点出发，依次按页边界对齐地向下查找，找到最长匹配的前缀节点及其长度。如果一个节点只有部分匹配，则调用 `split_at` 将该节点一分为二（类似操作系统中"块分裂"），保证树的结构精确对应到页边界。

**前缀插入**（第 136-146 行）：`insert_prefix` 在请求完成 prefill 后被调用，将新计算的 KV 页面插入基数树，供后续请求复用。注意插入长度强制对齐到 `page_size`（通过 `align_down`），尾部不满一页的 token 不参与缓存，避免部分页被共享后引发正确性问题。

### 6.3.5 LRU 驱逐策略（第 148-175 行）

当空闲页不足时，`evict` 方法负责从基数树中驱逐最旧的叶节点：

```python
# radix_cache.py，第 155-173 行
leave_nodes = self._collect_leave_nodes_for_evict()
heapq.heapify(leave_nodes)
# ...
while evicted_size < size:
    node = heapq.heappop(leave_nodes)
    # 驱逐后，若父节点变成叶节点且引用计数为 0，继续加入候选堆
    if parent.is_leaf() and parent.ref_count == 0:
        heapq.heappush(leave_nodes, parent)
```

驱逐策略的核心逻辑：

1. **只驱逐叶节点**：叶节点的 KV 数据没有被任何其他缓存节点依赖，可以安全删除。
2. **时间戳排序（近似 LRU）**：`RadixTreeNode` 实现了 `__lt__` 方法（第 83-84 行）按 `timestamp` 排序，最旧的节点优先被驱逐。每次通过 `_tree_walk` 访问节点时都会更新其 `timestamp`（第 228 行），因此最近被命中的前缀会被保留。
3. **引用计数保护**：`ref_count > 0` 的节点正在被活跃请求使用，不可驱逐。`lock_handle` / `unlock_handle` 方法负责维护引用计数，防止正在使用的 KV 数据被误驱逐。
4. **级联删除**：驱逐一个叶节点后，其父节点可能变成新的叶节点，若父节点同样未被引用，则进入候选队列，实现路径上的级联回收。

---

## 6.4 设计决策：为什么这样实现

### 为什么选择基数树而不是哈希表？

最直观的前缀缓存实现是哈希表：以 token ID 序列的哈希值为键，存储对应的 KV 页面指针。但哈希表的问题在于：如果一个请求的前缀是 `[A, B, C, D]`，另一个是 `[A, B, C, D, E, F]`，哈希表无法自动识别后者是前者的扩展，二者会被独立存储，导致 `[A, B, C, D]` 对应的 KV 页被重复存储两份。

基数树天然表达了"前缀包含"关系：短前缀节点是长前缀节点的祖先，共享 KV 页面只需让二者指向同一棵子树的节点即可。驱逐时从叶节点开始也保证了数据引用的完整性。

### `page_size > 1` 的取舍

`page_size = 1` 最灵活，但每个 token 都需要一个独立的槽位条目，`page_table` 管理开销和 GPU kernel 的 gather 操作代价都更高。`page_size = 16` 时，一次分配/回收涉及 16 个连续槽位，管理开销降低 16 倍，同时 memory access pattern 更规整，有利于 GPU cache line 利用率。代价是最多浪费 `page_size - 1 = 15` 个槽位的内部碎片。实际部署中，`page_size` 通常设为 16 或 32。

### 引用计数 vs. 标记清除

选择引用计数（`ref_count`）而非垃圾回收式的标记清除，原因在于 LLM 推理的实时性要求：不能在关键的调度循环中触发全量扫描。引用计数的增减操作是 O(树深度) 的，且都发生在调度器的 CPU 侧，不阻塞 GPU 计算流。

### `lazy_free_region` 上下文管理器

`cache.py` 第 93-104 行的 `lazy_free_region` 是一个微妙的优化。在处理一个 batch 的结果时，多个请求可能同时完成，需要归还 KV 页面。如果每次 `_free` 都立即 `torch.cat`，会产生多次 CPU 内存分配。`lazy_free_region` 将所有待释放的槽位收集到 `lazy_free_list`，在 `finally` 块中一次性 `torch.cat` 合并，减少内存碎片和分配次数。

---

## 6.5 动手实验

### 实验一：观察分页分配的物理布局

在 Python 交互式环境中，直接实例化核心组件，观察 `page_table` 的填充过程：

```python
import torch
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.cache import CacheManager

device = torch.device("cpu")  # 用 CPU 方便调试
page_size = 4
num_pages = 16
max_running_reqs = 4
max_seq_len = 64

# 构造 page_table（共享张量）
page_table = torch.zeros(max_running_reqs, max_seq_len, dtype=torch.int32, device=device)

table_mgr = TableManager(max_running_reqs, page_table)
cache_mgr = CacheManager(num_pages, page_size, page_table, type="naive")

print("初始 free_slots:", cache_mgr.free_slots)
# 期望输出: tensor([ 0,  4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60])
```

**验证点**：`free_slots` 中每个值均为 `page_size` 的倍数，确认了"以页为单位"的分配粒度。

接着模拟一次分配：

```python
from minisgl.core import Req
from minisgl.kvcache.naive_cache import NaivePrefixCache

# 模拟一个长度为 10 的请求，需要 ceil(10/4) = 3 页
table_idx = table_mgr.allocate()
fake_handle = NaivePrefixCache(device).match_prefix(torch.tensor([])).cuda_handle

from minisgl.core import Req
req = Req(
    input_ids=torch.arange(10, dtype=torch.int32),
    table_idx=table_idx,
    cached_len=0,
    output_len=5,
    uid=0,
    cache_handle=fake_handle,
    sampling_params=None,
)
req._device_len = 10  # 模拟已 prefill 10 个 token

cache_mgr.allocate_paged([req])
print("分配后 page_table[0, :12]:", page_table[0, :12])
# 期望：前 12 个位置（3 页 × 4）填入了物理槽位编号
print("剩余 free_slots 数量:", len(cache_mgr.free_slots))
# 期望：16 - 3 = 13
```

### 实验二：验证基数树前缀共享

```python
import torch
from minisgl.kvcache.radix_cache import RadixPrefixCache

# 需要先设置 global ctx 中的 page_size，这里用 monkey-patch 模拟
import minisgl.core as _core
_core._global_ctx = type("Ctx", (), {"page_size": 4})()

device = torch.device("cpu")
cache = RadixPrefixCache(device)

# 模拟两个共享前缀的请求
ids_A = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32)      # 请求 A
ids_B = torch.tensor([1, 2, 3, 4, 9, 10, 11, 12], dtype=torch.int32)   # 请求 B（前 4 个 token 相同）

# 为请求 A 分配槽位并插入缓存
slots_A = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
result_A = cache.insert_prefix(ids_A, slots_A)
print("A 插入后缓存大小:", cache.size_info)

# 请求 B 匹配前缀
match_B = cache.match_prefix(ids_B)
print("B 匹配到的前缀长度:", match_B.cuda_handle.cached_len)
# 期望：4（第一页 [1,2,3,4] 命中）
matched_indices = match_B.cuda_handle.get_matched_indices()
print("B 命中的物理槽位:", matched_indices)
# 期望：tensor([0, 1, 2, 3])，与请求 A 的前 4 个槽位完全相同
```

**关键观察**：`matched_indices` 与 `slots_A[:4]` 完全相同，证明两个请求共享了同一批物理 KV 槽位，没有任何复制。

### 进阶实验：观察 LRU 驱逐的级联效果

```python
# 继续上面的 cache 实例
# 再插入一个更长的前缀让树有层次结构
ids_C = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16], dtype=torch.int32)
slots_C = torch.tensor(list(range(16)), dtype=torch.int32)
cache.insert_prefix(ids_C, slots_C)

print("插入 C 后可驱逐大小:", cache.size_info.evictable_size)

# 驱逐 4 个槽位（1 页）
evicted = cache.evict(4)
print("驱逐的槽位:", evicted)
print("驱逐后可驱逐大小:", cache.size_info.evictable_size)
```

观察被驱逐的是哪个叶节点，以及驱逐后树的结构变化。尝试修改各请求的访问顺序，验证时间戳最旧的节点优先被驱逐。

---

## 6.6 小结

本章围绕 LLM 推理中的显存管理问题，系统讲解了 mini-sglang 的分页 KV Cache 机制：

1. **物理存储层**（`MHAKVCache`）：将 KV 矩阵组织为 `[num_pages, page_size, ...]` 的分页张量，`page_size` 控制分配粒度与内部碎片的权衡。

2. **逻辑映射层**（`TableManager` + `page_table`）：`token_pool` 存储 token ID，`page_table` 存储物理槽位编号，二者共享相同的二维坐标系（请求 × 序列位置），实现逻辑地址到物理地址的解耦。

3. **分配与回收**（`CacheManager`）：`free_slots` 以页为单位跟踪空闲空间，`_allocate` / `_free` 实现按需分配，`lazy_free_region` 优化批量释放的开销。

4. **前缀共享**（`RadixPrefixCache`）：基数树以 token ID 序列为键组织已缓存的 KV 页面，相同前缀的请求共享同一批物理槽位，避免重复计算。

5. **LRU 驱逐**：通过引用计数区分可驱逐与受保护节点，以时间戳为优先级从叶节点开始驱逐，支持级联回收，保证了显存的动态复用。

**与后续章节的连接**：`page_table` 中存储的物理槽位编号是连接调度器与推理内核的"合同"——第 7 章将看到 Attention 内核如何根据这些槽位编号，在非连续的物理内存上正确完成分页注意力计算（Paged Attention）。第 8 章将讨论 prefill 与 decode 的调度策略，届时会看到 `CacheManager.available_size` 和 `prefix_cache.size_info` 如何影响调度器决定接受哪些请求进入 prefill 队列。
