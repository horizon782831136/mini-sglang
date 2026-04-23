# 附录E：KV 缓存（kvcache/）

本附录详细说明 `python/minisgl/kvcache/` 目录下五个文件的实现，包括抽象基类接口、MHA KV 缓存物理存储、基数树前缀缓存和朴素前缀缓存。

---

## E.1 `kvcache/__init__.py` — 工厂函数与注册表

### 文件职责

提供 KV 缓存池和前缀缓存的工厂函数，并通过注册表机制支持按名称动态选择前缀缓存实现。

### 公开函数

#### `create_kvcache_pool`

```python
def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> BaseKVCachePool:
```

**参数：**
- `model_config`：模型配置，提供 `num_kv_heads`、`num_layers`、`head_dim`
- `num_pages`：总页数（已加 1 个 dummy page，由 Engine 负责加上）
- `page_size`：每页包含的 token 数
- `dtype`：存储精度（如 `bfloat16`）
- `device`：目标 GPU 设备

**返回值：** `MHAKVCache` 实例（当前仅支持 MHA；注释中标注了未来支持 MLA 的扩展点）

**功能：** 创建 MHA KV 缓存池，内部直接构造 `MHAKVCache`，TP 头数切分在 `MHAKVCache.__init__` 内部完成。

#### `create_prefix_cache`

```python
def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
```

**参数：**
- `device`：目标设备
- `type`：缓存类型字符串，当前支持 `"naive"` 和 `"radix"`

**返回值：** 对应类型的 `BasePrefixCache` 实例

**功能：** 通过注册表 `SUPPORTED_CACHE_MANAGER` 按名称查找并调用对应的工厂函数。

#### 注册表

```python
SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")

@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device) -> NaivePrefixCache: ...

@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device) -> RadixPrefixCache: ...
```

`CacheManagerCreator` 是协议类型，约束注册函数的签名为 `(device: torch.device) -> BasePrefixCache`。

---

## E.2 `kvcache/base.py` — 抽象基类

### 文件职责

定义 KV 缓存系统的全部抽象接口，包括物理存储池（`BaseKVCachePool`）、逻辑前缀缓存（`BasePrefixCache`）、缓存句柄（`BaseCacheHandle`）以及相关数据结构。

### 公开类

#### `BaseKVCachePool`（抽象类）

```python
class BaseKVCachePool(ABC):
```

KV 缓存物理存储层的接口。负责 key/value tensor 的实际分配和读写，与分页机制直接交互。

**抽象方法：**

```python
@abstractmethod
def k_cache(self, index: int) -> torch.Tensor: ...
```
获取第 `index` 层的 key 缓存 tensor，返回形状为 `(num_pages, page_size, local_kv_heads, head_dim)`。

```python
@abstractmethod
def v_cache(self, index: int) -> torch.Tensor: ...
```
获取第 `index` 层的 value 缓存 tensor，形状同上。

```python
@abstractmethod
def store_kv(
    self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
) -> None: ...
```
将当前步骤计算出的 key/value 写入缓存指定位置：
- `k`、`v`：当前步骤的 key/value，shape `(seq_len, num_heads, head_dim)`
- `out_loc`：目标物理位置索引，shape `(seq_len,)`，每个值是展平后的 token slot 编号
- `layer_id`：当前 transformer 层索引

**抽象属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `device` | `torch.device` | 缓存所在设备 |
| `dtype` | `torch.dtype` | 缓存数据类型 |
| `num_layers` | `int` | transformer 层数 |

#### `BaseCacheHandle`（抽象冻结数据类）

```python
@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int
```

缓存句柄，表示对某段已缓存前缀的引用。`cached_len` 字段记录该句柄对应的前缀长度（以 token 为单位）。

**抽象方法：**

```python
@abstractmethod
def get_matched_indices(self) -> torch.Tensor: ...
```
返回该句柄对应的所有物理 slot 索引，shape `(cached_len,)`，dtype `int32`，设备与缓存相同。这些索引可直接用于填充 page_table。

#### `SizeInfo`

```python
class SizeInfo(NamedTuple):
    evictable_size: int   # 当前可驱逐的 token 数（ref_count == 0）
    protected_size: int   # 当前被保护的 token 数（ref_count > 0）

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size
```

描述前缀缓存的容量使用情况，调度器用其决定何时触发驱逐。

#### `InsertResult`

```python
class InsertResult(NamedTuple):
    cached_len: int          # 插入前已在缓存中的前缀长度（这部分旧索引需要被释放）
    handle: BaseCacheHandle  # 插入后指向完整前缀的缓存句柄
```

#### `MatchResult`

```python
class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle  # 匹配到的前缀对应的缓存句柄
```

#### `BasePrefixCache`（抽象类）

```python
class BasePrefixCache(ABC):
```

KV 缓存逻辑层的接口。管理前缀到物理 slot 的映射关系，提供前缀匹配、插入、驱逐等操作。

**抽象方法：**

```python
@abstractmethod
def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
```
锁定或解锁一个缓存句柄。锁定（`unlock=False`）时引用计数加 1，节点变为不可驱逐；解锁（`unlock=True`）时引用计数减 1，归零后变为可驱逐。

**使用规范：** `match_prefix` 返回的句柄在被使用前必须先 `lock_handle`，否则在 `evict` 调用期间可能被驱逐而导致数据损坏。

```python
@abstractmethod
def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
```
**参数：** `input_ids`，shape `(seq_len,)` — 待匹配的输入 token 序列

**返回值：** `MatchResult`，包含匹配到的最长前缀对应的句柄

**语义：** 只读操作，不修改缓存状态。返回的 `handle.cached_len` 为已命中的前缀长度；`handle.get_matched_indices()` 返回这些 token 对应的物理 slot 编号。

```python
@abstractmethod
def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
```
**参数：**
- `input_ids`：shape `(seq_len,)` — 要插入的完整前缀 token 序列
- `indices`：shape `(seq_len,)` — 对应的物理 slot 编号（由 page 分配器分配）

**返回值：** `InsertResult`，`cached_len` 字段为插入前已命中的长度（对应的旧 slot 可以被释放）

**语义：** 读写操作，将新前缀写入缓存结构。

```python
@abstractmethod
def evict(self, size: int) -> torch.Tensor:
```
**参数：** `size` — 需要驱逐的 token 数（以 token 为单位）

**返回值：** 被驱逐的物理 slot 索引，shape `(evicted_size,)`

**语义：** 实际驱逐量可能大于请求量（以节点为粒度）。`evict(0)` 始终安全且无副作用。

```python
@abstractmethod
def reset(self) -> None:
```
重置整个缓存状态。

```python
@property
@abstractmethod
def size_info(self) -> SizeInfo:
```
获取当前缓存的可驱逐和受保护 token 数量。

```python
@abstractmethod
def check_integrity(self) -> None:
```
完整性校验，检测缓存数据结构的一致性，发现异常时抛出异常。

---

## E.3 `kvcache/mha_pool.py` — MHA KV 缓存池

### 文件职责

实现多头注意力（MHA）场景下 KV 缓存的物理存储，通过单块预分配的六维 tensor 管理所有层、所有页的 key/value 数据。

### 公开类

#### `MHAKVCache`

```python
class MHAKVCache(BaseKVCachePool):
    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
```

**参数：**

| 参数 | 说明 |
|------|------|
| `num_kv_heads` | 模型的 KV 头数（全局，TP 切分在内部完成） |
| `num_layers` | transformer 层数 |
| `head_dim` | 每个注意力头的维度 |
| `num_pages` | 总页数（含 1 个 dummy page） |
| `page_size` | 每页的 token 容量 |
| `dtype` | 存储精度 |
| `device` | 目标 GPU |

### 六维 Tensor 物理布局

`MHAKVCache` 的核心是一块统一预分配的六维 tensor `_kv_buffer`：

```python
self._kv_buffer = torch.empty(
    (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
    device=device,
    dtype=dtype,
)
```

**各维度含义：**

| 维度 | 大小 | 含义 |
|------|------|------|
| 第 0 维 | 2 | K/V 区分（0 = key，1 = value） |
| 第 1 维 | `num_layers` | transformer 层索引 |
| 第 2 维 | `num_pages` | 页编号（物理页，含 dummy page） |
| 第 3 维 | `page_size` | 页内 token 偏移 |
| 第 4 维 | `local_kv_heads` | 本 TP rank 负责的 KV 头数 |
| 第 5 维 | `head_dim` | 注意力头维度 |

**TP 切分：** `local_kv_heads = div_even(num_kv_heads, tp_info.size)`，每个 rank 持有等分的 KV 头，`div_even` 会断言整除。

**K/V 分离视图：**

```python
self._k_buffer = self._kv_buffer[0]   # shape: (num_layers, num_pages, page_size, local_kv_heads, head_dim)
self._v_buffer = self._kv_buffer[1]   # shape: (num_layers, num_pages, page_size, local_kv_heads, head_dim)
```

两者是 `_kv_buffer` 的切片视图，零拷贝，共享底层存储。

**平坦存储形状：**

```python
self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
```

用于在 `store_kv` 中将多维缓存展平为 `[token_slot, head, dim]` 的三维视图，配合 `store_cache` kernel 的索引方式。

**内存布局示意：**

```
_kv_buffer[k_or_v, layer, page, token_in_page, head, dim]
           ──────  ─────  ────  ─────────────  ────  ───
              2      L     P         S           H     D
```

整块 tensor 在 GPU 内存中连续存储，layout 为 C-contiguous（行主序），最内层维度（`head_dim`）在内存中连续，有利于 attention kernel 的 coalesced memory access。

### 方法实现

```python
def k_cache(self, index: int) -> torch.Tensor:
    return self._k_buffer[index]

def v_cache(self, index: int) -> torch.Tensor:
    return self._v_buffer[index]
```

按层索引返回对应层的 key/value 缓存，shape `(num_pages, page_size, local_kv_heads, head_dim)`。

```python
def store_kv(
    self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
) -> None:
    from minisgl.kernel import store_cache

    store_cache(
        k_cache=self._k_buffer[layer_id].view(self._storage_shape),
        v_cache=self._v_buffer[layer_id].view(self._storage_shape),
        indices=out_loc,
        k=k,
        v=v,
    )
```

**关键细节：** 通过 `.view(self._storage_shape)` 将 `(num_pages, page_size, local_kv_heads, head_dim)` 重塑为 `(num_pages * page_size, local_kv_heads, head_dim)`，即把 page 和页内偏移合并为单一的 token slot 维度。随后 `store_cache` kernel 以 `out_loc[i]` 为目标 slot 编号，将 `k[i]`、`v[i]` 写入对应位置。

`store_cache` 是一个自定义 CUDA kernel（位于 `minisgl.kernel`），相比简单的索引赋值有更好的 GPU 利用率。

---

## E.4 `kvcache/radix_cache.py` — 基数树前缀缓存

### 文件职责

基于基数树（Radix Tree，也称压缩前缀树）实现带 LRU 驱逐策略的前缀 KV 缓存，允许不同请求复用相同前缀的 KV 计算结果，从而减少重复计算。

### 公开类

#### `RadixTreeNode`

```python
class RadixTreeNode:
    counter: int = 0  # 全局单调递增计数器，用于生成唯一 uuid

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
```

基数树的节点类，每个节点代表一段前缀区间（key）及其对应的物理 slot 索引（value）。

**关键字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `key_fn` | `Callable` | 将 token 序列转换为字典键的函数（见 `_get_key_fn`） |
| `children` | `Dict[Any, RadixTreeNode]` | 子节点字典，键为首个 page 的 token tuple |
| `_parent` | `RadixTreeNode \| None` | 父节点引用 |
| `ref_count` | `int` | 引用计数，> 0 表示被锁定不可驱逐 |
| `uuid` | `int` | 全局唯一节点 ID |
| `timestamp` | `int` | 最近访问时间戳（`time.monotonic_ns()`），LRU 排序依据 |
| `_key` | `torch.Tensor` | 该节点代表的 token id 段，shape `(length,)` |
| `_value` | `torch.Tensor` | 对应的物理 slot 索引，shape `(length,)` |
| `_length` | `int` | 该节点覆盖的 token 数 |

**关键方法：**

```python
def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
```
设置节点的 key/value，断言二者长度相等。

```python
def set_parent(self, parent: RadixTreeNode) -> None:
```
将当前节点注册为 `parent` 的子节点，`key_fn(self._key)` 作为字典键。

```python
def get_match_len(self, input_ids: torch.Tensor) -> int:
```
调用 `fast_compare_key`（自定义 CUDA kernel）比较 `self._key` 与 `input_ids` 的最长公共前缀长度。

```python
def split_at(self, pos: int) -> RadixTreeNode:
```
在位置 `pos` 处将当前节点一分为二，返回新建的前半段节点。具体操作：
1. 新建节点 `new_node`，`key/value = self._key[:pos] / self._value[:pos]`，继承 `self` 的 `ref_count` 和 `timestamp`
2. 将 `new_node` 注册为 `self.parent` 的子节点（替换 `self`）
3. `self` 的 `key/value` 更新为后半段 `[pos:]`，将 `self` 注册为 `new_node` 的子节点

```python
def __lt__(self, other: RadixTreeNode) -> bool:
    return self.timestamp < other.timestamp
```
定义小于运算，使节点可直接放入 `heapq`，按 `timestamp` 升序（越旧优先级越高，越先被驱逐）。

#### `RadixCacheHandle`

```python
@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode
```

基数树缓存句柄。`node` 字段指向树中的某个节点，代表匹配或插入的前缀终点。

```python
def get_matched_indices(self) -> torch.Tensor:
    node = self.node
    value_list: List[torch.Tensor] = []
    while not node.is_root():
        value_list.append(node.value)
        node = node.parent
    value_list.reverse()
    return torch.cat(value_list)
```

从当前节点向上遍历到根节点，收集路径上所有节点的 `value`（物理 slot 索引），反转后拼接，得到完整前缀的物理地址序列。时间复杂度 O(depth)，实践中树深度有限。

#### `RadixPrefixCache`

```python
class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
```

**关键字段：**

| 字段 | 说明 |
|------|------|
| `device` | 目标设备 |
| `page_size` | 从全局上下文读取，决定对齐单位 |
| `key_fn` | 首 page 的 token tuple 作为子节点字典键 |
| `empty_tensor` | 空 int32 tensor，用于返回零长度结果 |
| `evictable_size` | 当前可驱逐的 token 总数（`ref_count == 0` 的节点之和） |
| `protected_size` | 当前受保护的 token 总数（`ref_count > 0` 的节点之和） |
| `root_node` | 根节点，`ref_count = 1`（永远受保护，不参与驱逐） |

### 基数树核心算法

#### `_tree_walk` — 前缀匹配遍历

```python
def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
```

**功能：** 从根节点出发沿树向下遍历，找到与 `input_ids` 最长匹配的节点和已匹配长度。

**算法步骤（行 205-230）：**

```python
while prefix_len < indice_len:
    # 1. 以当前未匹配段的第一个 page 的 token tuple 查找子节点
    child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
    if child_node is None:
        return node, prefix_len          # 无匹配，返回当前位置

    node = child_node

    # 2. 精确比较当前子节点的 key 与 input_ids[prefix_len:]
    match_len = node.get_match_len(input_ids[prefix_len:])
    match_len = align_down(match_len, self.page_size)   # 对齐到 page 边界
    prefix_len += match_len

    # 3. 如果只是部分匹配（key 比 input_ids 短了一截），需要分裂节点
    if match_len != node.length:
        node = node.split_at(match_len)  # 分裂，返回前半段
        return node, prefix_len

    # 4. 完全匹配当前节点，更新时间戳（LRU 访问记录）
    node.timestamp = tic
```

**关键设计：**
- 子节点字典的键为首个 page 的 token tuple（由 `key_fn` 生成），因此字典查找是 O(1)
- 使用 `fast_compare_key` CUDA kernel 做逐 token 比较，比 Python 循环快得多
- 匹配长度向下对齐到 `page_size`，确保所有操作以 page 为最小粒度
- 部分匹配时调用 `split_at` 原地分裂节点，使基数树保持结构正确性

#### `match_prefix` — 前缀查找

```python
def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
    node, prefix_len = self._tree_walk(input_ids)
    return MatchResult(RadixCacheHandle(prefix_len, node))
```

纯查找，不修改树结构（但会更新时间戳）。

#### `insert_prefix` — 前缀插入

```python
def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
    insert_len = align_down(len(input_ids), self.page_size)
    input_ids, indices = input_ids[:insert_len], indices[:insert_len]
    node, prefix_len = self._tree_walk(input_ids)
    if prefix_len != insert_len:
        new_node = RadixTreeNode(self.key_fn)
        new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
        new_node.set_parent(node)
        self.evictable_size += new_node.length
        node = new_node
    return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))
```

**步骤：**
1. 将插入长度向下对齐到 page 边界
2. 调用 `_tree_walk` 找到最长匹配节点和已匹配长度
3. 若未全部匹配（`prefix_len < insert_len`）：
   - 创建新节点，存储未匹配部分的 token id（`key`）和 slot 索引（`value`）
   - `indices[prefix_len:].clone()` 复制索引数据，避免外部修改影响缓存
   - 将新节点接入树，更新 `evictable_size`
4. 返回 `InsertResult`，其中 `cached_len = prefix_len` 表示已命中部分（这些 slot 已有缓存，调用方可将其释放）

#### `evict` — LRU 驱逐算法

```python
def evict(self, size: int) -> torch.Tensor:
```

**算法实现（行 148-175）：**

```python
# 收集所有叶子节点中 ref_count == 0 的节点
leave_nodes = self._collect_leave_nodes_for_evict()
# 建立最小堆（按 timestamp 升序，最旧的在堆顶）
heapq.heapify(leave_nodes)

while evicted_size < size:
    node = heapq.heappop(leave_nodes)           # 弹出最旧的叶子
    evicted_size += node.length
    evicted_indices.append(node.value)           # 收集被驱逐的 slot 索引
    self.evictable_size -= node.length
    parent = node.parent
    del parent.children[self.key_fn(node._key)] # 从父节点子字典中删除
    # 父节点若也变成了叶子且 ref_count == 0，加入候选堆
    if parent.is_leaf() and parent.ref_count == 0:
        heapq.heappush(leave_nodes, parent)

return torch.cat(evicted_indices)
```

**LRU 堆排序实现：**
- `RadixTreeNode.__lt__` 定义了按 `timestamp` 的比较，`heapq` 是最小堆，堆顶始终是时间戳最小（最久未访问）的节点
- 每次驱逐最旧叶子后，检查其父节点是否成为新的可驱逐叶子，并推入堆中，实现"自底向上"的链式驱逐
- 根节点因 `ref_count = 1` 永远不会进入驱逐候选

**`_collect_leave_nodes_for_evict` 收集候选叶子：**

```python
def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
    nodes: List[RadixTreeNode] = [self.root_node]
    leave_nodes: List[RadixTreeNode] = []
    while len(nodes) > 0:
        node = nodes.pop()
        if node.is_leaf():
            if node.ref_count == 0:
                leave_nodes.append(node)
        else:
            for child in node.children.values():
                nodes.append(child)
    return leave_nodes
```

深度优先遍历整棵树，收集所有 `ref_count == 0` 的叶子节点作为初始驱逐候选。时间复杂度 O(N)，N 为树中节点总数。

### `lock_handle` — 引用计数管理

```python
def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
```

**锁定路径（`unlock=False`，行 124-130）：**
```python
while not node.is_root():
    if node.ref_count == 0:              # 从 0 变 1：从可驱逐变为受保护
        self.evictable_size -= node.length
        self.protected_size += node.length
    node.ref_count += 1
    node = node.parent
```

**解锁路径（`unlock=True`，行 116-123）：**
```python
while not node.is_root():
    node.ref_count -= 1
    assert node.ref_count >= 0
    if node.ref_count == 0:              # 从 1 变 0：从受保护变为可驱逐
        self.evictable_size += node.length
        self.protected_size -= node.length
    node = node.parent
```

两个路径都从句柄对应的节点一直向上遍历到根节点，对路径上所有节点（即整条前缀链）修改引用计数。这保证了即使中间节点也不会在被引用时被驱逐。

### `_get_key_fn` — 子节点字典键函数

```python
def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()         # 单 token：整数键
    return lambda x: tuple(x[:page_size].tolist())  # 多 token：tuple 键
```

字典键的设计需要兼顾两个需求：
1. 可哈希（dict 键的要求）
2. 能区分不同的子树分支

`page_size == 1` 时直接用首个 token id 的整数值作键，比 tuple 更高效。`page_size > 1` 时用首 page 的 token tuple，确保在多 token 一页的场景下键的唯一性。

---

## E.5 `kvcache/naive_cache.py` — 朴素前缀缓存

### 文件职责

实现一个"空操作"版本的前缀缓存，不进行任何前缀复用，每次请求都重新计算所有 KV，用于功能验证、基准测试和不需要前缀缓存的场景。

### 公开类

#### `NaiveCacheHandle`

```python
class NaiveCacheHandle(BaseCacheHandle):
    empty_tensor: torch.Tensor  # 类变量，由 NaivePrefixCache 初始化时注入

    def __init__(self):
        super().__init__(cached_len=0)

    def get_matched_indices(self) -> torch.Tensor:
        return self.empty_tensor
```

**关键设计：**
- `cached_len` 固定为 0：永远报告没有命中任何前缀
- `get_matched_indices()` 返回空 tensor：没有可复用的物理 slot
- `empty_tensor` 是类变量（非实例变量），由 `NaivePrefixCache.__init__` 通过 `NaiveCacheHandle.empty_tensor = ...` 注入，所有句柄实例共享同一个空 tensor，避免重复分配

#### `NaivePrefixCache`

```python
class NaivePrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        NaiveCacheHandle.empty_tensor = self.empty_tensor
        super().__init__()
```

**各方法实现：**

```python
def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
    pass  # 无操作：没有任何缓存节点需要保护
```

```python
def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
    return MatchResult(NaiveCacheHandle())  # 始终返回 cached_len=0 的句柄
```

```python
def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
    return InsertResult(0, NaiveCacheHandle())  # 不存储任何数据
```

```python
def evict(self, size: int) -> torch.Tensor:
    if size == 0:
        return self.empty_tensor
    raise NotImplementedError("NaiveCacheManager does not support eviction.")
    # 因为从不缓存，evictable_size 始终为 0，调用 evict(>0) 属于逻辑错误
```

```python
def reset(self) -> None:
    pass  # 无状态，无需重置
```

```python
@property
def size_info(self) -> SizeInfo:
    return SizeInfo(evictable_size=0, protected_size=0)
    # 永远报告缓存为空，不会触发驱逐
```

```python
def check_integrity(self) -> None:
    pass  # 无状态，无需校验
```

**设计意图：** `NaivePrefixCache` 是 Null Object 模式的典型应用。调度器无需对缓存类型做特殊判断，只需统一调用 `BasePrefixCache` 接口；当使用朴素缓存时，所有前缀相关操作变为无操作，调度逻辑退化为每次请求完整重算 KV。

---

## E.6 模块整体架构

### 两层结构

KV 缓存系统分为独立的两层，各层可独立替换：

```
┌────────────────────────────────────────────┐
│          逻辑层（前缀缓存）                  │
│   BasePrefixCache                          │
│   ├── NaivePrefixCache  (无复用)            │
│   └── RadixPrefixCache  (基数树 + LRU)      │
│       管理: token id → 物理 slot 的映射      │
└────────────────────────────────────────────┘
              │ 物理 slot 索引（int32 tensor）
              ▼
┌────────────────────────────────────────────┐
│          物理层（KV 缓存池）                  │
│   BaseKVCachePool                          │
│   └── MHAKVCache  (六维 tensor)             │
│       管理: 实际 key/value 数据的分配与读写   │
└────────────────────────────────────────────┘
```

两层之间通过**物理 slot 索引**（`torch.int32` 张量）交互，上层（调度器/前缀缓存）负责分配和管理 slot 编号，下层（KV 缓存池）负责在对应 slot 中存取实际数据。

### 数据流

一次完整的请求处理流程中，KV 缓存模块的数据流如下：

```
新请求到来
    │
    ▼
match_prefix(input_ids)
    │ 返回 handle（cached_len, node）
    ▼
lock_handle(handle)          # 保护命中的缓存，防止被驱逐
    │
    ▼
handle.get_matched_indices() # 获取命中前缀的物理 slot
    │ 填入 page_table
    ▼
prefill 阶段
    │ model.forward() 计算新 token 的 KV
    ▼
kv_cache.store_kv(k, v, out_loc, layer_id)  # 写入物理 slot
    │
    ▼
insert_prefix(input_ids, new_indices)
    │ 返回 InsertResult(cached_len, new_handle)
    ▼
lock_handle(new_handle)      # 保护新插入的缓存
    │
    ▼
decode 阶段（多轮）
    │
    ▼
请求完成
    │
    ▼
lock_handle(handle, unlock=True)  # 释放引用，节点变为可驱逐

（页分配器在需要时调用 evict()）
```

### 引用计数不变式

`RadixPrefixCache` 维护以下不变式，确保缓存一致性：

1. **根节点 `ref_count == 1`**：根节点永远受保护
2. **活跃请求持有的节点 `ref_count >= 1`**：被锁定的节点不可被驱逐
3. **`evictable_size + protected_size == 树中所有节点长度之和`**：两个计数器之和等于树的总大小
4. **驱逐只针对叶子节点**：非叶子节点（有子节点的节点）即使 `ref_count == 0` 也不会被直接驱逐，只有其所有子节点都被驱逐后，它才可能成为叶子并进入驱逐候选

### 页大小对齐约束

所有前缀操作以 `page_size` 为最小粒度（`align_down`）：
- 插入长度对齐：`insert_len = align_down(len(input_ids), page_size)`
- 匹配长度对齐：`match_len = align_down(fast_compare_key(...), page_size)`
- 这确保每个基数树节点存储的 token 数始终是 `page_size` 的倍数，与 KV 缓存的分页机制完全对齐
