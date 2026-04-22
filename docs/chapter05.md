# 第 5 章：调度器核心——如何高效地同时服务多个请求

## 1. 背景问题

在单请求推理场景下，生活很美好：把输入喂进模型，等它一步步生成输出，完成后返回给用户。但一旦需要**同时服务多个用户**，三个矛盾立刻浮出水面：

**矛盾一：显存是有限的。** 每个请求都需要在 GPU 显存中存放 KV Cache。一个 7B 参数模型，在 FP16 精度下，32 层、128 个头的情况下，单个 token 的 KV Cache 约 0.5 MB。一个 4096 token 的长对话，就要占用 2 GB 显存。10 个并发请求就是 20 GB——直接 OOM。

**矛盾二：Prefill 和 Decode 的计算特性截然不同。** Prefill 是一次性处理整个输入序列（比如 1000 个 token），是 compute-bound 操作；Decode 是每步只生成 1 个 token，是 memory-bandwidth-bound 操作。把它们粗暴地混在一批里执行，两种操作会相互干扰，GPU 利用率低下。

**矛盾三：请求的生命周期参差不齐。** 有的请求 Prefill 很短，有的很长；有的生成 10 个 token 就结束，有的要生成 2000 个。调度器必须在任意时刻决定：下一步该处理哪些请求、处理多少。

本章的主角 `Scheduler` 以及它的三位助手——`PrefillManager`、`DecodeManager`、`CacheManager`——就是专门解决上述三个矛盾的。

---

## 2. 核心概念讲解

### 2.1 Prefill 与 Decode 的本质区别

可以用一个厨房类比来理解：

- **Prefill** 就像"备菜"：把整道菜的所有食材一次性切好、腌好，送进炉子。输入序列有多长，就要处理多少 token。Attention 矩阵是 `[seq_len, seq_len]` 的，计算量与 `seq_len²` 成正比，是标准的矩阵乘法密集型任务。
- **Decode** 就像"出菜"：每次只出一道菜（生成一个 token），新 token 只需要与历史 KV Cache 做 attention，计算量很小，瓶颈在于从显存读取 KV Cache 的带宽。

这两种操作不能混在一个 batch 里的原因：若把一个 1000-token 的 Prefill 请求和 10 个 Decode 请求塞进同一批，Prefill 的大矩阵计算会大幅延迟 Decode 请求的响应时间（即 TTFT——Time To First Token 和 ITL——Inter-Token Latency 都会恶化）。

### 2.2 Paged KV Cache 与显存管理

类比操作系统的虚拟内存分页：与其给每个请求分配一整块连续显存，不如把显存切成固定大小的**页（Page）**，按需分配。`CacheManager` 维护一个空闲页池（`free_slots`），请求需要多少页就分配多少页，完成后归还。

### 2.3 Chunked Prefill：长输入的解法

对于超长输入（如 32K token 的上下文），一次性 Prefill 会瞬间耗尽显存和 token 预算。`PrefillManager` 引入 **Chunked Prefill**：将长输入切成多个小块，每轮调度只处理一个 chunk，多轮完成后再进入 Decode 阶段。被分块处理的请求用 `ChunkedReq` 类型标记，暂不加入 Decode 队列。

### 2.4 Prefill-First 调度策略

`Scheduler._schedule_next_batch()` 的策略非常简洁（见 `scheduler.py` 第 221–224 行）：

```python
batch = (
    self.prefill_manager.schedule_next_batch(self.prefill_budget)
    or self.decode_manager.schedule_next_batch()
)
```

**Prefill 优先**：每轮调度先尝试安排 Prefill，只有没有 Prefill 任务时才切换到 Decode。这保证了新请求能尽快完成 KV Cache 建立，进入生成阶段。`prefill_budget`（`max_extend_tokens`，默认 8192）则限制了每轮 Prefill 消耗的 token 总量，防止单轮 Prefill 阻塞 Decode 太久。

---

## 3. 核心代码导读

### 3.1 请求的数据结构（`core.py`）

理解调度器，先理解请求的状态机。每个请求用 `Req` 对象表示（`core.py` 第 29–68 行），关键字段：

| 字段 | 含义 |
|------|------|
| `cached_len` | 已经写入 KV Cache 的 token 数（Attention 可以直接读取） |
| `device_len` | 当前在设备上的 token 数（包括本轮待处理的） |
| `max_device_len` | 最大序列长度（`input_len + max_output_len`） |

三个核心属性从这些字段派生：

- `extend_len = device_len - cached_len`：本轮需要计算 Attention 的 token 数
- `remain_len = max_device_len - device_len`：还能继续生成多少 token
- `can_decode`：`remain_len > 0`，即是否还需要继续生成

每次 Decode 完成一步，调用 `complete_one()` 方法（第 52–54 行），`cached_len` 追上 `device_len`，`device_len` 再加 1（新 token 进队）。

### 3.2 主调度循环（`scheduler.py`）

调度器的主循环有两个模式：`normal_loop` 和 `overlap_loop`（第 108–131 行）。在默认的 `overlap_loop` 模式下，CPU 端处理**上一批次**结果的同时，GPU 执行**当前批次**的前向计算，实现 CPU/GPU overlap。

每轮循环的三步：

1. **接收新消息**（`receive_msg`）：从前端接收新的用户请求或 abort 指令
2. **调度决策**（`_schedule_next_batch`）：决定下一批次执行什么
3. **处理上一批次的输出**（`_process_last_data`）：采样结果写回、判断是否结束、归还资源

`_schedule_next_batch` 完整路径（第 219–225 行）：

```python
def _schedule_next_batch(self) -> ForwardInput | None:
    batch = (
        self.prefill_manager.schedule_next_batch(self.prefill_budget)
        or self.decode_manager.schedule_next_batch()
    )
    return self._prepare_batch(batch) if batch else None
```

若两个 manager 都返回 `None`，则本轮空转（调度器 idle）。

### 3.3 PrefillManager：Chunked Prefill 的分块策略（`prefill.py`）

`PrefillManager.schedule_next_batch` 的核心逻辑（第 126–151 行）：

1. 创建 `PrefillAdder`，初始化两个约束：
   - `token_budget`：本轮最多处理多少个新 token（来自 `prefill_budget`）
   - `reserved_size`：当前 Decode 队列已占用的显存估算（`decode_manager.inflight_tokens`），预留出来不能分配给新 Prefill

2. 遍历 `pending_list`，对每个请求调用 `adder.try_add_one`：
   - 若 `token_budget` 耗尽，停止添加
   - 若可用显存不足（`estimated_len + reserved_size > available_size`），停止添加

3. `_add_one_req` 中（第 65–90 行），核心分块逻辑：

```python
chunk_size = min(self.token_budget, remain_len)
is_chunked = chunk_size < remain_len
CLS = ChunkedReq if is_chunked else Req
self.token_budget -= chunk_size
```

若 `chunk_size < remain_len`，说明一次处理不完，创建 `ChunkedReq`（而非普通 `Req`）。`ChunkedReq` 重写了两个方法（第 24–29 行）：
- `append_host` 抛出异常——分块请求不该被采样出 token
- `can_decode` 返回 `False`——分块请求不进入 Decode 队列

处理完 chunk 后，`pending_req.chunked_req` 保存当前进度，下一轮调度时直接续上（第 96–102 行），无需重新分配资源。

分块后 `pending_list` 的更新（第 150 行）：

```python
self.pending_list = chunked_list + self.pending_list[len(reqs):]
```

未处理完的 chunked 请求插回队首，保证下轮优先继续处理。

### 3.4 DecodeManager：维护运行中的 Decode 批次（`decode.py`）

`DecodeManager` 的设计非常简洁：用一个 `Set[Req]` 维护当前所有运行中的请求（第 11–39 行）。

关键方法：

- **`filter_reqs(reqs)`**（第 14–15 行）：每轮前向后调用，将本轮参与计算的请求合并到 `running_reqs`，同时过滤掉 `can_decode == False` 的已完成请求：

```python
def filter_reqs(self, reqs: Iterable[Req]) -> None:
    self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}
```

注意：Prefill 批次完成后，其中的普通 `Req`（非 `ChunkedReq`）会通过这里进入 `running_reqs`，开始 Decode。

- **`inflight_tokens`**（第 28–30 行）：估算所有 Decode 请求未来还需要的显存页数，供 `PrefillAdder` 做保守预留：

```python
tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
return sum(req.remain_len for req in self.running_reqs) + tokens_reserved
```

每个请求额外保留 `page_size - 1` 个 token 的空间，是因为页对齐导致的内碎片——最坏情况下每个请求的最后一页会浪费 `page_size - 1` 个 slot。

### 3.5 CacheManager：KV 页面的生命周期（`cache.py`）

`CacheManager` 负责显存的分页分配与前缀缓存（第 15–125 行）。

**分配路径**（`allocate_paged`，第 42–53 行）：

对每个请求，计算其当前需要哪些页（从 `cached_len` 到 `device_len`），调用 `_allocate` 从 `free_slots` 中取出，然后写入 `page_table`（二维张量，shape `[max_req, max_seq_len]`）。若空闲页不足，触发 LRU 驱逐（第 107–109 行）：

```python
if needed_pages > (free_pages := len(self.free_slots)):
    evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
    self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
```

**释放路径**（`cache_req`，第 55–79 行）：

当请求完成时（`finished=True`），将已计算的 KV Cache 插入前缀缓存树，尾部多余的页（无法插入缓存的 trailing token）归还到 `free_slots`。这是 prefix caching 的关键：同一前缀的后续请求可以直接复用已有的 KV Cache，跳过 Prefill 计算。

**延迟释放**（`lazy_free_region`，第 93–104 行）：

在 `_process_last_data` 中使用 context manager 包裹批量释放操作，避免在处理每个请求时频繁修改 `free_slots` tensor，降低 CPU 开销。

---

## 4. 设计决策

### 4.1 为什么 Prefill 优先而不是 Decode 优先？

**Prefill 优先**保证吞吐量：每个新请求只有完成 Prefill 才能开始 Decode，积压的 Prefill 越多意味着越多请求在等待，系统总体进度停滞。

**Decode 优先**则可以降低已在生成的请求的 ITL（Inter-Token Latency），适合对延迟极为敏感的场景。代码注释（`scheduler.py` 第 220 行）也预留了这个扩展点：`# TODO: support other policies: e.g. DECODE first`。

### 4.2 Chunked Prefill vs. 直接拒绝长请求

另一种方案是直接拒绝超出 `max_extend_tokens` 的请求，实现更简单。但 Chunked Prefill 的优势在于：
- 可以处理任意长度的输入（只要最终序列长度不超过 `max_seq_len`）
- 长输入可以与 Decode 任务交错执行，不会长时间独占 GPU

代价是增加了状态管理复杂度：`ChunkedReq` 需要跨轮次保持状态，`pending_list` 需要正确重排。

### 4.3 `inflight_tokens` 的保守预留

`PrefillAdder` 初始化时将 Decode 队列的全部剩余 token 估算值作为保留量。这是一个**悲观估计**：实际上不需要一次性分配完所有剩余页。但保守估计换来的是安全性——不会因为过度分配 Prefill 导致 Decode 请求在下轮无页可用。

### 4.4 两流（Two-Stream）Overlap 调度

`scheduler.py` 使用了两个 CUDA stream（`self.stream` 和 `self.engine.stream`）：CPU 端的元数据准备（构建 page_table 索引、拷贝 input_ids）在 `self.stream` 上执行，GPU 计算在 `engine.stream` 上执行，通过 `stream.wait_stream` 同步。这样上一批次 GPU 计算期间，CPU 可以同步处理上上批次的输出结果，实现 CPU/GPU pipeline。

---

## 5. 动手实验

### 实验一：观察 Chunked Prefill 的分块行为

在 `PrefillManager.schedule_next_batch` 中加入日志，观察长输入被分成几块处理。

在 `prefill.py` 第 148 行（`if len(reqs) == 0: return None` 之后）插入：

```python
for req in reqs:
    chunk_type = "CHUNKED" if isinstance(req, ChunkedReq) else "FULL"
    print(f"[Prefill] uid={req.uid} type={chunk_type} "
          f"cached={req.cached_len} device={req.device_len}")
```

然后发送一个 token 数超过 `max_extend_tokens`（默认 8192）的请求，观察输出。预期会看到多轮 `CHUNKED` 日志，最后一轮出现 `FULL`。

### 实验二：测量 Prefill Budget 对吞吐量的影响

修改 `SchedulerConfig.max_extend_tokens` 的值，分别设置为 512、2048、8192，用相同的 benchmark 请求集合测试吞吐量（tokens/s）和首 token 时延（TTFT）。

```python
# 在启动服务器时修改
config = SchedulerConfig(
    model_path="...",
    max_extend_tokens=512,  # 尝试不同值
)
```

**预期结论**：`max_extend_tokens` 越小，单次 Prefill 阻塞 Decode 的时间越短（ITL 更稳定），但 Prefill 需要更多轮次，Prefill 吞吐量下降，TTFT 增大。

### 实验三：验证 CacheManager 的完整性检查

`CacheManager.check_integrity` 方法（`cache.py` 第 81–91 行）验证 `free_pages + cache_pages == num_pages`。在调度器空闲时（`run_when_idle`）会自动调用。

可以在测试中手动触发：

```python
# 在 Scheduler 空闲时调用
scheduler.cache_manager.check_integrity()
```

若不抛出异常，说明显存没有泄漏。可以故意在 `_free_req_resources` 中注释掉 `cache_manager.cache_req` 调用，再跑请求，观察 `check_integrity` 报错，理解每一步资源释放的必要性。

### 实验四（进阶）：实现 Decode 优先调度

参考 `_schedule_next_batch` 的注释，实现一个 Decode 优先策略：

```python
def _schedule_next_batch(self) -> ForwardInput | None:
    batch = (
        self.decode_manager.schedule_next_batch()
        or self.prefill_manager.schedule_next_batch(self.prefill_budget)
    )
    return self._prepare_batch(batch) if batch else None
```

对比两种策略在高并发负载下的 P99 ITL（第 99 百分位的 inter-token latency）。

---

## 6. 小结

本章系统梳理了 mini-sglang 调度器的核心设计：

| 组件 | 职责 | 核心机制 |
|------|------|----------|
| `Scheduler` | 主循环、消息路由、结果处理 | Two-stream overlap，Prefill 优先 |
| `PrefillManager` | 管理待 Prefill 的请求队列 | Chunked Prefill，token/显存双约束 |
| `DecodeManager` | 维护正在 Decode 的请求集合 | Set 维护，inflight_tokens 保守估计 |
| `CacheManager` | KV Cache 页面分配与回收 | 分页管理，前缀缓存，LRU 驱逐 |

**核心设计哲学**：在 GPU 显存这个"硬约束"下，用贪心策略（Prefill 优先 + token budget）最大化 GPU 利用率，同时通过 Chunked Prefill 保证长请求不被饿死。

**与后续章节的连接**：调度器生成的 `Batch` 对象，接下来会交给**注意力后端**（第 6 章）处理——`batch.attn_metadata` 字段正是调度器为注意力计算准备的元数据，包含每个请求的页表映射信息。`CacheManager` 维护的前缀缓存（Radix Tree），将在**第 7 章**中详细展开。
