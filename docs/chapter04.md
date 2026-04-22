# 第 4 章：Tokenizer 与增量解码——流式输出的文字是怎么来的

## 4.1 背景问题：一个"简单"的需求背后的工程挑战

当你在 ChatGPT 或类似界面里看到文字一个个蹦出来时，直觉上这件事似乎很简单：模型每一步生成一个 token，把 token 解码成文字，再推送给客户端就行了。

然而实际落地时会遇到一个棘手的矛盾：**BPE tokenizer 的编码单位（token）和人类可读文字的边界（字符/词）并不对齐**。

具体来说有三个层面的问题：

1. **字节级别的 BPE（Byte-level BPE）把 Unicode 字符拆成字节序列**。一个中文字符在 UTF-8 下占 3 个字节，BPE 词表可能把这 3 个字节映射到 1 到 3 个不同的 token。如果只解码当前 token，很可能得到一个不完整的 UTF-8 序列，进而产生乱码（Python 的 `�` 替换字符，即 U+FFFD）。

2. **英文单词边界模糊**。`tokenizer.decode([token_id])` 对单个 token 的解码结果往往带有空格前缀（如 `" hello"`），但在还未收到下一个 token 时，无法判断当前 token 是某个词的一部分还是完整的词。

3. **流式场景下每一步只有一个新 token**，不能等全部生成完再解码（那就不叫流式了）。

这三个矛盾叠加在一起，要求我们设计一个**增量解码（incremental detokenization）**机制：每收到一个新 token 就能安全、正确地输出尽可能多的可读文本，同时保留还不能确认的"模糊边界"等待下一个 token 到来后再决策。

mini-sglang 对应的解决方案就是 `DetokenizeManager`，本章将深入剖析其三指针设计。

---

## 4.2 核心概念：从 messages 到 token ids

### 4.2.1 Chat Template 的作用

HuggingFace 的 `apply_chat_template` 本质上是一段 Jinja2 模板，把结构化的 `messages` 列表渲染成模型训练时见过的格式化字符串，再调用 `tokenizer.encode` 转成 `input_ids`。

以 Qwen3 为例，一个典型的 `messages` 输入：

```python
messages = [
    {"role": "system", "content": "你是一个助手。"},
    {"role": "user",   "content": "你好！"},
]
```

经过 `apply_chat_template` 后会变成类似：

```
<|im_start|>system
你是一个助手。<|im_end|>
<|im_start|>user
你好！<|im_end|>
<|im_start|>assistant
```

注意末尾的 `<|im_start|>assistant\n` 是 `add_generation_prompt=True` 自动添加的，它告诉模型"现在轮到你作答了"。

`tokenize.py` 中的 `TokenizeManager` 正是在这里做分发：若输入是 `list`（即 messages），走 chat template 路径；若是 `str`，直接 encode。

**`python/minisgl/tokenizer/tokenize.py` 第 17-31 行：**

```python
for msg in msgs:
    if isinstance(msg.text, list):
        prompt = self.tokenizer.apply_chat_template(
            msg.text,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,      # 关闭 Qwen3 的 thinking 模式
        )
        assert isinstance(prompt, str)
    else:
        prompt = msg.text
    input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
    results.append(input_ids.view(-1).to(torch.int32))
```

这里有两个细节值得注意：
- `tokenize=False` 先拿到字符串，再统一由 `encode` 处理，便于调试时打印中间结果。
- `to(torch.int32)` 而非 `int64`：节省显存，int32 已经足以覆盖所有主流词表大小（通常 < 200K）。

### 4.2.2 为什么 decode 单个 token 会乱码

以 Llama/Qwen 使用的 SentencePiece 或 tiktoken 为例，它们底层都是 Byte-level BPE。词表里同时包含字节级别的 token（如 `<0xE4>` `<0xB8>` `<0xAD>`，对应"中"字的 UTF-8 三字节）和已经合并好的多字节 token（如直接对应"中"字的 token）。

当模型在某个推理步骤生成了 `<0xE4>` 这个 token，单独调用 `tokenizer.decode([<0xE4>])` 会得到一个不完整的 UTF-8 序列，Python 的 `str` 无法表示它，于是 HuggingFace 把它替换为 `"<0xE4>"` 或 `"?"` 等形式，流式推给客户端就是乱码。

正确做法是**把之前所有已生成的 token 和新 token 一起 decode**，让 tokenizer 有机会把多个字节级 token 拼合成完整字符。但这样会重复 decode 历史 token，效率低下，且无法区分"本次新增了哪些文本"。

`DetokenizeManager` 的三指针设计解决了这两个问题。

---

## 4.3 核心代码导读：`DetokenizeManager` 三指针设计

完整实现见 `python/minisgl/tokenizer/detokenize.py`，共 112 行。

### 4.3.1 状态定义（第 54-60 行）

```python
@dataclass
class DecodeStatus:
    decoded_ids: List[int]   # 全部已生成的 token id 序列
    decoded_str: str         # 已确认可以输出的累积字符串
    read_offset: int         # 上一次成功 decode 时，decoded_ids 的长度
    surr_offset: int         # "surrogate 区域"起始位置，用于探测乱码
    sent_offset: int         # 已经发送给客户端的 decoded_str 长度
```

三个指针在 `decoded_ids` 数组上的位置关系为：

```
decoded_ids: [0 ... surr_offset ... read_offset ... len-1]
              ╰── 已确认安全 ──╯╰── 可能有乱码 ──╯╰── 新 token ──╯
```

`sent_offset` 是独立作用于 `decoded_str` 字符串的指针，记录已发送的字符数量。

### 4.3.2 核心 decode 逻辑（第 70-111 行）

每次调用 `detokenize(msgs)` 时，对每个请求：

**第一步：追加新 token，构造两段待 decode 的 id 序列**（第 83-86 行）

```python
if not (msg.finished and msg.next_token == self.eos_token_id):
    s.decoded_ids.append(msg.next_token)
read_ids.append(s.decoded_ids[s.surr_offset :])   # 从 surr 到末尾
surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])  # 从 surr 到 read
```

- **`read_ids`**：`surr_offset` 到末尾的所有 id，包含新 token。
- **`surr_ids`**：`surr_offset` 到 `read_offset` 的 id，即上一次解码时"模糊区域"的 id，不含新 token。

两者都以 `surr_offset` 为起点，是因为 `surr_offset` 之前的 token 已经安全解码过，不需要重复处理。

**第二步：批量 decode**（第 88-89 行）

```python
read_texts = self.tokenizer.batch_decode(read_ids)
surr_texts = self.tokenizer.batch_decode(surr_ids)
```

`batch_decode` 的开销主要在反查词表，将两次 decode 合并为一个 batch 调用，均摊了 Python 函数调用开销。

**第三步：差分得到新文本，处理乱码**（第 94-103 行）

```python
new_text = read_str[len(surr_str):]
if len(new_text) > 0 and not new_text.endswith("?"):
    # 没有乱码：确认这段文本，推进 surr_offset 和 read_offset
    output_str = s.decoded_str + new_text
    s.decoded_str = output_str
    s.surr_offset = s.read_offset
    s.read_offset = len(s.decoded_ids)
else:
    # 有乱码或为空：只输出 find_printable_text 认为安全的部分
    new_text = find_printable_text(new_text)
    output_str = s.decoded_str + new_text
```

关键逻辑是差分：`read_str[len(surr_str):]`。由于 `surr_ids` 是 `read_ids` 的前缀，`read_str` 和 `surr_str` 共享同一段开头，直接截取尾部就得到新增的文本。

当 `new_text` 以 `"?"` 结尾（即 U+FFFD，HuggingFace 对无效 UTF-8 的替换符），说明最后若干个 token 还未构成完整字符，**不推进 `surr_offset` 和 `read_offset`**，让这部分 id 留在"模糊区域"，等下一个 token 到来后一起重新 decode。

**第四步：计算增量输出**（第 105-106 行）

```python
incremental_output = output_str[s.sent_offset:]
s.sent_offset = len(output_str)
```

`sent_offset` 保证每次只向调用方返回本轮新增的文本片段，避免重复发送。

### 4.3.3 `find_printable_text`：何时可以安全输出

**`python/minisgl/tokenizer/detokenize.py` 第 35-51 行：**

```python
def find_printable_text(text: str):
    if text.endswith("\n"):
        return text                    # 换行符是强制刷新点
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text                    # 末尾是完整 CJK 字符，可全部输出
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]               # 倒数第二是 CJK，保留末尾一个字符等待
    else:
        return text[: text.rfind(" ") + 1]  # 英文：输出到最后一个空格
```

这里体现了 BPE 解码的两类边界问题的不同处理策略：

- **CJK（中日韩）字符**：每个汉字在词表里通常有独立 token（或至多两个 token 合并），一旦字符出现在 `text` 末尾说明 UTF-8 解码已完成，可以安全输出。倒数第二是 CJK 时，保留最后一个字符（可能是英文 token，其后可能还有延续）。
- **英文/空格分隔的语言**：BPE 的 token 边界通常和空格对齐，输出到最后一个空格之前是安全的，最后一个"词片段"等待后续 token 确认是否完整。

### 4.3.4 三指针的状态机图解

以生成"你好 world"为例，假设 tokenizer 把它切成 4 个 token：`[你好][▁world]` 中，"你好"可能是 1 个 token，"world" 带空格前缀是 1 个 token。假设实际被切成更细粒度：`[<0xE4>][<0xBD>][<0xA0>][<0xE5>][<0xA5>][<0xBD>][▁world]`（纯字节级，仅作示意）：

```
步骤1：收到 <0xE4>
  decoded_ids = [e4]
  read_ids = [e4], surr_ids = []
  new_text = "?" （不完整 UTF-8）
  → surr/read 不推进，sent_offset 不变，输出 ""

步骤3：收到 <0xA0>（UTF-8 第3字节）
  decoded_ids = [e4, bd, a0]
  read_ids = [e4, bd, a0], surr_ids = []
  new_text = "你"（完整字符！）
  → 推进 surr_offset=0→3, read_offset=0→3
  → sent_offset=0→1，输出 "你"

步骤4-6：类似处理"好"

步骤7：收到 ▁world
  new_text = " world"，不以乱码结尾
  find_printable_text → " " (截到最后空格)，或若无后续则全量输出
```

（实际上现代 Qwen/Llama tokenizer 通常已合并了常见汉字 token，"你好"很可能是 1-2 个 token，不会逐字节生成。上面示例仅为说明极端情况。）

---

## 4.4 系统架构：tokenizer 进程的职责

`server.py` 中的 `tokenize_worker` 是一个独立进程（`multiprocessing.Process`），通过 ZMQ 消息队列与 API 服务器（前端）和调度器（后端）通信：

```
API Server  ──TokenizeMsg──→  tokenize_worker  ──UserMsg(input_ids)──→  Scheduler
API Server  ←──UserReply───   tokenize_worker  ←──DetokenizeMsg(token)── Scheduler
```

`tokenize_worker` 内部：
- 收到 `TokenizeMsg`：调用 `TokenizeManager.tokenize()`，把 messages/prompt 转为 `input_ids`，转发给 scheduler。
- 收到 `DetokenizeMsg`（每个生成步骤发来一个 `next_token`）：调用 `DetokenizeManager.detokenize()`，得到增量字符串，封装为 `UserReply` 发回前端。

**`python/minisgl/tokenizer/server.py` 第 61-85 行：**

```python
while True:
    pending_msg = _unwrap_msg(recv_listener.get())
    while len(pending_msg) < local_bs and not recv_listener.empty():
        pending_msg.extend(_unwrap_msg(recv_listener.get()))

    detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
    tokenize_msg   = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
    ...
    if len(detokenize_msg) > 0:
        replies = detokenize_manager.detokenize(detokenize_msg)
        ...
        send_frontend.put(batch_output)
```

这里有一个微批处理（micro-batching）优化：先拉取一条消息，如果队列非空且 batch 未满（`< local_bs`），继续拉取。这样在高并发下多个请求的 `DetokenizeMsg` 会被合并为一个 batch，由 `batch_decode` 统一处理，显著降低 Python 函数调用开销。

### `ensure_ascii=False` 的重要性

在 `api_server.py` 的流式输出路径（第 156 行和第 177 行）：

```python
yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
```

`json.dumps` 默认 `ensure_ascii=True`，会把所有非 ASCII 字符转义为 `\uXXXX` 形式。对于中文内容，"你好"会变成 `"\u4f60\u597d"`，客户端虽然能正确解析，但：

1. **流量放大**：每个中文字符从 3 字节（UTF-8）膨胀为 6 字节（`\uXXXX`），增加网络传输量约 2 倍。
2. **调试困难**：在 curl 或日志中看到的是转义序列，不直观。

设置 `ensure_ascii=False` 后，JSON 字符串直接输出 UTF-8 编码的中文，配合 `.encode()` 转为 `bytes` 正确传输。

---

## 4.5 设计决策与替代方案

### 为什么用差分而不是 token-by-token decode？

最朴素的方案是维护一个"已 decode 的文本"，每次直接 decode 所有历史 token：

```python
# 朴素方案（低效）
full_text = tokenizer.decode(all_ids)
new_text = full_text[len(prev_text):]
```

问题在于 `decode(all_ids)` 的时间复杂度是 O(N)，N 是序列长度。对于长文本生成（几千 token），总开销是 O(N²)。

mini-sglang 的设计只 decode `surr_offset` 之后的片段，已经确认安全的历史部分不再重复处理，均摊下来接近 O(1)。

### 为什么保留 surr 区域而不直接 decode 最新 token？

HuggingFace `tokenizer.decode([single_id])` 对某些 token 的处理可能带有 BOS/EOS/空格的副作用。更根本的原因是：单个字节级 token 本身无法形成合法 UTF-8，必须放在上下文中一起 decode 才能得到正确字符。`surr_offset` 保留了"待确认区域"的 id，使得重新 decode 时 tokenizer 拥有完整的字节序列。

### 为什么在独立进程而不是协程里做 tokenize/detokenize？

Tokenize 操作（尤其是 `apply_chat_template`）是 CPU 密集型，涉及 Jinja2 模板渲染和 BPE 匹配；`batch_decode` 同样是纯 CPU 工作。在 asyncio 事件循环内执行会阻塞整个前端，导致响应延迟上升。独立进程隔离了 GIL，也便于水平扩展（可以启动多个 tokenizer worker）。

---

## 4.6 动手实验

### 实验一：直接观察乱码现象

**目标**：亲眼验证逐 token decode 的乱码问题。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# 编码一个中文句子
text = "你好世界"
ids = tokenizer.encode(text, add_special_tokens=False)
print("token ids:", ids)

# 逐 token decode，观察每一步的输出
for i, tok_id in enumerate(ids):
    partial = tokenizer.decode([tok_id])
    cumulative = tokenizer.decode(ids[:i+1])
    print(f"  step {i}: single={repr(partial)}, cumulative={repr(cumulative)}")
```

预期观察：某些步骤的 `single` 输出包含 `?` 或乱码，而 `cumulative` 在字节凑齐后给出正确汉字。

### 实验二：验证三指针的 surr_offset 推进逻辑

**目标**：在不启动完整服务器的情况下，单独测试 `DetokenizeManager`。

```python
from transformers import AutoTokenizer
from minisgl.tokenizer.detokenize import DetokenizeManager
from minisgl.message import DetokenizeMsg

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
manager = DetokenizeManager(tokenizer)

text = "你好 world"
ids = tokenizer.encode(text, add_special_tokens=False)
print("input text:", repr(text))
print("token count:", len(ids))

uid = 42
outputs = []
for i, tok_id in enumerate(ids):
    finished = (i == len(ids) - 1)
    msg = DetokenizeMsg(uid=uid, next_token=tok_id, finished=finished)
    result = manager.detokenize([msg])
    outputs.append(result[0])
    print(f"  step {i} (token={tok_id}): incremental={repr(result[0])}")

print("reconstructed:", repr("".join(outputs)))
```

观察每一步 `incremental` 的值，注意：
- 字节级 token 出现时输出为 `""`。
- 字符凑齐后一次性输出完整字符。
- 英文 token 可能滞后一步（等待空格确认边界）。

### 实验三：Chat Template 的输出

**目标**：查看 `apply_chat_template` 的实际输出，理解 token id 的构成。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

messages = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user",   "content": "1+1等于几？"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
print("=== rendered prompt ===")
print(repr(prompt))

ids = tokenizer.encode(prompt, return_tensors="pt")
print("\n=== token count ===", ids.shape)
print("=== first 10 tokens ===", ids[0, :10].tolist())
print("=== last 10 tokens ===", ids[0, -10:].tolist())
```

注意观察：
- 模板是否添加了 `<|im_start|>` / `<|im_end|>` 等特殊 token？
- `add_generation_prompt=True` 在末尾添加了什么？
- 最后一个 token 对应的是什么字符串？（用 `tokenizer.decode([last_id])` 查看）

### 进阶实验：测量 batch_decode 的性能收益

对比单条 decode 循环和 batch_decode 的耗时，感受 micro-batching 的价值：

```python
import time
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# 模拟 32 个并发请求，每个有 ~50 个 token 的历史
batch_size = 32
seq_len = 50
import random
all_id_seqs = [[random.randint(0, 10000) for _ in range(seq_len)] for _ in range(batch_size)]

# 方案A：循环逐条 decode
t0 = time.perf_counter()
for _ in range(100):
    results = [tokenizer.decode(ids) for ids in all_id_seqs]
t1 = time.perf_counter()
print(f"loop decode:  {(t1-t0)*10:.1f} ms per call")

# 方案B：batch_decode
t0 = time.perf_counter()
for _ in range(100):
    results = tokenizer.batch_decode(all_id_seqs)
t1 = time.perf_counter()
print(f"batch_decode: {(t1-t0)*10:.1f} ms per call")
```

---

## 4.7 小结

本章从流式输出的工程需求出发，梳理了 tokenizer 模块的两大职责：

| 职责 | 实现类 | 关键点 |
|------|--------|--------|
| 输入处理 | `TokenizeManager` | `apply_chat_template` 统一处理 messages 和 plain text；`int32` 节省显存 |
| 增量解码 | `DetokenizeManager` | 三指针（`surr_offset` / `read_offset` / `sent_offset`）差分解码；`find_printable_text` 安全边界判断 |

核心设计要点：

1. **`surr_offset`** 是乱码防护的关键：它圈出"可能存在字节级不完整 token"的区域，每次从这里重新 decode，让 tokenizer 有机会拼合完整字符。
2. **`find_printable_text`** 处理英文词边界：以空格为安全刷新点，避免把一个词的前半部分提前推送出去。
3. **`ensure_ascii=False`** 是 API 层的必要配置：直接输出 UTF-8 字符，避免中文内容被 JSON 转义膨胀。
4. **micro-batching + `batch_decode`** 是吞吐量的关键优化：把多个请求的 detokenize 合并处理，摊薄 Python 调用开销。

**与后续章节的连接**：

`DetokenizeMsg` 中的 `next_token` 由调度器（Scheduler）在每个 decode step 后发出。第 5 章将深入调度器的工作循环，理解 prefill 和 decode 两个阶段如何交替执行，以及 KV Cache 如何在批次之间复用，进而理解 `DetokenizeMsg` 的生产时序。
