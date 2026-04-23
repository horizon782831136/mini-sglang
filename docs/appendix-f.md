# 附录 F：模型实现与神经网络层代码详解

本附录对 mini-sglang 项目中 `models/` 和 `layers/` 两个子包的全部源码进行逐文件的深度解析，涵盖每个公开类与函数的签名、参数语义、返回值以及关键实现细节。

---

## 第一节：models/ —— 模型层

### F.1.1 `models/base.py` — 模型基类

**文件职责**：定义所有 LLM 模型必须遵守的抽象接口，将 `BaseLLMModel` 同时继承自标准抽象基类 `ABC` 和层框架基类 `BaseOP`，以在模型级别统一 `forward` 协议。

#### 类：`BaseLLMModel`

```python
# python/minisgl/models/base.py，第 12-14 行
class BaseLLMModel(ABC, BaseOP):
    @abstractmethod
    def forward(self) -> torch.Tensor: ...
```

| 方面 | 说明 |
|------|------|
| 继承链 | `ABC`（抽象基类约束）、`BaseOP`（提供 `state_dict` / `load_state_dict` 能力） |
| 抽象方法 | `forward(self) -> torch.Tensor`：**无参数**，从全局上下文 `get_global_ctx()` 读取本次前向所需的 batch，返回 logits 张量 |

**关键设计**：`forward` 不接受任何显式参数，这是 mini-sglang 的核心约定——所有运行时状态（输入 token id、位置编码、注意力元数据等）均通过全局上下文 `GlobalContext` 传递，使模型实现与调度逻辑解耦。

---

### F.1.2 `models/config.py` — 模型配置

**文件职责**：提供两个不可变数据类 `RotaryConfig` 和 `ModelConfig`，以及从 HuggingFace `PretrainedConfig` 自动构建 `ModelConfig` 的工厂方法。

#### 数据类：`RotaryConfig`

```python
# python/minisgl/models/config.py，第 9-15 行
@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None
```

| 字段 | 类型 | 含义 |
|------|------|------|
| `head_dim` | `int` | 每个注意力头的维度大小，即 `hidden_size / num_attention_heads` |
| `rotary_dim` | `int` | 实际参与旋转编码的维度数；当前实现中要求 `rotary_dim == head_dim`（见 `rotary.py` 第 23 行的断言） |
| `max_position` | `int` | 预计算 cos/sin 缓存时覆盖的最大序列长度，来自 `config.max_position_embeddings` |
| `base` | `float` | RoPE 频率基数，对应 Llama 系列常用的 `rope_theta`（默认 10000.0，Llama-3 使用 500000.0） |
| `scaling` | `Dict[str, Any] \| None` | RoPE 缩放配置字典，`None` 表示使用标准 RoPE；非 `None` 时包含 `rope_type`（如 `"llama3"`）及缩放参数 |

#### 数据类：`ModelConfig`

```python
# python/minisgl/models/config.py，第 18-36 行
@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
```

所有字段均为只读（`frozen=True`）。各字段详细说明如下：

| 字段 | 来源字段 | 含义 |
|------|----------|------|
| `num_layers` | `config.num_hidden_layers` | Transformer 解码器层数 |
| `num_qo_heads` | `config.num_attention_heads` | Query/Output 注意力头数 |
| `num_kv_heads` | `config.num_key_value_heads`（缺省等于 `num_qo_heads`） | Key/Value 头数；`num_qo_heads / num_kv_heads` 即 GQA 分组比 |
| `head_dim` | `config.head_dim`（缺省 `hidden_size / num_attention_heads`） | 每头维度，部分模型（如 Qwen3）在 config 中显式指定 |
| `hidden_size` | `config.hidden_size` | 模型隐藏层宽度 |
| `vocab_size` | `config.vocab_size` | 词表大小，决定嵌入矩阵和 LM Head 的第一维 |
| `intermediate_size` | `config.intermediate_size` | 稠密 MLP 的中间层宽度（gate/up projection 的输出维度） |
| `rms_norm_eps` | `config.rms_norm_eps` | RMSNorm 中防止除零的 epsilon 值 |
| `rotary_config` | 由多个字段构建 | 见上文 `RotaryConfig` |
| `hidden_act` | `config.hidden_act` | 激活函数名称，支持 `"silu"`（Llama/Qwen3）和 `"gelu"`；用于在 `GatedMLP` 中选择激活函数 |
| `tie_word_embeddings` | `getattr(config, "tie_word_embeddings", False)` | 是否将 `embed_tokens` 权重与 LM Head 权重绑定 |
| `num_experts` | `config.num_local_experts` 或 `config.num_experts`（缺省 0） | MoE 专家总数；非 MoE 模型为 0 |
| `num_experts_per_tok` | `config.num_experts_per_tok`（缺省 0） | 每个 token 激活的专家数（top-k） |
| `moe_intermediate_size` | `config.moe_intermediate_size`（缺省 0） | MoE 专家内部 FFN 的中间维度，独立于稠密 `intermediate_size` |
| `norm_topk_prob` | `config.norm_topk_prob`（缺省 False） | 是否对 top-k 路由权重重新归一化；Qwen3-MoE 设为 `True` |
| `model_type` | `config.model_type`（缺省 `"llama"`） | 模型类型字符串，用于 `is_moe` 属性判断 |
| `architectures` | `config.architectures`（缺省 `["LlamaForCausalLM"]`） | HuggingFace 架构名列表，用于注册表查找 |

#### 属性：`is_moe`

```python
# python/minisgl/models/config.py，第 38-40 行
@property
def is_moe(self) -> bool:
    return "moe" in self.model_type
```

通过检查 `model_type` 字符串中是否包含 `"moe"` 子串来判断是否为混合专家模型，例如 `"qwen3_moe"` 返回 `True`，`"qwen3"` 返回 `False`。

#### 类方法：`from_hf`

```python
# python/minisgl/models/config.py，第 42-78 行
@classmethod
def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
```

**参数**：
- `config: PretrainedConfig`：HuggingFace transformers 库加载的预训练模型配置对象

**返回值**：`ModelConfig` 实例

**实现细节**：使用 `getattr(config, field, default)` 模式处理不同模型架构中字段命名不一致的问题，所有可能缺失的字段均提供安全默认值。`RotaryConfig` 在该方法中内联构建。

---

### F.1.3 `models/register.py` — 模型注册表

**文件职责**：维护架构名到模块路径及类名的映射字典，通过延迟 `importlib` 导入实现按需加载。

#### 注册表字典：`_MODEL_REGISTRY`

```python
# python/minisgl/models/register.py，第 5-10 行
_MODEL_REGISTRY = {
    "LlamaForCausalLM":    (".llama",     "LlamaForCausalLM"),
    "Qwen2ForCausalLM":    (".qwen2",     "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM":    (".qwen3",     "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),
}
```

字典的键是 HuggingFace `config.architectures[0]` 的值，值是一个二元组 `(相对模块路径, 类名)`。

#### 函数：`get_model_class`

```python
# python/minisgl/models/register.py，第 13-19 行
def get_model_class(model_architecture: str, model_config: ModelConfig):
```

**参数**：
- `model_architecture: str`：架构名，如 `"LlamaForCausalLM"`
- `model_config: ModelConfig`：已构建的模型配置对象

**返回值**：已实例化的模型对象（`BaseLLMModel` 子类实例）

**实现细节**：
1. 从 `_MODEL_REGISTRY` 查找对应的 `(module_path, class_name)` 元组，未找到则抛出 `ValueError`
2. 调用 `importlib.import_module(module_path, package=__package__)` 进行**相对导入**，延迟至首次调用时才真正导入模型模块
3. 通过 `getattr(module, class_name)` 获取类对象，立即以 `model_config` 为参数实例化并返回

**设计意图**：注册表模式使新增模型仅需在字典中添加一行，无需修改任何调用方代码；延迟导入则避免了所有模型模块在启动时全部加载。

---

### F.1.4 `models/weight.py` — 权重加载

**文件职责**：从 SafeTensors 格式的权重文件中加载参数，完成张量并行分片（sharding）和权重合并（merging）两个关键变换后返回可直接载入模型的 `state_dict`。

#### 私有函数：`_shard_state_dict`

```python
# python/minisgl/models/weight.py，第 13-42 行
def _shard_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
```

**参数**：`state_dict`：从文件加载的原始完整权重字典

**返回值**：按当前 TP rank 分片后的权重字典

**分片规则**（取决于权重 key 中的子串）：

| 匹配子串 | 分片方式 | 对应层 |
|----------|----------|--------|
| `.q_proj`, `.k_proj`, `.v_proj`, `.gate_proj`, `.up_proj` | `dim=0` 均匀切分，取第 `r` 块 | 列并行投影（Column Parallel） |
| `.o_proj`, `.down_proj` | `dim=1` 均匀切分，取第 `r` 块 | 行并行投影（Row Parallel） |
| `lm_head`, `embed_tokens` | 词表维度（`dim=0`）按 `div_ceil` 切分 | 词表并行嵌入 |
| 其余 | 不切分，全量复制 | RMSNorm、bias 等 |

词表分片使用向上取整（`div_ceil`）而非平均分割，因此最后一个 rank 可能持有比其他 rank 少的词向量，实际有效范围由 `vocab_start_idx:vocab_end_idx` 精确计算：

```python
# python/minisgl/models/weight.py，第 34-39 行
num_embeddings_per_partition = div_ceil(num_embeddings, n)
vocab_start_idx = r * num_embeddings_per_partition
vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
```

#### 私有函数：`_merge_state_dict`

```python
# python/minisgl/models/weight.py，第 45-68 行
def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
```

**参数**：分片后（或原始）的权重字典

**返回值**：合并后的权重字典，键名已重命名

**合并规则**：

| 输入键 | 合并操作 | 输出键 |
|--------|----------|--------|
| `.q_proj` + `.k_proj` + `.v_proj` | `torch.cat([q, k, v], dim=0)` | `.qkv_proj` |
| `.gate_proj` + `.up_proj` | `torch.cat([gate, up], dim=0)` | `.gate_up_proj` |
| `.k_proj`, `.v_proj`, `up_proj`（孤立） | 跳过（已在上一步消费） | — |
| 其余 | 直接复制 | 不变 |

**关键细节**：合并操作通过直接修改传入的 `state_dict`（使用 `del`）来避免内存浪费，被消费的键从原字典中删除。合并后的权重 key 与模型代码中 `LinearQKVMerged`（`.qkv_proj`）和 `LinearColParallelMerged`（`.gate_up_proj`）的参数名完全对应。

#### 公开函数：`load_weight`

```python
# python/minisgl/models/weight.py，第 71-87 行
def load_weight(model_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
```

**参数**：
- `model_path: str`：HuggingFace 模型路径（本地路径或 Hub 模型 ID）
- `device: torch.device`：权重加载目标设备

**返回值**：经过分片与合并处理的最终 `state_dict`

**执行流程**：
1. `download_hf_weight(model_path)` 确保权重文件已下载到本地
2. 按文件名排序，逐个用 `safetensors.safe_open` 直接在目标设备上加载（`device=device_str`），避免 CPU 到 GPU 的额外拷贝
3. TP rank 0 显示 tqdm 进度条，其余 rank 禁用（`disable_tqdm`）
4. TP size > 1 时调用 `_shard_state_dict` 分片
5. 最终调用 `_merge_state_dict` 合并 QKV 和 gate/up

---

### F.1.5 `models/utils.py` — 模型工具函数（共用组件）

**文件职责**：提供可被多种模型架构复用的标准组件：门控 MLP（`GatedMLP`）、MoE MLP（`MoEMLP`）和带 RoPE 的多头注意力（`RopeAttn`）。

#### 类：`GatedMLP`

```python
# python/minisgl/models/utils.py，第 25-50 行
class GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

**构造参数**：`config: ModelConfig`

**内部结构**：
- `gate_up_proj: LinearColParallelMerged`：将 `hidden_size` 投影到 `[intermediate_size, intermediate_size]`（gate 和 up 的合并列并行投影）
- `act_fn`：根据 `config.hidden_act` 从 `{"silu": silu_and_mul, "gelu": gelu_and_mul}` 中选择激活函数，不支持的激活函数立即抛出 `ValueError`
- `down_proj: LinearRowParallel`：将 `intermediate_size` 投影回 `hidden_size`

**`forward` 流程**：
1. `gate_up = gate_up_proj(x)`：一次矩阵乘法同时得到 gate 和 up 两部分，合并在 `dim=0`
2. `y = act_fn(gate_up)`：对合并结果施加门控激活（`silu_and_mul` 内部将输入对半切分，前半乘以 silu(后半)）
3. `return down_proj(y)`：行并行投影并 all-reduce

中间张量 `x` 和 `gate_up` 在使用后立即 `del` 以释放显存，通过 `@nvtx_annotate("MLP")` 装饰器标注 NVTX 性能剖析范围。

#### 类：`MoEMLP`

```python
# python/minisgl/models/utils.py，第 53-78 行
class MoEMLP(BaseOP):
    def __init__(self, config: ModelConfig): ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
```

**构造参数**：`config: ModelConfig`

**内部结构**：
- `experts: MoELayer`：管理所有专家权重并调用融合 MoE 计算后端
- `gate: LinearReplicated`：路由器线性层，输入 `hidden_size`，输出 `num_experts`（全量复制，非分片）

**`forward` 流程**：
1. 展平输入为 `[num_tokens, hidden_dim]`
2. 通过 `gate` 计算每个 token 对所有专家的路由 logits
3. 调用 `experts.forward(hidden_states, router_logits)` 执行 top-k 选路和融合计算
4. 恢复原始 shape 后返回

#### 类：`RopeAttn`

```python
# python/minisgl/models/utils.py，第 81-125 行
class RopeAttn(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
    ): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

**构造参数**：
- `config: ModelConfig`
- `layer_id: int`：所在层的索引，传递给 `AttentionLayer` 用于 KV-Cache 定位
- `has_attn_bias: bool`（关键字参数）：QKV 投影是否带 bias；Qwen2 为 `True`，Llama/Qwen3 为 `False`
- `has_qk_norm: bool`（关键字参数）：是否对 Q、K 做 RMSNorm；Qwen3 系列为 `True`

**内部结构**：
- `qkv_proj: LinearQKVMerged`：合并 QKV 投影
- `q_norm / k_norm: RMSNorm | None`：QK 归一化（仅 `has_qk_norm=True` 时创建）
- `attn: AttentionLayer`：执行 RoPE + 注意力计算
- `o_proj: LinearOProj`：输出投影并 all-reduce

**`forward` 流程**：
1. `qkv = qkv_proj(x)`：一次投影同时得到 Q、K、V
2. `o = attn(qkv)`：内部拆分 QKV，施加可选 QK norm，应用 RoPE，调用 attention backend
3. `return o_proj(o)`：行并行输出投影

注意 `q_norm` 和 `k_norm` 的实例被同时传入 `RopeAttn` 的 `self` 属性和 `AttentionLayer` 构造函数——两处持有同一对象，状态共享，权重加载只需走一条路径。

---

### F.1.6 `models/llama.py` — Llama-3 实现

**文件职责**：基于通用组件组装 Llama-3 因果语言模型，三层嵌套结构 `LlamaForCausalLM → LlamaModel → LlamaDecoderLayer`。

#### 类：`LlamaDecoderLayer`

```python
# python/minisgl/models/llama.py，第 18-43 行
class LlamaDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int): ...
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...
```

**与 Qwen2/Qwen3 的差异**：`LlamaAttn` 初始化时使用默认参数 `has_attn_bias=False`、`has_qk_norm=False`，即无 attention bias 也无 QK norm。

**`forward` 签名说明**：接受和返回 `(x, residual)` 对，配合 `RMSNormFused` 的 fused add+norm 机制——`residual` 携带上一层的归一化前输出，避免单独做残差加法再归一化的两次内存读写。

#### 类：`LlamaModel`

```python
# python/minisgl/models/llama.py，第 46-65 行
class LlamaModel(BaseOP):
    def __init__(self, config: ModelConfig): ...
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...
```

**`forward` 细节**：首层调用 `layer.forward(x, residual=None)`，`RMSNormFused` 在 `residual is None` 时退化为普通 rmsnorm 并将原始 `x` 作为新 residual 返回；从第二层起传入非 None residual，执行融合的 add+rmsnorm。

#### 类：`LlamaForCausalLM`

```python
# python/minisgl/models/llama.py，第 68-85 行
class LlamaForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig): ...
    def forward(self) -> torch.Tensor: ...
```

**构造细节**：当 `config.tie_word_embeddings=True` 时，`ParallelLMHead` 的 `tied_embedding` 参数接收 `self.model.embed_tokens` 的**同一引用**，LM Head 的 forward 直接使用 embed_tokens 的权重矩阵做线性变换，无额外参数。

**`forward` 细节**：从 `get_global_ctx().batch.input_ids` 获取 token id，经过 `LlamaModel` 得到隐藏状态，再经 `ParallelLMHead` 得到 logits。

---

### F.1.7 `models/qwen2.py` — Qwen2 实现

**文件职责**：与 Llama 结构完全相同，唯一区别是 attention bias 设置：Qwen2 的 QKV 投影带 bias（`has_attn_bias=True`），Qwen3 有 QK norm 而 Qwen2 没有。

```python
# python/minisgl/models/qwen2.py，第 20 行
self.self_attn = Qwen2Attn(config, layer_id, has_qk_norm=False, has_attn_bias=True)
```

其余结构（`Qwen2Model`、`Qwen2ForCausalLM`）与 Llama 对应类完全同构，仅类名不同。

---

### F.1.8 `models/qwen3.py` — Qwen3 实现

**文件职责**：Qwen3 稠密版本，与 Qwen2 的差异是启用了 QK RMSNorm（`has_qk_norm=True`），同时去掉了 attention bias（`has_attn_bias` 默认为 `False`）。

```python
# python/minisgl/models/qwen3.py，第 20 行
self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
```

**QK Norm 作用**：对每个注意力头的 Q 和 K 向量独立做 RMSNorm，稳定训练时的注意力权重幅度，是 Qwen3 相比 Qwen2 的主要结构改进。

---

### F.1.9 `models/qwen3_moe.py` — Qwen3-MoE 实现

**文件职责**：Qwen3 混合专家版本，与稠密 Qwen3 的唯一区别是将 `GatedMLP` 替换为 `MoEMLP`。

```python
# python/minisgl/models/qwen3_moe.py，第 11 行
from .utils import MoEMLP as Qwen3MLP  # 稠密版本用 GatedMLP
```

注意 `qwen3_moe.py` 中的 `Qwen3DecoderLayer` 和 `Qwen3Model` 类名与 `qwen3.py` 中的同名类**完全相同**，但位于不同模块，注册表按模块精确定位，不存在冲突。

`Qwen3MoeForCausalLM.forward` 结构与其他模型完全一致，MoE 逻辑完全封装在 `MoEMLP` 内部。

---

## 第二节：layers/ —— 神经网络层

### F.2.1 `layers/base.py` — 层基类

**文件职责**：提供整个框架的核心基类 `BaseOP`，以及专用于无状态操作的 `StateLessOP` 和列表容器 `OPList`，三者共同实现了不依赖 `torch.nn.Module` 的轻量级参数管理机制。

#### 私有函数：`_collect_expert_keys`

```python
# python/minisgl/layers/base.py，第 14-45 行
def _collect_expert_keys(
    state_dict: _STATE_DICT, prefix: str, param_name: str
) -> List[str]:
```

**功能**：在 `state_dict` 中收集属于某个专家组的所有权重键。

**两阶段查找策略**：
1. **快速路径（O(num_experts) 直接查找）**：对 `idx=0,1,2,...` 依次构造 `"{prefix}.{idx}.{param_name}{suffix}"` 直接在字典中查找，遇到第一个缺失的 idx 则停止
2. **慢速回退（线性扫描）**：若快速路径没有找到任何键（非标准命名约定），线性遍历整个 `state_dict`，用正则 `r"experts\.(\d+)\."` 提取专家编号并排序

#### 类：`BaseOP`

```python
# python/minisgl/layers/base.py，第 52-106 行
class BaseOP:
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT: ...
    def load_state_dict(self, state_dict: _STATE_DICT, *, prefix: str = "", _internal: bool = False) -> None: ...
```

**`state_dict` 实现**：递归遍历 `self.__dict__`，跳过以 `_` 开头的私有属性，将 `torch.Tensor` 类型的属性收集为叶子节点，将 `BaseOP` 子类属性递归收集，键名为点分隔的前缀路径（如 `"model.layers.0.self_attn.qkv_proj.weight"`）。

**`load_state_dict` 实现**：同样递归遍历 `self.__dict__`，关键逻辑在两种情况下分支：

```python
# python/minisgl/layers/base.py，第 80-98 行
if isinstance(param, torch.Tensor):
    if "experts" in prefix:
        # 专家权重：收集所有专家的同名参数，stack 成三维张量
        matched_keys = _collect_expert_keys(state_dict, prefix, name)
        items = [state_dict.pop(k) for k in matched_keys]
        item = torch.stack(items, dim=0)  # [num_experts, out, in]
    else:
        # 普通参数：直接按键名弹出
        item = state_dict.pop(_concat_prefix(prefix, name))
    assert param.shape == item.shape and param.dtype == item.dtype
    setattr(self, name, item)
```

**专家权重处理的关键细节**：HuggingFace 的 MoE 模型通常将每个专家的权重存储为独立的 key（如 `experts.0.gate_proj.weight`、`experts.1.gate_proj.weight`），而 `MoELayer` 内部将所有专家的同名矩阵合并为一个三维张量（`[num_experts, out_dim, in_dim]`）。`_collect_expert_keys` + `torch.stack` 完成了这个维度扩展的合并操作。

**完整性校验**：非内部调用（`_internal=False`，即根节点）时，若 `state_dict` 中仍有剩余键，抛出 `RuntimeError`，确保权重完全匹配。

#### 类：`StateLessOP`

```python
# python/minisgl/layers/base.py，第 109-126 行
class StateLessOP(BaseOP):
```

**用途**：用于没有可学习参数的操作层（如 `AttentionLayer`、`RotaryEmbedding`），重写 `state_dict` 返回空字典，`load_state_dict` 仅做完整性检查而不实际加载任何参数。

#### 类：`OPList`

```python
# python/minisgl/layers/base.py，第 129-154 行
class OPList(BaseOP, Generic[T]):
    def __init__(self, ops: List[T]): ...
    def state_dict(...) -> _STATE_DICT: ...
    def load_state_dict(...) -> None: ...
```

**用途**：类似 `torch.nn.ModuleList`，为 `BaseOP` 层的有序列表提供统一的 `state_dict` / `load_state_dict`，键名为数字索引（如 `"layers.0"`、`"layers.1"`）。

---

### F.2.2 `layers/linear.py` — 线性层（张量并行）

**文件职责**：提供五种支持张量并行的线性层变体，统一建模"全局尺寸 vs 本地分片尺寸"的概念。

#### 类：`_LinearTPImpl`（内部基类）

```python
# python/minisgl/layers/linear.py，第 13-32 行
class _LinearTPImpl(BaseOP):
    def __init__(
        self,
        full_isize: int, full_osize: int,
        local_isize: int, local_osize: int,
        has_bias: bool,
    ): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

**核心思想**：用 `full_*` 记录全局维度（用于文档/验证），用 `local_*` 决定实际分配的张量大小。`weight` 的形状为 `[local_osize, local_isize]`，`bias` 的形状为 `[local_osize]`（可选）。

`forward` 仅调用 `F.linear(x, self.weight, self.bias)`，无任何通信，通信由子类视需要添加。

#### 类：`LinearReplicated`

```python
# python/minisgl/layers/linear.py，第 35-53 行
class LinearReplicated(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool): ...
```

`local_isize == full_isize`，`local_osize == full_osize`——**不分片**，每个 GPU 持有完整权重矩阵。用于 MoE 路由器（`gate`），因为路由决策需要看到完整的专家集合。

#### 类：`LinearColParallelMerged`

```python
# python/minisgl/layers/linear.py，第 56-68 行
class LinearColParallelMerged(_LinearTPImpl):
    def __init__(self, input_size: int, output_sizes: List[int], has_bias: bool): ...
```

**列并行（Column Parallel）**：沿输出维度（`dim=0`，即 weight 的行）切分。接受多个输出尺寸列表 `output_sizes`（对应合并矩阵中的多个分块），每个尺寸必须能被 `tp_size` 整除（由 `div_even` 强制检查）。

用于 `gate_up_proj`（gate 和 up 合并的列并行投影），本地持有 `[sum(output_sizes) / tp_size, input_size]` 的权重。

#### 类：`LinearQKVMerged`

```python
# python/minisgl/layers/linear.py，第 71-88 行
class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self, hidden_size: int, head_dim: int,
        num_qo_heads: int, num_kv_heads: int, has_bias: bool
    ): ...
```

**GQA 感知的 QKV 合并列并行**：

```
GQA_ratio = num_qo_heads / num_kv_heads
local_num_kv = num_kv_heads / tp_size
full_osize  = (GQA_ratio + 2) * num_kv_heads * head_dim
local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
```

`+2` 对应一组 K 头和一组 V 头，`GQA_ratio` 对应 Q 头。以 Llama-3-8B（`num_qo_heads=32, num_kv_heads=8`）为例，`GQA_ratio=4`，每个 KV 头对应 4 个 Q 头，合并矩阵输出维度按 `[Q, K, V]` 排列。

#### 类：`LinearOProj`

```python
# python/minisgl/layers/linear.py，第 91-106 行
class LinearOProj(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

**行并行输出投影（特化版）**：`local_isize = input_size / tp_size`（沿输入维度分片）。`forward` 在矩阵乘法后调用 `all_reduce` 汇总各 rank 的部分和。专为注意力输出投影（`o_proj`）设计，语义上等同于 `LinearRowParallel` 但名称更清晰。

#### 类：`LinearRowParallel`

```python
# python/minisgl/layers/linear.py，第 109-127 行
class LinearRowParallel(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

**行并行（Row Parallel）**：与 `LinearOProj` 结构完全相同，`local_isize = input_size / tp_size`，`forward` 后做 `all_reduce`。用于 `down_proj`（MLP 的输出投影）。

**张量并行通信汇总**：

| 类 | 分片维度 | 通信操作 |
|----|----------|----------|
| `LinearReplicated` | 无分片 | 无 |
| `LinearColParallelMerged` | `weight` 行方向（output dim） | 无（输出已分片，由下游 Row Parallel 的 all-reduce 归聚） |
| `LinearQKVMerged` | `weight` 行方向 | 无 |
| `LinearOProj` | `weight` 列方向（input dim） | all-reduce |
| `LinearRowParallel` | `weight` 列方向 | all-reduce |

---

### F.2.3 `layers/embedding.py` — 嵌入层

**文件职责**：提供词表并行嵌入 `VocabParallelEmbedding` 和并行 LM Head `ParallelLMHead`，两者共享同一套词表分区逻辑。

#### 类：`VocabParallelEmbedding`

```python
# python/minisgl/layers/embedding.py，第 14-51 行
class VocabParallelEmbedding(BaseOP):
    def __init__(self, num_embeddings: int, embedding_dim: int): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

**分区计算**：

```python
# python/minisgl/layers/embedding.py，第 25-28 行
self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
start_idx = self.num_embeddings_tp * tp_rank
finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
self.vocab_range = (start_idx, finish_idx - start_idx)
self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
```

使用 `div_ceil` 向上取整，所有 rank 持有大小相同的权重矩阵（末尾 rank 的部分词向量为填充），`vocab_range` 记录该 rank 负责的词表起始索引和实际长度。

**`forward` 的 `torch.where` 实现（张量并行 embedding 的核心）**：

```python
# python/minisgl/layers/embedding.py，第 37-40 行
mask = (x >= start) & (x < start + length)
clamped = torch.clamp(x - start, 0, length - 1)
y = torch.where(
    mask.unsqueeze(-1),
    F.embedding(clamped, self.weight),
    torch.zeros(x.shape[0], self.weight.shape[1],
                device=self.weight.device, dtype=self.weight.dtype)
)
```

**工作原理**：
1. `mask`：布尔张量，标记哪些 token id 落在当前 rank 的词表分区内
2. `clamped`：将 token id 平移并裁剪到 `[0, length-1]` 范围，使 `F.embedding` 可以正常查表（不会越界）——即使 token id 不在当前 rank 的范围内，钳制后的值也是合法下标，但结果会被 `torch.where` 掩掉
3. `torch.where`：仅保留有效 rank 的嵌入结果，其余填零
4. `all_reduce`：各 rank 将自己负责的嵌入行填入，其余行为零；all_reduce 将所有 rank 的结果相加，等效于每个 token 取到正确的嵌入向量

这个实现的巧妙之处在于：通过"安全钳制 + 掩码清零 + all-reduce"完全避免了条件分支和动态索引，对 GPU 高度友好。

#### 类：`ParallelLMHead`

```python
# python/minisgl/layers/embedding.py，第 54-119 行
class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int, embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

**Tied Embedding 处理**：当 `tie_word_embeddings=True` 时，`load_state_dict` 直接从 `state_dict` 中弹出并丢弃 `lm_head.weight`（权重已在 `embed_tokens` 中加载），`state_dict` 方法同样返回空字典，避免双重保存。

**`forward` 实现细节**：

```python
# python/minisgl/layers/embedding.py，第 97-107 行
if batch.is_prefill:
    indices = batch.attn_metadata.get_last_indices(bs)
    x = x[indices].contiguous()  # 仅取每个序列最后一个 token 的隐藏状态
module = self.tied_embedding or self
logits = F.linear(x, module.weight, self.bias)
```

prefill 阶段只需要每个序列**最后一个** token 的 logits 用于采样，通过 `get_last_indices` 提前切片，避免对整个序列长度的矩阵乘法。

**多 GPU all-gather 重组逻辑**：

```python
# python/minisgl/layers/embedding.py，第 108-119 行
output_tensor = self._comm.all_gather(logits)
if bs == 1:
    return output_tensor.view(1, -1)[:, :self.num_embeddings]
output_tensor = output_tensor.view((self.tp_size,) + input_shape)
output_tensor = output_tensor.movedim(0, -1)
output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
return output_tensor[:, :self.num_embeddings]
```

all-gather 将各 rank 的局部 logits 拼接，通过 `movedim` + `reshape` 将 `[tp_size, batch, vocab_shard]` 重排为 `[batch, tp_size * vocab_shard]`，最后截取到实际词表大小 `num_embeddings`（去除末尾填充）。

---

### F.2.4 `layers/norm.py` — 归一化层

**文件职责**：封装 FlashInfer 的 RMSNorm 和融合 add+RMSNorm 算子，提供 `RMSNorm` 和 `RMSNormFused` 两个类。

#### 类：`RMSNorm`

```python
# python/minisgl/layers/norm.py，第 8-20 行
class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def forward_inplace(self, x: torch.Tensor) -> None: ...
```

**参数**：
- `size: int`：归一化的特征维度
- `eps: float`：数值稳定性 epsilon，来自 `config.rms_norm_eps`

**`forward`**：调用 `flashinfer.rmsnorm(x, self.weight, self.eps)`，返回归一化结果（**不修改输入**）。

**`forward_inplace`**：调用同一函数但传入 `out=x`，**原地修改**输入，用于 QK-norm（`AttentionLayer` 中对 Q、K 的每头归一化）。

#### 类：`RMSNormFused`

```python
# python/minisgl/layers/norm.py，第 23-38 行
class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None: ...
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...
```

**返回值**：`(normalized, residual)` 元组

**两种行为模式**：

```python
# python/minisgl/layers/norm.py，第 32-38 行
if residual is None:
    return self.rmsnorm(x, self.weight, self.eps), x  # 首层：普通 norm，原始 x 成为新 residual
else:
    self.fused_add_rmsnorm(x, residual, self.weight, self.eps)  # 后续层：原地 add+norm
    return x, residual
```

**融合操作的语义**（`fused_add_rmsnorm` 原地修改 `x` 和 `residual`）：
- `residual += x`（residual 累积残差）
- `x = rmsnorm(residual)`（x 成为归一化结果）

这样，模型中的残差连接（`output = x + residual`）和后续归一化（`normalized = rmsnorm(output)`）被融合为单次内存访问，相比分离操作节省约一半的显存读写。

---

### F.2.5 `layers/attention.py` — 注意力层

**文件职责**：实现 `AttentionLayer`，负责将 QKV 投影结果分拆、执行可选 QK norm、应用 RoPE，并委托 attention backend（FlashAttention/FlashInfer 等）完成实际的注意力计算和 KV-Cache 操作。

#### 类：`AttentionLayer`

```python
# python/minisgl/layers/attention.py，第 18-57 行
class AttentionLayer(StateLessOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int, num_kv_heads: int, head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ): ...
    def forward(self, qkv: torch.Tensor) -> torch.Tensor: ...
```

继承 `StateLessOP`：`AttentionLayer` 本身不拥有任何可学习参数（权重在 `qkv_proj`、`o_proj`、`q_norm`、`k_norm` 中），因此 `state_dict` 返回空字典。

**构造逻辑**：

```python
# python/minisgl/layers/attention.py，第 32-36 行
self.num_qo_heads = div_even(num_qo_heads, tp_size)   # 本地 Q 头数
self.num_kv_heads = div_even(num_kv_heads, tp_size)   # 本地 KV 头数
self.qo_attn_dim = self.num_qo_heads * head_dim        # Q 维度
self.kv_attn_dim = self.num_kv_heads * head_dim        # KV 维度
```

**`forward` 流程**：

```python
# python/minisgl/layers/attention.py，第 47-57 行
q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
if self.q_norm:
    self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
if self.k_norm:
    self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
q, k = self.rotary.forward(ctx.batch.positions, q, k)
q = q.view(-1, self.num_qo_heads, self.head_dim)
o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
return o.view(-1, self.qo_attn_dim)
```

1. `split`：按 `[qo_dim, kv_dim, kv_dim]` 从合并 QKV 张量中切分
2. QK norm：以 `[num_tokens, num_heads, head_dim]` 视图原地归一化（每头独立归一化）
3. RoPE：对 Q、K 原地施加旋转位置编码
4. `attn_backend.forward`：实际的 FlashAttention 调用，传入 `layer_id` 用于定位 KV-Cache 槽位
5. 重整形输出为 `[num_tokens, qo_attn_dim]` 供 `o_proj` 使用

---

### F.2.6 `layers/rotary.py` — RoPE 位置编码

**文件职责**：实现旋转位置编码（Rotary Position Embedding），支持标准 RoPE 和 Llama-3 风格的频率缩放变体，通过 `@functools.cache` 在进程内全局缓存 RoPE 实例。

#### 类：`RotaryEmbedding`

```python
# python/minisgl/layers/rotary.py，第 12-52 行
class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int, rotary_dim: int, max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...
```

**cos/sin 缓存预计算**：

```python
# python/minisgl/layers/rotary.py，第 24-32 行
inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
if post_process is not None:
    inv_freq = post_process(inv_freq)
t = torch.arange(max_position_embeddings, dtype=torch.float)
freqs = torch.einsum("i,j -> ij", t, inv_freq)
cos = freqs.cos()
sin = freqs.sin()
self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
```

标准 RoPE 的逆频率公式：`inv_freq[i] = 1 / (base^(2i/d))`（`i=0,1,...,d/2-1`）。通过外积 `einsum("i,j->ij", positions, inv_freq)` 一次性计算所有位置的频率，拼接 cos 和 sin 缓存为 `[max_position, head_size]`（`head_size = cos_dim + sin_dim = rotary_dim/2 + rotary_dim/2 = rotary_dim`）。

**`post_process` 钩子**：允许在固化缓存之前对 `inv_freq` 进行修改，Llama-3 的 RoPE 缩放通过此机制实现（见 `_get_rope`）。

**`forward`**：调用 `flashinfer.apply_rope_with_cos_sin_cache_inplace`，直接原地修改 query 和 key，无额外内存分配。返回修改后的同一对象（`query`, `key`）。

**约束**：当前实现中 `rotary_dim == head_size` 是强制要求（第 23 行的 `assert`），`head_size` 必须为 `[64, 128, 256, 512]` 之一（FlashInfer 算子的支持范围）。

#### 私有函数：`_get_rope`

```python
# python/minisgl/layers/rotary.py，第 55-90 行
def _get_rope(
    head_dim: int, rotary_dim: int, max_position: int, base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
```

处理 `rope_scaling` 为 `None`（标准 RoPE）或包含 `rope_type: "llama3"` 的 Llama-3 缩放 RoPE。

**Llama-3 RoPE 缩放的 `post_process` 实现**：

```python
# python/minisgl/layers/rotary.py，第 72-86 行
def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
    wave_len = 2 * math.pi / inv_freq
    if low_freq_factor == high_freq_factor:
        return torch.where(
            wave_len < original_max_position / high_freq_factor,
            inv_freq,
            inv_freq / scaling_factor,
        )
    delta = high_freq_factor - low_freq_factor
    smooth = (original_max_position / wave_len - low_freq_factor) / delta
    smooth = torch.clamp(smooth, 0, 1)
    factor = (1 - smooth) / scaling_factor + smooth
    return factor * inv_freq
```

Llama-3 的 RoPE 缩放根据每个频率分量的波长与 `original_max_position` 的比值进行分段处理：
- **高频分量**（波长短）：不缩放，保留原始 `inv_freq`
- **低频分量**（波长长）：除以 `scaling_factor` 进行缩放，扩展远程依赖能力
- **中间区域**：通过 `smooth` 平滑插值，避免硬截断

#### 全局函数：`get_rope`（缓存工厂）

```python
# python/minisgl/layers/rotary.py，第 101-119 行
@functools.cache
def get_rope(
    head_dim: int, rotary_dim: int, max_position: int, base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
```

**`@functools.cache` 装饰**：相同参数的调用返回同一个 `RotaryEmbedding` 实例，确保整个模型中所有层共享同一份 cos/sin 缓存，节省显存（cos/sin 缓存大小为 `max_position × head_size × 4 bytes`，对 Llama-3 约 16 MB）。

**参数类型约束**：`functools.cache` 要求所有参数可哈希，`rope_scaling` 字典通过转换为嵌套元组（`Tuple[Tuple[str, Any], ...]`）满足该要求；`None` 值（标准 RoPE）也是可哈希的。

**Meta 设备处理**：

```python
# python/minisgl/layers/rotary.py，第 110-118 行
t = torch.tensor([])
if t.device == torch.device("meta"):
    if _ROPE_DEVICE is None:
        raise RuntimeError(...)
    with torch.device(_ROPE_DEVICE):
        return _get_rope(...)
```

当全局默认设备为 `meta`（用于延迟初始化的模型并行场景）时，RoPE 的 cos/sin 缓存无法在 meta 设备上计算，因此强制切换到 `_ROPE_DEVICE`（由 `set_rope_device()` 预先设置）。

#### 函数：`set_rope_device`

```python
# python/minisgl/layers/rotary.py，第 96-98 行
def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device
```

在模型初始化前调用，指定 RoPE 缓存应分配的目标设备（通常为实际的 CUDA 设备）。

---

### F.2.7 `layers/activation.py` — 激活函数

**文件职责**：封装 FlashInfer 的融合门控激活函数，提供 `silu_and_mul` 和 `gelu_and_mul` 两个函数。

#### 函数：`silu_and_mul`

```python
# python/minisgl/layers/activation.py，第 9-12 行
def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import silu_and_mul
    return silu_and_mul(x, out=out)
```

**参数**：
- `x: torch.Tensor`：形状为 `[..., 2 * intermediate_size]` 的合并 gate+up 张量
- `out: torch.Tensor | None`：可选的输出缓冲区（用于原地写入，避免额外内存分配）

**返回值**：形状为 `[..., intermediate_size]` 的张量，计算 `x[:, :d] * silu(x[:, d:])`，其中 `d = intermediate_size`

FlashInfer 的融合实现在单次内核调用中完成切分、silu 计算和逐元素相乘，相比分离操作降低内存带宽消耗。

#### 函数：`gelu_and_mul`

与 `silu_and_mul` 完全相同结构，激活函数替换为 GELU，供 `hidden_act="gelu"` 的模型使用。

---

### F.2.8 `layers/moe.py` — MoE 层

**文件职责**：`MoELayer` 封装所有专家权重张量，并通过全局上下文获取 MoE backend 进行前向计算，同时处理张量并行下的 all-reduce 聚合。

#### 类：`MoELayer`

```python
# python/minisgl/layers/moe.py，第 9-59 行
class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int, top_k: int,
        hidden_size: int, intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ): ...
    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor): ...
```

**权重布局**：

```python
# python/minisgl/layers/moe.py，第 33-43 行
intermediate_size_per_partition = div_even(intermediate_size, tp_size)
self.gate_up_proj = torch.empty(
    num_experts,
    2 * intermediate_size_per_partition,   # gate 和 up 合并
    hidden_size,
)
self.down_proj = torch.empty(
    num_experts,
    hidden_size,
    intermediate_size_per_partition,
)
```

- `gate_up_proj`：形状 `[E, 2 * (N/tp), H]`，`E` 为专家数，`N` 为 `intermediate_size`，`H` 为 `hidden_size`
- `down_proj`：形状 `[E, H, N/tp]`

张量并行的分片体现在 `intermediate_size_per_partition`（中间维度在 tp 维度上均匀切分）。

**`forward`**：

```python
# python/minisgl/layers/moe.py，第 45-59 行
final_hidden_states = ctx.moe_backend.forward(
    hidden_states=hidden_states,
    w1=self.gate_up_proj, w2=self.down_proj,
    gating_output=router_logits,
    topk=self.top_k,
    renormalize=self.renormalize,
    activation=self.activation,
    apply_router_weight_on_input=self.apply_router_weight_on_input,
)
if self.tp_size > 1:
    final_hidden_states = self._comm.all_reduce(final_hidden_states)
```

前向计算完全委托给 `moe_backend`（通常为 `FusedMoe`），其结果为各 TP rank 上的部分和，需要 all-reduce 归聚。

---

## 第三节：moe/ —— MoE 后端

### F.3.1 `moe/base.py` — MoE 基类

**文件职责**：定义 MoE 计算后端的抽象接口，供不同实现（如 Triton 融合核、原生 PyTorch）遵循。

#### 类：`BaseMoeBackend`

```python
# python/minisgl/moe/base.py，第 6-18 行
class BaseMoeBackend(ABC):
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor, w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int, renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor: ...
```

**抽象方法参数**：

| 参数 | 含义 |
|------|------|
| `hidden_states` | 输入隐藏状态，形状 `[num_tokens, hidden_size]` |
| `w1` | 所有专家的 gate+up 权重，形状 `[num_experts, 2*intermediate_size_per_tp, hidden_size]` |
| `w2` | 所有专家的 down 权重，形状 `[num_experts, hidden_size, intermediate_size_per_tp]` |
| `gating_output` | 路由器输出 logits，形状 `[num_tokens, num_experts]` |
| `topk` | 每个 token 激活的专家数 |
| `renormalize` | 是否对 top-k 权重归一化 |
| `activation` | 激活函数名称（`"silu"` 或 `"gelu"`） |
| `apply_router_weight_on_input` | 是否在 gate_proj（而非 down_proj）处应用路由权重 |

---

### F.3.2 `moe/fused.py` — 融合 MoE 实现

**文件职责**：基于 Triton 自定义 CUDA 核的高性能 MoE 实现，包含 top-k softmax、block 对齐、融合专家 GEMM 等核心函数。

#### 函数：`fused_topk`

```python
# python/minisgl/moe/fused.py，第 9-28 行
def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int, renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

**功能**：对路由 logits 做 softmax 并取 top-k。

**返回值**：`(topk_weights, topk_ids)` 形状均为 `[M, topk]`，`weights` 为 float32，`ids` 为 int32。

**实现细节**：
- 调用 `sgl_kernel.topk_softmax` 完成一步融合的 softmax+topk
- 若 `renormalize=True`，对 topk_weights 按行归一化（除以行和加 1e-8 防止除零）
- 若提供 `num_token_non_padded`，将填充 token 的专家 id 设为 -1（告知后续核忽略这些 token）

#### 函数：`moe_align_block_size`

```python
# python/minisgl/moe/fused.py，第 31-89 行
def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

**功能**：将 token-专家分配关系重排为按专家分组、block 对齐的格式，以便 Triton grouped GEMM 核高效执行。

**返回值**：
- `sorted_token_ids`：按专家编号排序的 token id（附填充），形状 `[max_num_tokens_padded]`
- `expert_ids`：每个 M 方向 block 对应的专家编号，形状 `[max_num_m_blocks]`
- `num_tokens_post_padded`：标量，实际填充后的 token 数

**填充策略**：每个专家负责的 token 数向上取整到 `block_size` 的倍数，填充 token 使用越界索引（`= total_tokens`）标记，后续矩阵乘法核通过边界检查忽略这些位置。

#### 函数：`get_default_config` / `try_get_optimal_moe_config`

根据 `(M, E, N, K, topk)` 参数自适应选择 Triton kernel 的 tiling 参数：

```python
# python/minisgl/moe/fused.py，第 92-113 行
if M <= E:  # decode 阶段，token 数少
    config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}
else:       # prefill 阶段，token 数多
    config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}
```

#### 函数：`fused_experts_impl`

```python
# python/minisgl/moe/fused.py，第 127-227 行
def fused_experts_impl(
    hidden_states, w1, w2, topk_weights, topk_ids,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
```

**四阶段计算流程**：

1. **预分配缓存**：

```python
cache = torch.empty(M * topk * max(N, w2.shape[1]), ...)
intermediate_cache1 = cache[:M*topk*N].view(M, topk, N)         # [tokens, topk, 2*inter]
intermediate_cache2 = torch.empty(M*topk, N//2, ...)             # [tokens*topk, inter]
intermediate_cache3 = cache[:M*topk*w2.shape[1]].view(M, topk, hidden)  # [tokens, topk, hidden]
```

`cache1` 和 `cache3` 共用同一块内存（`cache`），因为它们不同时使用，此设计节省约一半的中间缓存显存。

2. **第一次 grouped GEMM（gate+up）**：

```python
fused_moe_kernel_triton(
    curr_hidden_states, w1, intermediate_cache1,
    curr_topk_weights, curr_topk_ids,
    sorted_token_ids, expert_ids, num_tokens_post_padded,
    apply_router_weight_on_input,  # 若 True 则在此处乘以路由权重
    topk_ids.shape[1], config, compute_type=compute_type,
)
```

结果存入 `intermediate_cache1`（`[M, topk, 2*inter_per_tp]`）。

3. **激活函数**：

```python
FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)
```

将形状展平后调用 `silu_and_mul`/`gelu_and_mul`，输出形状 `[M*topk, inter_per_tp]`。

4. **第二次 grouped GEMM（down）与加权归约**：

```python
fused_moe_kernel_triton(
    intermediate_cache2, w2, intermediate_cache3,
    curr_topk_weights, curr_topk_ids,
    ..., not apply_router_weight_on_input,  # 若未在第一步乘权重，则在此处乘
    1, config, ...
)
moe_sum_reduce_triton(intermediate_cache3, out_hidden_states[...])
```

将 `[M, topk, hidden]` 沿 `topk` 维度按路由权重加权求和，得到最终 `[M, hidden]`。

#### 类：`FusedMoe`

```python
# python/minisgl/moe/fused.py，第 230-256 行
class FusedMoe(BaseMoeBackend):
    def forward(self, ...) -> torch.Tensor:
        topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk, renormalize)
        return fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids, activation, apply_router_weight_on_input)
```

`BaseMoeBackend` 的具体实现，组合 `fused_topk` 和 `fused_experts_impl` 两步，是 mini-sglang 默认的 MoE 计算后端。

---

## 附录：模型架构对比表

| 特性 | Llama-3 | Qwen2 | Qwen3 | Qwen3-MoE |
|------|---------|-------|-------|-----------|
| QKV bias | 否 | 是 | 否 | 否 |
| QK-norm | 否 | 否 | 是 | 是 |
| MLP 类型 | GatedMLP (SiLU) | GatedMLP (SiLU) | GatedMLP (SiLU) | MoEMLP (SiLU) |
| RoPE 缩放 | Llama3 缩放 | 无/自定义 | 无/自定义 | 无/自定义 |
| Weight Tie | 可选 | 可选 | 可选 | 可选 |

## 附录：权重加载数据流

```
HuggingFace SafeTensors 文件
        |
        v (safetensors.safe_open，直接加载到目标设备)
  原始 state_dict
        |
        +-- tp_size > 1 --> _shard_state_dict()
        |                    +-- q/k/v/gate/up_proj: chunk(dim=0)[rank]
        |                    +-- o/down_proj:         chunk(dim=1)[rank]
        |                    +-- embed_tokens/lm_head: vocab 切片
        |
        v
  分片后 state_dict
        |
        v _merge_state_dict()
        +-- q+k+v   --> qkv_proj    (cat dim=0)
        +-- gate+up --> gate_up_proj (cat dim=0)
        |
        v
  最终 state_dict
        |
        v model.load_state_dict()
  模型参数就位
```
