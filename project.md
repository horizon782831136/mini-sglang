# Mini-SGLang 项目结构说明

## 项目概述

Mini-SGLang 是一个轻量级但高性能的大语言模型（LLM）推理框架，是 [SGLang](https://github.com/sgl-project/sglang) 的精简实现。该项目旨在以约 **5,000 行 Python 代码**揭示现代 LLM 服务系统的核心设计，既可作为推理引擎使用，也可作为研究和学习参考。

### 核心特性

- **高性能推理**：达到业界领先的吞吐量和延迟
- **Radix Cache**：基于基数树的 KV 缓存前缀复用，共享前缀的请求可复用缓存
- **Chunked Prefill**：分块预填充，减少长输入的内存占用
- **Overlap Scheduling**：重叠调度，将 CPU 元数据处理与 GPU 计算并行执行
- **Tensor Parallelism**：张量并行，支持多 GPU 扩展
- **优化算子**：集成 FlashAttention、FlashInfer、TensorRT-LLM 等高性能注意力后端

---

## 目录结构

```
mini-sglang/
├── python/
│   └── minisgl/                    # 主代码包
│       ├── __init__.py
│       ├── __main__.py             # 入口点: python -m minisgl
│       ├── core.py                 # 核心数据类 (Req, Batch, SamplingParams, Context)
│       ├── shell.py                # 交互式 Shell 模式
│       ├── env.py                  # 环境变量配置
│       ├── attention/              # 注意力后端 (FlashAttention, FlashInfer, TRT-LLM)
│       ├── benchmark/              # 基准测试工具
│       ├── distributed/            # 张量并行 (all-reduce, all-gather, NCCL)
│       ├── engine/                 # 推理引擎 (前向传播, CUDA Graph, 采样)
│       ├── kernel/                 # 自定义 CUDA/Triton 算子
│       │   ├── triton/             # Triton 实现 (融合 MoE)
│       │   └── csrc/               # C++/CUDA 源码
│       ├── kvcache/                # KV 缓存管理 (池化, 前缀缓存)
│       ├── layers/                 # 神经网络层 (线性层, 归一化, 注意力, RoPE, MoE)
│       ├── llm/                    # 高层 LLM 接口
│       ├── message/                # 进程间通信消息类型 (ZMQ)
│       ├── models/                 # 模型实现 (Llama, Qwen2, Qwen3, Qwen3-MoE)
│       ├── moe/                    # Mixture of Experts 实现与调度
│       ├── scheduler/              # 请求调度 (预填充, 解码, 缓存管理)
│       ├── server/                 # API 服务器 (FastAPI, OpenAI 兼容)
│       ├── tokenizer/              # 分词/反分词
│       └── utils/                  # 工具函数 (日志, ZMQ 封装, 注册表)
├── benchmark/                      # 基准测试脚本
├── docs/                           # 文档
├── tests/                          # 单元测试
├── pyproject.toml                  # 项目配置
└── Dockerfile                      # Docker 构建
```

---

## 各模块详细说明

### 1. 核心数据模块 (`core.py`)

定义了整个推理框架的基础数据结构：

| 类名 | 说明 |
|------|------|
| `Req` | 单个推理请求，包含 input_ids、table_idx、cache_handle、采样参数等 |
| `Batch` | 一批请求的集合，标记当前阶段（`prefill` 或 `decode`） |
| `SamplingParams` | 采样参数（temperature、top_k、top_p、max_tokens） |
| `Context` | 全局推理上下文，持有模型、KV 缓存、注意力后端等引用 |

### 2. 推理引擎 (`engine/`)

推理执行的核心模块，负责模型的前向计算、采样和 CUDA Graph 管理。

| 组件 | 文件 | 说明 |
|------|------|------|
| `Engine` | `engine.py` | 主推理 Worker，管理模型、Context、KV 缓存、注意力后端和 CUDA Graph 重放 |
| `EngineConfig` | `config.py` | 引擎配置（模型路径、TP 度、页面大小等） |
| `ForwardOutput` | `engine.py` | 前向传播的输出（next_tokens_gpu 等） |
| `BatchSamplingArgs` | `engine.py` | 批量采样参数 |
| CudaGraphRunner | `graph.py` | CUDA Graph 捕获与重放，减少 kernel launch 开销 |
| Sampler | `sampler.py` | 采样器，支持 temperature、top-k、top-p 等采样策略 |

**关键流程**：`Engine.forward_batch()` 接收 Batch，执行模型前向传播，调用采样器生成 token。

### 3. 调度器 (`scheduler/`)

请求调度的核心模块，协调预填充和解码两个阶段的执行。

| 组件 | 文件 | 说明 |
|------|------|------|
| `Scheduler` | `scheduler.py` | 主调度器，运行在每个 TP Worker 进程中 |
| `SchedulerConfig` | `config.py` | 调度器配置 |
| `PrefillManager` | `prefill.py` | 预填充管理，处理新请求的输入 token 处理（支持分块） |
| `DecodeManager` | `decode.py` | 解码管理，管理已预填充请求的自回归生成 |
| `CacheManager` | `cache.py` | 缓存管理，分配/释放 KV 缓存页面 |
| `TableManager` | `table.py` | 表管理，管理 token pool 和 page table |
| `SchedulerIOMixin` | `io.py` | I/O 混入类，处理 ZMQ 消息收发和tokenizer/detokenizer通信 |

**调度策略**：
- **Overlap Scheduling**（默认）：将当前 batch 的 GPU 计算与上一 batch 的 CPU 后处理重叠执行，隐藏 CPU 延迟
- **Normal Scheduling**：非重叠模式，顺序执行

**两阶段调度**：
1. **Prefill 阶段**：处理新请求的输入 token，将 KV 值写入缓存
2. **Decode 阶段**：自回归生成新的 token，每次每个请求生成一个 token

### 4. 注意力后端 (`attention/`)

可插拔的注意力计算后端，通过注册表机制支持多种实现：

| 后端 | 文件 | 说明 |
|------|------|------|
| `BaseAttnBackend` | `base.py` | 注意力后端协议基类 |
| `FlashAttentionBackend` | `flashattention.py` | 基于 FlashAttention (v3/v4) 的实现 |
| `FlashInferBackend` | `flashinfer.py` | 基于 FlashInfer 的实现，在 decode 阶段性能优秀 |
| `TensorRTLLMBackend` | `trtllm.py` | 基于 TensorRT-LLM fmha 的实现 |
| `HybridBackend` | `hybrid.py` | 混合后端，prefill 和 decode 阶段分别使用不同后端 |

**设计模式**：使用 `Registry` 注册表模式，通过 `SUPPORTED_ATTN_BACKEND` 注册和查询后端实现。

### 5. KV 缓存 (`kvcache/`)

KV（Key-Value）缓存的管理模块，基于分页机制实现高效内存管理：

| 组件 | 文件 | 说明 |
|------|------|------|
| `BaseKVCachePool` | `base.py` | KV 缓存池抽象基类 |
| `BasePrefixCache` | `base.py` | 前缀缓存抽象基类 |
| `BaseCacheHandle` | `base.py` | 缓存句柄基类 |
| `MHAKVCache` | `mha_pool.py` | 多头注意力 KV 缓存池（基于分页） |
| `RadixPrefixCache` | `radix_cache.py` | 基于基数树的前缀缓存，支持共享前缀复用 |
| `NaivePrefixCache` | `naive_cache.py` | 简单前缀缓存，不共享前缀 |

**分页机制**：KV 缓存以固定大小的 Page 为单位管理，类似于操作系统的虚拟内存分页机制。每个 Page 存储固定数量的 token 的 KV 值。

**Radix Cache 核心思想**：将 token 序列的前缀组织为基数树，相同前缀的多个请求可以共享同一组 KV 缓存页面，避免重复计算。

### 6. 模型实现 (`models/`)

支持多种主流 LLM 架构的模型实现：

| 组件 | 文件 | 说明 |
|------|------|------|
| `BaseLLMModel` | `base.py` | 模型基类，定义前向传播接口 |
| `ModelConfig` | `config.py` | 模型配置（层数、隐藏维度、注意力头数等） |
| `RotaryConfig` | `config.py` | 旋转位置编码 (RoPE) 配置 |
| `LlamaModel` | `llama.py` | Llama-3 架构实现 |
| `Qwen2Model` | `qwen2.py` | Qwen2 架构实现 |
| `Qwen3Model` | `qwen3.py` | Qwen3 架构实现 |
| `Qwen3MoEModel` | `qwen3_moe.py` | Qwen3 MoE（混合专家）架构实现 |
| `get_model_class` | `register.py` | 模型工厂，根据架构名实例化对应模型 |
| `load_weight` | `weight.py` | 模型权重加载（支持 HuggingFace 格式） |

**设计模式**：使用 `SUPPORTED_MODELS` 注册表，模型通过装饰器 `@SUPPORTED_MODELS.register("Qwen2ForCausalLM")` 自动注册。

### 7. 神经网络层 (`layers/`)

构成模型的基础层实现，支持张量并行：

| 组件 | 文件 | 说明 |
|------|------|------|
| `LinearColParallelMerged` | `linear.py` | 列并行线性层（合并QKV投影） |
| `LinearRowParallel` | `linear.py` | 行并行线性层（O投影） |
| `LinearQKVMerged` | `linear.py` | QKV 合并投影层 |
| `VocabParallelEmbedding` | `linear.py` | 词表并行嵌入层 |
| `ParallelLMHead` | `linear.py` | 并行语言模型头 |
| `RMSNorm` / `RMSNormFused` | `norm.py` | RMS 归一化层 |
| `AttentionLayer` | `attention.py` | 注意力层 |
| `get_rope` / `set_rope_device` | `rope.py` | 旋转位置编码 (RoPE) |
| `silu_and_mul` / `gelu_and_mul` | `activation.py` | 激活函数 |
| `MoELayer` | `moe.py` | MoE（混合专家）层 |

### 8. 分布式 (`distributed/`)

张量并行通信原语，支持多 GPU 协同推理：

| 组件 | 文件 | 说明 |
|------|------|------|
| `PyNCCLCommunicator` | `pynccl.py` | NCCL 通信器封装 |
| `all_reduce` | `parallel.py` | All-Reduce 操作，用于行并行层的梯度聚合 |
| `all_gather` | `parallel.py` | All-Gather 操作，用于列并行层的输出拼接 |
| `broadcast` | `parallel.py` | Broadcast 操作，用于同步调度决策 |
| `DistributedInfo` | `parallel.py` | 分布式信息（TP rank、world size、设备信息） |

**通信架构**：Rank 0 作为主调度器，通过 NCCL Broadcast 将调度决策同步到所有 Rank。

### 9. 自定义算子 (`kernel/`)

针对关键操作的高性能 CUDA/Triton 算子：

| 算子 | 文件 | 说明 |
|------|------|------|
| `fused_moe_kernel_triton` | `triton/fused_moe.py` | 基于 Triton 的融合 MoE 算子 |
| `indexing` | C++/CUDA | 索引操作算子 |
| `radix` | C++/CUDA | 基数树键值比较算子 |
| `store_cache` | C++/CUDA | 缓存存储算子 |
| `pynccl` | C++/CUDA | NCCL Python 绑定 |

### 10. API 服务器 (`server/`)

提供 OpenAI 兼容的 HTTP API：

| 组件 | 文件 | 说明 |
|------|------|------|
| `launch_server` | `api_server.py` | 主入口，启动所有子进程 |
| FastAPI App | `api_server.py` | OpenAI 兼容的 API 端点 |
| `args.py` | `args.py` | CLI 参数解析 |

**启动流程**：`launch_server()` 启动 Tokenizer Worker、Detokenizer Worker、Scheduler Worker（每个 GPU 一个）和 HTTP 服务器。

### 11. 消息类型 (`message/`)

基于 ZMQ 的进程间通信消息定义：

| 消息类型 | 说明 |
|----------|------|
| `UserMsg` | 来自用户的推理请求 |
| `BatchBackendMsg` | 批量后端消息 |
| `UserReply` | 返回给用户的推理结果 |
| `BatchFrontendMsg` | 批量前端消息 |
| `TokenizeMsg` | 分词消息 |
| `DetokenizeMsg` | 反分词消息 |
| `AbortBackendMsg` | 中止请求消息 |
| `ExitMsg` | 退出消息 |

### 12. 工具函数 (`utils/`)

| 组件 | 文件 | 说明 |
|------|------|------|
| `Registry` | `registry.py` | 通用注册表，支持插件的注册和查询 |
| `ZmqQueue` / `ZmqAsyncQueue` | `zmq.py` | ZeroMQ 同步/异步队列封装 |
| `init_logger` | `logger.py` | 日志初始化 |
| `load_tokenizer` | `hf.py` | HuggingFace 分词器加载 |
| `download_hf_weight` | `hf.py` | HuggingFace 权重下载 |
| `is_sm90_supported` / `is_sm100_supported` | `gpu.py` | GPU 架构特性检测 |

### 13. LLM 接口 (`llm/`)

提供高层次 Python API：

```python
from minisgl import LLM

llm = LLM(model="Qwen/Qwen3-0.6B")
result = llm.generate("Hello, world!")
```

| 组件 | 文件 | 说明 |
|------|------|------|
| `LLM` | `llm.py` | 高层 LLM 封装，在子进程中启动服务器并通过 ZMQ 通信 |

### 14. 交互式 Shell (`shell.py`)

提供命令行交互式对话模式，用户可直接在终端与模型对话。

### 15. 环境变量 (`env.py`)

通过环境变量控制运行时行为，如 `DISABLE_OVERLAP_SCHEDULING`、`OVERLAP_EXTRA_SYNC` 等。

### 16. MoE 模块 (`moe/`)

| 组件 | 文件 | 说明 |
|------|------|------|
| MoE Dispatcher | `dispatcher.py` | 专家分发逻辑（token 到专家的路由） |
| MoE 调度策略 | 多文件 | 支持不同的专家调度和负载均衡策略 |

### 17. Tokenizer (`tokenizer/`)

| 组件 | 文件 | 说明|
|------|------|------|
| Tokenizer Worker | `tokenizer.py` | 独立分词进程，通过 ZMQ 与调度器通信|
| Detokenizer Worker | `detokenizer.py` | 独立反分词进程，将 token ID 转回文本|

---

## 模块间数据流

```
用户请求
    │
    ▼
┌─────────────────┐     ┌────────────────┐
│  API Server     │────▶│  Tokenizer     │ (ZMQ)
│  (FastAPI)      │     │  Worker        │
└────────┬────────┘     └───────┬────────┘
         │                      │
         │                      ▼
         │              ┌────────────────┐
         │              │  Scheduler 0   │ (Rank 0, ZMQ + NCCL Broadcast)
         │              │  (主调度器)     │
         │              └────────┬───────┘
         │                       │ NCCL Broadcast
         │           ┌───────────┼───────────┐
         │           ▼           ▼           ▼
         │    ┌──────────┐ ┌──────────┐ ┌──────────┐
         │    │Scheduler 1│ │Scheduler 2│ │Scheduler N│  (每个 GPU 一个)
         │    └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
         │          │             │             │
         │          ▼             ▼             ▼
         │    ┌─────────────────────────────────────┐
         │    │         Engine (每个 GPU)             │
         │    │  ┌────────────┐  ┌───────────────┐  │
         │    │  │   Model    │──│   Attention   │  │
         │    │  │  (Layers)  │  │   Backend     │  │
         │    │  └────────────┘  └───────────────┘  │
         │    │        │                │            │
         │    │        ▼                ▼            │
         │    │  ┌──────────────────────────────┐   │
         │    │  │       KV-Cache Pool          │   │
         │    │  │  ┌────────────────────────┐  │   │
         │    │  │  │   Radix Prefix Cache   │  │   │
         │    │  │  └────────────────────────┘  │   │
         │    │  └──────────────────────────────┘   │
         │    └──────────────────────────────────────┘
         │                       │
         │                       ▼
         │              ┌────────────────┐
         │              │  Detokenizer   │ (ZMQ)
         │              │  Worker        │
         │              └───────┬────────┘
         │                      │
         └──────────────────────┘
                    │
                    ▼
              用户响应
```

---

## 核心设计模式

### 1. 注册表模式 (Registry Pattern)

广泛用于插件系统，允许在不修改代码的情况下切换实现：
- 注意力后端：`SUPPORTED_ATTN_BACKEND`
- 缓存管理器：`SUPPORTED_CACHE_MANAGER`
- MoE 后端：`SUPPORTED_MOE_BACKEND`
- 模型架构：`SUPPORTED_MODELS`

### 2. 工厂模式 (Factory Pattern)

通过工厂函数创建具体实现：
- `create_model()` → 根据架构名创建模型
- `create_attention_backend()` → 根据名称创建注意力后端
- `create_prefix_cache()` → 根据类型创建前缀缓存
- `create_kvcache_pool()` → 创建 KV 缓存池

### 3. 数据类驱动 (Dataclass-based State)

广泛使用 Python dataclass 管理请求状态和配置：
- `Req`、`Batch`、`SamplingParams` 管理请求生命周期
- `ModelConfig`、`SchedulerConfig`、`EngineConfig` 管理系统配置

### 4. 多进程架构 (Process-based Distribution)

基于多进程的分布式架构：
- **ZMQ**：用于控制消息的进程间通信
- **NCCL**：用于 GPU 间的张量通信（all-reduce、broadcast 等）
- 各进程职责分离：API Server、Tokenizer、Scheduler、Detokenizer 各自独立运行

### 5. 分页 KV 缓存 (Page-based KV Cache)

类似操作系统的虚拟内存分页机制：
- KV 缓存以固定大小的 Page 为单位分配
- 支持 Page 的动态分配和回收
- Radix Cache 实现 Page 的共享和复用

### 6. 两阶段调度 (Two-Phase Scheduling)

- **Prefill 阶段**：处理输入 token，计算并缓存 KV 值
- **Decode 阶段**：自回归生成，每次每个请求生成一个 token
- 两个阶段有不同的优化策略（如 Chunked Prefill、CUDA Graph 等）

---

## 入口方式

### CLI 方式
```bash
python -m minisgl --model <model_name> [--tp N] [--port N] [--shell]
```

### Python API 方式
```python
from minisgl import LLM
llm = LLM(model="Qwen/Qwen3-0.6B")
result = llm.generate("Hello, world!")
```

### Docker 方式
```bash
docker run minisgl --model Qwen/Qwen3-0.6B
```
