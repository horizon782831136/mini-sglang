<p align="center">
<img width="400" src="/assets/logo.png">
</p>

# Mini-SGLang

A **lightweight yet high-performance** inference framework for Large Language Models.

---

Mini-SGLang is a compact implementation of [SGLang](https://github.com/sgl-project/sglang), designed to demystify the complexities of modern LLM serving systems. With a compact codebase of **~5,000 lines of Python**, it serves as both a capable inference engine and a transparent reference for researchers and developers.

## ✨ Key Features

- **High Performance**: Achieves state-of-the-art throughput and latency with advanced optimizations.
- **Lightweight & Readable**: A clean, modular, and fully type-annotated codebase that is easy to understand and modify.
- **Advanced Optimizations**:
  - **Radix Cache**: Reuses KV cache for shared prefixes across requests.
  - **Chunked Prefill**: Reduces peak memory usage for long-context serving.
  - **Overlap Scheduling**: Hides CPU scheduling overhead with GPU computation.
  - **Tensor Parallelism**: Scales inference across multiple GPUs.
  - **Optimized Kernels**: Integrates **FlashAttention** and **FlashInfer** for maximum efficiency.
  - ...

## 🚀 Quick Start

> **⚠️ Platform Support**: Mini-SGLang currently supports **Linux only** (x86_64 and aarch64). Windows and macOS are not supported due to dependencies on Linux-specific CUDA kernels (`sgl-kernel`, `flashinfer`). We recommend using [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) on Windows or Docker for cross-platform compatibility.

### 1. Environment Setup

We recommend using `uv` for a fast and reliable installation (note that `uv` does not conflict with `conda`).

```bash
# Create a virtual environment (Python 3.10+ recommended)
uv venv --python=3.12
source .venv/bin/activate
```

**Prerequisites**: Mini-SGLang relies on CUDA kernels that are JIT-compiled. Ensure you have the **NVIDIA CUDA Toolkit** installed and that its version matches your driver's version. You can check your driver's CUDA capability with `nvidia-smi`.

### 2. Installation

Install Mini-SGLang directly from the source:

```bash
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
```

<details>
<summary><b>💡 Installing on Windows (WSL2)</b></summary>

Since Mini-SGLang requires Linux-specific dependencies, Windows users should use WSL2:

1. **Install WSL2** (if not already installed):
   ```powershell
   # In PowerShell (as Administrator)
   wsl --install
   ```

2. **Install CUDA on WSL2**:
   - Follow [NVIDIA's WSL2 CUDA guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   - Ensure your Windows GPU drivers support WSL2

3. **Install Mini-SGLang in WSL2**:
   ```bash
   # Inside WSL2 terminal
   git clone https://github.com/sgl-project/mini-sglang.git
   cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
   uv pip install -e .
   ```

4. **Access from Windows**: The server will be accessible at `http://localhost:8000` from Windows browsers and applications.

</details>

<details>
<summary><b>🐳 Running with Docker</b></summary>

**Prerequisites**:
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. **Build the Docker image**:
   ```bash
   docker build -t minisgl .
   ```

2. **Run the server**:
   ```bash
   docker run --gpus all -p 1919:1919 \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

3. **Run in interactive shell mode**:
   ```bash
   docker run -it --gpus all \
       minisgl --model Qwen/Qwen3-0.6B --shell
   ```

4. **Using Docker Volumes for persistent caches** (recommended for faster subsequent startups):
   ```bash
   docker run --gpus all -p 1919:1919 \
       -v huggingface_cache:/app/.cache/huggingface \
       -v tvm_cache:/app/.cache/tvm-ffi \
       -v flashinfer_cache:/app/.cache/flashinfer \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

</details>

### 3. Online Serving

Launch an OpenAI-compatible API server with a single command.

```bash
# Deploy Qwen/Qwen3-0.6B on a single GPU
python -m minisgl --model "Qwen/Qwen3-0.6B"

# Deploy meta-llama/Llama-3.1-70B-Instruct on 4 GPUs with Tensor Parallelism, on port 30000
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000
```

Once the server is running, you can send requests using standard tools like `curl` or any OpenAI-compatible client.

### 4. Interactive Shell

Chat with your model directly in the terminal by adding the `--shell` flag.

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

![shell-example](https://lmsys.org/images/blog/minisgl/shell.png)

You can also use `/reset` to clear the chat history.

## Benchmark

### Offline inference

See [bench.py](./benchmark/offline/bench.py) for more details. Set `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` for ablation study on overlap scheduling.

Test Configuration:

- Hardware: 1xH200 GPU.
- Model: Qwen3-0.6B, Qwen3-14B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100-1024 tokens
- Output Length: Randomly sampled between 100-1024 tokens

![offline](https://lmsys.org/images/blog/minisgl/offline.png)

### Online inference

See [benchmark_qwen.py](./benchmark/online/bench_qwen.py) for more details.

Test Configuration:

- Hardware: 4xH200 GPU, connected by NVLink.
- Model: Qwen3-32B
- Dataset: [Qwen trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_traceA_blksz_16.jsonl), replaying first 1000 requests.

Launch command:

```bash
# Mini-SGLang
python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --cache naive

# SGLang
python3 -m sglang.launch_server --model "Qwen/Qwen3-32B" --tp 4 \
    --disable-radix --port 1919 --decode-attention flashinfer
```

> **Note**: If you encounter network issues when downloading models from HuggingFace, try using `--model-source modelscope` to download from ModelScope instead:
> ```bash
> python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --model-source modelscope
> ```

![online](https://lmsys.org/images/blog/minisgl/online.png)

## 📚 Learn More

- **[Detailed Features](./docs/features.md)**: Explore all available features and command-line arguments.
- **[System Architecture](./docs/structures.md)**: Dive deep into the design and data flow of Mini-SGLang.

---

## 项目结构说明

### 目录结构

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

### 各模块详细说明

#### 1. 核心数据模块 (`core.py`)

| 类名 | 说明 |
|------|------|
| `Req` | 单个推理请求，包含 input_ids、table_idx、cache_handle、采样参数等 |
| `Batch` | 一批请求的集合，标记当前阶段（`prefill` 或 `decode`） |
| `SamplingParams` | 采样参数（temperature、top_k、top_p、max_tokens） |
| `Context` | 全局推理上下文，持有模型、KV 缓存、注意力后端等引用 |

#### 2. 推理引擎 (`engine/`)

| 组件 | 文件 | 说明 |
|------|------|------|
| `Engine` | `engine.py` | 主推理 Worker，管理模型、Context、KV 缓存、注意力后端和 CUDA Graph 重放 |
| `EngineConfig` | `config.py` | 引擎配置（模型路径、TP 度、页面大小等） |
| `CudaGraphRunner` | `graph.py` | CUDA Graph 捕获与重放，减少 kernel launch 开销 |
| `Sampler` | `sampler.py` | 采样器，支持 temperature、top-k、top-p 等采样策略 |

#### 3. 调度器 (`scheduler/`)

| 组件 | 文件 | 说明 |
|------|------|------|
| `Scheduler` | `scheduler.py` | 主调度器，运行在每个 TP Worker 进程中 |
| `PrefillManager` | `prefill.py` | 预填充管理，处理新请求的输入 token（支持分块） |
| `DecodeManager` | `decode.py` | 解码管理，管理已预填充请求的自回归生成 |
| `CacheManager` | `cache.py` | 缓存管理，分配/释放 KV 缓存页面 |

**调度策略**：默认使用 Overlap Scheduling，将当前 batch 的 GPU 计算与上一 batch 的 CPU 后处理重叠执行。

#### 4. 注意力后端 (`attention/`)

| 后端 | 说明 |
|------|------|
| `FlashAttentionBackend` | 基于 FlashAttention (v3/v4) 的实现 |
| `FlashInferBackend` | 基于 FlashInfer 的实现，decode 阶段性能优秀 |
| `TensorRTLLMBackend` | 基于 TensorRT-LLM fmha 的实现 |
| `HybridBackend` | 混合后端，prefill 和 decode 阶段分别使用不同后端 |

#### 5. KV 缓存 (`kvcache/`)

| 组件 | 说明 |
|------|------|
| `MHAKVCache` | 多头注意力 KV 缓存池（基于分页） |
| `RadixPrefixCache` | 基于基数树的前缀缓存，支持共享前缀复用 |
| `NaivePrefixCache` | 简单前缀缓存，不共享前缀 |

#### 6. 模型实现 (`models/`)

| 组件 | 说明 |
|------|------|
| `LlamaModel` | Llama-3 架构实现 |
| `Qwen2Model` | Qwen2 架构实现 |
| `Qwen3Model` | Qwen3 架构实现 |
| `Qwen3MoEModel` | Qwen3 MoE（混合专家）架构实现 |

### 模块间数据流

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
         │              │  Scheduler 0   │ (Rank 0)
         │              └────────┬───────┘
         │                       │ NCCL Broadcast
         │           ┌───────────┼───────────┐
         │           ▼           ▼           ▼
         │    ┌──────────┐ ┌──────────┐ ┌──────────┐
         │    │Scheduler 1│ │Scheduler 2│ │Scheduler N│
         │    └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
         │          └─────────────┼─────────────┘
         │                        ▼
         │    ┌─────────────────────────────────────┐
         │    │  Engine (每个 GPU)                   │
         │    │  Model (Layers) + Attention Backend  │
         │    │  KV-Cache Pool + Radix Prefix Cache  │
         │    └──────────────────┬──────────────────┘
         │                       ▼
         │              ┌────────────────┐
         │              │  Detokenizer   │ (ZMQ)
         │              └───────┬────────┘
         └──────────────────────┘
                    │
                    ▼
              用户响应
```

### 核心设计模式

- **注册表模式**：注意力后端、缓存管理器、模型架构均通过 `Registry` 注册，支持灵活切换实现
- **工厂模式**：`create_model()`、`create_attention_backend()` 等工厂函数按需创建具体实现
- **多进程架构**：API Server、Tokenizer、Scheduler、Detokenizer 各自独立进程，ZMQ 传递控制消息，NCCL 负责 GPU 间张量通信
- **分页 KV 缓存**：类似操作系统虚拟内存分页，KV 缓存以固定大小 Page 为单位分配与回收
- **两阶段调度**：Prefill 阶段处理输入 token，Decode 阶段自回归生成，各有独立优化策略
