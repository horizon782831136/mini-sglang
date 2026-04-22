#!/bin/bash
python -m minisgl --model-path /mnt/c/Users/Admin/liuwei/models/Qwen3-0.6B --attn fi --cuda-graph-max-bs 1 --max-running-requests 1 --memory-ratio 0.7 --host localhost --port 8000 --max-prefill-length 8192
