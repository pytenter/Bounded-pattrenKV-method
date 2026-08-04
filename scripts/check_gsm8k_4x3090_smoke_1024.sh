#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
nvidia-smi --id=4,5,6,7 --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
"$PYTHON_BIN" scripts/summarize_gsm8k.py --mode smoke --results-dir results/gsm8k/smoke_1024 --expected-samples 50 --methods fp16 kivi patternkv || true
