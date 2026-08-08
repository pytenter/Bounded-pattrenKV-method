#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B}"
LOG_DIR="${LOG_DIR:-run/aime24_pseudodecode_3090_8gpu}"
GPUS="${GPUS:-0,1,2,3,4,5}"

mkdir -p "$LOG_DIR"

export PYTHONPATH="$ROOT/quant:$ROOT:${PYTHONPATH:-}"

"$PYTHON_BIN" scripts/run_aime24_pseudodecode_formal.py precompute-fp16 --model-path "$MODEL_PATH"
"$PYTHON_BIN" scripts/run_aime24_pseudodecode_formal.py launch --model-path "$MODEL_PATH" --gpus "$GPUS"
"$PYTHON_BIN" scripts/summarize_aime24_pseudodecode.py
