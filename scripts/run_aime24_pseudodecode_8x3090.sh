#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/blockgtq-repro/envs/blockgtq/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B}"
LOG_DIR="${LOG_DIR:-run/aime24_pseudodecode_3090_8gpu}"

mkdir -p "$LOG_DIR"

"$PYTHON_BIN" scripts/prepare_aime24_pseudodecode_3090.py --model-path "$MODEL_PATH" --python-bin "$PYTHON_BIN"

echo "Formal pseudo/static workers are gated. Complete worker implementations and preflight gates before launching long-run jobs."
echo "Reference generation can be launched independently with scripts/generate_aime24_pseudodecode_references.py."
