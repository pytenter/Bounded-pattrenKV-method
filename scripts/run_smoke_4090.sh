#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
PROMPT_FILE="${PROMPT_FILE:-}"
DEVICE="${DEVICE:-cuda:0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATTERNKV_DEBUG_STATS="${PATTERNKV_DEBUG_STATS:-1}"

python scripts/run_smoke.py \
  --model-path "$MODEL_PATH" \
  --method fp16 \
  --device "$DEVICE" \
  --dtype float16 \
  --max-new-tokens 16 \
  ${PROMPT_FILE:+--prompt-file "$PROMPT_FILE"} \
  --output-json results/smoke_fp16.json 2>&1 | tee logs/smoke_fp16.log

python scripts/run_smoke.py \
  --model-path "$MODEL_PATH" \
  --method patternkv \
  --device "$DEVICE" \
  --dtype float16 \
  --k-bits 2 \
  --v-bits 2 \
  --group-size 128 \
  --residual-length 128 \
  --num-k-base 32 \
  --num-v-base 32 \
  --max-new-tokens 160 \
  ${PROMPT_FILE:+--prompt-file "$PROMPT_FILE"} \
  --output-json results/smoke_patternkv.json 2>&1 | tee logs/smoke_patternkv.log

