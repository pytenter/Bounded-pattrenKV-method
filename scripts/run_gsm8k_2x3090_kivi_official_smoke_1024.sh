#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
DATA_PATH="${DATA_PATH:-data/gsm8k/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/gsm8k/kivi_official_smoke_1024_g32r32}"
STATUS_DIR="${STATUS_DIR:-run/gsm8k/kivi_official_smoke_1024_g32r32}"
LOG_DIR="${LOG_DIR:-logs/gsm8k/kivi_official_smoke_1024_g32r32}"
GPU_A="${GPU_A:-4}"
GPU_B="${GPU_B:-5}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
GROUP_SIZE="${GROUP_SIZE:-32}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-32}"

mkdir -p "$LOG_DIR" "$STATUS_DIR"

run_shard() {
  local shard="$1"
  local gpu="$2"
  local log="$LOG_DIR/gpu${gpu}_official_kivi_shard${shard}of2.log"
  local pidfile="$STATUS_DIR/gpu${gpu}_official_kivi_shard${shard}of2.pid"

  env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" bench/bench_gsm8k_patternkv.py \
    --method kivi_official \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --num-samples "$NUM_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --seed 0 \
    --shard-id "$shard" \
    --num-shards 2 \
    --output-dir "$OUTPUT_DIR" \
    --status-dir "$STATUS_DIR" \
    --skip-existing \
    --dtype float16 \
    --k-bits 2 \
    --v-bits 2 \
    --group-size "$GROUP_SIZE" \
    --residual-length "$RESIDUAL_LENGTH" \
    --physical-gpu-id "$gpu" \
    --mode smoke > "$log" 2>&1 &

  echo "$!" > "$pidfile"
  echo "started shard=$shard gpu=$gpu pid=$(cat "$pidfile") log=$log"
}

run_shard 0 "$GPU_A"
sleep "${LAUNCH_STAGGER_SECONDS:-15}"
run_shard 1 "$GPU_B"

wait

"$PYTHON_BIN" scripts/summarize_gsm8k.py \
  --mode smoke \
  --results-dir "$OUTPUT_DIR" \
  --expected-samples "$NUM_SAMPLES" \
  --num-shards 2 \
  --methods kivi_official \
  --report-md reports/gsm8k_kivi_official_smoke_1024_g32r32.md \
  --report-json reports/gsm8k_kivi_official_smoke_1024_g32r32.json

