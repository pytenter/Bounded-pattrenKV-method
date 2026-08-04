#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
GSM8K_DATA_PATH="${GSM8K_DATA_PATH:-data/gsm8k/test.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
NUM_SAMPLES=50
NUM_SHARDS=4

mkdir -p logs/gsm8k/smoke run/gsm8k/smoke results/gsm8k/smoke

run_worker() {
  local method="$1"
  local shard="$2"
  local gpu="$3"
  local log="logs/gsm8k/smoke/${method}/gpu${gpu}_shard${shard}.log"
  local pidfile="run/gsm8k/smoke/${method}/gpu${gpu}_shard${shard}.pid"
  local metafile="run/gsm8k/smoke/${method}/gpu${gpu}_shard${shard}.meta"
  mkdir -p "logs/gsm8k/smoke/${method}" "run/gsm8k/smoke/${method}" "results/gsm8k/smoke/${method}"
  printf 'gpu=%s\nmethod=%s\nmode=smoke\nshard=%s\nnum_samples=%s\nmax_new_tokens=%s\nlog=%s\nstarted_at=%s\n' \
    "$gpu" "$method" "$shard" "$NUM_SAMPLES" "$MAX_NEW_TOKENS" "$log" "$(date --iso-8601=seconds)" > "$metafile"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" bench/bench_gsm8k_patternkv.py \
    --method "$method" \
    --model-path "$MODEL_PATH" \
    --data-path "$GSM8K_DATA_PATH" \
    --num-samples "$NUM_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --seed 0 \
    --shard-id "$shard" \
    --num-shards "$NUM_SHARDS" \
    --physical-gpu-id "$gpu" \
    --mode smoke \
    --skip-existing > "$log" 2>&1 &
  echo $! > "$pidfile"
  echo "started method=${method} shard=${shard} gpu=${gpu} pid=$! log=${log}"
}

wait_for_pids() {
  local rc=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  return "$rc"
}

echo "[$(date --iso-8601=seconds)] continuing GSM8K smoke on idle GPUs"

kivi_pids=()
if [[ -s run/gsm8k/smoke/kivi/gpu3_shard0.pid ]] && kill -0 "$(cat run/gsm8k/smoke/kivi/gpu3_shard0.pid)" 2>/dev/null; then
  kivi_pids+=("$(cat run/gsm8k/smoke/kivi/gpu3_shard0.pid)")
elif pgrep -f "bench_gsm8k_patternkv.py --method kivi .*--shard-id 0 .*--physical-gpu-id 3" >/dev/null; then
  kivi_pids+=("$(pgrep -f "bench_gsm8k_patternkv.py --method kivi .*--shard-id 0 .*--physical-gpu-id 3" | head -1)")
fi

run_worker kivi 1 1
kivi_pids+=("$!")
run_worker kivi 2 5
kivi_pids+=("$!")
run_worker kivi 3 7
kivi_pids+=("$!")

wait_for_pids "${kivi_pids[@]}" || true
"$PYTHON_BIN" scripts/summarize_gsm8k.py --mode smoke --expected-samples "$NUM_SAMPLES" --methods kivi || true

run_worker patternkv 0 1
p0="$!"
run_worker patternkv 1 3
p1="$!"
run_worker patternkv 2 5
p2="$!"
run_worker patternkv 3 7
p3="$!"
wait_for_pids "$p0" "$p1" "$p2" "$p3" || true

"$PYTHON_BIN" scripts/summarize_gsm8k.py --mode smoke --expected-samples "$NUM_SAMPLES" --methods fp16 kivi patternkv || true
echo "[$(date --iso-8601=seconds)] idle GPU smoke continuation finished"
