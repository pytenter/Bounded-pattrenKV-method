#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
GSM8K_DATA_PATH="${GSM8K_DATA_PATH:-data/gsm8k/test.jsonl}"
MAX_NEW_TOKENS=1024
NUM_SAMPLES=50
NUM_SHARDS=4
GPU_IDS=(4 5 6 7)
BUSY_MIB="${GPU_MEMORY_BUSY_MIB:-2000}"

mkdir -p logs/gsm8k/smoke_1024 run/gsm8k/smoke_1024 results/gsm8k/smoke_1024 reports

precheck() {
  nvidia-smi -i 4,5,6,7
  "$PYTHON_BIN" - <<PY
import subprocess
busy = []
out = subprocess.check_output(["nvidia-smi", "--id=4,5,6,7", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
for line in out.strip().splitlines():
    idx, mem = [x.strip() for x in line.split(",")]
    if int(mem) > int("$BUSY_MIB"):
        busy.append((idx, int(mem)))
if busy:
    raise SystemExit(f"GPU 4-7 busy above threshold MiB, not launching: {busy}")
PY
}

launch_method() {
  local method="$1"
  mkdir -p "logs/gsm8k/smoke_1024/${method}" "run/gsm8k/smoke_1024/${method}" "results/gsm8k/smoke_1024/${method}"
  local pids=()
  for shard in 0 1 2 3; do
    local gpu="${GPU_IDS[$shard]}"
    local log="logs/gsm8k/smoke_1024/${method}/gpu${gpu}_shard${shard}.log"
    local pidfile="run/gsm8k/smoke_1024/${method}/gpu${gpu}_shard${shard}.pid"
    local metafile="run/gsm8k/smoke_1024/${method}/gpu${gpu}_shard${shard}.meta"
    printf 'gpu=%s\nmethod=%s\nmode=smoke\nshard=%s\nnum_samples=%s\nmax_new_tokens=%s\nlog=%s\nstarted_at=%s\n' \
      "$gpu" "$method" "$shard" "$NUM_SAMPLES" "$MAX_NEW_TOKENS" "$log" "$(date --iso-8601=seconds)" > "$metafile"
    env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" bench/bench_gsm8k_patternkv.py \
      --method "$method" \
      --model-path "$MODEL_PATH" \
      --data-path "$GSM8K_DATA_PATH" \
      --num-samples "$NUM_SAMPLES" \
      --num-shards "$NUM_SHARDS" \
      --shard-id "$shard" \
      --physical-gpu-id "$gpu" \
      --mode smoke \
      --dtype float16 \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --output-dir results/gsm8k/smoke_1024 \
      --status-dir run/gsm8k/smoke_1024 \
      --skip-existing > "$log" 2>&1 &
    echo $! > "$pidfile"
    pids+=("$!")
    echo "started ${method} shard=${shard} gpu=${gpu} pid=$!"
    sleep 8
  done
  local rc=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  "$PYTHON_BIN" scripts/summarize_gsm8k.py --mode smoke --results-dir results/gsm8k/smoke_1024 --expected-samples "$NUM_SAMPLES" --methods "$method" || rc=1
  return "$rc"
}

precheck
"$PYTHON_BIN" -m pytest tests/test_gsm8k_parser.py tests/test_gsm8k_stop_reason.py -q
for method in fp16 kivi patternkv; do
  launch_method "$method"
done
"$PYTHON_BIN" scripts/summarize_gsm8k.py --mode smoke --results-dir results/gsm8k/smoke_1024 --expected-samples "$NUM_SAMPLES" --methods fp16 kivi patternkv --report-md reports/gsm8k_smoke_1024.md --report-json reports/gsm8k_smoke_1024.json || true
