#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local Llama-3.1-8B-Instruct directory}"
GSM8K_DATA_PATH="${GSM8K_DATA_PATH:-data/gsm8k/test.jsonl}"
GPU_IDS=(4 5 6 7)
NUM_SHARDS=4
NUM_SAMPLES=1319
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-8}"

mkdir -p logs/gsm8k/full run/gsm8k/full reports results/gsm8k/full

if [[ ! -f reports/gsm8k_smoke_3methods_50.json ]]; then
  echo "Missing smoke report. Run scripts/run_gsm8k_4x3090_smoke.sh first." >&2
  exit 1
fi
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
s = json.loads(Path("reports/gsm8k_smoke_3methods_50.json").read_text(encoding="utf-8"))
if not s.get("pass"):
    raise SystemExit("Smoke report is not PASS; refusing full run.")
if sum(m["length_truncated"] for m in s["methods"]):
    raise SystemExit("Smoke contains length_truncated rows; rerun smoke with MAX_NEW_TOKENS=1024 before full.")
PY

launch_method() {
  local method="$1"
  mkdir -p "logs/gsm8k/full/${method}" "run/gsm8k/full/${method}" "results/gsm8k/full/${method}"
  local pids=()
  for shard in 0 1 2 3; do
    local gpu="${GPU_IDS[$shard]}"
    local log="logs/gsm8k/full/${method}/gpu${gpu}_shard${shard}.log"
    local pidfile="run/gsm8k/full/${method}/gpu${gpu}_shard${shard}.pid"
    local metafile="run/gsm8k/full/${method}/gpu${gpu}_shard${shard}.meta"
    printf 'gpu=%s\nmethod=%s\nmode=full\nshard=%s\nnum_samples=%s\nmax_new_tokens=%s\nlog=%s\nstarted_at=%s\n' \
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
      --mode full \
      --skip-existing > "$log" 2>&1 &
    echo $! > "$pidfile"
    pids+=("$!")
    echo "started method=${method} gpu=${gpu} shard=${shard} pid=$! log=${log}"
    sleep "$LAUNCH_STAGGER_SECONDS"
  done
  local rc=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  "$PYTHON_BIN" scripts/summarize_gsm8k.py --mode full --expected-samples "$NUM_SAMPLES" --methods "$method" --require-complete
  return "$rc"
}

for method in fp16 kivi patternkv; do
  launch_method "$method"
done
"$PYTHON_BIN" scripts/summarize_gsm8k.py \
  --mode full \
  --expected-samples "$NUM_SAMPLES" \
  --methods fp16 kivi patternkv \
  --report-csv reports/gsm8k_3methods_full.csv \
  --final-status-md reports/gsm8k_final_status.md \
  --require-complete
echo "GSM8K full PASS"
