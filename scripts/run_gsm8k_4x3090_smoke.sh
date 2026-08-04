#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local Llama-3.1-8B-Instruct directory}"
GSM8K_DATA_PATH="${GSM8K_DATA_PATH:-data/gsm8k/test.jsonl}"
GPU_IDS=(4 5 6 7)
NUM_SHARDS=4
NUM_SAMPLES=50
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
GPU_MEMORY_BUSY_MIB="${GPU_MEMORY_BUSY_MIB:-2000}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-8}"

mkdir -p logs/gsm8k/smoke run/gsm8k/smoke reports results/gsm8k/smoke

precheck() {
  echo "[precheck] GPUs 4-7"
  nvidia-smi --id=4,5,6,7 --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  "$PYTHON_BIN" - <<PY
import json
import subprocess
import sys
from pathlib import Path
model = Path("$MODEL_PATH")
data = Path("$GSM8K_DATA_PATH")
required = ["config.json", "tokenizer_config.json", "model.safetensors.index.json"]
missing = [str(model / x) for x in required if not (model / x).exists()]
if missing:
    raise SystemExit("missing model files: " + json.dumps(missing))
if not data.exists():
    raise SystemExit(f"missing GSM8K data: {data}; run scripts/prepare_gsm8k_modelscope.py")
rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 1319:
    raise SystemExit(f"expected 1319 GSM8K rows, got {len(rows)}")
out = subprocess.check_output(["nvidia-smi", "--id=4,5,6,7", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
busy = []
for line in out.strip().splitlines():
    idx, mem = [x.strip() for x in line.split(",")]
    if int(mem) > int("$GPU_MEMORY_BUSY_MIB"):
        busy.append((idx, mem))
if busy:
    raise SystemExit(f"GPU memory above threshold MiB: {busy}")
print("precheck ok")
PY
  "$PYTHON_BIN" -m pytest tests/test_gsm8k_parser.py -q
}

launch_method() {
  local method="$1"
  mkdir -p "logs/gsm8k/smoke/${method}" "run/gsm8k/smoke/${method}" "results/gsm8k/smoke/${method}"
  local pids=()
  for shard in 0 1 2 3; do
    local gpu="${GPU_IDS[$shard]}"
    local log="logs/gsm8k/smoke/${method}/gpu${gpu}_shard${shard}.log"
    local pidfile="run/gsm8k/smoke/${method}/gpu${gpu}_shard${shard}.pid"
    local metafile="run/gsm8k/smoke/${method}/gpu${gpu}_shard${shard}.meta"
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
  "$PYTHON_BIN" scripts/summarize_gsm8k.py --mode smoke --expected-samples "$NUM_SAMPLES" --methods "$method" --require-complete
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
summary = json.loads(Path("reports/gsm8k_smoke_3methods_50.json").read_text(encoding="utf-8"))
truncated = sum(m["length_truncated"] for m in summary["methods"])
if truncated:
    raise SystemExit("${method} smoke has %d length_truncated rows; archive current smoke output and rerun with MAX_NEW_TOKENS=1024 before continuing." % truncated)
PY
  return "$rc"
}

precheck
for method in fp16 kivi patternkv; do
  launch_method "$method"
done
"$PYTHON_BIN" scripts/summarize_gsm8k.py --mode smoke --expected-samples "$NUM_SAMPLES" --methods fp16 kivi patternkv --require-complete
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path("reports/gsm8k_smoke_3methods_50.json").read_text(encoding="utf-8"))
truncated = sum(m["length_truncated"] for m in summary["methods"])
if truncated:
    raise SystemExit(f"Smoke has {truncated} length_truncated rows; rerun smoke with MAX_NEW_TOKENS=1024 before full.")
print("GSM8K smoke PASS")
PY
