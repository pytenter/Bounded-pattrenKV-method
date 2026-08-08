#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python}"
RESULT_DIR="${RESULT_DIR:-results/aime24_int2_wave1_v100_8gpu_wave1a4}"
RUN_DIR="${RUN_DIR:-run/aime24_int2_wave1_v100_8gpu_wave1a4}"
REFERENCE_DIR="$RESULT_DIR/fp16_reference_trajectories"
EXPECTED_REFERENCE_TASKS="${EXPECTED_REFERENCE_TASKS:-12}"
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"

mkdir -p "$RUN_DIR"

count_reference_records() {
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
base = Path("results/aime24_int2_wave1_v100_8gpu_wave1a4/fp16_reference_trajectories")
rows = 0
errors = 0
for path in base.glob("*/*.json"):
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        errors += 1
        continue
    if rec.get("max_new_tokens") == 32768 and rec.get("generated_token_ids"):
        rows += 1
    if rec.get("error") or rec.get("runtime_error"):
        errors += 1
print(f"{rows} {errors}")
PY
}

while true; do
  read -r records errors < <(count_reference_records)
  echo "[$(date -Iseconds)] fp16_reference_records=$records errors=$errors"
  if [[ "$errors" != "0" ]]; then
    echo "FP16 reference has errors; aborting formal Wave1A4." >&2
    exit 1
  fi
  if [[ "$records" -ge "$EXPECTED_REFERENCE_TASKS" ]]; then
    break
  fi
  sleep "$SLEEP_SECONDS"
done

"$PYTHON_BIN" scripts/run_wave1a4_attention_observer.py --mode fp16-manifest

CUDA_VISIBLE_DEVICES="${WAVE1A4_FP16_CAPTURE_GPU:-1}" "$PYTHON_BIN" scripts/run_wave1a4_attention_observer.py \
  --mode capture-fp16 \
  --clear-csvs

configs=(
  pattern_rolling_k2v2_s0_r128
  pattern_rolling_k2v2_s16_r128
  pattern_rolling_k2v2_s128_r128
  kivi_rolling_k2v2_s0_r128
  kivi_rolling_k2v2_s16_r128
  kivi_rolling_k2v2_s128_r128
)
gpus=(${WAVE1A4_TEACHER_GPUS:-1 2 3 4 5 6})
pids=()
for idx in "${!configs[@]}"; do
  config="${configs[$idx]}"
  gpu="${gpus[$idx]}"
  log="$RUN_DIR/wave1a4_teacher_${config}_gpu${gpu}.log"
  echo "[$(date -Iseconds)] teacher-config $config on logical GPU $gpu"
  (
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/run_wave1a4_attention_observer.py \
      --mode teacher-config \
      --config-name "$config"
  ) > "$log" 2>&1 &
  pids+=("$!")
  echo "${pids[-1]}" > "$RUN_DIR/wave1a4_teacher_${config}.pid"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != "0" ]]; then
  echo "At least one teacher-config worker failed; check $RUN_DIR/wave1a4_teacher_*.log" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/summarize_wave1a4_attention_mechanism.py
echo "[$(date -Iseconds)] Wave1A4 formal offline driver complete"
