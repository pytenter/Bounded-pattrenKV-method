#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
DATA_DIR="${LONGBENCH_DATA_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper_repro_v2/longbench}"
STATUS_DIR="${STATUS_DIR:-run/paper_repro_v2/longbench}"
LOG_DIR="${LOG_DIR:-logs/paper_repro_v2/longbench}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-31500}"
SEED="${SEED:-0}"
MODE="${MODE:-paper_v2_full}"
METHODS="${METHODS:-fp16 kivi_paper_g128 patternkv_paper}"

mkdir -p "$OUTPUT_DIR" "$STATUS_DIR" "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at ${PYTHON_BIN}; set PYTHON_BIN to the patternkv env python" >&2
  exit 2
fi

nvidia-smi
"$PYTHON_BIN" - <<'PY'
import torch
print("Visible GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
assert torch.cuda.device_count() >= 6, f"Expected at least 6 visible GPUs, found {torch.cuda.device_count()}"
PY

TASK_SPLITS=(
  "narrativeqa qasper multifieldqa_en multifieldqa_zh"
  "hotpotqa 2wikimqa musique dureader"
  "gov_report qmsum multi_news vcsum"
  "trec triviaqa samsum"
  "lsht passage_count passage_retrieval_en"
  "passage_retrieval_zh lcc repobench-p"
)

launch_gpu() {
  local gpu="$1"
  local tasks="$2"
  local log="${LOG_DIR}/gpu${gpu}.log"
  local pidfile="${STATUS_DIR}/gpu${gpu}.pid"
  local metafile="${STATUS_DIR}/gpu${gpu}.meta"
  printf 'gpu=%s\nmode=%s\nmethods=%s\ntasks=%s\nnum_samples=%s\nmax_input_length=%s\nstarted_at=%s\nlog=%s\n' \
    "$gpu" "$MODE" "$METHODS" "$tasks" "$NUM_SAMPLES" "$MAX_INPUT_LENGTH" "$(date --iso-8601=seconds)" "$log" > "$metafile"
  (
    set -euo pipefail
    for method in $METHODS; do
      echo "[$(date --iso-8601=seconds)] start gpu=${gpu} method=${method} tasks=${tasks}"
      cmd=("$PYTHON_BIN" bench/bench_longbench_patternkv.py \
        --method "$method" \
        --tasks $tasks \
        --num-samples "$NUM_SAMPLES" \
        --model-path "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --status-dir "$STATUS_DIR" \
        --mode "$MODE" \
        --gpu-id "$gpu" \
        --max-input-length "$MAX_INPUT_LENGTH" \
        --seed "$SEED" \
        --skip-existing)
      if [[ -n "$DATA_DIR" ]]; then
        cmd+=(--data-dir "$DATA_DIR")
      fi
      env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "${cmd[@]}"
      echo "[$(date --iso-8601=seconds)] done gpu=${gpu} method=${method}"
    done
  ) > "$log" 2>&1 &
  echo $! > "$pidfile"
  echo "started gpu=${gpu} pid=$(cat "$pidfile") log=${log} tasks=${tasks}"
}

for gpu in 0 1 2 3 4 5; do
  launch_gpu "$gpu" "${TASK_SPLITS[$gpu]}"
done

echo "Use scripts/check_longbench_6x3090.sh with OUTPUT_DIR=${OUTPUT_DIR} for progress, or inspect ${STATUS_DIR}/*.status.json."
