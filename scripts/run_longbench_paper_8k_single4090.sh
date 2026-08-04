#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/Bounded-pattrenKV-method}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/meta-llama/Llama-3.1-8B-Instruct}"
DATA_DIR="${DATA_DIR:-/root/Block-kvcache-experiment/data/LongBench}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-8192}"
OUTPUT_TAG="${OUTPUT_TAG:-longbench_full_8k_4090}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper_repro_v2/${OUTPUT_TAG}}"
STATUS_DIR="${STATUS_DIR:-run/paper_repro_v2/${OUTPUT_TAG}}"
LOG_DIR="${LOG_DIR:-logs/paper_repro_v2/${OUTPUT_TAG}}"
METHOD_FILTER="${METHOD_FILTER:-}"
TASK_FILTER="${TASK_FILTER:-}"
SAMPLE_FILTER="${SAMPLE_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
RETRY_FAILED="${RETRY_FAILED:-0}"
RETRY_OOM="${RETRY_OOM:-0}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR" "$STATUS_DIR" "$LOG_DIR"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be exactly 0" >&2
  exit 2
fi
if [[ "$MAX_INPUT_LENGTH" != "8192" ]]; then
  echo "MAX_INPUT_LENGTH must be 8192 for longbench_full_8k_4090" >&2
  exit 2
fi
if [[ "$OUTPUT_DIR" == *"longbench_full_strict"* ]]; then
  echo "Refusing to write 8K results into strict directory: $OUTPUT_DIR" >&2
  exit 2
fi

cmd=(
  "$PYTHON_BIN" scripts/run_longbench_paper_8k_single4090.py
  --model-path "$MODEL_PATH"
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --status-dir "$STATUS_DIR"
  --max-input-length "$MAX_INPUT_LENGTH"
)
[[ -n "$METHOD_FILTER" ]] && cmd+=(--method-filter "$METHOD_FILTER")
[[ -n "$TASK_FILTER" ]] && cmd+=(--task-filter "$TASK_FILTER")
[[ -n "$SAMPLE_FILTER" ]] && cmd+=(--sample-filter "$SAMPLE_FILTER")
[[ "$DRY_RUN" == "1" ]] && cmd+=(--dry-run)
[[ "$RESUME" == "1" ]] && cmd+=(--resume)
[[ "$RETRY_FAILED" == "1" ]] && cmd+=(--retry-failed)
[[ "$RETRY_OOM" == "1" ]] && cmd+=(--retry-oom)

echo "physical GPU count used: 1"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH=$MODEL_PATH"
echo "MAX_INPUT_LENGTH=$MAX_INPUT_LENGTH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "RESUME=$RESUME RETRY_FAILED=$RETRY_FAILED RETRY_OOM=$RETRY_OOM"
echo "OOM policy: record and continue; no fallback below 8192"
echo "Methods: ${METHOD_FILTER:-fp16 kivi_paper_g128 patternkv_paper}"
echo "Task filter: ${TASK_FILTER:-all 21 tasks}"

if [[ "$DRY_RUN" == "1" ]]; then
  "${cmd[@]}"
else
  for method in fp16 kivi_paper_g128 patternkv_paper; do
    if [[ -n "$METHOD_FILTER" && " $METHOD_FILTER " != *" $method "* ]]; then
      continue
    fi
    echo "[$(date --iso-8601=seconds)] phase will run method=$method" | tee -a "$LOG_DIR/launcher.phase.log"
  done
  "${cmd[@]}"
fi
