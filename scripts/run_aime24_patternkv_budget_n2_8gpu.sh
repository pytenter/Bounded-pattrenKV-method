#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper_repro_v2/aime24_budget_n2}"
STATUS_DIR="${STATUS_DIR:-run/paper_repro_v2/aime24_budget_n2}"
LOG_DIR="${LOG_DIR:-logs/paper_repro_v2/aime24_budget_n2}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"
BASE_SEED="${BASE_SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
WORKER_GPU_IDS="${WORKER_GPU_IDS:-0 1 2 3 4 5 6 7}"
METHOD_FILTER="${METHOD_FILTER:-fp16 kivi_paper_g128 patternkv_paper}"
STARTUP_STAGGER_SECONDS="${STARTUP_STAGGER_SECONDS:-5}"

mkdir -p "$OUTPUT_DIR" "$STATUS_DIR" "$LOG_DIR"

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH=/path/to/DeepSeek-R1-Distill-Llama-8B is required; refusing full run." >&2
  exit 2
fi

read -r -a gpu_ids <<< "$WORKER_GPU_IDS"
num_workers="${#gpu_ids[@]}"
main_log="${LOG_DIR}/main_$(date +%Y%m%d_%H%M%S).log"
echo "AIME24 budget run num_workers=${num_workers} num_samples=${NUM_SAMPLES} max_new_tokens=${MAX_NEW_TOKENS}" | tee -a "$main_log"
nvidia-smi | tee -a "$main_log"

for method in $METHOD_FILTER; do
  echo "[$(date --iso-8601=seconds)] phase start method=${method}" | tee -a "$main_log"
  pids=()
  rc=0
  for idx in "${!gpu_ids[@]}"; do
    gpu="${gpu_ids[$idx]}"
    log="${LOG_DIR}/${method}_gpu${gpu}.log"
    env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" bench/bench_aime24_patternkv.py \
      --method "$method" \
      --model-path "$MODEL_PATH" \
      --output-dir "$OUTPUT_DIR" \
      --status-dir "$STATUS_DIR" \
      --experiment-id aime24_budget_n2 \
      --num-samples "$NUM_SAMPLES" \
      --worker-index "$idx" \
      --num-workers "$num_workers" \
      --gpu-id "$gpu" \
      --base-seed "$BASE_SEED" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      > "$log" 2>&1 &
    pids+=("$!")
    echo "started method=${method} gpu=${gpu} worker=${idx}/${num_workers} pid=${pids[-1]} log=${log}" | tee -a "$main_log"
    sleep "$STARTUP_STAGGER_SECONDS"
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  echo "[$(date --iso-8601=seconds)] phase done method=${method} rc=${rc}" | tee -a "$main_log"
done

"$PYTHON_BIN" scripts/summarize_aime24_results.py \
  --results-dir "$OUTPUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --methods fp16 kivi_paper_g128 patternkv_paper \
  --report-md reports/paper_repro_v2/aime24/results_summary.md \
  --report-json reports/paper_repro_v2/aime24/results_summary.json | tee -a "$main_log" || true
