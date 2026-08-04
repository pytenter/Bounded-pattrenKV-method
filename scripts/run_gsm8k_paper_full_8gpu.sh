#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper_repro_v2/gsm8k_full}"
STATUS_DIR="${STATUS_DIR:-run/paper_repro_v2/gsm8k_full}"
LOG_DIR="${LOG_DIR:-logs/paper_repro_v2/gsm8k_full}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
METHOD_FILTER="${METHOD_FILTER:-fp16 kivi_paper_g128 patternkv_paper}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$OUTPUT_DIR" "$STATUS_DIR" "$LOG_DIR" reports/paper_repro_v2/gsm8k_full
read -r -a gpus <<< "$GPU_IDS"
num_workers="${#gpus[@]}"
main_log="${LOG_DIR}/main_$(date +%Y%m%d_%H%M%S).log"
echo "GSM8K paper full start workers=${num_workers} methods=${METHOD_FILTER}" | tee -a "$main_log"
for method in $METHOD_FILTER; do
  pids=()
  echo "phase start ${method}" | tee -a "$main_log"
  for idx in "${!gpus[@]}"; do
    gpu="${gpus[$idx]}"
    log="${LOG_DIR}/${method}_gpu${gpu}.log"
    pidfile="${STATUS_DIR}/${method}_gpu${gpu}.pid"
    cmd=("$PYTHON_BIN" bench/bench_gsm8k_paper.py --method "$method" --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR" --status-dir "$STATUS_DIR" --experiment-id gsm8k_full_paper --worker-index "$idx" --num-workers "$num_workers" --gpu-id "$gpu" --max-new-tokens "$MAX_NEW_TOKENS")
    if [[ "$DRY_RUN" == "1" ]]; then cmd+=(--dry-run); fi
    nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "${cmd[@]}" > "$log" 2>&1 &
    echo $! > "$pidfile"
    pids+=("$!")
    echo "started method=${method} gpu=${gpu} worker=${idx}/${num_workers} pid=${pids[-1]} log=${log}" | tee -a "$main_log"
    sleep 3
  done
  if [[ "$DRY_RUN" == "1" ]]; then
    for pid in "${pids[@]}"; do wait "$pid" || true; done
  else
    echo "phase ${method} launched; waiting before next phase is handled by this script"
    for pid in "${pids[@]}"; do wait "$pid" || true; done
  fi
  echo "phase done ${method}" | tee -a "$main_log"
done
"$PYTHON_BIN" scripts/summarize_gsm8k_paper_results.py --results-dir "$OUTPUT_DIR" --methods fp16 kivi_paper_g128 patternkv_paper --report-md reports/paper_repro_v2/gsm8k_full/summary.md --report-json reports/paper_repro_v2/gsm8k_full/summary.json || true
