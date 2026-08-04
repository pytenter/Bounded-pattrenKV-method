#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-31500}"
STRICT_PAPER_MODE="${STRICT_PAPER_MODE:-1}"
OUTPUT_TAG="${OUTPUT_TAG:-full_strict}"
if [[ "$STRICT_PAPER_MODE" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-results/paper_repro_v2/longbench_full_strict}"
  LOG_DIR="${LOG_DIR:-logs/paper_repro_v2/longbench_full_strict}"
  STATUS_DIR="${STATUS_DIR:-run/paper_repro_v2/longbench_full_strict}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-results/paper_repro_v2/longbench_full_8k_cap}"
  LOG_DIR="${LOG_DIR:-logs/paper_repro_v2/longbench_full_8k_cap}"
  STATUS_DIR="${STATUS_DIR:-run/paper_repro_v2/longbench_full_8k_cap}"
fi
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
METHOD_FILTER="${METHOD_FILTER:-fp16 kivi_paper_g128 patternkv_paper}"
TASK_FILTER="${TASK_FILTER:-narrativeqa qasper multifieldqa_en multifieldqa_zh hotpotqa 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec triviaqa samsum lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$STATUS_DIR"
if [[ "$STRICT_PAPER_MODE" == "1" && "$MAX_INPUT_LENGTH" != "31500" ]]; then
  echo "STRICT_PAPER_MODE=1 requires MAX_INPUT_LENGTH=31500" >&2
  exit 2
fi
if [[ "$STRICT_PAPER_MODE" == "1" && ! -f reports/paper_repro_v2/longbench_memory_qualification/report.json ]]; then
  echo "Missing memory qualification report; refusing strict full." >&2
  exit 2
fi
if [[ "$STRICT_PAPER_MODE" == "1" ]]; then
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
s=json.loads(Path("reports/paper_repro_v2/longbench_memory_qualification/report.json").read_text())
if s.get("status") != "STRICT_QUALIFICATION_PASS":
    raise SystemExit("Memory qualification did not PASS; refusing strict full.")
PY
fi
read -r -a gpus <<< "$GPU_IDS"
num_workers="${#gpus[@]}"
main_log="${LOG_DIR}/main_$(date +%Y%m%d_%H%M%S).log"
echo "LongBench paper v2 start strict=${STRICT_PAPER_MODE} max_input=${MAX_INPUT_LENGTH}" | tee -a "$main_log"
for method in $METHOD_FILTER; do
  pids=()
  for idx in "${!gpus[@]}"; do
    gpu="${gpus[$idx]}"
    tasks=()
    n=0
    for task in $TASK_FILTER; do
      if (( n % num_workers == idx )); then tasks+=("$task"); fi
      n=$((n+1))
    done
    [[ "${#tasks[@]}" == "0" ]] && continue
    cmd=("$PYTHON_BIN" bench/bench_longbench_patternkv.py --method "$method" --tasks "${tasks[@]}" --num-samples 0 --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR" --status-dir "$STATUS_DIR" --mode "paper_v2_${OUTPUT_TAG}" --gpu-id "$gpu" --max-input-length "$MAX_INPUT_LENGTH" --skip-existing)
    if [[ "$DRY_RUN" == "1" ]]; then echo "DRY ${cmd[*]}"; continue; fi
    log="${LOG_DIR}/${method}_gpu${gpu}.log"
    nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "${cmd[@]}" > "$log" 2>&1 &
    pids+=("$!")
    echo "started method=${method} gpu=${gpu} pid=${pids[-1]} tasks=${tasks[*]} log=${log}" | tee -a "$main_log"
    sleep 3
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
done
