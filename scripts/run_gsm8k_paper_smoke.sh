#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper_repro_v2/gsm8k_smoke}"
STATUS_DIR="${STATUS_DIR:-run/paper_repro_v2/gsm8k_smoke}"
LOG_DIR="${LOG_DIR:-logs/paper_repro_v2/gsm8k_smoke}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
METHODS="${METHOD_FILTER:-fp16 kivi_paper_g128 patternkv_paper}"
mkdir -p "$OUTPUT_DIR" "$STATUS_DIR" "$LOG_DIR" reports/paper_repro_v2/gsm8k_full
rc=0
i=0
for method in $METHODS; do
  gpu="$i"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" bench/bench_gsm8k_paper.py \
    --method "$method" --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR" --status-dir "$STATUS_DIR" \
    --experiment-id gsm8k_paper_smoke --problem-ids 0 1 2 --gpu-id "$gpu" --max-new-tokens "$MAX_NEW_TOKENS" \
    > "${LOG_DIR}/${method}_gpu${gpu}.log" 2>&1 || rc=1
  i=$((i+1))
done
"$PYTHON_BIN" scripts/summarize_gsm8k_paper_results.py --results-dir "$OUTPUT_DIR" --methods $METHODS --report-md reports/paper_repro_v2/gsm8k_full/smoke_test_report.md --report-json reports/paper_repro_v2/gsm8k_full/smoke_test_report.json || rc=1
nvidia-smi
exit "$rc"
