#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper_repro_v2/aime24_eta}"
STATUS_DIR="${STATUS_DIR:-run/paper_repro_v2/aime24_eta}"
LOG_DIR="${LOG_DIR:-logs/paper_repro_v2/aime24_eta}"
BASE_SEED="${BASE_SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"

mkdir -p "$OUTPUT_DIR" "$STATUS_DIR" "$LOG_DIR" reports/paper_repro_v2/aime24

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required for ETA; refusing to start." >&2
  exit 2
fi

declare -a methods=(fp16 kivi_paper_g128 patternkv_paper)
rc=0
for i in 0 1 2; do
  method="${methods[$i]}"
  env CUDA_VISIBLE_DEVICES="$i" PYTHONUNBUFFERED=1 "$PYTHON_BIN" bench/bench_aime24_patternkv.py \
    --method "$method" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --status-dir "$STATUS_DIR" \
    --experiment-id aime24_eta \
    --num-samples 1 \
    --problem-ids 0 10 20 \
    --gpu-id "$i" \
    --base-seed "$BASE_SEED" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    > "${LOG_DIR}/${method}_gpu${i}.log" 2>&1 || rc=1 &
done
wait || rc=1

"$PYTHON_BIN" scripts/summarize_aime24_results.py \
  --results-dir "$OUTPUT_DIR" \
  --num-samples 1 \
  --methods fp16 kivi_paper_g128 patternkv_paper \
  --report-md reports/paper_repro_v2/aime24/eta_report.md \
  --report-json reports/paper_repro_v2/aime24/eta_report.json || rc=1

exit "$rc"
