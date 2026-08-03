#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper_repro_v2/aime24_smoke}"
STATUS_DIR="${STATUS_DIR:-run/paper_repro_v2/aime24_smoke}"
LOG_DIR="${LOG_DIR:-logs/paper_repro_v2/aime24_smoke}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
BASE_SEED="${BASE_SEED:-42}"
METHODS="${METHOD_FILTER:-fp16 kivi_paper_g128 patternkv_paper}"

mkdir -p "$OUTPUT_DIR" "$STATUS_DIR" "$LOG_DIR" reports/paper_repro_v2/aime24

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required for AIME smoke; refusing to use a non-DeepSeek fallback." >&2
  "$PYTHON_BIN" - <<'PY'
from bench.aime_utils import search_model_candidates
import json
print("DeepSeek-R1-Distill-Llama-8B candidates:", json.dumps(search_model_candidates(), ensure_ascii=False))
PY
  exit 2
fi

rc=0
for i in 0 1 2; do
  method="$(echo "$METHODS" | tr ' ' '\n' | sed -n "$((i+1))p")"
  [[ -z "${method:-}" ]] && continue
  log="${LOG_DIR}/${method}_gpu${i}.log"
  env CUDA_VISIBLE_DEVICES="$i" PYTHONUNBUFFERED=1 "$PYTHON_BIN" bench/bench_aime24_patternkv.py \
    --method "$method" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --status-dir "$STATUS_DIR" \
    --experiment-id aime24_smoke \
    --num-samples 1 \
    --problem-ids 0 \
    --gpu-id "$i" \
    --base-seed "$BASE_SEED" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    > "$log" 2>&1 || rc=1
done

"$PYTHON_BIN" scripts/summarize_aime24_results.py \
  --results-dir "$OUTPUT_DIR" \
  --num-samples 1 \
  --methods $METHODS \
  --report-md reports/paper_repro_v2/aime24/smoke_test_report.md \
  --report-json reports/paper_repro_v2/aime24/smoke_test_report.json || rc=1

nvidia-smi
exit "$rc"
