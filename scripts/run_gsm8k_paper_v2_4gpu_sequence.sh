#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper_repro_v2/gsm8k_full_2048}"
STATUS_ROOT="${STATUS_ROOT:-run/paper_repro_v2/gsm8k_full_2048_sequence_4gpu}"
LOG_ROOT="${LOG_ROOT:-logs/paper_repro_v2/gsm8k_full_2048_sequence_4gpu}"
REPORT_DIR="${REPORT_DIR:-reports/paper_repro_v2/gsm8k_full_2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
GPU_IDS="${GPU_IDS:-0 1 2 6}"
METHOD_ORDER="${METHOD_ORDER:-kivi_paper_g128 patternkv_paper fp16}"
EXPECTED="${EXPECTED:-1319}"
POLL_SECONDS="${POLL_SECONDS:-120}"
LOCK_FILE="${LOCK_FILE:-run/paper_repro_v2/gsm8k_full_2048_sequence_4gpu.lock}"

mkdir -p "$STATUS_ROOT" "$LOG_ROOT" "$REPORT_DIR" "$(dirname "$LOCK_FILE")"

read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} > 4 )); then
  echo "ERROR: GPU_IDS has ${#GPUS[@]} GPUs; this sequence is capped at 4." >&2
  exit 2
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "ERROR: another GSM8K 4-GPU sequence appears to be running: $LOCK_FILE" >&2
  exit 2
fi

MAIN_LOG="${LOG_ROOT}/sequence_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

count_method() {
  local method="$1"
  "$PYTHON_BIN" - "$OUTPUT_DIR" "$method" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
method = sys.argv[2]
files = sorted((root / method).glob("p*.json"))
rows = []
bad = 0
for path in files:
    try:
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        bad += 1
ids = [row.get("problem_id") for row in rows if isinstance(row.get("problem_id"), int)]
print(len(set(ids)), len(rows), bad)
PY
}

method_unique_count() {
  local method="$1"
  read -r unique _ _ < <(count_method "$method")
  echo "$unique"
}

method_complete() {
  local method="$1"
  local unique total bad
  read -r unique total bad < <(count_method "$method")
  [[ "$unique" -ge "$EXPECTED" && "$total" -ge "$EXPECTED" && "$bad" -eq 0 ]]
}

method_alive() {
  local method="$1"
  pgrep -af "bench/bench_gsm8k_paper.py .*--method ${method} .*--output-dir ${OUTPUT_DIR}" >/dev/null 2>&1
}

print_method_summary() {
  "$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY' | tee -a "$MAIN_LOG"
import json
from pathlib import Path

root = Path(__import__("sys").argv[1])
for method in ("kivi_paper_g128", "patternkv_paper", "fp16"):
    files = sorted((root / method).glob("p*.json"))
    rows = []
    bad = 0
    for path in files:
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            bad += 1
    ids = {r.get("problem_id") for r in rows if isinstance(r.get("problem_id"), int)}
    correct = sum(1 for r in rows if r.get("is_correct") is True)
    length = sum(1 for r in rows if r.get("stop_reason") == "length" or r.get("hit_max_new_tokens"))
    err = sum(1 for r in rows if r.get("error") or r.get("stop_reason") in ("error", "invalid_json"))
    oom = sum(1 for r in rows if r.get("stop_reason") == "oom")
    acc = 100.0 * correct / len(rows) if rows else 0.0
    print(f"{method}: unique={len(ids)}/1319 files={len(rows)} acc={acc:.2f}% correct={correct} length={length} error={err} oom={oom} bad_json={bad}")
PY
}

launch_method() {
  local method="$1"
  local status_dir="${STATUS_ROOT}/${method}"
  local log_dir="${LOG_ROOT}/${method}"
  mkdir -p "$status_dir" "$log_dir"
  log "launch method=${method} gpu_ids='${GPU_IDS}' max_new_tokens=${MAX_NEW_TOKENS}"
  env \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL_PATH="$MODEL_PATH" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    GPU_IDS="$GPU_IDS" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    STATUS_DIR="$status_dir" \
    LOG_DIR="$log_dir" \
    METHOD_FILTER="$method" \
    bash scripts/run_gsm8k_paper_full_8gpu.sh 2>&1 | tee -a "$MAIN_LOG"
}

log "GSM8K paper-v2 4-GPU sequence start"
log "methods=${METHOD_ORDER}"
log "gpu_ids=${GPU_IDS}"
log "output_dir=${OUTPUT_DIR}"

for method in $METHOD_ORDER; do
  while true; do
    if method_complete "$method"; then
      log "method=${method} complete; skipping launch"
      break
    fi

    if method_alive "$method"; then
      read -r unique total bad < <(count_method "$method")
      log "method=${method} already running; progress unique=${unique}/${EXPECTED} files=${total} bad_json=${bad}; waiting ${POLL_SECONDS}s"
      sleep "$POLL_SECONDS"
      continue
    fi

    read -r unique total bad < <(count_method "$method")
    log "method=${method} not complete and not running; progress unique=${unique}/${EXPECTED} files=${total} bad_json=${bad}"
    launch_method "$method"

    if method_complete "$method"; then
      log "method=${method} complete after launch"
      break
    fi

    read -r unique total bad < <(count_method "$method")
    log "method=${method} launch returned before completion; progress unique=${unique}/${EXPECTED} files=${total} bad_json=${bad}; will retry after ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
  done

  print_method_summary
done

log "all requested methods reached ${EXPECTED} unique GSM8K results"
"$PYTHON_BIN" scripts/summarize_gsm8k_paper_results.py \
  --results-dir "$OUTPUT_DIR" \
  --methods fp16 kivi_paper_g128 patternkv_paper \
  --report-md "${REPORT_DIR}/summary_4gpu_sequence.md" \
  --report-json "${REPORT_DIR}/summary_4gpu_sequence.json" \
  2>&1 | tee -a "$MAIN_LOG" || true
log "sequence done"
