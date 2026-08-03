#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-20}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at ${PYTHON_BIN}; set PYTHON_BIN to the patternkv env python" >&2
  exit 1
fi

complete() {
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

tasks = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "trec", "passage_retrieval_en", "lcc"]
missing = []
errors = 0
for task in tasks:
    path = Path("results/longbench/kivi") / f"{task}.jsonl"
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors += sum(1 for row in rows if row.get("error"))
    if len(rows) < 50:
        missing.append(f"kivi/{task}:{len(rows)}/50")
print("KIVI_COUNTS", " ".join(missing) if missing else "all>=50", "errors=", errors, flush=True)
raise SystemExit(0 if not missing and errors == 0 else 1)
PY
}

round=0
while ! complete; do
  round=$((round + 1))
  echo "===== KIVI FULL RESUME ROUND ${round} $(date -Is) ====="
  LAUNCH_STAGGER_SECONDS="$LAUNCH_STAGGER_SECONDS" bash scripts/run_longbench_6x3090_kivi_full.sh
  while pgrep -f 'bench/bench_longbench_patternkv.py.*--method kivi' >/dev/null 2>&1; do
    sleep 60
    complete || true
  done
  echo "===== KIVI WORKERS IDLE $(date -Is) ====="
  complete || true
done

echo "===== KIVI FULL COUNTS COMPLETE $(date -Is) ====="
"$PYTHON_BIN" bench/summarize_longbench.py \
  --expected-samples 50 \
  --methods fp16 kivi \
  --summary-name kivi_vs_fp16 \
  --report-path reports/kivi_longbench_8x50.md \
  --require-complete

"$PYTHON_BIN" bench/summarize_longbench.py \
  --expected-samples 50 \
  --methods fp16 patternkv kivi \
  --summary-name all_methods \
  --report-path reports/longbench_fp16_patternkv_kivi_8x50.md \
  --require-complete
