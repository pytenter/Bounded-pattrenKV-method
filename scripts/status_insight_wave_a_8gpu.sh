#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/insight_v2/wave_a_8gpu}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT/reports/insight_v2/wave_a_8gpu}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/insight_v2/wave_a_8gpu}"
RUN_ROOT="${RUN_ROOT:-$ROOT/run/insight_v2/wave_a_8gpu}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
cd "$ROOT"
export REPORT_ROOT

echo "== Insight Wave A 8GPU status =="
if [[ -f "$REPORT_ROOT/manifest.json" ]]; then
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads((Path(os.environ["REPORT_ROOT"]) / "manifest.json").read_text(encoding="utf-8"))
print(f"ready_to_launch={manifest['readiness']['ready_to_launch']} total_planned={manifest['total_planned']}")
print(f"model_complete={manifest['readiness']['model_complete']} model_path={manifest['model_path']}")
print(f"longbench={manifest['longbench_data_dir']}")
print(f"gsm8k={manifest['gsm8k_data_path']}")
for job in manifest["jobs"]:
    print(f"job gpu={job['gpu']} dataset={job['dataset']} task={job['task']} selected={job['limit']}")
PY
else
  echo "manifest=missing"
fi

echo
echo "== GPU status =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

echo
echo "== PID files =="
find "$RUN_ROOT" -maxdepth 2 -type f \( -name '*.pid' -o -name 'gpu*.current_task' \) -print 2>/dev/null | sort || true

echo
echo "== Last logs =="
for log in "$LOG_ROOT"/gpu*_*/run.log; do
  [[ -f "$log" ]] || continue
  echo "--- $log ---"
  tail -n 12 "$log"
done
