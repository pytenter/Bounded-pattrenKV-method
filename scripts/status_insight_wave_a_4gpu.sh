#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-results/insight_v2/wave_a}"
REPORT_ROOT="${REPORT_ROOT:-reports/insight_v2/wave_a}"
LOG_ROOT="${LOG_ROOT:-logs/insight_v2/wave_a}"
RUN_ROOT="${RUN_ROOT:-run/insight_v2/wave_a}"
WAVE_A_GPU_IDS="${WAVE_A_GPU_IDS:-4 5 6 7}"
cd "$ROOT"

echo "== Insight Wave A 4GPU status =="
if [[ -f "$RUN_ROOT/launcher.pid" ]]; then
  launcher_pid="$(cat "$RUN_ROOT/launcher.pid")"
  if [[ "$launcher_pid" =~ ^[0-9]+$ ]] && kill -0 "$launcher_pid" 2>/dev/null; then
    echo "launcher_pid=$launcher_pid status=running"
  else
    echo "launcher_pid=$launcher_pid status=not-running"
  fi
else
  echo "launcher_pid=missing"
fi

echo
echo "== GPU4-7 =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F, '$1 ~ /^[[:space:]]*[4-7][[:space:]]*$/ {print}'

echo
echo "== Queue PIDs =="
running_tasks_file="$(mktemp)"
trap 'rm -f "$running_tasks_file"' EXIT
for gpu in $WAVE_A_GPU_IDS; do
  pidfile="$RUN_ROOT/gpu${gpu}.queue.pid"
  current="$RUN_ROOT/gpu${gpu}.current_task"
  if [[ -f "$pidfile" ]]; then
    pid="$(cat "$pidfile")"
    state="not-running"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      state="running"
    fi
    task="idle"
    [[ -f "$current" ]] && task="$(cat "$current")"
    if [[ "$state" == "running" && "$task" != "idle" ]]; then
      echo "$task" >> "$running_tasks_file"
    elif [[ "$state" != "running" && "$task" != "idle" ]]; then
      task="stale:$task"
    fi
    echo "gpu=$gpu queue_pid=$pid status=$state current_task=$task"
  else
    echo "gpu=$gpu queue_pid=missing current_task=unknown"
  fi
done

echo
echo "== Task Summary =="
export RUNNING_TASKS_FILE="$running_tasks_file"
"$PYTHON_BIN" - <<'PY'
import json
import math
import os
import time
from pathlib import Path

result_root = Path(os.environ.get("RESULT_ROOT", "results/insight_v2/wave_a"))
report_root = Path(os.environ.get("REPORT_ROOT", "reports/insight_v2/wave_a"))
running_tasks_path = Path(os.environ["RUNNING_TASKS_FILE"])
running_tasks = set(running_tasks_path.read_text(encoding="utf-8").splitlines()) if running_tasks_path.exists() else set()
plan = [
    ("longbench", "hotpotqa", 12),
    ("longbench", "samsum", 12),
    ("longbench", "passage_retrieval_en", 12),
    ("longbench", "passage_retrieval_zh", 12),
    ("longbench", "dureader", 12),
    ("gsm8k", "gsm8k", 50),
]

def bad_number(value):
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, dict):
        return any(bad_number(v) for v in value.values())
    if isinstance(value, list):
        return any(bad_number(v) for v in value)
    return False

total = {"selected": 0, "completed": 0, "failed": 0, "running": 0, "missing": 0, "oom": 0, "hook_errors": 0, "observer_completed": 0, "observer_missing": 0}
now = time.time()
for dataset, task, selected in plan:
    gen_dir = result_root / "generation" / dataset / task
    obs_dir = result_root / "observer" / dataset / task
    gen_files = sorted(gen_dir.glob("*.json")) if gen_dir.exists() else []
    obs_files = sorted(obs_dir.glob("*.json")) if obs_dir.exists() else []
    completed = failed = oom = hook_errors = nan_inf = 0
    mtimes = []
    for path in gen_files:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            failed += 1
            continue
        mtimes.append(path.stat().st_mtime)
        err = rec.get("error")
        stop = rec.get("stop_reason")
        obs_status = rec.get("observer_status")
        completed += int(not err and stop not in {"error", "oom"} and obs_status == "completed")
        failed += int(bool(err) or stop == "error")
        oom += int(stop == "oom" or (isinstance(err, str) and "OutOfMemory" in err))
        hook_errors += int(isinstance(err, str) and ("InsightHookError" in err or "hook" in err.lower()))
        nan_inf += int(bad_number(rec))
    obs_completed = 0
    obs_missing = max(selected - len(obs_files), 0)
    for path in obs_files:
        try:
            obs = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        obs_completed += int(obs.get("status") == "completed" and not bad_number(obs))
    missing = max(selected - len(gen_files), 0)
    running = 1 if f"{dataset}/{task}" in running_tasks else 0
    last = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(max(mtimes))) if mtimes else "never"
    print(
        f"{dataset}/{task}: selected={selected} completed={completed} failed={failed} running={running} "
        f"missing={missing} oom={oom} hook_errors={hook_errors} nan_inf={nan_inf} "
        f"observer_completed={obs_completed} observer_missing={obs_missing} last_update={last}"
    )
    total["selected"] += selected
    total["completed"] += completed
    total["failed"] += failed
    total["running"] += running
    total["missing"] += missing
    total["oom"] += oom
    total["hook_errors"] += hook_errors
    total["observer_completed"] += obs_completed
    total["observer_missing"] += obs_missing
print("TOTAL:", " ".join(f"{k}={v}" for k, v in total.items()))
completion = {
    "schema_version": "insight_v2.wave_a_status",
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    **total,
}
report_root.mkdir(parents=True, exist_ok=True)
(report_root / "status_latest.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo
echo "== Last 20 log lines per task =="
for log in "$LOG_ROOT"/gpu*_*/run.log "$LOG_ROOT"/gpu*_queue.log; do
  [[ -f "$log" ]] || continue
  echo
  echo "--- $log ---"
  tail -n 20 "$log"
done
