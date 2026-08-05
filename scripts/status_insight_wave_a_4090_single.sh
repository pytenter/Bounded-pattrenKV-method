#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/patternkv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/insight_v2/wave_a_4090_single}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT/reports/insight_v2/wave_a_4090_single}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/insight_v2/wave_a_4090_single}"
RUN_ROOT="${RUN_ROOT:-$ROOT/run/insight_v2/wave_a_4090_single}"
cd "$ROOT"

echo "== Insight Wave A single RTX 4090 status =="
for name in launcher.pid worker.pid current_task current_sample state; do
  path="$RUN_ROOT/$name"
  [[ -f "$path" ]] && echo "$name=$(tr '\n' ' ' < "$path")"
done
if [[ -f "$RUN_ROOT/launcher.pid" ]]; then
  pid="$(cat "$RUN_ROOT/launcher.pid")"
  if kill -0 "$pid" 2>/dev/null; then echo "launcher_alive=true"; else echo "launcher_alive=false"; fi
fi
if [[ -f "$RUN_ROOT/worker.pid" ]]; then
  pid="$(cat "$RUN_ROOT/worker.pid")"
  if kill -0 "$pid" 2>/dev/null; then echo "worker_alive=true"; else echo "worker_alive=false"; fi
fi

echo
echo "== RTX 4090 =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | grep -E 'RTX 4090' || true

echo
echo "== Task counts =="
PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}" \
RESULT_ROOT="$RESULT_ROOT" REPORT_ROOT="$REPORT_ROOT" RUN_ROOT="$RUN_ROOT" "$PYTHON_BIN" - <<'PY'
import json, os, time
from pathlib import Path
from insight_wave_a_4090_utils import PLAN, load_reference, plan_samples, result_path, is_completed_generation, is_completed_observer
root = Path(os.environ["RESULT_ROOT"])
reference = load_reference()
rows = plan_samples(reference)
total = {k: 0 for k in ("planned","completed","failed","running","missing","oom","hook_errors","observer_completed","observer_missing")}
for dataset, task, planned in PLAN:
    samples = [r for r in rows if r["dataset"] == dataset and r["task"] == task]
    completed = failed = oom = hook = obs_completed = 0
    for sample in samples:
        gen = result_path(root / "generation", sample, "oracle")
        obs = result_path(root / "observer", sample, "oracle")
        if is_completed_generation(gen):
            completed += 1
        elif gen.exists():
            failed += 1
            try:
                payload = json.loads(gen.read_text())
                oom += int(payload.get("stop_reason") == "oom")
                hook += int("hook" in str(payload.get("error", "")).lower())
            except Exception:
                pass
        obs_completed += int(is_completed_observer(obs))
    missing = max(planned - completed - failed, 0)
    obs_missing = max(planned - obs_completed, 0)
    running = int((Path(os.environ["RUN_ROOT"]) / "current_task").read_text().strip() == f"{dataset}/{task}") if (Path(os.environ["RUN_ROOT"]) / "current_task").exists() else 0
    print(f"{dataset}/{task}: planned={planned} completed={completed} failed={failed} running={running} missing={missing} oom={oom} hook_errors={hook} observer_completed={obs_completed} observer_missing={obs_missing}")
    for key, value in (("planned",planned),("completed",completed),("failed",failed),("running",running),("missing",missing),("oom",oom),("hook_errors",hook),("observer_completed",obs_completed),("observer_missing",obs_missing)):
        total[key] += value
print("TOTAL:", " ".join(f"{k}={v}" for k,v in total.items()))
Path(os.environ["REPORT_ROOT"]).mkdir(parents=True, exist_ok=True)
(Path(os.environ["REPORT_ROOT"]) / "status_latest.json").write_text(json.dumps({"generated_at":time.time(),**total}, indent=2, sort_keys=True) + "\n")
PY

echo
echo "== Last 30 log lines =="
for log in "$LOG_ROOT"/*.log; do
  [[ -f "$log" ]] || continue
  echo "--- $log ---"
  tail -n 30 "$log"
done
