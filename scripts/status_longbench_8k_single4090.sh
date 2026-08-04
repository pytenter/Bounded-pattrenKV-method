#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/root/Bounded-pattrenKV-method}"
OUTPUT_TAG="${OUTPUT_TAG:-longbench_full_8k_4090}"
LOG_DIR="$PROJECT_ROOT/logs/paper_repro_v2/${OUTPUT_TAG}"
STATUS_DIR="$PROJECT_ROOT/run/paper_repro_v2/${OUTPUT_TAG}"
RESULT_DIR="$PROJECT_ROOT/results/paper_repro_v2/${OUTPUT_TAG}"
PID_FILE="$LOG_DIR/launcher.pid"
echo "launcher PID: $(cat "$PID_FILE" 2>/dev/null || echo none)"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then echo "launcher: alive"; else echo "launcher: not running"; fi
if [[ -f "$STATUS_DIR/runner.status.json" ]]; then
  python - <<PY
import json
p="$STATUS_DIR/runner.status.json"
d=json.load(open(p))
print("worker PID:", d.get("worker_pid"))
print("phase:", d.get("phase"))
print("method:", d.get("current_method"))
print("task:", d.get("current_task"))
print("sample:", d.get("current_sample"))
print("planned:", d.get("planned_total"))
PY
else
  echo "runner status: missing"
fi
python - <<PY
import json
from pathlib import Path
base=Path("$RESULT_DIR")
completed=success=oom=error=0
latest=None
for p in base.glob("*/*.jsonl"):
    latest=max(latest or p.stat().st_mtime, p.stat().st_mtime)
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        try: r=json.loads(line)
        except Exception: continue
        completed+=1
        if r.get("stop_reason")=="oom": oom+=1
        elif r.get("stop_reason")=="error": error+=1
        else: success+=1
print("completed:", completed)
print("success:", success)
print("OOM:", oom)
print("error:", error)
print("recent result mtime:", latest)
PY
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
df -h "$PROJECT_ROOT" | tail -n +2
echo "--- recent launcher log ---"
tail -n 30 "$LOG_DIR/launcher.nohup.log" 2>/dev/null || true
