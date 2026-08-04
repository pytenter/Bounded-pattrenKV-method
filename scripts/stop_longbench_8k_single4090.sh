#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/root/Bounded-pattrenKV-method}"
OUTPUT_TAG="${OUTPUT_TAG:-longbench_full_8k_4090}"
LOG_DIR="$PROJECT_ROOT/logs/paper_repro_v2/${OUTPUT_TAG}"
PID_FILE="$LOG_DIR/launcher.pid"
mkdir -p "$LOG_DIR"
if [[ ! -f "$PID_FILE" ]]; then
  echo "No launcher pid file: $PID_FILE"
  exit 0
fi
pid="$(cat "$PID_FILE")"
echo "stopping launcher pid=$pid at $(date --iso-8601=seconds)" | tee -a "$LOG_DIR/stop.log"
if kill -0 "$pid" 2>/dev/null; then
  kill -TERM "$pid" || true
  for _ in {1..60}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "launcher stopped after SIGTERM" | tee -a "$LOG_DIR/stop.log"
      exit 0
    fi
    sleep 2
  done
  echo "launcher still alive after wait; sending SIGKILL to pid=$pid" | tee -a "$LOG_DIR/stop.log"
  kill -KILL "$pid" || true
else
  echo "launcher already stopped" | tee -a "$LOG_DIR/stop.log"
fi
