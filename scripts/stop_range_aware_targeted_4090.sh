#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_NAME="${TARGET_NAME:-range_aware_targeted_4090}"
RUN_ROOT="${RUN_ROOT:-$ROOT/run/insight_v2/$TARGET_NAME}"

if ! find "$RUN_ROOT" -maxdepth 2 -type f -name '*.pid' -print -quit 2>/dev/null | grep -q .; then
  echo "No range-aware targeted 4090 PID files found."
  exit 0
fi

find "$RUN_ROOT" -maxdepth 2 -type f -name '*.pid' -print | sort | while read -r pidfile; do
  pid="$(cat "$pidfile")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping PID $pid from $pidfile"
    kill "$pid"
  fi
done
