#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/run/insight_v2/wave_a_8gpu}"
cd "$ROOT"

if ! find "$RUN_ROOT" -maxdepth 2 -type f -name '*.pid' -print -quit 2>/dev/null | grep -q .; then
  echo "No Insight Wave A 8GPU PID files found."
  exit 0
fi

find "$RUN_ROOT" -maxdepth 2 -type f -name '*.pid' -print | sort | while read -r pidfile; do
  pid="$(cat "$pidfile")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping PID $pid from $pidfile"
    kill "$pid"
  fi
done
