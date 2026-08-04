#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! compgen -G "run/insight_v2/*.pid" > /dev/null; then
  echo "No insight_v2 PID files found."
  exit 0
fi

for pidfile in run/insight_v2/*.pid; do
  pid="$(cat "$pidfile")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping PID $pid from $pidfile"
    kill "$pid"
  fi
done
