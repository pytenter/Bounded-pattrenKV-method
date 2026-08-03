#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

shopt -s nullglob
for pidfile in run/longbench/*.pid; do
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [[ -n "$pid" ]] || continue
  if kill -0 "$pid" 2>/dev/null; then
    echo "stopping pid=${pid} from ${pidfile}"
    kill "$pid" 2>/dev/null || true
  else
    echo "pid not running: ${pid} from ${pidfile}"
  fi
done

echo "Only PIDs recorded under run/longbench/*.pid were signaled."
