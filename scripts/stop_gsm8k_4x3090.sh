#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for pidfile in run/gsm8k/*/*/*.pid; do
  [[ -e "$pidfile" ]] || continue
  pid="$(cat "$pidfile")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "stopping pid=${pid} from ${pidfile}"
    kill "$pid" || true
  fi
done
