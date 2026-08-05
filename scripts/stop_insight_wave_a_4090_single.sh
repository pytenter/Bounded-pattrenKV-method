#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/run/insight_v2/wave_a_4090_single}"
cd "$ROOT"
if [[ ! -d "$RUN_ROOT" ]]; then
  echo "No single-4090 run directory: $RUN_ROOT"
  exit 0
fi
mapfile -t pidfiles < <(find "$RUN_ROOT" -type f \( -name 'launcher.pid' -o -name 'worker.pid' \) -print | sort -u)
pids=()
for file in "${pidfiles[@]}"; do
  pid="$(cat "$file" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] && pids+=("$pid:$file")
done
if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No valid single-4090 experiment PIDs found."
  exit 0
fi
echo "Sending SIGTERM only to recorded experiment PIDs"
for item in "${pids[@]}"; do
  pid="${item%%:*}"
  file="${item#*:}"
  if kill -0 "$pid" 2>/dev/null; then
    echo "TERM pid=$pid source=$file"
    kill -TERM "$pid" || true
  fi
done
for _ in $(seq 1 30); do
  alive=0
  for item in "${pids[@]}"; do
    pid="${item%%:*}"
    kill -0 "$pid" 2>/dev/null && alive=1
  done
  [[ "$alive" -eq 0 ]] && break
  sleep 2
done
for item in "${pids[@]}"; do
  pid="${item%%:*}"
  file="${item#*:}"
  if kill -0 "$pid" 2>/dev/null; then
    echo "KILL pid=$pid source=$file"
    kill -KILL "$pid" || true
  fi
done
echo stopping > "$RUN_ROOT/state"
