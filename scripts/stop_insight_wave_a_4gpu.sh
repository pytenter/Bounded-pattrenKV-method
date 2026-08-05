#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-run/insight_v2/wave_a}"
cd "$ROOT"

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "No Wave A run directory: $RUN_ROOT"
  exit 0
fi

mapfile -t pidfiles < <(
  find "$RUN_ROOT" -type f \( \
    -name 'launcher.pid' -o \
    -name 'gpu*.queue.pid' -o \
    -name 'worker.pid' \
  \) -print | sort -u
)

if [[ "${#pidfiles[@]}" -eq 0 ]]; then
  echo "No Wave A PID files found."
  exit 0
fi

pids=()
for pidfile in "${pidfiles[@]}"; do
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ "$pid" =~ ^[0-9]+$ ]]; then
    pids+=("$pid:$pidfile")
  fi
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No valid Wave A PIDs found."
  exit 0
fi

echo "Sending SIGTERM to Wave A PIDs only:"
for entry in "${pids[@]}"; do
  pid="${entry%%:*}"
  pidfile="${entry#*:}"
  if kill -0 "$pid" 2>/dev/null; then
    echo "  TERM pid=$pid source=$pidfile"
    kill -TERM "$pid" || true
  else
    echo "  stale pid=$pid source=$pidfile"
  fi
done

for _ in $(seq 1 24); do
  alive=0
  for entry in "${pids[@]}"; do
    pid="${entry%%:*}"
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
    fi
  done
  [[ "$alive" -eq 0 ]] && break
  sleep 5
done

still_alive=()
for entry in "${pids[@]}"; do
  pid="${entry%%:*}"
  pidfile="${entry#*:}"
  if kill -0 "$pid" 2>/dev/null; then
    still_alive+=("$pid:$pidfile")
  fi
done

if [[ "${#still_alive[@]}" -gt 0 ]]; then
  echo "Some Wave A PIDs did not exit after SIGTERM; sending SIGKILL to those exact PIDs:"
  for entry in "${still_alive[@]}"; do
    pid="${entry%%:*}"
    pidfile="${entry#*:}"
    echo "  KILL pid=$pid source=$pidfile"
    kill -KILL "$pid" || true
  done
fi

echo "Wave A stop command complete."
