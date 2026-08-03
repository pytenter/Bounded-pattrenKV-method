#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
GPU_MEMORY_BUSY_MIB="${GPU_MEMORY_BUSY_MIB:-2000}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at ${PYTHON_BIN}; set PYTHON_BIN to the patternkv env python" >&2
  exit 1
fi

mkdir -p logs/longbench/kivi run/longbench

free_enough() {
  "$PYTHON_BIN" - <<PY
import subprocess
import sys

limit = int("${GPU_MEMORY_BUSY_MIB}")
target = set(range(6))
out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
busy = []
for line in out.splitlines():
    idx_s, used_s = [part.strip() for part in line.split(",", 1)]
    idx = int(idx_s)
    used = int(used_s)
    if idx in target and used > limit:
        busy.append((idx, used))
if busy:
    print("busy " + " ".join(f"gpu{idx}={used}MiB" for idx, used in busy))
    raise SystemExit(1)
print("free")
PY
}

echo "===== WAITING FOR GPUS 0-5 $(date -Is) ====="
while ! free_enough; do
  sleep "$CHECK_INTERVAL_SECONDS"
done

echo "===== STARTING KIVI FULL $(date -Is) ====="
GPU_MEMORY_BUSY_MIB="$GPU_MEMORY_BUSY_MIB" bash scripts/resume_longbench_6x3090_kivi_until_complete.sh
