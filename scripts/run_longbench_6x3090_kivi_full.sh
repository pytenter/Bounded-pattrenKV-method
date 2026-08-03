#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-20}"
GPU_MEMORY_BUSY_MIB="${GPU_MEMORY_BUSY_MIB:-2000}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at ${PYTHON_BIN}; set PYTHON_BIN to the patternkv env python" >&2
  exit 1
fi

mkdir -p logs/longbench/kivi results/longbench/kivi run/longbench

echo "[precheck] nvidia-smi"
nvidia-smi
"$PYTHON_BIN" - <<'PY'
import torch
print("Visible GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
assert torch.cuda.device_count() >= 6, f"Expected 6 visible GPUs, but found {torch.cuda.device_count()}"
PY

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
    print("[precheck] GPUs 0-5 are not free enough for KIVI:", file=sys.stderr)
    for idx, used in busy:
        print(f"  gpu={idx} used={used} MiB > {limit} MiB", file=sys.stderr)
    raise SystemExit(2)
print(f"[precheck] GPUs 0-5 memory <= {limit} MiB; launching KIVI")
PY

launch_worker() {
  local gpu="$1"
  shift
  local tasks=("$@")
  local log="logs/longbench/kivi/gpu${gpu}_full.log"
  local pidfile="run/longbench/gpu${gpu}_kivi_full.pid"
  local metafile="run/longbench/gpu${gpu}_kivi_full.meta"
  if [[ -s "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "worker already running for gpu ${gpu}: pid $(cat "$pidfile")"
    return
  fi
  printf 'gpu=%s\nmethod=kivi\nmode=full\nnum_samples=50\ntasks=%s\nlog=%s\n' "$gpu" "${tasks[*]}" "$log" > "$metafile"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" bench/bench_longbench_patternkv.py \
      --method kivi \
      --tasks "${tasks[@]}" \
      --num-samples 50 \
      --model-path "$MODEL_PATH" \
      --gpu-id "$gpu" \
      --mode full \
      --max-input-length 8192 \
      --k-bits 2 \
      --v-bits 2 \
      --group-size 128 \
      --residual-length 128 \
      --output-dir results/longbench \
      --status-dir run/longbench \
      --skip-existing > "$log" 2>&1 &
  echo $! > "$pidfile"
  echo "started gpu=${gpu} method=kivi pid=$(cat "$pidfile") log=${log} tasks=${tasks[*]}"
  sleep "$LAUNCH_STAGGER_SECONDS"
}

launch_worker 0 qasper gov_report
launch_worker 1 multifieldqa_en
launch_worker 2 hotpotqa
launch_worker 3 2wikimqa trec
launch_worker 4 passage_retrieval_en
launch_worker 5 lcc

echo "KIVI full workers launched. Use scripts/check_longbench_6x3090.sh to monitor."
