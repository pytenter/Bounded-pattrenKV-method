#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
RESULTS_DIR="${RESULTS_DIR:-results/longbench_official_kivi_8x50}"
STATUS_DIR="${STATUS_DIR:-run/longbench_official_kivi_8x50}"
LOG_DIR="${LOG_DIR:-logs/longbench/kivi_official_8x50}"
GPU_MEMORY_BUSY_MIB="${GPU_MEMORY_BUSY_MIB:-2000}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-12}"
GROUP_SIZE="${GROUP_SIZE:-32}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-128}"
K_BITS="${K_BITS:-2}"
V_BITS="${V_BITS:-2}"

TASKS=(qasper multifieldqa_en hotpotqa 2wikimqa gov_report trec passage_retrieval_en lcc)
GPUS=(0 1 2 3 4 5 6 7)

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$RESULTS_DIR/kivi_official"

echo "[precheck] $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

"$PYTHON_BIN" - <<PY
import subprocess
import sys

limit = int("${GPU_MEMORY_BUSY_MIB}")
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
busy = []
for line in out.splitlines():
    idx_s, used_s = [x.strip() for x in line.split(",", 1)]
    idx, used = int(idx_s), int(used_s)
    if idx in set(range(8)) and used > limit:
        busy.append((idx, used))
if busy:
    print("[precheck] Some target GPUs are busy; aborting to avoid stealing GPUs.", file=sys.stderr)
    for idx, used in busy:
        print(f"  gpu={idx} used={used} MiB > {limit} MiB", file=sys.stderr)
    raise SystemExit(2)
print(f"[precheck] GPUs 0-7 memory <= {limit} MiB; launching official KIVI LongBench full.")
PY

launch_one() {
  local gpu="$1"
  local task="$2"
  local log="$LOG_DIR/gpu${gpu}_${task}.log"
  local pidfile="$STATUS_DIR/gpu${gpu}_${task}.pid"
  local metafile="$STATUS_DIR/gpu${gpu}_${task}.meta"

  printf 'gpu=%s\nmethod=kivi_official\nmode=full\nnum_samples=50\ntask=%s\nlog=%s\n' \
    "$gpu" "$task" "$log" > "$metafile"

  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" bench/bench_longbench_patternkv.py \
    --method kivi_official \
    --tasks "$task" \
    --num-samples 50 \
    --model-path "$MODEL_PATH" \
    --gpu-id "$gpu" \
    --mode full \
    --max-input-length 8192 \
    --k-bits "$K_BITS" \
    --v-bits "$V_BITS" \
    --group-size "$GROUP_SIZE" \
    --residual-length "$RESIDUAL_LENGTH" \
    --output-dir "$RESULTS_DIR" \
    --status-dir "$STATUS_DIR" \
    --skip-existing > "$log" 2>&1 &

  echo "$!" > "$pidfile"
  echo "started gpu=$gpu task=$task pid=$(cat "$pidfile") log=$log"
}

pids=()
for i in "${!TASKS[@]}"; do
  launch_one "${GPUS[$i]}" "${TASKS[$i]}"
  pids+=("$(cat "$STATUS_DIR/gpu${GPUS[$i]}_${TASKS[$i]}.pid")")
  sleep "$LAUNCH_STAGGER_SECONDS"
done

echo "[wait] pids=${pids[*]}"
rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done

echo "[summarize] $(date -Is)"
"$PYTHON_BIN" bench/summarize_longbench.py \
  --results-dir "$RESULTS_DIR" \
  --expected-samples 50 \
  --methods kivi_official \
  --summary-name kivi_official_8x50 \
  --report-path reports/kivi_official_longbench_8x50.md \
  --require-complete || rc=1

echo "[done] rc=$rc $(date -Is)"
exit "$rc"

