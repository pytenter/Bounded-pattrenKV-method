#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
LONGBENCH_DATA_DIR="${LONGBENCH_DATA_DIR:-/data/zypan/PatternKV-repro/datasets/LongBench}"
GSM8K_DATA_PATH="${GSM8K_DATA_PATH:-/data/zypan/PatternKV-repro/datasets/gsm8k/gsm8k_test.jsonl}"
WAVE_A_ALLOWED_GPU_IDS="${WAVE_A_ALLOWED_GPU_IDS:-4 5 6 7}"
RESULT_ROOT="${RESULT_ROOT:-results/insight_v2/wave_a}"
REPORT_ROOT="${REPORT_ROOT:-reports/insight_v2/wave_a}"
LOG_ROOT="${LOG_ROOT:-logs/insight_v2/wave_a}"
RUN_ROOT="${RUN_ROOT:-run/insight_v2/wave_a}"
SELECTED_SAMPLES_JSON="${SELECTED_SAMPLES_JSON:-reports/insight_v1/v0/selected_samples.json}"
GPU_MEMORY_THRESHOLD_MIB="${GPU_MEMORY_THRESHOLD_MIB:-1024}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

cd "$ROOT"
mkdir -p "$RESULT_ROOT/generation" "$RESULT_ROOT/observer" "$REPORT_ROOT" "$LOG_ROOT" "$RUN_ROOT"
export WAVE_A_ALLOWED_GPU_IDS GPU_MEMORY_THRESHOLD_MIB RESULT_ROOT REPORT_ROOT LOG_ROOT RUN_ROOT

for gpu in $WAVE_A_ALLOWED_GPU_IDS; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]] || (( gpu < 4 || gpu > 7 )); then
    echo "Idle Wave A launcher is restricted to GPU4-7; got GPU $gpu" >&2
    exit 4
  fi
done

if ! "$PYTHON_BIN" scripts/check_insight_wave_a_gate.py; then
  echo "Wave A idle launch blocked by gate." >&2
  exit 2
fi

idle_gpus="$("$PYTHON_BIN" - <<'PY'
import os
import subprocess

allowed = [int(x) for x in os.environ["WAVE_A_ALLOWED_GPU_IDS"].split()]
threshold = int(os.environ["GPU_MEMORY_THRESHOLD_MIB"])
uuid_rows = subprocess.check_output(["nvidia-smi", "-L"], text=True)
uuid_to_index = {}
for line in uuid_rows.splitlines():
    if line.startswith("GPU "):
        idx = int(line.split(":", 1)[0].split()[1])
        uuid = line.rsplit("UUID: ", 1)[1].rstrip(")")
        uuid_to_index[uuid] = idx
mem_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
memory = {}
for line in mem_rows.splitlines():
    idx_s, used_s = [x.strip() for x in line.split(",", 1)]
    memory[int(idx_s)] = int(used_s)
apps_rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL)
busy = set()
for line in apps_rows.splitlines():
    if not line.strip():
        continue
    uuid = line.split(",", 1)[0].strip()
    idx = uuid_to_index.get(uuid)
    if idx in allowed:
        busy.add(idx)
idle = [idx for idx in allowed if memory.get(idx, 0) <= threshold and idx not in busy]
print(" ".join(str(x) for x in idle))
PY
)"

if [[ -z "$idle_gpus" ]]; then
  echo "No idle Wave A GPU found in: $WAVE_A_ALLOWED_GPU_IDS"
  if (( DRY_RUN )); then
    echo "dry_run=true; no model loaded and no result generated."
    exit 0
  fi
  exit 5
fi

queue_specs=(
  "retrieval|longbench:passage_retrieval_en:12,longbench:passage_retrieval_zh:12"
  "hotpot_samsum|longbench:hotpotqa:12,longbench:samsum:12"
  "dureader|longbench:dureader:12"
  "gsm8k|gsm8k:gsm8k:50"
)

echo "idle_gpus=$idle_gpus"
echo "queue_order=${queue_specs[*]}"

if (( DRY_RUN )); then
  echo "dry_run=true; no model loaded and no result generated."
  exit 0
fi

launch_queue() {
  local gpu="$1"
  local queue_name="$2"
  local spec="$3"
  local queue_log="$LOG_ROOT/gpu${gpu}_${queue_name}_queue.log"
  setsid bash -c '
    set -euo pipefail
    cd "$1"
    gpu="$2"
    queue_name="$3"
    spec="$4"
    PYTHON_BIN="$5"
    MODEL_PATH="$6"
    LONGBENCH_DATA_DIR="$7"
    GSM8K_DATA_PATH="$8"
    RESULT_ROOT="$9"
    REPORT_ROOT="${10}"
    LOG_ROOT="${11}"
    RUN_ROOT="${12}"
    SELECTED_SAMPLES_JSON="${13}"
    echo $$ > "$RUN_ROOT/gpu${gpu}.queue.pid"
    trap "exit 143" TERM INT
    IFS="," read -r -a jobs <<< "$spec"
    rc=0
    for job in "${jobs[@]}"; do
      IFS=":" read -r dataset task limit <<< "$job"
      echo "$dataset/$task" > "$RUN_ROOT/gpu${gpu}.current_task"
      task_run="$RUN_ROOT/gpu${gpu}_${task}"
      task_log_dir="$LOG_ROOT/gpu${gpu}_${task}"
      mkdir -p "$task_run" "$task_log_dir" "$REPORT_ROOT/$task"
      date --iso-8601=seconds > "$task_run/started_at.txt"
      args=(
        bench/bench_pattern_insight.py
        --dataset "$dataset"
        --tasks "$task"
        --selected-samples-json "$SELECTED_SAMPLES_JSON"
        --model-path "$MODEL_PATH"
        --data-dir "$LONGBENCH_DATA_DIR"
        --gsm8k-data-path "$GSM8K_DATA_PATH"
        --output-dir "$RESULT_ROOT/generation"
        --observer-output-root "$RESULT_ROOT/observer"
        --insight-output-dir "$REPORT_ROOT/$task"
        --gpu-id "$gpu"
        --insight-level oracle
        --oracle-samples-per-head 8
        --oracle-layers 0 7 15 23 31
        --max-input-length 8192
        --limit "$limit"
        --skip-existing
        --resume
      )
      if [[ "$dataset" == "gsm8k" ]]; then
        args+=(--max-new-tokens 2048)
      fi
      env CUDA_VISIBLE_DEVICES="$gpu" \
        PATTERNKV_INSIGHT=1 \
        PATTERNKV_INSIGHT_LEVEL=oracle \
        PATTERNKV_INSIGHT_ORACLE_LAYERS=0,7,15,23,31 \
        PATTERNKV_INSIGHT_SAMPLE_TOKENS=8 \
        PATTERNKV_INSIGHT_SEED=0 \
        PATTERNKV_INSIGHT_OUTPUT="$RESULT_ROOT/observer" \
        PYTHONUNBUFFERED=1 \
        "$PYTHON_BIN" "${args[@]}" >> "$task_log_dir/run.log" 2>&1 &
      child="$!"
      echo "$child" > "$task_run/worker.pid"
      wait "$child" || rc=1
      echo "$rc" > "$task_run/exit_code.txt"
      date --iso-8601=seconds > "$task_run/finished_at.txt"
    done
    rm -f "$RUN_ROOT/gpu${gpu}.current_task"
    exit "$rc"
  ' bash "$ROOT" "$gpu" "$queue_name" "$spec" "$PYTHON_BIN" "$MODEL_PATH" "$LONGBENCH_DATA_DIR" "$GSM8K_DATA_PATH" "$RESULT_ROOT" "$REPORT_ROOT" "$LOG_ROOT" "$RUN_ROOT" "$SELECTED_SAMPLES_JSON" > "$queue_log" 2>&1 &
  echo "launched gpu=$gpu queue=$queue_name pid=$! log=$queue_log"
}

read -r -a idle <<< "$idle_gpus"
launched=0
for idx in "${!idle[@]}"; do
  [[ "$idx" -lt "${#queue_specs[@]}" ]] || break
  queue_name="${queue_specs[$idx]%%|*}"
  spec="${queue_specs[$idx]#*|}"
  launch_queue "${idle[$idx]}" "$queue_name" "$spec"
  launched=$((launched + 1))
done

echo "launched_queues=$launched"
