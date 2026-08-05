#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/meta-llama/Llama-3.1-8B-Instruct}"
LONGBENCH_DATA_DIR="${LONGBENCH_DATA_DIR:-/root/Block-kvcache-experiment/data/LongBench}"
GSM8K_DATA_PATH="${GSM8K_DATA_PATH:-$ROOT/datasets/gsm8k/gsm8k_test.jsonl}"
V100_MANIFEST_SOURCE="${V100_MANIFEST_SOURCE:-/root/Bounded-pattrenKV-method/reports/insight_v2/wave_a_8gpu/manifest.json}"
PHYSICAL_4090_ID="${PHYSICAL_4090_ID:-}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/insight_v2/wave_a_4090_single}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT/reports/insight_v2/wave_a_4090_single}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/insight_v2/wave_a_4090_single}"
RUN_ROOT="${RUN_ROOT:-$ROOT/run/insight_v2/wave_a_4090_single}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

cd "$ROOT"
export PYTHON_BIN MODEL_PATH LONGBENCH_DATA_DIR GSM8K_DATA_PATH PHYSICAL_4090_ID V100_MANIFEST_SOURCE

if [[ -z "$PHYSICAL_4090_ID" ]]; then
  PHYSICAL_4090_ID="$("$PYTHON_BIN" - <<'PY'
import subprocess
rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"], text=True).splitlines()
targets = [line.split(",", 1)[0].strip() for line in rows if "RTX 4090" in line]
if len(targets) != 1:
    raise SystemExit(f"expected exactly one RTX 4090, found {len(targets)}")
print(targets[0])
PY
)"
fi
export PHYSICAL_4090_ID

mkdir -p "$RESULT_ROOT/generation" "$RESULT_ROOT/observer" "$REPORT_ROOT" "$LOG_ROOT" "$RUN_ROOT"
"$PYTHON_BIN" scripts/prepare_insight_wave_a_4090.py \
  --report-root "$REPORT_ROOT" \
  --model-path "$MODEL_PATH" \
  --longbench-data-dir "$LONGBENCH_DATA_DIR" \
  --gsm8k-data-path "$GSM8K_DATA_PATH" \
  --v100-manifest "$V100_MANIFEST_SOURCE" \
  >/dev/null
"$PYTHON_BIN" scripts/insight_wave_a_4090_control.py materialize --output "$RUN_ROOT/selected_samples_4090.json" >/dev/null

if ! CUDA_VISIBLE_DEVICES="$PHYSICAL_4090_ID" "$PYTHON_BIN" scripts/check_insight_wave_a_4090_gate.py --report-root "$REPORT_ROOT"; then
  echo "single-4090 Wave A blocked: gate did not pass" >&2
  exit 2
fi

if ! "$PYTHON_BIN" - "$PHYSICAL_4090_ID" <<'PY'
import subprocess, sys
idx = int(sys.argv[1])
rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True).splitlines()
used = {int(a.strip()): int(b.strip()) for a,b in (line.split(",", 1) for line in rows)}
apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL).strip()
if used.get(idx, 10**9) > 1024 or apps:
    raise SystemExit(f"target GPU is busy: memory={used.get(idx)} apps={apps!r}")
PY
then
  echo "single-4090 launch blocked by occupancy guard" >&2
  exit 5
fi

echo "hardware: RTX 4090 physical_gpu_id=$PHYSICAL_4090_ID local_cuda_device=0"
echo "gate: passed"
echo "tasks: hotpotqa=12 passage_retrieval_en=12 passage_retrieval_zh=12 samsum=12 dureader=12 gsm8k=80 total=140"
echo "model_path=$MODEL_PATH"
echo "longbench_data_dir=$LONGBENCH_DATA_DIR"
echo "gsm8k_data_path=$GSM8K_DATA_PATH"
echo "result_root=$RESULT_ROOT"
echo "runtime_commit=$(git rev-parse HEAD)"
echo "v100_reference_sha256=$(python -c 'import json; print(json.load(open("'"$REPORT_ROOT"'/reference_manifest.json"))["v100_manifest_sha256"])')"
echo "gsm8k_ids_sha256=$(python -c 'import json; print(json.load(open("'"$REPORT_ROOT"'/reference_manifest.json"))["gsm8k_problem_ids_ordered_sha256"])')"
if (( DRY_RUN )); then
  echo "dry_run=true; no model loaded and no result generated."
  exit 0
fi

echo "$$" > "$RUN_ROOT/launcher.pid"
echo "$0" > "$RUN_ROOT/launcher_script.txt"
date --iso-8601=seconds > "$RUN_ROOT/launcher_started_at.txt"
echo "running" > "$RUN_ROOT/state"

trap 'if [[ -n "${WORKER_PID:-}" ]] && kill -0 "$WORKER_PID" 2>/dev/null; then kill -TERM "$WORKER_PID" 2>/dev/null || true; fi; echo stopping > "$RUN_ROOT/state"; exit 143' TERM INT

run_task() {
  local dataset="$1" task="$2"
  local pending_file="$RUN_ROOT/pending_${task}.json"
  "$PYTHON_BIN" scripts/insight_wave_a_4090_control.py pending --dataset "$dataset" --task "$task" --result-root "$RESULT_ROOT" --output "$pending_file" >/dev/null
  local count
  count="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["count"])' "$pending_file")"
  if [[ "$count" == "0" ]]; then
    echo "skip task=$dataset/$task reason=all samples complete"
    return 0
  fi
  echo "$dataset/$task" > "$RUN_ROOT/current_task"
  "$PYTHON_BIN" - "$pending_file" "$RUN_ROOT/current_sample" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))["pending"]
value = rows[0].get("sample_id") or f"problem_id={rows[0].get('problem_id')}"
open(sys.argv[2], "w").write(str(value) + "\n")
PY
  args=(
    bench/bench_pattern_insight.py
    --dataset "$dataset"
    --tasks "$task"
    --selected-samples-json "$RUN_ROOT/selected_samples_4090.json"
    --model-path "$MODEL_PATH"
    --data-dir "$LONGBENCH_DATA_DIR"
    --gsm8k-data-path "$GSM8K_DATA_PATH"
    --output-dir "$RESULT_ROOT/generation"
    --observer-output-root "$RESULT_ROOT/observer"
    --insight-output-dir "$REPORT_ROOT/$task"
    --gpu-id 0
    --insight-level oracle
    --oracle-samples-per-head 8
    --oracle-layers 0 7 15 23 31
    --max-input-length 8192
    --skip-existing
    --resume
  )
  if [[ "$dataset" == "longbench" ]]; then
    mapfile -t ids < <("$PYTHON_BIN" -c 'import json,sys; print("\n".join(str(x["sample_id"]) for x in json.load(open(sys.argv[1]))["pending"]))' "$pending_file")
    args+=(--sample-ids "${ids[@]}")
  else
    mapfile -t ids < <("$PYTHON_BIN" -c 'import json,sys; print("\n".join(str(x["problem_id"]) for x in json.load(open(sys.argv[1]))["pending"]))' "$pending_file")
    args+=(--problem-ids "${ids[@]}" --max-new-tokens 2048)
  fi
  local task_dir="$RUN_ROOT/${dataset}_${task}"
  local log="$LOG_ROOT/${dataset}_${task}.log"
  mkdir -p "$task_dir" "$LOG_ROOT" "$REPORT_ROOT/$task"
  echo "$(date --iso-8601=seconds)" > "$task_dir/started_at.txt"
  echo "$log" > "$task_dir/log.txt"
  echo "pending_count=$count selected_id_count=${#ids[@]}" > "$task_dir/selection_check.txt"
  if [[ "${#ids[@]}" != "$count" ]]; then
    echo "$dataset/$task selection identity mismatch: pending=$count ids=${#ids[@]}" >&2
    echo failed > "$RUN_ROOT/state"
    return 6
  fi
  env CUDA_VISIBLE_DEVICES="$PHYSICAL_4090_ID" \
    PATTERNKV_INSIGHT=1 PATTERNKV_INSIGHT_LEVEL=oracle \
    PATTERNKV_INSIGHT_ORACLE_LAYERS=0,7,15,23,31 \
    PATTERNKV_INSIGHT_SAMPLE_TOKENS=8 PATTERNKV_INSIGHT_SEED=0 \
    PATTERNKV_INSIGHT_OUTPUT="$RESULT_ROOT/observer" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" "${args[@]}" > "$log" 2>&1 &
  WORKER_PID="$!"
  echo "$WORKER_PID" > "$task_dir/worker.pid"
  echo "$WORKER_PID" > "$RUN_ROOT/worker.pid"
  set +e
  wait "$WORKER_PID"
  local rc="$?"
  set -e
  echo "$rc" > "$task_dir/exit_code.txt"
  date --iso-8601=seconds > "$task_dir/finished_at.txt"
  if [[ "$rc" != "0" ]]; then
    echo "$dataset/$task failed rc=$rc" >&2
    echo failed > "$RUN_ROOT/state"
    return "$rc"
  fi
  rm -f "$RUN_ROOT/current_task" "$RUN_ROOT/current_sample" "$RUN_ROOT/worker.pid"
}

run_task longbench hotpotqa
run_task longbench passage_retrieval_en
run_task longbench passage_retrieval_zh
run_task longbench samsum
run_task longbench dureader
run_task gsm8k gsm8k

"$PYTHON_BIN" scripts/summarize_insight_wave_a_4090.py --result-root "$RESULT_ROOT" --report-root "$REPORT_ROOT"
echo completed > "$RUN_ROOT/state"
date --iso-8601=seconds > "$RUN_ROOT/launcher_finished_at.txt"
echo "single-4090 Wave A launcher finished"
