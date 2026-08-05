#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
LONGBENCH_DATA_DIR="${LONGBENCH_DATA_DIR:-/data/zypan/PatternKV-repro/datasets/LongBench}"
GSM8K_DATA_PATH="${GSM8K_DATA_PATH:-/data/zypan/PatternKV-repro/datasets/gsm8k/gsm8k_test.jsonl}"
WAVE_A_GPU_IDS="${WAVE_A_GPU_IDS:-4 5 6 7}"
SELECTED_SAMPLES_JSON="${SELECTED_SAMPLES_JSON:-reports/insight_v1/v0/selected_samples.json}"
RESULT_ROOT="${RESULT_ROOT:-results/insight_v2/wave_a}"
REPORT_ROOT="${REPORT_ROOT:-reports/insight_v2/wave_a}"
LOG_ROOT="${LOG_ROOT:-logs/insight_v2/wave_a}"
RUN_ROOT="${RUN_ROOT:-run/insight_v2/wave_a}"
GPU_MEMORY_THRESHOLD_MIB="${GPU_MEMORY_THRESHOLD_MIB:-1024}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

cd "$ROOT"
mkdir -p "$RESULT_ROOT/generation" "$RESULT_ROOT/observer" "$REPORT_ROOT" "$LOG_ROOT" "$RUN_ROOT"

read -r -a gpu_ids <<< "$WAVE_A_GPU_IDS"
if [[ "${#gpu_ids[@]}" -ne 4 ]]; then
  echo "Need exactly 4 GPU ids for Wave A, got: $WAVE_A_GPU_IDS" >&2
  exit 4
fi
for gpu in "${gpu_ids[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]] || (( gpu < 4 || gpu > 7 )); then
    echo "Wave A may only use GPU4-7; got GPU $gpu" >&2
    exit 4
  fi
done

export WAVE_A_GPU_IDS PYTHON_BIN MODEL_PATH LONGBENCH_DATA_DIR GSM8K_DATA_PATH
export SELECTED_SAMPLES_JSON RESULT_ROOT REPORT_ROOT LOG_ROOT RUN_ROOT GPU_MEMORY_THRESHOLD_MIB
export DRY_RUN

"$PYTHON_BIN" - <<'PY'
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from insight.io import atomic_write_json, atomic_write_text

gpu_ids = [int(x) for x in os.environ["WAVE_A_GPU_IDS"].split()]
queues = [
    {"gpu": gpu_ids[0], "jobs": [{"dataset": "longbench", "task": "hotpotqa", "limit": 12}, {"dataset": "longbench", "task": "samsum", "limit": 12}]},
    {"gpu": gpu_ids[1], "jobs": [{"dataset": "longbench", "task": "passage_retrieval_en", "limit": 12}, {"dataset": "longbench", "task": "passage_retrieval_zh", "limit": 12}]},
    {"gpu": gpu_ids[2], "jobs": [{"dataset": "longbench", "task": "dureader", "limit": 12}]},
    {"gpu": gpu_ids[3], "jobs": [{"dataset": "gsm8k", "task": "gsm8k", "limit": 50}]},
]
payload = {
    "schema_version": "insight_v2.wave_a_4gpu_manifest",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": "dry_run" if os.environ.get("DRY_RUN") == "1" else "prepared",
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "python_path": os.environ["PYTHON_BIN"],
    "model_path": os.environ["MODEL_PATH"],
    "longbench_data_dir": os.environ["LONGBENCH_DATA_DIR"],
    "gsm8k_data_path": os.environ["GSM8K_DATA_PATH"],
    "selected_samples_json": os.environ["SELECTED_SAMPLES_JSON"],
    "result_root": os.environ["RESULT_ROOT"],
    "report_root": os.environ["REPORT_ROOT"],
    "log_root": os.environ["LOG_ROOT"],
    "run_root": os.environ["RUN_ROOT"],
    "total_planned": 110,
    "patternkv_config": {
        "method": "patternkv_paper",
        "k_bits": 2,
        "v_bits": 2,
        "group_size": 128,
        "residual_length": 128,
        "num_k_base": 32,
        "num_v_base": 32,
        "pattern_group": 128,
        "pattern_position": "post-RoPE",
        "key_axis": "per-channel",
        "value_axis": "per-token",
        "asymmetric": True,
    },
    "insight_config": {
        "PATTERNKV_INSIGHT": "1",
        "PATTERNKV_INSIGHT_LEVEL": "oracle",
        "PATTERNKV_INSIGHT_ORACLE_LAYERS": "0,7,15,23,31",
        "PATTERNKV_INSIGHT_SAMPLE_TOKENS": "8",
        "PATTERNKV_INSIGHT_SEED": "0",
    },
    "generation_config": {
        "batch_size": 1,
        "do_sample": False,
        "longbench_max_input": 8192,
        "longbench_max_gen": "task-specific MAX_NEW_TOKENS from LongBench runner",
        "gsm8k_max_new_tokens": 2048,
    },
    "queues": queues,
    "gates": {
        "quant_reference_validation": "reports/insight_v2/quant_reference_validation.json",
        "parity_report": "reports/insight_v2/parity_report.json",
        "micro_smoke_report": "reports/insight_v2/micro_smoke_report.json",
        "wave_a_gate": "reports/insight_v2/wave_a_gate.json",
    },
}
atomic_write_json(Path(os.environ["REPORT_ROOT"]) / "manifest.json", payload)
lines = ["# Insight Wave A 4GPU Manifest", "", f"total_planned: `{payload['total_planned']}`", "", "| gpu | queue | selected |", "|---:|---|---:|"]
for q in queues:
    lines.append(f"| {q['gpu']} | {' -> '.join(j['task'] for j in q['jobs'])} | {sum(j['limit'] for j in q['jobs'])} |")
atomic_write_text(Path(os.environ["REPORT_ROOT"]) / "manifest.md", "\n".join(lines) + "\n")
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
PY

echo "== Wave A 4GPU dry-run plan =="
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads((Path(os.environ["REPORT_ROOT"]) / "manifest.json").read_text())
for q in payload["queues"]:
    print(f"GPU{q['gpu']}:")
    print("  " + " -> ".join(j["task"] for j in q["jobs"]))
    for job in q["jobs"]:
        print(f"    {job['task']}: selected={job['limit']} dataset={job['dataset']}")
print("data:")
print(f"  LongBench={payload['longbench_data_dir']}")
print(f"  GSM8K={payload['gsm8k_data_path']}")
print(f"model={payload['model_path']}")
print(f"results={payload['result_root']}")
print(f"PatternKV={payload['patternkv_config']}")
print(f"Insight={payload['insight_config']}")
print(f"Generation={payload['generation_config']}")
PY

test -x "$PYTHON_BIN"
test -r "$MODEL_PATH/config.json"
test -r "$LONGBENCH_DATA_DIR/data/hotpotqa.jsonl"
test -r "$LONGBENCH_DATA_DIR/data/samsum.jsonl"
test -r "$LONGBENCH_DATA_DIR/data/dureader.jsonl"
test -r "$LONGBENCH_DATA_DIR/data/passage_retrieval_en.jsonl"
test -r "$LONGBENCH_DATA_DIR/data/passage_retrieval_zh.jsonl"
test -r "$GSM8K_DATA_PATH"
test -r "$SELECTED_SAMPLES_JSON"

if ! "$PYTHON_BIN" scripts/check_insight_wave_a_gate.py; then
  cat > "$RUN_ROOT/wave_a.blocked" <<EOF
Wave A not launched. See reports/insight_v2/wave_a_gate.json and .md.
EOF
  echo "Wave A blocked by gate." >&2
  exit 2
fi
echo "gate_status=passed"

busy_report="$("$PYTHON_BIN" - <<'PY'
import os
import json
import subprocess

selected = {int(x) for x in os.environ["WAVE_A_GPU_IDS"].split()}
threshold = int(os.environ["GPU_MEMORY_THRESHOLD_MIB"])
uuid_rows = subprocess.check_output(["nvidia-smi", "-L"], text=True)
uuid_to_index = {}
for line in uuid_rows.splitlines():
    if not line.startswith("GPU "):
        continue
    idx = int(line.split(":", 1)[0].split()[1])
    uuid = line.rsplit("UUID: ", 1)[1].rstrip(")")
    uuid_to_index[uuid] = idx
mem_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
mem = {}
for line in mem_rows.splitlines():
    idx_s, used_s, util_s = [x.strip() for x in line.split(",")]
    idx = int(idx_s)
    if idx in selected:
        mem[idx] = {"used": int(used_s), "util": int(util_s)}
apps_rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL)
apps = {idx: [] for idx in selected}
for line in apps_rows.splitlines():
    if not line.strip():
        continue
    uuid, pid, proc, used = [x.strip() for x in line.split(",", 3)]
    idx = uuid_to_index.get(uuid)
    if idx in selected:
        apps[idx].append({"pid": pid, "process_name": proc, "used_memory": used})
busy = []
for idx in sorted(selected):
    if mem.get(idx, {}).get("used", 0) > threshold or apps.get(idx):
        busy.append({"gpu": idx, "memory_used_mib": mem.get(idx, {}).get("used"), "utilization": mem.get(idx, {}).get("util"), "compute_apps": apps.get(idx, [])})
print(json.dumps({"busy": busy, "memory": mem, "threshold_mib": threshold}, sort_keys=True))
PY
)"
echo "gpu_guard=$busy_report"
if [[ "$busy_report" != *'"busy": []'* ]]; then
  echo "$busy_report" > "$RUN_ROOT/gpu_guard_blocked.json"
  echo "Wave A not launched; GPU4-7 occupancy guard failed." >&2
  exit 5
fi

if (( DRY_RUN )); then
  echo "dry_run=true; no model loaded and no result generated."
  exit 0
fi

echo "$$" > "$RUN_ROOT/launcher.pid"
date --iso-8601=seconds > "$RUN_ROOT/launcher_started_at.txt"

run_job() {
  local gpu="$1"
  local dataset="$2"
  local task="$3"
  local limit="$4"
  local log_dir="$LOG_ROOT/gpu${gpu}_${task}"
  local task_run_dir="$RUN_ROOT/gpu${gpu}_${task}"
  mkdir -p "$log_dir" "$task_run_dir" "$RESULT_ROOT/generation" "$RESULT_ROOT/observer"
  echo "$dataset/$task" > "$RUN_ROOT/gpu${gpu}.current_task"
  echo "$(date --iso-8601=seconds)" > "$task_run_dir/started_at.txt"
  local args=(
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
    "$PYTHON_BIN" "${args[@]}" > "$log_dir/run.log" 2>&1 &
  local pid="$!"
  echo "$pid" > "$task_run_dir/worker.pid"
  wait "$pid"
  local rc="$?"
  echo "$rc" > "$task_run_dir/exit_code.txt"
  echo "$(date --iso-8601=seconds)" > "$task_run_dir/finished_at.txt"
  return "$rc"
}

run_queue() {
  local gpu="$1"
  shift
  echo "$$" > "$RUN_ROOT/gpu${gpu}.queue.pid"
  trap 'if [[ -n "${child_pid:-}" ]] && kill -0 "$child_pid" 2>/dev/null; then kill -TERM "$child_pid"; fi; exit 143' TERM INT
  local rc=0
  while (( "$#" )); do
    local dataset="$1"
    local task="$2"
    local limit="$3"
    shift 3
    run_job "$gpu" "$dataset" "$task" "$limit" || rc=1
  done
  rm -f "$RUN_ROOT/gpu${gpu}.current_task"
  exit "$rc"
}

run_queue "${gpu_ids[0]}" longbench hotpotqa 12 longbench samsum 12 > "$LOG_ROOT/gpu${gpu_ids[0]}_queue.log" 2>&1 &
echo "$!" > "$RUN_ROOT/gpu${gpu_ids[0]}.queue.pid"
run_queue "${gpu_ids[1]}" longbench passage_retrieval_en 12 longbench passage_retrieval_zh 12 > "$LOG_ROOT/gpu${gpu_ids[1]}_queue.log" 2>&1 &
echo "$!" > "$RUN_ROOT/gpu${gpu_ids[1]}.queue.pid"
run_queue "${gpu_ids[2]}" longbench dureader 12 > "$LOG_ROOT/gpu${gpu_ids[2]}_queue.log" 2>&1 &
echo "$!" > "$RUN_ROOT/gpu${gpu_ids[2]}.queue.pid"
run_queue "${gpu_ids[3]}" gsm8k gsm8k 50 > "$LOG_ROOT/gpu${gpu_ids[3]}_queue.log" 2>&1 &
echo "$!" > "$RUN_ROOT/gpu${gpu_ids[3]}.queue.pid"

wait
