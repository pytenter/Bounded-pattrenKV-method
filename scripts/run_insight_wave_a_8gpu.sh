#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
MODEL_PATH="${MODEL_PATH:-$HOME/modelscope_models/Llama-3.1-8B-Instruct}"
LONGBENCH_DATA_DIR="${LONGBENCH_DATA_DIR:-$ROOT/datasets/LongBench}"
GSM8K_DATA_PATH="${GSM8K_DATA_PATH:-$ROOT/datasets/gsm8k/gsm8k_test.jsonl}"
SELECTED_SAMPLES_JSON="${SELECTED_SAMPLES_JSON:-$ROOT/reports/insight_v1/v0/selected_samples.json}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/insight_v2/wave_a_8gpu}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT/reports/insight_v2/wave_a_8gpu}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/insight_v2/wave_a_8gpu}"
RUN_ROOT="${RUN_ROOT:-$ROOT/run/insight_v2/wave_a_8gpu}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
GPU_MEMORY_THRESHOLD_MIB="${GPU_MEMORY_THRESHOLD_MIB:-1024}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

cd "$ROOT"
mkdir -p "$RESULT_ROOT/generation" "$RESULT_ROOT/observer" "$REPORT_ROOT" "$LOG_ROOT" "$RUN_ROOT"

PYTHON_PREFIX="$("$PYTHON_BIN" - <<'PY'
import sys
print(sys.prefix)
PY
)"
if [[ -x "$PYTHON_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]]; then
  export CC="${CC:-$PYTHON_PREFIX/bin/x86_64-conda-linux-gnu-gcc}"
  export CXX="${CXX:-$PYTHON_PREFIX/bin/x86_64-conda-linux-gnu-g++}"
fi
if [[ -x "$PYTHON_PREFIX/bin/nvcc" ]]; then
  export CUDA_HOME="${CUDA_HOME:-$PYTHON_PREFIX}"
fi

read -r -a gpu_ids <<< "$GPU_IDS"
if [[ "${#gpu_ids[@]}" -ne 8 ]]; then
  echo "Need exactly 8 GPU ids, got: $GPU_IDS" >&2
  exit 4
fi
for gpu in "${gpu_ids[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "GPU id must be numeric, got: $gpu" >&2
    exit 4
  fi
done

export PYTHON_BIN MODEL_PATH LONGBENCH_DATA_DIR GSM8K_DATA_PATH SELECTED_SAMPLES_JSON
export RESULT_ROOT REPORT_ROOT LOG_ROOT RUN_ROOT GPU_IDS GPU_MEMORY_THRESHOLD_MIB DRY_RUN

"$PYTHON_BIN" - <<'PY'
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from insight.io import atomic_write_json, atomic_write_text

root = Path.cwd()
gpu_ids = [int(x) for x in os.environ["GPU_IDS"].split()]
selected_path = Path(os.environ["SELECTED_SAMPLES_JSON"])
payload = json.loads(selected_path.read_text(encoding="utf-8"))
gsm8k_ids = []
seen_gsm8k_ids = set()
for row in payload.get("selected", []):
    if row.get("dataset") != "gsm8k" or row.get("task") != "gsm8k":
        continue
    problem_id = int(row["problem_id"])
    if problem_id in seen_gsm8k_ids:
        continue
    seen_gsm8k_ids.add(problem_id)
    gsm8k_ids.append(problem_id)
gsm8k_shards = [gsm8k_ids[i::3] for i in range(3)]
jobs = [
    {"gpu": gpu_ids[0], "dataset": "longbench", "task": "hotpotqa", "limit": 12, "problem_ids": []},
    {"gpu": gpu_ids[1], "dataset": "longbench", "task": "passage_retrieval_en", "limit": 12, "problem_ids": []},
    {"gpu": gpu_ids[2], "dataset": "longbench", "task": "passage_retrieval_zh", "limit": 12, "problem_ids": []},
    {"gpu": gpu_ids[3], "dataset": "longbench", "task": "samsum", "limit": 12, "problem_ids": []},
    {"gpu": gpu_ids[4], "dataset": "longbench", "task": "dureader", "limit": 12, "problem_ids": []},
]
for offset, shard in enumerate(gsm8k_shards):
    jobs.append(
        {
            "gpu": gpu_ids[5 + offset],
            "dataset": "gsm8k",
            "task": "gsm8k",
            "limit": len(shard),
            "problem_ids": shard,
        }
    )

model_path = Path(os.environ["MODEL_PATH"])
index_path = model_path / "model.safetensors.index.json"
model_complete = index_path.exists()
missing_weights = []
if model_complete:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for name in sorted(set(index.get("weight_map", {}).values())):
        path = model_path / name
        if not path.exists() or path.stat().st_size == 0:
            model_complete = False
            missing_weights.append(name)
if any(model_path.glob("*.incomplete")):
    model_complete = False
    missing_weights.extend(path.name for path in model_path.glob("*.incomplete"))

longbench_required = ["hotpotqa", "passage_retrieval_en", "passage_retrieval_zh", "samsum", "dureader"]
longbench_dir = Path(os.environ["LONGBENCH_DATA_DIR"])
longbench_zip_exists = (longbench_dir / "data.zip").exists() or (longbench_dir / "LongBench.zip").exists()
readiness = {
    "python_exists": Path(os.environ["PYTHON_BIN"]).exists(),
    "model_config_exists": (model_path / "config.json").exists(),
    "model_complete": model_complete,
    "missing_model_files": sorted(set(missing_weights)),
    "selected_samples_exists": selected_path.exists(),
    "gsm8k_exists": Path(os.environ["GSM8K_DATA_PATH"]).exists(),
    "longbench_zip_exists": longbench_zip_exists,
    "longbench_files": {
        task: (longbench_dir / f"{task}.jsonl").exists() or (longbench_dir / "data" / f"{task}.jsonl").exists()
        for task in longbench_required
    },
}
readiness["ready_to_launch"] = (
    readiness["python_exists"]
    and readiness["model_config_exists"]
    and readiness["model_complete"]
    and readiness["selected_samples_exists"]
    and readiness["gsm8k_exists"]
    and (readiness["longbench_zip_exists"] or all(readiness["longbench_files"].values()))
)
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"

manifest = {
    "schema_version": "insight_v2.wave_a_8gpu_manifest",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": "dry_run" if os.environ.get("DRY_RUN") == "1" else "prepared",
    "commit": commit,
    "python_path": os.environ["PYTHON_BIN"],
    "model_path": os.environ["MODEL_PATH"],
    "longbench_data_dir": os.environ["LONGBENCH_DATA_DIR"],
    "gsm8k_data_path": os.environ["GSM8K_DATA_PATH"],
    "selected_samples_json": os.environ["SELECTED_SAMPLES_JSON"],
    "result_root": os.environ["RESULT_ROOT"],
    "report_root": os.environ["REPORT_ROOT"],
    "log_root": os.environ["LOG_ROOT"],
    "run_root": os.environ["RUN_ROOT"],
    "total_planned": sum(job["limit"] for job in jobs),
    "jobs": jobs,
    "readiness": readiness,
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
}
report_root = Path(os.environ["REPORT_ROOT"])
run_root = Path(os.environ["RUN_ROOT"])
atomic_write_json(report_root / "manifest.json", manifest)
atomic_write_json(run_root / "jobs.json", {"jobs": jobs})
lines = [
    "# Insight Wave A 8GPU Manifest",
    "",
    f"ready_to_launch: `{readiness['ready_to_launch']}`",
    f"total_planned: `{manifest['total_planned']}`",
    "",
    "| gpu | dataset | task | selected | shard |",
    "|---:|---|---|---:|---|",
]
for job in jobs:
    shard = ",".join(str(x) for x in job["problem_ids"]) if job["problem_ids"] else "-"
    lines.append(f"| {job['gpu']} | {job['dataset']} | {job['task']} | {job['limit']} | {shard} |")
atomic_write_text(report_root / "manifest.md", "\n".join(lines) + "\n")
print(json.dumps({"ready_to_launch": readiness["ready_to_launch"], "total_planned": manifest["total_planned"], "jobs": len(jobs)}, sort_keys=True))
PY

if (( DRY_RUN )); then
  echo "dry_run=true; manifest generated and no model loaded."
  exit 0
fi

if ! "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads((Path(os.environ["REPORT_ROOT"]) / "manifest.json").read_text(encoding="utf-8"))
raise SystemExit(0 if manifest["readiness"]["ready_to_launch"] else 3)
PY
then
  echo "Wave A 8GPU not launched; readiness failed. See $REPORT_ROOT/manifest.json" >&2
  exit 3
fi

if ! "$PYTHON_BIN" scripts/check_insight_wave_a_gate.py; then
  echo "Wave A 8GPU not launched; gate failed." >&2
  exit 2
fi

busy_report="$("$PYTHON_BIN" - <<'PY'
import json
import os
import subprocess

selected = {int(x) for x in os.environ["GPU_IDS"].split()}
threshold = int(os.environ["GPU_MEMORY_THRESHOLD_MIB"])
rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
busy = []
for row in rows.splitlines():
    idx_s, used_s, util_s = [x.strip() for x in row.split(",")]
    idx = int(idx_s)
    used = int(used_s)
    if idx in selected and used > threshold:
        busy.append({"gpu": idx, "memory_used_mib": used, "utilization": int(util_s)})
print(json.dumps({"busy": busy, "threshold_mib": threshold}, sort_keys=True))
PY
)"
echo "gpu_guard=$busy_report"
if [[ "$busy_report" != *'"busy": []'* ]]; then
  echo "$busy_report" > "$RUN_ROOT/gpu_guard_blocked.json"
  echo "Wave A 8GPU not launched; selected GPU occupancy guard failed." >&2
  exit 5
fi

echo "$$" > "$RUN_ROOT/launcher.pid"
date --iso-8601=seconds > "$RUN_ROOT/launcher_started_at.txt"

"$PYTHON_BIN" - <<'PY' | while IFS=$'\t' read -r gpu dataset task limit problem_ids; do
import json
import os
from pathlib import Path

payload = json.loads((Path(os.environ["RUN_ROOT"]) / "jobs.json").read_text(encoding="utf-8"))
for job in payload["jobs"]:
    print(
        "\t".join(
            [
                str(job["gpu"]),
                job["dataset"],
                job["task"],
                str(job["limit"]),
                ",".join(str(x) for x in job.get("problem_ids", [])),
            ]
        )
    )
PY
  log_dir="$LOG_ROOT/gpu${gpu}_${task}"
  task_run_dir="$RUN_ROOT/gpu${gpu}_${task}"
  report_dir="$REPORT_ROOT/${dataset}_${task}_gpu${gpu}"
  mkdir -p "$log_dir" "$task_run_dir" "$report_dir"
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
    --insight-output-dir "$report_dir"
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
    if [[ -n "$problem_ids" ]]; then
      IFS=',' read -r -a shard_ids <<< "$problem_ids"
      args+=(--problem-ids "${shard_ids[@]}")
    fi
  fi
  WAVE_GPU="$gpu" \
    PATTERNKV_INSIGHT_OUTPUT="$RESULT_ROOT/observer" \
    RUN_CURRENT="$RUN_ROOT/gpu${gpu}.current_task" \
    setsid bash -c '
    set -euo pipefail
    echo $$ > "$1"
    echo "$2/$3" > "$4"
    shift 4
    env CUDA_VISIBLE_DEVICES="$WAVE_GPU" \
      PATTERNKV_INSIGHT=1 \
      PATTERNKV_INSIGHT_LEVEL=oracle \
      PATTERNKV_INSIGHT_ORACLE_LAYERS=0,7,15,23,31 \
      PATTERNKV_INSIGHT_SAMPLE_TOKENS=8 \
      PATTERNKV_INSIGHT_SEED=0 \
      PATTERNKV_INSIGHT_OUTPUT="$PATTERNKV_INSIGHT_OUTPUT" \
      PYTHONUNBUFFERED=1 \
      "$@"
    rm -f "$RUN_CURRENT"
  ' bash "$task_run_dir/worker.pid" "$dataset" "$task" "$RUN_ROOT/gpu${gpu}.current_task" "$PYTHON_BIN" "${args[@]}" \
    > "$log_dir/run.log" 2>&1 &
  echo "$!" > "$RUN_ROOT/gpu${gpu}.queue.pid"
  echo "launched gpu=$gpu dataset=$dataset task=$task limit=$limit pid=$! log=$log_dir/run.log"
done
