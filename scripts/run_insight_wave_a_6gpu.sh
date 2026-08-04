#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
LONGBENCH_DATA_DIR="${LONGBENCH_DATA_DIR:-$ROOT/datasets/LongBench}"
GSM8K_DATA_PATH="${GSM8K_DATA_PATH:-datasets/gsm8k/gsm8k_test.jsonl}"
cd "$ROOT"

mkdir -p reports/insight_v2 logs/insight_v2 run/insight_v2 results/insight_v2/generation results/insight_v2/observer

read -r -a gpu_ids <<< "${GPU_IDS:-0 1 2 3 4 5}"
if [[ "${#gpu_ids[@]}" -lt 6 ]]; then
  echo "Need 6 GPU ids, got: ${GPU_IDS:-0 1 2 3 4 5}" >&2
  exit 4
fi
export WAVE_A_GPU_IDS="${gpu_ids[*]}"

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from insight.io import atomic_write_json, atomic_write_text

gpu_ids = [int(x) for x in os.environ["WAVE_A_GPU_IDS"].split()]
jobs = [
    {"gpu": gpu_ids[0], "dataset": "longbench", "task": "hotpotqa", "limit": 12},
    {"gpu": gpu_ids[1], "dataset": "longbench", "task": "passage_retrieval_en", "limit": 12},
    {"gpu": gpu_ids[2], "dataset": "longbench", "task": "passage_retrieval_zh", "limit": 12},
    {"gpu": gpu_ids[3], "dataset": "longbench", "task": "samsum", "limit": 12},
    {"gpu": gpu_ids[4], "dataset": "longbench", "task": "dureader", "limit": 12},
    {"gpu": gpu_ids[5], "dataset": "gsm8k", "task": "gsm8k", "limit": 50},
]
payload = {
    "schema_version": "insight_v2.wave_a_manifest",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": "prepared",
    "jobs": jobs,
        "gates": {
            "quant_reference_validation": "reports/insight_v2/quant_reference_validation.json",
            "parity_report": "reports/insight_v2/parity_report.json",
            "micro_smoke_report": "reports/insight_v2/micro_smoke_report.json",
        },
}
atomic_write_json(Path("reports/insight_v2/wave_a_manifest.json"), payload)
lines = ["# Wave A Manifest", "", "| gpu | dataset | task | limit |", "|---:|---|---|---:|"]
for job in jobs:
    lines.append(f"| {job['gpu']} | {job['dataset']} | {job['task']} | {job['limit']} |")
atomic_write_text(Path("reports/insight_v2/wave_a_manifest.md"), "\n".join(lines) + "\n")
PY

if ! "$PYTHON_BIN" scripts/check_insight_wave_a_gate.py; then
  cat > run/insight_v2/wave_a.blocked <<EOF
Wave A not launched. See reports/insight_v2/wave_a_gate.json and .md.
EOF
  echo "Wave A blocked by gate."
  exit 2
fi

busy_gpus="$("$PYTHON_BIN" - <<'PY'
import os
import subprocess

selected = {int(x) for x in os.environ["WAVE_A_GPU_IDS"].split()}
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
busy = []
for line in out.strip().splitlines():
    idx_s, mem_s = [part.strip() for part in line.split(",", 1)]
    idx = int(idx_s)
    mem = int(mem_s)
    if idx in selected and mem > 1024:
        busy.append(f"{idx}:{mem}MiB")
print(" ".join(busy))
PY
)"
if [[ -n "$busy_gpus" ]]; then
  cat > run/insight_v2/wave_a.blocked <<EOF
Wave A not launched. Selected GPU(s) appear occupied: $busy_gpus
EOF
  echo "Wave A not launched; selected GPU(s) appear occupied: $busy_gpus" >&2
  exit 5
fi

declare -a datasets=(longbench longbench longbench longbench longbench gsm8k)
declare -a tasks=(hotpotqa passage_retrieval_en passage_retrieval_zh samsum dureader gsm8k)
declare -a limits=(12 12 12 12 12 50)

for i in 0 1 2 3 4 5; do
  gpu="${gpu_ids[$i]}"
  dataset="${datasets[$i]}"
  task="${tasks[$i]}"
  limit="${limits[$i]}"
  log="logs/insight_v2/wave_a_gpu${gpu}_${dataset}_${task}.log"
  pidfile="run/insight_v2/wave_a_gpu${gpu}_${dataset}_${task}.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "skip active $pidfile pid=$(cat "$pidfile")"
    continue
  fi
  CUDA_VISIBLE_DEVICES="$gpu" PATTERNKV_INSIGHT=1 PATTERNKV_INSIGHT_LEVEL=oracle \
    LONGBENCH_DATA_DIR="$LONGBENCH_DATA_DIR" \
    "$PYTHON_BIN" bench/bench_pattern_insight.py \
      --dataset "$dataset" \
      --tasks "$task" \
      --model-path "$MODEL_PATH" \
      --data-dir "$LONGBENCH_DATA_DIR" \
      --gsm8k-data-path "$GSM8K_DATA_PATH" \
      --gpu-id "$gpu" \
      --insight-level oracle \
      --oracle-samples-per-head 8 \
      --max-input-length 8192 \
      --max-new-tokens "$([[ "$dataset" = gsm8k ]] && echo 2048 || echo 1024)" \
      --limit "$limit" \
      --skip-existing \
      > "$log" 2>&1 &
  echo $! > "$pidfile"
  echo "launched gpu=$gpu dataset=$dataset task=$task limit=$limit pid=$(cat "$pidfile") log=$log"
done
