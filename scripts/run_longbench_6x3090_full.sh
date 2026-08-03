#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-}"
PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-15}"
if [[ -z "$MICROMAMBA_BIN" ]]; then
  if command -v micromamba >/dev/null 2>&1; then
    MICROMAMBA_BIN="$(command -v micromamba)"
  elif [[ -x /data/zypan/kvarn-repro/tools/bin/micromamba ]]; then
    MICROMAMBA_BIN="/data/zypan/kvarn-repro/tools/bin/micromamba"
  else
    echo "micromamba not found; set MICROMAMBA_BIN=/path/to/micromamba" >&2
    exit 1
  fi
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at ${PYTHON_BIN}; set PYTHON_BIN to the patternkv env python" >&2
  exit 1
fi

mkdir -p logs/longbench/fp16 logs/longbench/patternkv results/longbench/fp16 results/longbench/patternkv run/longbench

echo "[gate] checking smoke completeness: every method/task must have at least 2 rows with non-empty predictions and empty error fields"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
tasks = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "trec", "passage_retrieval_en", "lcc"]
bad = []
for method in ["fp16", "patternkv"]:
    for task in tasks:
        path = Path("results/longbench") / method / f"{task}.jsonl"
        rows = []
        if path.exists():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        first = rows[:2]
        if len(first) < 2:
            bad.append(f"{method}/{task}: only {len(first)} smoke rows")
            continue
        if any(r.get("error") for r in first):
            bad.append(f"{method}/{task}: smoke error field present")
        if any(not str(r.get("prediction") or "").strip() for r in first):
            bad.append(f"{method}/{task}: empty smoke prediction")
if bad:
    print("SMOKE GATE FAIL")
    print("\n".join(bad))
    raise SystemExit(1)
print("SMOKE GATE PASS")
PY

echo "[precheck] nvidia-smi"
nvidia-smi
"$PYTHON_BIN" - <<'PY'
import torch
print("Visible GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
assert torch.cuda.device_count() >= 6, f"Expected 6 visible GPUs, but found {torch.cuda.device_count()}"
PY

launch_worker() {
  local gpu="$1"
  local method="$2"
  shift 2
  local tasks=("$@")
  local log="logs/longbench/${method}/gpu${gpu}_full.log"
  local pidfile="run/longbench/gpu${gpu}_${method}_full.pid"
  local metafile="run/longbench/gpu${gpu}_${method}_full.meta"
  if [[ -s "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "worker already running for gpu ${gpu}: pid $(cat "$pidfile")"
    return
  fi
  printf 'gpu=%s\nmethod=%s\nmode=full\nnum_samples=50\ntasks=%s\nlog=%s\n' "$gpu" "$method" "${tasks[*]}" "$log" > "$metafile"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" bench/bench_longbench_patternkv.py \
      --method "$method" \
      --tasks "${tasks[@]}" \
      --num-samples 50 \
      --model-path "$MODEL_PATH" \
      --gpu-id "$gpu" \
      --mode full \
      --max-input-length 8192 \
      --output-dir results/longbench \
      --status-dir run/longbench \
      --skip-existing > "$log" 2>&1 &
  echo $! > "$pidfile"
  echo "started gpu=${gpu} method=${method} pid=$(cat "$pidfile") log=${log} tasks=${tasks[*]}"
  sleep "$LAUNCH_STAGGER_SECONDS"
}

launch_worker 0 fp16 qasper gov_report
launch_worker 1 fp16 multifieldqa_en trec lcc
launch_worker 2 fp16 hotpotqa 2wikimqa passage_retrieval_en
launch_worker 3 patternkv qasper gov_report
launch_worker 4 patternkv multifieldqa_en trec lcc
launch_worker 5 patternkv hotpotqa 2wikimqa passage_retrieval_en

echo "Full workers launched. Use scripts/check_longbench_6x3090.sh to monitor."
