#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/qinch2023/v100_aime24_aime25_quality_work/Bounded-pattrenKV-method"
PYBIN="/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python"
EXP="qwen3_8b_aime24_patternkv_paper_v1"
CURRENT_RE="bench/bench_aime24_qwen3_generalization.py --worker --methods (PATTERN_BASE|CAUSAL_V4_25)"

cd "$ROOT"
export PYTHONPATH="$ROOT/vendor/transformers_4_51_runtime:$ROOT"
export TORCH_CUDA_ARCH_LIST="7.0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "run/$EXP/logs"

while pgrep -f "$CURRENT_RE" >/dev/null; do
  ts="$(date '+%F %T %z')"
  pids="$(pgrep -af "$CURRENT_RE" | awk '{print $1}' | tr '\n' ' ')"
  echo "$ts waiting_for_current_qwen3_pattern_causal pids=$pids"
  "$PYBIN" bench/bench_aime24_qwen3_generalization.py --status || true
  sleep 600
done

echo "$(date '+%F %T %z') current_qwen3_pattern_causal_finished"
"$PYBIN" bench/bench_aime24_qwen3_patternkv_paper.py

launch_worker() {
  local gpu="$1"
  case "$gpu" in 0|1|2|3) ;; *) echo "Refuse GPU $gpu"; exit 3;; esac
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYBIN" bench/bench_aime24_qwen3_patternkv_paper.py --worker \
    > "run/$EXP/logs/patternkv_paper_g${gpu}.log" 2>&1 &
  echo "$! $gpu PATTERNKV_PAPER patternkv_paper_g${gpu}"
}

{
  date '+%F %T %z'
  launch_worker 0
  launch_worker 1
  launch_worker 2
  launch_worker 3
} | tee "run/$EXP/formal_pids.txt"

echo "$(date '+%F %T %z') launched_patternkv_paper_workers"
