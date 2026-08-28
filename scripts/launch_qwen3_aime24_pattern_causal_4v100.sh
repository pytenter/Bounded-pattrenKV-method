#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/qinch2023/v100_aime24_aime25_quality_work/Bounded-pattrenKV-method"
PYBIN="/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python"
cd "$ROOT"
GATES="reports/qwen3_8b_aime24_native_generalization_v1/preflight_gate.json"
if ! "$PYBIN" - <<'PY'
import json, sys
from pathlib import Path
p=Path('reports/qwen3_8b_aime24_native_generalization_v1/preflight_gate.json')
g=json.loads(p.read_text())
required=['MODEL_IDENTITY_GATE','QWEN3_NATIVE_CLASS_GATE','PROMPT_GATE','LOGITS_PARITY_GATE','FIXED_SUBSET_GATE']
missing=[k for k in required if g.get(k)!='PASS']
if missing or g.get('FORMAL_CAP_SMOKE_GATE')!='PASS':
    print({'blocked': missing, 'formal_cap': g.get('FORMAL_CAP_SMOKE_GATE')})
    sys.exit(1)
PY
then
  echo "Refusing full formal launch: preflight gates are not fully PASS."
  exit 2
fi
mkdir -p run/qwen3_8b_aime24_native_generalization_v1/logs
export PYTHONPATH="$ROOT/vendor/transformers_4_51_runtime:$ROOT"
export TORCH_CUDA_ARCH_LIST="7.0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
launch_worker() {
  local gpu="$1" methods="$2" name="$3"
  case "$gpu" in 0|1|2|3) ;; *) echo "Refuse GPU $gpu"; exit 3;; esac
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYBIN" bench/bench_aime24_qwen3_generalization.py --worker --methods "$methods" \
    > "run/qwen3_8b_aime24_native_generalization_v1/logs/${name}.log" 2>&1 &
  echo "$! $gpu $methods $name"
}
{
  date '+%F %T %z'
  launch_worker 0 PATTERN_BASE qwen3_pattern_g0
  launch_worker 1 PATTERN_BASE qwen3_pattern_g1
  launch_worker 2 CAUSAL_V4_25 qwen3_causal_g2
  launch_worker 3 CAUSAL_V4_25 qwen3_causal_g3
} | tee run/qwen3_8b_aime24_native_generalization_v1/formal_pids.txt
