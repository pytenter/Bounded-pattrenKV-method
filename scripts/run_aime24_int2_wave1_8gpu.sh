#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B}"
RESULT_DIR="${RESULT_DIR:-results/aime24_int2_wave1_v100_8gpu}"
LOG_DIR="${LOG_DIR:-run/aime24_int2_wave1_v100_8gpu}"
REPORT_DIR="${REPORT_DIR:-reports/aime24_int2_wave1_v100_8gpu}"
SELECTED_TASKS="${SELECTED_TASKS:-configs/aime24_wave1_selected_tasks.json}"
EXPERIMENT_ID="${EXPERIMENT_ID:-aime24_int2_wave1_v100_8gpu}"

CONFIGS=(
  "0 kivi_k2v2_s0_r128 kivi_official 2 2 0 128 none"
  "1 pattern_k2v2_s0_r128 patternkv 2 2 0 128 none"
  "2 kivi_k2v2_s64_r256 kivi_official 2 2 64 256 none"
  "3 pattern_k2v2_s64_r256 patternkv 2 2 64 256 none"
  "4 pattern_k4v2_s0_r128 patternkv 4 2 0 128 none"
  "5 pattern_k2v4_s0_r128 patternkv 2 4 0 128 none"
)

BLOCKED_WAVE1B_CONFIGS=(
  "6 pattern_magnitude_kmix_v2_s0_r128 patternkv 2 2 0 128 artifacts/aime24_wave1_masks/PLACEHOLDER_NOT_FOR_RESULTS_magnitude_key_int4_mask.pt blocked_wave1b"
  "7 pattern_queryaware_kmix_v2_s0_r128 patternkv 2 2 0 128 artifacts/aime24_wave1_masks/PLACEHOLDER_NOT_FOR_RESULTS_query_aware_key_int4_mask.pt blocked_wave1b"
)

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$REPORT_DIR"

ensure_ready() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found or not executable: $PYTHON_BIN" >&2
    return 2
  fi
  if [[ ! -r "$MODEL_PATH/config.json" ]]; then
    echo "Local model config not found: $MODEL_PATH/config.json" >&2
    return 2
  fi
  "$PYTHON_BIN" scripts/prepare_aime24_int2_wave1.py
}

config_hash_for() {
  local config_name="$1" method="$2" k_bits="$3" v_bits="$4" sink="$5" recent="$6" mask="$7" max_new_tokens="$8"
  "$PYTHON_BIN" - "$MODEL_PATH" "$config_name" "$method" "$k_bits" "$v_bits" "$sink" "$recent" "$mask" "$max_new_tokens" <<'PY'
import json, sys
from bench.aime24_int2_wave1 import stable_hash
model, config_name, method, k, v, sink, recent, mask, max_new = sys.argv[1:]
print(stable_hash({"dataset":"aime24","model_path":model,"config_name":config_name,"method":method,"k_bits":int(k),"v_bits":int(v),"group_size":128,"sink_length":int(sink),"recent_length":int(recent),"mask":mask,"max_new_tokens":int(max_new),"temperature":0.6,"top_p":0.95,"do_sample":True,"repetition_penalty":1.0,"batch_size":1,"force_think_prefix":True}))
PY
}

launch_one() {
  local gpu="$1" config_name="$2" method="$3" k_bits="$4" v_bits="$5" sink="$6" recent="$7" mask="$8" mode="$9"
  local max_new_tokens=32768
  local selected="$SELECTED_TASKS"
  local output_dir="$RESULT_DIR/wave1a"
  local validate_cache=0
  if [[ "$mode" == "wave1a-smoke" ]]; then
    max_new_tokens=1024
    validate_cache=1
    selected="$LOG_DIR/smoke_selected_tasks.json"
    "$PYTHON_BIN" - "$SELECTED_TASKS" "$selected" <<'PY'
import json, sys
src, dst = sys.argv[1:]
rows = json.load(open(src, encoding="utf-8"))[:2]
json.dump(rows, open(dst, "w", encoding="utf-8"), indent=2, sort_keys=True)
PY
  elif [[ "$mode" == "wave1a-long-smoke" ]]; then
    max_new_tokens=4096
    validate_cache=1
    selected="$LOG_DIR/long_smoke_selected_tasks.json"
    "$PYTHON_BIN" - "$SELECTED_TASKS" "$selected" <<'PY'
import json, sys
src, dst = sys.argv[1:]
rows = json.load(open(src, encoding="utf-8"))[:2]
json.dump(rows, open(dst, "w", encoding="utf-8"), indent=2, sort_keys=True)
PY
  fi
  local mask_hash=""
  local mixed_ratio="0"
  if [[ "$mask" != "none" ]]; then
    mixed_ratio="0.125"
    mask_hash="$("$PYTHON_BIN" - "$mask" <<'PY'
import sys, torch
from bench.aime24_int2_wave1 import mask_hash
obj = torch.load(sys.argv[1], map_location="cpu")
print(obj.get("mask_hash") or mask_hash(obj["mask"]))
PY
)"
  fi
  local cfg_hash
  cfg_hash="$(config_hash_for "$config_name" "$method" "$k_bits" "$v_bits" "$sink" "$recent" "$mask" "$max_new_tokens")"
  echo "GPU ID=$gpu config name=$config_name task count=$("$PYTHON_BIN" - <<PY
import json; print(len(json.load(open("$selected"))))
PY
) model path=$MODEL_PATH output directory=$RESULT_DIR git commit=$(git rev-parse HEAD) config hash=$cfg_hash"
  local log="$LOG_DIR/${config_name}.${mode}.log"
  local cmd=(
    "$PYTHON_BIN" bench/bench_aime24_patternkv.py
    --method "$method"
    --config-name "$config_name"
    --model-path "$MODEL_PATH"
    --output-dir "$output_dir"
    --status-dir "$LOG_DIR"
    --experiment-id "$EXPERIMENT_ID"
    --selected-tasks "$selected"
    --k-bits "$k_bits"
    --v-bits "$v_bits"
    --group-size 128
    --sink-length "$sink"
    --recent-length "$recent"
    --max-new-tokens "$max_new_tokens"
    --model-dtype float16
    --temperature 0.6
    --top-p 0.95
    --repetition-penalty 1.0
    --force-think-prefix
    --overwrite-invalid
    --retry-failed
    --mixed-key-int4-ratio "$mixed_ratio"
    --mixed-key-mask-hash "$mask_hash"
  )
  if [[ "$mask" != "none" ]]; then
    cmd+=(--mixed-key-mask-path "$mask")
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export MODEL_PATH
    if [[ "$validate_cache" == "1" ]]; then
      export PATTERNKV_CACHE_VALIDATE=1
    else
      unset PATTERNKV_CACHE_VALIDATE
    fi
    "${cmd[@]}"
  ) > "$log" 2>&1 &
  echo "$!" > "$LOG_DIR/${config_name}.${mode}.pid"
}

case "${1:-}" in
  dry-run)
    ensure_ready || exit $?
    for spec in "${CONFIGS[@]}"; do
      read -r gpu config_name method k_bits v_bits sink recent mask <<<"$spec"
      config_hash_for "$config_name" "$method" "$k_bits" "$v_bits" "$sink" "$recent" "$mask" 32768
    done
    ;;
  wave1a-smoke|wave1a-long-smoke|wave1a-full)
    mode="$1"
    if [[ "$mode" == "wave1a-full" ]]; then
      mode="wave1a-full"
    fi
    echo "Wave 1A uses 6 of 8 GPUs"
    ensure_ready || exit $?
    for spec in "${CONFIGS[@]}"; do
      read -r gpu config_name method k_bits v_bits sink recent mask <<<"$spec"
      launch_one "$gpu" "$config_name" "$method" "$k_bits" "$v_bits" "$sink" "$recent" "$mask" "$mode"
    done
    failures=0
    for spec in "${CONFIGS[@]}"; do
      read -r _ config_name _ <<<"$spec"
      pid="$(cat "$LOG_DIR/${config_name}.${mode}.pid")"
      if ! wait "$pid"; then
        failures=$((failures + 1))
      fi
    done
    exit "$failures"
    ;;
  status)
    for pidfile in "$LOG_DIR"/*.pid; do
      [[ -e "$pidfile" ]] || continue
      pid="$(cat "$pidfile")"
      if kill -0 "$pid" 2>/dev/null; then
        echo "$(basename "$pidfile"): running pid=$pid"
      else
        echo "$(basename "$pidfile"): not running pid=$pid"
      fi
    done
    ;;
  summarize-wave1a)
    "$PYTHON_BIN" scripts/summarize_aime24_int2_wave1.py --results-dir "$RESULT_DIR/wave1a" --report-dir "$REPORT_DIR/wave1a"
    ;;
  *)
    echo "usage: bash scripts/run_aime24_int2_wave1_8gpu.sh {dry-run|wave1a-smoke|wave1a-long-smoke|wave1a-full|status|summarize-wave1a}" >&2
    exit 2
    ;;
esac
