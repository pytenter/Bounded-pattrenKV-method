#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B}"
RESULT_DIR="${RESULT_DIR:-results/aime24_full30_3seed}"
RUN_DIR="${RUN_DIR:-run/aime24_full30_3seed}"
REPORT_DIR="${REPORT_DIR:-reports/aime24_full30_3seed}"
EXPERIMENT_ID="${EXPERIMENT_ID:-aime24_full30_3seed_formal}"

CONFIGS=(
  "0 fp16 fp16 16 16 0 0 0 segmented segmented_rolling"
  "1 patternkv_paper patternkv_paper 2 2 0 128 128 legacy legacy_tuple_chunked"
  "2 pattern_rolling_s0_r128 patternkv 2 2 0 128 128 segmented segmented_rolling"
  "3 pattern_rolling_s16_r128 patternkv 2 2 16 128 128 segmented segmented_rolling"
  "4 kivi_paper kivi_paper_g128 2 2 0 128 128 segmented segmented_rolling"
  "5 kivi_rolling_s0_r128 kivi_official 2 2 0 128 128 segmented segmented_rolling"
  "6 kivi_rolling_s16_r128 kivi_official 2 2 16 128 128 segmented segmented_rolling"
)

mkdir -p "$RESULT_DIR" "$RUN_DIR" "$REPORT_DIR"

ensure_ready() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found or not executable: $PYTHON_BIN" >&2
    return 2
  fi
  if [[ ! -r "$MODEL_PATH/config.json" ]]; then
    echo "Local model config not found: $MODEL_PATH/config.json" >&2
    return 2
  fi
  "$PYTHON_BIN" scripts/aime24_full30_3seed_formal.py prepare
}

launch_one() {
  local gpu="$1" config_name="$2" method="$3" k_bits="$4" v_bits="$5" sink="$6" recent="$7" residual="$8" cache_path="$9" cache_mode="${10}" selected="${11}" phase="${12}"
  local log="$RUN_DIR/${phase}.${config_name}.gpu${gpu}.log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export MODEL_PATH
    export PYTHONUNBUFFERED=1
    if [[ "$cache_mode" == "segmented_rolling" ]]; then
      export PATTERNKV_CACHE_VALIDATE=1
    else
      unset PATTERNKV_CACHE_VALIDATE
    fi
    "$PYTHON_BIN" bench/bench_aime24_patternkv.py \
      --method "$method" \
      --config-name "$config_name" \
      --model-path "$MODEL_PATH" \
      --dataset-path datasets/aime/aime24.jsonl \
      --output-dir "$RESULT_DIR" \
      --status-dir "$RUN_DIR" \
      --experiment-id "$EXPERIMENT_ID" \
      --selected-tasks "$selected" \
      --use-manifest-seed \
      --k-bits "$k_bits" \
      --v-bits "$v_bits" \
      --group-size 128 \
      --sink-length "$sink" \
      --recent-length "$recent" \
      --residual-length "$residual" \
      --max-new-tokens 32768 \
      --model-dtype float16 \
      --temperature 0.6 \
      --top-p 0.95 \
      --repetition-penalty 1.0 \
      --force-think-prefix \
      --overwrite-invalid \
      --retry-failed \
      --patternkv-cache-path "$cache_path" \
      --patternkv-cache-mode "$cache_mode"
  ) > "$log" 2>&1 &
  echo "$!" > "$RUN_DIR/${phase}.${config_name}.pid"
  echo "started phase=${phase} gpu=${gpu} config=${config_name} method=${method} pid=$! log=${log}"
}

wait_phase() {
  local phase="$1"
  local failures=0
  for spec in "${CONFIGS[@]}"; do
    read -r _ config_name _ <<<"$spec"
    local pid_file="$RUN_DIR/${phase}.${config_name}.pid"
    [[ -f "$pid_file" ]] || continue
    local pid
    pid="$(cat "$pid_file")"
    if ! wait "$pid"; then
      failures=$((failures + 1))
    fi
  done
  return "$failures"
}

case "${1:-}" in
  prepare)
    ensure_ready
    ;;
  preflight)
    ensure_ready
    selected="$RUN_DIR/preflight_selected_tasks.json"
    for spec in "${CONFIGS[@]}"; do
      read -r gpu config_name method k_bits v_bits sink recent residual cache_path cache_mode <<<"$spec"
      launch_one "$gpu" "$config_name" "$method" "$k_bits" "$v_bits" "$sink" "$recent" "$residual" "$cache_path" "$cache_mode" "$selected" preflight
      sleep 3
    done
    wait_phase preflight
    "$PYTHON_BIN" scripts/aime24_full30_3seed_formal.py preflight-check
    ;;
  formal|resume)
    bash "$0" preflight
    selected="$RUN_DIR/formal_selected_tasks.json"
    for spec in "${CONFIGS[@]}"; do
      read -r gpu config_name method k_bits v_bits sink recent residual cache_path cache_mode <<<"$spec"
      launch_one "$gpu" "$config_name" "$method" "$k_bits" "$v_bits" "$sink" "$recent" "$residual" "$cache_path" "$cache_mode" "$selected" formal
      sleep 3
    done
    wait_phase formal
    "$PYTHON_BIN" scripts/aime24_full30_3seed_formal.py aggregate
    ;;
  aggregate)
    "$PYTHON_BIN" scripts/aime24_full30_3seed_formal.py aggregate
    ;;
  status)
    "$PYTHON_BIN" scripts/aime24_full30_3seed_formal.py status
    for pidfile in "$RUN_DIR"/*.pid; do
      [[ -e "$pidfile" ]] || continue
      pid="$(cat "$pidfile")"
      if kill -0 "$pid" 2>/dev/null; then
        echo "$(basename "$pidfile"): running pid=$pid"
      else
        echo "$(basename "$pidfile"): not running pid=$pid"
      fi
    done
    ;;
  *)
    echo "usage: $0 {prepare|preflight|formal|resume|aggregate|status}" >&2
    exit 2
    ;;
esac
