#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B}"
SELECTED_TASKS="${SELECTED_TASKS:-configs/aime24_wave1_selected_tasks.json}"
RESULT_DIR="${RESULT_DIR:-results/aime24_int2_wave1_v100_8gpu_wave1a4}"
REPORT_DIR="${REPORT_DIR:-reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism}"
RUN_DIR="${RUN_DIR:-run/aime24_int2_wave1_v100_8gpu_wave1a4}"
EXPERIMENT_ID="${EXPERIMENT_ID:-aime24_wave1a4_attention_mechanism}"

mkdir -p "$RESULT_DIR" "$REPORT_DIR" "$RUN_DIR"

ensure_ready() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found: $PYTHON_BIN" >&2
    return 2
  fi
  if [[ ! -r "$MODEL_PATH/config.json" ]]; then
    echo "Model not found: $MODEL_PATH" >&2
    return 2
  fi
  "$PYTHON_BIN" scripts/prepare_aime24_int2_wave1.py
}

run_fp16_reference() {
  local selected="$SELECTED_TASKS"
  local output_dir="$RESULT_DIR/fp16_reference_trajectories"
  local log="$RUN_DIR/fp16_reference.log"
  (
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    "$PYTHON_BIN" bench/bench_aime24_patternkv.py \
      --method fp16 \
      --config-name fp16_reference \
      --model-path "$MODEL_PATH" \
      --output-dir "$output_dir" \
      --status-dir "$RUN_DIR" \
      --experiment-id "$EXPERIMENT_ID" \
      --selected-tasks "$selected" \
      --max-new-tokens 32768 \
      --model-dtype float16 \
      --temperature 0.6 \
      --top-p 0.95 \
      --repetition-penalty 1.0 \
      --force-think-prefix \
      --overwrite-invalid \
      --retry-failed
  ) > "$log" 2>&1
}

case "${1:-}" in
  fp16-reference)
    ensure_ready || exit $?
    run_fp16_reference
    ;;
  fp16-reference-offline)
    ensure_ready || exit $?
    nohup setsid bash "$0" fp16-reference > "$RUN_DIR/fp16_reference.nohup.log" 2>&1 < /dev/null &
    echo "$!" > "$RUN_DIR/fp16_reference.pid"
    echo "fp16_reference_pid=$!"
    ;;
  observer-smoke)
    ensure_ready || exit $?
    "$PYTHON_BIN" scripts/run_wave1a4_attention_observer.py --mode smoke
    ;;
  fp16-manifest)
    ensure_ready || exit $?
    "$PYTHON_BIN" scripts/run_wave1a4_attention_observer.py --mode fp16-manifest
    ;;
  capture-fp16)
    ensure_ready || exit $?
    "$PYTHON_BIN" scripts/run_wave1a4_attention_observer.py --mode capture-fp16 "${@:2}"
    ;;
  capture-fp16-offline)
    ensure_ready || exit $?
    nohup setsid bash "$0" capture-fp16 "${@:2}" > "$RUN_DIR/capture_fp16.nohup.log" 2>&1 < /dev/null &
    echo "$!" > "$RUN_DIR/capture_fp16.pid"
    echo "capture_fp16_pid=$!"
    ;;
  teacher-config)
    ensure_ready || exit $?
    "$PYTHON_BIN" scripts/run_wave1a4_attention_observer.py --mode teacher-config "${@:2}"
    ;;
  teacher-config-offline)
    ensure_ready || exit $?
    config_name=""
    args=("${@:2}")
    for ((i=0; i<${#args[@]}; i++)); do
      if [[ "${args[$i]}" == "--config-name" && $((i+1)) -lt ${#args[@]} ]]; then
        config_name="${args[$((i+1))]}"
      fi
    done
    if [[ -z "$config_name" ]]; then
      echo "teacher-config-offline requires --config-name" >&2
      exit 2
    fi
    nohup setsid bash "$0" teacher-config "${args[@]}" > "$RUN_DIR/${config_name}.nohup.log" 2>&1 < /dev/null &
    echo "$!" > "$RUN_DIR/${config_name}.pid"
    echo "${config_name}_pid=$!"
    ;;
  formal-offline)
    ensure_ready || exit $?
    chmod +x scripts/run_wave1a4_formal_offline.sh
    nohup setsid bash scripts/run_wave1a4_formal_offline.sh > "$RUN_DIR/formal_offline.log" 2>&1 < /dev/null &
    echo "$!" > "$RUN_DIR/formal_offline.pid"
    echo "formal_offline_pid=$!"
    ;;
  summarize)
    "$PYTHON_BIN" scripts/summarize_wave1a4_attention_mechanism.py
    ;;
  status)
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
    echo "usage: bash scripts/run_aime24_wave1a4_attention_mechanism.sh {fp16-reference|fp16-reference-offline|fp16-manifest|observer-smoke|capture-fp16|capture-fp16-offline|teacher-config|teacher-config-offline|formal-offline|summarize|status}" >&2
    exit 2
    ;;
esac
