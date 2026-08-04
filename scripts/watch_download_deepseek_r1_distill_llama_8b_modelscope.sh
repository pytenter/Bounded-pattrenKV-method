#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_DIR="${MODEL_DIR:-/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B}"
LOG_DIR="${LOG_DIR:-logs/model_downloads}"
RUN_DIR="${RUN_DIR:-run/model_downloads}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STALL_RESTART_SECONDS="${STALL_RESTART_SECONDS:-600}"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$MODEL_DIR"

main_log="${LOG_DIR}/deepseek_r1_distill_llama_8b_modelscope_watch_$(date +%Y%m%d_%H%M%S).log"
pid_file="${RUN_DIR}/deepseek_r1_distill_llama_8b_modelscope.pid"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$main_log"
}

trap 'rc=$?; log "watch exiting rc=${rc} line=${LINENO}"' EXIT

bytes_downloaded() {
  find "$MODEL_DIR" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.incomplete' \) -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}'
}

complete() {
  [[ -s "$MODEL_DIR/model-00001-of-000002.safetensors" && -s "$MODEL_DIR/model-00002-of-000002.safetensors" ]] || return 1
  "$PYTHON_BIN" - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
idx = json.loads((root / "model.safetensors.index.json").read_text())
expected = int(idx["metadata"]["total_size"])
actual = sum((root / name).stat().st_size for name in {"model-00001-of-000002.safetensors", "model-00002-of-000002.safetensors"})
if actual != expected:
    raise SystemExit(1)
PY
}

alive_pid() {
  local pid="${1:-}"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  return 1
}

start_download() {
  local log_file="${LOG_DIR}/deepseek_r1_distill_llama_8b_modelscope_$(date +%Y%m%d_%H%M%S).log"
  nohup "$PYTHON_BIN" scripts/download_deepseek_r1_distill_llama_8b_modelscope.py > "$log_file" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$pid_file"
  log "started ModelScope download pid=${pid} log=${log_file}"
}

log "watch start model_dir=${MODEL_DIR} poll=${POLL_SECONDS}s stall_restart=${STALL_RESTART_SECONDS}s"

last_bytes="$(bytes_downloaded)"
last_growth_ts="$(date +%s)"

while true; do
  if complete; then
    log "download complete bytes=$(bytes_downloaded)"
    exit 0
  fi

  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if ! alive_pid "$pid"; then
    log "download process not alive; current_bytes=$(bytes_downloaded); restarting"
    start_download
    pid="$(cat "$pid_file")"
  fi

  sleep "$POLL_SECONDS"

  now_bytes="$(bytes_downloaded)"
  now_ts="$(date +%s)"
  if (( now_bytes > last_bytes )); then
    delta=$((now_bytes - last_bytes))
    log "progress pid=${pid} bytes=${now_bytes} delta=${delta}"
    last_bytes="$now_bytes"
    last_growth_ts="$now_ts"
  else
    stagnant=$((now_ts - last_growth_ts))
    log "no growth pid=${pid} bytes=${now_bytes} stagnant=${stagnant}s"
    if (( stagnant >= STALL_RESTART_SECONDS )); then
      if alive_pid "$pid"; then
        log "stalled for ${stagnant}s; terminating pid=${pid}"
        kill "$pid" || true
        sleep 10
        if alive_pid "$pid"; then
          kill -KILL "$pid" || true
        fi
      fi
      start_download
      last_growth_ts="$(date +%s)"
    fi
  fi
done
