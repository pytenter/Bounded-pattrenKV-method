#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
TARGET_NAME="${TARGET_NAME:-range_aware_targeted_4090}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/insight_v2/$TARGET_NAME}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT/reports/insight_v2/$TARGET_NAME}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/insight_v2/$TARGET_NAME}"
RUN_ROOT="${RUN_ROOT:-$ROOT/run/insight_v2/$TARGET_NAME}"
REFERENCE_MANIFEST="${REFERENCE_MANIFEST:-$ROOT/reports/insight_v2/wave_a_4090_single/reference_manifest.json}"
if [[ ! -f "$REFERENCE_MANIFEST" ]] && [[ -f "/tmp/patternkv-insight-wave-a-4090-runtime6c88/reports/insight_v2/wave_a_4090_single/reference_manifest.json" ]]; then
  REFERENCE_MANIFEST="/tmp/patternkv-insight-wave-a-4090-runtime6c88/reports/insight_v2/wave_a_4090_single/reference_manifest.json"
fi
SELECTED_JSON="$REPORT_ROOT/selected_25_samples.json"
PARITY_STATUS_JSON="$REPORT_ROOT/parity_status.json"
CURRENT_STATUS_JSON="$REPORT_ROOT/current_status.json"
MODEL_PATH="${MODEL_PATH:-}"
LONGBENCH_DATA_DIR="${LONGBENCH_DATA_DIR:-}"
GSM8K_DATA_PATH="${GSM8K_DATA_PATH:-$ROOT/datasets/gsm8k/gsm8k_test.jsonl}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_range_aware_targeted_4090.sh --dry-run
  bash scripts/run_range_aware_targeted_4090.sh --parity-only
  bash scripts/run_range_aware_targeted_4090.sh --run-targeted

Options:
  --dry-run       Print manifest and readiness without starting GPU
  --parity-only   Reserved for explicit parity execution
  --run-targeted  Reserved for explicit targeted execution; requires parity passed
  --resume        Accepted for future execution plumbing
  --sample-id ID  Accepted for future execution plumbing
  --task TASK     Accepted for future execution plumbing
EOF
}

mkdir -p "$REPORT_ROOT" "$RUN_ROOT" "$RESULT_ROOT" "$LOG_ROOT"

mode=""
resume="false"
sample_id=""
task_name=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--parity-only|--run-targeted) mode="$1"; shift ;;
    --resume) resume="true"; shift ;;
    --sample-id) sample_id="${2:-}"; shift 2 ;;
    --task) task_name="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$mode" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$SELECTED_JSON" ]] && [[ -f "$REFERENCE_MANIFEST" ]]; then
  "$PYTHON_BIN" "$ROOT/scripts/select_range_aware_targeted_samples_4090.py" \
    --reference-manifest "$REFERENCE_MANIFEST" \
    --output-root "$REPORT_ROOT"
fi

if [[ "$mode" == "--run-targeted" ]]; then
  if [[ ! -f "$PARITY_STATUS_JSON" ]]; then
    echo "Refusing --run-targeted: parity status missing."
    exit 3
  fi
  parity_status="$("$PYTHON_BIN" - <<'PY' "$PARITY_STATUS_JSON"
import json, sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read()).get("parity_status", "not_run"))
PY
)"
  if [[ "$parity_status" != "passed" ]]; then
    echo "Refusing --run-targeted: parity_status=$parity_status"
    exit 3
  fi
fi

if [[ "$mode" == "--parity-only" || "$mode" == "--run-targeted" ]]; then
  echo "GPU launch is intentionally not performed in this B1 preparation step."
  exit 4
fi

selected_sha="$("$PYTHON_BIN" - <<'PY' "$SELECTED_JSON"
import hashlib, sys
text = open(sys.argv[1], encoding="utf-8").read()
print(hashlib.sha256(text.encode("utf-8")).hexdigest())
PY
)"
task_counts="$("$PYTHON_BIN" - <<'PY' "$SELECTED_JSON"
import json, sys
from collections import Counter
rows = json.loads(open(sys.argv[1], encoding="utf-8").read())["selected"]
c = Counter(row["task"] for row in rows)
print(", ".join(f"{k}={c[k]}" for k in sorted(c)))
PY
)"

branch="$(git -C "$ROOT" branch --show-current)"
commit="$(git -C "$ROOT" rev-parse HEAD)"

cat <<EOF
branch: $branch
git commit: $commit
target name: $TARGET_NAME
selected manifest path: $SELECTED_JSON
selected manifest SHA256: $selected_sha
task counts: $task_counts
total samples=25
parity samples=3
model path: ${MODEL_PATH:-missing}
LongBench data path: ${LONGBENCH_DATA_DIR:-missing}
GSM8K data path: ${GSM8K_DATA_PATH:-missing}
result root: $RESULT_ROOT
report root: $REPORT_ROOT
log root: $LOG_ROOT
run root: $RUN_ROOT
observer schema: insight_v2.range_aware_aggregate_v1
sample records disabled: true
K axis description: [B,H,T,D] post-RoPE, transpose to [B,H,D,T], 128-token groups, per-channel range
V axis description: [B,H,T,D] per-token over head_dim=128
expected layers: 32
expected KV heads: 8
current parity status: not_run
GPU will start: no
EOF

cat > "$CURRENT_STATUS_JSON" <<EOF
{
  "implementation_ready": true,
  "cpu_tests_passed": false,
  "dry_run_passed": true,
  "parity_status": "not_run",
  "targeted_status": "not_run",
  "gpu_started": false
}
EOF
