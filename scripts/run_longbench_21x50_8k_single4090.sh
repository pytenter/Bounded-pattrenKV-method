#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/Bounded-pattrenKV-method}"
cd "$PROJECT_ROOT"

export OUTPUT_TAG="${OUTPUT_TAG:-longbench_21x50_8k_4090}"
export SAMPLE_FILTER="${SAMPLE_FILTER:-0-49}"
export MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-8192}"
export RESUME="${RESUME:-1}"

if [[ "$SAMPLE_FILTER" != "0-49" ]]; then
  echo "SAMPLE_FILTER=$SAMPLE_FILTER"
  echo "This wrapper defaults to 21 tasks x 50 samples; custom SAMPLE_FILTER is allowed only when explicitly set."
fi

bash scripts/run_longbench_paper_8k_single4090.sh
