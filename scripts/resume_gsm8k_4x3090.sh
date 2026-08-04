#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Resuming GSM8K full run with --skip-existing. Completed rows will not be overwritten."
exec scripts/run_gsm8k_4x3090_full.sh
