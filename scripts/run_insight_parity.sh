#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
cd "$(dirname "$0")/.."
"$PYTHON_BIN" scripts/run_insight_parity.py
