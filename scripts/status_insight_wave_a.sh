#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== GPU status =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo
echo "== Insight V2 run files =="
find run/insight_v2 -maxdepth 2 -type f -print 2>/dev/null | sort || true

echo
echo "== Wave A manifest =="
if [[ -f reports/insight_v2/wave_a_manifest.json ]]; then
  cat reports/insight_v2/wave_a_manifest.json
else
  echo "missing"
fi
