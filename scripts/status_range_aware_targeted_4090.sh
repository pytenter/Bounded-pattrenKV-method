#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_NAME="${TARGET_NAME:-range_aware_targeted_4090}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT/reports/insight_v2/$TARGET_NAME}"
RUN_ROOT="${RUN_ROOT:-$ROOT/run/insight_v2/$TARGET_NAME}"

echo "== Range-aware targeted 4090 status =="
if [[ -f "$REPORT_ROOT/current_status.json" ]]; then
  cat "$REPORT_ROOT/current_status.json"
else
  echo "current_status=missing"
fi

echo
echo "== PID files =="
find "$RUN_ROOT" -maxdepth 2 -type f -name '*.pid' -print 2>/dev/null | sort || true
