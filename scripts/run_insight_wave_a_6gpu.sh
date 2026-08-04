#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
cd "$ROOT"

mkdir -p reports/insight_v2 logs/insight_v2 run/insight_v2 results/insight_v2/generation results/insight_v2/observer

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
from insight.io import atomic_write_json, atomic_write_text

jobs = [
    {"gpu": 0, "dataset": "longbench", "task": "hotpotqa", "limit": 12},
    {"gpu": 1, "dataset": "longbench", "task": "passage_retrieval_en", "limit": 12},
    {"gpu": 2, "dataset": "longbench", "task": "passage_retrieval_zh", "limit": 12},
    {"gpu": 3, "dataset": "longbench", "task": "samsum", "limit": 12},
    {"gpu": 4, "dataset": "longbench", "task": "dureader", "limit": 12},
    {"gpu": 5, "dataset": "gsm8k", "task": "gsm8k", "limit": 50},
]
payload = {
    "schema_version": "insight_v2.wave_a_manifest",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": "prepared",
    "jobs": jobs,
    "gates": {
        "quant_reference_validation": "reports/insight_v2/quant_reference_validation.json",
        "parity_report": "reports/insight_v2/parity_report.json",
    },
}
atomic_write_json(Path("reports/insight_v2/wave_a_manifest.json"), payload)
lines = ["# Wave A Manifest", "", "| gpu | dataset | task | limit |", "|---:|---|---|---:|"]
for job in jobs:
    lines.append(f"| {job['gpu']} | {job['dataset']} | {job['task']} | {job['limit']} |")
atomic_write_text(Path("reports/insight_v2/wave_a_manifest.md"), "\n".join(lines) + "\n")
PY

quant_status="$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
p=Path("reports/insight_v2/quant_reference_validation.json")
print(json.loads(p.read_text()).get("status") if p.exists() else "missing")
PY
)"
parity_status="$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
p=Path("reports/insight_v2/parity_report.json")
print(json.loads(p.read_text()).get("status") if p.exists() else "missing")
PY
)"

if [[ "$quant_status" != "passed" || "$parity_status" != "passed" ]]; then
  cat > run/insight_v2/wave_a.blocked <<EOF
Wave A not launched.
quant_reference_validation=$quant_status
parity_report=$parity_status
EOF
  echo "Wave A blocked: quant=$quant_status parity=$parity_status"
  exit 2
fi

echo "Wave A gates passed, but real generation runner is not connected in this branch."
exit 3
