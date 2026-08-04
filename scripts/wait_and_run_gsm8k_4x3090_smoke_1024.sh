#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-120}"
BUSY_MIB="${GPU_MEMORY_BUSY_MIB:-2000}"
LOG="logs/gsm8k/wait_and_run_smoke_1024.log"
mkdir -p logs/gsm8k reports

echo "[$(date --iso-8601=seconds)] waiting for GPU 4,5,6,7 to become available" | tee -a "$LOG"
while true; do
  busy="$("$PYTHON_BIN" - <<PY
import subprocess
out = subprocess.check_output(["nvidia-smi", "--id=4,5,6,7", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
busy = []
for line in out.strip().splitlines():
    idx, mem = [x.strip() for x in line.split(",")]
    if int(mem) > int("$BUSY_MIB"):
        busy.append(f"{idx}:{mem}MiB")
print(" ".join(busy))
PY
)"
  if [[ -z "$busy" ]]; then
    echo "[$(date --iso-8601=seconds)] GPU 4-7 available; launching official smoke_1024" | tee -a "$LOG"
    break
  fi
  echo "[$(date --iso-8601=seconds)] still busy: ${busy}" | tee -a "$LOG"
  sleep "$CHECK_INTERVAL_SECONDS"
done

if scripts/run_gsm8k_4x3090_smoke_1024.sh >> "$LOG" 2>&1; then
  echo "[$(date --iso-8601=seconds)] smoke_1024 runner finished" | tee -a "$LOG"
else
  echo "[$(date --iso-8601=seconds)] smoke_1024 runner exited nonzero" | tee -a "$LOG"
fi

"$PYTHON_BIN" scripts/summarize_gsm8k.py --mode smoke --results-dir results/gsm8k/smoke_1024 --expected-samples 50 --methods fp16 kivi patternkv --report-md reports/gsm8k_smoke_1024.md --report-json reports/gsm8k_smoke_1024.json >> "$LOG" 2>&1 || true
"$PYTHON_BIN" scripts/analyze_gsm8k_smoke.py truncation --root results/gsm8k/smoke_1024 --md reports/gsm8k_smoke_1024_truncation_analysis.md --json reports/gsm8k_smoke_1024_truncation_analysis.json --csv reports/gsm8k_smoke_1024_truncated_samples.csv >> "$LOG" 2>&1 || true
"$PYTHON_BIN" scripts/analyze_gsm8k_smoke.py compare --a results/gsm8k/smoke_1024_existing_mixed_gpu --b results/gsm8k/smoke_1024 --md reports/gsm8k_smoke_existing_vs_official_1024.md --json reports/gsm8k_smoke_existing_vs_official_1024.json >> "$LOG" 2>&1 || true
"$PYTHON_BIN" - <<'PY' >> "$LOG" 2>&1
import json
from pathlib import Path
summary_path = Path("reports/gsm8k_smoke_1024.json")
if summary_path.exists():
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    methods = s.get("methods", [])
    pass_strict = bool(methods) and not s.get("issues") and all(
        m.get("rows") == 50
        and m.get("errors") == 0
        and m.get("oom_errors") == 0
        and m.get("empty_predictions") == 0
        and m.get("length_truncated") == 0
        for m in methods
    )
    status = "PASS" if pass_strict else "PARTIAL PASS"
    full = "YES" if pass_strict else "NO"
else:
    s = {"methods": [], "issues": ["missing reports/gsm8k_smoke_1024.json"]}
    status = "FAIL"
    full = "NO"
lines = [
    "# GSM8K Smoke 1024 Final Status",
    "",
    f"## 当前状态",
    "",
    status,
    "",
    "## 512结果",
    "",
    "No complete three-method 512-token result was found. Historical user-provided numbers are preserved as requested: FP16 82%, 1 truncation; KIVI fixed 36%, 20 truncations; PatternKV 84%, 1 truncation. Current complete three-method archived records are actually max_new_tokens=1024.",
    "",
    "## 1024结果",
    "",
]
for m in s.get("methods", []):
    lines.append(f"- {m['method']}: accuracy={m.get('accuracy_percent')}%, correct={m.get('correct')}/50, truncated={m.get('length_truncated')}, parser_failure={m.get('parser_failures')}, errors={m.get('errors')}, avg_output_tokens={m.get('avg_output_tokens')}, p95_output_tokens={m.get('p95_output_tokens')}, normal_eos_rate={m.get('normal_eos_rate_percent')}")
lines += [
    "",
    "## KIVI专项分析",
    "",
    "See `reports/gsm8k_smoke_1024_truncation_analysis.md` and `reports/gsm8k_smoke_512_truncation_analysis.md`.",
    "",
    f"FULL_RUN_ALLOWED = {full}",
]
Path("reports/gsm8k_smoke_1024_final_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote reports/gsm8k_smoke_1024_final_status.md with FULL_RUN_ALLOWED={full}")
PY
echo "[$(date --iso-8601=seconds)] wait-and-run pipeline finished" | tee -a "$LOG"
