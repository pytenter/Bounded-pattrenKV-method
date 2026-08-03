#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MICROMAMBA_BIN="${MICROMAMBA_BIN:-}"
PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
if [[ -z "$MICROMAMBA_BIN" ]]; then
  if command -v micromamba >/dev/null 2>&1; then
    MICROMAMBA_BIN="$(command -v micromamba)"
  elif [[ -x /data/zypan/kvarn-repro/tools/bin/micromamba ]]; then
    MICROMAMBA_BIN="/data/zypan/kvarn-repro/tools/bin/micromamba"
  else
    echo "micromamba not found; set MICROMAMBA_BIN=/path/to/micromamba" >&2
    exit 1
  fi
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at ${PYTHON_BIN}; set PYTHON_BIN to the patternkv env python" >&2
  exit 1
fi

echo "== GPU 0-5 utilization =="
nvidia-smi --id=0,1,2,3,4,5 --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

echo
echo "== Worker status =="
"$PYTHON_BIN" - <<'PY'
import json
import os
import signal
from datetime import datetime
from pathlib import Path

run = Path("run/longbench")
statuses = sorted(run.glob("gpu*_*.status.json"))
metas = sorted(run.glob("gpu*_*.meta"))
seen = set()

def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False

for status in statuses:
    data = json.loads(status.read_text(encoding="utf-8"))
    seen.add(status.stem.replace(".status", ""))
    print(f"{status.name}:")
    print(f"  pid={data.get('pid')} alive={alive(data.get('pid'))} gpu={data.get('physical_gpu_id')} logical_gpu={data.get('logical_gpu_id')} method={data.get('method')} mode={data.get('mode')}")
    print(f"  current_task={data.get('current_task')} current_sample={data.get('current_sample')}")
    print(f"  completed={data.get('completed_samples')}/{data.get('total_samples')} failures={data.get('failures')}")
    mtimes = data.get("result_mtimes") or {}
    for task, mtime in sorted(mtimes.items()):
        print(f"  result_mtime[{task}]={mtime}")

for meta in metas:
    stem = meta.stem
    if any(stem.startswith(s.split(".status")[0]) for s in seen):
        continue
    pidfile = run / (stem + ".pid")
    pid = pidfile.read_text().strip() if pidfile.exists() else None
    print(f"{meta.name}:")
    print(f"  pid={pid} alive={alive(pid) if pid else False}")
    print("  " + " ".join(line.strip() for line in meta.read_text(encoding="utf-8").splitlines()))

print("\n== Result counts ==")
tasks = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "trec", "passage_retrieval_en", "lcc"]
methods = ["fp16", "patternkv"]
if (Path("results/longbench") / "kivi").exists():
    methods.append("kivi")
for method in methods:
    for task in tasks:
        path = Path("results/longbench") / method / f"{task}.jsonl"
        rows = []
        if path.exists():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        failures = sum(1 for row in rows if row.get("error"))
        mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else "missing"
        print(f"{method}/{task}: completed={len(rows)} failures={failures} last_update={mtime}")
PY

echo
echo "== Compute apps memory =="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
