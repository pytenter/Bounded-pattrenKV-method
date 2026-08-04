#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"

echo "== GPU 4-7 utilization =="
nvidia-smi --id=4,5,6,7 --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

echo
echo "== Worker status =="
"$PYTHON_BIN" - <<'PY'
import json
import os
from datetime import datetime
from pathlib import Path

def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False

for meta in sorted(Path("run/gsm8k").glob("*/*/*.meta")):
    pidfile = meta.with_suffix(".pid")
    pid = pidfile.read_text().strip() if pidfile.exists() else None
    items = dict(line.split("=", 1) for line in meta.read_text(encoding="utf-8").splitlines() if "=" in line)
    status_files = list(meta.parent.glob(f"gpu{items.get('gpu')}_shard{items.get('shard')}.status.json"))
    status = {}
    if status_files:
        status = json.loads(status_files[0].read_text(encoding="utf-8"))
    out = Path(items.get("log", "")).as_posix()
    result = Path("results/gsm8k") / items.get("mode", "") / items.get("method", "") / f"shard_{items.get('shard')}.jsonl"
    completed = 0
    failures = 0
    if result.exists():
        for line in result.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            completed += 1
            try:
                failures += int(bool(json.loads(line).get("error")))
            except Exception:
                failures += 1
    mtime = datetime.fromtimestamp(result.stat().st_mtime).isoformat() if result.exists() else "missing"
    print(f"{items.get('mode')}/{items.get('method')}/shard_{items.get('shard')}: pid={pid} alive={alive(pid) if pid else False} gpu={items.get('gpu')} logical_gpu=0")
    print(f"  current_sample={status.get('current_sample_index')} completed={completed}/{status.get('expected_shard_samples', '?')} failures={failures}")
    print(f"  result_mtime={mtime} log={out}")

print("\n== Result totals ==")
for mode, expected in (("smoke", 50), ("full", 1319)):
    for method in ("fp16", "kivi", "patternkv"):
        rows = []
        for path in sorted((Path("results/gsm8k") / mode / method).glob("shard_*.jsonl")):
            rows.extend([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
        print(f"{mode}/{method}: rows={len(rows)}/{expected}")
PY

echo
echo "== Compute apps memory =="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
