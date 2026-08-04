#!/usr/bin/env python
"""Run or block Insight micro-smoke according to parity status."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.io import atomic_write_json, atomic_write_text


def main() -> None:
    parity_path = Path("reports/insight_v2/parity_report.json")
    parity = json.loads(parity_path.read_text(encoding="utf-8")) if parity_path.exists() else {"status": "missing"}
    if parity.get("status") != "passed":
        payload = {
            "schema_version": "insight_v2.micro_smoke_report",
            "status": "blocked",
            "reason": f"parity status is {parity.get('status')}",
            "rows": [],
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        }
        atomic_write_json(Path("reports/insight_v2/micro_smoke_report.json"), payload)
        atomic_write_text(Path("reports/insight_v2/micro_smoke_report.md"), "# Insight Micro-Smoke Report\n\nStatus: BLOCKED\n\nReason: parity has not passed.\n")
        print(json.dumps({"status": "blocked", "reason": payload["reason"]}, sort_keys=True))
        return
    raise SystemExit("micro-smoke launch requires explicit observer file generation commands; use bench/bench_pattern_insight.py for the three parity samples, then scripts/check_insight_micro_smoke.py")


if __name__ == "__main__":
    main()
