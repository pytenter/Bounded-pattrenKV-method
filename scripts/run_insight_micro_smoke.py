#!/usr/bin/env python
"""Run Insight micro-smoke validation after parity has passed."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.io import atomic_write_json, atomic_write_text

DEFAULT_OBSERVER_FILES = [
    Path(
        "results/insight_v2/parity/observer/oracle/longbench/hotpotqa/"
        "hotpotqa:b6352c61b4a748448ce38882861cd5ae5f7f2869a81e92a1_oracle_seed0.json"
    ),
    Path(
        "results/insight_v2/parity/observer/oracle/longbench/samsum/"
        "samsum:6b2677b451034aef716068c91304f0ce5e33440ebf8b4620_oracle_seed0.json"
    ),
    Path("results/insight_v2/parity/observer/oracle/gsm8k/gsm8k/p0809_oracle_seed0.json"),
]


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

    missing = [str(path) for path in DEFAULT_OBSERVER_FILES if not path.exists()]
    if missing:
        payload = {
            "schema_version": "insight_v2.micro_smoke_report",
            "status": "blocked",
            "reason": "missing oracle parity observer files",
            "missing": missing,
            "rows": [],
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        }
        atomic_write_json(Path("reports/insight_v2/micro_smoke_report.json"), payload)
        atomic_write_text(
            Path("reports/insight_v2/micro_smoke_report.md"),
            "# Insight Micro-Smoke Report\n\nStatus: BLOCKED\n\nReason: missing oracle parity observer files.\n",
        )
        print(json.dumps({"status": "blocked", "reason": payload["reason"], "missing": missing}, sort_keys=True))
        return

    cmd = [
        sys.executable,
        "scripts/check_insight_micro_smoke.py",
        "--observer-files",
        *[str(path) for path in DEFAULT_OBSERVER_FILES],
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
