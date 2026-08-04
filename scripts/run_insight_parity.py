#!/usr/bin/env python
"""Run or report PatternKV Insight observer parity checks."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.io import atomic_write_json, atomic_write_text


REPORT_JSON = Path("reports/insight_v2/parity_report.json")
REPORT_MD = Path("reports/insight_v2/parity_report.md")


def main() -> None:
    status = {
        "schema_version": "insight_v2.parity_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "blocked",
        "reason": "bench/bench_pattern_insight.py real generation lifecycle is not connected yet",
        "required_samples": [
            {"dataset": "longbench", "task": "hotpotqa", "count": 1},
            {"dataset": "longbench", "task": "samsum", "count": 1},
            {"dataset": "gsm8k", "task": "gsm8k", "count": 1},
        ],
        "required_comparison": [
            "input_token_ids",
            "generated_token_ids",
            "sha256",
            "generated_text",
            "score_or_is_correct",
            "stop_reason",
            "cache_length",
            "last_token_id",
        ],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    atomic_write_json(REPORT_JSON, status)
    lines = [
        "# Insight Parity Report",
        "",
        "Status: BLOCKED",
        "",
        f"Reason: {status['reason']}",
        "",
        "Token-level parity must pass before Wave A can run.",
    ]
    atomic_write_text(REPORT_MD, "\n".join(lines) + "\n")
    print(json.dumps({"status": "blocked", "report": str(REPORT_JSON)}, sort_keys=True))


if __name__ == "__main__":
    main()
