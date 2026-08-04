#!/usr/bin/env python
"""Validate Insight V2 micro-smoke observer files."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.io import atomic_write_json, atomic_write_text


def has_bad_number(value: Any) -> bool:
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, dict):
        return any(has_bad_number(v) for v in value.values())
    if isinstance(value, list):
        return any(has_bad_number(v) for v in value)
    return False


def check_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or []
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "non_empty": path.stat().st_size > 0,
        "schema_ok": payload.get("schema_version") == "insight_v2.observer",
        "status_completed": payload.get("status") == "completed",
        "multiple_heads": len({r.get("kv_head") for r in records if r.get("kv_head") is not None}) > 1,
        "v_gate_confusion_nonempty": "gate_vs_mse_oracle" in text,
        "v_matching_oracle_nonempty": any(r.get("hook") == "v_matching_oracle" for r in records),
        "k_conditional_oracle_nonempty": any(r.get("hook") == "k_conditional_oracle" for r in records),
        "dynamic_records_nonempty": any(r.get("hook") in {"decode_k", "decode_v"} for r in records),
        "dynamic_mse_not_placeholder": any(r.get("hook") == "decode_k" and (r.get("old_mse") != 0 or r.get("new_mse") != 0) for r in records),
        "has_nan_or_inf": has_bad_number(payload),
        "dropped_record_count_present": "dropped_record_count" in payload,
        "under_100mb": path.stat().st_size < 100 * 1024 * 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observer-files", nargs="*", type=Path, default=[])
    parser.add_argument("--report-json", type=Path, default=Path("reports/insight_v2/micro_smoke_report.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/insight_v2/micro_smoke_report.md"))
    args = parser.parse_args()

    rows = [check_file(path) for path in args.observer_files if path.exists()]
    if not args.observer_files:
        status = "blocked"
        reason = "micro-smoke observer files were not provided"
    elif len(rows) != len(args.observer_files):
        status = "blocked"
        reason = "one or more observer files are missing"
    elif all(
        r["non_empty"] and r["schema_ok"] and r["status_completed"] and r["v_gate_confusion_nonempty"] and r["dropped_record_count_present"] and r["under_100mb"] and not r["has_nan_or_inf"]
        for r in rows
    ):
        status = "passed"
        reason = None
    else:
        status = "failed"
        reason = "one or more observer integrity checks failed"

    payload = {"schema_version": "insight_v2.micro_smoke_report", "generated_at": datetime.now(timezone.utc).isoformat(), "status": status, "reason": reason, "rows": rows}
    atomic_write_json(args.report_json, payload)
    lines = ["# Insight Micro-Smoke Report", "", f"Status: {status.upper()}"]
    if reason:
        lines += ["", f"Reason: {reason}"]
    atomic_write_text(args.report_md, "\n".join(lines) + "\n")
    print(json.dumps({"status": status, "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
