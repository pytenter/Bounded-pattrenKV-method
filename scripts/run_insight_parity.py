#!/usr/bin/env python
"""Build token-level Insight parity report from generated records."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.io import atomic_write_json, atomic_write_text


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_triplet(off: dict[str, Any], basic: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    off_ids = [int(x) for x in off.get("generated_token_ids") or []]
    basic_ids = [int(x) for x in basic.get("generated_token_ids") or []]
    oracle_ids = [int(x) for x in oracle.get("generated_token_ids") or []]
    basic_equal = off_ids == basic_ids
    oracle_equal = off_ids == oracle_ids
    return {
        "dataset": off.get("dataset"),
        "task": off.get("task"),
        "problem_id": off.get("problem_id"),
        "sample_id": off.get("sample_id"),
        "off_hash": off.get("generated_token_ids_sha256"),
        "basic_hash": basic.get("generated_token_ids_sha256"),
        "oracle_hash": oracle.get("generated_token_ids_sha256"),
        "token_equal": basic_equal and oracle_equal,
        "score_equal": (off.get("score"), off.get("is_correct")) == (basic.get("score"), basic.get("is_correct")) == (oracle.get("score"), oracle.get("is_correct")),
        "stop_reason_equal": off.get("stop_reason") == basic.get("stop_reason") == oracle.get("stop_reason"),
        "generated_tokens_equal": off.get("generated_tokens") == basic.get("generated_tokens") == oracle.get("generated_tokens"),
        "first_basic_divergence": first_divergence(off_ids, basic_ids),
        "first_oracle_divergence": first_divergence(off_ids, oracle_ids),
        "observer_files": [basic.get("observer_output_path"), oracle.get("observer_output_path")],
        "observer_errors": [basic.get("error"), oracle.get("error")],
        "peak_memory": {
            "off": off.get("peak_memory_allocated"),
            "basic": basic.get("peak_memory_allocated"),
            "oracle": oracle.get("peak_memory_allocated"),
        },
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", nargs="*", type=Path, default=[])
    parser.add_argument("--basic", nargs="*", type=Path, default=[])
    parser.add_argument("--oracle", nargs="*", type=Path, default=[])
    parser.add_argument("--manifest", type=Path, default=Path("reports/insight_v2/parity_manifest.json"))
    parser.add_argument("--report-json", type=Path, default=Path("reports/insight_v2/parity_report.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/insight_v2/parity_report.md"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    missing_reason = None
    manifest_rows = []
    if args.manifest.exists():
        manifest_rows = json.loads(args.manifest.read_text(encoding="utf-8")).get("samples", [])
        blocked = [r for r in manifest_rows if str(r.get("identity_status", "")).startswith("blocked")]
        if blocked and not (args.off or args.basic or args.oracle):
            missing_reason = "; ".join(
                f"{r.get('dataset')}/{r.get('task')}: {r.get('identity_status')}" for r in blocked
            )
    if missing_reason is None and (args.off or args.basic or args.oracle):
        if not (len(args.off) == len(args.basic) == len(args.oracle) and args.off):
            missing_reason = "off/basic/oracle file counts differ or are empty"
        else:
            for off_path, basic_path, oracle_path in zip(args.off, args.basic, args.oracle):
                rows.append(compare_triplet(load_json(off_path), load_json(basic_path), load_json(oracle_path)))
    elif missing_reason is None:
        missing_reason = "parity generation files were not provided"

    if missing_reason:
        status = "blocked"
    elif all(r["token_equal"] and r["score_equal"] and r["stop_reason_equal"] for r in rows):
        status = "passed"
    else:
        status = "failed"

    payload = {
        "schema_version": "insight_v2.parity_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason": missing_reason,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "required_samples": [
            {"dataset": "longbench", "task": "hotpotqa", "count": 1},
            {"dataset": "longbench", "task": "samsum", "count": 1},
            {"dataset": "gsm8k", "task": "gsm8k", "count": 1},
        ],
        "manifest": str(args.manifest),
        "manifest_samples": manifest_rows,
        "rows": rows,
    }
    atomic_write_json(args.report_json, payload)
    lines = ["# Insight Parity Report", "", f"Status: {status.upper()}"]
    if missing_reason:
        lines += ["", f"Reason: {missing_reason}"]
    lines += ["", "| dataset | task | sample | token_equal | first_basic_divergence | first_oracle_divergence |", "|---|---|---|---:|---:|---:|"]
    for row in rows:
        sample = row.get("sample_id") or row.get("problem_id")
        lines.append(f"| {row.get('dataset')} | {row.get('task')} | {sample} | {row.get('token_equal')} | {row.get('first_basic_divergence')} | {row.get('first_oracle_divergence')} |")
    atomic_write_text(args.report_md, "\n".join(lines) + "\n")
    print(json.dumps({"status": status, "rows": len(rows), "report": str(args.report_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
