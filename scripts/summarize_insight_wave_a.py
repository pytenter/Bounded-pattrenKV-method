#!/usr/bin/env python
"""Summarize Wave A observer outputs when available."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.io import atomic_write_json, atomic_write_text


def main() -> None:
    out_dir = Path("reports/insight_v2/wave_a")
    out_dir.mkdir(parents=True, exist_ok=True)
    observer_files = sorted(Path("results/insight_v2/observer").rglob("*.json"))
    rows = []
    for path in observer_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"path": str(path), "status": "unreadable", "error": repr(exc)})
            continue
        rows.append(
            {
                "path": str(path),
                "status": payload.get("status"),
                "dataset": (payload.get("metadata") or {}).get("dataset"),
                "task": (payload.get("metadata") or {}).get("task"),
                "sample_id": (payload.get("metadata") or {}).get("sample_id"),
                "estimated_serialized_bytes": payload.get("estimated_serialized_bytes"),
                "dropped_record_count": payload.get("dropped_record_count"),
                "truncated": payload.get("truncated"),
            }
        )
    status = "completed" if rows and all(r.get("status") == "completed" for r in rows) else "partial_or_blocked"
    payload = {
        "schema_version": "insight_v2.wave_a_summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "observer_file_count": len(observer_files),
        "rows": rows,
    }
    atomic_write_json(out_dir / "completion.json", payload)
    lines = [
        "# Wave A Completion",
        "",
        f"Status: {status.upper()}",
        f"Observer files: {len(observer_files)}",
        "",
    ]
    if not observer_files:
        lines.append("No Wave A observer outputs are available.")
    atomic_write_text(out_dir / "completion.md", "\n".join(lines) + "\n")
    if observer_files:
        for name in (
            "pattern_gain_map.csv",
            "pattern_gain_map_by_head.csv",
            "pattern_harm_fraction.csv",
            "matching_oracle_gap.csv",
            "k_conditional_oracle.csv",
            "v_gate_confusion.csv",
            "dynamic_pattern_utility.csv",
        ):
            (out_dir / name).write_text("task,phase,layer,kv_head,kv_type,metric,count,mean\n", encoding="utf-8")
    atomic_write_text(out_dir / "summary.md", "\n".join(lines) + "\n")
    atomic_write_text(out_dir / "decision_matrix.md", "# Decision Matrix\n\nStatus: DATA INSUFFICIENT\n",)
    print(json.dumps({"status": status, "observer_file_count": len(observer_files)}, sort_keys=True))


if __name__ == "__main__":
    main()
