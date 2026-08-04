#!/usr/bin/env python
"""Gate Wave A launch on quant, parity, and micro-smoke evidence."""

from __future__ import annotations

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


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def has_bad_number(value: Any) -> bool:
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, dict):
        return any(has_bad_number(v) for v in value.values())
    if isinstance(value, list):
        return any(has_bad_number(v) for v in value)
    return False


def main() -> None:
    quant = load(Path("reports/insight_v2/quant_reference_validation.json"))
    parity = load(Path("reports/insight_v2/parity_report.json"))
    micro = load(Path("reports/insight_v2/micro_smoke_report.json"))
    reasons: list[str] = []
    if quant.get("status") != "passed":
        reasons.append(f"quant_reference_validation={quant.get('status')}")
    if parity.get("status") != "passed":
        reasons.append(f"parity={parity.get('status')} reason={parity.get('reason')}")
    if micro.get("status") != "passed":
        reasons.append(f"micro_smoke={micro.get('status')} reason={micro.get('reason')}")
    if has_bad_number(quant) or has_bad_number(parity) or has_bad_number(micro):
        reasons.append("NaN/Inf detected in gate reports")
    for row in micro.get("rows") or []:
        if row.get("size_bytes", 0) >= 100 * 1024 * 1024:
            reasons.append(f"observer_file_over_100mb={row.get('path')}")
        if row.get("has_nan_or_inf"):
            reasons.append(f"observer_nan_inf={row.get('path')}")
    for row in parity.get("rows") or []:
        peak = row.get("peak_memory") or {}
        off = peak.get("off") or 0
        oracle = peak.get("oracle") or 0
        if oracle and off and oracle - off > 6 * 1024**3:
            reasons.append(f"oracle_extra_memory_over_6gb={row.get('dataset')}/{row.get('task')}")
    status = "passed" if not reasons else "blocked"
    payload = {
        "schema_version": "insight_v2.wave_a_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reasons": reasons,
        "inputs": {
            "quant_status": quant.get("status"),
            "parity_status": parity.get("status"),
            "micro_smoke_status": micro.get("status"),
        },
    }
    atomic_write_json(Path("reports/insight_v2/wave_a_gate.json"), payload)
    lines = ["# Insight Wave A Gate", "", f"Status: {status.upper()}"]
    if reasons:
        lines += ["", "Reasons:"] + [f"- {r}" for r in reasons]
    atomic_write_text(Path("reports/insight_v2/wave_a_gate.md"), "\n".join(lines) + "\n")
    print(json.dumps({"status": status, "reasons": reasons}, sort_keys=True))
    raise SystemExit(0 if status == "passed" else 2)


if __name__ == "__main__":
    main()
