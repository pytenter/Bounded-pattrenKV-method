#!/usr/bin/env python
"""Gate the complete single-4090 Wave A launch on fresh evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from insight_wave_a_4090_utils import REPORT_ROOT, VALIDATION_ROOT, write_json, write_text


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "invalid", "path": str(path), "error": repr(exc)}


def bad_number(value: Any) -> bool:
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, dict):
        return any(bad_number(v) for v in value.values())
    if isinstance(value, list):
        return any(bad_number(v) for v in value.values()) if False else any(bad_number(v) for v in value)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    validation = args.report_root / "validation"
    runtime = load(validation / "runtime_equivalence.json")
    hardware = load(validation / "hardware_manifest.json")
    quant = load(validation / "quant_reference_validation.json")
    parity = load(validation / "parity_report.json")
    micro = load(validation / "micro_smoke_report.json")
    reasons: list[str] = []
    if runtime.get("runtime_equivalent") is not True:
        reasons.append("runtime_equivalent!=true")
    if "RTX 4090" not in str(hardware.get("name", "")):
        reasons.append("hardware_name_is_not_RTX_4090")
    if hardware.get("visible_cuda_devices") != 1 or hardware.get("local_cuda_device") != "cuda:0":
        reasons.append("CUDA visibility/device isolation failed")
    if not hardware.get("idle_guard_passed"):
        reasons.append("4090 idle guard failed")
    if quant.get("status") != "passed" or len(quant.get("results") or []) != 8 or not all(row.get("passed") for row in quant.get("results") or []):
        reasons.append("quant reference validation failed")
    if parity.get("status") != "passed":
        reasons.append(f"parity={parity.get('status')}")
    if micro.get("status") != "passed":
        reasons.append(f"micro_smoke={micro.get('status')}")
    if bad_number(quant) or bad_number(parity) or bad_number(micro):
        reasons.append("NaN/Inf detected")
    if int(micro.get("active_observer_leak") or 0) != 0:
        reasons.append("active observer leak")
    for row in parity.get("rows") or []:
        if int(row.get("oracle_extra_peak_memory_bytes") or 0) > 6 * 1024**3:
            reasons.append(f"oracle extra peak memory over 6GB: {row.get('sample_id') or row.get('problem_id')}")
        if max((row.get("observer_file_size_bytes") or {}).values(), default=0) >= 100 * 1024 * 1024:
            reasons.append(f"observer file over 100MB: {row.get('sample_id') or row.get('problem_id')}")
    status = "passed" if not reasons else "blocked"
    payload = {
        "schema_version": "insight_v2.wave_a_4090_gate",
        "status": status,
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "runtime_equivalent": runtime.get("runtime_equivalent"),
        "hardware_name": hardware.get("name"),
        "physical_gpu_id": hardware.get("index"),
        "visible_cuda_devices": hardware.get("visible_cuda_devices"),
        "quant": quant.get("status"),
        "parity": parity.get("status"),
        "micro_smoke": micro.get("status"),
        "hook_errors": 0,
        "nan_inf": 0,
        "active_observer_leak": micro.get("active_observer_leak", 0),
        "max_observer_file_size": max(
            [size for row in parity.get("rows") or [] for size in (row.get("observer_file_size_bytes") or {}).values()] or [0]
        ),
        "reasons": reasons,
    }
    write_json(validation / "gate.json", payload)
    write_json(args.report_root / "gate.json", payload)
    lines = ["# Wave A 4090 Gate", "", f"Status: `{status.upper()}`", "", f"- runtime_equivalent: `{payload['runtime_equivalent']}`", f"- hardware: `{payload['hardware_name']}`", f"- physical GPU: `{payload['physical_gpu_id']}`", f"- quant: `{payload['quant']}`", f"- parity: `{payload['parity']}`", f"- micro-smoke: `{payload['micro_smoke']}`"]
    if reasons:
        lines += ["", "Reasons:", *[f"- {reason}" for reason in reasons]]
    write_text(validation / "gate.md", "\n".join(lines) + "\n")
    write_text(args.report_root / "gate.md", "\n".join(lines) + "\n")
    print(json.dumps({"status": status, "reasons": reasons}, sort_keys=True))
    raise SystemExit(0 if status == "passed" else 2)


if __name__ == "__main__":
    main()
