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


def value_present(record: dict[str, Any], key: str) -> bool:
    return key in record and record.get(key) is not None


def non_empty(records: list[dict[str, Any]], hook: str, key: str) -> bool:
    return any(r.get("hook") == hook and value_present(r, key) for r in records)


def layer_head_coverage(records: list[dict[str, Any]], hook: str) -> dict[str, Any]:
    selected = [r for r in records if r.get("hook") == hook]
    layers = {r.get("layer_idx") for r in selected if r.get("layer_idx") is not None}
    heads = {r.get("kv_head") for r in selected if r.get("kv_head") is not None}
    return {
        "record_count": len(selected),
        "layer_count": len(layers),
        "kv_head_count": len(heads),
        "multiple_layers": len(layers) > 1,
        "multiple_kv_heads": len(heads) > 1,
    }


def confusion_total(confusion: dict[str, Any]) -> int:
    total = 0
    for key, value in confusion.items():
        if "gate_vs_mse_oracle" not in key or not isinstance(value, dict):
            continue
        total += sum(int(value.get(name) or 0) for name in ("true_positive", "true_negative", "false_positive", "false_negative"))
    return total


def oracle_not_worse(records: list[dict[str, Any]], hook: str) -> bool:
    selected = [r for r in records if r.get("hook") == hook]
    if not selected:
        return False
    eps = 1e-6
    for record in selected:
        if hook == "v_matching_oracle":
            oracle = record.get("oracle_mse")
            candidates = [record.get("l2_mse"), record.get("minmax_mse"), record.get("current_pattern_mse")]
        else:
            oracle = record.get("conditional_oracle_group_mse")
            candidates = [record.get("current_group_mse")]
        if oracle is None:
            return False
        for candidate in candidates:
            if candidate is not None and float(oracle) > float(candidate) + eps:
                return False
    return True


def check_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or []
    histograms = payload.get("histograms") or {}
    confusion = payload.get("confusion") or {}
    active_observer_count = payload.get("active_observer_count")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "non_empty": path.stat().st_size > 0,
        "schema_ok": payload.get("schema_version") == "insight_v2.observer",
        "status_completed": payload.get("status") == "completed",
        "error_empty": payload.get("error") in (None, ""),
        "estimated_serialized_bytes_present": "estimated_serialized_bytes" in payload,
        "dropped_record_count_present": "dropped_record_count" in payload,
        "active_observer_leak": active_observer_count not in (None, 0),
        "prefill_k": layer_head_coverage(records, "prefill_k"),
        "prefill_v": layer_head_coverage(records, "prefill_v"),
        "v_matching_oracle": layer_head_coverage(records, "v_matching_oracle"),
        "k_conditional_oracle": layer_head_coverage(records, "k_conditional_oracle"),
        "decode_k": layer_head_coverage(records, "decode_k"),
        "decode_v": layer_head_coverage(records, "decode_v"),
        "k_raw_mse_nonempty": non_empty(records, "prefill_k", "raw_mse"),
        "k_pattern_mse_nonempty": non_empty(records, "prefill_k", "pattern_mse"),
        "k_relative_benefit_nonempty": non_empty(records, "prefill_k", "relative_benefit"),
        "k_harmful_nonempty": non_empty(records, "prefill_k", "harmful"),
        "k_assignment_histogram_nonempty": any(key.startswith("prefill.k.") and key.endswith(".assignment_histogram") for key in histograms),
        "v_raw_mse_nonempty": non_empty(records, "prefill_v", "raw_mse"),
        "v_pattern_candidate_mse_nonempty": non_empty(records, "prefill_v", "pattern_candidate_mse"),
        "v_gate_confusion_total": confusion_total(confusion),
        "v_assignment_histogram_nonempty": any(key.startswith("prefill.v.") and key.endswith(".assignment_histogram") for key in histograms),
        "v_matching_fields_present": all(
            non_empty(records, "v_matching_oracle", key)
            for key in ("l2_mse", "minmax_mse", "current_pattern_mse", "oracle_mse")
        ),
        "v_matching_oracle_not_worse": oracle_not_worse(records, "v_matching_oracle"),
        "k_conditional_fields_present": all(
            non_empty(records, "k_conditional_oracle", key)
            for key in ("current_group_mse", "conditional_oracle_group_mse")
        ),
        "k_conditional_oracle_not_worse": oracle_not_worse(records, "k_conditional_oracle"),
        "decode_k_records_nonempty": any(r.get("hook") == "decode_k" for r in records),
        "decode_v_records_nonempty": any(r.get("hook") == "decode_v" for r in records),
        "decode_mse_not_placeholder": any(
            r.get("hook") == "decode_k" and (r.get("old_mse") not in (None, 0) or r.get("new_mse") not in (None, 0))
            for r in records
        ),
        "decode_candidate_assignment_fraction_present": non_empty(records, "decode_k", "candidate_assignment_fraction"),
        "decode_v_candidate_gate_applied_fraction_present": non_empty(records, "decode_v", "candidate_gate_accepted_fraction"),
        "has_nan_or_inf": has_bad_number(payload),
        "under_100mb": path.stat().st_size < 100 * 1024 * 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observer-files", nargs="*", type=Path, default=[])
    parser.add_argument("--report-json", type=Path, default=Path("reports/insight_v2/micro_smoke_report.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/insight_v2/micro_smoke_report.md"))
    args = parser.parse_args()

    rows = [check_file(path) for path in args.observer_files if path.exists()]
    base_ok = all(
        r["non_empty"]
        and r["schema_ok"]
        and r["status_completed"]
        and r["error_empty"]
        and r["estimated_serialized_bytes_present"]
        and r["dropped_record_count_present"]
        and r["under_100mb"]
        and not r["has_nan_or_inf"]
        and not r["active_observer_leak"]
        for r in rows
    )
    coverage = {
        "k_multiple_layers": any(r["prefill_k"]["multiple_layers"] for r in rows),
        "k_multiple_kv_heads": any(r["prefill_k"]["multiple_kv_heads"] for r in rows),
        "k_raw_mse_nonempty": any(r["k_raw_mse_nonempty"] for r in rows),
        "k_pattern_mse_nonempty": any(r["k_pattern_mse_nonempty"] for r in rows),
        "k_relative_benefit_nonempty": any(r["k_relative_benefit_nonempty"] for r in rows),
        "k_harmful_nonempty": any(r["k_harmful_nonempty"] for r in rows),
        "k_assignment_histogram_nonempty": any(r["k_assignment_histogram_nonempty"] for r in rows),
        "v_multiple_layers": any(r["prefill_v"]["multiple_layers"] for r in rows),
        "v_multiple_kv_heads": any(r["prefill_v"]["multiple_kv_heads"] for r in rows),
        "v_raw_mse_nonempty": any(r["v_raw_mse_nonempty"] for r in rows),
        "v_pattern_candidate_mse_nonempty": any(r["v_pattern_candidate_mse_nonempty"] for r in rows),
        "v_gate_confusion_total_gt_zero": sum(r["v_gate_confusion_total"] for r in rows) > 0,
        "v_assignment_histogram_nonempty": any(r["v_assignment_histogram_nonempty"] for r in rows),
        "v_matching_oracle_nonempty": any(r["v_matching_oracle"]["record_count"] > 0 for r in rows),
        "v_matching_fields_present": any(r["v_matching_fields_present"] for r in rows),
        "v_matching_oracle_not_worse": any(r["v_matching_oracle_not_worse"] for r in rows),
        "k_conditional_oracle_nonempty": any(r["k_conditional_oracle"]["record_count"] > 0 for r in rows),
        "k_conditional_fields_present": any(r["k_conditional_fields_present"] for r in rows),
        "k_conditional_oracle_not_worse": any(r["k_conditional_oracle_not_worse"] for r in rows),
        "decode_k_records_nonempty": any(r["decode_k_records_nonempty"] for r in rows),
        "decode_v_records_nonempty": any(r["decode_v_records_nonempty"] for r in rows),
        "decode_mse_not_placeholder": any(r["decode_mse_not_placeholder"] for r in rows),
        "decode_candidate_assignment_fraction_present": any(r["decode_candidate_assignment_fraction_present"] for r in rows),
        "decode_v_candidate_gate_applied_fraction_present": any(r["decode_v_candidate_gate_applied_fraction_present"] for r in rows),
    }
    if not args.observer_files:
        status = "blocked"
        reason = "micro-smoke observer files were not provided"
    elif len(rows) != len(args.observer_files):
        status = "blocked"
        reason = "one or more observer files are missing"
    elif base_ok and all(coverage.values()):
        status = "passed"
        reason = None
    else:
        status = "failed"
        reason = "one or more observer integrity checks failed"

    payload = {
        "schema_version": "insight_v2.micro_smoke_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason": reason,
        "base_ok": base_ok,
        "coverage": coverage,
        "rows": rows,
    }
    atomic_write_json(args.report_json, payload)
    lines = ["# Insight Micro-Smoke Report", "", f"Status: {status.upper()}"]
    if reason:
        lines += ["", f"Reason: {reason}"]
    atomic_write_text(args.report_md, "\n".join(lines) + "\n")
    print(json.dumps({"status": status, "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
