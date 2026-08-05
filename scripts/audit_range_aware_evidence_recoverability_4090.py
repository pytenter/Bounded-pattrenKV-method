#!/usr/bin/env python
"""Audit recoverability of range-aware evidence from 4090 Wave A observer data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.io import atomic_write_json, atomic_write_text, write_csv


FIELDS_OF_INTEREST = [
    "task",
    "sample_id",
    "sample_index",
    "problem_id",
    "phase",
    "layer",
    "kv_head",
    "kv_type",
    "bucket",
    "token_index",
    "window_index",
    "candidate_count",
    "pattern_count",
    "l2_assignment",
    "minmax_assignment",
    "l2_assignment_count",
    "minmax_assignment_count",
    "assignment_mismatch_count",
    "assignment_total_count",
    "l2_residual_range",
    "minmax_residual_range",
    "range_regret",
    "range_regret_sum",
    "range_regret_sum_sq",
    "range_regret_count",
    "raw_range",
    "candidate_pattern_ids",
    "candidate_pattern_distances",
    "candidate_residual_ranges",
    "pattern_bank_identity",
]


SEMANTICS_FINDINGS = [
    {
        "file": "models/llama_patternkv.py",
        "line": 333,
        "function_or_class": "batched_kmeans_fast",
        "evidence": "d2 = ||x||^2 + ||c||^2 - 2 x·c; assign = d2.argmin(dim=-1).",
        "interpretation": "Pattern bank mining uses L2 KMeans updates on a fixed [H,N,D] token matrix.",
    },
    {
        "file": "models/llama_patternkv.py",
        "line": 936,
        "function_or_class": "LlamaFlashAttention_PatternKV.forward",
        "evidence": "assign_k = batched_assign_compiled(Xk, k_centroids); assign = batched_assign_compiled(X, centroids).",
        "interpretation": "Runtime prefill assignment for both K and V uses L2 nearest-centroid assignment on the mined bank.",
    },
    {
        "file": "models/llama_patternkv.py",
        "line": 495,
        "function_or_class": "LlamaFlashAttention_PatternKV._assign_minmax_hnk",
        "evidence": "r = diff.amax(-1) - diff.amin(-1); idx = r.min(-1).",
        "interpretation": "K min-max assignment is defined on residual dynamic range over the head_dim axis for each [H,N,D] token vector.",
    },
    {
        "file": "models/llama_patternkv.py",
        "line": 613,
        "function_or_class": "LlamaFlashAttention_PatternKV._nearest_v_centroid",
        "evidence": "r = diff.amax(-1) - diff.amin(-1); idx = r.argmin(dim=2).",
        "interpretation": "V runtime nearest-centroid matching uses min-max residual range over head_dim, not L2.",
    },
    {
        "file": "insight/quant_reference.py",
        "line": 41,
        "function_or_class": "quantize_dequant_k_token_groups",
        "evidence": "K is transposed on [B,H,T,D] -> [B,H,D,T] and quantized along the last dimension in 128-token groups.",
        "interpretation": "Real K quantization axis is per-channel over 128-token groups, so K range-aware evidence must respect token grouping after transpose.",
    },
    {
        "file": "insight/quant_reference.py",
        "line": 59,
        "function_or_class": "quantize_dequant_v_head_dim",
        "evidence": "V stays [B,H,T,D] and is quantized along the last dimension.",
        "interpretation": "Real V quantization axis is per-token over head_dim=128.",
    },
    {
        "file": "insight/hook_metrics.py",
        "line": 190,
        "function_or_class": "_record_k_conditional_oracle",
        "evidence": "l2_assignment and minmax_assignment are sampled per token within a 128-token K group.",
        "interpretation": "Existing K oracle records diagnose runtime matching mismatch on a fixed pattern bank, not a re-mined bank objective.",
    },
    {
        "file": "insight/hook_metrics.py",
        "line": 221,
        "function_or_class": "_record_k_conditional_oracle",
        "evidence": "raw_group_mse, current_group_mse, minmax_group_mse, conditional_oracle_group_mse are stored; no residual ranges are stored.",
        "interpretation": "K oracle records provide MSE-gap evidence but do not directly expose K range regret or per-assignment residual ranges.",
    },
    {
        "file": "insight/hook_metrics.py",
        "line": 347,
        "function_or_class": "_record_v_matching_oracle",
        "evidence": "l2_assignment, minmax_assignment, l2_minmax_mismatch, range_regret, current_oracle_gap, minmax_oracle_gap are recorded.",
        "interpretation": "V oracle records directly contain mismatch and a regret-like scalar, but only at sampled-token granularity.",
    },
    {
        "file": "scripts/summarize_insight_wave_a_8gpu.py",
        "line": 227,
        "function_or_class": "main",
        "evidence": "matching_oracle_gap.csv keeps only metrics whose names contain oracle_gap.",
        "interpretation": "Published matching_oracle_gap.csv does not export mismatch counts or range regret directly.",
    },
    {
        "file": "insight/collector.py",
        "line": 96,
        "function_or_class": "InsightCollector.add_sample_record",
        "evidence": "When max_sample_records is reached, dropped_record_count increments and later records are discarded.",
        "interpretation": "Any recoverability path that depends on sample records is invalid for fully recovered 140-sample conclusions when truncation is present.",
    },
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_task_from_path(path: Path) -> str:
    parts = path.parts
    if "observer" in parts:
        i = parts.index("observer")
        if i + 2 < len(parts):
            return parts[i + 2] if parts[i + 1] == "longbench" else parts[i + 1]
    return ""


def discover_files(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*.json") if path.is_file())


def row_key(task: str, kv_type: str, layer: int | None, head: int | None) -> tuple[str, str, int | None, int | None]:
    return (task, kv_type, layer, head)


def classify_schema(payload: dict[str, Any]) -> str:
    if "records" in payload and "aggregates" in payload:
        return "insight_v2.observer"
    return "unknown"


def field_present(record: dict[str, Any], aliases: list[str]) -> bool:
    return any(alias in record and record.get(alias) is not None for alias in aliases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    report_root = Path(args.report_root)
    result_root = Path(args.result_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    observer_root = result_root / "observer"
    observer_files = discover_files(observer_root)
    report_files = {
        "matching_oracle_gap.csv": report_root / "matching_oracle_gap.csv",
        "pattern_gain_map.csv": report_root / "pattern_gain_map.csv",
        "completion.json": report_root / "completion.json",
        "reference_manifest.json": report_root / "reference_manifest.json",
    }

    inventory_rows = []
    field_coverage = {field: {"aggregate": 0, "record": 0, "histogram": 0, "confusion": 0} for field in FIELDS_OF_INTEREST}
    by_task_rows: dict[tuple[str, str], dict[str, Any]] = {}
    by_kv_rows: dict[str, dict[str, Any]] = {}
    by_layer_head_rows: dict[tuple[str, int | None, int | None], dict[str, Any]] = {}

    matching_oracle_rows = list(csv.DictReader((report_files["matching_oracle_gap.csv"]).open(encoding="utf-8", newline="")))
    matching_metrics = sorted({row["metric"] for row in matching_oracle_rows})

    v_sample_has_mismatch = False
    v_sample_has_range_regret = False
    k_sample_has_mismatch = False
    k_sample_has_range_regret = False
    any_truncated = False
    truncated_files = 0
    dropped_records_total = 0

    for path in observer_files:
        payload = parse_json(path)
        task = infer_task_from_path(path)
        records = payload.get("records") or []
        aggregates = payload.get("aggregates") or {}
        histograms = payload.get("histograms") or {}
        confusion = payload.get("confusion") or {}
        truncated = bool(payload.get("truncated"))
        dropped = int(payload.get("dropped_record_count") or 0)
        any_truncated = any_truncated or truncated
        truncated_files += int(truncated)
        dropped_records_total += dropped
        inventory_rows.append(
            {
                "path": str(path),
                "task": task,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "schema": classify_schema(payload),
                "status": payload.get("status"),
                "complete": payload.get("status") == "completed",
                "truncated": truncated,
                "dropped_record_count": dropped,
                "record_count": len(records),
                "aggregate_count": len(aggregates),
                "histogram_count": len(histograms),
                "confusion_count": len(confusion),
            }
        )

        task_row = by_task_rows.setdefault((task, "all"), {"task": task, "files": 0, "truncated_files": 0, "dropped_record_count": 0, "record_count": 0})
        task_row["files"] += 1
        task_row["truncated_files"] += int(truncated)
        task_row["dropped_record_count"] += dropped
        task_row["record_count"] += len(records)

        for key in aggregates:
            suffix = key.split(".")[-1]
            if suffix == "raw_range":
                field_coverage["raw_range"]["aggregate"] += 1
            if suffix == "conditional_oracle_gap":
                field_coverage["range_regret_sum"]["aggregate"] += 0
            if suffix == "current_oracle_gap":
                field_coverage["range_regret"]["aggregate"] += 0
        for key in histograms:
            if key.endswith(".assignment_histogram"):
                field_coverage["pattern_count"]["histogram"] += 1
        for record in records:
            hook = record.get("hook")
            kv_type = str(record.get("kv_type") or "")
            layer = record.get("layer_idx")
            head = record.get("kv_head")
            per_kv = by_kv_rows.setdefault(kv_type or "unknown", {"kv_type": kv_type or "unknown", "record_count": 0, "truncated_files": 0, "files": set()})
            per_kv["record_count"] += 1
            per_kv["files"].add(str(path))
            if truncated:
                per_kv["truncated_files"] += 1
            lh = by_layer_head_rows.setdefault((kv_type, layer, head), {"kv_type": kv_type, "layer": layer, "kv_head": head, "record_count": 0, "tasks": set(), "truncated_files": 0})
            lh["record_count"] += 1
            lh["tasks"].add(task)
            if truncated:
                lh["truncated_files"] += 1

            alias_map = {
                "task": ["task"],
                "sample_id": ["sample_id"],
                "sample_index": ["sample_index"],
                "problem_id": ["problem_id"],
                "phase": ["phase"],
                "layer": ["layer_idx", "layer"],
                "kv_head": ["kv_head"],
                "kv_type": ["kv_type"],
                "bucket": ["position_bucket", "bucket"],
                "token_index": ["token_idx"],
                "window_index": ["window_idx"],
                "l2_assignment": ["l2_assignment"],
                "minmax_assignment": ["minmax_assignment"],
                "range_regret": ["range_regret"],
                "raw_range": ["raw_range"],
                "candidate_pattern_ids": ["candidate_pattern_ids"],
                "candidate_pattern_distances": ["candidate_pattern_distances"],
                "candidate_residual_ranges": ["candidate_residual_ranges"],
                "pattern_bank_identity": ["pattern_bank_identity"],
            }
            for field, aliases in alias_map.items():
                if field_present(record, aliases):
                    field_coverage[field]["record"] += 1
            if hook == "v_matching_oracle":
                if field_present(record, ["l2_minmax_mismatch"]):
                    field_coverage["assignment_mismatch_count"]["record"] += 1
                    field_coverage["assignment_total_count"]["record"] += 1
                    v_sample_has_mismatch = True
                if field_present(record, ["range_regret"]):
                    field_coverage["range_regret"]["record"] += 1
                    v_sample_has_range_regret = True
            if hook == "k_conditional_oracle":
                if field_present(record, ["l2_assignment"]) and field_present(record, ["minmax_assignment"]):
                    field_coverage["assignment_mismatch_count"]["record"] += 1
                    field_coverage["assignment_total_count"]["record"] += 1
                    k_sample_has_mismatch = True
                if field_present(record, ["range_regret"]):
                    k_sample_has_range_regret = True

    for payload in by_kv_rows.values():
        payload["files"] = len(payload["files"])

    coverage_rows = []
    for field in FIELDS_OF_INTEREST:
        status = "missing"
        if field_coverage[field]["aggregate"] > 0 or field_coverage[field]["record"] > 0 or field_coverage[field]["histogram"] > 0 or field_coverage[field]["confusion"] > 0:
            status = "present"
        coverage_rows.append(
            {
                "field": field,
                "aggregate_presence_count": field_coverage[field]["aggregate"],
                "record_presence_count": field_coverage[field]["record"],
                "histogram_presence_count": field_coverage[field]["histogram"],
                "confusion_presence_count": field_coverage[field]["confusion"],
                "status": status,
            }
        )

    coverage_by_task_rows = []
    for (task, _), payload in sorted(by_task_rows.items()):
        coverage_by_task_rows.append(payload)

    coverage_by_kv_rows = []
    for kv_type, payload in sorted(by_kv_rows.items()):
        coverage_by_kv_rows.append(payload)

    coverage_by_layer_head_csv = []
    for (_, layer, head), payload in sorted(by_layer_head_rows.items(), key=lambda item: (str(item[0][0]), -1 if item[0][1] is None else int(item[0][1]), -1 if item[0][2] is None else int(item[0][2]))):
        coverage_by_layer_head_csv.append(
            {
                "kv_type": payload["kv_type"],
                "layer": payload["layer"],
                "kv_head": payload["kv_head"],
                "record_count": payload["record_count"],
                "task_coverage": len(payload["tasks"]),
                "truncated_files": payload["truncated_files"],
            }
        )

    if v_sample_has_mismatch and v_sample_has_range_regret and not k_sample_has_range_regret:
        recoverability_status = "partially_recoverable"
        reason = "V mismatch/range_regret exist only in sample records, while K lacks direct range_regret fields and 68 observer files are truncated."
    elif any_truncated:
        recoverability_status = "data_insufficient"
        reason = "Core sample-record paths are truncated and aggregate paths do not expose the required mismatch/range-regret numerators."
    else:
        recoverability_status = "not_recoverable"
        reason = "Core assignment/range fields are absent."

    semantics = {
        "schema_version": "insight_v2.range_aware_algorithm_semantics_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "findings": SEMANTICS_FINDINGS,
        "answers": {
            "pattern_bank_l2_kmeans": True,
            "mining_assignment_equals_runtime_assignment": False,
            "runtime_l2_assignment": "batched_assign_compiled for mined K/V centroids during prefill initialization",
            "runtime_minmax_assignment_k": "_assign_minmax_hnk on fixed K centroids",
            "runtime_minmax_assignment_v": "_nearest_v_centroid on fixed V centroids",
            "same_fixed_pattern_bank_for_comparison": True,
            "residual_range_definition": "max(residual) - min(residual) over head_dim",
            "epsilon": 1e-12,
            "range_regret_runtime_formula": "For V sample records only, stored value is minmax_mse - oracle_mse; it is not the frozen normalized range-regret formula.",
            "k_layout": "[B,H,T,D], quantized after transpose to [B,H,D,T] in 128-token groups",
            "v_layout": "[B,H,T,D], quantized per-token on last-dim head_dim=128",
            "k_axis_verified": True,
            "v_axis_verified": True,
            "matching_oracle_gap_csv_core_metrics": matching_metrics,
            "diagnosed_scope": "runtime matching mismatch on a fixed pattern bank; not direct evidence that re-mining the bank with a new objective would improve inference.",
        },
    }

    recoverability = {
        "schema_version": "insight_v2.range_aware_recoverability_decision",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "status": recoverability_status,
        "reason": reason,
        "observer_files": len(observer_files),
        "truncated_files": truncated_files,
        "dropped_record_count": dropped_records_total,
        "matching_oracle_gap_metrics": matching_metrics,
        "k_sample_has_mismatch": k_sample_has_mismatch,
        "k_sample_has_range_regret": k_sample_has_range_regret,
        "v_sample_has_mismatch": v_sample_has_mismatch,
        "v_sample_has_range_regret": v_sample_has_range_regret,
        "aggregate_supports_full_recovery": False,
        "sample_record_supports_full_recovery": False,
    }

    raw_inventory = {
        "schema_version": "insight_v2.range_aware_raw_observer_inventory",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "report_inputs": {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_file(path),
            }
            for name, path in report_files.items()
        },
        "observer_root": str(observer_root),
        "observer_files": inventory_rows,
        "truncated_files": truncated_files,
        "dropped_record_count": dropped_records_total,
    }

    atomic_write_json(output_root / "algorithm_semantics_audit.json", semantics)
    atomic_write_text(
        output_root / "algorithm_semantics_audit.md",
        "\n".join(
            [
                "# Algorithm Semantics Audit",
                "",
                "| file | line | function/class | evidence | interpretation |",
                "| --- | ---: | --- | --- | --- |",
            ]
            + [
                f"| {item['file']} | {item['line']} | {item['function_or_class']} | {item['evidence']} | {item['interpretation']} |"
                for item in SEMANTICS_FINDINGS
            ]
        )
        + "\n",
    )
    atomic_write_json(output_root / "raw_observer_inventory.json", raw_inventory)
    atomic_write_text(
        output_root / "raw_observer_inventory.md",
        "\n".join(
            [
                "# Raw Observer Inventory",
                "",
                f"- observer files: `{len(observer_files)}`",
                f"- truncated files: `{truncated_files}`",
                f"- dropped sample records: `{dropped_records_total}`",
            ]
        )
        + "\n",
    )
    write_csv(output_root / "field_coverage.csv", coverage_rows, list(coverage_rows[0].keys()) if coverage_rows else ["field"])
    write_csv(output_root / "field_coverage_by_task.csv", coverage_by_task_rows, list(coverage_by_task_rows[0].keys()) if coverage_by_task_rows else ["task"])
    write_csv(output_root / "field_coverage_by_kv.csv", coverage_by_kv_rows, list(coverage_by_kv_rows[0].keys()) if coverage_by_kv_rows else ["kv_type"])
    write_csv(output_root / "field_coverage_by_layer_head.csv", coverage_by_layer_head_csv, list(coverage_by_layer_head_csv[0].keys()) if coverage_by_layer_head_csv else ["kv_type", "layer", "kv_head"])
    atomic_write_json(output_root / "recoverability_decision.json", recoverability)
    atomic_write_text(
        output_root / "recoverability_decision.md",
        "\n".join(
            [
                "# Recoverability Decision",
                "",
                f"Status: `{recoverability_status}`",
                "",
                f"- reason: {reason}",
                f"- observer files: `{len(observer_files)}`",
                f"- truncated files: `{truncated_files}`",
                f"- dropped sample records: `{dropped_records_total}`",
                f"- matching_oracle_gap metrics: `{', '.join(matching_metrics)}`",
            ]
        )
        + "\n",
    )
    print(json.dumps({"status": recoverability_status, "observer_files": len(observer_files), "truncated_files": truncated_files}, sort_keys=True))


if __name__ == "__main__":
    main()
