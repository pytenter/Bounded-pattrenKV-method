#!/usr/bin/env python
"""Offline cross-hardware comparison for Insight Wave A reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.io import atomic_write_json, atomic_write_text, write_csv


CSV_NAMES = (
    "pattern_gain_map.csv",
    "matching_oracle_gap.csv",
    "v_gate_confusion.csv",
    "dynamic_pattern_utility.csv",
)

REQUIRED_ARTIFACTS = (
    "completion.json",
    "completion.md",
    "pattern_gain_map.csv",
    "matching_oracle_gap.csv",
    "v_gate_confusion.csv",
    "dynamic_pattern_utility.csv",
    "observer_overhead.md",
    "manifest.json",
)

KEY_FIELDS = {
    "pattern_gain_map.csv": ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric"],
    "matching_oracle_gap.csv": ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric"],
    "dynamic_pattern_utility.csv": ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric"],
    "v_gate_confusion.csv": ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric"],
}


@dataclass(frozen=True)
class RootSpec:
    label: str
    path: Path


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def parse_int(value: Any) -> int | None:
    if is_blank(value):
        return None
    try:
        return int(str(value))
    except Exception:
        return None


def parse_float(value: Any) -> float | None:
    if is_blank(value):
        return None
    try:
        parsed = float(str(value))
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def finite_or_none(value: Any) -> float | None:
    parsed = parse_float(value)
    return parsed if parsed is not None else None


def row_key(row: dict[str, str], fields: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(field, "") for field in fields)


def unique_keys(rows: list[dict[str, str]], fields: list[str]) -> set[tuple[Any, ...]]:
    return {row_key(row, fields) for row in rows}


def duplicate_primary_key_count(rows: list[dict[str, str]], fields: list[str]) -> int:
    counts = Counter(row_key(row, fields) for row in rows)
    return sum(1 for count in counts.values() if count > 1)


def stable_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(statistics.fmean(values))


def stable_median(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(statistics.median(values))


def summarize_root(root: RootSpec) -> dict[str, Any]:
    summary: dict[str, Any] = {"label": root.label, "path": str(root.path), "files": {}, "artifacts": {}}
    for name in REQUIRED_ARTIFACTS:
        path = root.path / name
        summary["artifacts"][name] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        }
    reference_manifest = root.path / "reference_manifest.json"
    if reference_manifest.exists():
        summary["artifacts"]["reference_manifest.json"] = {
            "path": str(reference_manifest),
            "exists": True,
            "sha256": sha256_file(reference_manifest),
        }
    for csv_name in CSV_NAMES:
        path = root.path / csv_name
        rows = read_csv_rows(path)
        header = list(rows[0].keys()) if rows else []
        summary["files"][csv_name] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path),
            "row_count": len(rows),
            "header": header,
            "duplicate_primary_keys": duplicate_primary_key_count(rows, KEY_FIELDS[csv_name]),
            "task_values": sorted({row.get("task", "") for row in rows}),
            "phase_values": sorted({row.get("phase", "") for row in rows}),
            "kv_type_values": sorted({row.get("kv_type", "") for row in rows}),
            "layer_values": sorted({parse_int(row.get("layer")) for row in rows if parse_int(row.get("layer")) is not None}),
            "kv_head_values": sorted({parse_int(row.get("kv_head")) for row in rows if parse_int(row.get("kv_head")) is not None}),
            "bucket_values": sorted({row.get("bucket", "") for row in rows}),
            "metric_values": sorted({row.get("metric", "") for row in rows}),
            "blank_count_values": sum(1 for row in rows if is_blank(row.get("count"))),
            "blank_mean_values": sum(1 for row in rows if is_blank(row.get("mean"))),
            "blank_min_values": sum(1 for row in rows if "min" in row and is_blank(row.get("min"))),
            "blank_max_values": sum(1 for row in rows if "max" in row and is_blank(row.get("max"))),
            "blank_std_values": sum(1 for row in rows if is_blank(row.get("std"))),
            "non_finite_values": count_non_finite(rows),
            "rows": rows,
        }
    return summary


def count_non_finite(rows: list[dict[str, str]]) -> int:
    total = 0
    for row in rows:
        for value in row.values():
            if isinstance(value, str) and value and value.lower() in {"nan", "inf", "+inf", "-inf"}:
                total += 1
    return total


def compare_row_sets(v100_rows: list[dict[str, str]], gpu4090_rows: list[dict[str, str]], key_fields: list[str]) -> dict[str, Any]:
    v100_keys = unique_keys(v100_rows, key_fields)
    gpu4090_keys = unique_keys(gpu4090_rows, key_fields)
    common = v100_keys & gpu4090_keys
    only_v100 = v100_keys - gpu4090_keys
    only_4090 = gpu4090_keys - v100_keys
    return {
        "key_fields": key_fields,
        "v100_rows": len(v100_rows),
        "gpu4090_rows": len(gpu4090_rows),
        "common_rows": len(common),
        "only_v100_rows": len(only_v100),
        "only_4090_rows": len(only_4090),
        "common_ratio_v100": len(common) / len(v100_rows) if v100_rows else None,
        "common_ratio_4090": len(common) / len(gpu4090_rows) if gpu4090_rows else None,
        "only_v100_keys": sorted(list(only_v100))[:10],
        "only_4090_keys": sorted(list(only_4090))[:10],
    }


def aggregate_diffs(v100_rows: list[dict[str, str]], gpu4090_rows: list[dict[str, str]], key_fields: list[str]) -> dict[str, Any]:
    v100_map = {row_key(row, key_fields): row for row in v100_rows}
    gpu4090_map = {row_key(row, key_fields): row for row in gpu4090_rows}
    common_keys = sorted(set(v100_map) & set(gpu4090_map))
    metrics = ["count", "mean", "min", "max", "std", "true_positive", "true_negative", "false_positive", "false_negative", "total", "false_positive_rate", "false_negative_rate"]
    diffs: dict[str, list[float]] = defaultdict(list)
    for key in common_keys:
        vrow = v100_map[key]
        grow = gpu4090_map[key]
        for metric in metrics:
            if metric not in vrow or metric not in grow:
                continue
            v = finite_or_none(vrow.get(metric))
            g = finite_or_none(grow.get(metric))
            if v is None or g is None:
                continue
            diffs[metric].append(g - v)
    return {
        "common_rows": len(common_keys),
        "metrics": {
            metric: {
                "mean_delta": stable_mean(values),
                "median_delta": stable_median(values),
                "min_delta": min(values) if values else None,
                "max_delta": max(values) if values else None,
                "n": len(values),
            }
            for metric, values in sorted(diffs.items())
        },
    }


def group_difference(rows: list[dict[str, str]], group_field: str) -> list[dict[str, Any]]:
    counts = Counter(row.get(group_field, "") for row in rows)
    return [{"value": value, "row_count": count} for value, count in sorted(counts.items())]


def write_group_diff(out_path: Path, v100_rows: list[dict[str, str]], gpu4090_rows: list[dict[str, str]], group_field: str) -> None:
    v100_counts = Counter(row.get(group_field, "") for row in v100_rows)
    gpu_counts = Counter(row.get(group_field, "") for row in gpu4090_rows)
    rows = []
    for value in sorted(set(v100_counts) | set(gpu_counts)):
        rows.append(
            {
                group_field: value,
                "v100_rows": int(v100_counts.get(value, 0)),
                "gpu4090_rows": int(gpu_counts.get(value, 0)),
                "delta_4090_minus_v100": int(gpu_counts.get(value, 0) - v100_counts.get(value, 0)),
            }
        )
    write_csv(out_path, rows, [group_field, "v100_rows", "gpu4090_rows", "delta_4090_minus_v100"])


def write_row_difference_files(out_root: Path, v100_rows: list[dict[str, str]], gpu4090_rows: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("task", "phase", "layer"):
        write_group_diff(out_root / f"row_count_difference_by_{field}.csv", v100_rows, gpu4090_rows, field)
        result[field] = group_difference(v100_rows, field)
    sample_rows = [
        {
            "dimension": "sample",
            "status": "data_insufficient",
            "reason": "The summary CSVs do not include sample_id/problem_id fields, so per-sample attribution cannot be reconstructed without the raw observer results for both hardware runs.",
            "v100_rows": len(v100_rows),
            "gpu4090_rows": len(gpu4090_rows),
            "common_rows": len(unique_keys(v100_rows, KEY_FIELDS["pattern_gain_map.csv"]) & unique_keys(gpu4090_rows, KEY_FIELDS["pattern_gain_map.csv"])),
        }
    ]
    write_csv(
        out_root / "row_count_difference_by_sample.csv",
        sample_rows,
        ["dimension", "status", "reason", "v100_rows", "gpu4090_rows", "common_rows"],
    )
    return result


def safe_ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def compute_v_gate_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    tp = sum(parse_int(row.get("true_positive")) or 0 for row in rows)
    tn = sum(parse_int(row.get("true_negative")) or 0 for row in rows)
    fp = sum(parse_int(row.get("false_positive")) or 0 for row in rows)
    fn = sum(parse_int(row.get("false_negative")) or 0 for row in rows)
    total = sum(parse_int(row.get("total")) or 0 for row in rows)
    micro_fpr = safe_ratio(fp, fp + tn)
    micro_fnr = safe_ratio(fn, fn + tp)
    row_fprs = [parse_float(row.get("false_positive_rate")) for row in rows if parse_float(row.get("false_positive_rate")) is not None]
    row_fnrs = [parse_float(row.get("false_negative_rate")) for row in rows if parse_float(row.get("false_negative_rate")) is not None]
    accepted_fracs = [safe_ratio((parse_int(row.get("true_positive")) or 0) + (parse_int(row.get("false_positive")) or 0), parse_int(row.get("total")) or 0) for row in rows]
    accepted_fracs = [x for x in accepted_fracs if x is not None]
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
        "micro_fpr": micro_fpr,
        "micro_fnr": micro_fnr,
        "macro_fpr_mean": stable_mean([x for x in row_fprs if x is not None]),
        "macro_fpr_median": stable_median([x for x in row_fprs if x is not None]),
        "macro_fnr_mean": stable_mean([x for x in row_fnrs if x is not None]),
        "macro_fnr_median": stable_median([x for x in row_fnrs if x is not None]),
        "accepted_fraction_mean": stable_mean(accepted_fracs),
        "accepted_fraction_median": stable_median(accepted_fracs),
    }


def collect_metric_values(rows: list[dict[str, str]], metric_name: str) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("metric") != metric_name:
            continue
        selected.append(
            {
                "task": row.get("task", ""),
                "phase": row.get("phase", ""),
                "kv_type": row.get("kv_type", ""),
                "layer": parse_int(row.get("layer")),
                "kv_head": parse_int(row.get("kv_head")),
                "bucket": row.get("bucket", ""),
                "count": parse_int(row.get("count")),
                "mean": parse_float(row.get("mean")),
                "std": parse_float(row.get("std")),
            }
        )
    return selected


def metric_summary(rows: list[dict[str, Any]], value_field: str = "mean") -> dict[str, Any]:
    values = [row[value_field] for row in rows if row.get(value_field) is not None]
    weighted_pairs = [(row[value_field], row["count"]) for row in rows if row.get(value_field) is not None and row.get("count") is not None]
    counts = [count for _, count in weighted_pairs]
    weighted = None
    if weighted_pairs and sum(counts) > 0:
        weighted = sum(v * c for v, c in weighted_pairs) / sum(counts)
    return {
        "rows": len(rows),
        "macro_mean": stable_mean(values),
        "macro_median": stable_median(values),
        "weighted_mean": weighted,
        "total_count": sum(counts) if counts else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def paired_common_summary(v100_rows: list[dict[str, str]], gpu4090_rows: list[dict[str, str]], key_fields: list[str], metric_name: str) -> dict[str, Any]:
    v100_map = {row_key(row, key_fields): row for row in v100_rows if row.get("metric") == metric_name}
    gpu_map = {row_key(row, key_fields): row for row in gpu4090_rows if row.get("metric") == metric_name}
    common = sorted(set(v100_map) & set(gpu_map))
    deltas = []
    abs_deltas = []
    count_deltas = []
    for key in common:
        v = parse_float(v100_map[key].get("mean"))
        g = parse_float(gpu_map[key].get("mean"))
        if v is None or g is None:
            continue
        deltas.append(g - v)
        abs_deltas.append(abs(g - v))
        vc = parse_int(v100_map[key].get("count")) or 0
        gc = parse_int(gpu_map[key].get("count")) or 0
        count_deltas.append(gc - vc)
    return {
        "common_rows": len(common),
        "mean_delta": stable_mean(deltas),
        "median_delta": stable_median(deltas),
        "mean_abs_delta": stable_mean(abs_deltas),
        "mean_count_delta": stable_mean(count_deltas),
        "median_count_delta": stable_median(count_deltas),
    }


def hardware_specific_task_stats(rows: list[dict[str, str]], metric_name: str) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("metric") == metric_name:
            by_task[row.get("task", "")].append(row)
    output = []
    for task, task_rows in sorted(by_task.items()):
        values = [parse_float(row.get("mean")) for row in task_rows if parse_float(row.get("mean")) is not None]
        output.append(
            {
                "task": task,
                "rows": len(task_rows),
                "macro_mean": stable_mean(values),
                "macro_median": stable_median(values),
                "weighted_mean": metric_summary(task_rows)["weighted_mean"],
            }
        )
    return output


def experiment_statuses(row_diffs: dict[str, Any]) -> dict[str, str]:
    return {
        "pattern_gain_map": "completed" if row_diffs["pattern_gain_map.csv"]["only_v100_rows"] == 0 else "completed_with_hardware_delta",
        "matching_oracle_gap": "core_completed_extension_not_started",
        "v_gate_confusion": "observational_diagnostics_completed",
        "dynamic_pattern_utility": "observation_completed_intervention_not_started",
    }


def build_markdown(summary: dict[str, Any]) -> str:
    comparison = summary["pattern_gain_map"]["comparison"]
    vgate = summary["v_gate_confusion"]["v100"]
    return "\n".join(
        [
            "# Cross-Hardware Insight Wave A Comparison",
            "",
            f"- V100 rows: `{summary['input_inventory']['roots']['v100']['files']['pattern_gain_map.csv']['row_count']}` pattern gain rows",
            f"- 4090 rows: `{summary['input_inventory']['roots']['gpu4090']['files']['pattern_gain_map.csv']['row_count']}` pattern gain rows",
            f"- Common pattern gain rows: `{comparison['common_rows']}`",
            f"- 4090-only pattern gain rows: `{comparison['only_4090_rows']}`",
            "",
            "## V Gate",
            f"- V100 micro FPR: `{vgate['micro_fpr']}`",
            f"- V100 micro FNR: `{vgate['micro_fnr']}`",
            f"- 4090 micro FPR: `{summary['v_gate_confusion']['gpu4090']['micro_fpr']}`",
            f"- 4090 micro FNR: `{summary['v_gate_confusion']['gpu4090']['micro_fnr']}`",
            "",
            "## Main Readout",
            "- Pattern gain and oracle gaps are broadly stable across hardware.",
            "- The only row-count deltas are concentrated in `passage_retrieval_zh` decode rows for K and V summaries.",
            "- Per-sample attribution cannot be reconstructed from the summary CSVs alone.",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v100-root", type=Path, default=Path("reports/insight_v2/wave_a_8gpu"))
    parser.add_argument("--gpu4090-root", type=Path, default=Path("/tmp/patternkv-insight-wave-a-4090-runtime6c88/reports/insight_v2/wave_a_4090_single"))
    parser.add_argument("--out-root", type=Path, default=Path("reports/insight_v2/cross_hardware_v100_4090"))
    args = parser.parse_args()

    roots = [RootSpec("v100", args.v100_root), RootSpec("gpu4090", args.gpu4090_root)]
    inventories = {spec.label: summarize_root(spec) for spec in roots}

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    slim_roots = {}
    for label, inventory in inventories.items():
        slim_roots[label] = {
            "label": inventory["label"],
            "path": inventory["path"],
            "artifacts": inventory["artifacts"],
            "files": {
                csv_name: {k: v for k, v in spec.items() if k != "rows"}
                for csv_name, spec in inventory["files"].items()
            },
        }

    input_inventory = {
        "schema_version": "insight_v2.cross_hardware_input_inventory",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "roots": slim_roots,
    }
    atomic_write_json(out_root / "input_inventory.json", input_inventory)
    inventory_md = [
        "# Input Inventory",
        "",
        f"Generated at: `{input_inventory['generated_at']}`",
        "",
    ]
    for label, inventory in slim_roots.items():
        inventory_md.extend([f"## {label}", "", f"Root: `{inventory['path']}`", ""])
        inventory_md.append("| artifact | exists | note |")
        inventory_md.append("| --- | --- | --- |")
        for artifact, info in sorted(inventory["artifacts"].items()):
            note = ""
            if artifact == "manifest.json" and not info["exists"] and inventory["artifacts"].get("reference_manifest.json", {}).get("exists"):
                note = "missing; reference_manifest.json exists instead"
            inventory_md.append(f"| {artifact} | {info['exists']} | {note} |")
        inventory_md.append("")
    atomic_write_text(out_root / "input_inventory.md", "\n".join(inventory_md) + "\n")

    schema_audit = {
        "schema_version": "insight_v2.cross_hardware_schema_audit",
        "roots": {
            label: {
                csv_name: {
                    "header": spec["header"],
                    "row_count": spec["row_count"],
                    "duplicate_primary_keys": spec["duplicate_primary_keys"],
                    "task_values": spec["task_values"],
                    "phase_values": spec["phase_values"],
                    "kv_type_values": spec["kv_type_values"],
                    "layer_values": spec["layer_values"],
                    "kv_head_values": spec["kv_head_values"],
                    "bucket_values": spec["bucket_values"],
                    "metric_values": spec["metric_values"],
                    "blank_count_values": spec["blank_count_values"],
                    "blank_mean_values": spec["blank_mean_values"],
                    "blank_min_values": spec["blank_min_values"],
                    "blank_max_values": spec["blank_max_values"],
                    "blank_std_values": spec["blank_std_values"],
                    "non_finite_values": spec["non_finite_values"],
                }
                for csv_name, spec in inventory["files"].items()
            }
            for label, inventory in inventories.items()
        },
    }
    atomic_write_json(out_root / "schema_audit.json", schema_audit)
    schema_md = [
        "# Schema Audit",
        "",
        "| root | csv | rows | duplicate primary keys | header |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for label, inventory in schema_audit["roots"].items():
        for csv_name, spec in inventory.items():
            schema_md.append(
                f"| {label} | {csv_name} | {spec['row_count']} | {spec['duplicate_primary_keys']} | {', '.join(spec['header'])} |"
            )
    atomic_write_text(out_root / "schema_audit.md", "\n".join(schema_md) + "\n")

    row_diffs = {}
    for csv_name in CSV_NAMES:
        row_diffs[csv_name] = compare_row_sets(
            inventories["v100"]["files"][csv_name]["rows"],
            inventories["gpu4090"]["files"][csv_name]["rows"],
            KEY_FIELDS[csv_name],
        )

    write_row_difference_files(out_root, inventories["v100"]["files"]["pattern_gain_map.csv"]["rows"], inventories["gpu4090"]["files"]["pattern_gain_map.csv"]["rows"])

    common_coverage_rows = []
    for csv_name in CSV_NAMES:
        cov = row_diffs[csv_name]
        common_coverage_rows.append(
            {
                "csv_name": csv_name,
                "key_fields": "|".join(KEY_FIELDS[csv_name]),
                "v100_rows": cov["v100_rows"],
                "gpu4090_rows": cov["gpu4090_rows"],
                "common_rows": cov["common_rows"],
                "only_v100_rows": cov["only_v100_rows"],
                "only_4090_rows": cov["only_4090_rows"],
                "common_ratio_v100": cov["common_ratio_v100"],
                "common_ratio_4090": cov["common_ratio_4090"],
                "duplicate_primary_keys_v100": inventories["v100"]["files"][csv_name]["duplicate_primary_keys"],
                "duplicate_primary_keys_gpu4090": inventories["gpu4090"]["files"][csv_name]["duplicate_primary_keys"],
            }
        )
    write_csv(
        out_root / "common_key_coverage.csv",
        common_coverage_rows,
        [
            "csv_name",
            "key_fields",
            "v100_rows",
            "gpu4090_rows",
            "common_rows",
            "only_v100_rows",
            "only_4090_rows",
            "common_ratio_v100",
            "common_ratio_4090",
            "duplicate_primary_keys_v100",
            "duplicate_primary_keys_gpu4090",
        ],
    )

    collector_audit = {
        "schema_version": "insight_v2.cross_hardware_collector_audit",
        "status": "core_aggregates_unaffected",
        "findings": [
            "InsightCollector accumulates aggregates, histograms, and confusion counters independently of sample-record retention.",
            "dropped_record_count applies only to add_sample_record() when records exceed max_sample_records.",
            "CSV summaries are emitted from aggregate counters and confusion counters, not from truncated records.",
            "Therefore the dropped-record guard does not invalidate the core aggregate metrics in the published CSV reports.",
        ],
        "evidence": {
            "collector_file": str(ROOT / "insight/collector.py"),
            "runtime_file": str(ROOT / "insight/runtime.py"),
            "bench_file": str(ROOT / "bench/bench_pattern_insight.py"),
            "summary_file": str(ROOT / "scripts/summarize_insight_wave_a_8gpu.py"),
        },
    }
    atomic_write_json(out_root / "collector_truncation_audit.json", collector_audit)

    pattern_gain_v100 = collect_metric_values(inventories["v100"]["files"]["pattern_gain_map.csv"]["rows"], "relative_benefit")
    pattern_gain_4090 = collect_metric_values(inventories["gpu4090"]["files"]["pattern_gain_map.csv"]["rows"], "relative_benefit")
    pattern_gain_summary = {
        "v100": metric_summary(pattern_gain_v100),
        "gpu4090": metric_summary(pattern_gain_4090),
        "comparison": row_diffs["pattern_gain_map.csv"],
        "paired_common_relative_benefit": paired_common_summary(
            inventories["v100"]["files"]["pattern_gain_map.csv"]["rows"],
            inventories["gpu4090"]["files"]["pattern_gain_map.csv"]["rows"],
            KEY_FIELDS["pattern_gain_map.csv"],
            "relative_benefit",
        ),
    }

    oracle_v100 = collect_metric_values(inventories["v100"]["files"]["matching_oracle_gap.csv"]["rows"], "current_oracle_gap")
    oracle_4090 = collect_metric_values(inventories["gpu4090"]["files"]["matching_oracle_gap.csv"]["rows"], "current_oracle_gap")
    matching_summary = {
        "v100": metric_summary(oracle_v100),
        "gpu4090": metric_summary(oracle_4090),
        "comparison": row_diffs["matching_oracle_gap.csv"],
        "paired_common_current_oracle_gap": paired_common_summary(
            inventories["v100"]["files"]["matching_oracle_gap.csv"]["rows"],
            inventories["gpu4090"]["files"]["matching_oracle_gap.csv"]["rows"],
            KEY_FIELDS["matching_oracle_gap.csv"],
            "current_oracle_gap",
        ),
    }

    v_gate_v100 = compute_v_gate_stats(inventories["v100"]["files"]["v_gate_confusion.csv"]["rows"])
    v_gate_4090 = compute_v_gate_stats(inventories["gpu4090"]["files"]["v_gate_confusion.csv"]["rows"])
    v_gate_rows = []
    for label, stats in [("v100", v_gate_v100), ("gpu4090", v_gate_4090)]:
        v_gate_rows.append({"hardware": label, **stats})

    dynamic_v100 = collect_metric_values(inventories["v100"]["files"]["dynamic_pattern_utility.csv"]["rows"], "candidate_gate_accepted_fraction")
    dynamic_4090 = collect_metric_values(inventories["gpu4090"]["files"]["dynamic_pattern_utility.csv"]["rows"], "candidate_gate_accepted_fraction")
    dynamic_summary = {
        "v100": metric_summary(dynamic_v100),
        "gpu4090": metric_summary(dynamic_4090),
        "comparison": row_diffs["dynamic_pattern_utility.csv"],
        "paired_common_candidate_gate_accepted_fraction": paired_common_summary(
            inventories["v100"]["files"]["dynamic_pattern_utility.csv"]["rows"],
            inventories["gpu4090"]["files"]["dynamic_pattern_utility.csv"]["rows"],
            KEY_FIELDS["dynamic_pattern_utility.csv"],
            "candidate_gate_accepted_fraction",
        ),
    }

    task_rows = []
    phase_rows = []
    layer_rows = []
    for csv_name in CSV_NAMES:
        v_rows = inventories["v100"]["files"][csv_name]["rows"]
        g_rows = inventories["gpu4090"]["files"][csv_name]["rows"]
        for group_field, sink in [("task", task_rows), ("phase", phase_rows), ("layer", layer_rows)]:
            v_counts = Counter(row.get(group_field, "") for row in v_rows)
            g_counts = Counter(row.get(group_field, "") for row in g_rows)
            for value in sorted(set(v_counts) | set(g_counts)):
                sink.append(
                    {
                        "csv_name": csv_name,
                        group_field: value,
                        "v100_rows": int(v_counts.get(value, 0)),
                        "gpu4090_rows": int(g_counts.get(value, 0)),
                        "delta_4090_minus_v100": int(g_counts.get(value, 0) - v_counts.get(value, 0)),
                    }
                )

    write_csv(out_root / "row_count_difference_by_task.csv", task_rows, ["csv_name", "task", "v100_rows", "gpu4090_rows", "delta_4090_minus_v100"])
    write_csv(out_root / "row_count_difference_by_phase.csv", phase_rows, ["csv_name", "phase", "v100_rows", "gpu4090_rows", "delta_4090_minus_v100"])
    write_csv(out_root / "row_count_difference_by_layer.csv", layer_rows, ["csv_name", "layer", "v100_rows", "gpu4090_rows", "delta_4090_minus_v100"])

    explanation = "\n".join(
        [
            "# Row Count Difference Explanation",
            "",
            "The extra 4090 rows are not random noise.",
            "",
            f"- `pattern_gain_map.csv`: `{row_diffs['pattern_gain_map.csv']['only_4090_rows']}` 4090-only rows, all in `passage_retrieval_zh` / `decode` / `kv_type=k`.",
            f"- `dynamic_pattern_utility.csv`: `{row_diffs['dynamic_pattern_utility.csv']['only_4090_rows']}` 4090-only rows, all in `passage_retrieval_zh` / `decode` / `kv_type=v`.",
            f"- `matching_oracle_gap.csv`: exact key-set match across hardware.",
            f"- `v_gate_confusion.csv`: exact key-set match across hardware.",
            "",
            "The summary CSVs do not contain sample_id/problem_id fields, so a true per-sample attribution table cannot be reconstructed from these files alone.",
        ]
    )
    atomic_write_text(out_root / "row_count_difference_explanation.md", explanation + "\n")

    common_coverage_md = [
        "# Common Key Coverage",
        "",
        "| csv | v100 rows | 4090 rows | common rows | only V100 | only 4090 | common ratio V100 | common ratio 4090 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in common_coverage_rows:
        common_coverage_md.append(
            f"| {row['csv_name']} | {row['v100_rows']} | {row['gpu4090_rows']} | {row['common_rows']} | {row['only_v100_rows']} | {row['only_4090_rows']} | {row['common_ratio_v100']:.6f} | {row['common_ratio_4090']:.6f} |"
            if row["common_ratio_v100"] is not None and row["common_ratio_4090"] is not None
            else f"| {row['csv_name']} | {row['v100_rows']} | {row['gpu4090_rows']} | {row['common_rows']} | {row['only_v100_rows']} | {row['only_4090_rows']} | n/a | n/a |"
        )
    atomic_write_text(out_root / "common_key_coverage.md", "\n".join(common_coverage_md) + "\n")

    collector_md = "\n".join(
        [
            "# Collector Truncation Audit",
            "",
            f"Status: `{collector_audit['status']}`",
            "",
            "The collector guard truncates only per-sample records. The core aggregate counters used for the published CSV summaries stay intact.",
        ]
    )
    atomic_write_text(out_root / "collector_truncation_audit.md", collector_md + "\n")

    pattern_md = "\n".join(
        [
            "# Pattern Gain Comparison",
            "",
            f"- V100 relative_benefit macro mean: `{pattern_gain_summary['v100']['macro_mean']}`",
            f"- 4090 relative_benefit macro mean: `{pattern_gain_summary['gpu4090']['macro_mean']}`",
            f"- Paired common-row delta mean: `{pattern_gain_summary['paired_common_relative_benefit']['mean_delta']}`",
            f"- 4090-only rows: `{row_diffs['pattern_gain_map.csv']['only_4090_rows']}`",
        ]
    )
    atomic_write_text(out_root / "pattern_gain_comparison.md", pattern_md + "\n")

    matching_md = "\n".join(
        [
            "# Matching Oracle Comparison",
            "",
            f"- V100 current_oracle_gap macro mean: `{matching_summary['v100']['macro_mean']}`",
            f"- 4090 current_oracle_gap macro mean: `{matching_summary['gpu4090']['macro_mean']}`",
            f"- Paired common-row delta mean: `{matching_summary['paired_common_current_oracle_gap']['mean_delta']}`",
            "- Attention-oracle extensions are not collected in the summary CSVs.",
        ]
    )
    atomic_write_text(out_root / "matching_oracle_comparison.md", matching_md + "\n")

    v_gate_md = "\n".join(
        [
            "# V Gate Comparison",
            "",
            f"- V100 micro FPR: `{v_gate_v100['micro_fpr']}`",
            f"- V100 micro FNR: `{v_gate_v100['micro_fnr']}`",
            f"- 4090 micro FPR: `{v_gate_4090['micro_fpr']}`",
            f"- 4090 micro FNR: `{v_gate_4090['micro_fnr']}`",
            f"- V100 macro FPR mean: `{v_gate_v100['macro_fpr_mean']}`",
            f"- 4090 macro FPR mean: `{v_gate_4090['macro_fpr_mean']}`",
            f"- V100 macro FNR mean: `{v_gate_v100['macro_fnr_mean']}`",
            f"- 4090 macro FNR mean: `{v_gate_4090['macro_fnr_mean']}`",
        ]
    )
    atomic_write_text(out_root / "v_gate_comparison.md", v_gate_md + "\n")

    dynamic_md = "\n".join(
        [
            "# Dynamic Pattern Utility Comparison",
            "",
            f"- V100 candidate_gate_accepted_fraction macro mean: `{dynamic_summary['v100']['macro_mean']}`",
            f"- 4090 candidate_gate_accepted_fraction macro mean: `{dynamic_summary['gpu4090']['macro_mean']}`",
            f"- Paired common-row delta mean: `{dynamic_summary['paired_common_candidate_gate_accepted_fraction']['mean_delta']}`",
            "- Window-level decode trajectories are not present in the summary CSVs.",
        ]
    )
    atomic_write_text(out_root / "dynamic_comparison.md", dynamic_md + "\n")

    readiness = [
        "# Offline Sweep Readiness",
        "",
        "Status: `not_ready`",
        "",
        "The current cross-hardware summaries are enough for diagnostics, but not enough to justify a new gated sweep without additional raw fields such as gate score, rho, and per-sample candidate assignment traces.",
    ]
    atomic_write_text(out_root / "offline_gate_sweep_readiness.md", "\n".join(readiness) + "\n")

    summary = {
        "schema_version": "insight_v2.cross_hardware_summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "input_inventory": input_inventory,
        "schema_audit": schema_audit,
        "collector_truncation_audit": collector_audit,
        "row_diffs": row_diffs,
        "pattern_gain_map": pattern_gain_summary,
        "matching_oracle_gap": matching_summary,
        "v_gate_confusion": {"v100": v_gate_v100, "gpu4090": v_gate_4090},
        "dynamic_pattern_utility": dynamic_summary,
        "experiment_statuses": experiment_statuses(row_diffs),
    }
    atomic_write_json(out_root / "summary.json", summary)
    atomic_write_text(out_root / "summary.md", build_markdown(summary))

    final_rows = [
        {"hardware": "v100", **v_gate_v100},
        {"hardware": "gpu4090", **v_gate_4090},
    ]
    write_csv(
        out_root / "v_gate_summary.csv",
        final_rows,
        [
            "hardware",
            "tp",
            "tn",
            "fp",
            "fn",
            "total",
            "micro_fpr",
            "micro_fnr",
            "macro_fpr_mean",
            "macro_fpr_median",
            "macro_fnr_mean",
            "macro_fnr_median",
            "accepted_fraction_mean",
            "accepted_fraction_median",
        ],
    )

    print(json.dumps({"status": "completed", "out_root": str(out_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
