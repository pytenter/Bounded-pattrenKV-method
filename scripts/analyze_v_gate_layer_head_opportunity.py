#!/usr/bin/env python
"""Offline V-gate layer/head opportunity analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.io import atomic_write_json, atomic_write_text, write_csv


POSITIVE_TASKS = {"hotpotqa", "passage_retrieval_en", "passage_retrieval_zh"}
NEGATIVE_TASKS = {"samsum", "dureader"}
GSM8K_TASKS = {"gsm8k"}
TASK_GROUPS = {
    "positive": POSITIVE_TASKS,
    "negative": NEGATIVE_TASKS,
    "gsm8k": GSM8K_TASKS,
}

CSV_NAME = "v_gate_confusion.csv"
COMMON_KEY_FIELDS = ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric"]
AGG_KEY_FIELDS = ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric"]
PAIR_KEY_FIELDS = ["task", "layer", "kv_head"]
LAYER_HEAD_KEY_FIELDS = ["layer", "kv_head"]


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_root(preferred: Path, required_file: str) -> Path:
    candidates = [preferred]
    if not preferred.is_absolute():
        candidates.append(ROOT / preferred)
    candidates.append(Path("/tmp") / preferred.name)
    for base in (ROOT / "reports" / "insight_v2", ROOT / "results" / "insight_v2", Path("/tmp")):
        if base.exists():
            for path in base.rglob(preferred.name):
                candidates.append(path)
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))
        if candidate.exists() and (candidate / required_file).exists():
            return candidate
    return preferred


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
        out = float(str(value))
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def row_key(row: dict[str, Any], fields: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(field, "") for field in fields)


def unique_values(rows: list[dict[str, Any]], field: str) -> list[Any]:
    vals = sorted({row.get(field, "") for row in rows})
    return vals


def duplicate_row_count(rows: list[dict[str, Any]]) -> int:
    counts = Counter(tuple(sorted(row.items())) for row in rows)
    return sum(1 for count in counts.values() if count > 1)


def duplicate_primary_key_count(rows: list[dict[str, Any]], fields: list[str]) -> int:
    counts = Counter(row_key(row, fields) for row in rows)
    return sum(1 for count in counts.values() if count > 1)


def aggregate_rows(rows: list[dict[str, Any]], group_fields: list[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(lambda: {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0, "row_count": 0})
    for row in rows:
        key = row_key(row, group_fields)
        g = groups[key]
        g["row_count"] += 1
        for field in ("true_positive", "true_negative", "false_positive", "false_negative"):
            g[field] += parse_int(row.get(field)) or 0
    return groups


def metric_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, Any]:
    total = tp + tn + fp + fn
    accepted = tp + fp
    pos = tp + fn
    neg = tn + fp
    micro_fpr = fp / neg if neg else None
    micro_fnr = fn / pos if pos else None
    precision = tp / accepted if accepted else None
    recall = tp / pos if pos else None
    acceptance = accepted / total if total else None
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "total": total,
        "candidate_count": total,
        "accepted_count": accepted,
        "positive_support": pos,
        "negative_support": neg,
        "micro_fpr": micro_fpr,
        "micro_fnr": micro_fnr,
        "precision_micro": precision,
        "recall_micro": recall,
        "acceptance_micro": acceptance,
    }


def rate_or_none(num: int, den: int) -> float | None:
    return num / den if den else None


def mean_or_none(values: Iterable[float]) -> float | None:
    vals = list(values)
    return float(statistics.fmean(vals)) if vals else None


def median_or_none(values: Iterable[float]) -> float | None:
    vals = list(values)
    return float(statistics.median(vals)) if vals else None


def p95_or_none(values: Iterable[float]) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    idx = int(math.ceil(0.95 * len(vals))) - 1
    idx = min(max(idx, 0), len(vals) - 1)
    return float(vals[idx])


def bootstrap_ci(
    units: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
    metric_fn,
) -> tuple[float | None, float | None, float | None]:
    if not units:
        return None, None, None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        sampled = [units[rng.randrange(len(units))] for _ in range(len(units))]
        value = metric_fn(sampled)
        if value is not None and math.isfinite(value):
            estimates.append(float(value))
    if not estimates:
        return None, None, None
    estimates.sort()
    lo = estimates[max(0, int(0.025 * len(estimates)))]
    hi = estimates[min(len(estimates) - 1, int(0.975 * len(estimates)))]
    return float(mean_or_none(estimates)), float(lo), float(hi)


def paired_bootstrap_ci(
    left_units: list[dict[str, Any]],
    right_units: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
    metric_fn,
) -> tuple[float | None, float | None, float | None]:
    if not left_units or not right_units or len(left_units) != len(right_units):
        return None, None, None
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(repetitions):
        idxs = [rng.randrange(len(left_units)) for _ in range(len(left_units))]
        left_sample = [left_units[i] for i in idxs]
        right_sample = [right_units[i] for i in idxs]
        lv = metric_fn(left_sample)
        rv = metric_fn(right_sample)
        if lv is None or rv is None:
            continue
        diffs.append(float(rv - lv))
    if not diffs:
        return None, None, None
    diffs.sort()
    lo = diffs[max(0, int(0.025 * len(diffs)))]
    hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
    return float(mean_or_none(diffs)), float(lo), float(hi)


def rank_average(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx = rank_average(x)
    ry = rank_average(y)
    mx = statistics.fmean(rx)
    my = statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (denx * deny) if denx and deny else None


def summarize_unit_counts(unit: dict[str, Any]) -> dict[str, Any]:
    return metric_from_counts(unit["true_positive"], unit["true_negative"], unit["false_positive"], unit["false_negative"])


def build_row(base: dict[str, Any], metric_name: str | None = None) -> dict[str, Any]:
    row = dict(base)
    if metric_name is not None:
        row["metric_name"] = metric_name
    return row


def summarize_group_rows(rows: list[dict[str, Any]], group_fields: list[str], bootstrap_seed: int, bootstrap_repetitions: int) -> list[dict[str, Any]]:
    grouped = aggregate_rows(rows, group_fields)
    output: list[dict[str, Any]] = []
    for key, unit in sorted(grouped.items(), key=lambda item: item[0]):
        stats = summarize_unit_counts(unit)
        unit_list = [unit]
        macro_est, macro_lo, macro_hi = bootstrap_ci(unit_list, repetitions=bootstrap_repetitions, seed=bootstrap_seed, metric_fn=lambda sample: mean_or_none([summarize_unit_counts(x)["micro_fpr"] for x in sample if summarize_unit_counts(x)["micro_fpr"] is not None]))
        fnr_est, fnr_lo, fnr_hi = bootstrap_ci(unit_list, repetitions=bootstrap_repetitions, seed=bootstrap_seed, metric_fn=lambda sample: mean_or_none([summarize_unit_counts(x)["micro_fnr"] for x in sample if summarize_unit_counts(x)["micro_fnr"] is not None]))
        row = {field: value for field, value in zip(group_fields, key)}
        row.update(
            {
                **stats,
                "fpr_estimate": stats["micro_fpr"],
                "fpr_bootstrap_mean": macro_est,
                "fpr_bootstrap_ci_low": macro_lo,
                "fpr_bootstrap_ci_high": macro_hi,
                "fnr_estimate": stats["micro_fnr"],
                "fnr_bootstrap_mean": fnr_est,
                "fnr_bootstrap_ci_low": fnr_lo,
                "fnr_bootstrap_ci_high": fnr_hi,
            }
        )
        output.append(row)
    return output


def task_macro_stats(task_rows: list[dict[str, Any]], bootstrap_seed: int, bootstrap_repetitions: int) -> dict[str, Any]:
    grouped = aggregate_rows(task_rows, ["task"])
    task_units = list(grouped.values())

    def macro_fpr(sample: list[dict[str, Any]]) -> float | None:
        vals = []
        for unit in sample:
            stats = summarize_unit_counts(unit)
            if stats["micro_fpr"] is not None:
                vals.append(stats["micro_fpr"])
        return mean_or_none(vals)

    def macro_fnr(sample: list[dict[str, Any]]) -> float | None:
        vals = []
        for unit in sample:
            stats = summarize_unit_counts(unit)
            if stats["micro_fnr"] is not None:
                vals.append(stats["micro_fnr"])
        return mean_or_none(vals)

    fpr_est, fpr_lo, fpr_hi = bootstrap_ci(task_units, repetitions=bootstrap_repetitions, seed=bootstrap_seed, metric_fn=macro_fpr)
    fnr_est, fnr_lo, fnr_hi = bootstrap_ci(task_units, repetitions=bootstrap_repetitions, seed=bootstrap_seed, metric_fn=macro_fnr)
    task_means_fpr = [summarize_unit_counts(unit)["micro_fpr"] for unit in task_units if summarize_unit_counts(unit)["micro_fpr"] is not None]
    task_means_fnr = [summarize_unit_counts(unit)["micro_fnr"] for unit in task_units if summarize_unit_counts(unit)["micro_fnr"] is not None]
    task_means_acc = [summarize_unit_counts(unit)["acceptance_micro"] for unit in task_units if summarize_unit_counts(unit)["acceptance_micro"] is not None]
    return {
        "task_macro_fpr": mean_or_none(task_means_fpr),
        "task_macro_fnr": mean_or_none(task_means_fnr),
        "task_macro_acceptance": mean_or_none(task_means_acc),
        "task_macro_fpr_bootstrap_mean": fpr_est,
        "task_macro_fpr_bootstrap_ci_low": fpr_lo,
        "task_macro_fpr_bootstrap_ci_high": fpr_hi,
        "task_macro_fnr_bootstrap_mean": fnr_est,
        "task_macro_fnr_bootstrap_ci_low": fnr_lo,
        "task_macro_fnr_bootstrap_ci_high": fnr_hi,
        "support": len(task_units),
    }


def global_stats(rows: list[dict[str, Any]], bootstrap_seed: int, bootstrap_repetitions: int) -> dict[str, Any]:
    grouped = aggregate_rows(rows, ["task", "layer", "kv_head", "bucket"])
    units = list(grouped.values())

    def metric_fpr(sample: list[dict[str, Any]]) -> float | None:
        tp = sum(x["true_positive"] for x in sample)
        tn = sum(x["true_negative"] for x in sample)
        fp = sum(x["false_positive"] for x in sample)
        fn = sum(x["false_negative"] for x in sample)
        return rate_or_none(fp, fp + tn)

    def metric_fnr(sample: list[dict[str, Any]]) -> float | None:
        tp = sum(x["true_positive"] for x in sample)
        tn = sum(x["true_negative"] for x in sample)
        fp = sum(x["false_positive"] for x in sample)
        fn = sum(x["false_negative"] for x in sample)
        return rate_or_none(fn, fn + tp)

    fpr_est, fpr_lo, fpr_hi = bootstrap_ci(units, repetitions=bootstrap_repetitions, seed=bootstrap_seed, metric_fn=metric_fpr)
    fnr_est, fnr_lo, fnr_hi = bootstrap_ci(units, repetitions=bootstrap_repetitions, seed=bootstrap_seed, metric_fn=metric_fnr)
    tp = sum(u["true_positive"] for u in units)
    tn = sum(u["true_negative"] for u in units)
    fp = sum(u["false_positive"] for u in units)
    fn = sum(u["false_negative"] for u in units)
    micro = metric_from_counts(tp, tn, fp, fn)
    return {
        **micro,
        "fpr_bootstrap_mean": fpr_est,
        "fpr_bootstrap_ci_low": fpr_lo,
        "fpr_bootstrap_ci_high": fpr_hi,
        "fnr_bootstrap_mean": fnr_est,
        "fnr_bootstrap_ci_low": fnr_lo,
        "fnr_bootstrap_ci_high": fnr_hi,
        "support": len(units),
    }


def task_group_stats(rows: list[dict[str, Any]], group_tasks: set[str]) -> dict[str, Any]:
    selected = [row for row in rows if row["task"] in group_tasks]
    grouped = aggregate_rows(selected, ["task"])
    task_units = list(grouped.values())
    tp = sum(u["true_positive"] for u in task_units)
    tn = sum(u["true_negative"] for u in task_units)
    fp = sum(u["false_positive"] for u in task_units)
    fn = sum(u["false_negative"] for u in task_units)
    micro = metric_from_counts(tp, tn, fp, fn)
    macro_fpr = mean_or_none([summarize_unit_counts(u)["micro_fpr"] for u in task_units if summarize_unit_counts(u)["micro_fpr"] is not None])
    macro_fnr = mean_or_none([summarize_unit_counts(u)["micro_fnr"] for u in task_units if summarize_unit_counts(u)["micro_fnr"] is not None])
    macro_acc = mean_or_none([summarize_unit_counts(u)["acceptance_micro"] for u in task_units if summarize_unit_counts(u)["acceptance_micro"] is not None])
    return {**micro, "task_macro_fpr": macro_fpr, "task_macro_fnr": macro_fnr, "task_macro_acceptance": macro_acc, "support": len(task_units)}


def safe_ordinal_label(fnr_v100: float | None, fnr_4090: float | None, fpr_v100: float | None, fpr_4090: float | None, pos_sup_v100: int, pos_sup_4090: int) -> str:
    if fnr_v100 is None or fnr_4090 is None or fpr_v100 is None or fpr_4090 is None:
        return "data_insufficient"
    if fnr_v100 >= 0.2 and fnr_4090 >= 0.2 and fpr_v100 < 0.05 and fpr_4090 < 0.05 and pos_sup_v100 >= 100 and pos_sup_4090 >= 100:
        return "stable_high_fnr"
    if fnr_v100 >= 0.2 and fnr_4090 >= 0.2 and fpr_v100 < 0.05 and fpr_4090 < 0.05:
        return "stable_high_fnr_low_support"
    if fpr_v100 < 0.05 and fpr_4090 < 0.05:
        return "stable_low_fpr" if abs(fnr_v100 - fnr_4090) <= 0.05 and (fnr_v100 - fnr_4090) * (fnr_v100 - fnr_4090) >= 0 else "unstable_across_hardware"
    if fpr_v100 >= 0.05 or fpr_4090 >= 0.05:
        return "high_fp_risk"
    return "low_fnr"


def compute_common_pair_rows(v100_rows: list[dict[str, Any]], gpu4090_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    left = aggregate_rows(v100_rows, PAIR_KEY_FIELDS + ["bucket"])
    right = aggregate_rows(gpu4090_rows, PAIR_KEY_FIELDS + ["bucket"])
    common_keys = sorted(set(left) & set(right))
    output = []
    for key in common_keys:
        l = left[key]
        r = right[key]
        ls = summarize_unit_counts(l)
        rs = summarize_unit_counts(r)
        output.append(
            {
                "task": key[0],
                "layer": key[1],
                "kv_head": key[2],
                "v100_tp": l["true_positive"],
                "v100_tn": l["true_negative"],
                "v100_fp": l["false_positive"],
                "v100_fn": l["false_negative"],
                "gpu4090_tp": r["true_positive"],
                "gpu4090_tn": r["true_negative"],
                "gpu4090_fp": r["false_positive"],
                "gpu4090_fn": r["false_negative"],
                "v100_fpr": ls["micro_fpr"],
                "v100_fnr": ls["micro_fnr"],
                "gpu4090_fpr": rs["micro_fpr"],
                "gpu4090_fnr": rs["micro_fnr"],
                "fpr_abs_diff": abs((rs["micro_fpr"] or 0.0) - (ls["micro_fpr"] or 0.0)),
                "fnr_abs_diff": abs((rs["micro_fnr"] or 0.0) - (ls["micro_fnr"] or 0.0)),
                "fpr_delta": (rs["micro_fpr"] or 0.0) - (ls["micro_fpr"] or 0.0),
                "fnr_delta": (rs["micro_fnr"] or 0.0) - (ls["micro_fnr"] or 0.0),
                "v100_positive_support": l["true_positive"] + l["false_negative"],
                "gpu4090_positive_support": r["true_positive"] + r["false_negative"],
                "v100_negative_support": l["true_negative"] + l["false_positive"],
                "gpu4090_negative_support": r["true_negative"] + r["false_positive"],
                "v100_accepted_fraction": ls["acceptance_micro"],
                "gpu4090_accepted_fraction": rs["acceptance_micro"],
                "cross_hardware_consistent": (
                    ls["micro_fnr"] is not None
                    and rs["micro_fnr"] is not None
                    and ls["micro_fpr"] is not None
                    and rs["micro_fpr"] is not None
                    and (ls["micro_fnr"] - rs["micro_fnr"]) * (ls["micro_fnr"] - rs["micro_fnr"]) >= 0
                    and abs(ls["micro_fnr"] - rs["micro_fnr"]) <= 0.05
                    and ls["micro_fpr"] < 0.05
                    and rs["micro_fpr"] < 0.05
                ),
            }
        )
    return output


def collect_column_semantics(schema: dict[str, Any]) -> str:
    lines = ["# Column Semantics", "", "| column | meaning | status | evidence |", "| --- | --- | --- | --- |"]
    for col in schema["columns"]:
        evidence = schema["column_evidence"].get(col, "")
        meaning = schema["column_meanings"].get(col, "")
        status = schema["column_status"].get(col, "")
        lines.append(f"| {col} | {meaning} | {status} | {evidence} |")
    return "\n".join(lines) + "\n"


def bucket_summary(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    agg = aggregate_rows(rows, [field])
    out = []
    for key, unit in sorted(agg.items(), key=lambda item: str(item[0])):
        stats = summarize_unit_counts(unit)
        out.append({field: key[0], **stats})
    return out


def layer_head_summary(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    agg = aggregate_rows(rows, group_fields)
    out = []
    for key, unit in sorted(agg.items(), key=lambda item: item[0]):
        stats = summarize_unit_counts(unit)
        row = {field: value for field, value in zip(group_fields, key)}
        row.update(stats)
        out.append(row)
    return out


def avg_metrics(values: list[dict[str, Any]]) -> dict[str, Any]:
    fprs = [v["micro_fpr"] for v in values if v["micro_fpr"] is not None]
    fnrs = [v["micro_fnr"] for v in values if v["micro_fnr"] is not None]
    accs = [v["acceptance_micro"] for v in values if v["acceptance_micro"] is not None]
    return {
        "micro_fpr": mean_or_none(fprs),
        "micro_fnr": mean_or_none(fnrs),
        "acceptance_micro": mean_or_none(accs),
        "support": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v100-root", type=Path, default=Path("reports/insight_v2/wave_a_8gpu"))
    parser.add_argument("--gpu4090-root", type=Path, default=Path("reports/insight_v2/wave_a_4090_single"))
    parser.add_argument("--cross-hardware-root", type=Path, default=Path("reports/insight_v2/cross_hardware_v100_4090"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/insight_v2/v_gate_layer_head_opportunity"))
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    v100_root = resolve_root(args.v100_root, CSV_NAME)
    gpu4090_root = resolve_root(args.gpu4090_root, CSV_NAME)
    v100_csv = v100_root / CSV_NAME
    gpu4090_csv = gpu4090_root / CSV_NAME
    if not v100_csv.exists() or not gpu4090_csv.exists():
        raise SystemExit("missing v_gate_confusion.csv inputs")
    v100_rows = read_csv_rows(v100_csv)
    gpu4090_rows = read_csv_rows(gpu4090_csv)

    # Inventory
    input_inventory = {
        "schema_version": "insight_v2.v_gate_layer_head_opportunity_input_inventory",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "roots": {
            "v100": {
                "report_root": str(v100_root),
                "csv_path": str(v100_csv),
                "csv_sha256": sha256_file(v100_csv),
                "cross_hardware_root": str(args.cross_hardware_root),
            },
            "gpu4090": {
                "report_root": str(gpu4090_root),
                "csv_path": str(gpu4090_csv),
                "csv_sha256": sha256_file(gpu4090_csv),
                "cross_hardware_root": str(args.cross_hardware_root),
            },
        },
    }
    atomic_write_json(out_root / "input_inventory.json", input_inventory)
    atomic_write_text(
        out_root / "input_inventory.md",
        "\n".join(
            [
                "# Input Inventory",
                "",
                f"- v100 csv: `{v100_csv}`",
                f"- gpu4090 csv: `{gpu4090_csv}`",
                f"- cross hardware root: `{args.cross_hardware_root}`",
            ]
        )
        + "\n",
    )

    # Schema audit
    schema = {
        "schema_version": "insight_v2.v_gate_layer_head_opportunity_schema_audit",
        "hardware": {},
    }
    for label, rows, path in [("v100", v100_rows, v100_csv), ("gpu4090", gpu4090_rows, gpu4090_csv)]:
        schema["hardware"][label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": len(rows),
            "columns": list(rows[0].keys()) if rows else [],
            "dtype": {col: ("int" if all(parse_int(r.get(col)) is not None or is_blank(r.get(col)) for r in rows) else "float" if all(parse_float(r.get(col)) is not None or is_blank(r.get(col)) for r in rows) else "str") for col in (rows[0].keys() if rows else [])},
            "blank_values": {col: sum(1 for r in rows if is_blank(r.get(col))) for col in (rows[0].keys() if rows else [])},
            "nan_inf_values": {col: sum(1 for r in rows if parse_float(r.get(col)) is None and not is_blank(r.get(col)) and str(r.get(col)).lower() in {"nan", "inf", "+inf", "-inf"}) for col in (rows[0].keys() if rows else [])},
            "duplicate_complete_rows": duplicate_row_count(rows),
            "duplicate_primary_keys": duplicate_primary_key_count(rows, COMMON_KEY_FIELDS),
            "task_values": unique_values(rows, "task"),
            "phase_values": unique_values(rows, "phase"),
            "layer_min": min((parse_int(r.get("layer")) for r in rows if parse_int(r.get("layer")) is not None), default=None),
            "layer_max": max((parse_int(r.get("layer")) for r in rows if parse_int(r.get("layer")) is not None), default=None),
            "kv_head_min": min((parse_int(r.get("kv_head")) for r in rows if parse_int(r.get("kv_head")) is not None), default=None),
            "kv_head_max": max((parse_int(r.get("kv_head")) for r in rows if parse_int(r.get("kv_head")) is not None), default=None),
            "bucket_values": unique_values(rows, "bucket"),
            "metric_values": unique_values(rows, "metric"),
            "field_presence": {
                "TP_TN_FP_FN": all(col in rows[0] for col in ("true_positive", "true_negative", "false_positive", "false_negative")) if rows else False,
                "candidate_count": "candidate_count" in rows[0] if rows else False,
                "accepted_count": "accepted_count" in rows[0] if rows else False,
                "raw_mse": "raw_mse" in rows[0] if rows else False,
                "pattern_mse": "pattern_mse" in rows[0] if rows else False,
                "gate_score": "gate_score" in rows[0] if rows else False,
                "rho": "rho" in rows[0] if rows else False,
                "oracle_decision": "oracle_decision" in rows[0] if rows else False,
                "current_decision": "current_decision" in rows[0] if rows else False,
                "FP_penalty": "false_positive_penalty" in rows[0] if rows else False,
                "FN_opportunity": "false_negative_opportunity" in rows[0] if rows else False,
            },
            "zero_denominator_blank_false_positive_rate": sum(1 for r in rows if is_blank(r.get("false_positive_rate"))),
            "zero_denominator_blank_false_negative_rate": sum(1 for r in rows if is_blank(r.get("false_negative_rate"))),
        }
    atomic_write_json(out_root / "schema_audit.json", schema)
    atomic_write_text(
        out_root / "schema_audit.md",
        "\n".join(
            [
                "# Schema Audit",
                "",
                "| hardware | rows | columns | duplicate PKs | blank FPR | blank FNR |",
                "| --- | ---: | --- | ---: | ---: | ---: |",
            ]
            + [
                f"| {hw} | {spec['row_count']} | {len(spec['columns'])} | {spec['duplicate_primary_keys']} | {spec['zero_denominator_blank_false_positive_rate']} | {spec['zero_denominator_blank_false_negative_rate']} |"
                for hw, spec in schema["hardware"].items()
            ]
        )
        + "\n",
    )

    # Column semantics
    schema_cols = list(v100_rows[0].keys()) if v100_rows else []
    col_meanings = {
        "task": "LongBench/GSM8K task label; used as the primary task dimension.",
        "phase": "Model phase; summary CSV is prefill-only for this confusion table.",
        "kv_type": "V gate only; this CSV records V-side gate-vs-oracle confusion.",
        "layer": "Transformer layer index.",
        "kv_head": "Key/value head index.",
        "bucket": "Position bucket derived from sampled token position (first/middle/last).",
        "metric": "Fixed summary metric name; gate_vs_mse_oracle.",
        "true_positive": "Count of sampled tokens where gate accepted and oracle said pattern was better.",
        "true_negative": "Count of sampled tokens where gate rejected and oracle said pattern was worse.",
        "false_positive": "Count of sampled tokens where gate accepted but oracle said raw was better.",
        "false_negative": "Count of sampled tokens where gate rejected but oracle said pattern was better.",
        "total": "TP+TN+FP+FN; row-level candidate support for the sampled tokens.",
        "false_positive_rate": "FP / (FP + TN), blank when denominator is zero.",
        "false_negative_rate": "FN / (FN + TP), blank when denominator is zero.",
    }
    col_status = {
        "candidate_count": "not_collected",
        "accepted_count": "derivable_as_TP_plus_FP",
        "raw_mse": "not_collected_in_summary",
        "pattern_mse": "not_collected_in_summary",
        "gate_score": "not_collected",
        "rho": "not_collected_in_summary",
        "oracle_decision": "present_in_raw_observer_only",
        "current_decision": "present_in_raw_observer_only",
        "FP_penalty": "not_collected_in_summary",
        "FN_opportunity": "not_collected_in_summary",
    }
    col_evidence = {
        "task": "parsed directly from CSV.",
        "phase": "parsed directly from CSV.",
        "kv_type": "parsed directly from CSV.",
        "layer": "parsed directly from CSV.",
        "kv_head": "parsed directly from CSV.",
        "bucket": "present in CSV; populated by _bucket(token, total) in insight/hook_metrics.py.",
        "metric": "fixed by summarize_insight_wave_a_8gpu.py gate_vs_mse_oracle filter.",
        "true_positive": "record_prefill_v_metrics() adds confusion counts via observer.add_confusion().",
        "true_negative": "record_prefill_v_metrics() adds confusion counts via observer.add_confusion().",
        "false_positive": "record_prefill_v_metrics() adds confusion counts via observer.add_confusion().",
        "false_negative": "record_prefill_v_metrics() adds confusion counts via observer.add_confusion().",
        "total": "summarizer reconstructs this from TP/TN/FP/FN counters.",
        "false_positive_rate": "summarizer computes fp / (fp + tn).",
        "false_negative_rate": "summarizer computes fn / (fn + tp).",
    }
    column_semantics = {"columns": schema_cols, "column_meanings": col_meanings, "column_status": col_status, "column_evidence": col_evidence}
    atomic_write_text(out_root / "column_semantics.md", collect_column_semantics(column_semantics))

    # Collector semantics audit
    collector_audit = {
        "schema_version": "insight_v2.v_gate_layer_head_collector_semantics",
        "status": "aggregate_only",
        "findings": [
            {
                "file": "insight/hook_metrics.py",
                "line": 275,
                "function": "record_prefill_v_metrics",
                "evidence": "oracle = pattern_mse_vec < raw_mse_vec; gate is the current mask.",
                "interpretation": "Positive class means pattern wins over raw; gate acceptance is predicted positive.",
            },
            {
                "file": "insight/hook_metrics.py",
                "line": 277,
                "function": "record_prefill_v_metrics",
                "evidence": "tp = gate & oracle; tn = ~gate & ~oracle; fp = gate & ~oracle; fn = ~gate & oracle.",
                "interpretation": "TP/TN/FP/FN are token-level confusion counts, not per-sample labels.",
            },
            {
                "file": "insight/hook_metrics.py",
                "line": 282,
                "function": "record_prefill_v_metrics",
                "evidence": "observer.add_confusion(f\"{prefix}.gate_vs_mse_oracle\", ...).",
                "interpretation": "Confusion counters are accumulated directly into the collector.",
            },
            {
                "file": "insight/hook_metrics.py",
                "line": 283,
                "function": "record_prefill_v_metrics",
                "evidence": "raw_mse, pattern_candidate_mse, actual_selected_path_mse, relative_candidate_benefit are recorded as scalars.",
                "interpretation": "The raw/pattern MSE signals exist in raw observer output, but not in the summary confusion CSV.",
            },
            {
                "file": "insight/hook_metrics.py",
                "line": 299,
                "function": "record_prefill_v_metrics",
                "evidence": "sample record contains gate_current, gate_oracle, rho, false_positive_penalty, false_negative_opportunity.",
                "interpretation": "Candidate-level sweep fields exist only in sample records, not in v_gate_confusion.csv.",
            },
            {
                "file": "scripts/summarize_insight_wave_a_8gpu.py",
                "line": 209,
                "function": "main",
                "evidence": "confusion_store accumulates counters from observer payloads.",
                "interpretation": "CSV rows are built from complete confusion counters, not from truncated sample records.",
            },
            {
                "file": "scripts/summarize_insight_wave_a_8gpu.py",
                "line": 230,
                "function": "main",
                "evidence": "v_gate_rows are generated from aggregated confusion_store totals.",
                "interpretation": "The published CSV reflects aggregate counts at task/layer/head/bucket granularity.",
            },
        ],
        "answers": {
            "tp_tn_fp_fn_definition": "token-level gate-vs-oracle confusion over sampled prefill tokens.",
            "positive_class": "pattern better than raw (oracle true).",
            "gate_predicted_positive": "gate accepts the pattern path.",
            "confusion_update_granularity": "sampled token, then aggregated into layer/head/bucket rows.",
            "row_candidate_count": "total = TP + TN + FP + FN.",
            "count_cross_token_aggregated": True,
            "raw_pattern_mse_retention": "raw/pattern MSE retained as scalar summaries, not as per-candidate CSV columns.",
            "fn_opportunity_exact": False,
            "fp_penalty_exact": False,
            "gate_score_exists": False,
            "rho_exists": False,
        },
    }
    atomic_write_json(out_root / "collector_semantics_audit.json", collector_audit)
    atomic_write_text(
        out_root / "collector_semantics_audit.md",
        "\n".join(
            [
                "# Collector Semantics Audit",
                "",
                f"Status: `{collector_audit['status']}`",
                "",
                "| file | line | function | evidence | interpretation |",
                "| --- | ---: | --- | --- | --- |",
            ]
            + [f"| {item['file']} | {item['line']} | {item['function']} | {item['evidence']} | {item['interpretation']} |" for item in collector_audit["findings"]]
        )
        + "\n",
    )

    # Global / task group summaries
    hw_tables = {
        "v100": v100_rows,
        "gpu4090": gpu4090_rows,
    }
    global_rows = []
    task_rows = []
    layer_rows = []
    head_rows = []
    layer_head_rows = []
    task_layer_head_rows = []
    bucket_rows = []
    positive_negative_rows = []
    group_unit_cache = {}
    for hw, rows in hw_tables.items():
        hw_global = global_stats(rows, args.bootstrap_seed, args.bootstrap_repetitions)
        global_rows.append({"hardware": hw, "scope": "global", **hw_global})
        task_macro = task_macro_stats(rows, args.bootstrap_seed, args.bootstrap_repetitions)
        global_rows.append({"hardware": hw, "scope": "task_macro", **task_macro})
        for group_name, tasks in TASK_GROUPS.items():
            grp = task_group_stats(rows, tasks)
            global_rows.append({"hardware": hw, "scope": f"{group_name}_tasks", **grp})
            positive_negative_rows.append(
                {
                    "hardware": hw,
                    "task_group": group_name,
                    "micro_fpr": grp["micro_fpr"],
                    "micro_fnr": grp["micro_fnr"],
                    "task_macro_fpr": grp["task_macro_fpr"],
                    "task_macro_fnr": grp["task_macro_fnr"],
                    "precision_micro": grp["precision_micro"],
                    "recall_micro": grp["recall_micro"],
                    "acceptance_micro": grp["acceptance_micro"],
                    "positive_support": grp["positive_support"],
                    "negative_support": grp["negative_support"],
                    "fn_opportunity_status": "not_collected",
                    "fp_penalty_status": "not_collected",
                    "current_confusion_cost_balance": None,
                }
            )

        # Task-level
        task_grouped = aggregate_rows(rows, ["task"])
        for task, unit in sorted(task_grouped.items(), key=lambda item: item[0]):
            stats = summarize_unit_counts(unit)
            task_rows.append(
                {
                    "hardware": hw,
                    "task": task[0],
                    **stats,
                }
            )

        # Layer-level
        layer_grouped = aggregate_rows(rows, ["layer"])
        for layer, unit in sorted(layer_grouped.items(), key=lambda item: item[0]):
            stats = summarize_unit_counts(unit)
            layer_rows.append({"hardware": hw, "layer": layer[0], **stats})

        # Head-level
        head_grouped = aggregate_rows(rows, ["kv_head"])
        for head, unit in sorted(head_grouped.items(), key=lambda item: item[0]):
            stats = summarize_unit_counts(unit)
            head_rows.append({"hardware": hw, "kv_head": head[0], **stats})

        # Layer-head aggregated across tasks/buckets
        lh_grouped = aggregate_rows(rows, ["layer", "kv_head"])
        for key, unit in sorted(lh_grouped.items(), key=lambda item: item[0]):
            stats = summarize_unit_counts(unit)
            row = {"hardware": hw, "layer": key[0], "kv_head": key[1], **stats}
            layer_head_rows.append(row)

        # Task-layer-head aggregated across buckets
        tlh_grouped = aggregate_rows(rows, ["task", "layer", "kv_head"])
        for key, unit in sorted(tlh_grouped.items(), key=lambda item: item[0]):
            stats = summarize_unit_counts(unit)
            row = {"hardware": hw, "task": key[0], "layer": key[1], "kv_head": key[2], **stats}
            task_layer_head_rows.append(row)
            group_unit_cache.setdefault(hw, {})[(key[0], key[1], key[2])] = unit

        # Position bucket
        bucket_grouped = aggregate_rows(rows, ["bucket"])
        for key, unit in sorted(bucket_grouped.items(), key=lambda item: str(item[0])):
            stats = summarize_unit_counts(unit)
            bucket_rows.append({"hardware": hw, "bucket": key[0], **stats})

    write_csv(out_root / "v_gate_global_summary.csv", global_rows, ["hardware", "scope", "true_positive", "true_negative", "false_positive", "false_negative", "total", "candidate_count", "accepted_count", "positive_support", "negative_support", "micro_fpr", "micro_fnr", "precision_micro", "recall_micro", "acceptance_micro", "task_macro_fpr", "task_macro_fnr", "task_macro_acceptance", "task_macro_fpr_bootstrap_mean", "task_macro_fpr_bootstrap_ci_low", "task_macro_fpr_bootstrap_ci_high", "task_macro_fnr_bootstrap_mean", "task_macro_fnr_bootstrap_ci_low", "task_macro_fnr_bootstrap_ci_high", "fpr_bootstrap_mean", "fpr_bootstrap_ci_low", "fpr_bootstrap_ci_high", "fnr_bootstrap_mean", "fnr_bootstrap_ci_low", "fnr_bootstrap_ci_high", "support"])
    write_csv(out_root / "v_gate_by_task.csv", task_rows, ["hardware", "task", "true_positive", "true_negative", "false_positive", "false_negative", "total", "candidate_count", "accepted_count", "positive_support", "negative_support", "micro_fpr", "micro_fnr", "precision_micro", "recall_micro", "acceptance_micro"])
    write_csv(out_root / "v_gate_by_layer.csv", layer_rows, ["hardware", "layer", "true_positive", "true_negative", "false_positive", "false_negative", "total", "candidate_count", "accepted_count", "positive_support", "negative_support", "micro_fpr", "micro_fnr", "precision_micro", "recall_micro", "acceptance_micro"])
    write_csv(out_root / "v_gate_by_head.csv", head_rows, ["hardware", "kv_head", "true_positive", "true_negative", "false_positive", "false_negative", "total", "candidate_count", "accepted_count", "positive_support", "negative_support", "micro_fpr", "micro_fnr", "precision_micro", "recall_micro", "acceptance_micro"])
    write_csv(out_root / "v_gate_by_layer_head.csv", layer_head_rows, ["hardware", "layer", "kv_head", "true_positive", "true_negative", "false_positive", "false_negative", "total", "candidate_count", "accepted_count", "positive_support", "negative_support", "micro_fpr", "micro_fnr", "precision_micro", "recall_micro", "acceptance_micro"])
    write_csv(out_root / "v_gate_by_task_layer_head.csv", task_layer_head_rows, ["hardware", "task", "layer", "kv_head", "true_positive", "true_negative", "false_positive", "false_negative", "total", "candidate_count", "accepted_count", "positive_support", "negative_support", "micro_fpr", "micro_fnr", "precision_micro", "recall_micro", "acceptance_micro"])
    write_csv(out_root / "v_gate_by_position_bucket.csv", bucket_rows, ["hardware", "bucket", "true_positive", "true_negative", "false_positive", "false_negative", "total", "candidate_count", "accepted_count", "positive_support", "negative_support", "micro_fpr", "micro_fnr", "precision_micro", "recall_micro", "acceptance_micro"])
    write_csv(out_root / "v_gate_positive_negative_tasks.csv", positive_negative_rows, ["hardware", "task_group", "micro_fpr", "micro_fnr", "task_macro_fpr", "task_macro_fnr", "precision_micro", "recall_micro", "acceptance_micro", "positive_support", "negative_support", "fn_opportunity_status", "fp_penalty_status", "current_confusion_cost_balance"])

    # Cross-hardware stability across common task-layer-head units
    v100_task_lh = aggregate_rows(v100_rows, ["task", "layer", "kv_head"])
    gpu4090_task_lh = aggregate_rows(gpu4090_rows, ["task", "layer", "kv_head"])
    common_keys = sorted(set(v100_task_lh) & set(gpu4090_task_lh))
    stability_rows = []
    for key in common_keys:
        left = summarize_unit_counts(v100_task_lh[key])
        right = summarize_unit_counts(gpu4090_task_lh[key])
        fnr_v100 = left["micro_fnr"]
        fnr_4090 = right["micro_fnr"]
        fpr_v100 = left["micro_fpr"]
        fpr_4090 = right["micro_fpr"]
        stability_rows.append(
            {
                "task": key[0],
                "layer": key[1],
                "kv_head": key[2],
                "v100_fnr": fnr_v100,
                "gpu4090_fnr": fnr_4090,
                "v100_fpr": fpr_v100,
                "gpu4090_fpr": fpr_4090,
                "v100_accepted_fraction": left["acceptance_micro"],
                "gpu4090_accepted_fraction": right["acceptance_micro"],
                "v100_fn_count": left["false_negative"],
                "gpu4090_fn_count": right["false_negative"],
                "v100_fp_count": left["false_positive"],
                "gpu4090_fp_count": right["false_positive"],
                "v100_positive_support": left["positive_support"],
                "gpu4090_positive_support": right["positive_support"],
                "v100_negative_support": left["negative_support"],
                "gpu4090_negative_support": right["negative_support"],
                "fnr_abs_diff": abs((fnr_v100 or 0.0) - (fnr_4090 or 0.0)),
                "fpr_abs_diff": abs((fpr_v100 or 0.0) - (fpr_4090 or 0.0)),
                "accepted_fraction_abs_diff": abs((left["acceptance_micro"] or 0.0) - (right["acceptance_micro"] or 0.0)),
                "fn_count_abs_diff": abs(left["false_negative"] - right["false_negative"]),
                "fpr_rank": None,
                "fnr_rank": None,
                "accepted_fraction_rank": None,
                "fn_count_rank": None,
                "cross_hardware_consistent": safe_ordinal_label(fnr_v100, fnr_4090, fpr_v100, fpr_4090, left["positive_support"], right["positive_support"]) in {"stable_high_fnr", "stable_high_fnr_low_support", "stable_low_fpr"},
            }
        )
    # Ranks and labels
    for row in stability_rows:
        row["support_label"] = "low_support" if row["v100_positive_support"] < 100 or row["gpu4090_positive_support"] < 100 else ("high_support" if row["v100_positive_support"] >= 500 and row["gpu4090_positive_support"] >= 500 else "moderate_support")
        row["fnr_stability_label"] = (
            "stable"
            if row["v100_fnr"] is not None
            and row["gpu4090_fnr"] is not None
            and row["v100_fpr"] is not None
            and row["gpu4090_fpr"] is not None
            and row["fnr_abs_diff"] <= 0.05
            and row["v100_fpr"] < 0.05
            and row["gpu4090_fpr"] < 0.05
            else "unstable"
        )
        row["label"] = safe_ordinal_label(row["v100_fnr"], row["gpu4090_fnr"], row["v100_fpr"], row["gpu4090_fpr"], row["v100_positive_support"], row["gpu4090_positive_support"])
        row["readiness"] = "aggregate_only"
    # ranking scores
    def sort_key_high(row: dict[str, Any]) -> tuple:
        return (
            0 if row["cross_hardware_consistent"] else 1,
            0 if row["support_label"] == "high_support" else 1 if row["support_label"] == "moderate_support" else 2,
            -(min(row["v100_fnr"] or 0.0, row["gpu4090_fnr"] or 0.0)),
            -(row["v100_fn_count"] + row["gpu4090_fn_count"]),
            row["task"],
            row["layer"],
            row["kv_head"],
        )
    def sort_key_low(row: dict[str, Any]) -> tuple:
        v100_fpr = row["v100_fpr"] if row["v100_fpr"] is not None else 1.0
        gpu_fpr = row["gpu4090_fpr"] if row["gpu4090_fpr"] is not None else 1.0
        v100_acc = row["v100_accepted_fraction"] if row["v100_accepted_fraction"] is not None else 1.0
        gpu_acc = row["gpu4090_accepted_fraction"] if row["gpu4090_accepted_fraction"] is not None else 1.0
        return (
            0 if v100_fpr < 0.05 and gpu_fpr < 0.05 else 1,
            0 if row["cross_hardware_consistent"] else 1,
            v100_fpr + gpu_fpr,
            v100_acc + gpu_acc,
            -(min(row["v100_fnr"] or 0.0, row["gpu4090_fnr"] or 0.0)),
            row["task"],
            row["layer"],
            row["kv_head"],
        )
    stability_rows_sorted_high = sorted(stability_rows, key=sort_key_high)
    stability_rows_sorted_low = sorted(stability_rows, key=sort_key_low)
    for idx, row in enumerate(stability_rows_sorted_high, 1):
        row["high_fnr_rank"] = idx
    for idx, row in enumerate(stability_rows_sorted_low, 1):
        row["low_risk_rank"] = idx
    # Spearman summaries
    def collect_vals(field: str) -> tuple[list[float], list[float]]:
        left = [row[field.replace("v100_", "v100_")] for row in stability_rows if row[field.replace("v100_", "v100_")] is not None]
        right = [row[field.replace("v100_", "gpu4090_")] for row in stability_rows if row[field.replace("v100_", "v100_")] is not None and row[field.replace("v100_", "gpu4090_")] is not None]
        return left, right
    fnr_corr = spearman([row["v100_fnr"] for row in stability_rows if row["v100_fnr"] is not None and row["gpu4090_fnr"] is not None], [row["gpu4090_fnr"] for row in stability_rows if row["v100_fnr"] is not None and row["gpu4090_fnr"] is not None])
    fpr_corr = spearman([row["v100_fpr"] for row in stability_rows if row["v100_fpr"] is not None and row["gpu4090_fpr"] is not None], [row["gpu4090_fpr"] for row in stability_rows if row["v100_fpr"] is not None and row["gpu4090_fpr"] is not None])
    acc_corr = spearman([row["v100_accepted_fraction"] for row in stability_rows if row["v100_accepted_fraction"] is not None and row["gpu4090_accepted_fraction"] is not None], [row["gpu4090_accepted_fraction"] for row in stability_rows if row["v100_accepted_fraction"] is not None and row["gpu4090_accepted_fraction"] is not None])
    fn_corr = spearman([row["v100_fn_count"] for row in stability_rows], [row["gpu4090_fn_count"] for row in stability_rows]) if stability_rows else None
    common_stability_md = [
        "# V Gate Cross-Hardware Stability",
        "",
        f"- common task-layer-head units: `{len(stability_rows)}`",
        f"- spearman_fnr: `{fnr_corr}`",
        f"- spearman_fpr: `{fpr_corr}`",
        f"- spearman_accepted_fraction: `{acc_corr}`",
        f"- spearman_fn_count: `{fn_corr}`",
        f"- stable_high_fnr_candidates: `{sum(1 for row in stability_rows if row['label'] == 'stable_high_fnr')}`",
        f"- stable_high_fnr_low_support_candidates: `{sum(1 for row in stability_rows if row['label'] == 'stable_high_fnr_low_support')}`",
    ]
    write_csv(out_root / "v_gate_cross_hardware_stability.csv", stability_rows, ["task", "layer", "kv_head", "v100_fnr", "gpu4090_fnr", "v100_fpr", "gpu4090_fpr", "v100_accepted_fraction", "gpu4090_accepted_fraction", "v100_fn_count", "gpu4090_fn_count", "v100_fp_count", "gpu4090_fp_count", "v100_positive_support", "gpu4090_positive_support", "v100_negative_support", "gpu4090_negative_support", "fnr_abs_diff", "fpr_abs_diff", "accepted_fraction_abs_diff", "fn_count_abs_diff", "cross_hardware_consistent", "support_label", "fnr_stability_label", "label", "readiness", "high_fnr_rank", "low_risk_rank"])
    atomic_write_text(out_root / "v_gate_cross_hardware_stability.md", "\n".join(common_stability_md) + "\n")

    # Candidate rankings
    high_ranking = []
    low_ranking = []
    candidate_labels = []
    for row in stability_rows_sorted_high:
        high_ranking.append(
            {
                **row,
                "rank_score": (
                    1 if row["cross_hardware_consistent"] else 0,
                    1 if row["support_label"] == "high_support" else 0 if row["support_label"] == "moderate_support" else -1,
                    min(row["v100_fnr"] or 0.0, row["gpu4090_fnr"] or 0.0),
                    row["v100_fn_count"] + row["gpu4090_fn_count"],
                ),
            }
        )
    for row in stability_rows_sorted_low:
        low_ranking.append({**row})
    for row in stability_rows:
        candidate_labels.append(
            {
                "task": row["task"],
                "layer": row["layer"],
                "kv_head": row["kv_head"],
                "label": row["label"],
                "readiness": row["readiness"],
                "support_label": row["support_label"],
                "cross_hardware_consistent": row["cross_hardware_consistent"],
                "v100_fnr": row["v100_fnr"],
                "gpu4090_fnr": row["gpu4090_fnr"],
                "v100_fpr": row["v100_fpr"],
                "gpu4090_fpr": row["gpu4090_fpr"],
                "v100_positive_support": row["v100_positive_support"],
                "gpu4090_positive_support": row["gpu4090_positive_support"],
            }
        )
    write_csv(out_root / "high_fnr_layer_head_ranking.csv", high_ranking, ["task", "layer", "kv_head", "v100_fnr", "gpu4090_fnr", "v100_fpr", "gpu4090_fpr", "v100_accepted_fraction", "gpu4090_accepted_fraction", "v100_fn_count", "gpu4090_fn_count", "v100_fp_count", "gpu4090_fp_count", "v100_positive_support", "gpu4090_positive_support", "v100_negative_support", "gpu4090_negative_support", "fnr_abs_diff", "fpr_abs_diff", "accepted_fraction_abs_diff", "fn_count_abs_diff", "cross_hardware_consistent", "support_label", "fnr_stability_label", "label", "readiness", "high_fnr_rank", "low_risk_rank", "rank_score"])
    write_csv(out_root / "low_risk_layer_head_ranking.csv", low_ranking, ["task", "layer", "kv_head", "v100_fnr", "gpu4090_fnr", "v100_fpr", "gpu4090_fpr", "v100_accepted_fraction", "gpu4090_accepted_fraction", "v100_fn_count", "gpu4090_fn_count", "v100_fp_count", "gpu4090_fp_count", "v100_positive_support", "gpu4090_positive_support", "v100_negative_support", "gpu4090_negative_support", "fnr_abs_diff", "fpr_abs_diff", "accepted_fraction_abs_diff", "fn_count_abs_diff", "cross_hardware_consistent", "support_label", "fnr_stability_label", "label", "readiness", "high_fnr_rank", "low_risk_rank"])
    write_csv(out_root / "candidate_labels.csv", candidate_labels, ["task", "layer", "kv_head", "label", "readiness", "support_label", "cross_hardware_consistent", "v100_fnr", "gpu4090_fnr", "v100_fpr", "gpu4090_fpr", "v100_positive_support", "gpu4090_positive_support"])

    # Unsupported missed-benefit ranking notice
    atomic_write_text(
        out_root / "missed_benefit_layer_head_ranking_unavailable.md",
        "\n".join(
            [
                "# Missed-Benefit Ranking",
                "",
                "Status: `unavailable`",
                "",
                "FN opportunity is not collected in the summary CSVs, so a truthful missed-benefit ranking cannot be built from the current workspace.",
            ]
        )
        + "\n",
    )

    # Offline sweep readiness
    offline_sweep = {
        "schema_version": "insight_v2.v_gate_offline_sweep_readiness",
        "status": "aggregate_only",
        "reason": [
            "v_gate_confusion.csv only provides aggregate TP/TN/FP/FN counters.",
            "gate_score is not present in the summary CSVs.",
            "rho is not present in the summary CSVs.",
            "V100 raw observer data is not available in this workspace.",
            "FN opportunity and FP penalty are not collected in the summary CSVs.",
        ],
        "available_on_4090_raw_only": True,
        "summary_only": True,
        "can_reconstruct_threshold_sweep": False,
    }
    atomic_write_json(out_root / "offline_sweep_readiness.json", offline_sweep)
    atomic_write_text(
        out_root / "offline_sweep_readiness.md",
        "\n".join(
            [
                "# Offline Sweep Readiness",
                "",
                f"Status: `{offline_sweep['status']}`",
                "",
                *[f"- {reason}" for reason in offline_sweep["reason"]],
            ]
        )
        + "\n",
    )

    # Next minimal step
    next_step = {
        "schema_version": "insight_v2.v_gate_next_minimal_step",
        "status": "not_supported",
        "recommendation": "Do not implement a new recall-aware gate yet; the current cross-hardware evidence does not support it.",
        "why": "No task/layer/head pair satisfies the stable high-FNR + low-FPR + sufficient-support rule on both hardware runs.",
    }
    atomic_write_json(out_root / "next_minimal_step.json", next_step)
    atomic_write_text(
        out_root / "next_minimal_step.md",
        "\n".join(["# Next Minimal Step", "", f"Status: `{next_step['status']}`", "", next_step["recommendation"], "", next_step["why"]]) + "\n",
    )

    # Final summary
    global_v100 = [row for row in global_rows if row["hardware"] == "v100" and row["scope"] == "global"][0]
    global_4090 = [row for row in global_rows if row["hardware"] == "gpu4090" and row["scope"] == "global"][0]
    pos_v100 = [row for row in positive_negative_rows if row["hardware"] == "v100" and row["task_group"] == "positive"][0]
    neg_v100 = [row for row in positive_negative_rows if row["hardware"] == "v100" and row["task_group"] == "negative"][0]
    pos_4090 = [row for row in positive_negative_rows if row["hardware"] == "gpu4090" and row["task_group"] == "positive"][0]
    neg_4090 = [row for row in positive_negative_rows if row["hardware"] == "gpu4090" and row["task_group"] == "negative"][0]
    top_fnr_tasks = {
        "v100": sorted(
            [{"task": row["task"], "fnr": row["micro_fnr"], "fpr": row["micro_fpr"], "positive_support": row["positive_support"], "negative_support": row["negative_support"]} for row in task_rows if row["hardware"] == "v100"],
            key=lambda x: x["fnr"] if x["fnr"] is not None else -1,
            reverse=True,
        ),
        "gpu4090": sorted(
            [{"task": row["task"], "fnr": row["micro_fnr"], "fpr": row["micro_fpr"], "positive_support": row["positive_support"], "negative_support": row["negative_support"]} for row in task_rows if row["hardware"] == "gpu4090"],
            key=lambda x: x["fnr"] if x["fnr"] is not None else -1,
            reverse=True,
        ),
    }
    final_summary = {
        "schema_version": "insight_v2.v_gate_layer_head_final_summary",
        "v100": global_v100,
        "gpu4090": global_4090,
        "task_group": {
            "v100": {"positive": pos_v100, "negative": neg_v100},
            "gpu4090": {"positive": pos_4090, "negative": neg_4090},
        },
        "top_fnr_tasks": top_fnr_tasks,
        "highest_fn_layers": {
            "v100": sorted([{ "layer": row["layer"], "fn": row["false_negative"], "fnr": row["micro_fnr"], "fpr": row["micro_fpr"]} for row in layer_rows if row["hardware"] == "v100"], key=lambda x: x["fn"], reverse=True)[:5],
            "gpu4090": sorted([{ "layer": row["layer"], "fn": row["false_negative"], "fnr": row["micro_fnr"], "fpr": row["micro_fpr"]} for row in layer_rows if row["hardware"] == "gpu4090"], key=lambda x: x["fn"], reverse=True)[:5],
        },
        "highest_fn_heads": {
            "v100": sorted([{ "kv_head": row["kv_head"], "fn": row["false_negative"], "fnr": row["micro_fnr"], "fpr": row["micro_fpr"]} for row in head_rows if row["hardware"] == "v100"], key=lambda x: x["fn"], reverse=True)[:5],
            "gpu4090": sorted([{ "kv_head": row["kv_head"], "fn": row["false_negative"], "fnr": row["micro_fnr"], "fpr": row["micro_fpr"]} for row in head_rows if row["hardware"] == "gpu4090"], key=lambda x: x["fn"], reverse=True)[:5],
        },
        "stable_high_fnr_candidates": [row for row in candidate_labels if row["label"] == "stable_high_fnr"],
        "stable_high_fnr_low_support_candidates": [row for row in candidate_labels if row["label"] == "stable_high_fnr_low_support"],
        "fn_opportunity_status": "not_collected",
        "fp_penalty_status": "not_collected",
        "gate_score_exists": False,
        "rho_exists": False,
        "offline_sweep_readiness": offline_sweep["status"],
        "pre_registered_benefit_aware_v_gating": "not_supported",
        "recall_aware_v_gating": "not_supported",
        "only_next_step": next_step["recommendation"],
    }
    atomic_write_json(out_root / "final_summary.json", final_summary)
    atomic_write_text(
        out_root / "final_summary.md",
        "\n".join(
            [
                "# Final Summary",
                "",
                f"- V100 micro FPR: `{global_v100['micro_fpr']}`",
                f"- V100 micro FNR: `{global_v100['micro_fnr']}`",
                f"- 4090 micro FPR: `{global_4090['micro_fpr']}`",
                f"- 4090 micro FNR: `{global_4090['micro_fnr']}`",
                f"- task macro uses equal-weight task averages; layer_head macro uses equal-weight task-layer-head averages; sample macro is not collected.",
                f"- top FNR tasks V100: `{', '.join(item['task'] for item in top_fnr_tasks['v100'])}`",
                f"- top FNR tasks 4090: `{', '.join(item['task'] for item in top_fnr_tasks['gpu4090'])}`",
                f"- stable high FNR candidates: `{len(final_summary['stable_high_fnr_candidates'])}`",
                f"- stable high FNR low-support candidates: `{len(final_summary['stable_high_fnr_low_support_candidates'])}`",
                f"- FN opportunity: `{final_summary['fn_opportunity_status']}`",
                f"- FP penalty: `{final_summary['fp_penalty_status']}`",
                f"- gate score / rho: `{final_summary['gate_score_exists']}` / `{final_summary['rho_exists']}`",
                f"- offline sweep readiness: `{final_summary['offline_sweep_readiness']}`",
                f"- pre-registered benefit-aware gate: `{final_summary['pre_registered_benefit_aware_v_gating']}`",
                f"- recall-aware gate: `{final_summary['recall_aware_v_gating']}`",
                f"- next step: `{final_summary['only_next_step']}`",
            ]
        )
        + "\n",
    )

    print(json.dumps({"status": "completed", "output_root": str(out_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
