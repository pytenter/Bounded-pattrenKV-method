#!/usr/bin/env python
"""Offline concentration analysis for 4090 Pattern Gain Wave A."""

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.io import atomic_write_json, atomic_write_text, write_csv


CSV_KEYS = ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric"]
ANALYSIS_PHASE = "prefill"
K_MAIN_METRIC = "relative_benefit"
V_MAIN_METRIC = "relative_benefit"
V_PROXY_METRIC = "relative_candidate_benefit"
SECONDARY_METRICS = {"relative_mse_gain", "relative_range_gain", "range_contraction"}
EXPECTED_TASKS = ["dureader", "gsm8k", "hotpotqa", "passage_retrieval_en", "passage_retrieval_zh", "samsum"]
LAYER_BLOCKS = {
    0: range(0, 8),
    1: range(8, 16),
    2: range(16, 24),
    3: range(24, 32),
}


@dataclass(frozen=True)
class ResolvedPath:
    requested: str
    resolved: str | None
    exists: bool


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


def stable_sorted(values: Iterable[Any]) -> list[Any]:
    return sorted(values, key=lambda item: (str(type(item)), str(item)))


def row_key(row: dict[str, Any], fields: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(field, "") for field in fields)


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value))
    except Exception:
        return None


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(str(value))
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def mean_or_none(values: Iterable[float]) -> float | None:
    vals = list(values)
    return float(statistics.fmean(vals)) if vals else None


def median_or_none(values: Iterable[float]) -> float | None:
    vals = list(values)
    return float(statistics.median(vals)) if vals else None


def percentile_or_none(values: Iterable[float], q: float) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    idx = int(math.ceil(q * len(vals))) - 1
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
        sample = [units[rng.randrange(len(units))] for _ in range(len(units))]
        value = metric_fn(sample)
        if value is not None and math.isfinite(value):
            estimates.append(float(value))
    if not estimates:
        return None, None, None
    estimates.sort()
    lo = estimates[max(0, int(0.025 * len(estimates)))]
    hi = estimates[min(len(estimates) - 1, int(0.975 * len(estimates)))]
    return float(mean_or_none(estimates)), float(lo), float(hi)


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
    rx = rank_average(x)
    ry = rank_average(y)
    mx = statistics.fmean(rx)
    my = statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - my) ** 2 for b in ry))
    if not den_x or not den_y:
        return None
    return float(num / (den_x * den_y))


def jaccard(a: set[Any], b: set[Any]) -> float | None:
    if not a and not b:
        return None
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def resolve_root(preferred: Path, required_file: str) -> ResolvedPath:
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
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and (candidate / required_file).exists():
            return ResolvedPath(str(preferred), str(candidate), True)
    return ResolvedPath(str(preferred), None, False)


def row_to_numeric(row: dict[str, str]) -> dict[str, Any]:
    return {
        "task": row.get("task", ""),
        "phase": row.get("phase", ""),
        "kv_type": row.get("kv_type", ""),
        "layer": parse_int(row.get("layer")),
        "kv_head": parse_int(row.get("kv_head")),
        "bucket": row.get("bucket", ""),
        "metric": row.get("metric", ""),
        "count": parse_int(row.get("count")),
        "mean": parse_float(row.get("mean")),
        "std": parse_float(row.get("std")),
    }


def safe_sum(values: Iterable[float]) -> float:
    return float(sum(values))


def task_layer_head_aggregate(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    grouped: dict[tuple[str, int, int], dict[str, Any]] = defaultdict(lambda: {"count": 0, "contrib": 0.0, "row_count": 0})
    for row in rows:
        count = row["count"]
        mean = row["mean"]
        if count is None or mean is None:
            continue
        key = (str(row["task"]), int(row["layer"]), int(row["kv_head"]))
        grouped[key]["count"] += count
        grouped[key]["contrib"] += count * mean
        grouped[key]["row_count"] += 1
    return grouped


def layer_head_aggregate(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, Any]] = defaultdict(lambda: {"count": 0, "contrib": 0.0, "row_count": 0})
    for row in rows:
        count = row["count"]
        mean = row["mean"]
        if count is None or mean is None:
            continue
        key = (int(row["layer"]), int(row["kv_head"]))
        grouped[key]["count"] += count
        grouped[key]["contrib"] += count * mean
        grouped[key]["row_count"] += 1
    return grouped


def task_aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "contrib": 0.0, "row_count": 0})
    for row in rows:
        count = row["count"]
        mean = row["mean"]
        if count is None or mean is None:
            continue
        key = str(row["task"])
        grouped[key]["count"] += count
        grouped[key]["contrib"] += count * mean
        grouped[key]["row_count"] += 1
    return grouped


def row_profile(rows: list[dict[str, str]]) -> dict[str, Any]:
    numeric_rows = [row_to_numeric(r) for r in rows]
    counts = Counter(tuple(sorted(r.items())) for r in rows)
    duplicate_row_count = sum(1 for value in counts.values() if value > 1)
    primary_counts = Counter(row_key(r, CSV_KEYS) for r in rows)
    duplicate_pk_count = sum(1 for value in primary_counts.values() if value > 1)
    blanks = {field: sum(1 for row in rows if row.get(field, "") == "") for field in rows[0].keys()} if rows else {}
    bad_count = sum(1 for row in numeric_rows if row["count"] is not None and row["count"] < 0)
    nan_count = sum(1 for row in rows for value in row.values() if isinstance(value, str) and value.lower() == "nan")
    inf_count = sum(1 for row in rows for value in row.values() if isinstance(value, str) and value.lower() in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"})
    return {
        "row_count": len(rows),
        "duplicate_row_count": duplicate_row_count,
        "duplicate_primary_key_count": duplicate_pk_count,
        "blank_counts": blanks,
        "negative_count_rows": bad_count,
        "nan_string_count": nan_count,
        "inf_string_count": inf_count,
    }


def metric_rows(rows: list[dict[str, Any]], phase: str, kv_type: str, metric: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["phase"] == phase and row["kv_type"] == kv_type and row["metric"] == metric]


def unit_rows_for_metric(rows: list[dict[str, Any]], kv_type: str, metric: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [r for r in rows if r["phase"] == ANALYSIS_PHASE and r["kv_type"] == kv_type and r["metric"] == metric]
    task_units = task_layer_head_aggregate(selected)
    layer_units = layer_head_aggregate(selected)
    return selected, [{"task": task, "layer": layer, "kv_head": head, **payload} for (task, layer, head), payload in task_units.items()], [{"layer": layer, "kv_head": head, **payload} for (layer, head), payload in layer_units.items()]


def task_macro_layer_head(task_units: dict[tuple[str, int, int], dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    per_key: dict[tuple[int, int], list[float]] = defaultdict(list)
    per_key_counts: dict[tuple[int, int], list[int]] = defaultdict(list)
    task_names = sorted({task for task, _, _ in task_units})
    for task, layer, head in task_units:
        key = (layer, head)
        payload = task_units[(task, layer, head)]
        count = payload["count"]
        if count > 0:
            per_key[key].append(payload["contrib"] / count)
            per_key_counts[key].append(count)
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for key, values in per_key.items():
        out[key] = {
            "count": safe_sum(per_key_counts[key]),
            "contrib": float(statistics.fmean(values)) if values else None,
            "task_count": len(values),
        }
    return out


def unit_stats_from_layer_head(layer_units: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    units = []
    for (layer, head), payload in layer_units.items():
        count = payload["count"]
        contrib = payload["contrib"]
        units.append({"layer": layer, "kv_head": head, "count": count, "contrib": contrib})
    return {
        "units": units,
        "layer_head_count": len(units),
    }


def concentration_metrics(units: list[dict[str, Any]], *, bootstrap_seed: int, bootstrap_repetitions: int, use_task_macro: bool) -> dict[str, Any]:
    ordered = sorted(units, key=lambda item: (item["contrib"] if item["contrib"] is not None else float("-inf"), -item["layer"], -item["kv_head"]), reverse=True)
    positive = [max(float(unit["contrib"] or 0.0), 0.0) for unit in ordered]
    total_pos = sum(positive)
    total = sum(float(unit["contrib"] or 0.0) for unit in ordered)
    total_units = len(ordered)
    top_fracs = [0.01, 0.05, 0.10, 0.25, 0.50]
    top_shares = {}
    for frac in top_fracs:
        n = max(1, math.ceil(total_units * frac)) if total_units else 0
        top_shares[f"top_{int(frac * 100)}pct_positive_share"] = (sum(positive[:n]) / total_pos) if total_pos and n else None
    bottom_10_n = max(1, math.ceil(total_units * 0.10)) if total_units else 0
    bottom_25_n = max(1, math.ceil(total_units * 0.25)) if total_units else 0
    bottom_10_total = sum(float(unit["contrib"] or 0.0) for unit in ordered[-bottom_10_n:]) if bottom_10_n else None
    bottom_25_total = sum(float(unit["contrib"] or 0.0) for unit in ordered[-bottom_25_n:]) if bottom_25_n else None
    bottom_25_negative = sum(1 for unit in ordered[-bottom_25_n:] if float(unit["contrib"] or 0.0) < 0.0) / bottom_25_n if bottom_25_n else None
    gini = None
    hhi = None
    if total_pos > 0:
        shares = [p / total_pos for p in positive if p > 0]
        if shares:
            hhi = sum(s * s for s in shares)
            sorted_shares = sorted(shares)
            n = len(sorted_shares)
            cum = 0.0
            for i, s in enumerate(sorted_shares, start=1):
                cum += i * s
            gini = (2 * cum) / (n * sum(sorted_shares)) - (n + 1) / n if n and sum(sorted_shares) else None
    def micro_metric(sample: list[dict[str, Any]]) -> float | None:
        sample_pos = [max(float(unit["contrib"] or 0.0), 0.0) for unit in sample]
        sample_pos_total = sum(sample_pos)
        if not sample_pos_total:
            return None
        n = max(1, math.ceil(len(sample) * 0.25))
        return sum(sample_pos[:n]) / sample_pos_total
    def macro_metric(sample: list[dict[str, Any]]) -> float | None:
        return micro_metric(sample)
    ci_mean, ci_lo, ci_hi = bootstrap_ci(
        ordered,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
        metric_fn=macro_metric if use_task_macro else micro_metric,
    )
    return {
        "total_units": total_units,
        "total_contribution": total,
        "positive_contribution": sum(p for p in positive if p > 0),
        "negative_contribution": sum(float(unit["contrib"] or 0.0) for unit in ordered if float(unit["contrib"] or 0.0) < 0.0),
        "positive_fraction": (sum(1 for unit in ordered if float(unit["contrib"] or 0.0) > 0.0) / total_units) if total_units else None,
        "harmful_fraction": (sum(1 for unit in ordered if float(unit["contrib"] or 0.0) < 0.0) / total_units) if total_units else None,
        "zero_fraction": (sum(1 for unit in ordered if float(unit["contrib"] or 0.0) == 0.0) / total_units) if total_units else None,
        "median": median_or_none(float(unit["contrib"] or 0.0) for unit in ordered),
        "p25": percentile_or_none((float(unit["contrib"] or 0.0) for unit in ordered), 0.25),
        "p75": percentile_or_none((float(unit["contrib"] or 0.0) for unit in ordered), 0.75),
        "p95": percentile_or_none((float(unit["contrib"] or 0.0) for unit in ordered), 0.95),
        "weighted_mean": (total / sum(float(unit["count"] or 0) for unit in ordered)) if ordered and sum(float(unit["count"] or 0) for unit in ordered) else None,
        "bootstrap_mean": ci_mean,
        "bootstrap_ci_low": ci_lo,
        "bootstrap_ci_high": ci_hi,
        "top_shares": top_shares,
        "bottom_10_total": bottom_10_total,
        "bottom_25_total": bottom_25_total,
        "bottom_25_negative_fraction": bottom_25_negative,
        "gini": gini,
        "hhi": hhi,
    }


def normalize_units(layer_units: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    units = []
    for (layer, head), payload in layer_units.items():
        units.append({"layer": layer, "kv_head": head, "count": payload["count"], "contrib": payload["contrib"]})
    return units


def task_macro_units(task_units: dict[tuple[str, int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    per_key: dict[tuple[int, int], list[float]] = defaultdict(list)
    per_key_counts: dict[tuple[int, int], list[int]] = defaultdict(list)
    for (task, layer, head), payload in task_units.items():
        if payload["count"] > 0:
            per_key[(layer, head)].append(payload["contrib"] / payload["count"])
            per_key_counts[(layer, head)].append(payload["count"])
    units = []
    for key in stable_sorted(per_key.keys()):
        layer, head = key
        values = per_key[key]
        units.append(
            {
                "layer": layer,
                "kv_head": head,
                "count": sum(per_key_counts[key]),
                "contrib": float(statistics.fmean(values)) if values else None,
            }
        )
    return units


def task_level_stats(task_units: dict[tuple[str, int, int], dict[str, Any]]) -> dict[str, Any]:
    task_payload: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "contrib": 0.0, "layer_heads": set()})
    for (task, layer, head), payload in task_units.items():
        task_payload[task]["count"] += payload["count"]
        task_payload[task]["contrib"] += payload["contrib"]
        task_payload[task]["layer_heads"].add((layer, head))
    task_means = []
    for task, payload in task_payload.items():
        if payload["count"] > 0:
            task_means.append(payload["contrib"] / payload["count"])
    return {
        "task_count": len(task_payload),
        "task_macro_mean": mean_or_none(task_means),
        "task_median": median_or_none(task_means),
        "task_p25": percentile_or_none(task_means, 0.25),
        "task_p75": percentile_or_none(task_means, 0.75),
        "task_p95": percentile_or_none(task_means, 0.95),
        "task_coverage": len(task_payload),
    }


def stable_negative_flags(task_units: dict[tuple[str, int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    task_by_unit: dict[tuple[int, int], list[float]] = defaultdict(list)
    count_by_unit: dict[tuple[int, int], int] = defaultdict(int)
    tasks = sorted({task for task, _, _ in task_units})
    for (task, layer, head), payload in task_units.items():
        key = (layer, head)
        count_by_unit[key] += payload["count"]
        if payload["count"] > 0:
            task_by_unit[key].append(payload["contrib"] / payload["count"])
    rows = []
    for key in stable_sorted(task_by_unit.keys()):
        vals = task_by_unit[key]
        task_macro = mean_or_none(vals)
        rows.append(
            {
                "layer": key[0],
                "kv_head": key[1],
                "task_macro_mean": task_macro,
                "negative_task_count": sum(1 for v in vals if v < 0),
                "positive_task_count": sum(1 for v in vals if v > 0),
                "zero_task_count": sum(1 for v in vals if v == 0),
                "task_coverage": len(vals),
                "total_count": count_by_unit[key],
                "max_task_mean": max(vals) if vals else None,
                "stable_negative": bool(
                    task_macro is not None
                    and task_macro < 0
                    and sum(1 for v in vals if v < 0) >= 4
                    and len(vals) >= 5
                    and count_by_unit[key] >= 100
                    and (max(vals) if vals else 0.0) <= 0.05
                ),
            }
        )
    return rows


def layer_block_rows(layer_units: dict[tuple[int, int], dict[str, Any]], kv_type: str, metric: str) -> list[dict[str, Any]]:
    rows = []
    block_map = {layer: block for block, layers in LAYER_BLOCKS.items() for layer in layers}
    for (layer, head), payload in layer_units.items():
        block = block_map.get(layer, None)
        if block is None:
            continue
        rows.append(
            {
                "kv_type": kv_type,
                "metric": metric,
                "layer": layer,
                "kv_head": head,
                "layer_block": block,
                "count": payload["count"],
                "contrib": payload["contrib"],
            }
        )
    return rows


def choose_top_bottom_sets(units: list[dict[str, Any]], frac: float) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    ordered = sorted(units, key=lambda item: (item["contrib"] if item["contrib"] is not None else float("-inf"), -item["layer"], -item["kv_head"]), reverse=True)
    n = max(1, math.ceil(len(ordered) * frac)) if ordered else 0
    top = {(int(item["layer"]), int(item["kv_head"])) for item in ordered[:n]}
    bottom = {(int(item["layer"]), int(item["kv_head"])) for item in ordered[-n:]}
    return top, bottom


def analyze_metric(
    rows: list[dict[str, Any]],
    *,
    kv_type: str,
    metric: str,
    bootstrap_seed: int,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    selected = [row for row in rows if row["phase"] == ANALYSIS_PHASE and row["kv_type"] == kv_type and row["metric"] == metric]
    task_units = task_layer_head_aggregate(selected)
    layer_units = layer_head_aggregate(selected)
    layer_macro_units = task_macro_units(task_units)
    micro_units = normalize_units(layer_units)
    task_macro_stats = concentration_metrics(layer_macro_units, bootstrap_seed=bootstrap_seed, bootstrap_repetitions=bootstrap_repetitions, use_task_macro=True)
    micro_stats = concentration_metrics(micro_units, bootstrap_seed=bootstrap_seed, bootstrap_repetitions=bootstrap_repetitions, use_task_macro=False)
    task_stats = task_level_stats(task_units)
    stable_rows = stable_negative_flags(task_units)
    top_micro, bottom_micro = choose_top_bottom_sets(micro_units, 0.25)
    top_macro, bottom_macro = choose_top_bottom_sets(layer_macro_units, 0.25)
    task_macro_by_unit = { (row["layer"], row["kv_head"]): row for row in layer_macro_units }
    bottom_macro_rows = [row for row in layer_macro_units if (row["layer"], row["kv_head"]) in bottom_macro]
    stable_bottom_macro = sum(1 for row in bottom_macro_rows if next((s for s in stable_rows if s["layer"] == row["layer"] and s["kv_head"] == row["kv_head"]), {}).get("stable_negative"))
    return {
        "kv_type": kv_type,
        "metric": metric,
        "selected_rows": len(selected),
        "task_units": task_units,
        "layer_units": layer_units,
        "layer_macro_units": layer_macro_units,
        "micro_stats": micro_stats,
        "task_macro_stats": task_macro_stats,
        "task_stats": task_stats,
        "stable_rows": stable_rows,
        "top_micro": top_micro,
        "bottom_micro": bottom_micro,
        "top_macro": top_macro,
        "bottom_macro": bottom_macro,
        "stable_bottom_macro_fraction": (stable_bottom_macro / len(bottom_macro_rows)) if bottom_macro_rows else None,
        "bottom_macro_total_mean": mean_or_none(row["contrib"] for row in bottom_macro_rows),
        "analysis_rows": selected,
    }


def metric_availability(rows: list[dict[str, Any]], kv_type: str, metric: str) -> bool:
    return any(row for row in rows if row["phase"] == ANALYSIS_PHASE and row["kv_type"] == kv_type and row["metric"] == metric)


def build_global_table(metrics_by_kv: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for (kv_type, metric), result in stable_sorted(metrics_by_kv.items()):
        for view, stats in (("micro", result["micro_stats"]), ("task_macro", result["task_macro_stats"])):
            row = {
                "kv_type": kv_type,
                "metric": metric,
                "view": view,
                "selected_rows": result["selected_rows"],
                "task_count": result["task_stats"]["task_count"],
                "task_coverage": result["task_stats"]["task_coverage"],
                "layer_head_count": len(result["layer_units"]),
                "total_units": stats["total_units"],
                "weighted_mean": stats["weighted_mean"],
                "task_macro_mean": result["task_stats"]["task_macro_mean"],
                "median": stats["median"],
                "p25": stats["p25"],
                "p75": stats["p75"],
                "p95": stats["p95"],
                "positive_contribution": stats["positive_contribution"],
                "negative_contribution": stats["negative_contribution"],
                "positive_fraction": stats["positive_fraction"],
                "harmful_fraction": stats["harmful_fraction"],
                "zero_fraction": stats["zero_fraction"],
                "bootstrap_mean": stats["bootstrap_mean"],
                "bootstrap_ci_low": stats["bootstrap_ci_low"],
                "bootstrap_ci_high": stats["bootstrap_ci_high"],
                "gini": stats["gini"],
                "hhi": stats["hhi"],
                "top_1pct_positive_share": stats["top_shares"]["top_1pct_positive_share"],
                "top_5pct_positive_share": stats["top_shares"]["top_5pct_positive_share"],
                "top_10pct_positive_share": stats["top_shares"]["top_10pct_positive_share"],
                "top_25pct_positive_share": stats["top_shares"]["top_25pct_positive_share"],
                "top_50pct_positive_share": stats["top_shares"]["top_50pct_positive_share"],
                "bottom_10_total": stats["bottom_10_total"],
                "bottom_25_total": stats["bottom_25_total"],
                "bottom_25_negative_fraction": stats["bottom_25_negative_fraction"],
            }
            out.append(row)
    return out


def flatten_layer_rows(metrics_by_kv: dict[tuple[str, str], dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    out = []
    for (kv_type, metric), result in stable_sorted(metrics_by_kv.items()):
        if kind == "layer":
            for (layer, head), payload in stable_sorted(result["layer_units"].items()):
                out.append(
                    {
                        "kv_type": kv_type,
                        "metric": metric,
                        "layer": layer,
                        "kv_head": head,
                        "count": payload["count"],
                        "contrib": payload["contrib"],
                        "weighted_mean": (payload["contrib"] / payload["count"]) if payload["count"] else None,
                    }
                )
        elif kind == "head":
            per_head: dict[int, dict[str, Any]] = defaultdict(lambda: {"count": 0, "contrib": 0.0})
            for (layer, head), payload in result["layer_units"].items():
                per_head[head]["count"] += payload["count"]
                per_head[head]["contrib"] += payload["contrib"]
            for head in stable_sorted(per_head):
                payload = per_head[head]
                out.append(
                    {
                        "kv_type": kv_type,
                        "metric": metric,
                        "kv_head": head,
                        "count": payload["count"],
                        "contrib": payload["contrib"],
                        "weighted_mean": (payload["contrib"] / payload["count"]) if payload["count"] else None,
                    }
                )
    return out


def write_rank_tables(metrics_by_kv: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rank_rows = []
    spearman_rows = []
    overlap_rows = []
    sign_rows = []
    for (kv_type, metric), result in stable_sorted(metrics_by_kv.items()):
        task_units = result["task_units"]
        task_names = sorted({task for task, _, _ in task_units})
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for (task, layer, head), payload in task_units.items():
            score = payload["contrib"] / payload["count"] if payload["count"] else None
            by_task[task].append({"layer": layer, "kv_head": head, "score": score, "count": payload["count"], "contrib": payload["contrib"]})
        task_rank_maps: dict[str, dict[tuple[int, int], int]] = {}
        for task, items in by_task.items():
            ordered = sorted(items, key=lambda item: (item["score"] if item["score"] is not None else float("-inf"), -item["layer"], -item["kv_head"]), reverse=True)
            for rank, item in enumerate(ordered, start=1):
                rank_rows.append(
                    {
                        "kv_type": kv_type,
                        "metric": metric,
                        "task": task,
                        "layer": item["layer"],
                        "kv_head": item["kv_head"],
                        "rank": rank,
                        "score": item["score"],
                        "count": item["count"],
                        "contrib": item["contrib"],
                    }
                )
            task_rank_maps[task] = {(item["layer"], item["kv_head"]): idx + 1 for idx, item in enumerate(ordered)}
        for left_idx, left in enumerate(task_names):
            left_scores = task_rank_maps[left]
            left_items = by_task[left]
            for right in task_names[left_idx + 1 :]:
                right_scores = task_rank_maps[right]
                common = sorted(set(left_scores) & set(right_scores))
                if len(common) >= 2:
                    x = [left_scores[key] for key in common]
                    y = [right_scores[key] for key in common]
                    rho = spearman(x, y)
                else:
                    rho = None
                spearman_rows.append(
                    {
                        "kv_type": kv_type,
                        "metric": metric,
                        "task_left": left,
                        "task_right": right,
                        "common_count": len(common),
                        "spearman_rho": rho,
                    }
                )
                for pct, label in ((0.10, "10"), (0.25, "25")):
                    left_order = sorted(left_scores.items(), key=lambda item: item[1])
                    right_order = sorted(right_scores.items(), key=lambda item: item[1])
                    n = max(1, math.ceil(len(left_order) * pct))
                    top_left = {key for key, _ in left_order[:n]}
                    top_right = {key for key, _ in right_order[:n]}
                    bottom_left = {key for key, _ in left_order[-n:]}
                    bottom_right = {key for key, _ in right_order[-n:]}
                    overlap_rows.append(
                        {
                            "kv_type": kv_type,
                            "metric": metric,
                            "task_left": left,
                            "task_right": right,
                            "bucket": f"top_{label}pct",
                            "overlap": jaccard(top_left, top_right),
                            "set_size": len(top_left),
                            "intersection_size": len(top_left & top_right),
                        }
                    )
                    overlap_rows.append(
                        {
                            "kv_type": kv_type,
                            "metric": metric,
                            "task_left": left,
                            "task_right": right,
                            "bucket": f"bottom_{label}pct",
                            "overlap": jaccard(bottom_left, bottom_right),
                            "set_size": len(bottom_left),
                            "intersection_size": len(bottom_left & bottom_right),
                        }
                    )
        for key in sorted({key for task in task_names for key in task_rank_maps[task]}):
            vals = []
            for task in task_names:
                if key in task_rank_maps[task]:
                    vals.append((task, task_rank_maps[task][key], by_task[task][task_rank_maps[task][key] - 1]["score"]))
            if not vals:
                continue
            scores = [v[2] for v in vals if v[2] is not None]
            sign_rows.append(
                {
                    "kv_type": kv_type,
                    "metric": metric,
                    "layer": key[0],
                    "kv_head": key[1],
                    "task_coverage": len(vals),
                    "task_count": len(task_names),
                    "positive_task_count": sum(1 for _, _, score in vals if score is not None and score > 0),
                    "negative_task_count": sum(1 for _, _, score in vals if score is not None and score < 0),
                    "zero_task_count": sum(1 for _, _, score in vals if score == 0),
                    "mean_rank": mean_or_none(v[1] for v in vals),
                    "std_rank": float(statistics.pstdev(v[1] for v in vals)) if len(vals) > 1 else None,
                    "sign_consistency_rate": max(
                        sum(1 for _, _, score in vals if score is not None and score > 0) / len(vals),
                        sum(1 for _, _, score in vals if score is not None and score < 0) / len(vals),
                        sum(1 for _, _, score in vals if score == 0) / len(vals),
                    ),
                }
            )
    return rank_rows, spearman_rows, overlap_rows, sign_rows


def layer_block_analysis(metrics_by_kv: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    block_rows = []
    summary = {}
    for (kv_type, metric), result in stable_sorted(metrics_by_kv.items()):
        layer_units = result["layer_units"]
        for row in layer_block_rows(layer_units, kv_type, metric):
            block_rows.append(row)
        per_block: dict[int, dict[str, Any]] = defaultdict(lambda: {"count": 0, "contrib": 0.0, "layer_heads": set(), "positive": 0, "negative": 0})
        block_map = {layer: block for block, layers in LAYER_BLOCKS.items() for layer in layers}
        stable_layers = { (row["layer"], row["kv_head"]) for row in result["stable_rows"] if row["stable_negative"] }
        for (layer, head), payload in layer_units.items():
            block = block_map[layer]
            per_block[block]["count"] += payload["count"]
            per_block[block]["contrib"] += payload["contrib"]
            per_block[block]["layer_heads"].add((layer, head))
            if (layer, head) in stable_layers:
                per_block[block]["positive"] += 1
            if payload["contrib"] < 0:
                per_block[block]["negative"] += 1
        block_summary_rows = []
        for block in sorted(per_block):
            payload = per_block[block]
            block_summary_rows.append(
                {
                    "kv_type": kv_type,
                    "metric": metric,
                    "layer_block": block,
                    "layer_head_count": len(payload["layer_heads"]),
                    "count": payload["count"],
                    "weighted_gain": (payload["contrib"] / payload["count"]) if payload["count"] else None,
                    "task_macro_gain": mean_or_none(
                        (result["task_macro_units"][idx]["contrib"] if False else [])
                    ),
                    "positive_contribution": payload["contrib"] if payload["contrib"] > 0 else 0.0,
                    "negative_contribution": payload["contrib"] if payload["contrib"] < 0 else 0.0,
                    "harmful_fraction": payload["negative"] / len(payload["layer_heads"]) if payload["layer_heads"] else None,
                    "stable_negative_layer_head_count": payload["positive"],
                }
            )
        ordered = sorted(block_summary_rows, key=lambda row: (row["weighted_gain"] if row["weighted_gain"] is not None else float("-inf"), -row["layer_block"]), reverse=True)
        summary[(kv_type, metric)] = {
            "block_rows": block_summary_rows,
            "ordered": ordered,
            "highest_benefit_block": ordered[0]["layer_block"] if ordered else None,
            "lowest_benefit_block": ordered[-1]["layer_block"] if ordered else None,
            "target_kv_type": kv_type,
        }
    return block_rows, [row for _, result in stable_sorted(summary.items()) for row in result["block_rows"]], summary


def build_inventory(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        name: {
            "exists": path.exists(),
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()

    report_root = resolve_root(Path(args.report_root), "pattern_gain_map.csv")
    result_root = resolve_root(Path(args.result_root), "")
    report_path = Path(report_root.resolved or args.report_root)
    result_path = Path(result_root.resolved or args.result_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    raw_rows = read_csv_rows(report_path / "pattern_gain_map.csv")
    numeric_rows = [row_to_numeric(row) for row in raw_rows]

    # Schema audit.
    schema_audit = {
        "schema_version": "insight_v2.pattern_gain_concentration.schema_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "report_root": str(report_path),
        "result_root": str(result_path),
        "row_count": len(raw_rows),
        "columns": list(raw_rows[0].keys()) if raw_rows else [],
        "required_columns": CSV_KEYS + ["count", "mean", "std"],
        "primary_key_fields": CSV_KEYS,
        "duplicate_row_count": sum(1 for v in Counter(tuple(sorted(r.items())) for r in raw_rows).values() if v > 1),
        "duplicate_primary_key_count": sum(1 for v in Counter(row_key(r, CSV_KEYS) for r in raw_rows).values() if v > 1),
        "blank_counts": {field: sum(1 for row in raw_rows if row.get(field, "") == "") for field in (raw_rows[0].keys() if raw_rows else [])},
        "negative_count_rows": sum(1 for row in numeric_rows if row["count"] is not None and row["count"] < 0),
        "nan_string_count": sum(1 for row in raw_rows for value in row.values() if isinstance(value, str) and value.lower() == "nan"),
        "inf_string_count": sum(1 for row in raw_rows for value in row.values() if isinstance(value, str) and value.lower() in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}),
        "unique_values": {
            field: stable_sorted({row.get(field, "") for row in raw_rows}) for field in ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric"]
        },
    }

    tasks_present = stable_sorted({row["task"] for row in numeric_rows})
    phases_present = stable_sorted({row["phase"] for row in numeric_rows})
    kv_types_present = stable_sorted({row["kv_type"] for row in numeric_rows})
    layers_present = stable_sorted({parse_int(row["layer"]) for row in raw_rows if parse_int(row["layer"]) is not None})
    heads_present = stable_sorted({parse_int(row["kv_head"]) for row in raw_rows if parse_int(row["kv_head"]) is not None})

    schema_audit["layer_count"] = len(layers_present)
    schema_audit["kv_head_count"] = len(heads_present)
    schema_audit["has_k"] = "k" in kv_types_present
    schema_audit["has_v"] = "v" in kv_types_present
    schema_audit["has_prefill"] = "prefill" in phases_present
    schema_audit["has_decode"] = "decode" in phases_present
    schema_audit["task_coverage_complete"] = sorted(tasks_present) == EXPECTED_TASKS
    schema_audit["expected_tasks"] = EXPECTED_TASKS
    schema_audit["not_collected"] = {
        "min_max_fields": ["min", "max"],
        "missing_main_metric_v": V_MAIN_METRIC,
        "missing_if_any": [],
    }

    # Input inventory.
    input_inventory = {
        "schema_version": "insight_v2.pattern_gain_concentration.input_inventory",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "report_root": str(report_path),
        "result_root": str(result_path),
        "inputs": build_inventory(
            {
                "pattern_gain_map.csv": report_path / "pattern_gain_map.csv",
                "completion.json": report_path / "completion.json",
                "reference_manifest.json": report_path / "reference_manifest.json",
                "matching_oracle_gap.csv": report_path / "matching_oracle_gap.csv",
                "dynamic_pattern_utility.csv": report_path / "dynamic_pattern_utility.csv",
                "v_gate_confusion.csv": report_path / "v_gate_confusion.csv",
            }
        ),
    }

    semantics_lines = [
        "# Column Semantics",
        "",
        "| column | inferred type | notes |",
        "| --- | --- | --- |",
        "| task | categorical | dataset/task id |",
        "| phase | categorical | `prefill` or `decode` |",
        "| kv_type | categorical | `k` or `v` |",
        "| layer | integer | transformer layer index |",
        "| kv_head | integer | KV head index |",
        "| bucket | categorical | bucket partition label |",
        "| metric | categorical | metric name |",
        "| count | integer | sample count for the row |",
        "| mean | float | aggregate mean value |",
        "| std | float | aggregate standard deviation |",
        "",
        f"Main analysis phase: `{ANALYSIS_PHASE}`",
        f"Main K metric: `{K_MAIN_METRIC}`",
        f"V main metric: `{V_MAIN_METRIC}`",
        f"V proxy metric: `{V_PROXY_METRIC}`",
    ]

    metrics_by_kv: dict[tuple[str, str], dict[str, Any]] = {}
    selected_metrics = [
        ("k", K_MAIN_METRIC),
        ("k", "range_contraction"),
        ("v", V_MAIN_METRIC),
        ("v", V_PROXY_METRIC),
    ]
    for kv_type, metric in selected_metrics:
        if metric_availability(numeric_rows, kv_type, metric):
            metrics_by_kv[(kv_type, metric)] = analyze_metric(
                numeric_rows,
                kv_type=kv_type,
                metric=metric,
                bootstrap_seed=args.bootstrap_seed,
                bootstrap_repetitions=args.bootstrap_repetitions,
            )

    # Global tables.
    global_rows = build_global_table(metrics_by_kv)
    by_task_rows = []
    by_layer_rows = []
    by_head_rows = []
    by_layer_head_rows = []
    by_task_layer_head_rows = []
    for (kv_type, metric), result in stable_sorted(metrics_by_kv.items()):
        for task, payload in sorted(result["task_units"].items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
            task_name, layer, head = task
            by_task_layer_head_rows.append(
                {
                    "kv_type": kv_type,
                    "metric": metric,
                    "task": task_name,
                    "layer": layer,
                    "kv_head": head,
                    "count": payload["count"],
                    "contrib": payload["contrib"],
                    "weighted_mean": (payload["contrib"] / payload["count"]) if payload["count"] else None,
                }
            )
        task_payload = task_aggregate([row for row in result["analysis_rows"]])
        for task_name in sorted(task_payload):
            payload = task_payload[task_name]
            by_task_rows.append(
                {
                    "kv_type": kv_type,
                    "metric": metric,
                    "task": task_name,
                    "count": payload["count"],
                    "contrib": payload["contrib"],
                    "weighted_mean": (payload["contrib"] / payload["count"]) if payload["count"] else None,
                }
            )
        for (layer, head), payload in sorted(result["layer_units"].items()):
            by_layer_rows.append(
                {
                    "kv_type": kv_type,
                    "metric": metric,
                    "layer": layer,
                    "count": payload["count"],
                    "contrib": payload["contrib"],
                    "weighted_mean": (payload["contrib"] / payload["count"]) if payload["count"] else None,
                }
            )
            by_head_rows.append(
                {
                    "kv_type": kv_type,
                    "metric": metric,
                    "kv_head": head,
                    "count": payload["count"],
                    "contrib": payload["contrib"],
                    "weighted_mean": (payload["contrib"] / payload["count"]) if payload["count"] else None,
                }
            )
            by_layer_head_rows.append(
                {
                    "kv_type": kv_type,
                    "metric": metric,
                    "layer": layer,
                    "kv_head": head,
                    "count": payload["count"],
                    "contrib": payload["contrib"],
                    "weighted_mean": (payload["contrib"] / payload["count"]) if payload["count"] else None,
                }
            )

    # Concentration tables.
    concentration_rows = []
    concentration_summary = {}
    for (kv_type, metric), result in stable_sorted(metrics_by_kv.items()):
        for view_name, units in (("micro", normalize_units(result["layer_units"])), ("task_macro", result["layer_macro_units"])):
            ordered = sorted(units, key=lambda item: (item["contrib"] if item["contrib"] is not None else float("-inf"), -item["layer"], -item["kv_head"]), reverse=True)
            pos = [max(float(unit["contrib"] or 0.0), 0.0) for unit in ordered]
            pos_total = sum(pos)
            total_units = len(ordered)
            for frac in [0.01, 0.05, 0.10, 0.25, 0.50]:
                n = max(1, math.ceil(total_units * frac)) if total_units else 0
                concentration_rows.append(
                    {
                        "kv_type": kv_type,
                        "metric": metric,
                        "view": view_name,
                        "bucket": f"top_{int(frac * 100)}pct",
                        "share_of_positive_contribution": (sum(pos[:n]) / pos_total) if pos_total and n else None,
                        "unit_count": n,
                        "total_units": total_units,
                    }
                )
            concentration_rows.extend(
                [
                    {
                        "kv_type": kv_type,
                        "metric": metric,
                        "view": view_name,
                        "bucket": "bottom_10pct_total",
                        "share_of_positive_contribution": None,
                        "unit_count": max(1, math.ceil(total_units * 0.10)) if total_units else None,
                        "total_units": total_units,
                        "total_contribution": sum(float(unit["contrib"] or 0.0) for unit in ordered[-max(1, math.ceil(total_units * 0.10)) :]) if total_units else None,
                    },
                    {
                        "kv_type": kv_type,
                        "metric": metric,
                        "view": view_name,
                        "bucket": "bottom_25pct_total",
                        "share_of_positive_contribution": None,
                        "unit_count": max(1, math.ceil(total_units * 0.25)) if total_units else None,
                        "total_units": total_units,
                        "total_contribution": sum(float(unit["contrib"] or 0.0) for unit in ordered[-max(1, math.ceil(total_units * 0.25)) :]) if total_units else None,
                    },
                ]
            )
        concentration_summary[(kv_type, metric)] = {
            "micro": result["micro_stats"],
            "task_macro": result["task_macro_stats"],
            "stable_negative_bottom25_fraction": result["stable_bottom_macro_fraction"],
            "stable_negative_bottom25_mean": result["bottom_macro_total_mean"],
        }

    # Rank / overlap / sign stability.
    rank_rows, spearman_rows, overlap_rows, sign_rows = write_rank_tables(metrics_by_kv)

    # Block analysis.
    block_rows = []
    block_summary_rows = []
    block_ranking_lines = ["# Layer Block Ranking", ""]
    block_summary: dict[str, Any] = {}
    for (kv_type, metric), result in stable_sorted(metrics_by_kv.items()):
        block_map = {layer: block for block, layers in LAYER_BLOCKS.items() for layer in layers}
        per_block: dict[int, dict[str, Any]] = defaultdict(lambda: {"count": 0, "contrib": 0.0, "layer_heads": set(), "stable_negative": 0, "negative": 0})
        for (layer, head), payload in result["layer_units"].items():
            block = block_map.get(layer)
            if block is None:
                continue
            per_block[block]["count"] += payload["count"]
            per_block[block]["contrib"] += payload["contrib"]
            per_block[block]["layer_heads"].add((layer, head))
        stable_map = {(row["layer"], row["kv_head"]): row["stable_negative"] for row in result["stable_rows"]}
        for block in sorted(per_block):
            payload = per_block[block]
            stable_count = sum(1 for key in payload["layer_heads"] if stable_map.get(key))
            task_block_means = []
            for task in sorted({task for task, _, _ in result["task_units"]}):
                unit_vals = []
                for (layer, head) in payload["layer_heads"]:
                    task_payload = result["task_units"].get((task, layer, head))
                    if task_payload and task_payload["count"]:
                        unit_vals.append(task_payload["contrib"] / task_payload["count"])
                if unit_vals:
                    task_block_means.append(float(statistics.fmean(unit_vals)))
            row = {
                "kv_type": kv_type,
                "metric": metric,
                "layer_block": block,
                "layer_head_count": len(payload["layer_heads"]),
                "count": payload["count"],
                "weighted_gain": (payload["contrib"] / payload["count"]) if payload["count"] else None,
                "task_macro_gain": mean_or_none(task_block_means),
                "positive_contribution": payload["contrib"] if payload["contrib"] > 0 else 0.0,
                "negative_contribution": payload["contrib"] if payload["contrib"] < 0 else 0.0,
                "harmful_fraction": (sum(1 for (layer, head) in payload["layer_heads"] if result["layer_units"][(layer, head)]["contrib"] < 0) / len(payload["layer_heads"])) if payload["layer_heads"] else None,
                "stable_negative_layer_head_count": stable_count,
            }
            block_rows.append(row)
            block_summary_rows.append(row)
        ordered = sorted([row for row in block_rows if row["kv_type"] == kv_type and row["metric"] == metric], key=lambda row: (row["weighted_gain"] if row["weighted_gain"] is not None else float("-inf"), -row["layer_block"]), reverse=True)
        block_summary[(kv_type, metric)] = {
            "highest_benefit_block": ordered[0]["layer_block"] if ordered else None,
            "lowest_benefit_block": ordered[-1]["layer_block"] if ordered else None,
            "target_kv_type": kv_type,
        }
        block_ranking_lines.extend(
            [
                f"## {kv_type} / {metric}",
                "",
                f"- highest_benefit_block: `{block_summary[(kv_type, metric)]['highest_benefit_block']}`",
                f"- lowest_benefit_block: `{block_summary[(kv_type, metric)]['lowest_benefit_block']}`",
                f"- target_kv_type: `{kv_type}`",
                "",
            ]
        )

    # Decision.
    k_result = metrics_by_kv.get(("k", K_MAIN_METRIC))
    v_main_available = metric_availability(numeric_rows, "v", V_MAIN_METRIC)
    decision_reason = []
    if k_result:
        k_micro = k_result["micro_stats"]["top_shares"]["top_25pct_positive_share"]
        k_macro = k_result["task_macro_stats"]["top_shares"]["top_25pct_positive_share"]
        k_bottom_stable = k_result["stable_bottom_macro_fraction"]
        k_bottom_mean = k_result["bottom_macro_total_mean"]
        decision_reason.append(f"K top25 micro={k_micro}")
        decision_reason.append(f"K top25 macro={k_macro}")
        decision_reason.append(f"K bottom25 stable={k_bottom_stable}")
        decision_reason.append(f"K bottom25 mean={k_bottom_mean}")
    v_main_result = metrics_by_kv.get(("v", V_MAIN_METRIC))
    if v_main_result:
        decision_reason.append(f"V top25 micro={v_main_result['micro_stats']['top_shares']['top_25pct_positive_share']}")
        decision_reason.append(f"V top25 macro={v_main_result['task_macro_stats']['top_shares']['top_25pct_positive_share']}")
        decision_reason.append(f"V bottom25 stable={v_main_result['stable_bottom_macro_fraction']}")
    v_proxy_result = metrics_by_kv.get(("v", V_PROXY_METRIC))
    if v_proxy_result:
        decision_reason.append(f"V proxy top25 micro={v_proxy_result['micro_stats']['top_shares']['top_25pct_positive_share']}")
        decision_reason.append(f"V proxy top25 macro={v_proxy_result['task_macro_stats']['top_shares']['top_25pct_positive_share']}")
        decision_reason.append(f"V proxy bottom25 stable={v_proxy_result['stable_bottom_macro_fraction']}")
    if not v_main_available or not v_main_result:
        decision = "data_insufficient"
    else:
        decision = "not_supported"
        if k_result:
            if (
                (k_result["micro_stats"]["top_shares"]["top_25pct_positive_share"] or 0) > 0.70
                and (k_result["task_macro_stats"]["top_shares"]["top_25pct_positive_share"] or 0) > 0.70
            ) or (
                (k_result["stable_bottom_macro_fraction"] or 0) >= 0.75
                and (k_result["bottom_macro_total_mean"] or 0) < 0
            ):
                decision = "supported"
    if decision != "supported":
        if v_main_available and v_main_result:
            if (
                (v_main_result["micro_stats"]["top_shares"]["top_25pct_positive_share"] or 0) > 0.70
                and (v_main_result["task_macro_stats"]["top_shares"]["top_25pct_positive_share"] or 0) > 0.70
            ) or (
                (v_main_result["stable_bottom_macro_fraction"] or 0) >= 0.75
                and (v_main_result["bottom_macro_total_mean"] or 0) < 0
            ):
                decision = "supported"

    adaptive_decision = {
        "schema_version": "insight_v2.pattern_gain_concentration.adaptive_allocation_decision",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "decision": decision,
        "reason": decision_reason,
        "k": {
            "main_metric": K_MAIN_METRIC,
            "top25_micro": k_result["micro_stats"]["top_shares"]["top_25pct_positive_share"] if k_result else None,
            "top25_task_macro": k_result["task_macro_stats"]["top_shares"]["top_25pct_positive_share"] if k_result else None,
            "stable_bottom25_fraction": k_result["stable_bottom_macro_fraction"] if k_result else None,
            "bottom25_mean": k_result["bottom_macro_total_mean"] if k_result else None,
        },
        "v": {
            "main_metric": V_MAIN_METRIC,
            "main_metric_status": "not_collected" if not v_main_available else "collected",
            "main_top25_micro": v_main_result["micro_stats"]["top_shares"]["top_25pct_positive_share"] if v_main_result else None,
            "main_top25_task_macro": v_main_result["task_macro_stats"]["top_shares"]["top_25pct_positive_share"] if v_main_result else None,
            "main_stable_bottom25_fraction": v_main_result["stable_bottom_macro_fraction"] if v_main_result else None,
            "main_bottom25_mean": v_main_result["bottom_macro_total_mean"] if v_main_result else None,
            "proxy_metric": V_PROXY_METRIC,
            "proxy_top25_micro": v_proxy_result["micro_stats"]["top_shares"]["top_25pct_positive_share"] if v_proxy_result else None,
            "proxy_top25_task_macro": v_proxy_result["task_macro_stats"]["top_shares"]["top_25pct_positive_share"] if v_proxy_result else None,
            "proxy_stable_bottom25_fraction": v_proxy_result["stable_bottom_macro_fraction"] if v_proxy_result else None,
            "proxy_bottom25_mean": v_proxy_result["bottom_macro_total_mean"] if v_proxy_result else None,
        },
    }

    # Write outputs.
    atomic_write_json(output_root / "schema_audit.json", schema_audit)
    atomic_write_text(output_root / "schema_audit.md", "# Schema Audit\n\n" + json.dumps(schema_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_write_json(output_root / "input_inventory.json", input_inventory)
    atomic_write_text(output_root / "input_inventory.md", "# Input Inventory\n\n" + json.dumps(input_inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_write_text(output_root / "column_semantics.md", "\n".join(semantics_lines) + "\n")
    write_csv(output_root / "pattern_gain_global_kv.csv", global_rows, list(global_rows[0].keys()) if global_rows else ["kv_type", "metric", "view"])
    write_csv(output_root / "pattern_gain_by_task_kv.csv", by_task_rows, list(by_task_rows[0].keys()) if by_task_rows else ["kv_type", "metric", "task"])
    write_csv(output_root / "pattern_gain_by_layer.csv", by_layer_rows, list(by_layer_rows[0].keys()) if by_layer_rows else ["kv_type", "metric", "layer"])
    write_csv(output_root / "pattern_gain_by_head.csv", by_head_rows, list(by_head_rows[0].keys()) if by_head_rows else ["kv_type", "metric", "kv_head"])
    write_csv(output_root / "pattern_gain_by_layer_head.csv", by_layer_head_rows, list(by_layer_head_rows[0].keys()) if by_layer_head_rows else ["kv_type", "metric", "layer", "kv_head"])
    write_csv(output_root / "pattern_gain_by_task_layer_head.csv", by_task_layer_head_rows, list(by_task_layer_head_rows[0].keys()) if by_task_layer_head_rows else ["kv_type", "metric", "task", "layer", "kv_head"])
    write_csv(output_root / "pattern_gain_concentration.csv", concentration_rows, list(concentration_rows[0].keys()) if concentration_rows else ["kv_type", "metric", "view"])
    for kv_type in ("k", "v"):
        metric = K_MAIN_METRIC if kv_type == "k" else V_MAIN_METRIC
        result = metrics_by_kv.get((kv_type, metric))
        if result:
            rows_out = []
            for view_name, stats in (("micro", result["micro_stats"]), ("task_macro", result["task_macro_stats"])):
                rows_out.append(
                    {
                        "kv_type": kv_type,
                        "metric": metric,
                        "view": view_name,
                        "total_units": stats["total_units"],
                        "weighted_mean": stats["weighted_mean"],
                        "task_macro_mean": result["task_stats"]["task_macro_mean"],
                        "median": stats["median"],
                        "p25": stats["p25"],
                        "p75": stats["p75"],
                        "p95": stats["p95"],
                        "positive_contribution": stats["positive_contribution"],
                        "negative_contribution": stats["negative_contribution"],
                        "positive_fraction": stats["positive_fraction"],
                        "harmful_fraction": stats["harmful_fraction"],
                        "zero_fraction": stats["zero_fraction"],
                        "bootstrap_mean": stats["bootstrap_mean"],
                        "bootstrap_ci_low": stats["bootstrap_ci_low"],
                        "bootstrap_ci_high": stats["bootstrap_ci_high"],
                        "top_1pct_positive_share": stats["top_shares"]["top_1pct_positive_share"],
                        "top_5pct_positive_share": stats["top_shares"]["top_5pct_positive_share"],
                        "top_10pct_positive_share": stats["top_shares"]["top_10pct_positive_share"],
                        "top_25pct_positive_share": stats["top_shares"]["top_25pct_positive_share"],
                        "top_50pct_positive_share": stats["top_shares"]["top_50pct_positive_share"],
                        "bottom_10_total": stats["bottom_10_total"],
                        "bottom_25_total": stats["bottom_25_total"],
                        "bottom_25_negative_fraction": stats["bottom_25_negative_fraction"],
                        "gini": stats["gini"],
                        "hhi": stats["hhi"],
                    }
                )
            write_csv(output_root / f"pattern_gain_concentration_{kv_type}.csv", rows_out, list(rows_out[0].keys()))
    write_csv(output_root / "layer_head_rank_by_task.csv", rank_rows, list(rank_rows[0].keys()) if rank_rows else ["kv_type", "metric", "task"])
    write_csv(output_root / "layer_head_task_spearman.csv", spearman_rows, list(spearman_rows[0].keys()) if spearman_rows else ["kv_type", "metric", "task_left", "task_right"])
    write_csv(output_root / "layer_head_top_overlap.csv", overlap_rows, list(overlap_rows[0].keys()) if overlap_rows else ["kv_type", "metric", "task_left", "task_right", "bucket"])
    write_csv(output_root / "layer_head_sign_stability.csv", sign_rows, list(sign_rows[0].keys()) if sign_rows else ["kv_type", "metric", "layer", "kv_head"])
    write_csv(output_root / "pattern_gain_by_layer_block.csv", block_rows, list(block_rows[0].keys()) if block_rows else ["kv_type", "metric", "layer_block"])
    atomic_write_text(output_root / "layer_block_ranking.md", "\n".join(block_ranking_lines) + "\n")
    atomic_write_json(output_root / "adaptive_allocation_decision.json", adaptive_decision)
    decision_lines = [
        "# Adaptive Allocation Decision",
        "",
        f"Decision: `{adaptive_decision['decision']}`",
        "",
        *[f"- {item}" for item in decision_reason],
        "",
        f"K top25 micro: `{adaptive_decision['k']['top25_micro']}`",
        f"K top25 task_macro: `{adaptive_decision['k']['top25_task_macro']}`",
        f"V main metric status: `{adaptive_decision['v']['main_metric_status']}`",
        f"V main top25 micro: `{adaptive_decision['v']['main_top25_micro']}`",
        f"V main top25 task_macro: `{adaptive_decision['v']['main_top25_task_macro']}`",
        f"V proxy top25 micro: `{adaptive_decision['v']['proxy_top25_micro']}`",
        f"V proxy top25 task_macro: `{adaptive_decision['v']['proxy_top25_task_macro']}`",
    ]
    atomic_write_text(output_root / "adaptive_allocation_decision.md", "\n".join(decision_lines) + "\n")
    summary = {
        "schema_version": "insight_v2.pattern_gain_concentration.summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "input_root": str(report_path),
        "output_root": str(output_root),
        "tasks_present": tasks_present,
        "phase_set": phases_present,
        "kv_types_present": kv_types_present,
        "layer_count": len(layers_present),
        "kv_head_count": len(heads_present),
        "task_coverage_complete": schema_audit["task_coverage_complete"],
        "analysis_metrics": sorted(f"{k}:{m}" for k, m in metrics_by_kv.keys()),
        "decision": adaptive_decision["decision"],
        "k_top25_micro": adaptive_decision["k"]["top25_micro"],
        "k_top25_task_macro": adaptive_decision["k"]["top25_task_macro"],
        "v_main_top25_micro": adaptive_decision["v"]["main_top25_micro"],
        "v_main_top25_task_macro": adaptive_decision["v"]["main_top25_task_macro"],
        "v_proxy_top25_micro": adaptive_decision["v"]["proxy_top25_micro"],
        "v_proxy_top25_task_macro": adaptive_decision["v"]["proxy_top25_task_macro"],
    }
    atomic_write_json(output_root / "summary.json", summary)
    atomic_write_text(
        output_root / "summary.md",
        "\n".join(
            [
                "# Pattern Gain Concentration Summary",
                "",
                f"Decision: `{adaptive_decision['decision']}`",
                f"K top25 micro: `{adaptive_decision['k']['top25_micro']}`",
                f"K top25 task_macro: `{adaptive_decision['k']['top25_task_macro']}`",
                f"V main metric status: `{adaptive_decision['v']['main_metric_status']}`",
                f"V main top25 micro: `{adaptive_decision['v']['main_top25_micro']}`",
                f"V main top25 task_macro: `{adaptive_decision['v']['main_top25_task_macro']}`",
                f"V proxy top25 micro: `{adaptive_decision['v']['proxy_top25_micro']}`",
                f"V proxy top25 task_macro: `{adaptive_decision['v']['proxy_top25_task_macro']}`",
            ]
            + [""]
        ),
    )


if __name__ == "__main__":
    main()
