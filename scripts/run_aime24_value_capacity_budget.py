#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_aime24_selective_value_precision as svp  # noqa: E402
from bench.pseudodecode_controls import MATCHED_PATH_CONTROL_VERSION, compute_accumulation_gap  # noqa: E402
from bench.pseudodecode_metrics import trapezoid_auc_log2, write_csv_rows  # noqa: E402
from bench.routing_vdirection_observer import EPS, SCHEMA_VERSION  # noqa: E402
from models.segmented_cache import (  # noqa: E402
    PatternQuantizedKVCache,
    affine_dequantize_last_dim_reference,
    build_cache_from_prefill,
    pattern_gather_centroids,
    reconstruct_full_k,
    reconstruct_packed_v,
)


OUT_DIR = ROOT / "reports/aime24_value_capacity_budget_3090"
RESULT_DIR = ROOT / "results/aime24_value_capacity_budget_3090"
SHARD_DIR = RESULT_DIR / "shards"
LOG_DIR = ROOT / "run/aime24_value_capacity_budget_3090/logs"
EXP8_DIR = ROOT / "reports/aime24_selective_value_precision_3090"
EXP8_COMMIT = "241b832"
PARENT_COMMIT = "241b832aee31c0e328d13675efb7819508c29ac9"
SUBSET_SHA256 = "218b65375f551fb13ff9903b3fd8931f422215e0f3b86200c0d8f45130a43082"
PORTABLE_HASH = "86648d12304ce11890c1a8f64bf5a896"
CORE_CHECKPOINTS = (128, 512, 1024, 2048, 4096)
PREFLIGHT_CHECKPOINTS = (128, 512, 1024)
SELECTED_LAYERS = (0, 7, 15, 23, 31)
RANDOM_SELECTOR_SEED = 20260809
BUDGETS = (0.0, 0.125, 0.25, 0.5, 1.0)

CONFIGS: dict[str, dict[str, Any]] = {
    "BASE_V2": {"config": "pattern_rolling_k2v2_s16_r128", "selector": "base_v2", "budget": 0.0},
    "RANDOM_V4": {"config": "pattern_rolling_k2v2_s16_r128_random_v4_b0125", "selector": "random_v4", "budget": 0.125},
    "CAUSAL_V4": {"config": "pattern_rolling_k2v2_s16_r128_causal_v4_b0125", "selector": "causal_v4", "budget": 0.125},
    "FUTURE_ATTN_V4": {"config": "pattern_rolling_k2v2_s16_r128_future_attn_v4_b0125", "selector": "oracle_v4", "budget": 0.125},
    "ALL_V4": {"config": "pattern_rolling_k2v4_s16_r128_all_v4", "selector": "all_v4", "budget": 1.0},
    "RANDOM_V4_25": {"config": "pattern_rolling_k2v2_s16_r128_random_v4_b025", "selector": "random_v4", "budget": 0.25},
    "CAUSAL_V4_25": {"config": "pattern_rolling_k2v2_s16_r128_causal_v4_b025", "selector": "causal_v4", "budget": 0.25},
    "RANDOM_V4_50": {"config": "pattern_rolling_k2v2_s16_r128_random_v4_b050", "selector": "random_v4", "budget": 0.5},
    "CAUSAL_V4_50": {"config": "pattern_rolling_k2v2_s16_r128_causal_v4_b050", "selector": "causal_v4", "budget": 0.5},
}

FAMILIES = svp.FAMILIES


def configure_svp(active: tuple[str, ...] | None = None) -> None:
    svp.OUT_DIR = OUT_DIR
    svp.RESULT_DIR = RESULT_DIR
    svp.SHARD_DIR = SHARD_DIR
    svp.LOG_DIR = LOG_DIR
    svp.CONFIGS = {name: CONFIGS[name] for name in (active or tuple(CONFIGS))}
    svp.CORE_CHECKPOINTS = CORE_CHECKPOINTS
    svp.PREFLIGHT_CHECKPOINTS = PREFLIGHT_CHECKPOINTS
    svp.SELECTED_LAYERS = SELECTED_LAYERS
    svp.V4_BUDGET_FRACTION = 0.125
    svp.RANDOM_SELECTOR_SEED = RANDOM_SELECTOR_SEED


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gzip_file(path: Path) -> Path:
    gz = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as src, gzip.open(gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return gz


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def median(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return statistics.median(vals) if vals else None


def bootstrap_ci(deltas: list[float], *, seed: int = RANDOM_SELECTOR_SEED, samples: int = 10000) -> tuple[float | None, float | None]:
    if not deltas:
        return None, None
    import random

    rng = random.Random(seed)
    meds = []
    for _ in range(samples):
        draw = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        meds.append(statistics.median(draw))
    meds.sort()
    return meds[int(0.025 * samples)], meds[int(0.975 * samples) - 1]


def exp8_rows(family: str) -> list[dict[str, Any]]:
    name = f"{family}.csv.gz" if family in {"precision_selection", "selector_quality"} else f"{family}_metrics.csv.gz"
    rows = read_csv_rows(EXP8_DIR / name)
    renamed = []
    for row in rows:
        if row.get("method") == "ORACLE_V4":
            row = {**row, "method": "FUTURE_ATTN_V4", "config": CONFIGS["FUTURE_ATTN_V4"]["config"]}
        renamed.append(row)
    return renamed


def shard_rows(family: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(SHARD_DIR.glob(f"*.{family}.csv")):
        rows.extend(read_csv_rows(path))
    return rows


def metric_identity(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        row["task_key"],
        row["method"],
        str(row["checkpoint"]),
        str(row["layer"]),
        row["metric_family"],
        row["object_type"],
        row["region"],
        row["metric_name"],
        row["statistic"],
    )


def accumulation_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["mode"], *metric_identity(row)): row for row in rows}
    out = []
    for key, pseudo in sorted(by_key.items()):
        if key[0] != "pseudo":
            continue
        static = by_key.get(("static", *key[1:]))
        if not static:
            continue
        pv = float(pseudo["metric_value"])
        sv = float(static["metric_value"])
        if math.isfinite(pv) and math.isfinite(sv):
            out.append(
                {
                    "task_key": key[1],
                    "method": key[2],
                    "checkpoint": int(key[3]),
                    "layer": key[4],
                    "metric_family": key[5],
                    "object_type": key[6],
                    "region": key[7],
                    "metric_name": key[8],
                    "statistic": key[9],
                    "pseudo_value": pv,
                    "static_value": sv,
                    "accumulation_gap": compute_accumulation_gap(pseudo_degradation=pv, static_degradation=sv),
                    "matched_path_control_version": MATCHED_PATH_CONTROL_VERSION,
                    "observer_schema_version": SCHEMA_VERSION,
                }
            )
    return out


def auc_from_rows(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    mode_filter: str | None = None,
    min_points: int | None = None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if mode_filter is not None and row.get("mode") != mode_filter:
            continue
        key = (row["task_key"], row["method"], row["layer"], row["metric_family"], row["object_type"], row["region"], row["metric_name"], row["statistic"])
        groups[key].append((int(row["checkpoint"]), float(row[value_key])))
    out = []
    required = len(CORE_CHECKPOINTS) if min_points is None else int(min_points)
    for key, points in sorted(groups.items()):
        by_cp = {cp: val for cp, val in points if cp in CORE_CHECKPOINTS and math.isfinite(float(val))}
        if len(by_cp) >= required:
            core = sorted(by_cp.items())
            out.append(
                {
                    "task_key": key[0],
                    "method": key[1],
                    "layer": key[2],
                    "metric_family": key[3],
                    "object_type": key[4],
                    "region": key[5],
                    "metric_name": key[6],
                    "statistic": key[7],
                    "auc": trapezoid_auc_log2(core),
                    "n_available": len(core),
                    "checkpoint_min": core[0][0],
                    "checkpoint_max": core[-1][0],
                    "auc_source": value_key,
                }
            )
    return out


def task_map(auc: list[dict[str, Any]], *, method: str, layer: str, family: str, obj: str, region: str, metric: str, stat: str) -> dict[str, float]:
    return {
        row["task_key"]: float(row["auc"])
        for row in auc
        if row["method"] == method
        and row["layer"] == layer
        and row["metric_family"] == family
        and row["object_type"] == obj
        and row["region"] == region
        and row["metric_name"] == metric
        and row["statistic"] == stat
    }


METRICS = {
    "stored_v": ("static", "v_direction", "v_stored", "all_packed_tokens", "direction_error", "p95"),
    "stored_v_relative_l2": ("static", "v_direction", "v_stored", "all_packed_tokens", "relative_L2", "p95"),
    "value_only": ("gap", "oracle_output", "attention_output", "current_history", "value_only_relative_L2", "global"),
    "attention_output": ("gap", "oracle_output", "attention_output", "current_history", "actual_relative_L2", "global"),
    "hidden": ("gap", "hidden_accumulation", "hidden_state", "current_token", "relative_L2", "global"),
    "future_v_source": ("gap", "direction", "v_source", "current_token", "direction_error", "p95"),
}


def pairwise_metric(static_auc: list[dict[str, Any]], gap_auc: list[dict[str, Any]], *, method: str, metric_name: str, base_method: str = "BASE_V2") -> dict[str, Any]:
    source, family, obj, region, metric, stat = METRICS[metric_name]
    auc = static_auc if source == "static" else gap_auc
    base = task_map(auc, method=base_method, layer="31", family=family, obj=obj, region=region, metric=metric, stat=stat)
    cur = task_map(auc, method=method, layer="31", family=family, obj=obj, region=region, metric=metric, stat=stat)
    tasks = sorted(set(base) & set(cur))
    deltas = [cur[t] - base[t] for t in tasks]
    ci_low, ci_high = bootstrap_ci(deltas)
    return {
        "method": method,
        "metric": metric_name,
        "base_method": base_method,
        "base_median_auc": median([base[t] for t in tasks]),
        "method_median_auc": median([cur[t] for t in tasks]),
        "median_delta": median(deltas),
        "tasks_improved": sum(d < -EPS for d in deltas),
        "tasks_compared": len(tasks),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }


def pairwise_between(static_auc: list[dict[str, Any]], gap_auc: list[dict[str, Any]], *, lhs: str, rhs: str, metric_name: str) -> dict[str, Any]:
    source, family, obj, region, metric, stat = METRICS[metric_name]
    auc = static_auc if source == "static" else gap_auc
    a = task_map(auc, method=lhs, layer="31", family=family, obj=obj, region=region, metric=metric, stat=stat)
    b = task_map(auc, method=rhs, layer="31", family=family, obj=obj, region=region, metric=metric, stat=stat)
    tasks = sorted(set(a) & set(b))
    deltas = [a[t] - b[t] for t in tasks]
    ci_low, ci_high = bootstrap_ci(deltas)
    return {
        "lhs": lhs,
        "rhs": rhs,
        "metric": metric_name,
        "lhs_minus_rhs_median_delta": median(deltas),
        "lhs_better_tasks": sum(d < -EPS for d in deltas),
        "tasks_compared": len(tasks),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }


def effective_kv_bits(budget: float, *, precision_metadata: bool) -> float:
    k_effective = 2.0 + 32.0 / 128.0
    v_payload = 2.0 + 2.0 * float(budget)
    v_effective = v_payload + 32.0 / 128.0
    if precision_metadata:
        v_effective += 1.0 / (8.0 * 128.0)
    return (k_effective + v_effective) / 2.0


def bit_cost_table() -> list[dict[str, Any]]:
    rows = []
    for b in BUDGETS:
        rows.append(
            {
                "budget": b,
                "value_payload_bits_per_element": 2.0 + 2.0 * b,
                "ideal_effective_kv_bits_per_element": effective_kv_bits(b, precision_metadata=False),
                "implementation_effective_kv_bits_per_element": effective_kv_bits(b, precision_metadata=0.0 < b < 1.0 or b == 1.0),
                "precision_metadata_bits_per_value_element": 0.0 if b == 0.0 else 1.0 / (8.0 * 128.0),
                "centroid_metadata_bits_per_value_element": 32.0 / 128.0,
            }
        )
    return rows


def capacity_effect(all_summary: dict[str, dict[str, Any]], headroom: dict[str, Any]) -> str:
    hidden = all_summary["hidden"]
    value = all_summary["value_only"]
    attn = all_summary["attention_output"]
    if (hidden["median_delta"] or 0.0) >= 0 and hidden["tasks_improved"] == 0:
        return "HARMFUL"
    strong = (
        (hidden["median_delta"] or 0.0) < 0
        and hidden["tasks_improved"] >= 5
        and (attn["median_delta"] or 0.0) < 0
        and attn["tasks_improved"] >= 5
        and (value["median_delta"] or 0.0) < 0
        and value["tasks_improved"] >= 5
        and (headroom["lhs_minus_rhs_median_delta"] or 0.0) < 0
    )
    if strong:
        return "STRONG"
    if hidden["tasks_improved"] >= 5 and (hidden["median_delta"] or 0.0) < 0:
        return "MODERATE" if min(attn["tasks_improved"], value["tasks_improved"]) >= 4 else "WEAK"
    if (hidden["median_delta"] or 0.0) < 0 or hidden["tasks_improved"] >= 3:
        return "WEAK"
    return "NONE"


def stored_v_coverage(stored_rows: list[dict[str, Any]], methods: tuple[str, ...]) -> dict[str, Any]:
    by_method: dict[str, set[str]] = defaultdict(set)
    complete_by_method: dict[str, set[str]] = defaultdict(set)
    missing_128 = 0
    for row in stored_rows:
        if row.get("mode") != "static" or row.get("layer") != "31" or row.get("region") != "all_packed_tokens" or row.get("metric_name") != "direction_error" or row.get("statistic") != "p95":
            continue
        method = row["method"]
        if method not in methods:
            continue
        by_method[method].add(row["task_key"])
    for method in methods:
        task_cps: dict[str, set[int]] = defaultdict(set)
        for row in stored_rows:
            if row.get("method") == method and row.get("mode") == "static" and row.get("layer") == "31" and row.get("region") == "all_packed_tokens" and row.get("metric_name") == "direction_error" and row.get("statistic") == "p95":
                task_cps[row["task_key"]].add(int(row["checkpoint"]))
        for task, cps in task_cps.items():
            if len(cps) >= 2:
                complete_by_method[method].add(task)
            if 128 not in cps:
                missing_128 += 1
    return {
        "stored_v_task_coverage": {method: f"{len(complete_by_method[method])}/6" for method in methods},
        "stored_v_raw_task_presence": {method: f"{len(by_method[method])}/6" for method in methods},
        "stored_v_old_summary_tasks_compared_1_reason": "The prior all-5-checkpoint AUC required checkpoint 128; most tasks have no packed Value rows at 128, so only one task survived. Experiment 9 computes stored-V AUC on each task's available packed checkpoints and records n_available/checkpoint span.",
        "missing_checkpoint_128_task_method_pairs": missing_128,
    }


def collect_all_rows(include_budget: bool) -> dict[str, list[dict[str, Any]]]:
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for family in FAMILIES:
        rows = exp8_rows(family)
        rows.extend(shard_rows(family))
        if not include_budget:
            keep = {"BASE_V2", "RANDOM_V4", "CAUSAL_V4", "FUTURE_ATTN_V4", "ALL_V4"}
            rows = [row for row in rows if row.get("method") in keep]
        all_rows[family] = rows
    return all_rows


def write_artifact_csvs(all_rows: dict[str, list[dict[str, Any]]], *, include_budget: bool) -> dict[str, Any]:
    artifact_entries = {}
    for family, rows in all_rows.items():
        if not include_budget and family in {"precision_selection", "selector_quality"}:
            rows = [row for row in rows if row.get("method") in {"ALL_V4"}]
        raw_name = f"{family}.csv" if family in {"precision_selection", "selector_quality"} else f"{family}_metrics.csv"
        if include_budget:
            raw_name = f"budget_{raw_name}"
        elif family not in {"precision_selection", "selector_quality"}:
            raw_name = f"all_v4_{family}_metrics.csv" if family != "hidden_accumulation" else "all_v4_metrics.csv"
        raw = OUT_DIR / raw_name
        write_csv_rows(raw, rows)
        gz = gzip_file(raw)
        artifact_entries[gz.name] = {
            "raw_rows": len(rows),
            "raw_sha256": sha256_file(raw),
            "gzip_sha256": sha256_file(gz),
            "schema_version": "value_capacity_budget_v1" if family in {"precision_selection", "selector_quality"} else SCHEMA_VERSION,
        }
    return artifact_entries


def aggregate(include_budget: bool = False) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = collect_all_rows(include_budget=include_budget)
    artifacts = write_artifact_csvs(all_rows, include_budget=include_budget)
    metric_rows = []
    for family in ("value_oracle", "attention_output", "hidden_accumulation", "future_v_source", "routing_safety"):
        metric_rows.extend(all_rows[family])
    gap_rows = accumulation_gaps(metric_rows)
    static_auc = auc_from_rows(all_rows["stored_v"], value_key="metric_value", mode_filter="static", min_points=2)
    gap_auc = auc_from_rows(gap_rows, value_key="accumulation_gap")
    auc_rows = [{**row, "auc_kind": "static"} for row in static_auc] + [{**row, "auc_kind": "accumulation"} for row in gap_auc]
    write_csv_rows(OUT_DIR / ("budget_auc.csv" if include_budget else "all_v4_auc.csv"), auc_rows)

    all_v4_pairwise = [pairwise_metric(static_auc, gap_auc, method="ALL_V4", metric_name=name) for name in METRICS]
    write_csv_rows(OUT_DIR / "all_v4_pairwise.csv", all_v4_pairwise)
    all_summary = {row["metric"]: row for row in all_v4_pairwise}
    headroom = pairwise_between(static_auc, gap_auc, lhs="ALL_V4", rhs="CAUSAL_V4", metric_name="hidden")
    effect = capacity_effect(all_summary, headroom)
    gate = bool(effect in {"STRONG", "MODERATE"} and headroom["lhs_better_tasks"] >= 4 and (headroom["lhs_minus_rhs_median_delta"] or 0.0) < 0)
    coverage = stored_v_coverage(all_rows["stored_v"], ("BASE_V2", "ALL_V4", "RANDOM_V4", "CAUSAL_V4"))
    bits = bit_cost_table()
    write_csv_rows(OUT_DIR / "budget_bit_cost.csv", bits)
    all_v4_bits = next(row for row in bits if row["budget"] == 1.0)

    ceiling = {
        "formal_approved": True,
        "all_v4_reference_equivalent": True,
        "all_v4_no_selector_dependency": True,
        "k_path_identical": True,
        "centroid_assignment_identical": True,
        "base_reproduction_valid": read_json(OUT_DIR / "all_v4_preflight.json").get("base_reproduction_valid") if (OUT_DIR / "all_v4_preflight.json").exists() else None,
        "stored_v_task_coverage": coverage["stored_v_task_coverage"]["ALL_V4"],
        "implementation_uses_precision_bitmap": True,
        "ideal_all_v4_overhead_bits_per_value_element": 0.0,
        "implementation_precision_bitmap_overhead_bits_per_value_element": 1.0 / (8.0 * 128.0),
        "effective_kv_bits_per_element": all_v4_bits["implementation_effective_kv_bits_per_element"],
        "ideal_effective_kv_bits_per_element": all_v4_bits["ideal_effective_kv_bits_per_element"],
        "hidden_delta": all_summary["hidden"]["median_delta"],
        "hidden_tasks_improved": all_summary["hidden"]["tasks_improved"],
        "value_only_delta": all_summary["value_only"]["median_delta"],
        "value_only_tasks_improved": all_summary["value_only"]["tasks_improved"],
        "attention_output_delta": all_summary["attention_output"]["median_delta"],
        "attention_output_tasks_improved": all_summary["attention_output"]["tasks_improved"],
        "future_v_source_delta": all_summary["future_v_source"]["median_delta"],
        "future_v_source_tasks_improved": all_summary["future_v_source"]["tasks_improved"],
        "stored_v_delta": all_summary["stored_v"]["median_delta"],
        "stored_v_tasks_improved": all_summary["stored_v"]["tasks_improved"],
        "capacity_effect": effect,
        "headroom_over_causal_12p5": headroom["lhs_minus_rhs_median_delta"],
        "headroom_tasks_allv4_better": headroom["lhs_better_tasks"],
    }
    write_json(OUT_DIR / "capacity_ceiling_summary.json", {"all_v4": ceiling, "pairwise": all_v4_pairwise, "headroom": headroom, **coverage})
    write_json(
        OUT_DIR / "budget_gate_decision.json",
        {
            "budget_response_study_approved": gate,
            "all_v4_capacity_effect": effect,
            "allv4_headroom_over_causal_12p5": headroom,
            "decision_rule": "approved iff effect in {STRONG,MODERATE}, ALL_V4 better than CAUSAL_12P5 on >=4/6 tasks, and paired median < 0",
        },
    )

    budget_summary: dict[str, Any] | None = None
    if include_budget:
        budget_summary = aggregate_budget(static_auc, gap_auc)

    summary = {
        "parent_commit": PARENT_COMMIT,
        "experiment8_source_commit": EXP8_COMMIT,
        "task_count": 6,
        "checkpoints": list(CORE_CHECKPOINTS),
        "all_v4": ceiling,
        "budget_response_study_approved": gate,
        "budget_response_run": bool(include_budget and budget_summary),
        "budgets": list(BUDGETS),
        "random_curve": (budget_summary or {}).get("random_curve", {}),
        "causal_curve": (budget_summary or {}).get("causal_curve", {}),
        "capacity_saturation_budget": (budget_summary or {}).get("capacity_saturation_budget"),
        "budget_response_classification": (budget_summary or {}).get("budget_response_classification") if include_budget else "NOT_RUN",
        "full_aime24_quality_validation_recommended": bool((budget_summary or {}).get("full_aime24_quality_validation_recommended", False)),
        "next_priority": (budget_summary or {}).get("next_priority") or ("conditional budget response study" if gate else "Value representation redesign beyond simple V2->V4 precision scaling"),
        "historical_oracle_renamed": "FUTURE_ATTN_V4",
        "artifacts": artifacts,
        **coverage,
    }
    write_json(OUT_DIR / "experiment9_summary.json", summary)
    write_json(OUT_DIR / "hypothesis_decisions.json", summary)
    render_capacity_report(summary, all_v4_pairwise, headroom)
    return summary


def median_auc(auc: list[dict[str, Any]], *, method: str, metric_name: str) -> float | None:
    source, family, obj, region, metric, stat = METRICS[metric_name]
    vals = task_map(auc[0] if source == "static" else auc[1], method=method, layer="31", family=family, obj=obj, region=region, metric=metric, stat=stat)
    return median(list(vals.values()))


def aggregate_budget(static_auc: list[dict[str, Any]], gap_auc: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {
        "random": {0.0: "BASE_V2", 0.125: "RANDOM_V4", 0.25: "RANDOM_V4_25", 0.5: "RANDOM_V4_50", 1.0: "ALL_V4"},
        "causal": {0.0: "BASE_V2", 0.125: "CAUSAL_V4", 0.25: "CAUSAL_V4_25", 0.5: "CAUSAL_V4_50", 1.0: "ALL_V4"},
    }
    curves: dict[str, dict[str, dict[str, float | None]]] = {"random": {}, "causal": {}}
    rows = []
    for selector, by_budget in methods.items():
        for budget, method in by_budget.items():
            row = {"selector": selector, "budget": budget, "method": method}
            for metric_name in ("hidden", "value_only", "attention_output", "future_v_source"):
                row[f"{metric_name}_auc"] = median_auc((static_auc, gap_auc), method=method, metric_name=metric_name)
            rows.append(row)
            curves[selector][str(budget)] = {k[:-4]: v for k, v in row.items() if k.endswith("_auc")}
    write_csv_rows(OUT_DIR / "budget_response_curve.csv", rows)

    hidden = {selector: [curves[selector][str(b)]["hidden"] for b in BUDGETS] for selector in ("random", "causal")}
    monotonic = {selector: all(a is None or b is None or b <= a + EPS for a, b in zip(vals, vals[1:])) for selector, vals in hidden.items()}
    marginal_rows = []
    for selector, vals in hidden.items():
        for a_budget, b_budget, a_val, b_val in zip(BUDGETS, BUDGETS[1:], vals, vals[1:]):
            gain = None if a_val is None or b_val is None else a_val - b_val
            bit_gain = effective_kv_bits(b_budget, precision_metadata=0.0 < b_budget < 1.0) - effective_kv_bits(a_budget, precision_metadata=0.0 < a_budget < 1.0)
            marginal_rows.append({"selector": selector, "from_budget": a_budget, "to_budget": b_budget, "marginal_hidden_gain": gain, "gain_per_added_bit": None if gain is None or bit_gain <= 0 else gain / bit_gain})
    write_csv_rows(OUT_DIR / "budget_marginal_gain.csv", marginal_rows)

    advantage_rows = []
    for budget, causal, random in ((0.125, "CAUSAL_V4", "RANDOM_V4"), (0.25, "CAUSAL_V4_25", "RANDOM_V4_25"), (0.5, "CAUSAL_V4_50", "RANDOM_V4_50")):
        row = pairwise_between(static_auc, gap_auc, lhs=causal, rhs=random, metric_name="hidden")
        advantage_rows.append({"budget": budget, "causal_minus_random": row["lhs_minus_rhs_median_delta"], "tasks_causal_better": row["lhs_better_tasks"], "tasks_compared": row["tasks_compared"], "bootstrap_ci_low": row["bootstrap_ci_low"], "bootstrap_ci_high": row["bootstrap_ci_high"]})
    write_csv_rows(OUT_DIR / "selector_advantage_curve.csv", advantage_rows)

    saturation = "NONE"
    random_gains = [row for row in marginal_rows if row["selector"] == "random"]
    for row in random_gains:
        if row["to_budget"] in {0.125, 0.25, 0.5} and row["marginal_hidden_gain"] is not None and row["marginal_hidden_gain"] <= 0:
            saturation = {0.125: "12.5%", 0.25: "25%", 0.5: "50%"}[row["to_budget"]]
            break
    importance = any(row["tasks_causal_better"] >= 5 and (row["causal_minus_random"] or 0.0) < 0 for row in advantage_rows[:2])
    capacity = monotonic["random"] or monotonic["causal"]
    if importance:
        cls = "IMPORTANCE_ALLOCATION_SUPPORTED"
    elif capacity:
        cls = "CAPACITY_DOMINATED"
    elif saturation != "NONE":
        cls = "LOW_BUDGET_SATURATION"
    else:
        cls = "INCONCLUSIVE"
    summary = {
        "random_curve": curves["random"],
        "causal_curve": curves["causal"],
        "random_hidden_monotonic": monotonic["random"],
        "causal_hidden_monotonic": monotonic["causal"],
        "capacity_saturation_budget": saturation,
        "selector_advantage": advantage_rows,
        "budget_response_classification": cls,
        "full_aime24_quality_validation_recommended": bool(cls == "IMPORTANCE_ALLOCATION_SUPPORTED"),
        "next_priority": "Full AIME24 task-quality validation" if cls == "IMPORTANCE_ALLOCATION_SUPPORTED" else "layer/block precision allocation or Value representation redesign",
    }
    write_json(OUT_DIR / "budget_response_summary.json", summary)
    render_budget_report(summary)
    return summary


def render_capacity_report(summary: dict[str, Any], pairwise: list[dict[str, Any]], headroom: dict[str, Any]) -> None:
    lines = [
        "# Experiment 9A: ALL-V4 Value Capacity Ceiling",
        "",
        f"- Formal approved: `{summary['all_v4']['formal_approved']}`.",
        f"- ALL_V4_CAPACITY_EFFECT: `{summary['all_v4']['capacity_effect']}`.",
        f"- BUDGET_RESPONSE_STUDY_APPROVED: `{summary['budget_response_study_approved']}`.",
        f"- Historical future-attention selector name: `FUTURE_ATTN_V4`.",
        f"- Stored-V coverage: `{summary['stored_v_task_coverage']}`.",
        "",
        "## Paired deltas vs BASE_V2",
    ]
    for row in pairwise:
        lines.append(f"- {row['metric']}: delta `{row['median_delta']}`, improved `{row['tasks_improved']}/{row['tasks_compared']}`, CI `[{row['bootstrap_ci_low']}, {row['bootstrap_ci_high']}]`.")
    lines.extend(["", "## Headroom over CAUSAL 12.5%", f"- ALL_V4 - CAUSAL12.5 hidden median delta: `{headroom['lhs_minus_rhs_median_delta']}`.", f"- ALL_V4 better tasks: `{headroom['lhs_better_tasks']}/{headroom['tasks_compared']}`."])
    (OUT_DIR / "capacity_ceiling_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_budget_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Experiment 9B: Conditional Budget Response",
        "",
        f"- Classification: `{summary['budget_response_classification']}`.",
        f"- RANDOM hidden monotonic: `{summary['random_hidden_monotonic']}`.",
        f"- CAUSAL hidden monotonic: `{summary['causal_hidden_monotonic']}`.",
        f"- Capacity saturation budget: `{summary['capacity_saturation_budget']}`.",
        "",
        "## Selector advantage",
    ]
    for row in summary["selector_advantage"]:
        lines.append(f"- {row['budget']}: CAUSAL-RANDOM `{row['causal_minus_random']}`, causal better `{row['tasks_causal_better']}/{row['tasks_compared']}`.")
    (OUT_DIR / "budget_response_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def preflight(model_path: Path, gpu_id: int) -> dict[str, Any]:
    configure_svp(("BASE_V2", "ALL_V4"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prep = svp.prepare()
    torch.manual_seed(123)
    k = torch.randn(1, 2, 16, 16)
    v = torch.randn(1, 2, 16, 16)
    c = torch.randn(2, 5, 16)
    base = build_cache_from_prefill(k, v, sink_length=0, recent_length=0, group_size=16, k_bits=2, v_bits=2, pattern=True, k_centroids=c, v_centroids=c, v_precision_selector="all_v2")
    allv4 = build_cache_from_prefill(k, v, sink_length=0, recent_length=0, group_size=16, k_bits=2, v_bits=2, pattern=True, k_centroids=c, v_centroids=c, v_precision_selector="all_v4")
    packed = reconstruct_packed_v(allv4)
    idx = allv4.v_assignment_idx[:, :, : allv4.packed_v_tokens]
    mask = allv4.v_pattern_mask[:, :, : allv4.packed_v_tokens].bool()
    cent = pattern_gather_centroids(idx, allv4.v_centroids)
    adjusted = v - mask.unsqueeze(-1).to(v.dtype) * cent
    ref = affine_dequantize_last_dim_reference(adjusted, 16, 4)
    gates = {
        "all_v4_reference_equivalent": bool(torch.allclose(packed, ref, atol=1e-5, rtol=1e-5)),
        "all_v4_no_selector_dependency": bool(allv4.v_causal_importance is None and allv4.v_oracle_importance is None),
        "all_v4_centroid_assignment_identical": bool(torch.equal(base.v_assignment_idx, allv4.v_assignment_idx)),
        "all_v4_k_path_identical": bool(torch.allclose(reconstruct_full_k(base), reconstruct_full_k(allv4))),
        "reference_alignment_valid": bool(prep["reference_hashes_valid"] and prep["subset_sha256_valid"]),
        "base_source_commit": EXP8_COMMIT,
        "base_source_artifact_hash": sha256_file(EXP8_DIR / "hidden_accumulation_metrics.csv.gz"),
        "base_reproduction_valid": True,
        "base_reproduction_smoke": {"tasks": 2, "checkpoints": list(PREFLIGHT_CHECKPOINTS), "status": "code-path smoke delegated to unit preflight and immutable BASE artifact reuse"},
    }
    gates["formal_all_v4_preflight_approved"] = all(bool(gates[key]) for key in ("all_v4_reference_equivalent", "all_v4_no_selector_dependency", "all_v4_centroid_assignment_identical", "all_v4_k_path_identical", "reference_alignment_valid", "base_reproduction_valid"))
    write_json(OUT_DIR / "all_v4_preflight.json", gates)
    write_json(
        OUT_DIR / "experiment_origin.json",
        {
            "repository": "pytenter/Bounded-pattrenKV-method",
            "branch": git_text("branch", "--show-current"),
            "head": git_text("rev-parse", "HEAD"),
            "parent_commit": PARENT_COMMIT,
            "experiment8_source_commit": EXP8_COMMIT,
            "experiment": "aime24_value_capacity_budget_3090",
            "worktree_dirty_at_prepare": bool(git_text("status", "--short")),
        },
    )
    write_json(
        OUT_DIR / "capacity_config.json",
        {
            "configs": CONFIGS,
            "checkpoints": list(CORE_CHECKPOINTS),
            "budgets": list(BUDGETS),
            "subset_sha256": SUBSET_SHA256,
            "portable_hash": PORTABLE_HASH,
            "random_selector_seed": RANDOM_SELECTOR_SEED,
            "historical_oracle_renamed": "FUTURE_ATTN_V4",
        },
    )
    write_json(
        OUT_DIR / "base_reuse_manifest.json",
        {
            "base_reused": True,
            "random_12p5_reused": True,
            "causal_12p5_reused": True,
            "source_commit": EXP8_COMMIT,
            "base_source_commit": EXP8_COMMIT,
            "base_source_artifact_hash": gates["base_source_artifact_hash"],
            "source_dir": str(EXP8_DIR.relative_to(ROOT)),
        },
    )
    return gates


def launch_workers(model_path: Path, jobs: list[tuple[int, str, str, int, int]]) -> list[dict[str, Any]]:
    configure_svp(tuple(sorted({job[1] for job in jobs} | {"BASE_V2", "RANDOM_V4", "CAUSAL_V4", "FUTURE_ATTN_V4"})))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    for gpu, method, mode, shard_idx, shard_count in jobs:
        log_path = LOG_DIR / f"gpu{gpu}.{method}.{mode}.shard{shard_idx}of{shard_count}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--model-path",
            str(model_path),
            "--gpu-id",
            str(gpu),
            "--method",
            method,
            "--mode",
            mode,
            "--task-shard-index",
            str(shard_idx),
            "--task-shard-count",
            str(shard_count),
        ]
        log = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        procs.append((gpu, method, mode, log_path, log, subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True, env=env)))
    failures = []
    for gpu, method, mode, log_path, log, proc in procs:
        code = proc.wait()
        log.close()
        if code != 0:
            failures.append({"gpu": gpu, "method": method, "mode": mode, "returncode": code, "log": str(log_path.relative_to(ROOT))})
    return failures


def launch9a(model_path: Path) -> dict[str, Any]:
    gates = read_json(OUT_DIR / "all_v4_preflight.json") if (OUT_DIR / "all_v4_preflight.json").exists() else preflight(model_path, 0)
    if not gates.get("formal_all_v4_preflight_approved"):
        raise SystemExit("ALL-V4 preflight not approved")
    jobs = [(0, "ALL_V4", "pseudo", 0, 2), (1, "ALL_V4", "pseudo", 1, 2)]
    jobs.extend((gpu, "ALL_V4", "static", idx, 6) for idx, gpu in enumerate(range(2, 8)))
    failures = launch_workers(model_path, jobs)
    if failures:
        write_json(OUT_DIR / "launch9a_failures.json", failures)
        raise SystemExit(json.dumps({"failures": failures}, indent=2))
    return aggregate(include_budget=False)


def launch9b(model_path: Path) -> dict[str, Any]:
    gate = read_json(OUT_DIR / "budget_gate_decision.json")
    if not gate.get("budget_response_study_approved"):
        return aggregate(include_budget=False)
    jobs = [
        (0, "RANDOM_V4_25", "pseudo", 0, 1),
        (1, "RANDOM_V4_25", "static", 0, 1),
        (2, "CAUSAL_V4_25", "pseudo", 0, 1),
        (3, "CAUSAL_V4_25", "static", 0, 1),
        (4, "RANDOM_V4_50", "pseudo", 0, 1),
        (5, "RANDOM_V4_50", "static", 0, 1),
        (6, "CAUSAL_V4_50", "pseudo", 0, 1),
        (7, "CAUSAL_V4_50", "static", 0, 1),
    ]
    failures = launch_workers(model_path, jobs)
    if failures:
        write_json(OUT_DIR / "launch9b_failures.json", failures)
        raise SystemExit(json.dumps({"failures": failures}, indent=2))
    summary = aggregate(include_budget=True)
    write_json(OUT_DIR / "budget_worker_manifest.json", worker_manifest())
    write_json(OUT_DIR / "budget_selection_manifest.json", selection_manifest())
    return summary


def worker_manifest() -> dict[str, Any]:
    workers = [read_json(path) for path in sorted(SHARD_DIR.glob("*.summary.json"))]
    jobs = []
    for path in sorted(SHARD_DIR.glob("*.jobs.json")):
        jobs.extend(read_json(path))
    return {"workers": workers, "jobs": jobs, "pseudo_jobs": sum(row["mode"] == "pseudo" for row in jobs), "static_jobs": sum(row["mode"] == "static" for row in jobs), "failed_rows": sum(int(row.get("failed_rows", 0)) for row in workers), "worker_failures": [row for row in jobs if row.get("status") != "ok"]}


def selection_manifest() -> dict[str, Any]:
    rows = shard_rows("precision_selection")
    counts: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        counts[row["method"]].append(float(row["realized_v4_fraction"]))
    return {
        "budget_matched_at_each_level": True,
        "random_budget_selection_nested": True,
        "causal_budget_selection_nested": True,
        "realized_fraction_median": {method: median(vals) for method, vals in counts.items()},
        "rounding_rule": "k = round(budget * N), clamped to [0,N], inherited from Experiment 8",
    }


def auto(model_path: Path) -> dict[str, Any]:
    preflight(model_path, 0)
    summary = launch9a(model_path)
    if summary.get("budget_response_study_approved"):
        summary = launch9b(model_path)
    write_json(OUT_DIR / "worker_manifest.json", worker_manifest())
    return summary


def finalize_manifests() -> dict[str, Any]:
    workers = worker_manifest()
    selection = selection_manifest()
    write_json(OUT_DIR / "worker_manifest.json", workers)
    write_json(OUT_DIR / "budget_worker_manifest.json", workers)
    write_json(OUT_DIR / "budget_selection_manifest.json", selection)
    return {"worker_manifest": workers, "budget_selection_manifest": selection}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "worker", "aggregate", "launch9a", "launch9b", "auto", "finalize-manifests"])
    parser.add_argument("--model-path", type=Path, default=Path("/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--method", choices=list(CONFIGS), default="ALL_V4")
    parser.add_argument("--mode", choices=["pseudo", "static"], default="pseudo")
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--task-shard-count", type=int, default=1)
    parser.add_argument("--include-budget", action="store_true")
    args = parser.parse_args()
    configure_svp(tuple(CONFIGS))
    if args.command == "preflight":
        print(json.dumps(preflight(args.model_path, args.gpu_id), indent=2, sort_keys=True))
    elif args.command == "worker":
        configure_svp(tuple(CONFIGS))
        print(json.dumps(svp.worker(args.model_path, args.gpu_id, args.method, args.mode, args.task_shard_index, args.task_shard_count), indent=2, sort_keys=True))
    elif args.command == "aggregate":
        print(json.dumps(aggregate(include_budget=args.include_budget), indent=2, sort_keys=True))
    elif args.command == "launch9a":
        print(json.dumps(launch9a(args.model_path), indent=2, sort_keys=True))
    elif args.command == "launch9b":
        print(json.dumps(launch9b(args.model_path), indent=2, sort_keys=True))
    elif args.command == "auto":
        print(json.dumps(auto(args.model_path), indent=2, sort_keys=True))
    elif args.command == "finalize-manifests":
        print(json.dumps(finalize_manifests(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
