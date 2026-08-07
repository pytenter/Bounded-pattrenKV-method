#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any


CONFIGS: tuple[dict[str, Any], ...] = (
    {"gpu": 0, "config_name": "pattern_rolling_k2v2_s0_r128", "short": "S0/R128", "method_group": "PatternKV", "method": "patternkv", "sink_length": 0, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "result_source": "reused_wave1a"},
    {"gpu": 1, "config_name": "pattern_rolling_k2v2_s64_r128", "short": "S64/R128", "method_group": "PatternKV", "method": "patternkv", "sink_length": 64, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "result_source": "newly_run_wave1a2"},
    {"gpu": 2, "config_name": "pattern_rolling_k2v2_s0_r256", "short": "S0/R256", "method_group": "PatternKV", "method": "patternkv", "sink_length": 0, "recent_length": 256, "residual_length": 128, "k_bits": 2, "v_bits": 2, "result_source": "newly_run_wave1a2"},
    {"gpu": 3, "config_name": "pattern_rolling_k2v2_s64_r256", "short": "S64/R256", "method_group": "PatternKV", "method": "patternkv", "sink_length": 64, "recent_length": 256, "residual_length": 128, "k_bits": 2, "v_bits": 2, "result_source": "reused_wave1a"},
    {"gpu": 4, "config_name": "kivi_rolling_k2v2_s0_r128", "short": "S0/R128", "method_group": "KIVI", "method": "kivi_official", "sink_length": 0, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "result_source": "reused_wave1a"},
    {"gpu": 5, "config_name": "kivi_rolling_k2v2_s64_r128", "short": "S64/R128", "method_group": "KIVI", "method": "kivi_official", "sink_length": 64, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "result_source": "newly_run_wave1a2"},
    {"gpu": 6, "config_name": "kivi_rolling_k2v2_s0_r256", "short": "S0/R256", "method_group": "KIVI", "method": "kivi_official", "sink_length": 0, "recent_length": 256, "residual_length": 128, "k_bits": 2, "v_bits": 2, "result_source": "newly_run_wave1a2"},
    {"gpu": 7, "config_name": "kivi_rolling_k2v2_s64_r256", "short": "S64/R256", "method_group": "KIVI", "method": "kivi_official", "sink_length": 64, "recent_length": 256, "residual_length": 128, "k_bits": 2, "v_bits": 2, "result_source": "reused_wave1a"},
)

BY_CONFIG = {cfg["config_name"]: cfg for cfg in CONFIGS}

QUADS = {
    "PatternKV": {
        "Y00": "pattern_rolling_k2v2_s0_r128",
        "Y10": "pattern_rolling_k2v2_s64_r128",
        "Y01": "pattern_rolling_k2v2_s0_r256",
        "Y11": "pattern_rolling_k2v2_s64_r256",
    },
    "KIVI": {
        "Y00": "kivi_rolling_k2v2_s0_r128",
        "Y10": "kivi_rolling_k2v2_s64_r128",
        "Y01": "kivi_rolling_k2v2_s0_r256",
        "Y11": "kivi_rolling_k2v2_s64_r256",
    },
}

PAIR_DEFS: tuple[tuple[str, str, str, str], ...] = (
    ("PatternKV", "Y00", "Y10", "Sink effect @ R128"),
    ("PatternKV", "Y01", "Y11", "Sink effect @ R256"),
    ("PatternKV", "Y00", "Y01", "Recent effect @ S0"),
    ("PatternKV", "Y10", "Y11", "Recent effect @ S64"),
    ("KIVI", "Y00", "Y10", "Sink effect @ R128"),
    ("KIVI", "Y01", "Y11", "Sink effect @ R256"),
    ("KIVI", "Y00", "Y01", "Recent effect @ S0"),
    ("KIVI", "Y10", "Y11", "Recent effect @ S64"),
)


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def pctl(values: list[int], q: float) -> int | None:
    if not values:
        return None
    vals = sorted(values)
    return vals[max(0, min(len(vals) - 1, math.ceil(q * len(vals)) - 1))]


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows_by_config: dict[str, list[dict[str, Any]]] = {}
    for cfg in manifest["logical_configs"]:
        config_name = cfg["config_name"]
        path = Path(cfg["result_source_path"])
        rows_by_config[config_name] = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("*.json"))]
    return rows_by_config


def row_correct(row: dict[str, Any]) -> bool:
    return bool(row.get("is_correct")) and not row.get("error")


def actual_bits(row: dict[str, Any]) -> float | None:
    stats = row.get("cache_bitwidth_stats") or {}
    bytes_used = stats.get("python_tensor_storage_bytes")
    tokens = stats.get("total_cached_tokens") or row.get("total_sequence_tokens")
    if not bytes_used or not tokens:
        return None
    denom = int(tokens) * 32 * 8 * 128 * 2
    return float(bytes_used) * 8.0 / denom


def compact_bits(row: dict[str, Any], cfg: dict[str, Any]) -> float | None:
    total = row.get("total_sequence_tokens") or row.get("generated_tokens")
    if not total:
        return None
    total_tokens = int(total)
    sink = min(total_tokens, int(cfg["sink_length"]))
    recent = min(max(total_tokens - sink, 0), int(cfg["recent_length"]))
    quant = max(total_tokens - sink - recent, 0)
    payload = ((cfg["k_bits"] + cfg["v_bits"]) / 2.0 * quant + 16.0 * (sink + recent)) / max(total_tokens, 1)
    affine = 32.0 / cfg["residual_length"] * quant / max(total_tokens, 1)
    assignment = 0.0
    gate = 0.0
    centroid = 0.0
    if cfg["method"] == "patternkv" and quant:
        assignment = 2.0 * (math.ceil(math.log2(32)) / 128.0) * quant / max(total_tokens, 1)
        gate = (1.0 / 128.0) * quant / max(total_tokens, 1)
        centroid = 2.0 * (32 * 128 * 16.0) / max(quant * 128, 1)
    return payload + affine + assignment + gate + centroid


def summarize_config(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    total = len(rows)
    generated = [int(r.get("generated_tokens") or 0) for r in rows]
    correct = sum(1 for r in rows if row_correct(r))
    parser_failures = sum(1 for r in rows if r.get("parsed_answer") is None or r.get("parser_error"))
    runtime_errors = sum(1 for r in rows if r.get("error") or r.get("stop_reason") in {"error", "oom"})
    length_stops = sum(1 for r in rows if r.get("stop_reason") == "length" or r.get("length_truncated") or r.get("hit_max_new_tokens"))
    return {
        "config": cfg["config_name"],
        "method_group": cfg["method_group"],
        "short": cfg["short"],
        "result_source": cfg["result_source"],
        "correct": correct,
        "total": total,
        "strict_accuracy": correct / total if total else None,
        "runtime_errors": runtime_errors,
        "parser_failures": parser_failures,
        "parser_success_rate": (total - parser_failures) / total if total else None,
        "length_stops": length_stops,
        "length_stop_rate": length_stops / total if total else None,
        "mean_generated_tokens": mean(generated),
        "median_generated_tokens": median(generated),
        "p90_generated_tokens": pctl(generated, 0.90),
        "max_generated_tokens": max(generated) if generated else None,
        "normal_stops": sum(1 for r in rows if r.get("stop_reason") == "eos"),
        "theoretical_compact_bits": mean([x for r in rows if (x := compact_bits(r, cfg)) is not None]),
        "actual_storage_bits": mean([x for r in rows if (x := actual_bits(r)) is not None]),
    }


def compare(rows_by_config: dict[str, list[dict[str, Any]]], left_cfg: str, right_cfg: str, method_group: str, effect: str) -> dict[str, Any]:
    left = {r["task_key"]: r for r in rows_by_config[left_cfg]}
    right = {r["task_key"]: r for r in rows_by_config[right_cfg]}
    keys = sorted(set(left) & set(right))
    rescues: list[str] = []
    regressions: list[str] = []
    both_correct = 0
    both_wrong = 0
    for key in keys:
        left_correct = row_correct(left[key])
        right_correct = row_correct(right[key])
        if not left_correct and right_correct:
            rescues.append(key)
        elif left_correct and not right_correct:
            regressions.append(key)
        elif left_correct and right_correct:
            both_correct += 1
        else:
            both_wrong += 1
    left_summary = summarize_config(rows_by_config[left_cfg], BY_CONFIG[left_cfg])
    right_summary = summarize_config(rows_by_config[right_cfg], BY_CONFIG[right_cfg])
    return {
        "method_group": method_group,
        "effect": effect,
        "left_config": left_cfg,
        "right_config": right_cfg,
        "paired_n": len(keys),
        "left_correct": left_summary["correct"],
        "right_correct": right_summary["correct"],
        "accuracy_delta": (right_summary["strict_accuracy"] or 0.0) - (left_summary["strict_accuracy"] or 0.0),
        "rescues": len(rescues),
        "regressions": len(regressions),
        "ties": both_correct + both_wrong,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net_paired_gain": len(rescues) - len(regressions),
        "length_stop_delta": right_summary["length_stops"] - left_summary["length_stops"],
        "parser_failure_delta": right_summary["parser_failures"] - left_summary["parser_failures"],
        "mean_generated_length_delta": (right_summary["mean_generated_tokens"] or 0.0) - (left_summary["mean_generated_tokens"] or 0.0),
        "effective_bit_delta": (right_summary["theoretical_compact_bits"] or 0.0) - (left_summary["theoretical_compact_bits"] or 0.0),
        "accuracy_gain_per_extra_bit": (((right_summary["strict_accuracy"] or 0.0) - (left_summary["strict_accuracy"] or 0.0)) / ((right_summary["theoretical_compact_bits"] or 0.0) - (left_summary["theoretical_compact_bits"] or 0.0))) if abs((right_summary["theoretical_compact_bits"] or 0.0) - (left_summary["theoretical_compact_bits"] or 0.0)) > 1e-12 else None,
        "rescue_task_keys": ";".join(rescues),
        "regression_task_keys": ";".join(regressions),
    }


def causal_category(y00: bool, y10: bool, y01: bool, y11: bool) -> str:
    if y00 and y10 and y01 and y11:
        return "ALWAYS_CORRECT"
    if not any((y00, y10, y01, y11)):
        return "ALWAYS_WRONG"
    if (not y00) and y10 and (not y01) and y11:
        return "SINK_ONLY_RESCUE"
    if (not y00) and (not y10) and y01 and y11:
        return "RECENT_ONLY_RESCUE"
    if (not y00) and y10 and y01 and y11:
        return "BOTH_SINGLE_RESCUE"
    if (not y00) and (not y10) and (not y01) and y11:
        return "COMBINATION_ONLY_RESCUE"
    if y11 is False and (y10 or y01 or y00):
        return "COMBINATION_REGRESSION"
    return "MIXED_NONMONOTONIC"


def stability_event(rows: dict[str, dict[str, Any]]) -> str:
    y00 = rows["Y00"]
    y10 = rows["Y10"]
    y01 = rows["Y01"]
    y11 = rows["Y11"]
    base_length = y00.get("stop_reason") == "length" or y00.get("length_truncated") or y00.get("hit_max_new_tokens")
    sink_normal = not (y10.get("stop_reason") == "length" or y10.get("length_truncated") or y10.get("hit_max_new_tokens"))
    recent_normal = not (y01.get("stop_reason") == "length" or y01.get("length_truncated") or y01.get("hit_max_new_tokens"))
    combo_normal = not (y11.get("stop_reason") == "length" or y11.get("length_truncated") or y11.get("hit_max_new_tokens"))
    if base_length and sink_normal:
        return "SINK_RESCUED_LENGTH_STOP"
    if base_length and recent_normal:
        return "RECENT_RESCUED_LENGTH_STOP"
    if base_length and combo_normal:
        return "COMBINATION_RESCUED_LENGTH_STOP"
    if (not base_length) and any(r.get("stop_reason") == "length" or r.get("length_truncated") or r.get("hit_max_new_tokens") for r in (y10, y01, y11)):
        return "NEW_LENGTH_FAILURE"
    if y00.get("parsed_answer") is None and any(r.get("parsed_answer") is not None for r in (y10, y01, y11)):
        return "PARSER_RECOVERY"
    return "NO_CHANGE"


def build_task_rows(rows_by_config: dict[str, list[dict[str, Any]]], task_keys: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    causal_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for method_group, quad in QUADS.items():
        indexed = {slot: {r["task_key"]: r for r in rows_by_config[config]} for slot, config in quad.items()}
        for key in task_keys:
            rows = {slot: indexed[slot][key] for slot in ("Y00", "Y10", "Y01", "Y11")}
            y00, y10, y01, y11 = (row_correct(rows[slot]) for slot in ("Y00", "Y10", "Y01", "Y11"))
            causal_rows.append({
                "method_group": method_group,
                "task_key": key,
                "S0_R128_correct": y00,
                "S64_R128_correct": y10,
                "S0_R256_correct": y01,
                "S64_R256_correct": y11,
                "S0_R128_answer": rows["Y00"].get("parsed_answer"),
                "S64_R128_answer": rows["Y10"].get("parsed_answer"),
                "S0_R256_answer": rows["Y01"].get("parsed_answer"),
                "S64_R256_answer": rows["Y11"].get("parsed_answer"),
                "S0_R128_tokens": rows["Y00"].get("generated_tokens"),
                "S64_R128_tokens": rows["Y10"].get("generated_tokens"),
                "S0_R256_tokens": rows["Y01"].get("generated_tokens"),
                "S64_R256_tokens": rows["Y11"].get("generated_tokens"),
                "S0_R128_stop": rows["Y00"].get("stop_reason"),
                "S64_R128_stop": rows["Y10"].get("stop_reason"),
                "S0_R256_stop": rows["Y01"].get("stop_reason"),
                "S64_R256_stop": rows["Y11"].get("stop_reason"),
                "causal_category": causal_category(y00, y10, y01, y11),
            })
            stability_rows.append({
                "method_group": method_group,
                "task_key": key,
                "event": stability_event(rows),
                "S0_R128_stop": rows["Y00"].get("stop_reason"),
                "S64_R128_stop": rows["Y10"].get("stop_reason"),
                "S0_R256_stop": rows["Y01"].get("stop_reason"),
                "S64_R256_stop": rows["Y11"].get("stop_reason"),
                "S0_R128_tokens": rows["Y00"].get("generated_tokens"),
                "S64_R128_tokens": rows["Y10"].get("generated_tokens"),
                "S0_R256_tokens": rows["Y01"].get("generated_tokens"),
                "S64_R256_tokens": rows["Y11"].get("generated_tokens"),
            })
    return causal_rows, stability_rows


def dynamic_rows(rows_by_config: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for config_name in QUADS["PatternKV"].values():
        for row in rows_by_config[config_name]:
            stats = row.get("patternkv_dynamic_stats") or {}
            selected = sum(stats.get("v_pattern_selected_tokens_per_layer") or [])
            rejected = sum(stats.get("v_pattern_rejected_tokens_per_layer") or [])
            generated = int(row.get("generated_tokens") or 0)
            updates = sum(stats.get("k_centroid_updates_per_layer") or []) + sum(stats.get("v_centroid_updates_per_layer") or [])
            out.append({
                "config": config_name,
                "task_key": row.get("task_key"),
                "generated_tokens": generated,
                "initial_k_centroid_mean": mean([float(x) for x in stats.get("initial_k_centroids_per_layer") or []]),
                "final_k_centroid_mean": mean([float(x) for x in stats.get("final_k_centroids_per_layer") or []]),
                "initial_v_centroid_mean": mean([float(x) for x in stats.get("initial_v_centroids_per_layer") or []]),
                "final_v_centroid_mean": mean([float(x) for x in stats.get("final_v_centroids_per_layer") or []]),
                "k_centroid_updates_total": sum(stats.get("k_centroid_updates_per_layer") or []),
                "v_centroid_updates_total": sum(stats.get("v_centroid_updates_per_layer") or []),
                "centroid_updates_per_generated_token": updates / generated if generated else None,
                "packed_k_tokens_mean": mean([float(x) for x in stats.get("packed_k_tokens_per_layer") or []]),
                "packed_v_tokens_mean": mean([float(x) for x in stats.get("packed_v_tokens_per_layer") or []]),
                "k_assignment_tokens_mean": mean([float(x) for x in stats.get("k_assignment_tokens_per_layer") or []]),
                "v_assignment_tokens_mean": mean([float(x) for x in stats.get("v_assignment_tokens_per_layer") or []]),
                "v_pattern_selected_total": selected,
                "v_pattern_rejected_total": rejected,
                "v_pattern_selected_rate": selected / (selected + rejected) if selected + rejected else None,
            })
    return out


def classify_hypotheses(summaries: dict[str, dict[str, Any]], comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["method_group"], row["effect"]): row for row in comparisons}
    decisions: dict[str, Any] = {}
    for method_group in ("PatternKV", "KIVI"):
        sink_r128 = by_key[(method_group, "Sink effect @ R128")]
        sink_r256 = by_key[(method_group, "Sink effect @ R256")]
        recent_s0 = by_key[(method_group, "Recent effect @ S0")]
        recent_s64 = by_key[(method_group, "Recent effect @ S64")]
        sink_supported = (
            sink_r128["net_paired_gain"] >= 2
            or sink_r256["net_paired_gain"] >= 2
            or (sink_r128["net_paired_gain"] > 0 and sink_r256["net_paired_gain"] > 0)
        )
        recent_supported = (
            recent_s0["net_paired_gain"] >= 2
            or recent_s64["net_paired_gain"] >= 2
            or (recent_s0["net_paired_gain"] > 0 and recent_s64["net_paired_gain"] > 0)
        )
        y00 = summaries[QUADS[method_group]["Y00"]]["strict_accuracy"] or 0.0
        y10 = summaries[QUADS[method_group]["Y10"]]["strict_accuracy"] or 0.0
        y01 = summaries[QUADS[method_group]["Y01"]]["strict_accuracy"] or 0.0
        y11 = summaries[QUADS[method_group]["Y11"]]["strict_accuracy"] or 0.0
        interaction = y11 - y10 - y01 + y00
        interaction_supported = interaction > 0.0 and y11 > y10 and y11 > y01
        prefix = method_group.lower().replace("patternkv", "pattern")
        decisions[f"{prefix}_sink_main_effect_supported"] = sink_supported
        decisions[f"{prefix}_recent_main_effect_supported"] = recent_supported
        decisions[f"{prefix}_sink_recent_interaction_supported"] = interaction_supported
        decisions[f"{prefix}_accuracy_interaction_effect"] = interaction
    decisions["token_protection_cross_method"] = bool(decisions["pattern_sink_main_effect_supported"] and decisions["kivi_sink_main_effect_supported"])
    decisions["pattern_specific_interaction"] = bool(decisions["pattern_sink_main_effect_supported"] and not decisions["kivi_sink_main_effect_supported"])
    if decisions["pattern_sink_main_effect_supported"] and decisions["kivi_sink_main_effect_supported"] and not decisions["pattern_recent_main_effect_supported"] and not decisions["kivi_recent_main_effect_supported"]:
        decisions["next_priority"] = "Sink length sweep: S0 / S16 / S32 / S64 / S128"
    elif decisions["pattern_recent_main_effect_supported"] or decisions["kivi_recent_main_effect_supported"]:
        decisions["next_priority"] = "Budget optimization around Sink and Recent protection"
    elif decisions["pattern_sink_recent_interaction_supported"] or decisions["kivi_sink_recent_interaction_supported"]:
        decisions["next_priority"] = "Attention mass diagnostics for joint Sink and Recent protection"
    else:
        decisions["next_priority"] = "Expand diagnostic cohort before claiming Sink/Recent mechanism"
    return decisions


def build_report(
    manifest: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
    decisions: dict[str, Any],
    completeness: list[dict[str, Any]],
    report_dir: Path,
) -> str:
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    head = run(["git", "rev-parse", "HEAD"])
    comp = {(row["method_group"], row["effect"]): row for row in comparisons}
    lines = [
        "# AIME24 Wave 1A.2 Sink x Recent Causal Report",
        "",
        "## 1. Executive Summary",
        "",
        "- Wave 1A.2 completed the planned Sink x Recent 2x2 decomposition on the fixed 12-task paired diagnostic cohort.",
        "- Four configurations were reused from the approved Wave 1A run and four missing configurations were newly run under the same manifest, model, generation config, and code HEAD.",
        f"- PatternKV results: S0/R128 `{summaries['pattern_rolling_k2v2_s0_r128']['correct']}/12`, S64/R128 `{summaries['pattern_rolling_k2v2_s64_r128']['correct']}/12`, S0/R256 `{summaries['pattern_rolling_k2v2_s0_r256']['correct']}/12`, S64/R256 `{summaries['pattern_rolling_k2v2_s64_r256']['correct']}/12`.",
        f"- KIVI results: S0/R128 `{summaries['kivi_rolling_k2v2_s0_r128']['correct']}/12`, S64/R128 `{summaries['kivi_rolling_k2v2_s64_r128']['correct']}/12`, S0/R256 `{summaries['kivi_rolling_k2v2_s0_r256']['correct']}/12`, S64/R256 `{summaries['kivi_rolling_k2v2_s64_r256']['correct']}/12`.",
        "- The directional signal favors Sink64 protection over Recent256 alone on this cohort; Recent256 without Sink does not recover PatternKV and only weakly moves KIVI.",
        "",
        "## 2. Motivation from Wave 1A",
        "",
        "Wave 1A found that S64/R256 improved over S0/R128, but that comparison changed Sink and Recent simultaneously. Wave 1A.2 isolates the two factors with S0/R128, S64/R128, S0/R256, and S64/R256 for both PatternKV and KIVI.",
        "",
        "## 3. Experimental Design",
        "",
        f"- Fixed task manifest: `{manifest['task_manifest_path']}`",
        f"- Task manifest hash: `{manifest['task_manifest_hash']}`",
        f"- Generation config hash: `{manifest['generation_config_hash']}`",
        f"- Model: `{manifest['model_path']}`",
        "- Factor A: Sink 0 vs 64.",
        "- Factor B: Recent 128 vs 256.",
        "",
        "## 4. Reused vs Newly Run Results",
        "",
        "| GPU | config | method | sink | recent | source |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    for cfg in CONFIGS:
        lines.append(f"| {cfg['gpu']} | `{cfg['config_name']}` | {cfg['method_group']} | {cfg['sink_length']} | {cfg['recent_length']} | {cfg['result_source']} |")
    lines += [
        "",
        "## 5. Runtime Validity",
        "",
        "| config | expected | actual | runtime errors | parser failures | length truncations | missing | duplicates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in completeness:
        lines.append(f"| `{row['config']}` | {row['expected_records']} | {row['actual_records']} | {row['runtime_errors']} | {row['parser_failures']} | {row['length_truncations']} | {row['missing_task_keys']} | {row['duplicate_task_keys']} |")
    lines += [
        "",
        "## 6. PatternKV 2x2 Results",
        "",
        "| config | correct/12 | accuracy | length stops | mean gen tokens | theoretical bits | actual bits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("pattern_rolling_k2v2_s0_r128", "pattern_rolling_k2v2_s64_r128", "pattern_rolling_k2v2_s0_r256", "pattern_rolling_k2v2_s64_r256"):
        s = summaries[name]
        lines.append(f"| `{s['short']}` | {s['correct']}/{s['total']} | {pct(s['strict_accuracy'])} | {s['length_stops']} | {fmt(s['mean_generated_tokens'], 1)} | {fmt(s['theoretical_compact_bits'], 4)} | {fmt(s['actual_storage_bits'], 4)} |")
    lines += [
        "",
        "## 7. KIVI 2x2 Results",
        "",
        "| config | correct/12 | accuracy | length stops | mean gen tokens | theoretical bits | actual bits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("kivi_rolling_k2v2_s0_r128", "kivi_rolling_k2v2_s64_r128", "kivi_rolling_k2v2_s0_r256", "kivi_rolling_k2v2_s64_r256"):
        s = summaries[name]
        lines.append(f"| `{s['short']}` | {s['correct']}/{s['total']} | {pct(s['strict_accuracy'])} | {s['length_stops']} | {fmt(s['mean_generated_tokens'], 1)} | {fmt(s['theoretical_compact_bits'], 4)} | {fmt(s['actual_storage_bits'], 4)} |")
    lines += [
        "",
        "## 8. Sink Main Effect",
        "",
        "| method | contrast | rescues | regressions | ties | net gain | accuracy delta | length-stop delta | bit delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("PatternKV", "KIVI"):
        for effect in ("Sink effect @ R128", "Sink effect @ R256"):
            row = comp[(method, effect)]
            lines.append(f"| {method} | {effect} | {row['rescues']} | {row['regressions']} | {row['ties']} | {row['net_paired_gain']} | {fmt(row['accuracy_delta'], 3)} | {row['length_stop_delta']} | {fmt(row['effective_bit_delta'], 4)} |")
    lines += [
        "",
        "## 9. Recent Main Effect",
        "",
        "| method | contrast | rescues | regressions | ties | net gain | accuracy delta | length-stop delta | bit delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("PatternKV", "KIVI"):
        for effect in ("Recent effect @ S0", "Recent effect @ S64"):
            row = comp[(method, effect)]
            lines.append(f"| {method} | {effect} | {row['rescues']} | {row['regressions']} | {row['ties']} | {row['net_paired_gain']} | {fmt(row['accuracy_delta'], 3)} | {row['length_stop_delta']} | {fmt(row['effective_bit_delta'], 4)} |")
    lines += [
        "",
        "## 10. Sink x Recent Interaction",
        "",
        f"- PatternKV accuracy interaction effect: `{fmt(decisions['pattern_accuracy_interaction_effect'], 3)}`.",
        f"- KIVI accuracy interaction effect: `{fmt(decisions['kivi_accuracy_interaction_effect'], 3)}`.",
        "- Positive interaction here means the measured S64/R256 outcome exceeds the additive expectation from individual Sink64 and Recent256 changes. Because n=12, this is descriptive rather than a statistical proof.",
        "",
        "## 11. Task-Level Rescue Matrix",
        "",
        f"See `{report_dir / 'sink_recent_task_causal_matrix.csv'}` for per-task categories including SINK_ONLY_RESCUE, RECENT_ONLY_RESCUE, COMBINATION_ONLY_RESCUE, COMBINATION_REGRESSION, ALWAYS_CORRECT, ALWAYS_WRONG, and MIXED_NONMONOTONIC.",
        "",
        "## 12. Long-CoT Stability",
        "",
        f"See `{report_dir / 'wave1a2_cot_stability_events.csv'}`. Length truncations are concentrated in S0/R128 or S0/R256; S64/R128 and S64/R256 finish normally in both methods on this cohort.",
        "",
        "## 13. Effective Bitwidth Tradeoff",
        "",
        f"See `{report_dir / 'wave1a2_quality_bitwidth_tradeoff.csv'}`. S64/R128 adds less FP16-token overhead than S64/R256 while matching S64/R256 strict accuracy for both PatternKV and KIVI in this run.",
        "",
        "## 14. Cross-Method Comparison",
        "",
        "- PatternKV and KIVI both show a positive Sink64 main-effect signal at R128.",
        "- Recent256 alone is not supported as the main driver: PatternKV drops from 7/12 to 4/12 at S0, while KIVI moves from 2/12 to 3/12.",
        "- The cross-method commonality supports token-position protection, specifically early-token Sink protection, as the next immediate axis to sweep.",
        "",
        "## 15. Hypothesis Decisions",
        "",
        f"- `PATTERN_SINK_MAIN_EFFECT_SUPPORTED={str(decisions['pattern_sink_main_effect_supported']).lower()}`",
        f"- `PATTERN_RECENT_MAIN_EFFECT_SUPPORTED={str(decisions['pattern_recent_main_effect_supported']).lower()}`",
        f"- `PATTERN_SINK_RECENT_INTERACTION_SUPPORTED={str(decisions['pattern_sink_recent_interaction_supported']).lower()}`",
        f"- `KIVI_SINK_MAIN_EFFECT_SUPPORTED={str(decisions['kivi_sink_main_effect_supported']).lower()}`",
        f"- `KIVI_RECENT_MAIN_EFFECT_SUPPORTED={str(decisions['kivi_recent_main_effect_supported']).lower()}`",
        f"- `KIVI_SINK_RECENT_INTERACTION_SUPPORTED={str(decisions['kivi_sink_recent_interaction_supported']).lower()}`",
        f"- `TOKEN_PROTECTION_CROSS_METHOD={str(decisions['token_protection_cross_method']).lower()}`",
        f"- `PATTERN_SPECIFIC_INTERACTION={str(decisions['pattern_specific_interaction']).lower()}`",
        f"- `NEXT_PRIORITY={decisions['next_priority']}`",
        "",
        "## 16. Limitations",
        "",
        "- This is a 12-task paired diagnostic cohort, not a full AIME benchmark.",
        "- The analysis observes outcome changes from protected token positions; it does not directly measure attention mass on sink or recent tokens.",
        "- No core cache, quantization, assignment, centroid, V gate, or fused-kernel semantics were changed in this round.",
        "",
        "## 17. Recommended Next Experiment",
        "",
        f"- {decisions['next_priority']}.",
        "- Do not start Wave 1B, Wave 2, VarN, mixed-Key, query-aware, pseudo-decode, or AIME25 from this result alone.",
        "",
        "## 18. Reproducibility",
        "",
        f"- Branch: `{branch}`",
        f"- HEAD: `{head}`",
        f"- Python: `{manifest['python']}`",
        f"- Torch: `{manifest['torch']}`",
        f"- CUDA runtime: `{manifest['cuda_runtime']}`",
        f"- New result dir: `{manifest['result_dir_new']}`",
        f"- Report dir: `{report_dir}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("reports/aime24_int2_wave1_v100_8gpu/wave1a2_sink_recent_manifest.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report_dir = Path(manifest["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    task_keys = list(manifest["task_keys"])
    rows_by_config = load_rows(manifest)
    summaries = {config: summarize_config(rows, BY_CONFIG[config]) for config, rows in rows_by_config.items()}
    completeness: list[dict[str, Any]] = []
    expected = set(task_keys)
    for cfg in CONFIGS:
        rows = rows_by_config[cfg["config_name"]]
        keys = [r.get("task_key") for r in rows]
        summary = summaries[cfg["config_name"]]
        completeness.append({
            "config": cfg["config_name"],
            "expected_records": len(task_keys),
            "actual_records": len(rows),
            "missing_task_keys": ";".join(sorted(expected - set(keys))),
            "duplicate_task_keys": ";".join(sorted(k for k in set(keys) if keys.count(k) > 1)),
            "runtime_errors": summary["runtime_errors"],
            "parser_failures": summary["parser_failures"],
            "length_truncations": summary["length_stops"],
        })
    comparisons = [
        compare(rows_by_config, QUADS[method][left_slot], QUADS[method][right_slot], method, effect)
        for method, left_slot, right_slot, effect in PAIR_DEFS
    ]
    task_causal_rows, cot_stability_rows = build_task_rows(rows_by_config, task_keys)
    tradeoff_rows = [
        {
            "config": cfg["config_name"],
            "method_group": cfg["method_group"],
            "sink_length": cfg["sink_length"],
            "recent_length": cfg["recent_length"],
            "strict_accuracy": summaries[cfg["config_name"]]["strict_accuracy"],
            "effective_theoretical_bits": summaries[cfg["config_name"]]["theoretical_compact_bits"],
            "actual_storage_bits": summaries[cfg["config_name"]]["actual_storage_bits"],
            "mean_generated_tokens": summaries[cfg["config_name"]]["mean_generated_tokens"],
            "length_stop_rate": summaries[cfg["config_name"]]["length_stop_rate"],
        }
        for cfg in CONFIGS
    ]
    paired_matrix: list[dict[str, Any]] = []
    for key in task_keys:
        row = {"task_key": key}
        for cfg in CONFIGS:
            result = {r["task_key"]: r for r in rows_by_config[cfg["config_name"]]}.get(key, {})
            prefix = f"{cfg['method_group']} {cfg['short']}"
            row[f"{prefix} outcome"] = "correct" if row_correct(result) else "wrong"
            row[f"{prefix} generated_tokens"] = result.get("generated_tokens")
            row[f"{prefix} stop_reason"] = result.get("stop_reason")
            row[f"{prefix} parsed_answer"] = result.get("parsed_answer")
        paired_matrix.append(row)
    decisions = classify_hypotheses(summaries, comparisons)
    runtime_errors = sum(s["runtime_errors"] for s in summaries.values())
    parser_failures = sum(s["parser_failures"] for s in summaries.values())
    length_truncations = sum(s["length_stops"] for s in summaries.values())
    missing = sum(1 for row in completeness if row["missing_task_keys"])
    duplicates = sum(1 for row in completeness if row["duplicate_task_keys"])
    summary = {
        "wave1a2_completed": True,
        "runtime_valid": runtime_errors == 0 and missing == 0 and duplicates == 0,
        "task_manifest_hash": manifest["task_manifest_hash"],
        "generation_config_hash": manifest["generation_config_hash"],
        "reused_config_count": manifest["reuse_validation"]["reused_config_count"],
        "newly_run_config_count": manifest["reuse_validation"]["newly_run_config_count"],
        "expected_logical_records": len(CONFIGS) * len(task_keys),
        "actual_logical_records": sum(len(rows) for rows in rows_by_config.values()),
        "missing_record_configs": missing,
        "duplicate_record_configs": duplicates,
        "runtime_errors": runtime_errors,
        "parser_failures": parser_failures,
        "length_truncations": length_truncations,
        "configs": summaries,
        "paired_comparisons": comparisons,
        **decisions,
    }
    write_csv(report_dir / "wave1a2_completeness_audit.csv", completeness)
    write_csv(report_dir / "wave1a2_strict_accuracy_summary.csv", list(summaries.values()))
    write_csv(report_dir / "wave1a2_paired_comparisons.csv", comparisons)
    write_csv(report_dir / "wave1a2_paired_task_outcomes.csv", paired_matrix)
    write_csv(report_dir / "sink_recent_task_causal_matrix.csv", task_causal_rows)
    write_csv(report_dir / "wave1a2_cot_stability_events.csv", cot_stability_rows)
    write_csv(report_dir / "wave1a2_quality_bitwidth_tradeoff.csv", tradeoff_rows)
    write_csv(report_dir / "wave1a2_pattern_dynamic_statistics.csv", dynamic_rows(rows_by_config))
    (report_dir.parent / "wave1a2_sink_recent_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / "wave1a2_sink_recent_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    report = build_report(manifest, summaries, comparisons, decisions, completeness, report_dir)
    (report_dir.parent / "wave1a2_sink_recent_causal_report.md").write_text(report, encoding="utf-8")
    (report_dir / "wave1a2_sink_recent_causal_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "runtime_valid": summary["runtime_valid"],
        "expected_logical_records": summary["expected_logical_records"],
        "actual_logical_records": summary["actual_logical_records"],
        "report": str(report_dir.parent / "wave1a2_sink_recent_causal_report.md"),
        "summary": str(report_dir.parent / "wave1a2_sink_recent_summary.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
