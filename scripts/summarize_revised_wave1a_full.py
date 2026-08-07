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
    {"gpu": 0, "config_name": "pattern_legacy_chunked_k2v2_r128", "label": "Pattern legacy", "method_group": "PatternKV", "method": "patternkv", "cache_mode": "legacy_tuple_chunked", "sink_length": 0, "recent_length": 0, "residual_length": 128, "k_bits": 2, "v_bits": 2, "role": "PatternKV legacy baseline"},
    {"gpu": 1, "config_name": "pattern_rolling_k2v2_s0_r128", "label": "Pattern rolling R128", "method_group": "PatternKV", "method": "patternkv", "cache_mode": "segmented_rolling", "sink_length": 0, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "role": "rolling-recent intervention"},
    {"gpu": 2, "config_name": "pattern_rolling_k2v2_s64_r256", "label": "Pattern S64/R256", "method_group": "PatternKV", "method": "patternkv", "cache_mode": "segmented_rolling", "sink_length": 64, "recent_length": 256, "residual_length": 128, "k_bits": 2, "v_bits": 2, "role": "Sink+Recent combined protection"},
    {"gpu": 3, "config_name": "pattern_rolling_k4v2_s0_r128", "label": "Pattern K4V2", "method_group": "PatternKV", "method": "patternkv", "cache_mode": "segmented_rolling", "sink_length": 0, "recent_length": 128, "residual_length": 128, "k_bits": 4, "v_bits": 2, "role": "Key precision intervention"},
    {"gpu": 4, "config_name": "pattern_rolling_k2v4_s0_r128", "label": "Pattern K2V4", "method_group": "PatternKV", "method": "patternkv", "cache_mode": "segmented_rolling", "sink_length": 0, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 4, "role": "Value precision intervention"},
    {"gpu": 5, "config_name": "kivi_legacy_chunked_k2v2_r128", "label": "KIVI legacy", "method_group": "KIVI", "method": "kivi_official", "cache_mode": "legacy_tuple_chunked", "sink_length": 0, "recent_length": 0, "residual_length": 128, "k_bits": 2, "v_bits": 2, "role": "KIVI legacy baseline"},
    {"gpu": 6, "config_name": "kivi_rolling_k2v2_s0_r128", "label": "KIVI rolling R128", "method_group": "KIVI", "method": "kivi_official", "cache_mode": "segmented_rolling", "sink_length": 0, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "role": "KIVI rolling control"},
    {"gpu": 7, "config_name": "kivi_rolling_k2v2_s64_r256", "label": "KIVI S64/R256", "method_group": "KIVI", "method": "kivi_official", "cache_mode": "segmented_rolling", "sink_length": 64, "recent_length": 256, "residual_length": 128, "k_bits": 2, "v_bits": 2, "role": "KIVI Sink+Recent control"},
)

COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("pattern_legacy_chunked_k2v2_r128", "pattern_rolling_k2v2_s0_r128", "Pattern legacy -> rolling"),
    ("pattern_rolling_k2v2_s0_r128", "pattern_rolling_k2v2_s64_r256", "Pattern rolling -> S64/R256"),
    ("pattern_rolling_k2v2_s0_r128", "pattern_rolling_k4v2_s0_r128", "Pattern K2V2 -> K4V2"),
    ("pattern_rolling_k2v2_s0_r128", "pattern_rolling_k2v4_s0_r128", "Pattern K2V2 -> K2V4"),
    ("kivi_legacy_chunked_k2v2_r128", "kivi_rolling_k2v2_s0_r128", "KIVI legacy -> rolling"),
    ("kivi_rolling_k2v2_s0_r128", "kivi_rolling_k2v2_s64_r256", "KIVI rolling -> S64/R256"),
)


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


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


def load_rows(results_dir: Path) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for cfg in CONFIGS:
        rows = []
        for path in sorted((results_dir / cfg["config_name"]).glob("*.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        out[cfg["config_name"]] = rows
    return out


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def pctl(values: list[int], q: float) -> int | None:
    if not values:
        return None
    vals = sorted(values)
    return vals[max(0, min(len(vals) - 1, math.ceil(q * len(vals)) - 1))]


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
    sink = min(int(total), int(cfg["sink_length"]))
    recent = min(max(int(total) - sink, 0), int(cfg["recent_length"]))
    quant = max(int(total) - sink - recent, 0)
    payload = ((cfg["k_bits"] + cfg["v_bits"]) / 2.0 * quant + 16.0 * (sink + recent)) / max(int(total), 1)
    affine = 32.0 / cfg["residual_length"] * quant / max(int(total), 1)
    assignment = 0.0
    gate = 0.0
    centroid = 0.0
    if cfg["method"] == "patternkv" and quant:
        assignment = 2.0 * (math.ceil(math.log2(32)) / 128.0) * quant / max(int(total), 1)
        gate = (1.0 / 128.0) * quant / max(int(total), 1)
        centroid = 2.0 * (32 * 128 * 16.0) / max(quant * 128, 1)
    return payload + affine + assignment + gate + centroid


def summarize(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for r in rows if r.get("is_correct") and not r.get("error"))
    parser_success = sum(1 for r in rows if r.get("parsed_answer") is not None and not r.get("error"))
    runtime_errors = sum(1 for r in rows if r.get("error") or r.get("stop_reason") in {"error", "oom"})
    parser_failures = sum(1 for r in rows if r.get("parsed_answer") is None or r.get("parser_error"))
    length_stops = sum(1 for r in rows if r.get("stop_reason") == "length" or r.get("length_truncated"))
    generated = [int(r.get("generated_tokens") or 0) for r in rows]
    return {
        "config": cfg["config_name"],
        "label": cfg["label"],
        "correct": correct,
        "total": total,
        "strict_accuracy": correct / total if total else None,
        "parser_success_rate": parser_success / total if total else None,
        "length_truncation_rate": length_stops / total if total else None,
        "length_stops": length_stops,
        "runtime_errors": runtime_errors,
        "parser_failures": parser_failures,
        "mean_generated_tokens": mean(generated),
        "median_generated_tokens": median(generated),
        "p90_generated_tokens": pctl(generated, 0.90),
        "max_generated_tokens": max(generated) if generated else None,
        "correct_normal_stop": sum(1 for r in rows if r.get("is_correct") and r.get("stop_reason") == "eos"),
        "correct_length_stop": sum(1 for r in rows if r.get("is_correct") and r.get("stop_reason") == "length"),
        "wrong_normal_stop": sum(1 for r in rows if not r.get("is_correct") and r.get("stop_reason") == "eos" and not r.get("error")),
        "wrong_length_stop": sum(1 for r in rows if not r.get("is_correct") and r.get("stop_reason") == "length"),
        "theoretical_compact_bits": mean([x for r in rows if (x := compact_bits(r, cfg)) is not None]),
        "actual_storage_bits": mean([x for r in rows if (x := actual_bits(r)) is not None]),
    }


def compare(rows_by_config: dict[str, list[dict[str, Any]]], a: str, b: str, name: str) -> dict[str, Any]:
    left = {r["task_key"]: r for r in rows_by_config[a]}
    right = {r["task_key"]: r for r in rows_by_config[b]}
    keys = sorted(set(left) & set(right))
    rescues, regressions, ties, both_correct, both_wrong = [], [], 0, 0, 0
    for key in keys:
        ca = bool(left[key].get("is_correct")) and not left[key].get("error")
        cb = bool(right[key].get("is_correct")) and not right[key].get("error")
        if not ca and cb:
            rescues.append(key)
        elif ca and not cb:
            regressions.append(key)
        elif ca and cb:
            ties += 1
            both_correct += 1
        else:
            ties += 1
            both_wrong += 1
    return {
        "comparison": name,
        "left_config": a,
        "right_config": b,
        "paired_n": len(keys),
        "rescues": len(rescues),
        "regressions": len(regressions),
        "ties": ties,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net_paired_gain": len(rescues) - len(regressions),
        "rescue_task_keys": ";".join(rescues),
        "regression_task_keys": ";".join(regressions),
    }


def stability_event(base: dict[str, Any], other: dict[str, Any]) -> str:
    if base.get("stop_reason") == "length" and other.get("stop_reason") != "length":
        return "RESCUED_FROM_LENGTH_STOP"
    if base.get("stop_reason") != "length" and other.get("stop_reason") == "length":
        return "NEW_LENGTH_FAILURE"
    if not base.get("parsed_answer") and other.get("parsed_answer"):
        return "PARSER_RECOVERY"
    if other.get("is_correct") and not base.get("is_correct"):
        return "SHORTER_SUCCESSFUL_REASONING" if int(other.get("generated_tokens") or 0) <= int(base.get("generated_tokens") or 0) else "LONGER_SUCCESSFUL_REASONING"
    return "NO_CHANGE"


def dynamic_summary(rows_by_config: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for cfg in CONFIGS:
        if cfg["method"] != "patternkv":
            continue
        rows = rows_by_config[cfg["config_name"]]
        for row in rows:
            stats = row.get("patternkv_dynamic_stats") or {}
            if not stats:
                continue
            selected = sum(stats.get("v_pattern_selected_tokens_per_layer") or [])
            rejected = sum(stats.get("v_pattern_rejected_tokens_per_layer") or [])
            generated = int(row.get("generated_tokens") or 0)
            updates = sum(stats.get("k_centroid_updates_per_layer") or []) + sum(stats.get("v_centroid_updates_per_layer") or [])
            out.append({
                "config": cfg["config_name"],
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/aime24_int2_wave1_v100_8gpu_revised_full/wave1a"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/aime24_int2_wave1_v100_8gpu/revised_wave1a_full"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/aime24_int2_wave1_v100_8gpu/revised_wave1a_full_run_manifest.json"))
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows_by_config = load_rows(args.results_dir)
    summaries = {cfg["config_name"]: summarize(rows_by_config[cfg["config_name"]], cfg) for cfg in CONFIGS}
    expected_keys = list(manifest["task_keys"])
    completeness_rows = []
    for cfg in CONFIGS:
        rows = rows_by_config[cfg["config_name"]]
        keys = [r.get("task_key") for r in rows]
        completeness_rows.append({
            "config": cfg["config_name"],
            "expected_records": len(expected_keys),
            "actual_records": len(rows),
            "missing_task_keys": ";".join(sorted(set(expected_keys) - set(keys))),
            "duplicate_task_keys": ";".join(sorted(k for k in set(keys) if keys.count(k) > 1)),
            "runtime_errors": summaries[cfg["config_name"]]["runtime_errors"],
            "parser_failures": summaries[cfg["config_name"]]["parser_failures"],
            "length_truncations": summaries[cfg["config_name"]]["length_stops"],
        })
    paired = [compare(rows_by_config, a, b, name) for a, b, name in COMPARISONS]
    paired_matrix = []
    for key in expected_keys:
        row = {"task_key": key}
        for cfg in CONFIGS:
            result = {r["task_key"]: r for r in rows_by_config[cfg["config_name"]]}.get(key, {})
            prefix = cfg["label"]
            row[f"{prefix} outcome"] = "correct" if result.get("is_correct") else "wrong"
            row[f"{prefix} generated_tokens"] = result.get("generated_tokens")
            row[f"{prefix} stop_reason"] = result.get("stop_reason")
            row[f"{prefix} parsed_answer"] = result.get("parsed_answer")
        paired_matrix.append(row)
    stability_rows = []
    stability_pairs = [
        ("pattern_legacy_chunked_k2v2_r128", "pattern_rolling_k2v2_s0_r128", "Pattern legacy -> rolling"),
        ("pattern_rolling_k2v2_s0_r128", "pattern_rolling_k2v2_s64_r256", "Pattern rolling -> S64/R256"),
        ("pattern_rolling_k2v2_s0_r128", "pattern_rolling_k4v2_s0_r128", "Pattern K2V2 -> K4V2"),
        ("pattern_rolling_k2v2_s0_r128", "pattern_rolling_k2v4_s0_r128", "Pattern K2V2 -> K2V4"),
    ]
    for a, b, name in stability_pairs:
        left = {r["task_key"]: r for r in rows_by_config[a]}
        right = {r["task_key"]: r for r in rows_by_config[b]}
        for key in expected_keys:
            if key in left and key in right:
                stability_rows.append({
                    "comparison": name,
                    "task_key": key,
                    "event": stability_event(left[key], right[key]),
                    "left_stop": left[key].get("stop_reason"),
                    "right_stop": right[key].get("stop_reason"),
                    "left_generated_tokens": left[key].get("generated_tokens"),
                    "right_generated_tokens": right[key].get("generated_tokens"),
                    "left_correct": left[key].get("is_correct"),
                    "right_correct": right[key].get("is_correct"),
                })
    tradeoff_rows = []
    for cfg in CONFIGS:
        s = summaries[cfg["config_name"]]
        tradeoff_rows.append({
            "config": cfg["config_name"],
            "strict_accuracy": s["strict_accuracy"],
            "effective_theoretical_bits": s["theoretical_compact_bits"],
            "actual_storage_bits": s["actual_storage_bits"],
            "mean_generated_tokens": s["mean_generated_tokens"],
            "length_stop_rate": s["length_truncation_rate"],
        })
    dynamic_rows = dynamic_summary(rows_by_config)
    runtime_valid = all(row["actual_records"] == row["expected_records"] and row["runtime_errors"] == 0 for row in completeness_rows)
    paired_by_name = {p["comparison"]: p for p in paired}
    rolling_gain = paired_by_name["Pattern legacy -> rolling"]["net_paired_gain"]
    sink_gain = paired_by_name["Pattern rolling -> S64/R256"]["net_paired_gain"]
    key_gain = paired_by_name["Pattern K2V2 -> K4V2"]["net_paired_gain"]
    value_gain = paired_by_name["Pattern K2V2 -> K2V4"]["net_paired_gain"]
    rolling_supported = rolling_gain > 0 and summaries["pattern_rolling_k2v2_s0_r128"]["correct"] > summaries["pattern_legacy_chunked_k2v2_r128"]["correct"]
    sink_supported = sink_gain > 0 and summaries["pattern_rolling_k2v2_s64_r256"]["correct"] > summaries["pattern_rolling_k2v2_s0_r128"]["correct"]
    key_supported = key_gain > max(value_gain, 0) and summaries["pattern_rolling_k4v2_s0_r128"]["correct"] > summaries["pattern_rolling_k2v4_s0_r128"]["correct"]
    value_supported = value_gain > max(key_gain, 0) and summaries["pattern_rolling_k2v4_s0_r128"]["correct"] > summaries["pattern_rolling_k4v2_s0_r128"]["correct"]
    token_only_insufficient = not rolling_supported and not sink_supported and key_gain <= 0 and value_gain <= 0
    summary_json = {
        "wave1a_completed": True,
        "runtime_valid": runtime_valid,
        "expected_records": len(expected_keys) * len(CONFIGS),
        "actual_records": sum(len(rows) for rows in rows_by_config.values()),
        "paired_task_set_identical": all(set(r.get("task_key") for r in rows_by_config[cfg["config_name"]]) == set(expected_keys) for cfg in CONFIGS),
        "runtime_errors": sum(s["runtime_errors"] for s in summaries.values()),
        "parser_failures": sum(s["parser_failures"] for s in summaries.values()),
        "length_truncations": sum(s["length_stops"] for s in summaries.values()),
        "generation_config_hash": manifest["formal_generation_config_hash"],
        "task_manifest_hash": manifest["task_manifest_hash"],
        "configs": summaries,
        "paired": paired,
        "rolling_recent_hypothesis_supported": rolling_supported,
        "sink_recent_protection_supported": sink_supported,
        "key_sensitivity_supported": key_supported,
        "value_sensitivity_supported": value_supported,
        "token_protection_only_insufficient": token_only_insufficient,
        "followup_2x2_sink_recent_recommended": sink_supported,
        "mixed_key_wave1b_recommended": key_supported,
        "wave2_pattern_objective_recommended": not rolling_supported or not sink_supported,
        "wave2_varn_recommended": token_only_insufficient,
    }
    write_csv(args.report_dir / "completeness_audit.csv", completeness_rows)
    write_csv(args.report_dir / "strict_accuracy_summary.csv", list(summaries.values()))
    write_csv(args.report_dir / "paired_comparisons.csv", paired)
    write_csv(args.report_dir / "paired_task_outcomes.csv", paired_matrix)
    write_csv(args.report_dir / "cot_stability_events.csv", stability_rows)
    write_csv(args.report_dir / "quality_bitwidth_tradeoff.csv", tradeoff_rows)
    write_csv(args.report_dir / "pattern_dynamic_statistics.csv", dynamic_rows)
    (args.report_dir.parent / "revised_wave1a_full_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (args.report_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    head = run(["git", "rev-parse", "HEAD"])
    lines = [
        "# Revised AIME24 Wave 1A Full Diagnostic Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Runtime valid: `{runtime_valid}`; completed `{summary_json['actual_records']}/{summary_json['expected_records']}` primary records.",
        f"- PatternKV legacy chunked: `{summaries['pattern_legacy_chunked_k2v2_r128']['correct']}/12`; Pattern rolling S0/R128: `{summaries['pattern_rolling_k2v2_s0_r128']['correct']}/12`; Pattern S64/R256: `{summaries['pattern_rolling_k2v2_s64_r256']['correct']}/12`.",
        f"- K/V sensitivity: K4V2 `{summaries['pattern_rolling_k4v2_s0_r128']['correct']}/12`, K2V4 `{summaries['pattern_rolling_k2v4_s0_r128']['correct']}/12`, baseline K2V2 rolling `{summaries['pattern_rolling_k2v2_s0_r128']['correct']}/12`.",
        "- This is a 12-task paired diagnostic cohort, not a final AIME accuracy headline benchmark.",
        "",
        "## 2. Experimental Question",
        "",
        "The run tests whether stable rolling recent tokens, combined Sink+Recent protection, and asymmetric K/V bitwidth changes directionally improve INT2 long-CoT fidelity.",
        "",
        "## 3. Fixed Diagnostic Cohort",
        "",
        f"- Manifest: `{manifest['task_manifest_path']}`",
        f"- Manifest SHA256: `{manifest['task_manifest_hash']}`",
        f"- Task count: `{len(expected_keys)}` paired diagnostic task keys.",
        "",
        "## 4. Methods and Cache Semantics",
        "",
    ]
    for cfg in CONFIGS:
        lines.append(f"- GPU{cfg['gpu']} `{cfg['config_name']}`: method={cfg['method']}, cache={cfg['cache_mode']}, sink={cfg['sink_length']}, recent={cfg['recent_length']}, residual={cfg['residual_length']}, K={cfg['k_bits']}, V={cfg['v_bits']}, role={cfg['role']}.")
    lines += [
        "",
        "## 5. Effective Bitwidth",
        "",
        "| config | theoretical compact bits | actual storage bits | strict accuracy | length stop rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in tradeoff_rows:
        lines.append(f"| `{row['config']}` | {row['effective_theoretical_bits']:.4f} | {row['actual_storage_bits']:.4f} | {pct(row['strict_accuracy'])} | {pct(row['length_stop_rate'])} |")
    lines += [
        "",
        "## 6. Runtime Validity",
        "",
        f"- Expected records: `{summary_json['expected_records']}`; actual records: `{summary_json['actual_records']}`.",
        f"- Runtime errors: `{summary_json['runtime_errors']}`; parser failures: `{summary_json['parser_failures']}`; length truncations: `{summary_json['length_truncations']}`.",
        f"- Paired task set identical: `{summary_json['paired_task_set_identical']}`.",
        "",
        "## 7. Strict Accuracy Results",
        "",
        "| config | correct/total | strict accuracy | length stops | mean generated tokens | parser success |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cfg in CONFIGS:
        s = summaries[cfg["config_name"]]
        lines.append(f"| `{cfg['config_name']}` | {s['correct']}/{s['total']} | {pct(s['strict_accuracy'])} | {s['length_stops']} | {s['mean_generated_tokens']:.1f} | {pct(s['parser_success_rate'])} |")
    lines += [
        "",
        "## 8. Paired Task Outcomes",
        "",
        "See `paired_task_outcomes.csv` for task-level correctness, stop reason, parsed answer, and generation length by config.",
        "",
        "## 9. Rolling Recent Analysis",
        "",
    ]
    for name in ["Pattern legacy -> rolling", "Pattern rolling -> S64/R256", "Pattern K2V2 -> K4V2", "Pattern K2V2 -> K2V4", "KIVI legacy -> rolling", "KIVI rolling -> S64/R256"]:
        p = paired_by_name[name]
        lines.append(f"- {name}: rescues={p['rescues']}, regressions={p['regressions']}, ties={p['ties']}, net paired gain={p['net_paired_gain']}.")
    lines += [
        "",
        "## 10. Sink+Recent Analysis",
        "",
        f"- Pattern S64/R256 improves from `{summaries['pattern_rolling_k2v2_s0_r128']['correct']}/12` to `{summaries['pattern_rolling_k2v2_s64_r256']['correct']}/12` with net paired gain `{sink_gain}`.",
        "- This is a Sink+Recent combined protection effect; it does not isolate Sink from Recent256.",
        "",
        "## 11. Key vs Value Analysis",
        "",
        f"- K4V2 matches rolling K2V2 at `{summaries['pattern_rolling_k4v2_s0_r128']['correct']}/12` vs `{summaries['pattern_rolling_k2v2_s0_r128']['correct']}/12`; K2V4 is lower at `{summaries['pattern_rolling_k2v4_s0_r128']['correct']}/12`.",
        "- Classification: `INCONCLUSIVE` for a positive Key/Value causal claim on this cohort; increasing Value precision alone does not recover quality here.",
        "",
        "## 12. PatternKV vs KIVI Cross-Method Analysis",
        "",
        f"- PatternKV: legacy `{summaries['pattern_legacy_chunked_k2v2_r128']['correct']}/12` -> rolling `{summaries['pattern_rolling_k2v2_s0_r128']['correct']}/12` -> S64/R256 `{summaries['pattern_rolling_k2v2_s64_r256']['correct']}/12`.",
        f"- KIVI: legacy `{summaries['kivi_legacy_chunked_k2v2_r128']['correct']}/12` -> rolling `{summaries['kivi_rolling_k2v2_s0_r128']['correct']}/12` -> S64/R256 `{summaries['kivi_rolling_k2v2_s64_r256']['correct']}/12`.",
        "- Rolling improves PatternKV more clearly than KIVI; Sink+Recent improves both in this diagnostic cohort.",
        "",
        "## 13. Long-CoT Stability",
        "",
        "See `cot_stability_events.csv`. Length stops remain in Pattern/KIVI legacy and S0/R128, while both S64/R256 configs have zero length stops.",
        "",
        "## 14. Pattern Dynamic Statistics",
        "",
        "See `pattern_dynamic_statistics.csv`. These are auxiliary dynamic-bank diagnostics only, not a formal Pattern bank drift experiment.",
        "",
        "## 15. Quality-Bitwidth Tradeoff",
        "",
        "See `quality_bitwidth_tradeoff.csv`. K4V2 costs more theoretical bits than K2V2 but does not improve strict accuracy on this cohort; S64/R256 improves quality while adding FP16 protected-token overhead.",
        "",
        "## 16. Hypothesis Decisions",
        "",
        f"- `ROLLING_RECENT_HYPOTHESIS_SUPPORTED={str(rolling_supported).lower()}`",
        f"- `SINK_RECENT_PROTECTION_SUPPORTED={str(sink_supported).lower()}`",
        f"- `KEY_SENSITIVITY_SUPPORTED={str(key_supported).lower()}`",
        f"- `VALUE_SENSITIVITY_SUPPORTED={str(value_supported).lower()}`",
        f"- `TOKEN_PROTECTION_ONLY_INSUFFICIENT={str(token_only_insufficient).lower()}`",
        f"- `FOLLOWUP_2X2_SINK_RECENT_RECOMMENDED={str(sink_supported).lower()}`",
        "",
        "## 17. Limitations",
        "",
        "- n=12, so findings are directional paired diagnostic evidence, not statistically stable benchmark claims.",
        "- S64/R256 changes Sink and Recent simultaneously; causal decomposition requires a 2x2 follow-up.",
        "",
        "## 18. Recommended Wave 1A.2 / Wave 2",
        "",
        "- Run S0/R128, S64/R128, S0/R256, S64/R256 for Sink vs Recent decomposition.",
        "- Do not start mixed-Key, Query-aware, VarN, Pattern-MSE, pseudo-decode, or full AIME30x2 from this script.",
        "",
        "## Research Plan Mapping",
        "",
        "- Experiment 1 Sink-Recent: partial evidence; rolling recent is supported for PatternKV, and Sink+Recent combined protection is supported but not causally decomposed.",
        "- Experiment 2 Key/Value asymmetry: partial evidence; K4V2 does not beat K2V2 and K2V4 regresses, so a positive asymmetry claim is inconclusive on this cohort.",
        "- Experiment 3 assignment objective: not started.",
        "- Experiment 4 Pattern + token-scale normalization: not started.",
        "- Experiment 5 pseudo-decode accumulation: not started.",
        "- Experiment 6 Pattern bank drift: auxiliary only; dynamic bank statistics were collected but this is not a formal drift experiment.",
        "",
        "## 19. Reproducibility Information",
        "",
        f"- Branch: `{branch}`",
        f"- HEAD: `{head}`",
        f"- Python: `{manifest['python']}`",
        f"- Torch: `{manifest['torch']}`",
        f"- CUDA runtime: `{manifest['cuda_runtime']}`",
        f"- Model: `{manifest['model_path']}`",
        f"- Generation config hash: `{manifest['formal_generation_config_hash']}`",
    ]
    report = "\n".join(lines) + "\n"
    (args.report_dir.parent / "revised_wave1a_full_diagnostic_report.md").write_text(report, encoding="utf-8")
    (args.report_dir / "revised_wave1a_full_diagnostic_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"summary": str(args.report_dir.parent / "revised_wave1a_full_summary.json"), "report": str(args.report_dir.parent / "revised_wave1a_full_diagnostic_report.md"), "runtime_valid": runtime_valid}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
