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


SINKS = (0, 16, 32, 64, 128)
METHODS = ("PatternKV", "KIVI")


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
    out = {}
    for cfg in manifest["logical_configs"]:
        path = Path(cfg["source_result_path"])
        out[cfg["config_name"]] = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("*.json"))]
    return out


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
    valid_rows = [row for row in rows if not row.get("error")]
    generated = [int(r.get("generated_tokens") or 0) for r in valid_rows]
    correct = sum(1 for r in valid_rows if row_correct(r))
    parser_failures = sum(1 for r in valid_rows if r.get("parsed_answer") is None or r.get("parser_error"))
    runtime_errors = sum(1 for r in rows if r.get("error") or r.get("stop_reason") in {"error", "oom"})
    length_stops = sum(1 for r in valid_rows if r.get("stop_reason") == "length" or r.get("length_truncated") or r.get("hit_max_new_tokens"))
    valid = total == 12 and runtime_errors == 0
    return {
        "config": cfg["config_name"],
        "method_group": cfg["method_group"],
        "sink_length": cfg["sink_length"],
        "recent_length": cfg["recent_length"],
        "result_source": cfg["result_source"],
        "valid_for_quality": valid,
        "correct": correct,
        "total": total,
        "valid_records": len(valid_rows),
        "strict_accuracy": correct / len(valid_rows) if valid_rows else None,
        "runtime_errors": runtime_errors,
        "parser_failures": parser_failures,
        "parser_success_rate": (len(valid_rows) - parser_failures) / len(valid_rows) if valid_rows else None,
        "length_stops": length_stops,
        "length_stop_rate": length_stops / len(valid_rows) if valid_rows else None,
        "mean_generated_tokens": mean(generated),
        "median_generated_tokens": median(generated),
        "p90_generated_tokens": pctl(generated, 0.90),
        "max_generated_tokens": max(generated) if generated else None,
        "theoretical_compact_bits": mean([x for r in valid_rows if (x := compact_bits(r, cfg)) is not None]),
        "actual_storage_bits": mean([x for r in valid_rows if (x := actual_bits(r)) is not None]),
        "first_runtime_error": next((repr(r.get("error")) for r in rows if r.get("error")), None),
    }


def index_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["task_key"]: row for row in rows}


def compare(rows_by_config: dict[str, list[dict[str, Any]]], left_cfg: str, right_cfg: str, method_group: str, effect: str) -> dict[str, Any]:
    left = index_by_task(rows_by_config[left_cfg])
    right = index_by_task(rows_by_config[right_cfg])
    keys = sorted(set(left) & set(right))
    rescues: list[str] = []
    regressions: list[str] = []
    comparable = 0
    both_correct = 0
    both_wrong = 0
    for key in keys:
        if left[key].get("error") or right[key].get("error"):
            continue
        comparable += 1
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
    return {
        "method_group": method_group,
        "effect": effect,
        "left_config": left_cfg,
        "right_config": right_cfg,
        "paired_n": comparable,
        "comparison_valid": comparable == 12,
        "rescues": len(rescues),
        "regressions": len(regressions),
        "ties": both_correct + both_wrong,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net_paired_gain": len(rescues) - len(regressions),
        "rescue_task_keys": ";".join(rescues),
        "regression_task_keys": ";".join(regressions),
    }


def first_rescue_pattern(values: dict[int, bool]) -> str | int | None:
    if values[0]:
        return None
    seen_correct = False
    for sink in (16, 32, 64, 128):
        if values.get(sink):
            seen_correct = True
            later = [values.get(s) for s in SINKS if s > sink]
            if any(v is False for v in later):
                return "NON_MONOTONIC"
            return sink
    return None


def task_threshold_rows(rows_by_config: dict[str, list[dict[str, Any]]], configs: dict[tuple[str, int], str], task_keys: list[str]) -> list[dict[str, Any]]:
    out = []
    for method in METHODS:
        indexed = {sink: index_by_task(rows_by_config[configs[(method, sink)]]) for sink in SINKS}
        for key in task_keys:
            correctness = {}
            row = {"method_group": method, "task_key": key}
            for sink in SINKS:
                result = indexed[sink].get(key, {})
                ok = row_correct(result) if not result.get("error") else None
                correctness[sink] = ok
                row[f"S{sink}_correct"] = ok
                row[f"S{sink}_stop"] = result.get("stop_reason")
                row[f"S{sink}_tokens"] = result.get("generated_tokens")
                row[f"S{sink}_answer"] = result.get("parsed_answer")
                row[f"S{sink}_error"] = result.get("error")
            row["first_sink_that_rescues"] = first_rescue_pattern({k: v for k, v in correctness.items() if v is not None})
            vals = [correctness[s] for s in SINKS if correctness[s] is not None]
            row["non_monotonic"] = any(vals[i] and not vals[j] for i in range(len(vals)) for j in range(i + 1, len(vals)))
            out.append(row)
    return out


def stability_rows(rows_by_config: dict[str, list[dict[str, Any]]], configs: dict[tuple[str, int], str], task_keys: list[str]) -> list[dict[str, Any]]:
    out = []
    for method in METHODS:
        indexed = {sink: index_by_task(rows_by_config[configs[(method, sink)]]) for sink in SINKS}
        for key in task_keys:
            base = indexed[0].get(key, {})
            base_length = base.get("stop_reason") == "length" or base.get("length_truncated") or base.get("hit_max_new_tokens")
            events = []
            for sink in (16, 32, 64, 128):
                row = indexed[sink].get(key, {})
                if row.get("error"):
                    continue
                if base_length and row.get("stop_reason") != "length":
                    events.append("LENGTH_STOP_RESCUE")
                if (not row_correct(base)) and row_correct(row):
                    events.append("EARLY_SINK_RESCUE" if sink in (16, 32) else "LARGER_SINK_ADDITIONAL_RESCUE")
                if row_correct(base) and not row_correct(row):
                    events.append("SINK_REGRESSION")
                if base.get("parsed_answer") is None and row.get("parsed_answer") is not None:
                    events.append("PARSER_RECOVERY")
            event = "NO_CHANGE"
            if "SINK_REGRESSION" in events and any(e.endswith("RESCUE") for e in events):
                event = "NON_MONOTONIC"
            elif events:
                event = events[0]
            out.append({"method_group": method, "task_key": key, "event": event, "events": ";".join(sorted(set(events)))})
    return out


def infer_decisions(summaries: dict[str, dict[str, Any]], comparisons: list[dict[str, Any]], configs: dict[tuple[str, int], str]) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    by_pair = {(row["method_group"], row["effect"]): row for row in comparisons}
    for method in METHODS:
        valid_sinks = [sink for sink in SINKS if summaries[configs[(method, sink)]]["valid_for_quality"]]
        vs0 = {sink: by_pair.get((method, f"S{sink} vs S0")) for sink in (16, 32, 64, 128)}
        effect_supported = any(row and row["comparison_valid"] and row["net_paired_gain"] > 0 for row in vs0.values())
        minimum = next((sink for sink in (16, 32, 64, 128) if (row := vs0.get(sink)) and row["comparison_valid"] and row["net_paired_gain"] > 0), None)
        valid_quality = [sink for sink in valid_sinks if sink != 128]
        best_acc = max((summaries[configs[(method, sink)]]["strict_accuracy"] or -1.0) for sink in valid_quality)
        pareto_candidates = [sink for sink in valid_quality if (summaries[configs[(method, sink)]]["strict_accuracy"] or -1.0) >= best_acc]
        best_pareto = min(pareto_candidates) if pareto_candidates else None
        best_sink = max(valid_quality, key=lambda sink: (summaries[configs[(method, sink)]]["strict_accuracy"] or -1.0, -sink))
        saturation = best_sink
        for sink in (16, 32, 64):
            if sink not in valid_quality:
                continue
            current_acc = summaries[configs[(method, sink)]]["strict_accuracy"] or -1.0
            future = [future_sink for future_sink in valid_quality if future_sink > sink]
            if not future:
                continue
            future_best = max(summaries[configs[(method, future_sink)]]["strict_accuracy"] or -1.0 for future_sink in future)
            adjacent = by_pair.get((method, f"S{sink} -> S{next_sink(sink)}"))
            if future_best <= current_acc and adjacent and adjacent["comparison_valid"] and adjacent["net_paired_gain"] <= 0:
                saturation = sink
                break
        if saturation is None:
            saturation = "not_reached" if summaries[configs[(method, 128)]]["valid_for_quality"] else "blocked_by_S128_runtime_error"
        prefix = "pattern" if method == "PatternKV" else "kivi"
        decisions[f"{prefix}_sink_effect_supported"] = effect_supported
        decisions[f"{prefix}_sink_saturation_point"] = saturation
        decisions[f"{prefix}_minimum_effective_sink_length"] = minimum
        decisions[f"{prefix}_best_pareto_sink_length"] = best_pareto
        decisions[f"{prefix}_sink_sweep_monotonic_accuracy"] = is_monotonic([summaries[configs[(method, sink)]]["strict_accuracy"] for sink in valid_quality])
    decisions["cross_method_sink_effect_supported"] = bool(decisions["pattern_sink_effect_supported"] and decisions["kivi_sink_effect_supported"])
    decisions["cross_method_sink_scale_consistent"] = decisions["pattern_best_pareto_sink_length"] == decisions["kivi_best_pareto_sink_length"]
    if decisions["pattern_best_pareto_sink_length"] == decisions["kivi_best_pareto_sink_length"]:
        decisions["cross_method_recommended_sink_length"] = decisions["pattern_best_pareto_sink_length"]
    else:
        decisions["cross_method_recommended_sink_length"] = 16 if decisions["pattern_minimum_effective_sink_length"] == 16 and decisions["kivi_minimum_effective_sink_length"] == 16 else None
    decisions["full_aime24_validation_recommended"] = decisions["cross_method_recommended_sink_length"] is not None and not any(
        summaries[configs[(method, 128)]]["valid_for_quality"] is False for method in METHODS
    )
    decisions["attention_mass_diagnostic_recommended"] = True
    decisions["next_priority"] = "Fix or formally define S128 sink validation before any full AIME24 expansion; use S16 as current Pareto candidate from valid data."
    return decisions


def next_sink(sink: int) -> int:
    return {0: 16, 16: 32, 32: 64, 64: 128}[sink]


def is_monotonic(values: list[float | None]) -> bool:
    numeric = [v for v in values if v is not None]
    return all(numeric[i] <= numeric[i + 1] for i in range(len(numeric) - 1))


def build_report(manifest: dict[str, Any], summaries: dict[str, dict[str, Any]], comparisons: list[dict[str, Any]], decisions: dict[str, Any], report_dir: Path, configs: dict[tuple[str, int], str]) -> str:
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    head = run(["git", "rev-parse", "HEAD"])
    comp = {(row["method_group"], row["effect"]): row for row in comparisons}
    lines = [
        "# AIME24 Wave 1A.3 Sink Length Sweep Report",
        "",
        "## 1. Executive Summary",
        "",
        "- Wave 1A.3 ran the planned Sink sweep for S16, S32, and S128 while reusing validated S0 and S64 records.",
        "- S16 and S32 completed for both PatternKV and KIVI. S128 hit a repeatable `sink token count mismatch` validation error in both methods and is excluded from quality/Pareto decisions.",
        "- On valid data, PatternKV improves from S0 7/12 to S16 9/12 and S32 9/12; KIVI improves from S0 2/12 to S16 6/12 and S32 5/12.",
        "- The current Pareto candidate is S16/R128: it reaches the best observed PatternKV accuracy and most of the KIVI Sink benefit with lower bit cost than S32/S64.",
        "",
        "## 2. Motivation",
        "",
        "Wave 1A.2 showed that Sink64, not Recent256, was the main source of the S64/R256 gain. This sweep asks how much early-token FP16 protection is needed.",
        "",
        "## 3. Experimental Design",
        "",
        f"- Task manifest hash: `{manifest['task_manifest_hash']}`",
        f"- Generation config hash: `{manifest['generation_config_hash']}`",
        "- Fixed: segmented rolling, recent_length=128, K2V2, group_size=128.",
        "- Variable: sink_length in S0, S16, S32, S64, S128.",
        "",
        "## 4. Reuse Validation",
        "",
        f"- Reuse validation status: `{manifest['reuse_validation']['status']}`.",
        f"- Reused records: `{manifest['reuse_validation']['reused_record_count']}`.",
        f"- Planned new records: `{manifest['reuse_validation']['planned_new_record_count']}`.",
        "",
        "## 5. Runtime Validity",
        "",
        "| method | sink | records | valid records | runtime errors | valid for quality | first error |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for method in METHODS:
        for sink in SINKS:
            s = summaries[configs[(method, sink)]]
            lines.append(f"| {method} | {sink} | {s['total']} | {s['valid_records']} | {s['runtime_errors']} | {s['valid_for_quality']} | {s['first_runtime_error'] or ''} |")
    for method, title in (("PatternKV", "## 6. PatternKV Sink Sweep"), ("KIVI", "## 7. KIVI Sink Sweep")):
        lines += [
            "",
            title,
            "",
            "| sink | correct/valid | accuracy | length stops | mean tokens | median tokens | P90 tokens | theoretical bits | actual bits |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for sink in SINKS:
            s = summaries[configs[(method, sink)]]
            lines.append(f"| {sink} | {s['correct']}/{s['valid_records']} | {pct(s['strict_accuracy'])} | {s['length_stops']} | {fmt(s['mean_generated_tokens'], 1)} | {fmt(s['median_generated_tokens'], 1)} | {fmt(s['p90_generated_tokens'], 1)} | {fmt(s['theoretical_compact_bits'], 4)} | {fmt(s['actual_storage_bits'], 4)} |")
    lines += [
        "",
        "## 8. Paired Rescues and Regressions",
        "",
        "| method | comparison | paired n | valid | rescues | regressions | ties | net gain |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        lines.append(f"| {row['method_group']} | {row['effect']} | {row['paired_n']} | {row['comparison_valid']} | {row['rescues']} | {row['regressions']} | {row['ties']} | {row['net_paired_gain']} |")
    lines += [
        "",
        "## 9. Sink Saturation Analysis",
        "",
        f"- PatternKV saturation point: `{decisions['pattern_sink_saturation_point']}`.",
        f"- KIVI saturation point: `{decisions['kivi_sink_saturation_point']}`.",
        "- S128 cannot be used to determine whether saturation continues beyond S64 because the current implementation/validation path rejects most S128 tasks.",
        "",
        "## 10. Minimum Effective Sink Length",
        "",
        f"- PatternKV minimum effective Sink: `{decisions['pattern_minimum_effective_sink_length']}`.",
        f"- KIVI minimum effective Sink: `{decisions['kivi_minimum_effective_sink_length']}`.",
        "",
        "## 11. Quality-Bitwidth Pareto",
        "",
        f"- PatternKV best Pareto Sink from valid data: `{decisions['pattern_best_pareto_sink_length']}`.",
        f"- KIVI best Pareto Sink from valid data: `{decisions['kivi_best_pareto_sink_length']}`.",
        f"- Cross-method recommended Sink: `{decisions['cross_method_recommended_sink_length']}`.",
        f"- See `{report_dir / 'wave1a3_sink_quality_bitwidth_tradeoff.csv'}`.",
        "",
        "## 12. Long-CoT Stability",
        "",
        f"See `{report_dir / 'wave1a3_sink_cot_stability.csv'}`.",
        "",
        "## 13. Task-Level Sink Thresholds",
        "",
        f"See `{report_dir / 'wave1a3_sink_task_thresholds.csv'}`.",
        "",
        "## 14. Cross-Method Consistency",
        "",
        f"- `CROSS_METHOD_SINK_EFFECT_SUPPORTED={str(decisions['cross_method_sink_effect_supported']).lower()}`.",
        f"- `CROSS_METHOD_SINK_SCALE_CONSISTENT={str(decisions['cross_method_sink_scale_consistent']).lower()}`.",
        "- Both methods improve at S16 relative to S0, supporting cross-method early-token protection.",
        "",
        "## 15. Hypothesis Decisions",
        "",
    ]
    for key in [
        "pattern_sink_effect_supported",
        "kivi_sink_effect_supported",
        "cross_method_sink_effect_supported",
        "pattern_sink_saturation_point",
        "kivi_sink_saturation_point",
        "pattern_minimum_effective_sink_length",
        "kivi_minimum_effective_sink_length",
        "pattern_best_pareto_sink_length",
        "kivi_best_pareto_sink_length",
        "cross_method_recommended_sink_length",
        "full_aime24_validation_recommended",
        "attention_mass_diagnostic_recommended",
        "next_priority",
    ]:
        lines.append(f"- `{key.upper()}={decisions[key]}`")
    lines += [
        "",
        "## 16. Limitations",
        "",
        "- n=12 paired diagnostic cohort, not full AIME24 accuracy.",
        "- S128 is runtime-invalid under current validation, so conclusions are limited to S0/S16/S32/S64.",
        "- This experiment manipulates protected token positions; it does not directly measure attention mass on early tokens.",
        "",
        "## 17. Recommended Next Experiment",
        "",
        "- First resolve whether S128 should include decode-time early tokens in sink semantics or whether validation should reflect prefill-only sink behavior.",
        "- After that, rerun S128 or proceed with S16/S32-focused validation depending on the clarified semantics.",
        "- Do not start Wave 1B, Wave 2, full AIME24, AIME25, VarN, mixed-Key, or query-aware work from this script.",
        "",
        "## 18. Reproducibility",
        "",
        f"- Branch: `{branch}`",
        f"- HEAD: `{head}`",
        f"- Python: `{manifest['python']}`",
        f"- Torch: `{manifest['torch']}`",
        f"- CUDA runtime: `{manifest['cuda_runtime']}`",
        f"- Result dir: `{manifest['result_dir_new']}`",
        f"- Report dir: `{report_dir}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("reports/aime24_int2_wave1_v100_8gpu/wave1a3_sink_length_sweep_manifest.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report_dir = Path(manifest["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    configs = {(cfg["method_group"], cfg["sink_length"]): cfg["config_name"] for cfg in manifest["logical_configs"]}
    cfg_by_name = {cfg["config_name"]: cfg for cfg in manifest["logical_configs"]}
    task_keys = list(manifest["task_keys"])
    rows_by_config = load_rows(manifest)
    summaries = {name: summarize_config(rows, cfg_by_name[name]) for name, rows in rows_by_config.items()}
    comparisons = []
    for method in METHODS:
        base = configs[(method, 0)]
        for sink in (16, 32, 64, 128):
            comparisons.append(compare(rows_by_config, base, configs[(method, sink)], method, f"S{sink} vs S0"))
        for left, right in ((0, 16), (16, 32), (32, 64), (64, 128)):
            comparisons.append(compare(rows_by_config, configs[(method, left)], configs[(method, right)], method, f"S{left} -> S{right}"))
    threshold_rows = task_threshold_rows(rows_by_config, configs, task_keys)
    stability = stability_rows(rows_by_config, configs, task_keys)
    decisions = infer_decisions(summaries, comparisons, configs)
    completeness = []
    for cfg in manifest["logical_configs"]:
        rows = rows_by_config[cfg["config_name"]]
        keys = [row.get("task_key") for row in rows]
        summary = summaries[cfg["config_name"]]
        completeness.append({
            "config": cfg["config_name"],
            "expected_records": len(task_keys),
            "actual_records": len(rows),
            "valid_records": summary["valid_records"],
            "missing_task_keys": ";".join(sorted(set(task_keys) - set(keys))),
            "duplicate_task_keys": ";".join(sorted(k for k in set(keys) if keys.count(k) > 1)),
            "runtime_errors": summary["runtime_errors"],
            "parser_failures": summary["parser_failures"],
            "length_truncations": summary["length_stops"],
            "valid_for_quality": summary["valid_for_quality"],
        })
    tradeoff = []
    for method in METHODS:
        base_bits = summaries[configs[(method, 0)]]["theoretical_compact_bits"] or 0.0
        base_cfg = configs[(method, 0)]
        for sink in SINKS:
            name = configs[(method, sink)]
            summary = summaries[name]
            vs0 = next((row for row in comparisons if row["method_group"] == method and row["effect"] == f"S{sink} vs S0"), None)
            bits = summary["theoretical_compact_bits"]
            tradeoff.append({
                "method": method,
                "sink_length": sink,
                "recent_length": 128,
                "config": name,
                "strict_accuracy": summary["strict_accuracy"],
                "valid_for_quality": summary["valid_for_quality"],
                "rescues_vs_s0": 0 if sink == 0 else (vs0 or {}).get("rescues"),
                "regressions_vs_s0": 0 if sink == 0 else (vs0 or {}).get("regressions"),
                "net_gain_vs_s0": 0 if sink == 0 else (vs0 or {}).get("net_paired_gain"),
                "length_stop_rate": summary["length_stop_rate"],
                "mean_generated_tokens": summary["mean_generated_tokens"],
                "theoretical_effective_bits": bits,
                "actual_storage_bits": summary["actual_storage_bits"],
                "extra_bits_vs_s0": None if bits is None else bits - base_bits,
                "gain_per_extra_bit": None if sink == 0 or bits is None or bits == base_bits else ((vs0 or {}).get("net_paired_gain") or 0) / (bits - base_bits),
            })
    summary_json = {
        "wave1a3_completed": True,
        "runtime_valid": all(row["runtime_errors"] == 0 for row in completeness),
        "runtime_valid_excluding_s128": all(row["runtime_errors"] == 0 for row in completeness if "_s128_" not in row["config"]),
        "task_manifest_hash": manifest["task_manifest_hash"],
        "generation_config_hash": manifest["generation_config_hash"],
        "reuse_validation_status": manifest["reuse_validation"]["status"],
        "reused_configs": manifest["reuse_validation"]["reused_config_count"],
        "newly_run_configs": manifest["reuse_validation"]["newly_run_config_count"],
        "expected_logical_records": len(task_keys) * len(manifest["logical_configs"]),
        "actual_logical_records": sum(len(rows) for rows in rows_by_config.values()),
        "new_records": sum(len(rows_by_config[cfg["config_name"]]) for cfg in manifest["logical_configs"] if cfg["result_source"] == "newly_run"),
        "reused_records": sum(len(rows_by_config[cfg["config_name"]]) for cfg in manifest["logical_configs"] if cfg["result_source"] == "reused"),
        "runtime_errors": sum(row["runtime_errors"] for row in completeness),
        "parser_failures": sum(row["parser_failures"] for row in completeness),
        "length_truncations": sum(row["length_truncations"] for row in completeness),
        "missing_record_configs": sum(1 for row in completeness if row["missing_task_keys"]),
        "duplicate_record_configs": sum(1 for row in completeness if row["duplicate_task_keys"]),
        "paired_task_set_identical": all(set(row.get("task_key") for row in rows_by_config[cfg["config_name"]]) == set(task_keys) for cfg in manifest["logical_configs"]),
        "configs": summaries,
        "paired_comparisons": comparisons,
        **decisions,
    }
    write_csv(report_dir / "wave1a3_completeness_audit.csv", completeness)
    write_csv(report_dir / "wave1a3_sink_sweep_summary.csv", list(summaries.values()))
    write_csv(report_dir / "wave1a3_sink_paired_comparisons.csv", comparisons)
    write_csv(report_dir / "wave1a3_sink_task_thresholds.csv", threshold_rows)
    write_csv(report_dir / "wave1a3_sink_cot_stability.csv", stability)
    write_csv(report_dir / "wave1a3_sink_quality_bitwidth_tradeoff.csv", tradeoff)
    (report_dir / "wave1a3_sink_length_sweep_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir.parent / "wave1a3_sink_length_sweep_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    report = build_report(manifest, summaries, comparisons, decisions, report_dir, configs)
    (report_dir / "wave1a3_sink_length_sweep_report.md").write_text(report, encoding="utf-8")
    (report_dir.parent / "wave1a3_sink_length_sweep_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(report_dir.parent / "wave1a3_sink_length_sweep_report.md"), "summary": str(report_dir.parent / "wave1a3_sink_length_sweep_summary.json"), "runtime_valid": summary_json["runtime_valid"], "runtime_valid_excluding_s128": summary_json["runtime_valid_excluding_s128"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
