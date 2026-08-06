#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.aime24_int2_wave1 import CONFIGS, BitwidthConfig, classify_failure, effective_bitwidth, read_result_files, summarize_config, write_csv


COMPARISONS = (
    ("pattern_k2v2_s0_r128", "kivi_k2v2_s0_r128"),
    ("kivi_k2v2_s64_r256", "kivi_k2v2_s0_r128"),
    ("pattern_k2v2_s64_r256", "pattern_k2v2_s0_r128"),
    ("pattern_k4v2_s0_r128", "pattern_k2v2_s0_r128"),
    ("pattern_k2v4_s0_r128", "pattern_k2v2_s0_r128"),
    ("pattern_k4v2_s0_r128", "pattern_k2v4_s0_r128"),
)


def exact_two_sided_binomial(a_only: int, b_only: int) -> float | None:
    n = a_only + b_only
    if n == 0:
        return None
    observed = min(a_only, b_only)
    prob = sum(math.comb(n, i) for i in range(observed + 1)) / (2**n)
    return min(1.0, 2.0 * prob)


def paired_bootstrap_ci(diffs: list[int]) -> tuple[float | None, float | None]:
    if not diffs:
        return None, None
    vals = []
    n = len(diffs)
    for i in range(1000):
        acc = 0
        seed = (i * 1103515245 + 12345) & 0x7FFFFFFF
        for _ in range(n):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            acc += diffs[seed % n]
        vals.append(acc / n)
    vals.sort()
    return vals[24], vals[974]


def paired(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], name_a: str, name_b: str) -> dict[str, Any]:
    a = {r["task_key"]: r for r in rows_a}
    b = {r["task_key"]: r for r in rows_b}
    both_correct = both_wrong = a_only = b_only = 0
    diffs: list[int] = []
    for key in sorted(set(a) & set(b)):
        ca = bool(a[key].get("is_correct")) and not a[key].get("error")
        cb = bool(b[key].get("is_correct")) and not b[key].get("error")
        diffs.append((1 if ca else 0) - (1 if cb else 0))
        if ca and cb:
            both_correct += 1
        elif not ca and not cb:
            both_wrong += 1
        elif ca:
            a_only += 1
        else:
            b_only += 1
    lo, hi = paired_bootstrap_ci(diffs)
    return {
        "comparison": f"{name_a} vs {name_b}",
        "paired_n": len(diffs),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "paired_strict_accuracy_difference": sum(diffs) / len(diffs) if diffs else None,
        "mcnemar_exact_p_value": exact_two_sided_binomial(a_only, b_only),
        "bootstrap_ci95_low": lo,
        "bootstrap_ci95_high": hi,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/aime24_int2_wave1_v100_8gpu"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/aime24_int2_wave1_v100_8gpu"))
    parser.add_argument("--planned", type=int, default=12)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    config_names = [cfg["config_name"] for cfg in CONFIGS]
    rows_by_config = {name: read_result_files(args.results_dir, name) for name in config_names}
    summary = {"results_dir": str(args.results_dir), "diagnostic_note": "This is a diagnostic experiment, not a final significance claim.", "configs": {}, "paired": []}
    per_sample = []
    for cfg in CONFIGS:
        name = cfg["config_name"]
        rows = rows_by_config[name]
        summary["configs"][name] = summarize_config(rows, args.planned)
        for row in rows:
            per_sample.append(
                {
                    "config_name": name,
                    "problem_id": row.get("problem_id"),
                    "sample_id": row.get("sample_id"),
                    "seed": row.get("seed"),
                    "task_key": row.get("task_key"),
                    "correct": row.get("is_correct"),
                    "parsed_answer": row.get("parsed_answer"),
                    "reference_answer": row.get("reference_answer"),
                    "generated_tokens": row.get("generated_tokens"),
                    "stop_reason": row.get("stop_reason"),
                    "parser_error": row.get("parser_error"),
                    "failure_type": classify_failure(row) if not row.get("is_correct") else "",
                    "wall_time_seconds": row.get("wall_time_seconds"),
                    "peak_memory_reserved_bytes": row.get("peak_memory_reserved_bytes"),
                }
            )
    for a, b in COMPARISONS:
        summary["paired"].append(paired(rows_by_config.get(a, []), rows_by_config.get(b, []), a, b))
    bit_rows = []
    cache_segment_rows = []
    for cfg in CONFIGS:
        for row in rows_by_config[cfg["config_name"]]:
            stats = row.get("cache_segment_stats") or (row.get("cache_bitwidth_stats") or {}).get("cache_segment_stats") or {}
            cache_segment_rows.append(
                {
                    "config_name": cfg["config_name"],
                    "task_key": row.get("task_key"),
                    "problem_id": row.get("problem_id"),
                    "sample_id": row.get("sample_id"),
                    "sink_tokens": stats.get("sink_tokens"),
                    "packed_history_tokens": stats.get("packed_history_tokens"),
                    "pending_history_tokens": stats.get("pending_history_tokens"),
                    "recent_tokens": stats.get("recent_tokens"),
                    "total_tokens": stats.get("total_tokens"),
                    "k_assignment_tokens": stats.get("k_assignment_tokens"),
                    "v_assignment_tokens": stats.get("v_assignment_tokens"),
                    "stop_reason": row.get("stop_reason"),
                    "error": row.get("error"),
                }
            )
        for total_tokens in (4096, 8192, 16384, 32768):
            stats = effective_bitwidth(
                BitwidthConfig(
                    method=cfg["config_name"],
                    total_tokens=total_tokens,
                    sink_length=int(cfg["sink_length"]),
                    recent_length=int(cfg["recent_length"]),
                    k_bits=float(cfg["k_bits"]) + (2.0 * float(cfg.get("mixed_key_mode") is not None) * 0.125),
                    v_bits=float(cfg["v_bits"]),
                    mixed_ratio=0.125 if cfg.get("mixed_key_mode") else 0.0,
                )
            )
            bit_rows.append({"config_name": cfg["config_name"], "total_cache_length": total_tokens, **stats})
    (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.report_dir / "per_sample.csv", per_sample)
    write_csv(args.report_dir / "paired_comparisons.csv", summary["paired"])
    write_csv(args.report_dir / "bitwidth_accounting.csv", bit_rows)
    write_csv(args.report_dir / "cache_segment_audit.csv", cache_segment_rows)
    failure_lines = ["# Failure Analysis", "", "Failure types are heuristic labels: reasoning_error, length_truncation, parser_failure, repetition_loop, runtime_error.", ""]
    for name, rows in rows_by_config.items():
        counts: dict[str, int] = {}
        for row in rows:
            if row.get("is_correct"):
                continue
            counts[classify_failure(row)] = counts.get(classify_failure(row), 0) + 1
        failure_lines.append(f"- {name}: {counts}")
    (args.report_dir / "failure_analysis.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
    notes = [
        "# Implementation Notes",
        "",
        "- All wave1 outputs are written under `results/aime24_int2_wave1_v100_8gpu/`.",
        "- Strict accuracy is `correct / 12 selected task keys`; valid-only accuracy is secondary.",
        "- Mixed-key reference masks are fixed files under `artifacts/aime24_wave1_masks/` and must be regenerated from calibration traces before claiming query-aware causal evidence.",
        "- This is a diagnostic experiment, not a final significance conclusion.",
    ]
    (args.report_dir / "implementation_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    lines = ["# AIME24 INT2 Wave1 Summary", "", "> This is a diagnostic experiment, not a final significance claim.", "", "| config | completed | valid | correct | strict accuracy | length | eos | errors | avg tokens |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in summary["configs"].items():
        lines.append(f"| {name} | {item['completed']} | {item['valid_parsed']} | {item['correct']} | {item['strict_accuracy']} | {item['length_stop_count']} | {item['eos_count']} | {item['error_oom_count']} | {item['average_generated_tokens']} |")
    lines += ["", "## Paired Comparisons", "", "| comparison | n | both correct | both wrong | A only | B only | diff | p | ci95 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in summary["paired"]:
        lines.append(f"| {row['comparison']} | {row['paired_n']} | {row['both_correct']} | {row['both_wrong']} | {row['a_only_correct']} | {row['b_only_correct']} | {row['paired_strict_accuracy_difference']} | {row['mcnemar_exact_p_value']} | [{row['bootstrap_ci95_low']}, {row['bootstrap_ci95_high']}] |")
    (args.report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.report_dir / "summary.md"), "configs": summary["configs"]}, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
