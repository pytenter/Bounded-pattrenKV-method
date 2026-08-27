#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_gsm8k_selector_components_pilot import EXPERIMENT_ID, METHOD_LABELS, METHODS, SELECTORS


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_method(root: Path, method: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / method).glob("p*.json")):
        try:
            rec = read_json(path)
            rec["_path"] = str(path)
        except Exception as exc:
            rec = {"method": method, "stop_reason": "invalid_json", "error": repr(exc), "_path": str(path)}
        rows.append(rec)
    return rows


def pct(num: int, den: int) -> float | None:
    return round(100.0 * num / den, 4) if den else None


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def summarize(rows: list[dict[str, Any]], planned: int) -> dict[str, Any]:
    completed = len(rows)
    correct = sum(1 for r in rows if r.get("is_correct"))
    parsed = sum(1 for r in rows if r.get("parsed_answer") is not None)
    gen = [int(r.get("generated_tokens") or 0) for r in rows]
    input_tokens = [int(r.get("input_tokens") or 0) for r in rows]
    wall = [float(r.get("wall_time_seconds") or 0.0) for r in rows if r.get("wall_time_seconds") is not None]
    tps = [float(r.get("tokens_per_second") or 0.0) for r in rows if r.get("tokens_per_second") is not None]
    total_gen = sum(gen)
    total_wall = sum(wall)
    return {
        "planned": planned,
        "completed": completed,
        "correct": correct,
        "accuracy_completed": pct(correct, completed),
        "strict_accuracy": pct(correct, planned),
        "parse_success_rate": pct(parsed, completed),
        "eos_count": sum(1 for r in rows if r.get("stop_reason") == "eos"),
        "length_count": sum(1 for r in rows if r.get("stop_reason") == "length"),
        "oom_count": sum(1 for r in rows if r.get("stop_reason") == "oom"),
        "error_count": sum(1 for r in rows if r.get("stop_reason") in ("error", "invalid_json") or r.get("error")),
        "avg_input_tokens": mean(input_tokens),
        "avg_generated_tokens": mean(gen),
        "median_generated_tokens": statistics.median(gen) if gen else None,
        "avg_tokens_per_second_record_mean": mean(tps),
        "aggregate_tokens_per_second": round(total_gen / total_wall, 6) if total_wall else None,
        "avg_wall_time_seconds": mean(wall),
        "max_peak_reserved_bytes": max([int(r.get("peak_memory_reserved_bytes") or 0) for r in rows], default=None),
    }


def paired(left: list[dict[str, Any]], right: list[dict[str, Any]], left_name: str, right_name: str) -> dict[str, Any]:
    l = {int(r["problem_id"]): r for r in left if "problem_id" in r}
    rr = {int(r["problem_id"]): r for r in right if "problem_id" in r}
    keys = sorted(set(l) & set(rr))
    both_correct = both_wrong = left_only = right_only = 0
    deltas = []
    for key in keys:
        a = bool(l[key].get("is_correct"))
        b = bool(rr[key].get("is_correct"))
        deltas.append((1 if a else 0) - (1 if b else 0))
        if a and b:
            both_correct += 1
        elif not a and not b:
            both_wrong += 1
        elif a:
            left_only += 1
        else:
            right_only += 1
    return {
        "comparison": f"{left_name}_vs_{right_name}",
        "paired_n": len(keys),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "left_correct_right_wrong": left_only,
        "left_wrong_right_correct": right_only,
        "paired_accuracy_delta": round(sum(deltas) / len(deltas), 6) if deltas else None,
    }


def bootstrap_delta(left: list[dict[str, Any]], right: list[dict[str, Any]], seed: int = 20260827, draws: int = 10000) -> dict[str, Any]:
    import numpy as np

    l = {int(r["problem_id"]): int(bool(r.get("is_correct"))) for r in left if "problem_id" in r}
    r = {int(row["problem_id"]): int(bool(row.get("is_correct"))) for row in right if "problem_id" in row}
    keys = sorted(set(l) & set(r))
    diffs = np.array([l[k] - r[k] for k in keys], dtype=float)
    if not len(diffs):
        return {"paired_n": 0, "delta_mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        means.append(float(rng.choice(diffs, size=len(diffs), replace=True).mean()))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"paired_n": len(keys), "delta_mean": round(float(diffs.mean()), 6), "ci95": [round(float(lo), 6), round(float(hi), 6)], "draws": draws, "seed": seed}


def mcnemar(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    p = paired(left, right, "left", "right")
    b = int(p["left_correct_right_wrong"])
    c = int(p["left_wrong_right_correct"])
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
        p_value = min(1.0, 2.0 * tail)
    return {"discordant_left_only": b, "discordant_right_only": c, "exact_two_sided_p": round(p_value, 8), "note": "Exploratory only; n=50 pilot is not a paper claim."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results/gsm8k_selector_components_pilot_v1/pilot"))
    p.add_argument("--report-dir", type=Path, default=Path("reports/gsm8k_selector_components_pilot_v1"))
    args = p.parse_args()
    manifest = read_json(args.report_dir / "pilot_manifest.json")
    manifest_rows = manifest["rows"]
    stratum_by_pid = {int(r["problem_id"]): r["stratum"] for r in manifest_rows}
    rows = {method: read_method(args.results_dir, method) for method in METHODS}
    planned = len(manifest_rows)
    summaries = {method: summarize(method_rows, planned) for method, method_rows in rows.items()}
    summary_rows = []
    for method, data in summaries.items():
        row = {"method": method, "method_label": METHOD_LABELS[method], "selector": SELECTORS[method], **data}
        summary_rows.append(row)
    write_csv(args.report_dir / "method_summary.csv", summary_rows, list(summary_rows[0].keys()))

    stratum_rows = []
    for method, method_rows in rows.items():
        by_stratum = defaultdict(list)
        for row in method_rows:
            by_stratum[stratum_by_pid.get(int(row.get("problem_id", -1)), "UNKNOWN")].append(row)
        for stratum in sorted(by_stratum):
            stratum_rows.append({"method": method, "stratum": stratum, **summarize(by_stratum[stratum], sum(1 for r in manifest_rows if r["stratum"] == stratum))})
    write_csv(args.report_dir / "stratum_summary.csv", stratum_rows, list(stratum_rows[0].keys()) if stratum_rows else ["method", "stratum"])

    transitions = [paired(rows[m], rows["causal_v4_25"], m, "causal_v4_25") for m in ("importance_only_v4_25", "error_only_v4_25")]
    write_csv(args.report_dir / "paired_transitions.csv", transitions, list(transitions[0].keys()))
    write_json(args.report_dir / "paired_bootstrap.json", {m: bootstrap_delta(rows[m], rows["causal_v4_25"]) for m in ("importance_only_v4_25", "error_only_v4_25")})
    write_json(args.report_dir / "mcnemar_tests.json", {m: mcnemar(rows[m], rows["causal_v4_25"]) for m in ("importance_only_v4_25", "error_only_v4_25")})

    activation = []
    for method, method_rows in rows.items():
        for row in method_rows:
            stats = (row.get("cache_bitwidth_stats") or {}).get("cache_segment_stats") or {}
            activation.append(
                {
                    "method": method,
                    "problem_id": row.get("problem_id"),
                    "stratum": stratum_by_pid.get(int(row.get("problem_id", -1)), "UNKNOWN"),
                    "packed_history_tokens": stats.get("packed_history_tokens"),
                    "generated_tokens": row.get("generated_tokens"),
                    "tokens_per_second": row.get("tokens_per_second"),
                    "is_correct": row.get("is_correct"),
                }
            )
    write_csv(args.report_dir / "activation_analysis.csv", activation, list(activation[0].keys()) if activation else ["method"])
    write_csv(args.report_dir / "runtime_diagnostic.csv", [{"method": m, "avg_tokens_per_second": s["avg_tokens_per_second_record_mean"], "aggregate_tokens_per_second": s["aggregate_tokens_per_second"], "avg_generated_tokens": s["avg_generated_tokens"], "avg_input_tokens": s["avg_input_tokens"], "avg_wall_time_seconds": s["avg_wall_time_seconds"]} for m, s in summaries.items()], ["method", "avg_tokens_per_second", "aggregate_tokens_per_second", "avg_generated_tokens", "avg_input_tokens", "avg_wall_time_seconds"])

    compact_fields = ["problem_id", "method", "value_precision_selector", "is_correct", "parsed_answer", "reference_answer", "generated_tokens", "input_tokens", "tokens_per_second", "wall_time_seconds", "stop_reason", "physical_gpu_id", "config_hash"]
    compact = [{k: row.get(k) for k in compact_fields} for method in METHODS for row in rows[method]]
    write_csv(args.report_dir / "canonical_pilot_rows.csv", compact, compact_fields)
    with gzip.open(args.report_dir / "canonical_pilot_rows.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in compact:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    completeness = {"expected_per_method": planned, "methods": {m: {"completed": len(rows[m]), "missing_problem_ids": sorted(set(stratum_by_pid) - {int(r.get("problem_id")) for r in rows[m] if "problem_id" in r})} for m in METHODS}}
    completeness["pass"] = all(v["completed"] == planned and not v["missing_problem_ids"] for v in completeness["methods"].values())
    write_json(args.report_dir / "completeness_audit.json", completeness)
    gate = {
        "experiment_id": EXPERIMENT_ID,
        "complete": completeness["pass"],
        "no_method_errors": all(s["error_count"] == 0 and s["oom_count"] == 0 for s in summaries.values()),
        "same_50_problem_ids": all({int(r.get("problem_id")) for r in rows[m] if "problem_id" in r} == set(stratum_by_pid) for m in METHODS),
        "smoke_excluded": "smoke" not in str(args.results_dir),
        "pilot_to_paper_claim_allowed": False,
        "methods": summaries,
        "paired_vs_causal": transitions,
    }
    gate["pass"] = bool(gate["complete"] and gate["no_method_errors"] and gate["same_50_problem_ids"] and gate["smoke_excluded"] and not gate["pilot_to_paper_claim_allowed"])
    write_json(args.report_dir / "final_gate.json", gate)
    write_json(args.report_dir / "full_run_recommendation.json", {"recommend_full_run": True, "basis": "Matched 50-item pilot only; use paired accuracy/runtime signs to decide priority, not as paper evidence.", "pilot_to_paper_claim_allowed": False})
    (args.report_dir / "claim_audit.md").write_text("# Claim Audit\n\nThis pilot is matched and useful for triage, but it is not a paper-level GSM8K claim. Smoke outputs are excluded from aggregation.\n", encoding="utf-8")
    lines = ["# GSM8K Selector Component Pilot", "", "| method | completed | correct | acc % | avg tok/s | aggregate tok/s | avg gen toks |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary_rows:
        lines.append(f"| {row['method']} | {row['completed']} | {row['correct']} | {row['accuracy_completed']} | {row['avg_tokens_per_second_record_mean']} | {row['aggregate_tokens_per_second']} | {row['avg_generated_tokens']} |")
    lines += ["", "## Paired Vs CAUSAL", "", "```json", json.dumps(transitions, indent=2, ensure_ascii=False), "```"]
    (args.report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
