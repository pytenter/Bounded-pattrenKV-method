#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.aime_utils import METHODS, length_bucket, majority_vote, paired_stats


def read_results(root: Path, method: str) -> list[dict]:
    rows = []
    for path in sorted((root / method).glob("p*_s*_*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            rows.append({"method": method, "path": str(path), "stop_reason": "invalid_json", "error": "invalid_json"})
    return rows


def pct(n, d):
    return round(100.0 * n / d, 4) if d else None


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, math.ceil(q * len(values)) - 1)
    return values[idx]


def summarize_method(rows: list[dict], planned: int, num_samples: int) -> dict:
    valid = [r for r in rows if r.get("parsed_answer") is not None and not r.get("error")]
    correct = [r for r in valid if r.get("is_correct")]
    generated = [int(r.get("generated_tokens") or 0) for r in rows]
    walls = [float(r.get("wall_time_seconds") or 0) for r in rows if r.get("wall_time_seconds") is not None]
    tps = [float(r.get("tokens_per_second") or 0) for r in rows if r.get("tokens_per_second")]
    mem = [int(r.get("peak_memory_reserved_bytes") or 0) for r in rows if r.get("peak_memory_reserved_bytes")]
    by_sample = {}
    for sid in range(num_samples):
        sample_rows = [r for r in valid if int(r.get("sample_id", -1)) == sid]
        by_sample[f"sample_id_{sid}_accuracy"] = pct(sum(1 for r in sample_rows if r.get("is_correct")), len(sample_rows))
    by_problem = defaultdict(list)
    for r in valid:
        by_problem[int(r["problem_id"])].append(r.get("parsed_answer"))
    majority_correct = ties = 0
    for pid, answers in by_problem.items():
        vote = majority_vote(answers)
        if vote["tie"]:
            ties += 1
        ref = next((r.get("reference_answer") for r in valid if int(r["problem_id"]) == pid), None)
        if vote["answer"] is not None and vote["answer"] == ref:
            majority_correct += 1
    buckets = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in valid:
        b = length_bucket(int(r.get("generated_tokens") or 0))
        buckets[b]["n"] += 1
        buckets[b]["correct"] += int(bool(r.get("is_correct")))
    return {
        "planned_tasks": planned,
        "completed": len(rows),
        "valid_responses": len(valid),
        "correct": len(correct),
        "avg_at_n": pct(len(correct), len(valid)),
        "strict_avg": pct(len(correct), planned),
        "sample_accuracy": by_sample,
        "parse_rate": pct(len(valid), len(rows)),
        "parser_failures": sum(1 for r in rows if r.get("parsed_answer") is None),
        "eos_stop": sum(1 for r in rows if r.get("stop_reason") == "eos"),
        "length_stop": sum(1 for r in rows if r.get("stop_reason") == "length"),
        "oom": sum(1 for r in rows if r.get("stop_reason") == "oom"),
        "other_errors": sum(1 for r in rows if r.get("stop_reason") in ("error", "invalid_json")),
        "avg_generated_tokens": round(statistics.mean(generated), 2) if generated else None,
        "median_generated_tokens": statistics.median(generated) if generated else None,
        "p90_generated_tokens": quantile(generated, 0.90),
        "p95_generated_tokens": quantile(generated, 0.95),
        "avg_wall_time": round(statistics.mean(walls), 4) if walls else None,
        "p95_wall_time": quantile(walls, 0.95),
        "avg_tokens_per_second": round(statistics.mean(tps), 4) if tps else None,
        "peak_reserved_bytes_mean": round(statistics.mean(mem), 2) if mem else None,
        "peak_reserved_bytes_max": max(mem) if mem else None,
        "majority_correct": majority_correct,
        "majority_ties": ties,
        "majority_tie_rate": pct(ties, len(by_problem)),
        "length_buckets": {k: {**v, "accuracy": pct(v["correct"], v["n"])} for k, v in sorted(buckets.items())},
        "cache_bitwidth_samples": [r.get("cache_bitwidth_stats") for r in rows if r.get("cache_bitwidth_stats")][:3],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results/paper_repro_v2/aime24_budget_n2"))
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--report-md", type=Path, default=Path("reports/paper_repro_v2/aime24/results_summary.md"))
    p.add_argument("--report-json", type=Path, default=Path("reports/paper_repro_v2/aime24/results_summary.json"))
    args = p.parse_args()
    planned = 30 * args.num_samples
    rows_by_method = {m: read_results(args.results_dir, m) for m in args.methods}
    summary = {"results_dir": str(args.results_dir), "num_samples": args.num_samples, "methods": {}, "paired": []}
    for method, rows in rows_by_method.items():
        summary["methods"][method] = summarize_method(rows, planned, args.num_samples)
    for a, b in [("patternkv_paper", "kivi_paper_g128"), ("patternkv_paper", "fp16"), ("kivi_paper_g128", "fp16")]:
        if a in rows_by_method and b in rows_by_method:
            summary["paired"].append(paired_stats(rows_by_method[a], rows_by_method[b], a, b))
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# AIME24 Results Summary", "", f"results_dir: `{args.results_dir}`", "", "| method | completed | valid | correct | Avg@N | strict_avg | oom | length | eos | avg tokens | p95 wall |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m, s in summary["methods"].items():
        lines.append(f"| {m} | {s['completed']} | {s['valid_responses']} | {s['correct']} | {s['avg_at_n']} | {s['strict_avg']} | {s['oom']} | {s['length_stop']} | {s['eos_stop']} | {s['avg_generated_tokens']} | {s['p95_wall_time']} |")
    lines += ["", "## Paired", "", "```json", json.dumps(summary["paired"], indent=2, ensure_ascii=False), "```"]
    args.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
