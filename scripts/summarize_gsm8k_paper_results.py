#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.gsm8k_paper_utils import EXPECTED_GSM8K_TEST, METHODS


def read_method(root: Path, method: str) -> list[dict]:
    rows = []
    for path in sorted((root / method).glob("p*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            rows.append({"method": method, "stop_reason": "invalid_json", "error": "invalid_json", "path": str(path)})
    return rows


def pct(n, d):
    return round(100.0 * n / d, 4) if d else None


def q(values, quant):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(quant * len(values)) - 1)]


def summarize(rows: list[dict]) -> dict:
    completed = len(rows)
    correct = sum(1 for r in rows if r.get("is_correct"))
    parsed = sum(1 for r in rows if r.get("parsed_answer") is not None)
    gen = [int(r.get("generated_tokens") or 0) for r in rows]
    wall = [float(r.get("wall_time_seconds") or 0) for r in rows if r.get("wall_time_seconds") is not None]
    mem = [int(r.get("peak_memory_reserved_bytes") or 0) for r in rows if r.get("peak_memory_reserved_bytes")]
    return {
        "planned": EXPECTED_GSM8K_TEST,
        "completed": completed,
        "correct": correct,
        "accuracy_completed": pct(correct, completed),
        "strict_accuracy": pct(correct, EXPECTED_GSM8K_TEST),
        "parse_success_rate": pct(parsed, completed),
        "eos_count": sum(1 for r in rows if r.get("stop_reason") == "eos"),
        "length_count": sum(1 for r in rows if r.get("stop_reason") == "length"),
        "oom_count": sum(1 for r in rows if r.get("stop_reason") == "oom"),
        "error_count": sum(1 for r in rows if r.get("stop_reason") in ("error", "invalid_json") or r.get("error")),
        "avg_generated_tokens": round(statistics.mean(gen), 2) if gen else None,
        "median_generated_tokens": statistics.median(gen) if gen else None,
        "p95_generated_tokens": q(gen, 0.95),
        "avg_wall_time": round(statistics.mean(wall), 4) if wall else None,
        "p95_wall_time": q(wall, 0.95),
        "peak_memory_max": max(mem) if mem else None,
        "bitwidth_summary": [r.get("cache_bitwidth_stats") for r in rows if r.get("cache_bitwidth_stats")][:3],
    }


def paired(left: list[dict], right: list[dict], lname: str, rname: str) -> dict:
    l = {r.get("problem_id"): r for r in left}
    rr = {r.get("problem_id"): r for r in right}
    keys = sorted(set(l) & set(rr))
    both_c = both_w = lc_rw = lw_rc = 0
    diffs = []
    for k in keys:
        a = bool(l[k].get("is_correct"))
        b = bool(rr[k].get("is_correct"))
        diffs.append((1 if a else 0) - (1 if b else 0))
        if a and b:
            both_c += 1
        elif not a and not b:
            both_w += 1
        elif a:
            lc_rw += 1
        else:
            lw_rc += 1
    return {"comparison": f"{lname}-{rname}", "paired_n": len(keys), "both_correct": both_c, "both_wrong": both_w, "left_correct_right_wrong": lc_rw, "left_wrong_right_correct": lw_rc, "paired_accuracy_difference": round(sum(diffs) / len(diffs), 6) if diffs else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results/paper_repro_v2/gsm8k_full"))
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--report-md", type=Path, default=Path("reports/paper_repro_v2/gsm8k_full/summary.md"))
    p.add_argument("--report-json", type=Path, default=Path("reports/paper_repro_v2/gsm8k_full/summary.json"))
    args = p.parse_args()
    rows = {m: read_method(args.results_dir, m) for m in args.methods}
    summary = {"results_dir": str(args.results_dir), "methods": {m: summarize(v) for m, v in rows.items()}, "paired": []}
    for a, b in [("patternkv_paper", "kivi_paper_g128"), ("patternkv_paper", "fp16"), ("kivi_paper_g128", "fp16")]:
        if a in rows and b in rows:
            summary["paired"].append(paired(rows[a], rows[b], a, b))
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# GSM8K Paper Full Summary", "", "| method | planned | completed | correct | acc completed | strict acc | parse | eos | length | oom | error | avg gen | p95 wall | peak mem |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m, s in summary["methods"].items():
        lines.append(f"| {m} | {s['planned']} | {s['completed']} | {s['correct']} | {s['accuracy_completed']} | {s['strict_accuracy']} | {s['parse_success_rate']} | {s['eos_count']} | {s['length_count']} | {s['oom_count']} | {s['error_count']} | {s['avg_generated_tokens']} | {s['p95_wall_time']} | {s['peak_memory_max']} |")
    lines += ["", "## Paired", "", "```json", json.dumps(summary["paired"], indent=2, ensure_ascii=False), "```"]
    args.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
