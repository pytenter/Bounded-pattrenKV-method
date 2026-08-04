#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_METHODS = ["fp16", "kivi", "patternkv"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    bad = 0
    if not path.exists():
        return rows, bad
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def exact_mcnemar_p(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * prob)


def bootstrap_ci(rows_a: dict[int, dict], rows_b: dict[int, dict], rounds: int = 2000) -> dict[str, float] | None:
    common = sorted(set(rows_a) & set(rows_b))
    if not common:
        return None
    rng = random.Random(0)
    diffs = []
    n = len(common)
    for _ in range(rounds):
        total = 0
        for _ in range(n):
            idx = common[rng.randrange(n)]
            total += int(bool(rows_a[idx].get("correct"))) - int(bool(rows_b[idx].get("correct")))
        diffs.append(100.0 * total / n)
    diffs.sort()
    return {
        "p2_5": round(diffs[int(0.025 * (rounds - 1))], 4),
        "p50": round(diffs[int(0.5 * (rounds - 1))], 4),
        "p97_5": round(diffs[int(0.975 * (rounds - 1))], 4),
    }


def summarize_method(root: Path, mode: str, method: str, expected_samples: int, num_shards: int) -> dict[str, Any]:
    base_dir = root if root.name == mode or root.name.startswith(f"{mode}_") else root / mode
    method_dir = base_dir / method
    all_rows = []
    issues = []
    shard_counts = {}
    bad_json_lines = {}
    for shard_id in range(num_shards):
        path = method_dir / f"shard_{shard_id}.jsonl"
        rows, bad = read_jsonl(path)
        shard_counts[str(shard_id)] = len(rows)
        bad_json_lines[str(shard_id)] = bad
        if not path.exists():
            issues.append(f"missing_shard_file:{method}/shard_{shard_id}.jsonl")
        if bad:
            issues.append(f"bad_json_lines:{method}/shard_{shard_id}:{bad}")
        all_rows.extend(rows)

    by_index: dict[int, dict] = {}
    duplicates = []
    for row in all_rows:
        idx = row.get("sample_index")
        if not isinstance(idx, int):
            issues.append(f"invalid_sample_index:{row.get('sample_id')}")
            continue
        if idx in by_index:
            duplicates.append(idx)
        by_index[idx] = row
    if duplicates:
        issues.append(f"duplicate_sample_index:{method}:{sorted(set(duplicates))[:20]}")
    expected_set = set(range(expected_samples))
    present = set(by_index)
    missing = sorted(expected_set - present)
    extra = sorted(present - expected_set)
    if missing:
        issues.append(f"missing_sample_index:{method}:count={len(missing)} first={missing[:20]}")
    if extra:
        issues.append(f"extra_sample_index:{method}:count={len(extra)} first={extra[:20]}")

    rows = [by_index[i] for i in sorted(by_index)]
    errors = [r for r in rows if r.get("error")]
    empty_predictions = [r for r in rows if not str(r.get("prediction") or "").strip()]
    parser_failures = [r for r in rows if r.get("parser_failure")]
    truncated = [r for r in rows if r.get("length_truncated")]
    if errors:
        issues.append(f"errors:{method}:count={len(errors)}")
    if empty_predictions:
        issues.append(f"empty_prediction:{method}:count={len(empty_predictions)}")
    if len(rows) != expected_samples:
        issues.append(f"wrong_total_rows:{method}:{len(rows)}!=expected:{expected_samples}")

    if method == "kivi":
        bad_kivi = []
        for r in rows:
            ev = r.get("kivi_runtime_evidence") or {}
            if ev.get("axis_key") != 1 or ev.get("axis_value") != 0:
                bad_kivi.append(r.get("sample_index"))
        if bad_kivi:
            issues.append(f"kivi_axis_mismatch:count={len(bad_kivi)} first={bad_kivi[:20]}")
    if method == "patternkv":
        evidence_rows = []
        for r in rows:
            ev = r.get("patternkv_runtime_evidence") or {}
            if ev.get("packed_k_exists") and ev.get("packed_v_exists") and ev.get("v_mask_exists") and ev.get("k_assignment_exists"):
                evidence_rows.append(r)
        if rows and not evidence_rows:
            issues.append("patternkv_evidence_missing_or_unpacked")

    correct = sum(1 for r in rows if bool(r.get("correct")))
    output_tokens = [float(r.get("output_tokens") or 0) for r in rows]
    input_tokens = [float(r.get("input_tokens") or 0) for r in rows]
    latencies = [float(r.get("latency_s") or 0) for r in rows if r.get("latency_s") is not None]
    peak_reserved = [float(r.get("peak_reserved_bytes") or 0) / (1024**3) for r in rows]
    ended_with_eos = [r for r in rows if r.get("ended_with_eos")]
    merged_path = method_dir / "all.jsonl"
    if rows:
        write_jsonl(merged_path, rows)
    return {
        "method": method,
        "rows": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 6) if rows else None,
        "accuracy_percent": round(100.0 * correct / len(rows), 3) if rows else None,
        "parser_failures": len(parser_failures),
        "length_truncated": len(truncated),
        "errors": len(errors),
        "oom_errors": sum(1 for r in errors if r.get("error_type") == "OOM"),
        "empty_predictions": len(empty_predictions),
        "avg_input_tokens": round(statistics.mean(input_tokens), 2) if input_tokens else None,
        "avg_output_tokens": round(statistics.mean(output_tokens), 2) if output_tokens else None,
        "p50_output_tokens": round(percentile(output_tokens, 0.5), 2) if output_tokens else None,
        "p95_output_tokens": round(percentile(output_tokens, 0.95), 2) if output_tokens else None,
        "max_output_tokens": int(max(output_tokens)) if output_tokens else None,
        "avg_latency_s": round(statistics.mean(latencies), 4) if latencies else None,
        "peak_reserved_gb": round(max(peak_reserved), 3) if peak_reserved else None,
        "normal_eos_rate": round(len(ended_with_eos) / len(rows), 6) if rows else None,
        "normal_eos_rate_percent": round(100.0 * len(ended_with_eos) / len(rows), 3) if rows else None,
        "shard_counts": shard_counts,
        "bad_json_lines": bad_json_lines,
        "merged_path": str(merged_path) if rows else None,
        "issues": issues,
        "rows_by_index": by_index,
    }


def build_report(summary: dict[str, Any]) -> str:
    lines = []
    lines.append(f"# GSM8K {summary['mode']} Summary")
    lines.append("")
    lines.append(f"- created_at: `{summary['created_at']}`")
    lines.append(f"- expected_samples_per_method: `{summary['expected_samples']}`")
    lines.append(f"- status: `{'PASS' if summary['pass'] else 'PARTIAL'}`")
    lines.append("")
    lines.append("| method | rows | correct | accuracy | retention_vs_fp16 | delta_vs_fp16 | parser_fail | truncated | errors | peak_reserved_gb |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in summary["methods"]:
        lines.append(
            f"| {m['method']} | {m['rows']} | {m['correct']} | "
            f"{m['accuracy_percent'] if m['accuracy_percent'] is not None else 'NA'} | "
            f"{m.get('retention_vs_fp16_percent', 'NA')} | {m.get('delta_vs_fp16_points', 'NA')} | "
            f"{m['parser_failures']} | {m['length_truncated']} | {m['errors']} | {m['peak_reserved_gb']} |"
        )
    lines.append("")
    if summary.get("pairwise_patternkv_kivi"):
        p = summary["pairwise_patternkv_kivi"]
        lines.append("## PatternKV vs KIVI Paired")
        lines.append("")
        lines.append(f"- common_samples: `{p['common_samples']}`")
        lines.append(f"- both_correct: `{p['both_correct']}`")
        lines.append(f"- both_wrong: `{p['both_wrong']}`")
        lines.append(f"- only_patternkv_correct: `{p['only_patternkv_correct']}`")
        lines.append(f"- only_kivi_correct: `{p['only_kivi_correct']}`")
        lines.append(f"- mcnemar_exact_p: `{p['mcnemar_exact_p']}`")
        lines.append(f"- bootstrap_delta_accuracy_points_ci95: `{p['bootstrap_delta_accuracy_points_ci95']}`")
        lines.append("")
    if summary["issues"]:
        lines.append("## Issues")
        lines.append("")
        for issue in summary["issues"]:
            lines.append(f"- `{issue}`")
    else:
        lines.append("No integrity issues detected.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True)
    p.add_argument("--results-dir", type=Path, default=Path("results/gsm8k"))
    p.add_argument("--expected-samples", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--report-md", type=Path)
    p.add_argument("--report-json", type=Path)
    p.add_argument("--report-csv", type=Path)
    p.add_argument("--final-status-md", type=Path)
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args()

    methods = [summarize_method(args.results_dir, args.mode, method, args.expected_samples, args.num_shards) for method in args.methods]
    fp16 = next((m for m in methods if m["method"] == "fp16"), None)
    if fp16 and fp16["accuracy"] not in (None, 0):
        for method in methods:
            if method["accuracy"] is not None:
                method["retention_vs_fp16_percent"] = round(100.0 * method["accuracy"] / fp16["accuracy"], 3)
                method["delta_vs_fp16_points"] = round(100.0 * (method["accuracy"] - fp16["accuracy"]), 3)

    pairwise = None
    kivi = next((m for m in methods if m["method"] == "kivi"), None)
    patternkv = next((m for m in methods if m["method"] == "patternkv"), None)
    if kivi and patternkv:
        k_rows = kivi["rows_by_index"]
        p_rows = patternkv["rows_by_index"]
        common = sorted(set(k_rows) & set(p_rows))
        both_correct = both_wrong = only_p = only_k = 0
        for idx in common:
            pc = bool(p_rows[idx].get("correct"))
            kc = bool(k_rows[idx].get("correct"))
            both_correct += int(pc and kc)
            both_wrong += int((not pc) and (not kc))
            only_p += int(pc and not kc)
            only_k += int(kc and not pc)
        pairwise = {
            "common_samples": len(common),
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "only_patternkv_correct": only_p,
            "only_kivi_correct": only_k,
            "mcnemar_exact_p": None if not common else round(exact_mcnemar_p(only_p, only_k), 6) if exact_mcnemar_p(only_p, only_k) is not None else None,
            "bootstrap_delta_accuracy_points_ci95": bootstrap_ci(p_rows, k_rows) if common else None,
        }

    issues = []
    for method in methods:
        issues.extend(method["issues"])
    complete = not issues
    public_methods = []
    for method in methods:
        item = {k: v for k, v in method.items() if k != "rows_by_index"}
        public_methods.append(item)
    summary = {
        "created_at": utc_now(),
        "mode": args.mode,
        "expected_samples": args.expected_samples,
        "num_shards": args.num_shards,
        "methods": public_methods,
        "pairwise_patternkv_kivi": pairwise,
        "issues": issues,
        "pass": complete,
    }

    default_md = Path("reports") / ("gsm8k_smoke_3methods_50.md" if args.mode == "smoke" else "gsm8k_3methods_full.md")
    default_json = Path("reports") / ("gsm8k_smoke_3methods_50.json" if args.mode == "smoke" else "gsm8k_3methods_full.json")
    report_md = args.report_md or default_md
    report_json = args.report_json or default_json
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    report_md.write_text(build_report(summary), encoding="utf-8")
    if args.report_csv:
        args.report_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.report_csv.open("w", encoding="utf-8", newline="") as f:
            fields = ["method", "rows", "correct", "accuracy_percent", "retention_vs_fp16_percent", "delta_vs_fp16_points", "parser_failures", "length_truncated", "errors", "peak_reserved_gb"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for method in public_methods:
                writer.writerow({field: method.get(field) for field in fields})
    if args.final_status_md:
        label = "FULL RUN PASS" if complete else "FULL RUN PARTIAL"
        args.final_status_md.write_text(f"# GSM8K Final Status\n\n`{label}`\n\n" + build_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    if args.require_complete and not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
