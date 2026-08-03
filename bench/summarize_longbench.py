import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench._longbench_scorer import score_subtask
from bench.longbench_config import METRIC_NAMES


TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "trec",
    "passage_retrieval_en",
    "lcc",
)
DEFAULT_METHODS = ("fp16", "patternkv")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def has_nan_inf(value) -> bool:
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, dict):
        return any(has_nan_inf(v) for v in value.values())
    if isinstance(value, list):
        return any(has_nan_inf(v) for v in value)
    return False


def score_rows(task: str, rows: list[dict]) -> float:
    preds = [str(row.get("prediction") or "") for row in rows]
    refs = [list(row.get("answers") or []) for row in rows]
    all_classes = None
    for row in rows:
        if row.get("all_classes"):
            all_classes = list(row["all_classes"])
            break
    return score_subtask(task, preds, refs, all_classes=all_classes)["score"] if rows else math.nan


def patternkv_bit_accounting(rows: list[dict]) -> dict:
    evidences = [row.get("patternkv_runtime_evidence") for row in rows if row.get("patternkv_runtime_evidence")]
    mask_vals = [ev.get("v_mask_mean") for ev in evidences if ev.get("v_mask_mean") is not None]
    avg_out = sum(int(row.get("output_tokens") or 0) for row in rows) / len(rows) if rows else 0.0
    avg_in = sum(int(row.get("input_tokens") or 0) for row in rows) / len(rows) if rows else 0.0
    t_quant = max(avg_in + avg_out - 128, 1.0)
    current_cache = {
        "k_payload_bits_per_dim": 2.0,
        "k_scale_min_bits_per_dim": 0.25,
        "k_assignment_actual_cache_bits_per_dim": 0.5,
        "v_payload_bits_per_dim": 2.0,
        "v_scale_min_bits_per_dim": 0.25,
        "v_mask_actual_cache_bits_per_dim": 0.0625,
        "v_assignment_actual_cache_bits_per_dim": 0.5,
        "centroid_fp16_bits_per_dim_amortized_per_k_or_v": round(32 * 16 / t_quant, 4),
        "assumptions": "2-bit packed payload, group_size=128, head_dim=128, FP16 scale/min, current Python cache stores K/V assignments as torch.long and V mask as uint8.",
    }
    compact_cuda = dict(current_cache)
    compact_cuda["k_assignment_actual_cache_bits_per_dim"] = 0.125
    compact_cuda["v_assignment_actual_cache_bits_per_dim"] = 0.0625
    compact_cuda["assumptions"] = "Same payload and scale/min, but CUDA wrapper compact transfer dtypes: K assignment int16, V assignment uint8."
    return {
        "avg_v_mask_mean": round(sum(mask_vals) / len(mask_vals), 6) if mask_vals else None,
        "avg_input_tokens": round(avg_in, 2),
        "avg_output_tokens": round(avg_out, 2),
        "current_cache_layout_bits": current_cache,
        "cuda_compact_transfer_bits": compact_cuda,
    }


def summarize(args):
    base = args.results_dir
    methods = tuple(args.methods)
    detail_rows = []
    issues = []
    all_records = {}
    for method in methods:
        for task in TASKS:
            path = base / method / f"{task}.jsonl"
            rows = read_jsonl(path)
            all_records[(method, task)] = rows
            ids = [str(row.get("sample_id")) for row in rows]
            duplicate_ids = sorted({sid for sid in ids if ids.count(sid) > 1})
            errors = [row for row in rows if row.get("error")]
            empty = [row for row in rows if not str(row.get("prediction") or "").strip()]
            nan_inf = [row for row in rows if has_nan_inf(row)]
            if not path.exists():
                issues.append({"method": method, "task": task, "issue": "missing_file", "path": str(path)})
            if len(rows) != args.expected_samples:
                issues.append({"method": method, "task": task, "issue": "wrong_sample_count", "actual": len(rows), "expected": args.expected_samples})
            if duplicate_ids:
                issues.append({"method": method, "task": task, "issue": "duplicate_sample_id", "sample_ids": duplicate_ids[:20]})
            if errors:
                issues.append({"method": method, "task": task, "issue": "error_field_nonempty", "count": len(errors)})
            if empty:
                issues.append({"method": method, "task": task, "issue": "empty_prediction", "count": len(empty)})
            if nan_inf:
                issues.append({"method": method, "task": task, "issue": "nan_or_inf", "count": len(nan_inf)})
            score = score_rows(task, rows)
            detail_rows.append({
                "method": method,
                "task": task,
                "samples": len(rows),
                "expected_samples": args.expected_samples,
                "failures": len(errors),
                "empty_predictions": len(empty),
                "score": score,
                "metric": METRIC_NAMES[task],
                "avg_input_tokens": round(sum(int(row.get("input_tokens") or 0) for row in rows) / len(rows), 2) if rows else 0,
                "avg_output_tokens": round(sum(int(row.get("output_tokens") or 0) for row in rows) / len(rows), 2) if rows else 0,
                "last_mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None,
                "path": str(path),
            })
    for task in TASKS:
        reference_method = methods[0]
        ref_ids = {str(row.get("sample_id")) for row in all_records[(reference_method, task)]}
        for method in methods[1:]:
            cmp_ids = {str(row.get("sample_id")) for row in all_records[(method, task)]}
            if ref_ids == cmp_ids:
                continue
            issues.append({
                "reference_method": reference_method,
                "method": method,
                "task": task,
                "issue": "sample_id_set_mismatch",
                "reference_only": sorted(ref_ids - cmp_ids)[:20],
                "method_only": sorted(cmp_ids - ref_ids)[:20],
            })
    method_scores = {}
    for method in methods:
        scores = [row["score"] for row in detail_rows if row["method"] == method and not math.isnan(row["score"])]
        method_scores[method] = round(sum(scores) / len(scores), 4) if scores else math.nan
    baseline = methods[0]
    primary = methods[1] if len(methods) > 1 else methods[0]
    delta = method_scores[primary] - method_scores[baseline] if not any(math.isnan(method_scores[m]) for m in (baseline, primary)) else math.nan
    retention = 100.0 * method_scores[primary] / method_scores[baseline] if method_scores[baseline] and not math.isnan(delta) else math.nan
    total_records = sum(len(rows) for rows in all_records.values())
    complete = not issues and total_records == args.expected_samples * len(TASKS) * len(methods)
    if args.expected_samples == 50:
        status = "FULL RUN PASS" if complete else "FULL RUN PARTIAL"
    else:
        status = "SMOKE PASS" if complete else "SMOKE PARTIAL"
    summary = {
        "status": status,
        "expected_samples_per_task": args.expected_samples,
        "methods": list(methods),
        "baseline_method": baseline,
        "primary_method": primary,
        "expected_total_records": args.expected_samples * len(TASKS) * len(methods),
        "actual_total_records": total_records,
        "avg_normalized": method_scores,
        f"{primary}_minus_{baseline}": round(delta, 4) if not math.isnan(delta) else None,
        "quality_retention_percent": round(retention, 4) if not math.isnan(retention) else None,
        "issues": issues,
        "tasks": detail_rows,
        "patternkv_bit_accounting": patternkv_bit_accounting([row for (method, _), rows in all_records.items() if method == "patternkv" for row in rows]),
        "generated_at": datetime.now().isoformat(),
    }
    return summary


def write_outputs(summary: dict, args) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.summary_name == "summary" else f"_{args.summary_name}"
    (args.results_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (args.results_dir / f"summary{suffix}.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["method", "task", "samples", "expected_samples", "failures", "empty_predictions", "score", "metric", "avg_input_tokens", "avg_output_tokens", "last_mtime", "path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["tasks"]:
            writer.writerow(row)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PatternKV LongBench 8x50 Report",
        "",
        f"Status: {summary['status']}",
        f"Expected total records: {summary['expected_total_records']}",
        f"Actual total records: {summary['actual_total_records']}",
        f"Absolute delta ({summary['primary_method']} - {summary['baseline_method']}): {summary.get(summary['primary_method'] + '_minus_' + summary['baseline_method'])}",
        f"Quality retention percent: {summary['quality_retention_percent']}",
        "",
        "## Method Averages",
        "",
        "| method | avg_normalized |",
        "| --- | ---: |",
    ]
    for method, score in summary["avg_normalized"].items():
        lines.append(f"| {method} | {score} |")
    lines.extend([
        "",
        "## Per Task",
        "",
        "| method | task | samples | failures | empty | score | metric | avg input | avg output |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ])
    for row in summary["tasks"]:
        lines.append(f"| {row['method']} | {row['task']} | {row['samples']} | {row['failures']} | {row['empty_predictions']} | {row['score']} | {row['metric']} | {row['avg_input_tokens']} | {row['avg_output_tokens']} |")
    lines.extend(["", "## PatternKV Bit Accounting", "", "```json", json.dumps(summary["patternkv_bit_accounting"], indent=2, sort_keys=True), "```", "", "## Issues", ""])
    if summary["issues"]:
        lines.extend(["```json", json.dumps(summary["issues"], indent=2, ensure_ascii=False, sort_keys=True), "```"])
    else:
        lines.append("No integrity issues found.")
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/longbench"))
    parser.add_argument("--expected-samples", type=int, default=50)
    parser.add_argument("--report-path", type=Path, default=Path("reports/patternkv_longbench_8x50.md"))
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--summary-name", default="summary")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = summarize(args)
    write_outputs(summary, args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete and not summary["status"].endswith("PASS"):
        raise SystemExit(summary["status"])


if __name__ == "__main__":
    main()
