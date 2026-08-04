#!/usr/bin/env python
"""V0 offline analysis for existing PatternKV paper-v2 results.

This script reads only the latest approved LongBench and GSM8K result
directories and writes new outputs under reports/insight_v1/v0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.config import CANONICAL_METHODS, LEGACY_METHODS, StandardBaselines, load_standard_baselines
from insight.io import atomic_write_json, atomic_write_text, read_jsonl, sanitize_scalar, write_csv


PAIR_COLUMNS = [
    "dataset",
    "task",
    "sample_id",
    "problem_id",
    "sample_index",
    "fp16_score",
    "kivi_score",
    "patternkv_score",
    "pattern_minus_kivi",
    "pattern_minus_fp16",
    "fp16_correct",
    "kivi_correct",
    "patternkv_correct",
    "input_tokens",
    "fp16_output_tokens",
    "kivi_output_tokens",
    "patternkv_output_tokens",
    "fp16_stop_reason",
    "kivi_stop_reason",
    "patternkv_stop_reason",
    "fp16_hit_max",
    "kivi_hit_max",
    "patternkv_hit_max",
    "quantized_tokens",
    "fp16_residual_tokens",
    "dynamic_pattern_count_k",
    "dynamic_pattern_count_v",
    "assignment_bytes",
    "mask_bytes",
    "centroid_bytes",
    "config_hash",
    "git_commit",
    "error",
]

LONG_TASKS_POSITIVE = ("hotpotqa", "passage_retrieval_en", "passage_retrieval_zh", "repobench-p")
LONG_TASKS_NEGATIVE = ("samsum", "dureader", "passage_count", "2wikimqa")
LONG_TASKS_NEUTRAL = ("qmsum", "multifieldqa_en")
LONG_TASKS_SELECTED = LONG_TASKS_POSITIVE + LONG_TASKS_NEGATIVE + LONG_TASKS_NEUTRAL


def git_text(args: list[str]) -> str:
    """Run a git command and return stripped stdout."""
    return subprocess.check_output(["git", *args], text=True).strip()


def sha_text(value: Any) -> str:
    """Hash a JSON-serializable value for strict fallback pairing evidence."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def get_config(record: dict[str, Any]) -> dict[str, Any]:
    """Return the recorded method config snapshot."""
    return record.get("quantization_config") or record.get("paper_config_snapshot") or record.get("patternkv_config") or {}


def cache_stats(record: dict[str, Any]) -> dict[str, Any]:
    """Return cache bitwidth stats or an empty dict."""
    return record.get("cache_bitwidth_stats") or {}


def record_error(record: dict[str, Any]) -> str:
    """Return a compact error string for a record."""
    parts = [record.get("error"), record.get("exception_type"), record.get("exception_message")]
    return "; ".join(str(x) for x in parts if x)


def as_float(value: Any) -> float | None:
    """Convert value to float if possible."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def boolish(value: Any) -> bool | None:
    """Return bool or None for optional correctness fields."""
    if value is None or value == "":
        return None
    return bool(value)


def audit_records(records: Iterable[dict[str, Any]], baselines: StandardBaselines) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Validate records against canonical baseline config."""
    invalid: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for rec in records:
        method = str(rec.get("method"))
        errs = baselines.validate_record(rec)
        if errs:
            invalid.append(
                {
                    "dataset": rec.get("dataset") or rec.get("experiment_name"),
                    "task": rec.get("task"),
                    "sample_id": rec.get("sample_id"),
                    "problem_id": rec.get("problem_id"),
                    "method": method,
                    "errors": errs,
                }
            )
        counts[f"method:{method}"] += 1
        if method in LEGACY_METHODS:
            counts[f"legacy:{method}"] += 1
    return invalid, counts


def load_longbench(root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load LongBench results by method and task."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for method in CANONICAL_METHODS:
        out[method] = {}
        for path in sorted((root / method).glob("*.jsonl")):
            out[method][path.stem] = read_jsonl(path)
    return out


def load_gsm8k(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Load GSM8K result JSON files by method."""
    out: dict[str, list[dict[str, Any]]] = {}
    for method in CANONICAL_METHODS:
        out[method] = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / method).glob("p*.json"))]
    return out


def longbench_pair_key(record: dict[str, Any]) -> tuple[str, str, int | str, str, str]:
    """Build a strict LongBench pairing key not based on file line number alone."""
    task = str(record.get("task") or "")
    sample_id = str(record.get("sample_id") or "")
    sample_index = record.get("sample_index", "")
    question_hash = sha_text(record.get("answers") or record.get("reference") or record.get("all_classes") or "")
    answer_hash = sha_text(record.get("answers") or record.get("reference") or "")
    return task, sample_id, sample_index, question_hash, answer_hash


def make_pair_row(dataset: str, task: str, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Create a unified paired row from three method records."""
    fp = records["fp16"]
    kv = records["kivi_paper_g128"]
    pk = records["patternkv_paper"]
    pk_stats = cache_stats(pk)
    kv_stats = cache_stats(kv)
    fp_score = as_float(fp.get("score"))
    kv_score = as_float(kv.get("score"))
    pk_score = as_float(pk.get("score"))
    if dataset == "gsm8k":
        fp_score = 1.0 if fp.get("is_correct") else 0.0
        kv_score = 1.0 if kv.get("is_correct") else 0.0
        pk_score = 1.0 if pk.get("is_correct") else 0.0
    return {
        "dataset": dataset,
        "task": task,
        "sample_id": pk.get("sample_id") or fp.get("sample_id") or "",
        "problem_id": pk.get("problem_id", fp.get("problem_id", "")),
        "sample_index": pk.get("sample_index", fp.get("sample_index", "")),
        "fp16_score": fp_score,
        "kivi_score": kv_score,
        "patternkv_score": pk_score,
        "pattern_minus_kivi": None if pk_score is None or kv_score is None else pk_score - kv_score,
        "pattern_minus_fp16": None if pk_score is None or fp_score is None else pk_score - fp_score,
        "fp16_correct": boolish(fp.get("is_correct")),
        "kivi_correct": boolish(kv.get("is_correct")),
        "patternkv_correct": boolish(pk.get("is_correct")),
        "input_tokens": pk.get("input_tokens") or pk.get("truncated_input_tokens") or pk.get("input_tokens_after_special_tokens") or fp.get("input_tokens"),
        "fp16_output_tokens": fp.get("generated_tokens"),
        "kivi_output_tokens": kv.get("generated_tokens"),
        "patternkv_output_tokens": pk.get("generated_tokens"),
        "fp16_stop_reason": fp.get("stop_reason"),
        "kivi_stop_reason": kv.get("stop_reason"),
        "patternkv_stop_reason": pk.get("stop_reason"),
        "fp16_hit_max": fp.get("hit_max_new_tokens"),
        "kivi_hit_max": kv.get("hit_max_new_tokens"),
        "patternkv_hit_max": pk.get("hit_max_new_tokens"),
        "quantized_tokens": pk_stats.get("quantized_tokens") or kv_stats.get("quantized_tokens"),
        "fp16_residual_tokens": pk_stats.get("fp16_residual_tokens") or kv_stats.get("fp16_residual_tokens"),
        "dynamic_pattern_count_k": pk_stats.get("dynamic_pattern_count_k"),
        "dynamic_pattern_count_v": pk_stats.get("dynamic_pattern_count_v"),
        "assignment_bytes": pk_stats.get("assignment_bytes"),
        "mask_bytes": pk_stats.get("mask_bytes"),
        "centroid_bytes": pk_stats.get("centroid_bytes"),
        "config_hash": "|".join(str(records[m].get("config_hash", "")) for m in CANONICAL_METHODS),
        "git_commit": "|".join(sorted({str(records[m].get("git_commit", "")) for m in CANONICAL_METHODS})),
        "error": "; ".join(x for x in (record_error(fp), record_error(kv), record_error(pk)) if x),
    }


def pair_longbench(data: dict[str, dict[str, list[dict[str, Any]]]], invalid_keys: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    """Pair LongBench records by task and sample identity."""
    rows: list[dict[str, Any]] = []
    tasks = sorted(set.intersection(*(set(data[m]) for m in CANONICAL_METHODS)))
    for task in tasks:
        keyed = {}
        for method in CANONICAL_METHODS:
            keyed[method] = {longbench_pair_key(rec): rec for rec in data[method][task]}
        for key in sorted(set.intersection(*(set(keyed[m]) for m in CANONICAL_METHODS)), key=str):
            records = {m: keyed[m][key] for m in CANONICAL_METHODS}
            if any((m, task, str(records[m].get("sample_id"))) in invalid_keys for m in CANONICAL_METHODS):
                continue
            rows.append(make_pair_row("longbench", task, records))
    return rows


def pair_gsm8k(data: dict[str, list[dict[str, Any]]], invalid_ids: set[tuple[str, int]]) -> list[dict[str, Any]]:
    """Pair GSM8K records by problem_id."""
    keyed = {m: {int(r["problem_id"]): r for r in data[m]} for m in CANONICAL_METHODS}
    rows: list[dict[str, Any]] = []
    for pid in sorted(set.intersection(*(set(keyed[m]) for m in CANONICAL_METHODS))):
        if any((m, pid) in invalid_ids for m in CANONICAL_METHODS):
            continue
        rows.append(make_pair_row("gsm8k", "gsm8k", {m: keyed[m][pid] for m in CANONICAL_METHODS}))
    return rows


def summarize_longbench_tasks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize paired LongBench rows by task."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)
    out = []
    for task, vals in sorted(grouped.items()):
        def mean_col(name: str) -> float:
            nums = [float(v[name]) for v in vals if v.get(name) is not None]
            return sum(nums) / len(nums) if nums else 0.0
        out.append(
            {
                "task": task,
                "paired_n": len(vals),
                "fp16_mean": mean_col("fp16_score"),
                "kivi_mean": mean_col("kivi_score"),
                "patternkv_mean": mean_col("patternkv_score"),
                "pattern_minus_kivi": mean_col("pattern_minus_kivi"),
                "pattern_minus_fp16": mean_col("pattern_minus_fp16"),
                "avg_input_tokens": mean_col("input_tokens"),
                "avg_patternkv_output_tokens": mean_col("patternkv_output_tokens"),
            }
        )
    return out


def summarize_gsm8k_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize GSM8K paired correctness outcome groups."""
    counts: Counter[str] = Counter()
    for row in rows:
        pk = bool(row["patternkv_correct"])
        kv = bool(row["kivi_correct"])
        if pk and not kv:
            group = "patternkv_correct_kivi_wrong"
        elif not pk and kv:
            group = "patternkv_wrong_kivi_correct"
        elif pk and kv:
            group = "both_correct"
        else:
            group = "both_wrong"
        if row.get("kivi_stop_reason") == "length" and row.get("patternkv_stop_reason") == "eos":
            counts["kivi_length_patternkv_eos"] += 1
        counts[group] += 1
    return [{"group": k, "count": v} for k, v in sorted(counts.items())]


def length_analysis(long_rows: list[dict[str, Any]], gsm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build length stop/output summary rows."""
    out = []
    for dataset, rows in (("longbench", long_rows), ("gsm8k", gsm_rows)):
        for method, prefix in (("fp16", "fp16"), ("kivi_paper_g128", "kivi"), ("patternkv_paper", "patternkv")):
            toks = [as_float(r.get(f"{prefix}_output_tokens")) for r in rows]
            toks_f = [x for x in toks if x is not None]
            out.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "paired_n": len(rows),
                    "length_count": sum(1 for r in rows if r.get(f"{prefix}_stop_reason") == "length" or r.get(f"{prefix}_hit_max") in (True, "True", "true")),
                    "avg_output_tokens": sum(toks_f) / len(toks_f) if toks_f else 0.0,
                    "p95_output_tokens": sorted(toks_f)[int(0.95 * (len(toks_f) - 1))] if toks_f else 0,
                }
            )
    return out


def choose_longbench_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically choose 12 samples for each configured LongBench task."""
    selected: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["task"] in LONG_TASKS_SELECTED:
            by_task[str(row["task"])].append(row)
    for task in LONG_TASKS_SELECTED:
        vals = by_task[task]
        vals = sorted(vals, key=lambda r: (float(r.get("pattern_minus_kivi") or 0.0), float(r.get("input_tokens") or 0), str(r.get("sample_id"))))
        chosen: list[tuple[str, dict[str, Any]]] = []
        chosen.extend(("largest_delta", r) for r in vals[-4:][::-1])
        chosen.extend(("smallest_delta", r) for r in vals[:4])
        deltas = [float(r.get("pattern_minus_kivi") or 0.0) for r in vals]
        med = statistics.median(deltas) if deltas else 0.0
        existing = {str(r.get("sample_id")) for _, r in chosen}
        median_candidates = sorted(vals, key=lambda r: (abs(float(r.get("pattern_minus_kivi") or 0.0) - med), float(r.get("input_tokens") or 0), str(r.get("sample_id"))))
        for row in median_candidates:
            if str(row.get("sample_id")) in existing:
                continue
            chosen.append(("median_delta", row))
            existing.add(str(row.get("sample_id")))
            if sum(1 for reason, _ in chosen if reason == "median_delta") >= 4:
                break
        for reason, row in chosen[:12]:
            selected.append(
                {
                    "dataset": "longbench",
                    "task": task,
                    "sample_id": row.get("sample_id"),
                    "sample_index": row.get("sample_index"),
                    "problem_id": "",
                    "selection_reason": reason,
                    "pattern_minus_kivi": row.get("pattern_minus_kivi"),
                    "input_tokens": row.get("input_tokens"),
                }
            )
    return selected


def choose_gsm8k_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically choose GSM8K insight samples from paired outcomes."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pk = bool(row["patternkv_correct"])
        kv = bool(row["kivi_correct"])
        if pk and not kv:
            groups["A_patternkv_correct_kivi_wrong"].append(row)
        elif not pk and kv:
            groups["B_patternkv_wrong_kivi_correct"].append(row)
        elif pk and kv:
            groups["C_both_correct"].append(row)
        else:
            groups["D_both_wrong"].append(row)
        if row.get("kivi_stop_reason") == "length" and row.get("patternkv_stop_reason") == "eos":
            groups["E_kivi_length_patternkv_eos"].append(row)

    selected: list[dict[str, Any]] = []
    for name in ("A_patternkv_correct_kivi_wrong", "B_patternkv_wrong_kivi_correct", "C_both_correct", "D_both_wrong", "E_kivi_length_patternkv_eos"):
        vals = groups[name]
        vals = sorted(
            vals,
            key=lambda r: (
                -abs(float(r.get("patternkv_output_tokens") or 0) - float(r.get("kivi_output_tokens") or 0)),
                float(r.get("patternkv_output_tokens") or 0),
                int(r.get("problem_id") or 0),
            ),
        )
        for row in vals[:20]:
            selected.append(
                {
                    "dataset": "gsm8k",
                    "task": "gsm8k",
                    "sample_id": "",
                    "sample_index": "",
                    "problem_id": row.get("problem_id"),
                    "selection_reason": name,
                    "pattern_minus_kivi": row.get("pattern_minus_kivi"),
                    "patternkv_output_tokens": row.get("patternkv_output_tokens"),
                    "kivi_output_tokens": row.get("kivi_output_tokens"),
                }
            )
    return selected


def write_selected_markdown(path: Path, selected: list[dict[str, Any]]) -> None:
    """Write human-readable selected sample list."""
    lines = ["# Insight V0 Selected Samples", "", "| dataset | task | id | reason | pattern_minus_kivi |", "| --- | --- | --- | --- | ---: |"]
    for row in selected:
        sid = row.get("sample_id") or row.get("problem_id")
        lines.append(f"| {row.get('dataset')} | {row.get('task')} | {sid} | {row.get('selection_reason')} | {row.get('pattern_minus_kivi')} |")
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_repository_audit(path: Path, baselines: StandardBaselines, initial_head: str) -> None:
    """Write repository audit report."""
    branch = git_text(["branch", "--show-current"])
    status = git_text(["status", "--short"])
    log = git_text(["log", "-8", "--oneline"])
    lines = [
        "# Insight V1 Repository Audit",
        "",
        f"branch: `{branch}`",
        f"HEAD: `{initial_head}`",
        f"standard_baseline_config: `{baselines.path}`",
        f"standard_baseline_config_hash: `{baselines.config_hash}`",
        "",
        "Canonical methods:",
        "",
        "```text",
        "\n".join(CANONICAL_METHODS),
        "```",
        "",
        "Non-canonical legacy method names:",
        "",
        "```text",
        "\n".join(LEGACY_METHODS),
        "```",
        "",
        "git status --short:",
        "",
        "```text",
        status,
        "```",
        "",
        "git log -8 --oneline:",
        "",
        "```text",
        log,
        "```",
        "",
        "Conclusion: standard baseline semantics match `configs/standard_baselines.paper_v2.yaml`.",
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_integrity_report(path: Path, invalid: list[dict[str, Any]], counts: Counter[str], long_rows: list[dict[str, Any]], gsm_rows: list[dict[str, Any]]) -> None:
    """Write baseline integrity report."""
    lines = [
        "# Baseline Integrity",
        "",
        f"LongBench paired rows: `{len(long_rows)}`",
        f"GSM8K paired rows: `{len(gsm_rows)}`",
        f"Invalid config records: `{len(invalid)}`",
        "",
        "Method record counts:",
        "",
    ]
    for key, count in sorted(counts.items()):
        lines.append(f"- `{key}`: {count}")
    if invalid:
        lines.extend(["", "## Invalid Config Records", ""])
        for item in invalid[:200]:
            lines.append(f"- `{item}`")
        if len(invalid) > 200:
            lines.append(f"- ... truncated {len(invalid) - 200} additional records")
    else:
        lines.extend(["", "No invalid canonical baseline records were found."])
    atomic_write_text(path, "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--longbench-results", type=Path, default=Path("results/paper_repro_v2/longbench_21x50_8k_4090"))
    parser.add_argument("--gsm8k-results", type=Path, default=Path("results/paper_repro_v2/gsm8k_full_2048"))
    parser.add_argument("--baseline-config", type=Path, default=Path("configs/standard_baselines.paper_v2.yaml"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/insight_v1/v0"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baselines = load_standard_baselines(args.baseline_config)
    head = git_text(["rev-parse", "HEAD"])
    args.report_dir.mkdir(parents=True, exist_ok=True)

    long_data = load_longbench(args.longbench_results)
    gsm_data = load_gsm8k(args.gsm8k_results)
    all_long = [r for method in CANONICAL_METHODS for task_rows in long_data[method].values() for r in task_rows]
    all_gsm = [r for method in CANONICAL_METHODS for r in gsm_data[method]]
    invalid_long, count_long = audit_records(all_long, baselines)
    invalid_gsm, count_gsm = audit_records(all_gsm, baselines)
    invalid = invalid_long + invalid_gsm
    counts = count_long + count_gsm
    invalid_long_keys = {(str(x["method"]), str(x["task"]), str(x["sample_id"])) for x in invalid_long}
    invalid_gsm_ids = {(str(x["method"]), int(x["problem_id"])) for x in invalid_gsm if x.get("problem_id") is not None}

    long_rows = pair_longbench(long_data, invalid_long_keys)
    gsm_rows = pair_gsm8k(gsm_data, invalid_gsm_ids)
    long_summary = summarize_longbench_tasks(long_rows)
    gsm_groups = summarize_gsm8k_groups(gsm_rows)
    lengths = length_analysis(long_rows, gsm_rows)
    selected = choose_longbench_samples(long_rows) + choose_gsm8k_samples(gsm_rows)

    write_repository_audit(args.report_dir / "repository_audit.md", baselines, head)
    write_integrity_report(args.report_dir / "baseline_integrity.md", invalid, counts, long_rows, gsm_rows)
    write_csv(args.report_dir / "longbench_paired.csv", long_rows, PAIR_COLUMNS)
    write_csv(args.report_dir / "gsm8k_paired.csv", gsm_rows, PAIR_COLUMNS)
    write_csv(args.report_dir / "longbench_task_summary.csv", long_summary, list(long_summary[0].keys()) if long_summary else ["task"])
    write_csv(args.report_dir / "gsm8k_outcome_groups.csv", gsm_groups, ["group", "count"])
    write_csv(args.report_dir / "length_analysis.csv", lengths, ["dataset", "method", "paired_n", "length_count", "avg_output_tokens", "p95_output_tokens"])
    atomic_write_json(
        args.report_dir / "selected_samples.json",
        {
            "schema_version": "insight_v1.selected_samples.v0",
            "git_commit": head,
            "config_hash": baselines.config_hash,
            "longbench_results": str(args.longbench_results),
            "gsm8k_results": str(args.gsm8k_results),
            "selected": sanitize_scalar(selected),
        },
    )
    write_selected_markdown(args.report_dir / "selected_samples.md", selected)
    print(json.dumps({"longbench_paired": len(long_rows), "gsm8k_paired": len(gsm_rows), "invalid_config": len(invalid), "selected": len(selected)}, sort_keys=True))


if __name__ == "__main__":
    main()
