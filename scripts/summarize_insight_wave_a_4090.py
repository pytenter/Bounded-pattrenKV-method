#!/usr/bin/env python
"""Summarize isolated single-4090 Wave A generation and observer outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from insight_wave_a_4090_utils import PLAN, current_commit, is_completed_generation, is_completed_observer, load_reference, plan_samples, result_path, write_json, write_text


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parsed_key(key: str) -> tuple[str, str, int, int, str, str] | None:
    parts = key.split(".")
    if len(parts) < 6 or not parts[2].startswith("layer") or not parts[3].startswith("head"):
        return None
    try:
        return parts[0], parts[1], int(parts[2][5:]), int(parts[3][4:]), parts[4], ".".join(parts[5:])
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    reference = load_reference()
    samples = plan_samples(reference)
    generation_files = []
    observer_files = []
    rows_by_task: dict[str, dict[str, Any]] = {
        task: {"planned": planned, "generation_files": 0, "observer_files": 0, "completed": 0, "failed": 0, "oom": 0, "hook_errors": 0, "observer_completed": 0, "observer_missing": 0, "max_observer_bytes": 0, "dropped_record_count": 0, "generated_tokens": 0, "wall_time_seconds": 0.0}
        for _, task, planned in PLAN
    }
    metric_store: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: {"count": 0.0, "sum": 0.0, "sum_sq": 0.0})
    confusion_store: dict[tuple[Any, ...], dict[str, int]] = defaultdict(lambda: {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0})
    errors = []
    generation_commits = set()
    observer_commits = set()
    for sample in samples:
        gen = result_path(args.result_root / "generation", sample, "oracle")
        obs = result_path(args.result_root / "observer", sample, "oracle")
        if gen.exists():
            generation_files.append(gen)
            try:
                payload = json.loads(gen.read_text())
                task = sample["task"]
                row = rows_by_task[task]
                row["generation_files"] += 1
                row["generated_tokens"] += int(payload.get("generated_tokens") or len(payload.get("generated_token_ids") or []))
                row["wall_time_seconds"] += float(payload.get("wall_time_seconds") or 0.0)
                if payload.get("git_commit"):
                    generation_commits.add(str(payload["git_commit"]))
                if is_completed_generation(gen):
                    row["completed"] += 1
                else:
                    row["failed"] += 1
                    row["oom"] += int(payload.get("stop_reason") == "oom")
                    row["hook_errors"] += int("hook" in str(payload.get("error", "")).lower())
                    errors.append({"path": str(gen), "error": payload.get("error"), "stop_reason": payload.get("stop_reason")})
            except Exception as exc:
                errors.append({"path": str(gen), "error": repr(exc)})
        if obs.exists():
            observer_files.append(obs)
            try:
                payload = json.loads(obs.read_text())
                task = sample["task"]
                row = rows_by_task[task]
                row["observer_files"] += 1
                row["observer_completed"] += int(is_completed_observer(obs))
                row["observer_missing"] = max(row["planned"] - row["observer_completed"], 0)
                row["max_observer_bytes"] = max(row["max_observer_bytes"], int(payload.get("estimated_serialized_bytes") or obs.stat().st_size))
                row["dropped_record_count"] += int(payload.get("dropped_record_count") or 0)
                metadata = payload.get("metadata") or {}
                if metadata.get("git_commit"):
                    observer_commits.add(str(metadata["git_commit"]))
                for key, value in (payload.get("aggregates") or {}).items():
                    parsed = parsed_key(key)
                    if parsed is None or not isinstance(value, dict):
                        continue
                    count = float(value.get("count") or 0)
                    if count <= 0:
                        continue
                    k = (task, *parsed)
                    target = metric_store[k]
                    target["count"] += count
                    target["sum"] += float(value.get("sum") or 0)
                    target["sum_sq"] += float(value.get("sum_sq") or 0)
                for key, value in (payload.get("confusion") or {}).items():
                    parsed = parsed_key(key)
                    if parsed is None or not isinstance(value, dict):
                        continue
                    k = (task, *parsed)
                    target = confusion_store[k]
                    for name in target:
                        target[name] += int(value.get(name) or 0)
            except Exception as exc:
                errors.append({"path": str(obs), "error": repr(exc)})
    for row in rows_by_task.values():
        row["observer_missing"] = max(row["planned"] - row["observer_completed"], 0)
    completed = sum(row["completed"] for row in rows_by_task.values())
    failed = sum(row["failed"] for row in rows_by_task.values())
    observer_completed = sum(row["observer_completed"] for row in rows_by_task.values())
    status = "completed" if completed == 140 and len(generation_files) == 140 and observer_completed == 140 and failed == 0 else "partial_or_blocked"
    completion = {
        "schema_version": "insight_v2.wave_a_4090_completion",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "git_commit": current_commit(),
        "generation_git_commits": sorted(generation_commits),
        "observer_git_commits": sorted(observer_commits),
        "result_root": str(args.result_root),
        "report_root": str(args.report_root),
        "total_planned": 140,
        "generation_files": len(generation_files),
        "observer_files": len(observer_files),
        "completed": completed,
        "failed": failed,
        "oom": sum(row["oom"] for row in rows_by_task.values()),
        "hook_errors": sum(row["hook_errors"] for row in rows_by_task.values()),
        "observer_completed": observer_completed,
        "observer_missing": sum(row["observer_missing"] for row in rows_by_task.values()),
        "task_counts": rows_by_task,
        "generation_errors": errors[:100],
        "csv_rows": {},
    }
    args.report_root.mkdir(parents=True, exist_ok=True)
    write_json(args.report_root / "completion.json", completion)
    fields = ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric", "count", "mean", "std"]
    metric_rows = []
    for key, value in sorted(metric_store.items()):
        count = value["count"]
        mean = value["sum"] / count if count else None
        variance = max(value["sum_sq"] / count - mean * mean, 0) if count and mean is not None else None
        task, phase, kv_type, layer, head, bucket, metric = key
        metric_rows.append({"task": task, "phase": phase, "kv_type": kv_type, "layer": layer, "kv_head": head, "bucket": bucket, "metric": metric, "count": int(count), "mean": mean, "std": math.sqrt(variance) if variance is not None else None})
    gain = [row for row in metric_rows if row["metric"] in {"relative_benefit", "relative_mse_gain", "relative_candidate_benefit", "range_contraction", "relative_range_gain"}]
    gap = [row for row in metric_rows if "oracle_gap" in str(row["metric"]) or "oracle" in str(row["metric"])]
    dynamic = [row for row in metric_rows if row["metric"] in {"candidate_gate_accepted_fraction", "dead_pattern_fraction", "top1_pattern_share", "top4_pattern_share", "normalized_entropy", "gate_acceptance"}]
    gate = []
    for key, value in sorted(confusion_store.items()):
        task, phase, kv_type, layer, head, bucket, metric = key
        total = sum(value.values())
        gate.append({"task": task, "phase": phase, "kv_type": kv_type, "layer": layer, "kv_head": head, "bucket": bucket, "metric": metric, **value, "total": total, "false_positive_rate": value["false_positive"] / (value["false_positive"] + value["true_negative"]) if value["false_positive"] + value["true_negative"] else None, "false_negative_rate": value["false_negative"] / (value["false_negative"] + value["true_positive"]) if value["false_negative"] + value["true_positive"] else None})
    csv_fields = fields
    write_csv(args.report_root / "pattern_gain_map.csv", gain, csv_fields)
    write_csv(args.report_root / "matching_oracle_gap.csv", gap, csv_fields)
    write_csv(args.report_root / "dynamic_pattern_utility.csv", dynamic, csv_fields)
    write_csv(args.report_root / "v_gate_confusion.csv", gate, ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric", "true_positive", "true_negative", "false_positive", "false_negative", "total", "false_positive_rate", "false_negative_rate"])
    completion["csv_rows"] = {"pattern_gain_map": len(gain), "matching_oracle_gap": len(gap), "v_gate_confusion": len(gate), "dynamic_pattern_utility": len(dynamic)}
    write_json(args.report_root / "completion.json", completion)
    lines = ["# Insight Wave A 4090 Completion", "", f"Status: `{status}`", f"Total planned: `{completion['total_planned']}`", f"Completed: `{completed}`", f"Generation files: `{len(generation_files)}`", f"Observer completed: `{observer_completed}`", f"Failed: `{failed}`", f"OOM: `{completion['oom']}`", f"Hook errors: `{completion['hook_errors']}`", f"Observer missing: `{completion['observer_missing']}`", "", "| task | planned | completed | failed | observer completed | observer missing | OOM | hook errors | max observer bytes |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task, row in rows_by_task.items():
        lines.append(f"| {task} | {row['planned']} | {row['completed']} | {row['failed']} | {row['observer_completed']} | {row['observer_missing']} | {row['oom']} | {row['hook_errors']} | {row['max_observer_bytes']} |")
    lines += ["", "This is a collection-completeness report; it is not by itself a cross-hardware algorithm conclusion."]
    write_text(args.report_root / "completion.md", "\n".join(lines) + "\n")
    write_text(args.report_root / "summary.md", "\n".join(lines) + "\n")
    write_text(args.report_root / "observer_overhead.md", f"# Observer Overhead\n\n- observer files: `{len(observer_files)}`\n- max serialized bytes: `{max((row['max_observer_bytes'] for row in rows_by_task.values()), default=0)}`\n- dropped records: `{sum(row['dropped_record_count'] for row in rows_by_task.values())}`\n")


if __name__ == "__main__":
    main()
