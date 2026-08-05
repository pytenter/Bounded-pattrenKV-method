#!/usr/bin/env python
"""Summarize Insight Wave A 8GPU generation and observer outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.io import atomic_write_json, atomic_write_text


EXPECTED_TASKS = {
    "hotpotqa": 12,
    "passage_retrieval_en": 12,
    "passage_retrieval_zh": 12,
    "samsum": 12,
    "dureader": 12,
    "gsm8k": 80,
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()
    tmp.replace(path)


def task_from_generation(payload: dict[str, Any]) -> str:
    if payload.get("dataset") == "gsm8k":
        return "gsm8k"
    return str(payload.get("task") or "unknown")


def task_from_observer(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") or {}
    if metadata.get("dataset") == "gsm8k":
        return "gsm8k"
    return str(metadata.get("task") or "unknown")


def parsed_metric_key(key: str) -> dict[str, Any] | None:
    parts = key.split(".")
    if len(parts) < 6 or parts[2][:5] != "layer" or parts[3][:4] != "head":
        return None
    try:
        layer = int(parts[2][5:])
        head = int(parts[3][4:])
    except ValueError:
        return None
    return {
        "phase": parts[0],
        "kv_type": parts[1],
        "layer": layer,
        "kv_head": head,
        "bucket": parts[4],
        "metric": ".".join(parts[5:]),
    }


def accumulate_metric(store: dict[tuple[Any, ...], dict[str, float]], key: tuple[Any, ...], value: dict[str, Any]) -> None:
    count = float(value.get("count") or 0)
    if count <= 0:
        return
    row = store[key]
    row["count"] += count
    row["sum"] += float(value.get("sum") or 0.0)
    row["sum_sq"] += float(value.get("sum_sq") or 0.0)
    if value.get("min") is not None:
        row["min"] = min(row.get("min", math.inf), float(value["min"]))
    if value.get("max") is not None:
        row["max"] = max(row.get("max", -math.inf), float(value["max"]))


def finalize_metric_rows(store: dict[tuple[Any, ...], dict[str, float]], fields: list[str]) -> list[dict[str, Any]]:
    rows = []
    for key, value in sorted(store.items()):
        count = value["count"]
        mean = value["sum"] / count if count else None
        variance = max(value["sum_sq"] / count - mean * mean, 0.0) if count and mean is not None else None
        row = {name: part for name, part in zip(fields, key)}
        row.update(
            {
                "count": int(count),
                "mean": mean,
                "min": None if value.get("min") == math.inf else value.get("min"),
                "max": None if value.get("max") == -math.inf else value.get("max"),
                "std": math.sqrt(variance) if variance is not None else None,
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("results/insight_v2/wave_a_8gpu"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/insight_v2/wave_a_8gpu"))
    args = parser.parse_args()

    generation_files = sorted((args.result_root / "generation").rglob("*.json"))
    observer_files = sorted((args.result_root / "observer").rglob("*.json"))
    generated_at = datetime.now(timezone.utc).isoformat()
    task_counts = {
        task: {
            "planned": planned,
            "generation_files": 0,
            "observer_files": 0,
            "completed": 0,
            "failed": 0,
            "oom": 0,
            "hook_errors": 0,
            "observer_completed": 0,
            "observer_missing": 0,
            "truncated_observers": 0,
            "dropped_record_count": 0,
            "max_observer_bytes": 0,
            "generated_tokens": 0,
            "wall_time_seconds": 0.0,
        }
        for task, planned in EXPECTED_TASKS.items()
    }
    task_counts["unknown"] = {**task_counts["gsm8k"], "planned": 0}
    task_counts["unknown"].update({k: 0 for k in task_counts["unknown"] if k != "planned"})

    generation_errors: list[dict[str, Any]] = []
    generation_commits: set[str] = set()
    for path in generation_files:
        try:
            payload = read_json(path)
        except Exception as exc:
            generation_errors.append({"path": str(path), "error": repr(exc)})
            continue
        task = task_from_generation(payload)
        if payload.get("git_commit"):
            generation_commits.add(str(payload["git_commit"]))
        row = task_counts.setdefault(task, {**task_counts["unknown"]})
        row["generation_files"] += 1
        error = payload.get("error")
        stop = payload.get("stop_reason")
        if error or stop in {"error", "oom"}:
            row["failed"] += 1
            row["oom"] += int(stop == "oom" or "OutOfMemory" in str(error))
            if "InsightHookError" in str(error) or "hook" in str(error).lower():
                row["hook_errors"] += 1
            generation_errors.append({"path": str(path), "task": task, "stop_reason": stop, "error": error})
        else:
            row["completed"] += 1
        row["generated_tokens"] += int(payload.get("generated_tokens") or len(payload.get("generated_token_ids") or []))
        row["wall_time_seconds"] += float(payload.get("wall_time_seconds") or 0.0)

    metric_store: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: {"count": 0.0, "sum": 0.0, "sum_sq": 0.0})
    confusion_store: dict[tuple[Any, ...], dict[str, int]] = defaultdict(lambda: {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0})
    observer_errors: list[dict[str, Any]] = []
    observer_commits: set[str] = set()
    for path in observer_files:
        try:
            payload = read_json(path)
        except Exception as exc:
            observer_errors.append({"path": str(path), "error": repr(exc)})
            continue
        task = task_from_observer(payload)
        metadata = payload.get("metadata") or {}
        if metadata.get("git_commit"):
            observer_commits.add(str(metadata["git_commit"]))
        row = task_counts.setdefault(task, {**task_counts["unknown"]})
        row["observer_files"] += 1
        if payload.get("status") == "completed":
            row["observer_completed"] += 1
        else:
            observer_errors.append({"path": str(path), "task": task, "status": payload.get("status"), "error": payload.get("error")})
        row["truncated_observers"] += int(bool(payload.get("truncated")))
        row["dropped_record_count"] += int(payload.get("dropped_record_count") or 0)
        row["max_observer_bytes"] = max(row["max_observer_bytes"], int(payload.get("estimated_serialized_bytes") or path.stat().st_size))
        for key, value in (payload.get("aggregates") or {}).items():
            parsed = parsed_metric_key(key)
            if parsed is None:
                continue
            aggregate_key = (task, parsed["phase"], parsed["kv_type"], parsed["layer"], parsed["kv_head"], parsed["bucket"], parsed["metric"])
            accumulate_metric(metric_store, aggregate_key, value)
        for key, value in (payload.get("confusion") or {}).items():
            parsed = parsed_metric_key(key)
            if parsed is None:
                continue
            aggregate_key = (task, parsed["phase"], parsed["kv_type"], parsed["layer"], parsed["kv_head"], parsed["bucket"], parsed["metric"])
            target = confusion_store[aggregate_key]
            for name in target:
                target[name] += int(value.get(name) or 0)

    for task, row in task_counts.items():
        if task == "unknown" and row["generation_files"] == 0 and row["observer_files"] == 0:
            continue
        row["observer_missing"] = max(row["planned"] - row["observer_completed"], 0)

    metric_rows = finalize_metric_rows(
        metric_store,
        ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric"],
    )
    pattern_gain_rows = [row for row in metric_rows if row["metric"] in {"relative_benefit", "relative_mse_gain", "relative_candidate_benefit", "range_contraction", "relative_range_gain"}]
    oracle_gap_rows = [row for row in metric_rows if "oracle_gap" in str(row["metric"])]
    dynamic_rows = [row for row in metric_rows if row["metric"] in {"candidate_gate_accepted_fraction", "dead_pattern_fraction", "top1_pattern_share", "top4_pattern_share", "normalized_entropy", "gate_acceptance"}]
    v_gate_rows = []
    for key, value in sorted(confusion_store.items()):
        tp = value["true_positive"]
        tn = value["true_negative"]
        fp = value["false_positive"]
        fn = value["false_negative"]
        total = tp + tn + fp + fn
        v_gate_rows.append(
            {
                "task": key[0],
                "phase": key[1],
                "kv_type": key[2],
                "layer": key[3],
                "kv_head": key[4],
                "bucket": key[5],
                "metric": key[6],
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "total": total,
                "false_positive_rate": fp / (fp + tn) if fp + tn else None,
                "false_negative_rate": fn / (fn + tp) if fn + tp else None,
            }
        )

    total_planned = sum(EXPECTED_TASKS.values())
    total_completed = sum(row["completed"] for task, row in task_counts.items() if task != "unknown")
    total_failed = sum(row["failed"] for task, row in task_counts.items() if task != "unknown")
    total_observer_completed = sum(row["observer_completed"] for task, row in task_counts.items() if task != "unknown")
    status = "completed" if total_completed == total_planned and total_failed == 0 and total_observer_completed == total_planned else "partial_or_blocked"

    completion = {
        "schema_version": "insight_v2.wave_a_8gpu_completion",
        "generated_at": generated_at,
        "git_commit": git_commit(),
        "generation_git_commits": sorted(generation_commits),
        "observer_git_commits": sorted(observer_commits),
        "status": status,
        "result_root": str(args.result_root),
        "report_root": str(args.report_root),
        "total_planned": total_planned,
        "generation_files": len(generation_files),
        "observer_files": len(observer_files),
        "completed": total_completed,
        "failed": total_failed,
        "oom": sum(row["oom"] for task, row in task_counts.items() if task != "unknown"),
        "hook_errors": sum(row["hook_errors"] for task, row in task_counts.items() if task != "unknown"),
        "observer_completed": total_observer_completed,
        "observer_missing": sum(row["observer_missing"] for task, row in task_counts.items() if task != "unknown"),
        "task_counts": {task: row for task, row in sorted(task_counts.items()) if task != "unknown"},
        "generation_errors": generation_errors[:50],
        "observer_errors": observer_errors[:50],
        "csv_rows": {
            "pattern_gain_map": len(pattern_gain_rows),
            "matching_oracle_gap": len(oracle_gap_rows),
            "v_gate_confusion": len(v_gate_rows),
            "dynamic_pattern_utility": len(dynamic_rows),
        },
    }
    args.report_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.report_root / "completion.json", completion)

    task_lines = []
    for task, row in sorted(completion["task_counts"].items()):
        avg_tokens = row["generated_tokens"] / row["completed"] if row["completed"] else 0.0
        task_lines.append(
            f"| {task} | {row['planned']} | {row['completed']} | {row['failed']} | {row['observer_completed']} | "
            f"{row['observer_missing']} | {row['oom']} | {row['hook_errors']} | {avg_tokens:.1f} | {row['max_observer_bytes']} |"
        )
    lines = [
        "# Insight Wave A 8GPU Completion",
        "",
        f"Status: `{status}`",
        f"Generated at: `{generated_at}`",
        f"Git commit: `{completion['git_commit']}`",
        f"Result root: `{args.result_root}`",
        "",
        f"Total planned: `{total_planned}`",
        f"Completed: `{total_completed}`",
        f"Failed: `{total_failed}`",
        f"OOM: `{completion['oom']}`",
        f"Hook errors: `{completion['hook_errors']}`",
        f"Observer completed: `{total_observer_completed}`",
        f"Observer missing: `{completion['observer_missing']}`",
        "",
        "| task | planned | completed | failed | observer completed | observer missing | OOM | hook errors | avg generated tokens | max observer bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *task_lines,
        "",
        "## Generated Artifacts",
        "",
        f"- `pattern_gain_map.csv`: `{len(pattern_gain_rows)}` rows",
        f"- `matching_oracle_gap.csv`: `{len(oracle_gap_rows)}` rows",
        f"- `v_gate_confusion.csv`: `{len(v_gate_rows)}` rows",
        f"- `dynamic_pattern_utility.csv`: `{len(dynamic_rows)}` rows",
        "",
        "## Interpretation Guardrail",
        "",
        "This report confirms Wave A data collection completeness. It does not by itself claim a final PatternKV research conclusion.",
    ]
    atomic_write_text(args.report_root / "completion.md", "\n".join(lines) + "\n")
    atomic_write_text(args.report_root / "summary.md", "\n".join(lines) + "\n")
    atomic_write_text(
        args.report_root / "decision_matrix.md",
        "# Insight Wave A 8GPU Decision Matrix\n\n"
        "Status: Wave A collection completed. Use the CSV artifacts for layer/head and K/V evidence before making final claims.\n\n"
        "| Evidence | Artifact | Status |\n"
        "| --- | --- | --- |\n"
        "| Pattern gain map | `pattern_gain_map.csv` | available |\n"
        "| Matching oracle gap | `matching_oracle_gap.csv` | available |\n"
        "| V gate confusion | `v_gate_confusion.csv` | available |\n"
        "| Dynamic pattern utility | `dynamic_pattern_utility.csv` | available |\n",
    )
    atomic_write_text(
        args.report_root / "observer_overhead.md",
        "# Insight Wave A 8GPU Observer Footprint\n\n"
        f"- Observer files: `{len(observer_files)}`\n"
        f"- Max serialized observer bytes: `{max((row['max_observer_bytes'] for task, row in task_counts.items() if task != 'unknown'), default=0)}`\n"
        f"- Truncated observer files: `{sum(row['truncated_observers'] for task, row in task_counts.items() if task != 'unknown')}`\n"
        f"- Dropped observer records: `{sum(row['dropped_record_count'] for task, row in task_counts.items() if task != 'unknown')}`\n\n"
        "These values describe diagnostic output footprint, not normal PatternKV inference speed.\n",
    )

    metric_fields = ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric", "count", "mean", "min", "max", "std"]
    write_csv(args.report_root / "pattern_gain_map.csv", pattern_gain_rows, metric_fields)
    write_csv(args.report_root / "matching_oracle_gap.csv", oracle_gap_rows, metric_fields)
    write_csv(args.report_root / "dynamic_pattern_utility.csv", dynamic_rows, metric_fields)
    write_csv(
        args.report_root / "v_gate_confusion.csv",
        v_gate_rows,
        [
            "task",
            "phase",
            "kv_type",
            "layer",
            "kv_head",
            "bucket",
            "metric",
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
            "total",
            "false_positive_rate",
            "false_negative_rate",
        ],
    )
    print(json.dumps({"status": status, "completed": total_completed, "failed": total_failed, "observer_files": len(observer_files)}, sort_keys=True))


if __name__ == "__main__":
    main()
