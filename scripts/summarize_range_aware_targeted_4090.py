#!/usr/bin/env python
"""Summarize 4090 targeted range-aware aggregate observer outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

TASKS = ("hotpotqa", "passage_retrieval_en", "passage_retrieval_zh", "samsum", "dureader", "gsm8k")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _std(count: int, total: float, sum_sq: float, *, clamp_tolerance: float = 1e-9) -> float | None:
    if count <= 0:
        return None
    mean = total / count
    variance = (sum_sq / count) - (mean * mean)
    if variance < 0 and abs(variance) <= clamp_tolerance:
        variance = 0.0
    if variance < 0:
        raise ValueError(f"variance below tolerance: {variance}")
    return math.sqrt(variance)


def flatten_aggregate_row(task: str, row: dict[str, Any]) -> dict[str, Any]:
    out = {
        "task": task,
        "phase": row["phase"],
        "kv_type": row["kv_type"],
        "layer": row["layer"],
        "kv_head": row["kv_head"],
        "bucket": row["bucket"],
        "assignment_total_count": row["assignment_total_count"],
        "assignment_mismatch_count": row["assignment_mismatch_count"],
        "mismatch_rate": row["assignment_mismatch_count"] / row["assignment_total_count"] if row["assignment_total_count"] else None,
    }
    for prefix, key in (
        ("l2_range", "l2_residual_range"),
        ("minmax_range", "minmax_residual_range"),
        ("range_gain_absolute", "range_gain_absolute"),
        ("range_regret", "range_regret"),
    ):
        metric = row[key]
        out[f"{prefix}_count"] = metric["count"]
        out[f"{prefix}_mean"] = metric["sum"] / metric["count"] if metric["count"] else None
        out[f"{prefix}_std"] = _std(metric["count"], metric["sum"], metric["sum_sq"])
        out[f"{prefix}_min"] = metric["min"]
        out[f"{prefix}_max"] = metric["max"]
        out[f"{prefix}_sum"] = metric["sum"]
        out[f"{prefix}_sum_sq"] = metric["sum_sq"]
    return out


def compute_macro_metrics(rows: list[dict[str, Any]], kv_type: str) -> dict[str, float | int | str]:
    kv_rows = [row for row in rows if row["kv_type"] == kv_type]
    task_mismatch = []
    task_regret = []
    tasks_seen = set()
    mismatch_num = 0
    mismatch_den = 0
    regret_num = 0.0
    regret_den = 0
    for task in TASKS:
        task_rows = [row for row in kv_rows if row["task"] == task]
        if not task_rows:
            continue
        tasks_seen.add(task)
        task_mismatch_num = sum(int(row["assignment_mismatch_count"]) for row in task_rows)
        task_mismatch_den = sum(int(row["assignment_total_count"]) for row in task_rows)
        task_regret_num = sum(float(row["range_regret_sum"]) for row in task_rows)
        task_regret_den = sum(int(row["range_regret_count"]) for row in task_rows)
        mismatch_num += task_mismatch_num
        mismatch_den += task_mismatch_den
        regret_num += task_regret_num
        regret_den += task_regret_den
        task_mismatch.append(task_mismatch_num / task_mismatch_den)
        task_regret.append(task_regret_num / task_regret_den)
    return {
        "kv_type": kv_type,
        "tasks_complete": len(tasks_seen),
        "mismatch_micro": mismatch_num / mismatch_den if mismatch_den else None,
        "mismatch_task_macro": sum(task_mismatch) / len(task_mismatch) if task_mismatch else None,
        "range_regret_micro": regret_num / regret_den if regret_den else None,
        "range_regret_task_macro": sum(task_regret) / len(task_regret) if task_regret else None,
    }


def decide_status(metrics: list[dict[str, Any]], *, parity_status: str, data_complete: bool) -> dict[str, Any]:
    if parity_status != "passed" or not data_complete:
        return {"status": "targeted_data_insufficient"}
    supported = False
    for row in metrics:
        supported = supported or (
            (row["mismatch_micro"] or 0.0) > 0.20
            and (row["mismatch_task_macro"] or 0.0) > 0.20
            and (row["range_regret_micro"] or 0.0) > 0.05
            and (row["range_regret_task_macro"] or 0.0) > 0.05
            and row["tasks_complete"] == 6
        )
    return {"status": "targeted_supported" if supported else "targeted_not_supported"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--parity-status", default="not_run")
    args = parser.parse_args()

    observer_files = sorted((args.result_root / "observer").rglob("*.json"))
    flat_rows: list[dict[str, Any]] = []
    for path in observer_files:
        payload = read_json(path)
        task = str((payload.get("metadata") or {}).get("task") or "gsm8k")
        for row in payload.get("range_aware_aggregates") or []:
            flat_rows.append(flatten_aggregate_row(task, row))

    data_complete = bool(flat_rows)
    metrics = [compute_macro_metrics(flat_rows, kv_type) for kv_type in ("k", "v")]
    decision = decide_status(metrics, parity_status=args.parity_status, data_complete=data_complete)
    args.report_root.mkdir(parents=True, exist_ok=True)

    if flat_rows:
        _write_csv(args.report_root / "range_aware_evidence.csv", flat_rows, list(flat_rows[0].keys()))
    completion = {
        "schema_version": "insight_v2.range_aware_targeted_summary_v1",
        "observer_files": len(observer_files),
        "rows": len(flat_rows),
        "parity_status": args.parity_status,
        "status": decision["status"],
    }
    (args.report_root / "completion.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.report_root / "range_aware_decision.json").write_text(json.dumps({"metrics": metrics, **decision}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
