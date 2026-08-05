#!/usr/bin/env python
"""Audit whether V realized benefit can be recovered offline from 4090 raw observer files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.io import atomic_write_json, atomic_write_text, write_csv


CSV_KEYS = ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric"]
RAW_REQUIRED = ["task", "sample_id", "phase", "layer", "kv_head", "kv_type", "bucket"]
V_REALIZED_FIELDS = [
    "raw_mse",
    "pattern_candidate_mse",
    "actual_selected_path_mse",
    "relative_candidate_benefit",
    "gate_current",
    "gate_oracle",
    "false_positive_penalty",
    "false_negative_opportunity",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sorted(values: set[Any]) -> list[Any]:
    return sorted(values, key=lambda item: (str(type(item)), str(item)))


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value))
    except Exception:
        return None


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(str(value))
    except Exception:
        return None
    return out if math.isfinite(out) else None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_files(base: Path) -> list[Path]:
    if not base.exists():
        return []
    out = []
    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
            out.append(path)
    return sorted(out)


def infer_task_from_path(path: Path) -> str:
    parts = path.parts
    for idx, part in enumerate(parts):
        if part == "generation" or part == "observer":
            if idx + 2 < len(parts):
                if parts[idx + 1] == "longbench" and idx + 2 < len(parts):
                    return parts[idx + 2]
                return parts[idx + 1]
    return ""


def iter_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    payload = read_json(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if "records" in payload and isinstance(payload["records"], list):
            return [row for row in payload["records"] if isinstance(row, dict)]
        if "record" in payload and isinstance(payload["record"], dict):
            return [payload["record"]]
        return [payload]
    return []


def flatten_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for rec in records:
        row = dict(rec)
        for field in ("layer_idx", "kv_head", "sample_index", "token_idx", "window_idx"):
            if field in row:
                row[field] = parse_int(row[field])
        for field in ("raw_mse", "pattern_candidate_mse", "actual_selected_path_mse", "relative_candidate_benefit", "false_positive_penalty", "false_negative_opportunity"):
            if field in row:
                row[field] = parse_float(row[field])
        out.append(row)
    return out


def metric_formula(row: dict[str, Any]) -> float | None:
    raw = row.get("raw_mse")
    actual = row.get("actual_selected_path_mse")
    if raw is None or actual is None:
        return None
    eps = 1e-12
    return (raw - actual) / (raw + eps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    report_root = Path(args.report_root)
    result_root = Path(args.result_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pattern_gain = read_csv_rows(report_root / "pattern_gain_map.csv")
    completion = read_json(report_root / "completion.json")
    reference_manifest = read_json(report_root / "reference_manifest.json") if (report_root / "reference_manifest.json").exists() else {}

    observer_files = discover_files(result_root / "observer")
    generation_files = discover_files(result_root / "generation")

    observer_rows: list[dict[str, Any]] = []
    file_index_rows = []
    for path in observer_files:
        try:
            payload = read_json(path)
        except Exception:
            continue
        records = flatten_records(payload.get("records", [])) if isinstance(payload, dict) else []
        task_name = infer_task_from_path(path)
        for rec in records:
            if rec.get("kv_type") == "v" and rec.get("phase") == "prefill":
                observer_rows.append({"path": str(path), "task": task_name, **rec})
        file_index_rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "record_count": len(records),
                "v_prefill_record_count": sum(1 for rec in records if rec.get("kv_type") == "v" and rec.get("phase") == "prefill"),
            }
        )

    required_present = {field: any(field in row and row.get(field) is not None for row in observer_rows) for field in V_REALIZED_FIELDS}
    row_total = len(observer_rows)
    per_key: dict[tuple[str, str, str, int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observer_rows:
        key = (
            str(row.get("task", "")),
            str(row.get("phase", "")),
            str(row.get("kv_type", "")),
            int(row.get("layer_idx")) if row.get("layer_idx") is not None else -1,
            int(row.get("kv_head")) if row.get("kv_head") is not None else -1,
            str(row.get("position_bucket", row.get("bucket", ""))),
            "relative_benefit",
        )
        per_key[key].append(row)

    recovered_rows = []
    for key, rows in sorted(per_key.items()):
        values = [metric_formula(row) for row in rows]
        values = [v for v in values if v is not None]
        if not values:
            continue
        recovered_rows.append(
            {
                "task": key[0],
                "phase": key[1],
                "kv_type": key[2],
                "layer": key[3],
                "kv_head": key[4],
                "bucket": key[5],
                "metric": key[6],
                "count": len(values),
                "mean": sum(values) / len(values),
                "std": (sum((v - (sum(values) / len(values))) ** 2 for v in values) / len(values)) ** 0.5 if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        )

    summary = {
        "schema_version": "insight_v2.v_realized_benefit_recoverability",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "report_root": str(report_root),
        "result_root": str(result_root),
        "completion_status": completion.get("status"),
        "completion_completed": completion.get("completed"),
        "observer_files": len(observer_files),
        "generation_files": len(generation_files),
        "observer_v_prefill_rows": row_total,
        "required_present": required_present,
        "recoverable": all(required_present.values()),
        "recovered_metric_rows": len(recovered_rows),
        "pattern_gain_has_v_relative_benefit": any(r.get("metric") == "relative_benefit" and r.get("kv_type") == "v" for r in pattern_gain),
        "pattern_gain_v_relative_benefit_rows": sum(1 for r in pattern_gain if r.get("metric") == "relative_benefit" and r.get("kv_type") == "v"),
    }

    atomic_write_json(output_root / "metric_semantics_audit.json", summary)
    lines = [
        "# Metric Semantics Audit",
        "",
        f"Recoverable: `{summary['recoverable']}`",
        f"Observer V prefill rows: `{row_total}`",
        f"Recovered rows: `{len(recovered_rows)}`",
        f"Pattern gain has V relative_benefit rows: `{summary['pattern_gain_v_relative_benefit_rows']}`",
        "",
        "| field | present |",
        "| --- | --- |",
    ]
    for field, present in required_present.items():
        lines.append(f"| {field} | `{present}` |")
    atomic_write_text(output_root / "metric_semantics_audit.md", "\n".join(lines) + "\n")
    write_csv(output_root / "recovered_v_relative_benefit.csv", recovered_rows, ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric", "count", "mean", "std", "min", "max"])
    merged_rows = []
    for row in pattern_gain:
        merged_rows.append(dict(row))
    for row in recovered_rows:
        merged_rows.append(dict(row))
    write_csv(output_root / "pattern_gain_map.csv", merged_rows, ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric", "count", "mean", "std", "min", "max"])
    write_csv(output_root / "observer_file_index.csv", file_index_rows, ["path", "sha256", "record_count", "v_prefill_record_count"])
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
