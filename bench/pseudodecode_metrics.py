from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


CHECKPOINTS = (128, 512, 1024, 2048, 4096, 8192, 16384, 24576)
SPARSE_LAYERS = (0, 7, 15, 23, 31)


LOWER_IS_BETTER_METRICS = {
    "attention_output_relative_L2",
    "attention_output_MAE",
    "post_WO_relative_L2",
    "hidden_cosine_loss",
    "next_token_KL",
    "next_token_JS",
    "target_token_NLL_delta",
    "top1_disagreement",
    "k_norm_abs_ratio_error",
    "v_norm_abs_ratio_error",
    "k_directional_error",
    "v_directional_error",
}


def canonical_json_hash(payload: Any, length: int | None = None) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return digest if length is None else digest[:length]


def token_ids_sha256(token_ids: list[int]) -> str:
    return canonical_json_hash({"token_ids": [int(x) for x in token_ids]})


def full_trajectory_sha256(prompt_token_ids: list[int], generated_token_ids: list[int]) -> str:
    return canonical_json_hash(
        {
            "prompt_token_ids": [int(x) for x in prompt_token_ids],
            "generated_token_ids": [int(x) for x in generated_token_ids],
        }
    )


def write_json_gz(path: Path, payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_availability(generated_reference_tokens: int, checkpoints: Iterable[int] = CHECKPOINTS) -> list[dict[str, Any]]:
    rows = []
    for checkpoint in checkpoints:
        available = int(generated_reference_tokens) >= int(checkpoint)
        rows.append(
            {
                "checkpoint": int(checkpoint),
                "checkpoint_available": available,
                "availability_reason": "available" if available else "trajectory_too_short",
            }
        )
    return rows


def hidden_cosine_loss(hidden_cosine: float) -> float:
    return 1.0 - float(hidden_cosine)


def top1_disagreement(top1_agreement: bool | int | float) -> float:
    return 1.0 - float(bool(top1_agreement))


def accumulation_gap(pseudo_value: float, static_value: float) -> float:
    return float(pseudo_value) - float(static_value)


def trapezoid_auc_log2(points: Iterable[tuple[int, float]]) -> float | None:
    ordered = sorted((int(x), float(y)) for x, y in points if x and x > 0 and y is not None)
    if len(ordered) < 2:
        return None
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        lx0 = math.log2(x0)
        lx1 = math.log2(x1)
        auc += (lx1 - lx0) * (y0 + y1) * 0.5
    return auc


def quantile(values: list[float], q: float) -> float | None:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def iqr(values: list[float]) -> dict[str, float | None]:
    return {"q1": quantile(values, 0.25), "median": quantile(values, 0.5), "q3": quantile(values, 0.75)}


def grouped_median_iqr(rows: Iterable[dict[str, Any]], group_keys: tuple[str, ...], value_key: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        value = row.get(value_key)
        if value is None:
            continue
        buckets.setdefault(tuple(row.get(k) for k in group_keys), []).append(float(value))
    out = []
    for key, values in sorted(buckets.items()):
        stats = iqr(values)
        out.append(
            {
                **dict(zip(group_keys, key)),
                "n_available": len(values),
                "median": stats["median"],
                "iqr_low": stats["q1"],
                "iqr_high": stats["q3"],
                "p90": quantile(values, 0.90),
            }
        )
    return out


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired_delta(rows: Iterable[dict[str, Any]], *, left_config: str, right_config: str, value_key: str) -> dict[str, Any]:
    by_config_task = {(row.get("config"), row.get("task_key")): row for row in rows}
    tasks = sorted({task for config, task in by_config_task if config in {left_config, right_config}})
    deltas = []
    for task in tasks:
        left = by_config_task.get((left_config, task))
        right = by_config_task.get((right_config, task))
        if not left or not right:
            continue
        if left.get(value_key) is None or right.get(value_key) is None:
            continue
        deltas.append(float(right[value_key]) - float(left[value_key]))
    return {
        "left_config": left_config,
        "right_config": right_config,
        "paired_n": len(deltas),
        "median_delta": statistics.median(deltas) if deltas else None,
        "tasks_improved": sum(1 for x in deltas if x < 0),
        "tasks_regressed": sum(1 for x in deltas if x > 0),
        "ties": sum(1 for x in deltas if x == 0),
    }
