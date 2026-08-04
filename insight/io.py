"""I/O helpers for PatternKV Insight diagnostics."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def sanitize_scalar(value: Any) -> Any:
    """Return a JSON/CSV-safe scalar without NaN or Inf values."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): sanitize_scalar(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_scalar(v) for v in value]
    return value


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write text using a sibling temporary file and rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write JSON with stable formatting."""
    payload = json.dumps(sanitize_scalar(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dictionaries."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write CSV with fixed column order and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: sanitize_scalar(row.get(k)) for k in fieldnames})
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
