"""Bounded scalar collector for optional PatternKV Insight observer data."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from insight.config import InsightRuntimeConfig
from insight.io import atomic_write_json


@dataclass
class Aggregate:
    """Streaming aggregate that does not retain raw tensors."""

    count: int = 0
    total: float = 0.0
    sum_sq: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def update(self, value: float) -> None:
        """Add one finite scalar to the aggregate."""
        self.count += 1
        self.total += float(value)
        self.sum_sq += float(value) * float(value)
        self.min_value = float(value) if self.min_value is None else min(self.min_value, float(value))
        self.max_value = float(value) if self.max_value is None else max(self.max_value, float(value))

    def to_json(self) -> dict[str, Any]:
        """Serialize aggregate values."""
        mean = self.total / self.count if self.count else None
        return {
            "count": self.count,
            "sum": self.total,
            "sum_sq": self.sum_sq,
            "min": self.min_value,
            "max": self.max_value,
            "mean": mean,
        }


@dataclass
class InsightCollector:
    """Optional scalar observer collector.

    When disabled, all methods are no-ops and no output is written.
    """

    config: InsightRuntimeConfig
    aggregates: dict[str, Aggregate] = field(default_factory=dict)
    histograms: dict[str, Counter[int]] = field(default_factory=dict)
    confusion: dict[str, Counter[str]] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    dropped_record_count: int = 0
    peak_record_count: int = 0
    truncated: bool = False

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def add_scalar(self, key: str, value: float) -> None:
        """Record one scalar if collection is enabled."""
        if not self.enabled:
            return
        self.aggregates.setdefault(key, Aggregate()).update(float(value))

    def add_histogram(self, key: str, values: Any) -> None:
        """Record integer histogram counts from a small iterable or tensor."""
        if not self.enabled:
            return
        if torch.is_tensor(values):
            if values.numel() > 100_000:
                raise ValueError(f"histogram tensor for {key} is too large: {values.numel()} values")
            seq = values.detach().reshape(-1).to("cpu").tolist()
        elif isinstance(values, dict):
            for bucket, count in values.items():
                self.histograms.setdefault(key, Counter())[int(bucket)] += int(count)
            return
        else:
            seq = list(values)
        counter = self.histograms.setdefault(key, Counter())
        for value in seq:
            counter[int(value)] += 1

    def add_confusion(self, key: str, *, true_positive: int = 0, true_negative: int = 0, false_positive: int = 0, false_negative: int = 0) -> None:
        """Accumulate binary confusion counters."""
        if not self.enabled:
            return
        c = self.confusion.setdefault(key, Counter())
        c["true_positive"] += int(true_positive)
        c["true_negative"] += int(true_negative)
        c["false_positive"] += int(false_positive)
        c["false_negative"] += int(false_negative)

    def add_sample_record(self, record: dict[str, Any]) -> None:
        """Record a small scalar-only sample record if enabled."""
        if not self.enabled:
            return
        if len(self.records) >= self.config.max_sample_records:
            self.dropped_record_count += 1
            self.truncated = True
            self.peak_record_count = max(self.peak_record_count, len(self.records))
            return
        clean: dict[str, Any] = {}
        for key, value in record.items():
            clean[key] = self._clean_record_value(key, value)
        self.records.append(clean)
        self.peak_record_count = max(self.peak_record_count, len(self.records))

    @staticmethod
    def _clean_record_value(key: str, value: Any) -> Any:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            if value.numel() != 1:
                raise ValueError(f"sample record field {key!r} received non-scalar tensor with shape {tuple(value.shape)}")
            return value.detach().to("cpu").item()
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for k, v in value.items():
                if torch.is_tensor(v):
                    if v.numel() != 1:
                        raise ValueError(f"sample record field {key}.{k!r} received non-scalar tensor with shape {tuple(v.shape)}")
                    cleaned[str(k)] = v.detach().to("cpu").item()
                elif isinstance(v, (int, float, str, bool)) or v is None:
                    cleaned[str(k)] = v
                elif isinstance(v, (list, tuple)) and len(v) <= 16 and all(isinstance(x, (int, float, str, bool)) or x is None for x in v):
                    cleaned[str(k)] = list(v)
                elif isinstance(v, (list, tuple)) and len(v) <= 16 and all(isinstance(x, (list, tuple)) for x in v):
                    nested = []
                    for row in v:
                        if len(row) > 16 or not all(isinstance(x, (int, float, str, bool)) or x is None for x in row):
                            raise ValueError(f"sample record field {key}.{k!r} has unsupported nested list")
                        nested.append(list(row))
                    cleaned[str(k)] = nested
                else:
                    raise ValueError(f"sample record field {key}.{k!r} has unsupported value type {type(v).__name__}")
            return cleaned
        if isinstance(value, (list, tuple)):
            if len(value) > 512:
                raise ValueError(f"sample record field {key!r} has too many entries: {len(value)}")
            cleaned_list: list[Any] = []
            for item in value:
                if torch.is_tensor(item):
                    if item.numel() != 1:
                        raise ValueError(f"sample record field {key!r} contains non-scalar tensor with shape {tuple(item.shape)}")
                    cleaned_list.append(item.detach().to("cpu").item())
                elif isinstance(item, (int, float, str, bool)) or item is None:
                    cleaned_list.append(item)
                else:
                    raise ValueError(f"sample record field {key!r} contains unsupported value type {type(item).__name__}")
            return cleaned_list
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        raise ValueError(f"sample record field {key!r} has unsupported value type {type(value).__name__}")

    def estimated_serialized_bytes(self) -> int:
        """Return an approximate serialized payload size for boundedness checks."""
        payload = {
            "aggregates": {k: v.to_json() for k, v in sorted(self.aggregates.items())},
            "histograms": self._histograms_json(),
            "confusion": self._confusion_json(),
            "records": self.records,
        }
        return len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))

    def _histograms_json(self) -> dict[str, dict[str, int]]:
        return {k: {str(bucket): int(count) for bucket, count in sorted(v.items())} for k, v in sorted(self.histograms.items())}

    def _confusion_json(self) -> dict[str, dict[str, int]]:
        return {k: {name: int(count) for name, count in sorted(v.items())} for k, v in sorted(self.confusion.items())}

    def flush(self, path: Path) -> None:
        """Write aggregate JSON if enabled."""
        if not self.enabled:
            return
        atomic_write_json(
            path,
            {
                "schema_version": "insight_v2.collector",
                "insight_level": self.config.level,
                "seed": self.config.seed,
                "aggregates": {k: v.to_json() for k, v in sorted(self.aggregates.items())},
                "histograms": self._histograms_json(),
                "confusion": self._confusion_json(),
                "records": self.records,
                "estimated_serialized_bytes": self.estimated_serialized_bytes(),
                "dropped_record_count": self.dropped_record_count,
                "max_sample_records": self.config.max_sample_records,
                "peak_record_count": self.peak_record_count,
                "truncated": self.truncated,
            },
        )

    def clear(self) -> None:
        """Release per-sample records and aggregates."""
        self.aggregates.clear()
        self.histograms.clear()
        self.confusion.clear()
        self.records.clear()
        self.dropped_record_count = 0
        self.peak_record_count = 0
        self.truncated = False
