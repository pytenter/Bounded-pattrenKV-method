"""Bounded scalar collector for optional PatternKV Insight observer data."""

from __future__ import annotations

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
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def add_scalar(self, key: str, value: float) -> None:
        """Record one scalar if collection is enabled."""
        if not self.enabled:
            return
        self.aggregates.setdefault(key, Aggregate()).update(float(value))

    def add_sample_record(self, record: dict[str, Any]) -> None:
        """Record a small scalar-only sample record if enabled."""
        if not self.enabled:
            return
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if torch.is_tensor(value):
                clean[key] = value.detach().float().mean().item() if value.numel() else None
            else:
                clean[key] = value
        self.records.append(clean)

    def flush(self, path: Path) -> None:
        """Write aggregate JSON if enabled."""
        if not self.enabled:
            return
        atomic_write_json(
            path,
            {
                "schema_version": "insight_v1",
                "insight_level": self.config.level,
                "seed": self.config.seed,
                "aggregates": {k: v.to_json() for k, v in sorted(self.aggregates.items())},
                "records": self.records,
            },
        )

    def clear(self) -> None:
        """Release per-sample records and aggregates."""
        self.aggregates.clear()
        self.records.clear()
