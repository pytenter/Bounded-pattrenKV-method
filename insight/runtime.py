"""Per-process read-only PatternKV Insight observer runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from insight.collector import InsightCollector
from insight.config import InsightRuntimeConfig
from insight.errors import InsightHookError
from insight.io import atomic_write_json


REQUIRED_METADATA = (
    "dataset",
    "task",
    "sample_id",
    "problem_id",
    "sample_index",
    "selection_reason",
    "model_path",
    "method",
    "seed",
    "git_commit",
    "config_hash",
)


@dataclass
class InsightObserver:
    """Owns one sample's bounded collector and metadata."""

    metadata: dict[str, Any]
    config: InsightRuntimeConfig
    collector: InsightCollector = field(init=False)
    status: str = "running"

    def __post_init__(self) -> None:
        self.collector = InsightCollector(self.config)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def add_scalar(self, key: str, value: float) -> None:
        self.collector.add_scalar(key, value)

    def add_histogram(self, key: str, values: Any) -> None:
        self.collector.add_histogram(key, values)

    def add_confusion(self, key: str, **counts: int) -> None:
        self.collector.add_confusion(key, **counts)

    def add_sample_record(self, record: dict[str, Any]) -> None:
        self.collector.add_sample_record(record)

    def add_range_aware_aggregate(self, **payload: Any) -> None:
        self.collector.add_range_aware_aggregate(**payload)

    def write(self, output_path: Path, *, status: str, error: str | None = None) -> None:
        payload = {
            "schema_version": self.config.schema_version,
            "status": status,
            "error": error,
            "metadata": self.metadata,
            "insight_level": self.config.level,
            "seed": self.config.seed,
            "aggregates": {k: v.to_json() for k, v in sorted(self.collector.aggregates.items())},
            "histograms": self.collector._histograms_json(),
            "confusion": self.collector._confusion_json(),
            "records": self.collector.records,
            "sample_records_enabled": self.config.sample_records_enabled,
            "range_aware_aggregates": self.collector._range_aware_json(),
            "estimated_serialized_bytes": self.collector.estimated_serialized_bytes(),
            "dropped_record_count": self.collector.dropped_record_count,
            "max_sample_records": self.collector.config.max_sample_records,
            "peak_record_count": self.collector.peak_record_count,
            "truncated": self.collector.truncated,
        }
        atomic_write_json(output_path, payload)


_ACTIVE_OBSERVER: InsightObserver | None = None


def begin_sample(metadata: dict[str, Any], runtime_config: InsightRuntimeConfig) -> InsightObserver | None:
    """Start one sample observer and register it for lightweight model hooks."""
    global _ACTIVE_OBSERVER
    if not runtime_config.enabled:
        _ACTIVE_OBSERVER = None
        return None
    if _ACTIVE_OBSERVER is not None:
        raise RuntimeError("nested PatternKV Insight samples are not supported")
    missing = [field for field in REQUIRED_METADATA if field not in metadata]
    if missing:
        raise ValueError(f"missing insight metadata fields: {missing}")
    _ACTIVE_OBSERVER = InsightObserver(dict(metadata), runtime_config)
    return _ACTIVE_OBSERVER


def get_active_observer() -> InsightObserver | None:
    """Return the active observer, or None when collection is disabled."""
    return _ACTIVE_OBSERVER


def end_sample(output_path: Path) -> None:
    """Flush and clear the active observer."""
    global _ACTIVE_OBSERVER
    observer = _ACTIVE_OBSERVER
    _ACTIVE_OBSERVER = None
    if observer is not None:
        observer.status = "completed"
        observer.write(output_path, status="completed")
        observer.collector.clear()


def abort_sample(error: BaseException | str, output_path: Path) -> None:
    """Flush error state and clear the active observer."""
    global _ACTIVE_OBSERVER
    observer = _ACTIVE_OBSERVER
    _ACTIVE_OBSERVER = None
    if observer is not None:
        observer.status = "aborted"
        if isinstance(error, InsightHookError):
            observer.add_sample_record(
                {
                    "hook": error.hook_name,
                    "phase": error.phase,
                    "kv_type": error.kv_type,
                    "layer_idx": error.layer_idx,
                    "kv_head": error.kv_head,
                    "exception_type": error.exception_type,
                    "exception_message": error.exception_message,
                    "tensor_shapes": error.tensor_shapes,
                }
            )
        observer.write(output_path, status="aborted", error=repr(error))
        observer.collector.clear()
