from pathlib import Path

import pytest
import torch

from insight.collector import InsightCollector
from insight.config import InsightRuntimeConfig


def test_collector_caps_sample_records_and_counts_drops(tmp_path: Path):
    collector = InsightCollector(InsightRuntimeConfig(enabled=True, output=tmp_path, max_sample_records=2))
    collector.add_sample_record({"i": 0})
    collector.add_sample_record({"i": 1})
    collector.add_sample_record({"i": 2})
    assert len(collector.records) == 2
    assert collector.dropped_record_count == 1
    assert collector.truncated is True
    assert collector.peak_record_count == 2


def test_collector_rejects_non_scalar_tensor_record(tmp_path: Path):
    collector = InsightCollector(InsightRuntimeConfig(enabled=True, output=tmp_path))
    with pytest.raises(ValueError, match="non-scalar tensor"):
        collector.add_sample_record({"bad": torch.ones(2)})


def test_collector_histograms_and_confusion_are_serialized(tmp_path: Path):
    collector = InsightCollector(InsightRuntimeConfig(enabled=True, output=tmp_path))
    collector.add_histogram("assign", torch.tensor([0, 1, 1, 2]))
    collector.add_confusion("gate", true_positive=1, false_negative=2)
    out = tmp_path / "collector.json"
    collector.flush(out)
    text = out.read_text()
    assert '"assign"' in text
    assert '"false_negative": 2' in text
    assert '"estimated_serialized_bytes"' in text
