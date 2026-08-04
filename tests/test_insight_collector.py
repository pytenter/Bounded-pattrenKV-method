from pathlib import Path

from insight.collector import InsightCollector
from insight.config import InsightRuntimeConfig


def test_observer_disabled_creates_no_stats(tmp_path: Path):
    collector = InsightCollector(InsightRuntimeConfig(enabled=False, output=tmp_path))
    collector.add_scalar("x", 1.0)
    collector.add_sample_record({"a": 1})
    out = tmp_path / "stats.json"
    collector.flush(out)
    assert not out.exists()
    assert collector.aggregates == {}
    assert collector.records == []


def test_observer_enabled_flushes_scalar_stats(tmp_path: Path):
    collector = InsightCollector(InsightRuntimeConfig(enabled=True, output=tmp_path))
    collector.add_scalar("x", 1.0)
    collector.add_scalar("x", 3.0)
    out = tmp_path / "stats.json"
    collector.flush(out)
    text = out.read_text()
    assert '"mean": 2.0' in text
