from pathlib import Path

import pytest

from insight.config import InsightRuntimeConfig
from insight.runtime import abort_sample, begin_sample, end_sample, get_active_observer


def _metadata():
    return {
        "dataset": "gsm8k",
        "task": "gsm8k",
        "sample_id": "gsm8k:0",
        "problem_id": 0,
        "sample_index": 0,
        "selection_reason": "unit",
        "model_path": "/tmp/model",
        "method": "patternkv_paper",
        "seed": 0,
        "git_commit": "abc",
        "config_hash": "hash",
    }


def test_disabled_runtime_has_no_active_observer(tmp_path: Path):
    cfg = InsightRuntimeConfig(enabled=False, output=tmp_path)
    assert begin_sample(_metadata(), cfg) is None
    assert get_active_observer() is None


def test_runtime_flushes_and_clears(tmp_path: Path):
    cfg = InsightRuntimeConfig(enabled=True, output=tmp_path, max_sample_records=4)
    obs = begin_sample(_metadata(), cfg)
    assert obs is get_active_observer()
    obs.add_scalar("x", 1.0)
    out = tmp_path / "observer.json"
    end_sample(out)
    assert get_active_observer() is None
    assert '"status": "completed"' in out.read_text()


def test_runtime_rejects_nested_and_abort_clears(tmp_path: Path):
    cfg = InsightRuntimeConfig(enabled=True, output=tmp_path)
    begin_sample(_metadata(), cfg)
    with pytest.raises(RuntimeError):
        begin_sample(_metadata(), cfg)
    out = tmp_path / "abort.json"
    abort_sample("boom", out)
    assert get_active_observer() is None
    assert '"status": "aborted"' in out.read_text()
