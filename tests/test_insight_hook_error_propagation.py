from pathlib import Path

import pytest
import torch

from insight.config import InsightRuntimeConfig
from insight.errors import InsightHookError
from insight.hook_metrics import record_prefill_k_metrics
from insight.runtime import abort_sample, begin_sample, end_sample, get_active_observer


def _metadata():
    return {
        "dataset": "longbench",
        "task": "hotpotqa",
        "sample_id": "x",
        "problem_id": None,
        "sample_index": 0,
        "selection_reason": "unit",
        "model_path": "/tmp/model",
        "method": "patternkv_paper",
        "seed": 0,
        "git_commit": "abc",
        "config_hash": "hash",
    }


def test_disabled_hook_is_noop_on_bad_shapes(tmp_path: Path):
    cfg = InsightRuntimeConfig(enabled=False, output=tmp_path)
    begin_sample(_metadata(), cfg)
    record_prefill_k_metrics(
        key_states=torch.zeros(1),
        key_states_quant=torch.zeros(1),
        assignments=torch.zeros(1, dtype=torch.long),
        k_base=torch.zeros(1),
        key_states_full=torch.zeros(1),
        layer_idx=0,
        bits=2,
        group_size=128,
    )
    assert get_active_observer() is None


def test_enabled_hook_error_propagates_and_abort_writes_context(tmp_path: Path):
    cfg = InsightRuntimeConfig(enabled=True, output=tmp_path)
    begin_sample(_metadata(), cfg)
    with pytest.raises(InsightHookError) as err:
        record_prefill_k_metrics(
            key_states=torch.zeros(1),
            key_states_quant=torch.zeros(1),
            assignments=torch.zeros(1, dtype=torch.long),
            k_base=torch.zeros(1),
            key_states_full=torch.zeros(1),
            layer_idx=3,
            bits=2,
            group_size=128,
        )
    out = tmp_path / "abort.json"
    abort_sample(err.value, out)
    text = out.read_text()
    assert '"status": "aborted"' in text
    assert "record_prefill_k_metrics" in text
    assert "tensor_shapes" in text
