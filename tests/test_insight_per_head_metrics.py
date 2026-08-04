from pathlib import Path

import torch

from insight.config import InsightRuntimeConfig
from insight.hook_metrics import record_prefill_k_metrics
from insight.runtime import begin_sample, end_sample


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


def test_each_kv_head_has_independent_prefill_k_stats(tmp_path: Path):
    cfg = InsightRuntimeConfig(enabled=True, output=tmp_path, sample_tokens=2, max_sample_records=64)
    begin_sample(_metadata(), cfg)
    raw = torch.randn(1, 2, 128, 8, dtype=torch.float16)
    base = torch.zeros(2, 4, 8, dtype=torch.float16)
    assignments = torch.zeros(1, 2, 128, dtype=torch.long)
    assignments[:, 1, :] = 1
    record_prefill_k_metrics(
        key_states=raw,
        key_states_quant=raw,
        assignments=assignments,
        k_base=base,
        key_states_full=raw,
        layer_idx=5,
        bits=2,
        group_size=128,
    )
    out = tmp_path / "observer.json"
    end_sample(out)
    text = out.read_text()
    assert "prefill.k.layer5.head0" in text
    assert "prefill.k.layer5.head1" in text
