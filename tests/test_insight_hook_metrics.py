from pathlib import Path

import torch

from insight.config import InsightRuntimeConfig
from insight.hook_metrics import record_prefill_k_metrics, record_prefill_v_metrics
from insight.runtime import begin_sample, end_sample


def _metadata():
    return {
        "dataset": "longbench",
        "task": "hotpotqa",
        "sample_id": "0",
        "problem_id": None,
        "sample_index": 0,
        "selection_reason": "unit",
        "model_path": "/tmp/model",
        "method": "patternkv_paper",
        "seed": 0,
        "git_commit": "abc",
        "config_hash": "hash",
    }


def test_prefill_k_hook_records_bounded_scalars(tmp_path: Path):
    cfg = InsightRuntimeConfig(enabled=True, output=tmp_path, sample_tokens=4, max_sample_records=8)
    begin_sample(_metadata(), cfg)
    raw = torch.randn(1, 1, 128, 4, dtype=torch.float16)
    base = torch.zeros(1, 2, 4, dtype=torch.float16)
    assignments = torch.zeros(1, 1, 128, dtype=torch.long)
    residual = raw.clone()
    record_prefill_k_metrics(
        key_states=raw,
        key_states_quant=residual,
        assignments=assignments,
        k_base=base,
        key_states_full=raw,
        layer_idx=0,
        bits=2,
        group_size=128,
    )
    out = tmp_path / "k.json"
    end_sample(out)
    text = out.read_text()
    assert "prefill.k.layer0.raw_mse" in text
    assert "assignment_histogram" in text


def test_prefill_v_hook_records_gate_confusion(tmp_path: Path):
    cfg = InsightRuntimeConfig(enabled=True, output=tmp_path, sample_tokens=2, max_sample_records=8)
    begin_sample(_metadata(), cfg)
    raw = torch.randn(1, 1, 2, 128, dtype=torch.float16)
    residual = raw.clone()
    idx = torch.zeros(1, 1, 2, dtype=torch.long)
    centroids = torch.zeros(1, 2, 128, dtype=torch.float16)
    mask = torch.tensor([[[1, 0]]], dtype=torch.bool)
    record_prefill_v_metrics(
        value_states_quant=residual,
        idx_q=idx,
        v_centroids=centroids,
        v_mask_q=mask,
        value_states_full=raw,
        layer_idx=0,
        bits=2,
        group_size=128,
    )
    out = tmp_path / "v.json"
    end_sample(out)
    text = out.read_text()
    assert "prefill.v.layer0.gate_acceptance" in text
    assert "gate_vs_selected_oracle" in text
