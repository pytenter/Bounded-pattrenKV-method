from pathlib import Path

import torch

import insight.hook_metrics as hm
from insight.config import InsightRuntimeConfig
from insight.hook_metrics import record_prefill_v_metrics
from insight.runtime import begin_sample, end_sample


def _metadata():
    return {
        "dataset": "longbench",
        "task": "samsum",
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


def test_v_gate_confusion_is_per_token_with_all_four_cells(tmp_path: Path, monkeypatch):
    def fake_v_token_mse(raw, pattern, *, bits, group_size):
        if pattern is None:
            return torch.tensor([1.0], device=raw.device).expand(raw.shape[0])
        return torch.tensor([0.5, 1.5, 0.5, 1.5], device=raw.device)[: raw.shape[0]]

    monkeypatch.setattr(hm, "_v_token_mse", fake_v_token_mse)
    cfg = InsightRuntimeConfig(enabled=True, output=tmp_path, sample_tokens=1, max_sample_records=64)
    begin_sample(_metadata(), cfg)
    raw = torch.zeros(4, 1, 1, 128, dtype=torch.float16)
    idx = torch.zeros(4, 1, 1, dtype=torch.long)
    centroids = torch.zeros(1, 32, 128, dtype=torch.float16)
    gate = torch.tensor([[[1]], [[1]], [[0]], [[0]]], dtype=torch.bool)
    record_prefill_v_metrics(
        value_states_quant=raw,
        idx_q=idx,
        v_centroids=centroids,
        v_mask_q=gate,
        value_states_full=raw,
        layer_idx=0,
        bits=2,
        group_size=128,
    )
    out = tmp_path / "v.json"
    end_sample(out)
    text = out.read_text()
    assert '"true_positive": 1' in text
    assert '"true_negative": 1' in text
    assert '"false_positive": 1' in text
    assert '"false_negative": 1' in text
