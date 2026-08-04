import json
from pathlib import Path

import torch

from insight.config import InsightRuntimeConfig
from insight.hook_metrics import record_decode_k_window_metrics, record_decode_v_window_metrics
from insight.runtime import begin_sample, end_sample


def _metadata():
    return {
        "dataset": "gsm8k",
        "task": "gsm8k",
        "sample_id": "x",
        "problem_id": 0,
        "sample_index": 0,
        "selection_reason": "unit",
        "model_path": "/tmp/model",
        "method": "patternkv_paper",
        "seed": 0,
        "git_commit": "abc",
        "config_hash": "hash",
    }


def test_decode_k_records_true_old_new_mse(tmp_path: Path):
    cfg = InsightRuntimeConfig(enabled=True, output=tmp_path, max_sample_records=64)
    begin_sample(_metadata(), cfg)
    window = torch.randn(1, 1, 128, 8, dtype=torch.float16)
    old_base = torch.zeros(1, 1, 8, dtype=torch.float16)
    new_base = torch.cat([old_base, window.mean(dim=2)], dim=1)
    record_decode_k_window_metrics(window_raw=window, old_k_base=old_base, new_k_base=new_base, layer_idx=0, window_idx=0, bits=2, group_size=128)
    out = tmp_path / "k.json"
    end_sample(out)
    rows = json.loads(out.read_text())["records"]
    row = [r for r in rows if r.get("hook") == "decode_k"][0]
    assert "old_mse" in row and "new_mse" in row
    assert row["old_mse"] >= 0
    assert row["new_mse"] >= 0


def test_decode_k_handles_multi_head_assignments(tmp_path: Path):
    cfg = InsightRuntimeConfig(enabled=True, output=tmp_path, max_sample_records=64)
    begin_sample(_metadata(), cfg)
    window = torch.randn(1, 2, 128, 8, dtype=torch.float16)
    old_base = torch.zeros(2, 2, 8, dtype=torch.float16)
    new_base = torch.cat([old_base, window.mean(dim=(0, 2)).unsqueeze(1)], dim=1)
    record_decode_k_window_metrics(window_raw=window, old_k_base=old_base, new_k_base=new_base, layer_idx=0, window_idx=0, bits=2, group_size=128)
    out = tmp_path / "k_multi.json"
    end_sample(out)
    rows = [r for r in json.loads(out.read_text())["records"] if r.get("hook") == "decode_k"]
    assert {r["kv_head"] for r in rows} == {0, 1}


def test_decode_v_separates_candidate_assignment_and_gate_application(tmp_path: Path):
    cfg = InsightRuntimeConfig(enabled=True, output=tmp_path, max_sample_records=64)
    begin_sample(_metadata(), cfg)
    window = torch.randn(1, 1, 2, 128, dtype=torch.float16)
    old_centroids = torch.zeros(1, 1, 128, dtype=torch.float16)
    new_centroids = torch.cat([old_centroids, window.mean(dim=2)], dim=1)
    old_idx = torch.zeros(1, 1, 2, dtype=torch.long)
    new_idx = torch.ones(1, 1, 2, dtype=torch.long)
    old_mask = torch.zeros(1, 1, 2, dtype=torch.bool)
    new_mask = torch.tensor([[[1, 0]]], dtype=torch.bool)
    record_decode_v_window_metrics(
        window_raw=window,
        old_v_centroids=old_centroids,
        new_v_centroids=new_centroids,
        old_idx=old_idx,
        new_idx=new_idx,
        old_mask=old_mask,
        new_mask=new_mask,
        layer_idx=0,
        window_idx=0,
        bits=2,
        group_size=128,
    )
    out = tmp_path / "v.json"
    end_sample(out)
    row = [r for r in json.loads(out.read_text())["records"] if r.get("hook") == "decode_v"][0]
    assert row["candidate_assignment_fraction"] == 1.0
    assert row["candidate_gate_accepted_fraction"] == 0.5
