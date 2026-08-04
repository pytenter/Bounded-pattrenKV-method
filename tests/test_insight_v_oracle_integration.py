import json
from pathlib import Path

import torch

from insight.config import InsightRuntimeConfig
from insight.hook_metrics import record_prefill_v_metrics
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


def test_v_mse_oracle_is_no_worse_than_current_candidate(tmp_path: Path):
    cfg = InsightRuntimeConfig(enabled=True, output=tmp_path, level="oracle", sample_tokens=1, oracle_layers=(0,), max_sample_records=128)
    begin_sample(_metadata(), cfg)
    raw = torch.randn(1, 1, 1, 128, dtype=torch.float16)
    idx = torch.zeros(1, 1, 1, dtype=torch.long)
    centroids = torch.randn(1, 32, 128, dtype=torch.float16)
    mask = torch.ones(1, 1, 1, dtype=torch.bool)
    record_prefill_v_metrics(
        value_states_quant=raw,
        idx_q=idx,
        v_centroids=centroids,
        v_mask_q=mask,
        value_states_full=raw,
        layer_idx=0,
        bits=2,
        group_size=128,
    )
    out = tmp_path / "observer.json"
    end_sample(out)
    payload = json.loads(out.read_text())
    oracle_rows = [r for r in payload["records"] if r.get("hook") == "v_matching_oracle"]
    assert oracle_rows
    row = oracle_rows[0]
    assert row["oracle_mse"] <= row["current_pattern_mse"] + 1e-9
