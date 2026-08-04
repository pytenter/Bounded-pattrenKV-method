import json
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


def test_k_conditional_oracle_records_true_group_and_assignment(tmp_path: Path):
    cfg = InsightRuntimeConfig(enabled=True, output=tmp_path, level="oracle", sample_tokens=1, oracle_layers=(0,), max_sample_records=128)
    begin_sample(_metadata(), cfg)
    raw = torch.randn(1, 1, 128, 8, dtype=torch.float16)
    base = torch.randn(1, 32, 8, dtype=torch.float16)
    assignments = torch.zeros(1, 1, 128, dtype=torch.long)
    record_prefill_k_metrics(
        key_states=raw,
        key_states_quant=raw,
        assignments=assignments,
        k_base=base,
        key_states_full=raw,
        layer_idx=0,
        bits=2,
        group_size=128,
    )
    out = tmp_path / "observer.json"
    end_sample(out)
    payload = json.loads(out.read_text())
    rows = [r for r in payload["records"] if r.get("hook") == "k_conditional_oracle"]
    assert rows
    row = rows[0]
    assert row["group_start_token"] == 0
    assert row["current_assignment"] == 0
    assert 0 <= row["conditional_oracle_assignment"] < 32
