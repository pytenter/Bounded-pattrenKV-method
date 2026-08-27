from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

import models.qwen3_patternkv_system as system


def test_qwen3_compressed_k_no_full_reconstruct():
    source = inspect.getsource(system._compressed_attention)
    assert "reconstruct_full_k" not in source
    assert "reconstruct_full_v" not in source
    assert "torch.cat(parts" not in source


def test_qwen3_compressed_v_no_full_reconstruct():
    source = inspect.getsource(system.patternkv_mixed_value_attention)
    assert "reconstruct_full_v" not in source
    assert "reconstruct_packed_v" not in source


def test_qwen3_reference_vs_compressed_semantics():
    assert system.Qwen3ForCausalLM_PatternKVCompressed.__name__ == "Qwen3ForCausalLM_PatternKVCompressed"
    assert system.Qwen3Attention_PatternKVCompressed.__name__ == "Qwen3Attention_PatternKVCompressed"


def test_qwen3_v4_fraction_25():
    cfg = SimpleNamespace(num_hidden_layers=1, patternkv_v4_budget_fraction=0.25)
    cache = system.Qwen3PatternKVCompressedCache(cfg)
    assert cache.config.patternkv_v4_budget_fraction == 0.25


def test_qwen3_selector_updates():
    source = inspect.getsource(system._compressed_attention)
    assert "update_value_causal_importance" in source


def test_qwen3_request_reset():
    cfg = SimpleNamespace(num_hidden_layers=2)
    a = system.Qwen3PatternKVCompressedCache(cfg)
    b = system.Qwen3PatternKVCompressedCache(cfg)
    assert a.layer_caches is not b.layer_caches
    assert a.get_seq_length(0) == 0
    assert b.get_seq_length(0) == 0


def test_qwen3_true_batch_b2():
    q = torch.randn(2, 32, 1, 128)
    k = torch.randn(2, 8, 17, 128)
    scores = system.patternkv_request_invariant_qk_scores(q, k, 4)
    assert scores.shape == (2, 32, 1, 17)


def test_qwen3_true_batch_b4():
    q = torch.randn(4, 32, 1, 128)
    k = torch.randn(4, 8, 9, 128)
    scores = system.patternkv_request_invariant_qk_scores(q, k, 4)
    assert scores.shape == (4, 32, 1, 9)


def test_qwen3_no_serial_dispatch():
    system.reset_qwen3_compressed_counters()
    counters = system.get_qwen3_compressed_counters()
    assert counters["serial_request_forward_dispatches"] == 0
    assert counters["serial_attention_dispatches"] == 0


def test_qwen3_timed_window_purity():
    counters = system.get_qwen3_compressed_counters()
    assert counters["refill_calls"] == 0
    assert counters["membership_changes"] == 0
    assert system.compressed_backend_counters_pass(counters)


def test_qwen3_v100_system_gate():
    counters = system.get_qwen3_compressed_counters()
    assert counters["historical_fp16_k_materialization_calls"] == 0
    assert counters["historical_fp16_v_materialization_calls"] == 0
