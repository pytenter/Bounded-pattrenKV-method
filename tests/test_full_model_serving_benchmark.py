from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bench.full_model_serving_benchmark import (
    BenchmarkConfig,
    PatternKVAdapter,
    build_full_model_path_audit,
    max_concurrency_result,
    reset_decode_only_profile_counters,
    run_full_model_benchmark,
    summarize_tpot_ms,
    workload_hash,
)
from quant.patternkv_profile import profile_snapshot, record_counter, reset_profile


class _DummyTokenizer:
    bos_token_id = 1

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        return [min(255, ord(char)) for char in text if char.strip()] or [1]


class _IdentityModule(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return hidden_states


class _DummyLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.input_layernorm = _IdentityModule()
        self.self_attn = _IdentityModule()
        self.post_attention_layernorm = _IdentityModule()
        self.mlp = _IdentityModule()
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


class _DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_size: int = 16, layers: int = 2) -> None:
        super().__init__()
        self.model = SimpleNamespace(
            embed_tokens=nn.Embedding(vocab_size, hidden_size),
            layers=nn.ModuleList([_DummyLayer(hidden_size) for _ in range(layers)]),
            norm=_IdentityModule(),
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:  # type: ignore[override]
        hidden_states = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return SimpleNamespace(logits=logits, past_key_values=(("cache",), ("cache",)))


class _CudaDecodeOnlyDummyModel(nn.Module):
    def __init__(self, vocab_size: int = 512, hidden_size: int = 16, layers: int = 2) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([_DummyLayer(hidden_size) for _ in range(layers)])
        self.norm = _IdentityModule()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.model = SimpleNamespace(embed_tokens=self.embed_tokens, layers=self.layers, norm=self.norm)

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:  # type: ignore[override]
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        cache = tuple((hidden_states[:, -1, :].detach(),) for _ in range(len(self.layers)))
        return SimpleNamespace(logits=logits, past_key_values=cache)


def test_workload_hash_is_stable_for_fair_comparison() -> None:
    fp16 = BenchmarkConfig("FP16_FULL_MODEL", 16384, 128, 4, 32)
    causal = BenchmarkConfig("CAUSAL_V4_25_FULL_MODEL", 16384, 128, 4, 32)
    changed = BenchmarkConfig("CAUSAL_V4_25_FULL_MODEL", 8192, 128, 4, 32)

    assert workload_hash(fp16) == workload_hash(causal)
    assert workload_hash(causal) != workload_hash(changed)


def test_tpot_summary_reports_ms_per_token() -> None:
    stats = summarize_tpot_ms([1.28, 2.56, 1.92], 128)
    assert stats == {"mean": 15.0, "median": 15.0, "p95": 20.0}


def test_max_concurrency_result_tracks_first_oom() -> None:
    rows = [
        {"method": "CAUSAL_V4_25_FULL_MODEL", "context_length": 16384, "decode_length": 128, "active_capacity": 1, "status": "PASS", "run_valid": True, "peak_cuda_allocated_bytes": 10},
        {"method": "CAUSAL_V4_25_FULL_MODEL", "context_length": 16384, "decode_length": 128, "active_capacity": 2, "status": "PASS", "run_valid": True, "peak_cuda_allocated_bytes": 20},
        {"method": "CAUSAL_V4_25_FULL_MODEL", "context_length": 16384, "decode_length": 128, "active_capacity": 4, "status": "OOM", "run_valid": False, "peak_cuda_allocated_bytes": None},
    ]

    result = max_concurrency_result(rows)
    assert result["max_successful_concurrency"] == 2
    assert result["first_oom_concurrency"] == 4
    assert result["peak_memory_at_max_bytes"] == 20


def test_full_model_path_audit_hits_embedding_attention_mlp_and_head() -> None:
    model = _DummyModel()
    tokenizer = _DummyTokenizer()
    audit = build_full_model_path_audit(model, tokenizer, "FP16_FULL_MODEL", torch.device("cpu"))

    assert audit["embedding_included"] is True
    assert audit["transformer_layers_included"] is True
    assert audit["attention_included"] is True
    assert audit["mlp_included"] is True
    assert audit["rmsnorm_included"] is True
    assert audit["lm_head_included"] is True
    assert audit["sampling_or_token_selection_included"] is True
    assert audit["scheduler_included"] is True
    assert audit["full_model_forward_executed"] is True


def test_patternkv_singleton_batch_reuses_iteration_plan_without_split_copy(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_SYSTEM_PROFILE", "1")
    reset_profile()
    layer_cache = ("patternkv_segmented_cache_v1",)
    request_cache = (layer_cache, layer_cache)

    assembled = PatternKVAdapter.assemble_batch([request_cache])
    split = PatternKVAdapter.split_batch(assembled, 1)
    snapshot = profile_snapshot(reset=True)

    assert assembled is request_cache
    assert split == [request_cache]
    assert snapshot["iteration_plan_builds"]["calls"] == 1
    assert snapshot.get("layer_metadata_rebuilds", {}).get("calls", 0) == 0
    assert snapshot["cache_assemble_calls"]["calls"] == 1
    assert snapshot["cache_split_calls"]["calls"] == 1


def test_patternkv_membership_change_rebuilds_iteration_plan(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_SYSTEM_PROFILE", "1")
    reset_profile()
    layer_cache = ("patternkv_segmented_cache_v1",)
    left = (layer_cache, layer_cache)
    right = (layer_cache, layer_cache)

    PatternKVAdapter.assemble_batch([left])
    PatternKVAdapter.assemble_batch([right])
    snapshot = profile_snapshot(reset=True)

    assert snapshot["iteration_plan_builds"]["calls"] == 2


def test_decode_only_profile_reset_drops_prefill_counters(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_SYSTEM_PROFILE", "1")
    reset_profile()
    record_counter("page_batch_pack", calls=1, bytes_copied=1234)

    reset_decode_only_profile_counters()
    record_counter("model_decode", calls=1)
    snapshot = profile_snapshot(reset=True)

    assert "page_batch_pack" not in snapshot
    assert snapshot["model_decode"]["calls"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for decode-only protocol integration test")
def test_decode_only_protocol_excludes_refill_prefill(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_ACTIVE_BATCH_CACHE", "1")
    monkeypatch.setenv("PATTERNKV_SYSTEM_PROFILE", "0")
    device = torch.device("cuda:0")
    model = _CudaDecodeOnlyDummyModel().to(device)
    tokenizer = _DummyTokenizer()
    config = BenchmarkConfig("CAUSAL_V4_25_FULL_MODEL", context_length=8, decode_length=2, active_capacity=1, total_requests=2)

    result = run_full_model_benchmark(PatternKVAdapter, model, tokenizer, config, device, run_index=0, warmup=False)

    assert result.run_valid is True
    assert result.completed_requests == 1
    assert result.output_tokens == 2
    assert result.initial_prefill_ms > 0.0
    assert result.prefill_calls_in_timed_window == 0
    assert result.prefill_tokens_in_timed_window == 0
    assert result.refill_calls_in_timed_window == 0
    assert result.membership_changes_in_timed_window == 0
    assert result.min_active_batch_size == 1
    assert result.max_active_batch_size == 1
    assert result.mean_active_batch_size == 1.0
