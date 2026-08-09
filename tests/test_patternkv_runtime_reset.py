from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from models.llama_patternkv import collect_patternkv_dynamic_stats, reset_patternkv_runtime_state
from models.segmented_cache import build_cache_from_prefill, serialize_cache


class DummyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_base = torch.ones(2, 3, 16)
        self.v_centroids = torch.ones(2, 4, 16)
        self.num_k_bases = 3
        self.num_v_bases = 4


def test_reset_patternkv_runtime_state_clears_layer_centroids() -> None:
    layers = [SimpleNamespace(self_attn=DummyAttention()), SimpleNamespace(self_attn=DummyAttention())]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    reset_patternkv_runtime_state(model)
    assert all(layer.self_attn.k_base is None for layer in layers)
    assert all(layer.self_attn.v_centroids is None for layer in layers)


def test_repeated_sample_cache_state_is_deterministic_after_reset() -> None:
    key = torch.arange(1 * 2 * 8 * 16, dtype=torch.float16).reshape(1, 2, 8, 16) / 11
    value = key + 0.25
    k_centroids = torch.zeros(2, 1, 16, dtype=torch.float16)
    v_centroids = torch.zeros(2, 1, 16, dtype=torch.float16)

    def build_serialized():
        return serialize_cache(
            build_cache_from_prefill(
                key,
                value,
                sink_length=0,
                recent_length=4,
                group_size=4,
                k_bits=2,
                v_bits=2,
                pattern=True,
                k_centroids=k_centroids,
                v_centroids=v_centroids,
            )
        )

    first = build_serialized()
    second = build_serialized()
    assert len(first) == len(second)
    assert first[0] == second[0]
    assert torch.equal(first[3], second[3])
    assert first[23] is False
    assert second[23] is False
    assert torch.equal(first[28], second[28])


def test_collect_dynamic_stats_reports_assignment_alignment() -> None:
    key = torch.arange(1 * 2 * 8 * 16, dtype=torch.float16).reshape(1, 2, 8, 16) / 13
    value = key + 0.5
    cache = build_cache_from_prefill(
        key,
        value,
        sink_length=0,
        recent_length=4,
        group_size=4,
        k_bits=2,
        v_bits=2,
        pattern=True,
        k_centroids=torch.zeros(2, 1, 16, dtype=torch.float16),
        v_centroids=torch.zeros(2, 1, 16, dtype=torch.float16),
    )
    layer = SimpleNamespace(self_attn=SimpleNamespace(num_k_bases=1, num_v_bases=1))
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    stats = collect_patternkv_dynamic_stats(model, [serialize_cache(cache)])
    assert stats["k_assignment_tokens_per_layer"] == [4]
    assert stats["v_assignment_tokens_per_layer"] == [4]
    assert stats["packed_k_tokens_per_layer"] == [4]
    assert stats["packed_v_tokens_per_layer"] == [4]
