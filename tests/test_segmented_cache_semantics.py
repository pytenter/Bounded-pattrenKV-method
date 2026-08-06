from __future__ import annotations

import math

import pytest
import torch

from models.segmented_cache import (
    PatternQuantizedKVCache,
    append_decode,
    build_cache_from_prefill,
    cache_segment_stats,
    deserialize_cache,
    reconstruct_full_k,
    reconstruct_full_v,
    serialize_cache,
    validate_cache,
)


def make_kv(tokens: int, *, heads: int = 2, dim: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(heads * tokens * dim, dtype=torch.float16).reshape(1, heads, tokens, dim)
    return base / 100, (base + 7) / 100


@pytest.mark.parametrize("tokens", [2, 5, 8, 13, 15])
def test_prefill_partition_reconstructs_token_order(tokens: int) -> None:
    sink_length = 4
    recent_length = 4
    group_size = 4
    key, value = make_kv(tokens)
    cache = build_cache_from_prefill(
        key,
        value,
        sink_length=sink_length,
        recent_length=recent_length,
        group_size=group_size,
        k_bits=2,
        v_bits=2,
    )
    stats = cache_segment_stats(cache)
    assert stats["sink_tokens"] == min(tokens, sink_length)
    assert stats["recent_tokens"] == min(max(tokens - sink_length, 0), recent_length)
    assert stats["sink_tokens"] + stats["packed_history_tokens"] + stats["pending_history_tokens"] + stats["recent_tokens"] == tokens
    assert reconstruct_full_k(cache).shape == key.shape
    assert reconstruct_full_v(cache).shape == value.shape


def test_decode_rolls_recent_without_quantizing_sink() -> None:
    sink_length = 3
    recent_length = 4
    group_size = 4
    key, value = make_kv(5, dim=16)
    cache = build_cache_from_prefill(key, value, sink_length=sink_length, recent_length=recent_length, group_size=group_size, k_bits=2, v_bits=2)
    original_sink = cache.sink_k.clone()
    all_k = [key]
    all_v = [value]
    for _ in range(100):
        next_k, next_v = make_kv(1, dim=16)
        next_k = next_k + len(all_k)
        next_v = next_v + len(all_v)
        all_k.append(next_k)
        all_v.append(next_v)
        append_decode(cache, next_k, next_v)
        expected_k = torch.cat(all_k, dim=2)
        expected_v = torch.cat(all_v, dim=2)
        assert torch.equal(cache.sink_k, original_sink)
        assert cache.recent_k.shape[2] == min(max(expected_k.shape[2] - sink_length, 0), recent_length)
        assert torch.equal(cache.recent_k, expected_k[:, :, -cache.recent_k.shape[2] :, :])
        assert reconstruct_full_k(cache).shape == expected_k.shape
        assert reconstruct_full_v(cache).shape == expected_v.shape
        validate_cache(cache)


@pytest.mark.parametrize("k_bits,v_bits", [(2, 2), (4, 2), (2, 4)])
def test_segmented_attention_matches_reconstructed_reference_shape(k_bits: int, v_bits: int) -> None:
    torch.manual_seed(0)
    key, value = make_kv(21, dim=16)
    cache = build_cache_from_prefill(key, value, sink_length=4, recent_length=5, group_size=4, k_bits=k_bits, v_bits=v_bits)
    query = torch.randn(1, 2, 3, 16, dtype=torch.float16)
    full_k = reconstruct_full_k(cache)
    full_v = reconstruct_full_v(cache)
    scores = torch.matmul(query, full_k.transpose(2, 3)) / math.sqrt(query.shape[-1])
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(weights, full_v)
    assert out.shape == (1, 2, 3, 16)
    assert torch.isfinite(out).all()
    assert cache.packed_k.dtype == torch.int32
    assert cache.packed_v.dtype == torch.int32


def test_pattern_assignment_alignment_excludes_pending_recent() -> None:
    cache = PatternQuantizedKVCache(total_tokens=16, sink_length=4, recent_length=4, group_size=4, k_bits=2, v_bits=2)
    key, value = make_kv(16, dim=16)
    built = build_cache_from_prefill(key, value, sink_length=4, recent_length=4, group_size=4, k_bits=2, v_bits=2, pattern=True)
    assert isinstance(built, PatternQuantizedKVCache)
    built.k_assignments = torch.zeros(1, 2, built.packed_k_tokens, dtype=torch.long)
    built.v_assignment_idx = torch.zeros(1, 2, built.packed_v_tokens, dtype=torch.long)
    built.v_assignments = torch.ones(1, 2, built.packed_v_tokens, dtype=torch.uint8)
    validate_cache(built)
    built.k_assignments = torch.zeros(1, 2, built.packed_k_tokens + built.recent_k.shape[2], dtype=torch.long)
    with pytest.raises(ValueError):
        validate_cache(built)
    assert cache.total_tokens == 16


def test_long_rolling_crosses_multiple_pack_cycles() -> None:
    key, value = make_kv(6, dim=16)
    cache = build_cache_from_prefill(key, value, sink_length=4, recent_length=8, group_size=4, k_bits=2, v_bits=2)
    for step in range(64):
        next_k, next_v = make_kv(1, dim=16)
        append_decode(cache, next_k + step, next_v + step)
    stats = cache_segment_stats(cache)
    assert stats["sink_tokens"] == 4
    assert stats["recent_tokens"] == 8
    assert stats["packed_history_tokens"] >= 56
    assert stats["total_tokens"] == 70
    validate_cache(cache)


def test_cache_serialization_round_trip() -> None:
    key, value = make_kv(17, dim=16)
    cache = build_cache_from_prefill(key, value, sink_length=4, recent_length=5, group_size=4, k_bits=4, v_bits=4, pattern=True)
    assert isinstance(cache, PatternQuantizedKVCache)
    cache.k_assignments = torch.zeros(1, 2, cache.packed_k_tokens, dtype=torch.long)
    cache.v_assignments = torch.ones(1, 2, cache.packed_v_tokens, dtype=torch.uint8)
    cache.v_assignment_idx = torch.zeros(1, 2, cache.packed_v_tokens, dtype=torch.long)
    restored = deserialize_cache(serialize_cache(cache), pattern=True)
    assert cache_segment_stats(restored) == cache_segment_stats(cache)
    assert torch.equal(reconstruct_full_k(restored), reconstruct_full_k(cache))
    assert torch.equal(restored.k_assignments, cache.k_assignments)
