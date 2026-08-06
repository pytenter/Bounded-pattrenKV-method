from __future__ import annotations

import torch

from models.segmented_cache import (
    append_decode,
    build_cache_from_prefill,
    cache_segment_stats,
    reconstruct_full_k,
    reconstruct_full_v,
)


def _kv(tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.arange(1 * 2 * tokens * 8, dtype=torch.float16).reshape(1, 2, tokens, 8) / 31
    return key, key + 0.125


def _pattern_cache(tokens: int):
    key, value = _kv(tokens)
    return build_cache_from_prefill(
        key,
        value,
        sink_length=0,
        recent_length=0,
        group_size=4,
        k_bits=4,
        v_bits=4,
        pattern=True,
        k_centroids=torch.zeros(2, 1, 8, dtype=torch.float16),
        v_centroids=torch.zeros(2, 1, 8, dtype=torch.float16),
        cache_mode="segmented_chunked",
        chunk_length=4,
    )


def test_segmented_chunked_matches_legacy_chunk_formula() -> None:
    for tokens in range(10):
        cache = _pattern_cache(tokens)
        stats = cache_segment_stats(cache)
        assert stats["packed_history_tokens"] == (tokens // 4) * 4
        assert stats["chunk_tokens"] == tokens % 4
        assert stats["pending_history_tokens"] == tokens % 4
        assert stats["recent_tokens"] == 0
        assert stats["sink_tokens"] == 0
        assert stats["k_assignment_tokens"] in (None, (tokens // 4) * 4)
        assert stats["v_assignment_tokens"] in (None, (tokens // 4) * 4)


def test_segmented_chunked_decode_reconstruction_order() -> None:
    cache = _pattern_cache(3)
    for step in range(5):
        key, value = _kv(1)
        append_decode(cache, key + step, value + step)
    assert cache.packed_k_tokens == 8
    assert cache_segment_stats(cache)["chunk_tokens"] == 0
    assert cache.centroid_updates_k == 2
    assert cache.centroid_updates_v == 2
    assert cache.k_assignments.shape[2] == cache.packed_k_tokens
    assert cache.v_assignment_idx.shape[2] == cache.packed_v_tokens
    assert cache.v_pattern_mask.shape[2] == cache.packed_v_tokens
    assert reconstruct_full_k(cache).shape[2] == cache.total_tokens
    assert reconstruct_full_v(cache).shape[2] == cache.total_tokens


def test_rolling_semantics_are_intentionally_different_from_chunked() -> None:
    key, value = _kv(6)
    chunked = build_cache_from_prefill(
        key,
        value,
        sink_length=0,
        recent_length=0,
        group_size=4,
        k_bits=4,
        v_bits=4,
        pattern=True,
        k_centroids=torch.zeros(2, 1, 8, dtype=torch.float16),
        v_centroids=torch.zeros(2, 1, 8, dtype=torch.float16),
        cache_mode="segmented_chunked",
        chunk_length=4,
    )
    rolling = build_cache_from_prefill(
        key,
        value,
        sink_length=0,
        recent_length=4,
        group_size=4,
        k_bits=4,
        v_bits=4,
        pattern=True,
        k_centroids=torch.zeros(2, 1, 8, dtype=torch.float16),
        v_centroids=torch.zeros(2, 1, 8, dtype=torch.float16),
        cache_mode="segmented_rolling",
        chunk_length=4,
    )
    assert chunked.packed_k_tokens == 4
    assert cache_segment_stats(chunked)["chunk_tokens"] == 2
    assert rolling.packed_k_tokens == 0
    assert cache_segment_stats(rolling)["pending_history_tokens"] == 2
    assert cache_segment_stats(rolling)["recent_tokens"] == 4
