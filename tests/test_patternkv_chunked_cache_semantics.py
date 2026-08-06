from __future__ import annotations

import torch

from models.segmented_cache import (
    append_decode,
    build_cache_from_prefill,
    cache_segment_stats,
    deserialize_cache,
    serialize_cache,
)


def _kv(tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens == 0:
        key = torch.empty(1, 2, 0, 8, dtype=torch.float16)
    else:
        key = torch.arange(1 * 2 * tokens * 8, dtype=torch.float16).reshape(1, 2, tokens, 8) / 17
    return key, key + 0.25


def _centroids() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(2, 1, 8, dtype=torch.float16), torch.zeros(2, 1, 8, dtype=torch.float16)


def test_chunked_decode_cadence() -> None:
    key, value = _kv(0)
    k_centroids, v_centroids = _centroids()
    cache = build_cache_from_prefill(
        key,
        value,
        sink_length=0,
        recent_length=0,
        group_size=4,
        k_bits=4,
        v_bits=4,
        pattern=True,
        k_centroids=k_centroids,
        v_centroids=v_centroids,
        cache_mode="segmented_chunked",
        chunk_length=4,
    )
    expected = {
        1: (0, 1, 0),
        2: (0, 2, 0),
        3: (0, 3, 0),
        4: (4, 0, 1),
        5: (4, 1, 1),
        8: (8, 0, 2),
    }
    for step in range(1, 9):
        next_k, next_v = _kv(1)
        append_decode(cache, next_k + step, next_v + step)
        if step in expected:
            packed, chunk, updates = expected[step]
            assert cache.packed_k_tokens == packed
            assert cache.packed_v_tokens == packed
            assert cache_segment_stats(cache)["chunk_tokens"] == chunk
            assert cache.centroid_updates_k == updates
            assert cache.centroid_updates_v == updates


def test_chunked_prefill_cadence() -> None:
    for tokens in (0, 1, 3, 4, 5, 7, 8, 9):
        key, value = _kv(tokens)
        k_centroids, v_centroids = _centroids()
        cache = build_cache_from_prefill(
            key,
            value,
            sink_length=0,
            recent_length=0,
            group_size=4,
            k_bits=4,
            v_bits=4,
            pattern=True,
            k_centroids=k_centroids,
            v_centroids=v_centroids,
            cache_mode="segmented_chunked",
            chunk_length=4,
        )
        assert cache.packed_k_tokens == (tokens // 4) * 4
        assert cache.packed_v_tokens == (tokens // 4) * 4
        assert cache_segment_stats(cache)["chunk_tokens"] == tokens % 4
        assert cache.centroid_updates_k == 0
        assert cache.centroid_updates_v == 0


def test_chunked_serialization_preserves_mode_and_buffers() -> None:
    key, value = _kv(5)
    k_centroids, v_centroids = _centroids()
    cache = build_cache_from_prefill(
        key,
        value,
        sink_length=0,
        recent_length=0,
        group_size=4,
        k_bits=4,
        v_bits=4,
        pattern=True,
        k_centroids=k_centroids,
        v_centroids=v_centroids,
        cache_mode="segmented_chunked",
        chunk_length=4,
    )
    restored = deserialize_cache(serialize_cache(cache), pattern=True)
    assert restored.cache_mode == "segmented_chunked"
    assert restored.chunk_length == 4
    assert restored.packed_k_tokens == 4
    assert cache_segment_stats(restored)["chunk_tokens"] == 1
    assert torch.equal(restored.k_assignments, cache.k_assignments)
    assert torch.equal(restored.v_assignment_idx, cache.v_assignment_idx)
    assert torch.equal(restored.v_pattern_mask, cache.v_pattern_mask)
