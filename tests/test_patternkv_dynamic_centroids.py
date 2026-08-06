from __future__ import annotations

import torch

from models.segmented_cache import (
    PatternQuantizedKVCache,
    append_decode,
    build_cache_from_prefill,
    cache_segment_stats,
    pattern_chebyshev_center_per_head,
    pattern_gather_centroids,
    validate_cache,
)


def make_kv(tokens: int, *, heads: int = 2, dim: int = 16, offset: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(heads * tokens * dim, dtype=torch.float16).reshape(1, heads, tokens, dim)
    return base / 17 + offset, base / 19 + offset


def test_dynamic_k_centroid_creation_and_assignment_alignment() -> None:
    key, value = make_kv(6)
    k_centroids = torch.zeros(2, 1, 16, dtype=torch.float16)
    v_centroids = torch.zeros(2, 1, 16, dtype=torch.float16)
    cache = build_cache_from_prefill(
        key,
        value,
        sink_length=2,
        recent_length=4,
        group_size=4,
        k_bits=2,
        v_bits=2,
        pattern=True,
        k_centroids=k_centroids,
        v_centroids=v_centroids,
    )
    assert isinstance(cache, PatternQuantizedKVCache)
    assert cache.packed_k_tokens == 0
    for step in range(3):
        next_k, next_v = make_kv(1, offset=10 + step)
        append_decode(cache, next_k, next_v)
        assert cache.centroid_updates_k == 0
    next_k, next_v = make_kv(1, offset=13)
    append_decode(cache, next_k, next_v)
    assert cache.centroid_updates_k == 1
    assert cache.k_centroids.shape == (2, 2, 16)
    assert cache.k_assignments.shape == (1, 2, 4)
    assert int(cache.k_assignments.max()) < cache.k_centroids.shape[1]
    assert cache.k_assignments.shape[2] == cache.packed_k_tokens
    assert pattern_gather_centroids(cache.k_assignments, cache.k_centroids).shape == (1, 2, 4, 16)
    validate_cache(cache)


def test_dynamic_centroid_is_per_head_chebyshev_center() -> None:
    window, _ = make_kv(4)
    x = window.permute(1, 0, 2, 3).reshape(2, 4, 16).contiguous()
    expected = (x.amin(dim=1, keepdim=True) + x.amax(dim=1, keepdim=True)) * 0.5
    assert torch.equal(pattern_chebyshev_center_per_head(x), expected)
    assert not torch.equal(expected[0], expected[1])


def test_dynamic_update_cadence_uses_pending_pack_window_only() -> None:
    key, value = make_kv(6)
    cache = build_cache_from_prefill(
        key,
        value,
        sink_length=2,
        recent_length=4,
        group_size=4,
        k_bits=4,
        v_bits=2,
        pattern=True,
        k_centroids=torch.zeros(2, 1, 16, dtype=torch.float16),
        v_centroids=torch.zeros(2, 1, 16, dtype=torch.float16),
    )
    updates = []
    packed = []
    for step in range(8):
        next_k, next_v = make_kv(1, offset=20 + step)
        append_decode(cache, next_k, next_v)
        updates.append(cache.centroid_updates_k)
        packed.append(cache.packed_k_tokens)
        stats = cache_segment_stats(cache)
        assert stats["recent_tokens"] == 4
        assert stats["sink_tokens"] == 2
        validate_cache(cache)
    assert updates == [0, 0, 0, 1, 1, 1, 1, 2]
    assert packed == [0, 0, 0, 4, 4, 4, 4, 8]
