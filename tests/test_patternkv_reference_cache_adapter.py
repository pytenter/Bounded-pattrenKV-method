from __future__ import annotations

import torch

from bench.patternkv_equivalence_reference import reference_view_from_legacy_tuple, reference_view_from_segmented_cache
from models.segmented_cache import CHUNKED_CACHE_MODE, PatternQuantizedKVCache, quantize_pack_k_reference, quantize_pack_v_reference


def _make_segmented_cache() -> PatternQuantizedKVCache:
    packed_k_input = torch.arange(256, dtype=torch.float16).reshape(1, 1, 16, 16)
    packed_v_input = torch.arange(256, 512, dtype=torch.float16).reshape(1, 1, 16, 16)
    chunk_k = torch.arange(16, dtype=torch.float16).reshape(1, 1, 1, 16) + 512
    chunk_v = torch.arange(16, dtype=torch.float16).reshape(1, 1, 1, 16) + 528
    packed_k, packed_k_scale, packed_k_zero = quantize_pack_k_reference(packed_k_input, group_size=16, bits=2)
    packed_v, packed_v_scale, packed_v_zero = quantize_pack_v_reference(packed_v_input, group_size=16, bits=2)
    return PatternQuantizedKVCache(
        packed_k=packed_k,
        packed_k_scale=packed_k_scale,
        packed_k_zero=packed_k_zero,
        packed_v=packed_v,
        packed_v_scale=packed_v_scale,
        packed_v_zero=packed_v_zero,
        pending_k=chunk_k,
        pending_v=chunk_v,
        total_tokens=17,
        packed_k_tokens=16,
        packed_v_tokens=16,
        sink_length=0,
        recent_length=0,
        group_size=16,
        k_bits=2,
        v_bits=2,
        cache_mode=CHUNKED_CACHE_MODE,
        chunk_length=16,
        k_assignments=torch.zeros(1, 1, 16, dtype=torch.long),
        v_assignment_idx=torch.zeros(1, 1, 16, dtype=torch.long),
        v_pattern_mask=torch.tensor([[[1] * 16]], dtype=torch.uint8),
        k_centroids=torch.tensor([[[0.5] * 16]], dtype=torch.float16),
        v_centroids=torch.tensor([[[1.0] * 16]], dtype=torch.float16),
    )


def test_reference_view_from_segmented_cache() -> None:
    cache = _make_segmented_cache()
    view = reference_view_from_segmented_cache(cache, num_attention_heads=2, num_key_value_heads=1, head_dim=16)
    assert view.cache_mode == CHUNKED_CACHE_MODE
    assert view.total_tokens == 17
    assert view.packed_k_tokens == 16
    assert view.packed_v_tokens == 16
    assert view.chunk_k is not None and view.chunk_k.shape == (1, 1, 1, 16)
    assert view.chunk_v is not None and view.chunk_v.shape == (1, 1, 1, 16)
    assert torch.equal(view.k_centroids, cache.k_centroids)
    assert torch.equal(view.v_pattern_mask, cache.v_pattern_mask)


def test_reference_view_from_legacy_tuple() -> None:
    cache = _make_segmented_cache()
    legacy = (
        cache.packed_k,
        None,
        cache.packed_k_scale,
        cache.packed_k_zero,
        cache.packed_v,
        None,
        cache.packed_v_scale,
        cache.packed_v_zero,
        cache.total_tokens,
        cache.k_assignments,
        cache.v_pattern_mask,
        cache.v_assignment_idx,
    )
    view = reference_view_from_legacy_tuple(
        legacy,
        chunk_k=cache.pending_k,
        chunk_v=cache.pending_v,
        k_centroids=cache.k_centroids,
        v_centroids=cache.v_centroids,
        total_tokens=cache.total_tokens,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    assert view.cache_mode == "legacy_tuple_chunked"
    assert view.total_tokens == 17
    assert view.packed_k_tokens == 16
    assert view.packed_v_tokens == 16
    assert torch.equal(view.k_assignments, cache.k_assignments)
    assert torch.equal(view.v_assignment_idx, cache.v_assignment_idx)
