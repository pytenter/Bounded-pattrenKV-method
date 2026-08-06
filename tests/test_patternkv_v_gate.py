from __future__ import annotations

import torch

from models.segmented_cache import (
    PatternQuantizedKVCache,
    build_cache_from_prefill,
    pattern_gather_centroids,
    pattern_v_threshold_and_mask,
    reconstruct_full_v,
    validate_cache,
)


def test_v_gate_true_subtracts_and_reconstructs_with_centroid() -> None:
    value = torch.tensor([[[[2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]]]], dtype=torch.float16)
    centroid = value.clone().squeeze(0).squeeze(1).unsqueeze(0)
    rho, mask = pattern_v_threshold_and_mask(value, value)
    assert rho.shape == (1, 1, 1, 1)
    assert mask.item() is True
    cache = build_cache_from_prefill(
        value,
        value,
        sink_length=0,
        recent_length=0,
        group_size=1,
        k_bits=4,
        v_bits=4,
        pattern=True,
        k_centroids=centroid,
        v_centroids=centroid,
        k_assignments=torch.zeros(1, 1, 1, dtype=torch.long),
        v_assignment_idx=torch.zeros(1, 1, 1, dtype=torch.long),
        v_pattern_mask=torch.ones(1, 1, 1, dtype=torch.uint8),
    )
    assert isinstance(cache, PatternQuantizedKVCache)
    assert torch.equal(cache.v_pattern_mask, torch.ones(1, 1, 1, dtype=torch.uint8))
    assert torch.allclose(reconstruct_full_v(cache), value, atol=1e-3, rtol=1e-3)
    validate_cache(cache)


def test_v_gate_false_quantizes_raw_without_adding_centroid() -> None:
    value = torch.tensor([[[[1.0, 1.5, 2.0, 2.5, 7.0, 7.5, 8.0, 8.5]]]], dtype=torch.float16)
    centroid = torch.full((1, 1, 8), 100.0, dtype=torch.float16)
    cache = build_cache_from_prefill(
        value,
        value,
        sink_length=0,
        recent_length=0,
        group_size=1,
        k_bits=4,
        v_bits=4,
        pattern=True,
        k_centroids=centroid,
        v_centroids=centroid,
        k_assignments=torch.zeros(1, 1, 1, dtype=torch.long),
        v_assignment_idx=torch.zeros(1, 1, 1, dtype=torch.long),
        v_pattern_mask=torch.zeros(1, 1, 1, dtype=torch.uint8),
    )
    raw_reconstruct = reconstruct_full_v(cache)
    assert raw_reconstruct.mean() < 20
    assert not torch.allclose(raw_reconstruct, value + centroid.unsqueeze(0), atol=1e-3, rtol=1e-3)
    validate_cache(cache)


def test_v_assignment_gather_and_mask_lengths() -> None:
    centroids = torch.arange(2 * 3 * 8, dtype=torch.float16).reshape(2, 3, 8)
    idx = torch.tensor([[[0, 1, 2, 1], [2, 1, 0, 2]]], dtype=torch.long)
    gathered = pattern_gather_centroids(idx, centroids)
    assert gathered.shape == (1, 2, 4, 8)
    assert torch.equal(gathered[0, 0, 2], centroids[0, 2])
