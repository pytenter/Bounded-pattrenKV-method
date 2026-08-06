from __future__ import annotations

import torch

from models.segmented_cache import (
    append_decode,
    build_cache_from_prefill,
    pattern_chebyshev_center_per_head,
    pattern_gather_centroids,
    reconstruct_full_k,
    reconstruct_full_v,
)


def legacy_minmax_assign(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    diff = x.unsqueeze(2) - centroids.unsqueeze(1)
    distance = diff.amax(dim=-1) - diff.amin(dim=-1)
    return distance.argmin(dim=-1)


def test_synthetic_dynamic_window_matches_legacy_chebyshev_and_minmax() -> None:
    window = torch.arange(1 * 2 * 4 * 8, dtype=torch.float16).reshape(1, 2, 4, 8) / 9
    x = window.permute(1, 0, 2, 3).reshape(2, 4, 8).contiguous()
    old_bank = torch.zeros(2, 1, 8, dtype=torch.float16)
    new_centroid = pattern_chebyshev_center_per_head(x).to(old_bank.dtype)
    legacy_bank = torch.cat([old_bank, new_centroid], dim=1)
    legacy_assign = legacy_minmax_assign(x, legacy_bank).view(2, 1, 4).permute(1, 0, 2).contiguous()
    gathered = pattern_gather_centroids(legacy_assign, legacy_bank)
    assert torch.equal(new_centroid, (x.amin(dim=1, keepdim=True) + x.amax(dim=1, keepdim=True)) * 0.5)
    assert gathered.shape == window.shape
    assert torch.equal(legacy_assign, torch.ones_like(legacy_assign))


def test_segmented_no_sink_recent_128_semantics_matches_legacy_residual_window() -> None:
    key = torch.arange(1 * 2 * 4 * 8, dtype=torch.float16).reshape(1, 2, 4, 8) / 7
    value = key + 0.125
    cache = build_cache_from_prefill(
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
    )
    for step in range(4):
        next_k = torch.full((1, 2, 1, 8), float(step + 1), dtype=torch.float16)
        next_v = next_k + 0.5
        append_decode(cache, next_k, next_v)
    assert cache.centroid_updates_k == 1
    assert cache.centroid_updates_v == 1
    assert cache.k_assignments.shape[2] == cache.packed_k_tokens == 4
    assert cache.v_assignment_idx.shape[2] == cache.packed_v_tokens == 4
    assert reconstruct_full_k(cache).shape[2] == 8
    assert reconstruct_full_v(cache).shape[2] == 8

