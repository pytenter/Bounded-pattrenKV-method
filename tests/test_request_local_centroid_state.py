from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.llama_patternkv import patternkv_mixed_value_attention
from models.segmented_cache import (
    PatternKVCentroidStatePool,
    _sync_cache_centroid_views,
    append_decode,
    build_cache_from_prefill,
    validate_cache,
)


GROUP_SIZE = 128
NH = 4
NH_KV = 2
HEAD_DIM = 128
FUSED_HEAD_DIM = 128


def _static_centroids(head_dim: int, *, device: torch.device) -> torch.Tensor:
    zeros = torch.zeros(NH_KV, 1, head_dim, dtype=torch.float16, device=device)
    return zeros


def _empty_cache(batch: int, slots: torch.Tensor, *, head_dim: int = HEAD_DIM, device: torch.device | None = None, **kwargs):
    device = device or slots.device
    key = torch.empty(batch, NH_KV, 0, head_dim, dtype=torch.float16, device=device)
    value = torch.empty_like(key)
    centroids = _static_centroids(head_dim, device=device)
    return build_cache_from_prefill(
        key,
        value,
        sink_length=0,
        recent_length=0,
        group_size=GROUP_SIZE,
        k_bits=2,
        v_bits=2,
        pattern=True,
        k_centroids=centroids,
        v_centroids=centroids,
        cache_mode="segmented_chunked",
        chunk_length=GROUP_SIZE,
        centroid_state_indices=slots,
        **kwargs,
    )


def _kv(batch: int, tokens: int, *, head_dim: int = HEAD_DIM, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.arange(tokens, dtype=torch.float32, device=device).view(1, 1, tokens, 1)
    dim = torch.arange(head_dim, dtype=torch.float32, device=device).view(1, 1, 1, head_dim)
    heads = torch.arange(NH_KV, dtype=torch.float32, device=device).view(1, NH_KV, 1, 1)
    req = torch.arange(batch, dtype=torch.float32, device=device).view(batch, 1, 1, 1)
    key = req * 17.0 + heads * 3.0 + pos * (0.03 + req * 0.01) + dim * 0.001
    value = req * -11.0 + heads * 5.0 + pos * (0.02 + req * 0.015) - dim * 0.0007
    return key.to(torch.float16), value.to(torch.float16)


def _append_tokens(cache, key: torch.Tensor, value: torch.Tensor, steps: int) -> None:
    for step in range(steps):
        append_decode(cache, key[:, :, step : step + 1, :], value[:, :, step : step + 1, :])
    validate_cache(cache)


def _independent_cache_for_row(key: torch.Tensor, value: torch.Tensor, row: int, steps: int):
    slots = torch.tensor([0], dtype=torch.long, device=key.device)
    cache = _empty_cache(1, slots, head_dim=key.shape[-1], device=key.device)
    _append_tokens(cache, key[row : row + 1], value[row : row + 1], steps)
    return cache


@pytest.mark.parametrize("batch,slot_values", [(2, [3, 0]), (4, [7, 2, 5, 1])])
@pytest.mark.parametrize("steps", [127, 128, 129, 255, 256, 257])
def test_dynamic_centroids_are_request_local_across_flush_boundaries(batch: int, slot_values: list[int], steps: int) -> None:
    device = torch.device("cpu")
    slots = torch.tensor(slot_values, dtype=torch.long, device=device)
    key, value = _kv(batch, steps, device=device)
    cache = _empty_cache(batch, slots, device=device)
    _append_tokens(cache, key, value, steps)

    expected_updates = steps // GROUP_SIZE
    assert cache.centroid_updates_k == expected_updates
    pool = cache.centroid_state_pool
    assert pool is not None
    for row, slot in enumerate(slots.tolist()):
        ref = _independent_cache_for_row(key, value, row, steps)
        ref_pool = ref.centroid_state_pool
        assert ref_pool is not None
        count = int(pool.k_counts[slot].item())
        ref_count = int(ref_pool.k_counts[0].item())
        assert count == ref_count == 1 + expected_updates
        torch.testing.assert_close(pool.k_centroid_pool[slot, :, :count, :], ref.k_centroids[:, :count, :], rtol=0, atol=0)
        torch.testing.assert_close(pool.v_centroid_pool[slot, :, :count, :], ref.v_centroids[:, :count, :], rtol=0, atol=0)
        if cache.k_assignments is not None:
            torch.testing.assert_close(cache.k_assignments[row : row + 1], ref.k_assignments, rtol=0, atol=0)
        if cache.v_assignment_idx is not None:
            torch.testing.assert_close(cache.v_assignment_idx[row : row + 1], ref.v_assignment_idx, rtol=0, atol=0)
            torch.testing.assert_close(cache.v_pattern_mask[row : row + 1], ref.v_pattern_mask, rtol=0, atol=0)


def test_centroid_slot_reorder_and_flush_mask_do_not_cross_contaminate() -> None:
    device = torch.device("cpu")
    slots = torch.tensor([5, 2], dtype=torch.long, device=device)
    key, value = _kv(2, GROUP_SIZE, device=device)
    cache = _empty_cache(2, slots, device=device, centroid_flush_mask=torch.tensor([True, False], device=device))
    _append_tokens(cache, key, value, GROUP_SIZE)

    pool = cache.centroid_state_pool
    assert pool is not None
    assert int(pool.k_counts[5].item()) == 2
    assert int(pool.k_counts[2].item()) == 1
    assert int(cache.k_assignments[1].max().item()) == 0

    before_slot5 = pool.k_centroid_pool[5, :, :2, :].clone()
    before_slot2 = pool.k_centroid_pool[2, :, :1, :].clone()
    cache.centroid_state_indices = torch.tensor([2, 5], dtype=torch.long, device=device)
    _sync_cache_centroid_views(cache)
    torch.testing.assert_close(cache.k_centroids[0, :, :1, :], before_slot2, rtol=0, atol=0)
    torch.testing.assert_close(cache.k_centroids[1, :, :2, :], before_slot5, rtol=0, atol=0)


def test_centroid_state_pool_reuses_freed_slots_with_clean_counts() -> None:
    device = torch.device("cpu")
    centroids = _static_centroids(HEAD_DIM, device=device)
    v_centroids = torch.cat([centroids, torch.ones_like(centroids)], dim=1)
    pool = PatternKVCentroidStatePool.create(centroids, v_centroids, max_slots=4, max_dynamic_centroids=2)
    slot = torch.tensor([3], dtype=torch.long, device=device)
    pool.allocate(slot)
    pool.k_counts[slot] += 2
    pool.v_counts[slot] += 1
    pool.update_counts_k[slot] += 2
    pool.update_counts_v[slot] += 1
    pool.last_flush_pos[slot] = 256
    pool.free(slot)
    assert not bool(pool.active[slot].item())
    assert int(pool.k_counts[slot].item()) == 1
    assert int(pool.v_counts[slot].item()) == 2
    assert int(pool.update_counts_k[slot].item()) == 0
    assert int(pool.update_counts_v[slot].item()) == 0
    assert int(pool.last_flush_pos[slot].item()) == 0
    pool.allocate(slot)
    assert bool(pool.active[slot].item())
    torch.testing.assert_close(pool.k_centroid_pool[3, :, :1, :], centroids, rtol=0, atol=0)
    torch.testing.assert_close(pool.v_centroid_pool[3, :, :2, :], v_centroids, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="request-local fused Value test requires a GPU")
def test_fused_page_value_uses_request_local_centroids(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_RUNTIME_NH", str(NH))
    monkeypatch.setenv("PATTERNKV_MIXED_V_BACKEND", "fused_page")
    device = torch.device("cuda")
    slots = torch.tensor([7, 1], dtype=torch.long, device=device)
    key, value = _kv(2, GROUP_SIZE, head_dim=FUSED_HEAD_DIM, device=device)
    cache = _empty_cache(
        2,
        slots,
        head_dim=FUSED_HEAD_DIM,
        device=device,
        value_objective="v_dir",
        v_precision_selector="causal_v4",
        v4_budget_fraction=0.25,
    )
    _append_tokens(cache, key, value, GROUP_SIZE)
    generator = torch.Generator(device=device).manual_seed(3090)
    weights = torch.softmax(torch.randn(2, NH, 1, cache.packed_v_tokens, device=device, dtype=torch.float16, generator=generator), dim=-1)
    module = SimpleNamespace(group_size=GROUP_SIZE, num_key_value_groups=NH // NH_KV, num_heads=NH, num_key_value_heads=NH_KV)
    got = patternkv_mixed_value_attention(module, cache, weights, cache.v_pattern_mask, cache.packed_v_tokens)
    refs = []
    for row in range(2):
        ref = _empty_cache(
            1,
            torch.tensor([0], dtype=torch.long, device=device),
            head_dim=FUSED_HEAD_DIM,
            device=device,
            value_objective="v_dir",
            v_precision_selector="causal_v4",
            v4_budget_fraction=0.25,
        )
        _append_tokens(ref, key[row : row + 1], value[row : row + 1], GROUP_SIZE)
        refs.append(patternkv_mixed_value_attention(module, ref, weights[row : row + 1], ref.v_pattern_mask, ref.packed_v_tokens))
    ref_out = torch.cat(refs, dim=0)
    torch.testing.assert_close(got, ref_out, rtol=0, atol=0)
