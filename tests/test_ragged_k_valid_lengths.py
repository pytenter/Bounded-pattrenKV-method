from __future__ import annotations

import torch

from models.segmented_cache import (
    PatternKVCentroidStatePool,
    PatternQuantizedKVCache,
    append_decode_rolling,
    assemble_ragged_patternkv_cache,
    build_k_segment_validity_mask,
    get_packed_k_tokens_per_request,
    get_ragged_k_counters,
    k_segment_valid_lengths,
    reset_ragged_k_counters,
    set_request_total_tokens,
)


def _actual_like_cache(total: int, packed_tokens: int, *, value: float) -> PatternQuantizedKVCache:
    bits = 2
    group_size = 128
    pack = 32 // bits
    payload_cols = (packed_tokens + pack - 1) // pack
    scale_cols = (packed_tokens + group_size - 1) // group_size
    lengths = _rolling_lengths(total, packed_tokens)
    k_static = torch.tensor([[[value], [value + 1.0]]])
    v_static = torch.tensor([[[value + 2.0], [value + 3.0]]])
    pool = PatternKVCentroidStatePool.create(k_static, v_static, max_slots=1, max_dynamic_centroids=0)
    pool.allocate(torch.tensor([0]))
    cache = PatternQuantizedKVCache(
        sink_k=torch.full((1, 1, lengths["sink"], 128), value),
        sink_v=torch.full((1, 1, lengths["sink"], 128), value),
        packed_k=torch.full((1, 1, 128, payload_cols), int(value), dtype=torch.int32),
        packed_k_scale=torch.ones((1, 1, 128, scale_cols)),
        packed_k_zero=torch.zeros((1, 1, 128, scale_cols)),
        packed_v=torch.zeros((1, 1, packed_tokens, 1), dtype=torch.int32),
        packed_v_scale=torch.zeros((1, 1, packed_tokens, 1)),
        packed_v_zero=torch.zeros((1, 1, packed_tokens, 1)),
        pending_k=torch.full((1, 1, lengths["pending"], 128), value),
        pending_v=torch.full((1, 1, lengths["pending"], 128), value),
        recent_k=torch.full((1, 1, lengths["recent"], 128), value),
        recent_v=torch.full((1, 1, lengths["recent"], 128), value),
        total_tokens=total,
        packed_k_tokens=packed_tokens,
        packed_v_tokens=packed_tokens,
        sink_length=16,
        recent_length=128,
        group_size=group_size,
        k_bits=bits,
        v_bits=2,
        k_assignments=torch.zeros((1, 1, packed_tokens), dtype=torch.long),
        v_assignments=torch.zeros((1, 1, packed_tokens), dtype=torch.uint8),
        v_assignment_idx=torch.zeros((1, 1, packed_tokens), dtype=torch.long),
        v_pattern_mask=torch.zeros((1, 1, packed_tokens), dtype=torch.uint8),
        k_centroids=pool.current_k(torch.tensor([0])),
        v_centroids=pool.current_v(torch.tensor([0])),
        v_precision_mask=torch.zeros((1, packed_tokens), dtype=torch.bool),
        packed_v4=torch.zeros((1, 1, 0, 1), dtype=torch.int32),
        packed_v4_scale=torch.zeros((1, 1, 0, 1)),
        packed_v4_zero=torch.zeros((1, 1, 0, 1)),
        packed_v4_tokens=0,
        centroid_state_pool=pool,
        centroid_state_indices=torch.tensor([0]),
    )
    set_request_total_tokens(cache, [total])
    cache.request_packed_k_tokens = torch.tensor([packed_tokens])
    cache.request_packed_v_tokens = torch.tensor([packed_tokens])
    cache.request_packed_v4_tokens = torch.tensor([0])
    return cache


def _rolling_lengths(total: int, packed: int) -> dict[str, int]:
    sink = min(total, 16)
    non_sink = max(total - sink, 0)
    recent = min(non_sink, 128)
    quantized = max(non_sink - recent, 0)
    return {"sink": sink, "pending": max(quantized - packed, 0), "recent": recent}


def _b2_cache() -> PatternQuantizedKVCache:
    return assemble_ragged_patternkv_cache([
        _actual_like_cache(384, 128, value=3.0),
        _actual_like_cache(513, 256, value=7.0),
    ])


def _value_parts(cache: PatternQuantizedKVCache) -> list[tuple[str, int]]:
    return [
        ("sink", cache.sink_k.shape[2]),
        ("packed", cache.packed_k_tokens),
        ("pending", cache.pending_k.shape[2]),
        ("recent", cache.recent_k.shape[2]),
    ]


def test_ragged_k_valid_lengths_from_cache() -> None:
    cache = _b2_cache()
    assert get_packed_k_tokens_per_request(cache).tolist() == [128, 256]
    lengths = k_segment_valid_lengths(cache)
    assert lengths["sink"].tolist() == [16, 16]
    assert lengths["packed"].tolist() == [128, 256]
    assert lengths["pending"].tolist() == [112, 113]
    assert lengths["recent"].tolist() == [128, 128]


def test_ragged_k_assignment_padding_shape() -> None:
    cache = _b2_cache()
    assert cache.packed_k.shape == (2, 1, 128, 16)
    assert cache.packed_k_scale.shape == (2, 1, 128, 2)
    assert cache.k_assignments.shape == (2, 1, 256)


def test_ragged_k_invalid_tail_masked() -> None:
    cache = _b2_cache()
    mask = build_k_segment_validity_mask(cache, _value_parts(cache))
    assert mask.shape == (2, 513)
    assert int((~mask[0]).sum().item()) == 129
    assert int((~mask[1]).sum().item()) == 0


def test_ragged_k_invalid_tail_sentinel_invariant() -> None:
    cache = _b2_cache()
    mask = build_k_segment_validity_mask(cache, _value_parts(cache))
    scores = torch.randn(2, 4, 1, 513)
    masked = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
    probs = torch.softmax(masked, dim=-1)
    mutated = scores.clone()
    mutated[0, :, :, ~mask[0]] = 1_000_000.0
    mutated_masked = mutated.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
    mutated_probs = torch.softmax(mutated_masked, dim=-1)
    assert torch.allclose(probs[0], mutated_probs[0], atol=0.0, rtol=0.0)
    assert torch.allclose(probs[1], mutated_probs[1], atol=0.0, rtol=0.0)


def test_ragged_k_short_row_does_not_read_long_row() -> None:
    cache = _b2_cache()
    mask = build_k_segment_validity_mask(cache, _value_parts(cache))
    scores = torch.randn(2, 4, 1, 513)
    base_a = torch.softmax(scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min), dim=-1)[0]
    scores[1, :, :, 300:] = -999_999.0
    got_a = torch.softmax(scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min), dim=-1)[0]
    assert torch.allclose(base_a, got_a, atol=0.0, rtol=0.0)


def test_ragged_k_reorder_preserves_valid_lengths() -> None:
    ab = assemble_ragged_patternkv_cache([_actual_like_cache(384, 128, value=3.0), _actual_like_cache(513, 256, value=7.0)])
    ba = assemble_ragged_patternkv_cache([_actual_like_cache(513, 256, value=7.0), _actual_like_cache(384, 128, value=3.0)])
    assert get_packed_k_tokens_per_request(ab).tolist() == [128, 256]
    assert get_packed_k_tokens_per_request(ba).tolist() == [256, 128]


def test_ragged_k_equal_length_matches_fixed_batch() -> None:
    cache = assemble_ragged_patternkv_cache([_actual_like_cache(512, 256, value=3.0), _actual_like_cache(512, 256, value=7.0)])
    mask = build_k_segment_validity_mask(cache, _value_parts(cache))
    assert bool(mask.all().item())
    assert get_packed_k_tokens_per_request(cache).tolist() == [256, 256]


def test_ragged_k_mask_gqa_shape() -> None:
    cache = _b2_cache()
    mask = build_k_segment_validity_mask(cache, _value_parts(cache))
    scores = torch.zeros(2, 32, 1, 513)
    masked = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
    assert masked.shape == (2, 32, 1, 513)
    assert (masked[0, :, :, ~mask[0]] < -1e20).all()


def test_ragged_k_no_fp16_history_materialization() -> None:
    reset_ragged_k_counters()
    cache = _b2_cache()
    build_k_segment_validity_mask(cache, _value_parts(cache))
    assert get_ragged_k_counters()["historical_fp16_k_materialization"] == 0


def test_ragged_k_decode_update_independent() -> None:
    cache = _b2_cache()
    set_request_total_tokens(cache, [385, 514])
    lengths = k_segment_valid_lengths(cache)
    assert lengths["pending"].tolist() == [113, 114]
    assert lengths["packed"].tolist() == [128, 256]


def test_ragged_decode_append_preserves_pending_valid_prefix() -> None:
    cache = _b2_cache()
    cache.recent_k[0, :, 0, :].fill_(42.0)
    cache.recent_v[0, :, 0, :].fill_(43.0)
    cache.pending_k[0, :, 112:, :].fill_(-100.0)
    cache.pending_v[0, :, 112:, :].fill_(-101.0)
    key_states = torch.full((2, 1, 1, 128), 99.0)
    value_states = torch.full((2, 1, 1, 128), 100.0)
    append_decode_rolling(cache, key_states, value_states)
    lengths = k_segment_valid_lengths(cache)
    assert lengths["pending"].tolist() == [113, 114]
    assert torch.equal(cache.pending_k[0, :, 112, :], torch.full((1, 128), 42.0))
    assert torch.equal(cache.pending_v[0, :, 112, :], torch.full((1, 128), 43.0))
    assert torch.equal(cache.recent_k[0, :, -1, :], torch.full((1, 128), 99.0))


def test_ragged_k_boundary_transition_independent() -> None:
    cache = assemble_ragged_patternkv_cache([
        _actual_like_cache(399, 128, value=3.0),
        _actual_like_cache(512, 256, value=7.0),
    ])
    before = k_segment_valid_lengths(cache)
    set_request_total_tokens(cache, [400, 513])
    after = k_segment_valid_lengths(cache)
    assert before["pending"].tolist() == [127, 112]
    assert after["pending"].tolist() == [128, 113]
    assert after["packed"].tolist() == [128, 256]
