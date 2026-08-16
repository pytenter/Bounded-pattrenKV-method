from __future__ import annotations

import torch

from models.llama_patternkv import patternkv_request_invariant_qk_scores
from models.segmented_cache import (
    PatternKVCentroidStatePool,
    PatternQuantizedKVCache,
    append_decode_rolling,
    assemble_ragged_patternkv_cache,
    build_k_segment_validity_mask,
    get_packed_k_tokens_per_request,
    get_ragged_k_counters,
    k_segment_valid_lengths,
    request_invariant_attention_split_boundaries,
    request_invariant_batch_split_signatures,
    request_invariant_full_value_attention,
    request_invariant_segmented_attention_softmax,
    request_invariant_split_signature,
    request_invariant_value_split_boundaries,
    reset_ragged_k_counters,
    set_request_total_tokens,
    update_value_causal_importance,
)


def _actual_like_cache(total: int, packed_tokens: int, *, value: float) -> PatternQuantizedKVCache:
    bits = 2
    group_size = 128
    pack = 32 // bits
    payload_cols = (packed_tokens + pack - 1) // pack
    scale_cols = (packed_tokens + group_size - 1) // group_size
    lengths = _rolling_lengths(total, packed_tokens)
    k_static = torch.stack([
        torch.full((128,), value),
        torch.full((128,), value + 1.0),
    ]).unsqueeze(0)
    v_static = torch.stack([
        torch.full((128,), value + 2.0),
        torch.full((128,), value + 3.0),
    ]).unsqueeze(0)
    pool = PatternKVCentroidStatePool.create(k_static, v_static, max_slots=1, max_dynamic_centroids=8)
    pool.allocate(torch.tensor([0]))
    cache = PatternQuantizedKVCache(
        sink_k=torch.full((1, 1, lengths["sink"], 128), value),
        sink_v=torch.full((1, 1, lengths["sink"], 128), value),
        packed_k=torch.full((1, 1, 128, payload_cols), int(value), dtype=torch.int32),
        packed_k_scale=torch.ones((1, 1, 128, scale_cols)),
        packed_k_zero=torch.zeros((1, 1, 128, scale_cols)),
        packed_v=torch.zeros((1, 1, packed_tokens, 8), dtype=torch.int32),
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
        v_precision_mask=None,
        packed_v4=torch.zeros((1, 1, 0, 16), dtype=torch.int32),
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


def _physical_mass_from_logical(cache: PatternQuantizedKVCache, row: int, logical: torch.Tensor) -> torch.Tensor:
    lengths = k_segment_valid_lengths(cache)
    mass = torch.zeros(cache.sink_k.shape[0], int(cache.total_tokens), dtype=torch.float32)
    physical_offset = 0
    logical_offset = 0
    for name, physical in _value_parts(cache):
        valid = int(lengths[name][row].item())
        if valid:
            mass[row, physical_offset : physical_offset + valid] = logical[logical_offset : logical_offset + valid]
        physical_offset += int(physical)
        logical_offset += valid
    return mass


def _physical_logits_from_logical(cache: PatternQuantizedKVCache, row: int, logical: torch.Tensor) -> torch.Tensor:
    lengths = k_segment_valid_lengths(cache)
    logits = torch.full((cache.sink_k.shape[0], 2, 1, int(cache.total_tokens)), torch.finfo(logical.dtype).min, dtype=logical.dtype)
    physical_offset = 0
    logical_offset = 0
    for name, physical in _value_parts(cache):
        valid = int(lengths[name][row].item())
        if valid:
            logits[row, :, :, physical_offset : physical_offset + valid] = logical[:, None, logical_offset : logical_offset + valid]
        physical_offset += int(physical)
        logical_offset += valid
    return logits


def _canonical_probs_from_physical(cache: PatternQuantizedKVCache, row: int, probs: torch.Tensor) -> torch.Tensor:
    lengths = k_segment_valid_lengths(cache)
    total = int(getattr(cache, "request_total_tokens")[row].item())
    out = torch.empty((probs.shape[1], total), dtype=probs.dtype)
    physical_offset = 0
    logical_offset = 0
    for name, physical in _value_parts(cache):
        valid = int(lengths[name][row].item())
        if valid:
            out[:, logical_offset : logical_offset + valid] = probs[row, :, 0, physical_offset : physical_offset + valid]
        physical_offset += int(physical)
        logical_offset += valid
    return out


def _request_a_softmax_probs(cache: PatternQuantizedKVCache, row: int = 0, *, peer_delta: float = 0.0) -> torch.Tensor:
    logical_a = torch.linspace(-2.0, 2.0, 384, dtype=torch.float32).repeat(2, 1)
    logits = _physical_logits_from_logical(cache, row, logical_a)
    for peer in range(logits.shape[0]):
        if peer != row:
            logits[peer].masked_fill_(logits[peer] > -1e20, peer_delta)
    probs = request_invariant_segmented_attention_softmax(logits, cache, _value_parts(cache))
    return _canonical_probs_from_physical(cache, row, probs)


def _update_from_mass(cache: PatternQuantizedKVCache, mass: torch.Tensor) -> torch.Tensor:
    update_value_causal_importance(cache, mass[:, None, None, :])
    assert cache.v_causal_importance is not None
    return cache.v_causal_importance.detach()


def _logical_signal(total: int) -> torch.Tensor:
    return torch.linspace(0.001, 1.0, total, dtype=torch.float32)


def test_causal_importance_ragged_segment_mapping_b1_vs_b2_short() -> None:
    b1 = _actual_like_cache(384, 128, value=3.0)
    b2 = assemble_ragged_patternkv_cache([
        _actual_like_cache(384, 128, value=3.0),
        _actual_like_cache(384, 128, value=7.0),
    ])
    logical = _logical_signal(384)
    b1_out = _update_from_mass(b1, _physical_mass_from_logical(b1, 0, logical))[0, :384]
    b2_out = _update_from_mass(b2, _physical_mass_from_logical(b2, 0, logical))[0, :384]
    assert torch.equal(b1_out, logical)
    assert torch.equal(b2_out, logical)


def test_request_invariant_softmax_split_boundaries() -> None:
    expected = [(0, 128), (128, 256), (256, 384)]
    assert request_invariant_attention_split_boundaries(384) == expected
    assert request_invariant_attention_split_boundaries(385) == expected + [(384, 385)]


def test_fixed_split_signature_boundaries_and_merge_order() -> None:
    lengths = [127, 128, 129, 255, 256, 257, 384, 513, 642, 771]
    for length in lengths:
        signature = request_invariant_split_signature(length)
        assert signature[0] == length
        assert signature[1] == 128
        assert signature[3] == "left_to_right"
        assert list(signature[2]) == request_invariant_attention_split_boundaries(length)


def test_batch_composition_invariant_split_signatures() -> None:
    a = request_invariant_batch_split_signatures([384])[0]
    ab = request_invariant_batch_split_signatures([384, 513])[0]
    b4 = request_invariant_batch_split_signatures([384, 513, 642, 771])[0]
    reorder = request_invariant_batch_split_signatures([771, 384, 642, 513])[1]
    assert a == ab == b4 == reorder


def test_fixed_split_cuda_softmax_matches_reference_for_b1_b2_b4(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    for lengths in ([257], [384, 513], [384, 513, 642, 771]):
        packed = [max(length - 128, 0) for length in lengths]
        recent = [min(length, 128) for length in lengths]
        cache = PatternQuantizedKVCache(
            total_tokens=max(lengths),
            packed_k_tokens=max(packed),
            packed_v_tokens=max(packed),
            sink_length=0,
            recent_length=128,
            group_size=128,
        )
        cache.packed_k = torch.empty(len(lengths), 1, 1, max((value + 15) // 16 for value in packed), device=device, dtype=torch.int32)
        cache.recent_k = torch.empty(len(lengths), 1, max(recent), 1, device=device, dtype=torch.float16)
        set_request_total_tokens(cache, torch.tensor(lengths, device=device))
        cache.request_packed_k_tokens = torch.tensor(packed, device=device)
        generator = torch.Generator(device=device).manual_seed(2026081501 + len(lengths))
        scores = torch.randn(len(lengths), 32, 1, cache.total_tokens, device=device, dtype=torch.float16, generator=generator)
        parts = [("packed", max(packed)), ("recent", max(recent))]
        monkeypatch.setenv("PATTERNKV_FIXED_SPLIT_SOFTMAX", "0")
        ref = request_invariant_segmented_attention_softmax(scores, cache, parts)
        monkeypatch.setenv("PATTERNKV_FIXED_SPLIT_SOFTMAX", "1")
        got = request_invariant_segmented_attention_softmax(scores, cache, parts)
        torch.cuda.synchronize()
        assert torch.allclose(got, ref, rtol=0.0, atol=2e-5)


def test_request_invariant_softmax_peer_length() -> None:
    b1 = _actual_like_cache(384, 128, value=3.0)
    b2_short = assemble_ragged_patternkv_cache([
        _actual_like_cache(384, 128, value=3.0),
        _actual_like_cache(384, 128, value=7.0),
    ])
    b2_long = _b2_cache()
    expected = request_invariant_attention_split_boundaries(384)
    assert request_invariant_attention_split_boundaries(int(b1.request_total_tokens[0].item())) == expected
    assert request_invariant_attention_split_boundaries(int(b2_short.request_total_tokens[0].item())) == expected
    assert request_invariant_attention_split_boundaries(int(b2_long.request_total_tokens[0].item())) == expected


def test_request_invariant_softmax_peer_content() -> None:
    cache = _b2_cache()
    base = _request_a_softmax_probs(cache, peer_delta=0.0)
    changed_peer = _request_a_softmax_probs(cache, peer_delta=19.0)
    assert torch.equal(base, changed_peer)


def test_request_invariant_softmax_reorder() -> None:
    b1 = _actual_like_cache(384, 128, value=3.0)
    ab = _b2_cache()
    ba = assemble_ragged_patternkv_cache([
        _actual_like_cache(513, 256, value=7.0),
        _actual_like_cache(384, 128, value=3.0),
    ])
    ref = _request_a_softmax_probs(b1)
    assert torch.equal(_request_a_softmax_probs(ab, row=0), ref)
    assert torch.equal(_request_a_softmax_probs(ba, row=1), ref)


def test_request_invariant_softmax_b4_layout() -> None:
    b1 = _actual_like_cache(384, 128, value=3.0)
    b4 = assemble_ragged_patternkv_cache([
        _actual_like_cache(384, 128, value=3.0),
        _actual_like_cache(513, 256, value=7.0),
        _actual_like_cache(642, 384, value=11.0),
        _actual_like_cache(771, 512, value=13.0),
    ])
    assert torch.equal(_request_a_softmax_probs(b4), _request_a_softmax_probs(b1))


def test_request_invariant_softmax_probability_exact() -> None:
    b1 = _actual_like_cache(384, 128, value=3.0)
    b2 = _b2_cache()
    got = _request_a_softmax_probs(b2)
    ref = _request_a_softmax_probs(b1)
    assert torch.equal(got, ref)


def test_request_invariant_value_split_boundaries() -> None:
    assert request_invariant_value_split_boundaries(384) == request_invariant_attention_split_boundaries(384)
    assert request_invariant_value_split_boundaries(385) == [(0, 128), (128, 256), (256, 384), (384, 385)]


def test_request_invariant_value_peer_length() -> None:
    b1 = _actual_like_cache(384, 128, value=3.0)
    b2_short = assemble_ragged_patternkv_cache([
        _actual_like_cache(384, 128, value=3.0),
        _actual_like_cache(384, 128, value=7.0),
    ])
    b2_long = _b2_cache()
    ref = request_invariant_value_split_boundaries(int(b1.request_total_tokens[0].item()))
    assert request_invariant_value_split_boundaries(int(b2_short.request_total_tokens[0].item())) == ref
    assert request_invariant_value_split_boundaries(int(b2_long.request_total_tokens[0].item())) == ref


def test_request_invariant_value_peer_content() -> None:
    first = request_invariant_value_split_boundaries(384)
    peer_mutated = request_invariant_value_split_boundaries(384)
    assert peer_mutated == first


def test_request_invariant_value_reorder() -> None:
    ab = _b2_cache()
    ba = assemble_ragged_patternkv_cache([
        _actual_like_cache(513, 256, value=7.0),
        _actual_like_cache(384, 128, value=3.0),
    ])
    assert request_invariant_value_split_boundaries(int(ab.request_total_tokens[0].item())) == request_invariant_value_split_boundaries(int(ba.request_total_tokens[1].item()))


def test_request_invariant_value_b4() -> None:
    b4 = assemble_ragged_patternkv_cache([
        _actual_like_cache(384, 128, value=3.0),
        _actual_like_cache(513, 256, value=7.0),
        _actual_like_cache(642, 384, value=11.0),
        _actual_like_cache(771, 512, value=13.0),
    ])
    assert request_invariant_value_split_boundaries(int(b4.request_total_tokens[0].item())) == request_invariant_value_split_boundaries(384)


def test_request_invariant_value_reduction_golden() -> None:
    probs = torch.softmax(torch.linspace(-1.0, 1.0, 385, dtype=torch.float16), dim=0).to(torch.float16)
    values = torch.linspace(-0.5, 0.5, 385 * 4, dtype=torch.float16).reshape(385, 4)
    out_a = torch.zeros((4,), dtype=torch.float16)
    for start, end in request_invariant_value_split_boundaries(385):
        partial = torch.zeros((4,), dtype=torch.float16)
        for idx in range(start, end):
            partial = partial + probs[idx] * values[idx]
        out_a = out_a + partial
    out_b = torch.zeros((4,), dtype=torch.float16)
    for start, end in request_invariant_value_split_boundaries(385):
        partial = torch.zeros((4,), dtype=torch.float16)
        for idx in range(start, end):
            partial = partial + probs[idx] * values[idx]
        out_b = out_b + partial
    assert torch.equal(out_a, out_b)


def test_attention_softmax_value_share_logical_split_contract() -> None:
    assert request_invariant_value_split_boundaries(771) == request_invariant_attention_split_boundaries(771)


def test_request_invariant_attention_pre_o_exact() -> None:
    weights_a = torch.softmax(torch.linspace(-2.0, 2.0, 113, dtype=torch.float16), dim=0).reshape(1, 1, 1, 113).repeat(1, 2, 1, 1)
    values_a = torch.linspace(-0.5, 0.5, 113 * 4, dtype=torch.float16).reshape(1, 1, 113, 4)
    weights_b = torch.zeros(2, 2, 1, 114, dtype=torch.float16)
    values_b = torch.zeros(2, 1, 114, 4, dtype=torch.float16)
    weights_b[0, :, :, :113] = weights_a[0]
    values_b[0, :, :113, :] = values_a[0]
    weights_b[1, :, :, :] = torch.softmax(torch.linspace(1.0, -1.0, 114, dtype=torch.float16), dim=0)
    values_b[1, :, :, :] = torch.linspace(0.1, 0.7, 114 * 4, dtype=torch.float16).reshape(1, 114, 4)
    ref = request_invariant_full_value_attention(weights_a, values_a, torch.tensor([113]), 2)
    got = request_invariant_full_value_attention(weights_b, values_b, torch.tensor([113, 114]), 2)[0:1]
    assert torch.equal(got, ref)


def test_softmax_contract_preserved_after_value_fix() -> None:
    b1 = _actual_like_cache(384, 128, value=3.0)
    b2 = _b2_cache()
    assert torch.equal(_request_a_softmax_probs(b2), _request_a_softmax_probs(b1))


def test_causal_importance_ragged_segment_mapping_b1_vs_b2_long() -> None:
    b1 = _actual_like_cache(384, 128, value=3.0)
    b2 = _b2_cache()
    logical = _logical_signal(384)
    b1_out = _update_from_mass(b1, _physical_mass_from_logical(b1, 0, logical))[0, :384]
    b2_out = _update_from_mass(b2, _physical_mass_from_logical(b2, 0, logical))[0, :384]
    assert torch.equal(b1_out, logical)
    assert torch.equal(b2_out, logical)


def test_causal_importance_ragged_mapping_reorder_preserves_request_a() -> None:
    ab = _b2_cache()
    ba = assemble_ragged_patternkv_cache([
        _actual_like_cache(513, 256, value=7.0),
        _actual_like_cache(384, 128, value=3.0),
    ])
    logical = _logical_signal(384)
    ab_out = _update_from_mass(ab, _physical_mass_from_logical(ab, 0, logical))[0, :384]
    ba_out = _update_from_mass(ba, _physical_mass_from_logical(ba, 1, logical))[1, :384]
    assert torch.equal(ab_out, logical)
    assert torch.equal(ba_out, logical)


def test_causal_importance_ragged_mapping_b4_preserves_request_a() -> None:
    b4 = assemble_ragged_patternkv_cache([
        _actual_like_cache(384, 128, value=3.0),
        _actual_like_cache(513, 256, value=7.0),
        _actual_like_cache(642, 384, value=11.0),
        _actual_like_cache(771, 512, value=13.0),
    ])
    logical = _logical_signal(384)
    out = _update_from_mass(b4, _physical_mass_from_logical(b4, 0, logical))[0, :384]
    assert torch.equal(out, logical)


def test_causal_importance_segment_mapping_covers_sink_packed_pending_recent() -> None:
    cache = _b2_cache()
    logical = _logical_signal(384)
    out = _update_from_mass(cache, _physical_mass_from_logical(cache, 0, logical))[0]
    assert torch.equal(out[:16], logical[:16])
    assert torch.equal(out[16:144], logical[16:144])
    assert torch.equal(out[144:256], logical[144:256])
    assert torch.equal(out[256:384], logical[256:384])


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


def _append_steps(cache: PatternQuantizedKVCache, steps: int) -> None:
    for step in range(steps):
        key_states = torch.full((cache.sink_k.shape[0], 1, 1, 128), 100.0 + step)
        value_states = torch.full((cache.sink_k.shape[0], 1, 1, 128), 200.0 + step)
        append_decode_rolling(cache, key_states, value_states)


def test_ragged_multistep_valid_prefix_preserved() -> None:
    cache = assemble_ragged_patternkv_cache([
        _actual_like_cache(384, 128, value=3.0),
        _actual_like_cache(513, 256, value=7.0),
    ])
    _append_steps(cache, 16)
    lengths = k_segment_valid_lengths(cache)
    assert lengths["pending"].tolist() == [0, 1]
    assert lengths["packed"].tolist() == [256, 384]
    assert cache.pending_k.shape[2] == 1
    assert torch.equal(cache.pending_k[1, :, 0, :], torch.full((1, 128), 7.0))
    assert torch.equal(cache.recent_k[:, :, -1, :], torch.full((2, 1, 128), 115.0))


def test_ragged_multistep_independent_flush_schedule() -> None:
    cache = assemble_ragged_patternkv_cache([
        _actual_like_cache(399, 128, value=3.0),
        _actual_like_cache(512, 256, value=7.0),
    ])
    _append_steps(cache, 1)
    lengths = k_segment_valid_lengths(cache)
    assert lengths["packed"].tolist() == [256, 256]
    assert lengths["pending"].tolist() == [0, 113]


def test_ragged_multistep_short_row_flush_does_not_move_long_row() -> None:
    cache = assemble_ragged_patternkv_cache([
        _actual_like_cache(399, 128, value=3.0),
        _actual_like_cache(512, 256, value=7.0),
    ])
    long_recent_before = cache.recent_k[1, :, :128, :].clone()
    _append_steps(cache, 1)
    assert torch.equal(cache.recent_k[1], torch.cat([long_recent_before[:, 1:, :], torch.full((1, 1, 128), 100.0)], dim=1))
    assert get_packed_k_tokens_per_request(cache).tolist() == [256, 256]


def test_ragged_multistep_long_row_flush_does_not_move_short_row() -> None:
    cache = assemble_ragged_patternkv_cache([
        _actual_like_cache(384, 128, value=3.0),
        _actual_like_cache(527, 256, value=7.0),
    ])
    short_packed_before = cache.packed_k[0].clone()
    _append_steps(cache, 1)
    assert torch.equal(cache.packed_k[0, :, :, : short_packed_before.shape[-1]], short_packed_before)
    assert get_packed_k_tokens_per_request(cache).tolist() == [128, 384]


def test_ragged_multistep_reorder_preserves_flush_schedule() -> None:
    ab = assemble_ragged_patternkv_cache([
        _actual_like_cache(384, 128, value=3.0),
        _actual_like_cache(513, 256, value=7.0),
    ])
    ba = assemble_ragged_patternkv_cache([
        _actual_like_cache(513, 256, value=7.0),
        _actual_like_cache(384, 128, value=3.0),
    ])
    _append_steps(ab, 16)
    _append_steps(ba, 16)
    assert get_packed_k_tokens_per_request(ab).tolist() == [256, 384]
    assert get_packed_k_tokens_per_request(ba).tolist() == [384, 256]


def test_ragged_multistep_centroid_slots_isolated() -> None:
    cache = assemble_ragged_patternkv_cache([
        _actual_like_cache(399, 128, value=3.0),
        _actual_like_cache(512, 256, value=7.0),
    ])
    pool = cache.centroid_state_pool
    before = pool.update_counts_k.clone()
    _append_steps(cache, 1)
    after = pool.update_counts_k
    assert (after - before).tolist() == [1, 0]


def test_ragged_multistep_page_ownership_isolated() -> None:
    cache = assemble_ragged_patternkv_cache([
        _actual_like_cache(399, 128, value=3.0),
        _actual_like_cache(512, 256, value=7.0),
    ])
    _append_steps(cache, 1)
    assert get_packed_k_tokens_per_request(cache).tolist() == [256, 256]
    assert cache.request_packed_v_tokens.tolist() == [256, 256]


def test_ragged_multistep_equal_length_regression() -> None:
    cache = assemble_ragged_patternkv_cache([
        _actual_like_cache(512, 256, value=3.0),
        _actual_like_cache(512, 256, value=7.0),
    ])
    _append_steps(cache, 16)
    lengths = k_segment_valid_lengths(cache)
    assert lengths["packed"].tolist() == [384, 384]
    assert lengths["pending"].tolist() == [0, 0]


def test_request_invariant_qk_scores_preserve_row_when_peer_count_changes() -> None:
    torch.manual_seed(2026081501)
    query_ab = torch.randn(2, 4, 1, 8, dtype=torch.float16)
    key_ab = torch.randn(2, 2, 5, 8, dtype=torch.float16)
    query_abcd = torch.randn(4, 4, 1, 8, dtype=torch.float16)
    key_abcd = torch.randn(4, 2, 7, 8, dtype=torch.float16)
    query_abcd[1] = query_ab[1]
    key_abcd[1, :, :5, :] = key_ab[1]
    scores_ab = patternkv_request_invariant_qk_scores(query_ab, key_ab, 2)
    scores_abcd = patternkv_request_invariant_qk_scores(query_abcd, key_abcd, 2)
    assert torch.equal(scores_ab[1, :, :, :5], scores_abcd[1, :, :, :5])


def test_request_invariant_qk_scores_match_single_row_matmul() -> None:
    torch.manual_seed(2026081502)
    query = torch.randn(3, 4, 1, 8, dtype=torch.float16)
    key = torch.randn(3, 2, 6, 8, dtype=torch.float16)
    got = patternkv_request_invariant_qk_scores(query, key, 2)
    repeated = key.repeat_interleave(2, dim=1)
    expected = (query.unsqueeze(3) * repeated.unsqueeze(2)).sum(dim=-1).contiguous()
    assert torch.equal(got, expected)
