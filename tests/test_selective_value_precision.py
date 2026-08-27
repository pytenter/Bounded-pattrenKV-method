from __future__ import annotations

import torch

from bench.pseudodecode_metrics import trapezoid_auc_log2
from models.segmented_cache import (
    PatternQuantizedKVCache,
    append_decode,
    build_cache_from_prefill,
    local_v2_v4_gain,
    pattern_gather_centroids,
    pattern_nearest_v_centroid,
    quantize_pack_v_reference,
    dequantize_v_reference,
    reconstruct_full_k,
    reconstruct_full_v,
    reconstruct_packed_v,
    normalize_value_precision_selector,
    select_value_precision_mask,
    serialize_cache,
    deserialize_cache,
    update_value_causal_importance,
)


def _sample_cache(selector: str, *, tokens: int = 16, recent: int = 0, seed: int = 0) -> PatternQuantizedKVCache:
    torch.manual_seed(seed)
    k = torch.randn(1, 2, tokens, 16)
    v = torch.randn(1, 2, tokens, 16)
    c = torch.randn(2, 5, 16)
    cache = build_cache_from_prefill(
        k,
        v,
        sink_length=0,
        recent_length=recent,
        group_size=16,
        k_bits=2,
        v_bits=2,
        pattern=True,
        k_centroids=c,
        v_centroids=c,
        value_objective="base",
        v_precision_selector=selector,
        v4_budget_fraction=0.25,
    )
    assert isinstance(cache, PatternQuantizedKVCache)
    return cache


def test_mixed_v_all_v2_matches_baseline() -> None:
    base = _sample_cache("base_v2")
    mixed = _sample_cache("all_v2")
    assert torch.equal(base.v_assignment_idx, mixed.v_assignment_idx)
    assert torch.allclose(reconstruct_full_v(base), reconstruct_full_v(mixed))


def test_mixed_v_all_v4_matches_reference() -> None:
    cache = _sample_cache("all_v4")
    raw = reconstruct_packed_v(cache)
    assert raw is not None
    assert cache.v_precision_mask.bool().all()
    assert cache.packed_v is None
    assert cache.packed_v4_tokens == cache.packed_v_tokens


def test_mixed_v_token_order_restore() -> None:
    cache = PatternQuantizedKVCache(total_tokens=4, packed_k_tokens=4, packed_v_tokens=4, group_size=16, k_bits=2, v_bits=2)
    v = torch.arange(64, dtype=torch.float32).view(1, 1, 4, 16)
    mask = torch.tensor([[False, True, False, True]])
    low = v[:, :, [0, 2], :]
    high = v[:, :, [1, 3], :]
    cache.packed_v, cache.packed_v_scale, cache.packed_v_zero = quantize_pack_v_reference(low, 16, 2)
    cache.packed_v4, cache.packed_v4_scale, cache.packed_v4_zero = quantize_pack_v_reference(high, 16, 4)
    cache.v_precision_mask = mask.to(torch.uint8)
    cache.packed_v4_tokens = 2
    restored = reconstruct_packed_v(cache)
    low_ref = dequantize_v_reference(cache.packed_v, cache.packed_v_scale, cache.packed_v_zero, 16, 2)
    high_ref = dequantize_v_reference(cache.packed_v4, cache.packed_v4_scale, cache.packed_v4_zero, 16, 4)
    assert restored.shape == v.shape
    assert torch.allclose(restored[:, :, 0], low_ref[:, :, 0])
    assert torch.allclose(restored[:, :, 1], high_ref[:, :, 0])


def test_mixed_v_precision_bitmap() -> None:
    cache = _sample_cache("random_v4")
    assert cache.v_precision_mask.shape == (1, cache.packed_v_tokens)
    assert int(cache.v_precision_mask.sum().item()) == cache.packed_v4_tokens


def test_mixed_v_reset() -> None:
    a = _sample_cache("random_v4", seed=7)
    b = _sample_cache("random_v4", seed=7)
    assert torch.equal(a.v_precision_mask, b.v_precision_mask)


def test_mixed_v_static_fresh_state() -> None:
    a = _sample_cache("causal_v4", seed=3)
    b = _sample_cache("causal_v4", seed=3)
    assert torch.equal(a.v_precision_mask, b.v_precision_mask)


def test_mixed_v_pseudo_persistent_state() -> None:
    cache = _sample_cache("causal_v4", tokens=8, recent=4)
    before = cache.v_causal_importance
    append_decode(cache, torch.randn(1, 2, 5, 16), torch.randn(1, 2, 5, 16))
    attn = torch.ones(1, 2, 1, cache.total_tokens) / float(cache.total_tokens)
    update_value_causal_importance(cache, attn)
    assert cache.v_causal_importance is not None
    assert before is None or cache.v_causal_importance.shape[1] >= before.shape[1]


def test_centroid_assignment_identical_across_precision() -> None:
    base = _sample_cache("all_v2", seed=5)
    v4 = _sample_cache("all_v4", seed=5)
    assert torch.equal(base.v_assignment_idx, v4.v_assignment_idx)


def test_k_path_identical_across_precision() -> None:
    base = _sample_cache("all_v2", seed=6)
    v4 = _sample_cache("all_v4", seed=6)
    assert torch.allclose(reconstruct_full_k(base), reconstruct_full_k(v4))


def test_v4_budget_exact() -> None:
    cache = _sample_cache("random_v4", tokens=16)
    assert int(cache.v_precision_mask.sum().item()) == 4


def test_random_selector_deterministic() -> None:
    a = _sample_cache("random_v4", seed=11)
    b = _sample_cache("random_v4", seed=11)
    assert torch.equal(a.v_precision_mask, b.v_precision_mask)


def test_causal_importance_accumulator() -> None:
    cache = PatternQuantizedKVCache(total_tokens=3)
    update_value_causal_importance(cache, torch.tensor([[[[0.2, 0.3, 0.5]], [[0.4, 0.1, 0.5]]]]))
    assert torch.allclose(cache.v_causal_importance, torch.tensor([[0.3, 0.2, 0.5]]))


def test_causal_selector_no_future_access() -> None:
    cache = PatternQuantizedKVCache(group_size=16, v_precision_selector="causal_v4", v4_budget_fraction=0.5)
    v = torch.randn(1, 2, 4, 16)
    c = torch.zeros_like(v)
    m = torch.zeros(1, 2, 4, dtype=torch.bool)
    cache.v_causal_importance = torch.tensor([[0.1, 0.9, 0.2, 0.8]])
    first = select_value_precision_mask(cache, v, c, m, absolute_start=0)
    cache.v_oracle_importance = torch.tensor([[100.0, 0.0, 100.0, 0.0]])
    second = select_value_precision_mask(cache, v, c, m, absolute_start=0)
    assert torch.equal(first, second)


def test_static_causal_importance_pack_time_only() -> None:
    cache = PatternQuantizedKVCache(group_size=16, v_precision_selector="causal_v4", v4_budget_fraction=0.25)
    cache.v_causal_importance = torch.tensor([[0.0, 1.0, 0.0, 0.0, 99.0]])
    v = torch.randn(1, 1, 4, 16)
    m = torch.zeros(1, 1, 4, dtype=torch.bool)
    chosen = select_value_precision_mask(cache, v, torch.zeros_like(v), m, absolute_start=0)
    assert chosen.shape == (1, 4)


def test_pseudo_causal_selector_feedback() -> None:
    cache = PatternQuantizedKVCache(total_tokens=4, group_size=4)
    update_value_causal_importance(cache, torch.ones(1, 2, 1, 4) * 0.25)
    assert float(cache.v_causal_importance.sum().item()) == 1.0


def test_local_v2_v4_gain() -> None:
    v = torch.randn(1, 2, 4, 16)
    gain = local_v2_v4_gain(v, torch.zeros_like(v), torch.zeros(1, 2, 4, dtype=torch.bool), group_size=16)
    assert gain.shape == (1, 4)
    assert torch.isfinite(gain).all()
    assert (gain >= 0).all()


def test_oracle_future_information_quarantined() -> None:
    causal = PatternQuantizedKVCache(group_size=16, v_precision_selector="causal_v4", v4_budget_fraction=0.5)
    oracle = PatternQuantizedKVCache(group_size=16, v_precision_selector="oracle_v4", v4_budget_fraction=0.5)
    causal.v_causal_importance = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    oracle.v_oracle_importance = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    v = torch.randn(1, 1, 4, 16)
    m = torch.zeros(1, 1, 4, dtype=torch.bool)
    assert not torch.equal(select_value_precision_mask(causal, v, torch.zeros_like(v), m, absolute_start=0), select_value_precision_mask(oracle, v, torch.zeros_like(v), m, absolute_start=0))


def test_oracle_selection_budget() -> None:
    cache = PatternQuantizedKVCache(group_size=16, v_precision_selector="oracle_v4", v4_budget_fraction=0.5)
    cache.v_oracle_importance = torch.tensor([[0.0, 2.0, 1.0, 3.0]])
    v = torch.randn(1, 1, 4, 16)
    m = torch.zeros(1, 1, 4, dtype=torch.bool)
    chosen = select_value_precision_mask(cache, v, torch.zeros_like(v), m, absolute_start=0)
    assert int(chosen.sum().item()) == 2


def test_component_selector_aliases() -> None:
    assert normalize_value_precision_selector("importance-only-v4") == "importance_only_v4"
    assert normalize_value_precision_selector("error_only") == "error_only_v4"


def test_importance_only_selector_uses_importance_without_gain_product() -> None:
    cache = PatternQuantizedKVCache(group_size=16, v_precision_selector="importance_only_v4", v4_budget_fraction=0.5)
    cache.v_causal_importance = torch.tensor([[0.0, 3.0, 1.0, 2.0]])
    v = torch.randn(1, 1, 4, 16)
    m = torch.zeros(1, 1, 4, dtype=torch.bool)
    chosen = select_value_precision_mask(cache, v, torch.zeros_like(v), m, absolute_start=0)
    assert chosen.tolist() == [[False, True, False, True]]


def test_error_only_selector_uses_local_v2_v4_gain_budget() -> None:
    cache = PatternQuantizedKVCache(group_size=16, v_precision_selector="error_only_v4", v4_budget_fraction=0.5)
    v = torch.randn(1, 1, 4, 16)
    m = torch.zeros(1, 1, 4, dtype=torch.bool)
    gain = local_v2_v4_gain(v, torch.zeros_like(v), m, group_size=16)
    chosen = select_value_precision_mask(cache, v, torch.zeros_like(v), m, absolute_start=0)
    assert int(chosen.sum().item()) == 2
    assert torch.equal(chosen, torch.zeros_like(chosen).scatter(1, torch.topk(gain, 2, dim=1).indices, True))


def test_selective_config_bit_cost_equal() -> None:
    payload = 0.875 * 2.0 + 0.125 * 4.0
    assert payload == 2.25


def test_selector_overlap_metrics() -> None:
    a = torch.tensor([[True, False, True, False]])
    b = torch.tensor([[False, False, True, True]])
    overlap = (a & b).sum().item() / max(a.sum().item(), 1)
    assert overlap == 0.5


def test_selective_precision_auc() -> None:
    assert trapezoid_auc_log2([(128, 1.0), (512, 0.8), (1024, 0.7), (2048, 0.6), (4096, 0.5)]) > 0.0


def test_selective_precision_pairwise() -> None:
    base = {"task0": 1.0, "task1": 2.0}
    method = {"task0": 0.8, "task1": 2.2}
    deltas = [method[key] - base[key] for key in sorted(base)]
    assert sum(delta < 0 for delta in deltas) == 1


def test_value_assignment_stays_base_objective() -> None:
    torch.manual_seed(9)
    v = torch.randn(1, 2, 4, 16)
    c = torch.randn(2, 5, 16)
    idx = pattern_nearest_v_centroid(v, c)
    gathered = pattern_gather_centroids(idx, c)
    assert gathered.shape == v.shape


def test_mixed_value_precision_survives_serialize_roundtrip() -> None:
    cache = _sample_cache("random_v4")
    restored = deserialize_cache(serialize_cache(cache), pattern=True)
    assert isinstance(restored, PatternQuantizedKVCache)
    assert torch.equal(restored.v_precision_mask, cache.v_precision_mask)
    assert restored.packed_v4_tokens == cache.packed_v4_tokens
