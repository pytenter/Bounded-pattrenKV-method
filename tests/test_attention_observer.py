from __future__ import annotations

import torch

from bench.attention_observer import (
    absolute_regions,
    cache_segment_regions,
    clone_signature,
    enrichment,
    probability_metrics,
    region_contributions,
    region_mass,
    repeat_kv_for_gqa,
    routing_value_decomposition,
    shadow_attention,
    tensor_pair_metrics,
)


def test_absolute_window_mass_and_enrichment() -> None:
    probs = torch.full((1, 2, 1, 256), 1.0 / 256, dtype=torch.float32)
    regions = absolute_regions(256)
    mass = region_mass(probs, regions)
    assert torch.allclose(mass["E16"], torch.full((1, 2, 1), 16 / 256))
    assert torch.allclose(mass["E128"], torch.full((1, 2, 1), 128 / 256))
    assert torch.allclose(enrichment(mass["E16"], region_tokens=16, context_tokens=256), torch.ones(1, 2, 1))


def test_cache_segment_mass_sums_to_one_for_disjoint_segments() -> None:
    probs = torch.full((1, 1, 1, 10), 0.1, dtype=torch.float32)
    regions = cache_segment_regions(sink_tokens=2, packed_history_tokens=3, pending_tokens=1, recent_tokens=4)
    mass = region_mass(probs, regions)
    total = mass["protected_sink"] + mass["packed_history"] + mass["pending_history"] + mass["recent"]
    assert torch.allclose(total, torch.ones_like(total))


def test_shadow_attention_is_finite_and_does_not_mutate_inputs() -> None:
    torch.manual_seed(0)
    query = torch.randn(1, 2, 1, 8)
    key = torch.randn(1, 2, 5, 8)
    value = torch.randn(1, 2, 5, 8)
    before = clone_signature((query, key, value))
    out = shadow_attention(query, key, value)
    after = clone_signature((query, key, value))
    assert before == after
    assert torch.isfinite(out["scores"]).all()
    assert torch.isfinite(out["probs"]).all()
    assert torch.isfinite(out["output"]).all()
    assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(1, 2, 1), atol=1e-6)


def test_repeat_kv_for_gqa_matches_expected_layout() -> None:
    kv = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    repeated = repeat_kv_for_gqa(kv, 4)
    assert repeated.shape == (1, 8, 3, 4)
    assert torch.equal(repeated[:, 0], kv[:, 0])
    assert torch.equal(repeated[:, 3], kv[:, 0])
    assert torch.equal(repeated[:, 4], kv[:, 1])


def test_region_contribution_and_decomposition() -> None:
    probs = torch.tensor([[[[0.7, 0.2, 0.1]]]], dtype=torch.float32)
    quant_probs = torch.tensor([[[[0.6, 0.3, 0.1]]]], dtype=torch.float32)
    value = torch.tensor([[[[1.0, 0.0], [0.0, 2.0], [2.0, 2.0]]]], dtype=torch.float32)
    quant_value = value + 0.1
    regions = cache_segment_regions(sink_tokens=1, packed_history_tokens=1, pending_tokens=0, recent_tokens=1)
    contrib = region_contributions(probs, value, regions)
    assert torch.allclose(contrib["protected_sink"], torch.tensor([[[[0.7, 0.0]]]]))
    fp16_output = torch.matmul(probs, value)
    decomp = routing_value_decomposition(
        fp16_probs=probs,
        quant_probs=quant_probs,
        fp16_value=value,
        quant_value=quant_value,
        fp16_output=fp16_output,
    )
    assert set(decomp) == {"routing_only", "value_only", "full"}
    assert decomp["value_only"]["relative_l2"] > 0
    assert probability_metrics(quant_probs, probs)["kl_ref_to_quant"] > 0
    assert tensor_pair_metrics(value, value)["relative_l2"] == 0
