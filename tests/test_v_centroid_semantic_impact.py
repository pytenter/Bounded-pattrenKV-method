from __future__ import annotations

import pytest
import torch

from bench.prefill_v_trace_utils import (
    centroid_only_counterfactual_metrics,
    reconstructed_v_metrics,
    run_v_state,
    same_attention_value_metrics,
    semantic_impact_level,
    semantic_state_comparator,
)


def _cache(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    value = torch.randn(1, 2, 128, 16, generator=gen)
    centroids = value[:, :, :4, :].squeeze(0).contiguous()
    return run_v_state(value, centroids, value_objective="base", group_size=16, bits=2, v4_budget_fraction=0.25, sink_length=0, recent_length=0)["cache"]


def test_semantic_state_comparator_exact() -> None:
    cache = _cache()
    result = semantic_state_comparator(cache, cache)
    assert result["status"] == "PASS"


def test_semantic_state_comparator_centroid_numerical_only() -> None:
    ref = _cache()
    got = _cache()
    got.v_centroids = ref.v_centroids + 1e-5
    got.operator_ready_page_pools.centroids = got.v_centroids
    result = semantic_state_comparator(ref, got)
    assert result["status"] == "NUMERICAL_ONLY_DIFFERENCE"


def test_semantic_state_comparator_assignment_failure() -> None:
    ref = _cache()
    got = _cache()
    got.v_assignment_idx = got.v_assignment_idx.clone()
    got.v_assignment_idx[:, :, 0] = (got.v_assignment_idx[:, :, 0] + 1) % got.v_centroids.shape[1]
    result = semantic_state_comparator(ref, got)
    assert result["status"] == "STRUCTURAL_FAIL"


def test_semantic_state_comparator_packed_payload_failure() -> None:
    ref = _cache()
    got = _cache()
    got.packed_v = got.packed_v.clone()
    got.packed_v.flatten()[0] += 1
    result = semantic_state_comparator(ref, got)
    assert result["status"] == "PHYSICAL_FAIL"


def test_semantic_state_comparator_large_centroid_failure() -> None:
    ref = _cache()
    got = _cache()
    got.v_centroids = ref.v_centroids + 0.1
    got.operator_ready_page_pools.centroids = got.v_centroids
    result = semantic_state_comparator(ref, got)
    assert result["status"] == "NUMERICAL_FAIL"


def test_reconstructed_v_metric_helper() -> None:
    cache = _cache()
    metrics = reconstructed_v_metrics(cache, cache)
    assert metrics["exact"] is True
    assert metrics["relative_l2"] == 0


def test_centroid_only_counterfactual_helper() -> None:
    cache = _cache()
    metrics = centroid_only_counterfactual_metrics(cache, cache.v_centroids + 1e-4)
    assert metrics["centroid_only_reconstructed_v_relative_l2"] > 0


def test_same_attention_value_output_helper() -> None:
    cache = _cache()
    metrics = reconstructed_v_metrics(cache, cache)
    attn = torch.randn(1, 4, 3, metrics["ref"].shape[2])
    out = same_attention_value_metrics(attn, metrics["got"], metrics["ref"], num_key_value_groups=2)
    assert out["relative_l2"] == pytest.approx(0.0)


def test_semantic_impact_classifier() -> None:
    assert semantic_impact_level(1e-5) == "NEGLIGIBLE"
    assert semantic_impact_level(5e-4) == "SMALL"
    assert semantic_impact_level(2e-3) == "MEANINGFUL"
