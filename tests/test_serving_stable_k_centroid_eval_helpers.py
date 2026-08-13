from __future__ import annotations

import pytest
import torch

from bench.run_serving_stable_k_centroid_eval import (
    aggregate_stage2,
    anchor_centroids,
    candidate_specs,
    canonicalize_k,
    decide,
    first_major_divergence,
    ratio,
    root_for_candidate,
    select_pareto,
    stability_improvement,
)


def test_quality_ratio_helper() -> None:
    assert ratio(0.11, 0.1) == 1.0999999999999999
    assert ratio(0.1, 0.0) is None


def test_stability_improvement_helper() -> None:
    assert stability_improvement(0.05, 0.01) == 5.0
    assert stability_improvement(0.05, 0.0) == "inf"


def test_candidate_pareto_selection() -> None:
    rows = [
        {"variant": "ANCHOR_ONLY", "qk_stability_improvement": 6.0, "qk_quality_ratio": 1.05, "k_quality_ratio": 1.02},
        {"variant": "KMEANS_1", "qk_stability_improvement": 4.0, "qk_quality_ratio": 1.01, "k_quality_ratio": 1.01},
        {"variant": "FIXED_B1_CENTROID_ORACLE", "qk_stability_improvement": "inf", "qk_quality_ratio": 1.0, "k_quality_ratio": 1.0},
    ]

    assert select_pareto(rows)["variant"] == "ANCHOR_ONLY"


def test_kmeans_iteration_sweep_helper() -> None:
    rows = [
        {"iteration": 0, "aligned_centroid_relative_l2": 0.0002},
        {"iteration": 1, "aligned_centroid_relative_l2": 0.003},
        {"iteration": 2, "aligned_centroid_relative_l2": 0.02},
    ]

    assert first_major_divergence(rows) == 2


def test_canonicalization_grid() -> None:
    k = torch.tensor([0.0, 0.24, 0.26, -0.26])

    assert torch.allclose(canonicalize_k(k, 0.5), torch.tensor([0.0, 0.0, 0.5, -0.5]))


def test_anchor_only_centroid() -> None:
    k = torch.arange(1 * 1 * 8 * 2, dtype=torch.float32).view(1, 1, 8, 2)

    centroids_a = anchor_centroids(k, 3, seed=0)
    centroids_b = anchor_centroids(k, 3, seed=0)

    assert torch.equal(centroids_a, centroids_b)
    assert centroids_a.shape == (1, 3, 2)


def test_stage2_aggregation() -> None:
    rows = [
        {"variant": "ANCHOR_ONLY", "qk_batch_rel_l2": 0.1, "qk_quality_ratio": 1.0},
        {"variant": "ANCHOR_ONLY", "qk_batch_rel_l2": 0.2, "qk_quality_ratio": 1.2},
        {"variant": "BASELINE_CURRENT", "qk_batch_rel_l2": 1.0, "qk_quality_ratio": 1.0},
    ]

    result = aggregate_stage2(rows, "ANCHOR_ONLY")

    assert result["cases"] == 2
    assert result["worst_qk_batch_rel_l2"] == pytest.approx(0.2)
    assert result["worst_qk_quality_ratio"] == pytest.approx(1.2)


def test_batch_invariant_control_labeling() -> None:
    oracle = {"k_proj_batch_relative_l2": 0.0, "qk_batch_rel_l2": 0.0}
    decision = decide([], None, oracle, baseline_reproduced=True)

    assert decision["classification"] == "BATCH_INVARIANT_KPROJ_CAUSAL_ORACLE_SUPPORTED"
    assert decision["next_task"] == "IMPLEMENT_BATCH_INVARIANT_KPROJ_PROTOTYPE"
    assert root_for_candidate("CANONICALIZED_P99") == "LOW_BIT_INPUT_NOISE_SENSITIVE_CLUSTER_BOUNDARIES"


def test_candidate_specs_include_required_families() -> None:
    specs = candidate_specs({"p95": 0.1, "p99": 0.2, "max": 0.4})
    names = {spec["name"] for spec in specs}

    assert {"BASELINE_CURRENT", "ANCHOR_ONLY", "KMEANS_1", "KMEANS_30", "CANONICALIZED_P95", "CANONICALIZED_P99", "CANONICALIZED_MAX"} <= names
