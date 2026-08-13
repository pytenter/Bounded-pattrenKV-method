from __future__ import annotations

import torch

from bench.run_prefill_kmeans_stability import (
    align_centroids_to_ref,
    amplification_factor,
    centroid_hungarian_matching,
    difference_rate,
    fixed_centroid_control,
    permutation_aligned_centroid_metrics,
    qk_impact_metrics,
    remap_assignments_to_ref,
    scaled_delta,
    gather_selected,
)


def test_centroid_hungarian_matching() -> None:
    ref = torch.tensor([[[0.0, 0.0], [10.0, 10.0], [3.0, 4.0]]])
    test = torch.tensor([[[10.0, 10.0], [3.0, 4.0], [0.0, 0.0]]])

    ref_to_test, test_to_ref, rows = centroid_hungarian_matching(ref, test)

    assert ref_to_test.tolist() == [[2, 0, 1]]
    assert test_to_ref.tolist() == [[1, 2, 0]]
    assert len(rows) == 3


def test_assignment_permutation_remap() -> None:
    assignments = torch.tensor([[[0, 1, 2, 1]]])
    test_to_ref = torch.tensor([[2, 0, 1]])

    aligned = remap_assignments_to_ref(assignments, test_to_ref)

    assert aligned.tolist() == [[[2, 0, 1, 0]]]


def test_permutation_aligned_centroid_metrics() -> None:
    ref = torch.tensor([[[0.0, 0.0], [10.0, 10.0]]])
    test = torch.tensor([[[10.0, 10.0], [0.0, 0.0]]])

    metrics = permutation_aligned_centroid_metrics(ref, test)

    assert metrics["raw_centroid_relative_l2"] > 0
    assert metrics["aligned_centroid_relative_l2"] == 0
    assert metrics["permutation_explained_fraction"] == 1.0
    assert torch.equal(align_centroids_to_ref(test, metrics["ref_to_test"]), ref)


def test_scaled_delta_generation() -> None:
    k_ref = torch.tensor([1.0, 2.0])
    delta = torch.tensor([0.5, -1.0])

    assert torch.allclose(scaled_delta(k_ref, delta, 2.0), torch.tensor([2.0, 0.0]))


def test_fixed_centroid_control_helper() -> None:
    k_ref = torch.tensor([[[[0.0, 0.0], [4.0, 4.0]]]])
    k_test = torch.tensor([[[[0.1, 0.0], [4.0, 4.1]]]])
    centroids = torch.tensor([[[0.0, 0.0], [4.0, 4.0]]])

    result = fixed_centroid_control(k_ref, k_test, centroids)

    assert result["assignment_difference_count"] == 0
    assert result["assignment_difference_rate"] == 0.0


def test_selected_centroid_equivalence() -> None:
    ref_centroids = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    test_centroids = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    assignments_ref = torch.tensor([[[0, 1]]])
    assignments_test = torch.tensor([[[1, 0]]])

    selected_ref = gather_selected(assignments_ref, ref_centroids)
    selected_test = gather_selected(assignments_test, test_centroids)

    assert torch.equal(selected_ref, selected_test)
    assert difference_rate(assignments_test, assignments_ref) == 1.0


def test_qk_impact_helper() -> None:
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]]])
    ref_k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    test_k = ref_k.clone()

    result = qk_impact_metrics(query, ref_k, test_k, num_key_value_heads=1, scale=1.0)

    assert result["qk_relative_l2"] == 0.0
    assert result["qk_max_abs"] == 0.0
    assert result["qk_cosine"] == 1.0


def test_amplification_factor_helper() -> None:
    assert amplification_factor(0.04, 0.0002) == 200.0
    assert amplification_factor(0.04, 0.0) is None
