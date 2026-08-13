from __future__ import annotations

import torch
import pytest

from bench.prefill_v_trace_utils import (
    assignment_metrics,
    bi_vproj_control,
    centroid_alignment_metrics,
    element_difference_rate,
    fixed_centroid_control,
    tensor_metric_dict,
    trace_reference_kmeans,
    v_centroid_amplification,
)


def test_v_trace_metrics_helper() -> None:
    ref = torch.tensor([1.0, 2.0, 3.0])
    got = torch.tensor([1.0, 2.5, 3.0])
    metrics = tensor_metric_dict(got, ref)
    assert metrics["exact"] is False
    assert metrics["max_abs"] == 0.5
    assert metrics["relative_l2"] > 0


def test_centroid_alignment_helper() -> None:
    ref = torch.tensor([[[0.0, 0.0], [4.0, 0.0]]])
    got = torch.tensor([[[4.0, 0.0], [0.0, 0.0]]])
    metrics = centroid_alignment_metrics(ref, got)
    assert metrics["raw_v_centroid_relative_l2"] > 0
    assert metrics["aligned_v_centroid_relative_l2"] == 0
    assert metrics["label_permutation_dominant"] is True


def test_v_centroid_amplification_helper() -> None:
    assert v_centroid_amplification(2e-2, 2e-4) == 100.0
    assert v_centroid_amplification(1.0, 0.0) == "inf"


def test_fixed_v_centroid_control_helper() -> None:
    value = torch.randn(1, 2, 128, 16, generator=torch.Generator().manual_seed(1))
    centroids = value[:, :, :4, :].squeeze(0).contiguous()
    result = fixed_centroid_control(value, value.clone(), centroids, value_objective="base", group_size=16, bits=2, v4_budget_fraction=0.25, sink_length=0, recent_length=0)
    assert result["fixed_centroid_assignment_difference_rate"] == 0
    assert result["fixed_centroid_mask_difference_rate"] == 0


def test_bi_vproj_control_uses_existing_v2_kernel() -> None:
    if not torch.cuda.is_available():
        return
    hidden = torch.randn(1, 5, 16, device="cuda", dtype=torch.float16)
    hidden_b2 = torch.cat([hidden, torch.randn_like(hidden)], dim=0)
    hidden_b4 = torch.cat([hidden_b2, torch.randn_like(hidden_b2)], dim=0)
    weight = torch.randn(8, 16, device="cuda", dtype=torch.float16)
    result = bi_vproj_control(hidden, hidden_b2, hidden_b4, weight, None)
    assert result["b1_b2_exact"] is True
    assert result["b1_b4_exact"] is True


def test_v_assignment_difference_helper() -> None:
    ref = torch.tensor([[[0, 1, 1, 0]]])
    got = torch.tensor([[[1, 0, 0, 1]]])
    got_to_ref = torch.tensor([[1, 0]])
    metrics = assignment_metrics(ref, got, got_to_ref)
    assert metrics["raw_assignment_difference_rate"] == 1
    assert metrics["aligned_assignment_difference_rate"] == 0


def test_v_mask_difference_helper() -> None:
    ref = torch.tensor([[True, False, True]])
    got = torch.tensor([[True, True, False]])
    assert element_difference_rate(got, ref) == pytest.approx(2 / 3)


def test_v_packed_difference_helper() -> None:
    ref = torch.tensor([1, 2, 3], dtype=torch.int32)
    got = torch.tensor([1, 0, 3], dtype=torch.int32)
    assert element_difference_rate(got, ref) == pytest.approx(1 / 3)


def test_trace_reference_kmeans_helper() -> None:
    x = torch.randn(2, 32, 4, generator=torch.Generator().manual_seed(2))
    trace = trace_reference_kmeans(x, 4, iters=3, seed=0)
    assert trace["initial_indices"].shape == (2, 4)
    assert trace["final_centroids"].shape == (2, 4, 4)
