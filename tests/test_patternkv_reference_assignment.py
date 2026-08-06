from __future__ import annotations

import torch

from bench.patternkv_equivalence_reference import (
    reference_chebyshev_center_fp16,
    reference_chebyshev_center_fp32,
    reference_minmax_assign,
    reference_minmax_distances,
    reference_top2,
    reference_v_gate,
)


def test_reference_chebyshev_center() -> None:
    x = torch.tensor([[[1.0, 4.0], [3.0, 2.0]]], dtype=torch.float16)
    assert torch.equal(reference_chebyshev_center_fp16(x), torch.tensor([[[2.0, 3.0]]], dtype=torch.float16))
    assert torch.equal(reference_chebyshev_center_fp32(x), torch.tensor([[[2.0, 3.0]]], dtype=torch.float32))


def test_reference_minmax_assignment() -> None:
    x = torch.tensor([[[0.0, 0.0], [4.0, 4.0]]])
    centroids = torch.tensor([[[0.0, 0.0], [5.0, 5.0]]])
    assign = reference_minmax_assign(x, centroids, compute_dtype=torch.float32)
    assert assign.tolist() == [[0, 0]]
    distances = reference_minmax_distances(x, centroids, compute_dtype=torch.float32)
    assert distances.shape == (1, 2, 2)


def test_reference_top2_margin() -> None:
    x = torch.tensor([[[1.0, 2.0]]])
    centroids = torch.tensor([[[1.0, 2.0], [1.0, 3.0]]])
    top2 = reference_top2(x, centroids, compute_dtype=torch.float32)
    assert top2["indices"].tolist() == [[[0, 1]]]
    assert top2["margin"].item() >= 0.0


def test_reference_v_gate_shapes() -> None:
    v = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])
    base = torch.zeros_like(v)
    rho, lhs, rhs, mask = reference_v_gate(v, base)
    assert rho.shape == lhs.shape == rhs.shape == mask.shape == (1, 1, 1)
