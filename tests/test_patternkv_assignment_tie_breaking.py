from __future__ import annotations

import torch

from bench.patternkv_equivalence_reference import reference_minmax_assign, reference_top2
from models.segmented_cache import _assign_minmax_hnk


def test_exact_tie_selects_lowest_index() -> None:
    x = torch.tensor([[[1.0, 1.0]]])
    centroids = torch.tensor([[[0.0, 0.0], [2.0, 2.0]]])
    assert reference_minmax_assign(x, centroids, compute_dtype=torch.float32).item() == 0
    top2 = reference_top2(x, centroids, compute_dtype=torch.float32)
    assert top2["distances"][0, 0, 0].item() == top2["distances"][0, 0, 1].item()


def test_segmented_block_tie_matches_full_reference() -> None:
    x = torch.tensor([[[1.0, 1.0]]])
    centroids = torch.tensor([[[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]])
    expected = reference_minmax_assign(x, centroids, compute_dtype=torch.float32)
    assert torch.equal(_assign_minmax_hnk(x, centroids, block_k=1), expected)
    assert torch.equal(_assign_minmax_hnk(x, centroids, block_k=2), expected)


def test_fp16_near_tie_can_be_reported() -> None:
    x = torch.tensor([[[1.0, 1.0001]]], dtype=torch.float32)
    centroids = torch.tensor([[[0.0, 0.0], [2.0, 2.0]]], dtype=torch.float32)
    fp16 = reference_top2(x, centroids, compute_dtype=torch.float16)
    fp32 = reference_top2(x, centroids, compute_dtype=torch.float32)
    assert fp16["margin"].numel() == fp32["margin"].numel() == 1
