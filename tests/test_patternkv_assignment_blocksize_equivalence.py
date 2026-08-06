from __future__ import annotations

import torch

from bench.patternkv_equivalence_reference import reference_minmax_assign
from models.segmented_cache import _assign_minmax_hnk


def test_block_sizes_match_full_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 7, 5)
    centroids = torch.randn(2, 9, 5)
    expected = reference_minmax_assign(x, centroids, compute_dtype=x.dtype)
    for block in (1, 2, 4, 32, 256):
        assert torch.equal(_assign_minmax_hnk(x, centroids, block_k=block), expected)
