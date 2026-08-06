from __future__ import annotations

import pytest
import torch

from bench.aime24_int2_wave1 import aggregate_query_importance, kv_head_query_groups


def test_gqa_query_groups_are_contiguous() -> None:
    assert kv_head_query_groups(32, 8) == [list(range(0, 4)), list(range(4, 8)), list(range(8, 12)), list(range(12, 16)), list(range(16, 20)), list(range(20, 24)), list(range(24, 28)), list(range(28, 32))]


def test_gqa_rejects_non_divisible_heads() -> None:
    with pytest.raises(ValueError):
        kv_head_query_groups(30, 8)


def test_query_importance_aggregates_to_kv_heads() -> None:
    x = torch.arange(1 * 4 * 3, dtype=torch.float32).reshape(1, 4, 3)
    out = aggregate_query_importance(x, 2)
    expected0 = x[:, [0, 1], :].mean(dim=1)
    expected1 = x[:, [2, 3], :].mean(dim=1)
    assert torch.allclose(out[:, 0, :], expected0)
    assert torch.allclose(out[:, 1, :], expected1)
