from __future__ import annotations

import torch
import pytest

from insight.collector import InsightCollector
from insight.config import InsightRuntimeConfig
from insight.range_aware_metrics import (
    RANGE_EPSILON,
    AggregateStats,
    compute_k_group_assignment_diagnostics,
    compute_v_assignment_diagnostics,
    merge_stats,
)


def test_v_assignment_diagnostics_preserve_negative_regret_and_tie_breaking():
    vectors = torch.tensor([[0.0, 2.0], [1.0, 1.0]], dtype=torch.float32)
    patterns = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32)
    diag = compute_v_assignment_diagnostics(vectors, patterns, epsilon=RANGE_EPSILON)
    assert diag.l2_assignment.tolist() == [0, 0]
    assert diag.minmax_assignment.tolist() == [0, 0]
    assert diag.mismatch_count == 0
    assert diag.total_count == 2
    assert (diag.range_regret <= 0).any()


def test_v_assignment_chunking_matches_non_chunked():
    vectors = torch.arange(15, dtype=torch.float32).view(5, 3)
    patterns = torch.arange(12, dtype=torch.float32).view(4, 3)
    a = compute_v_assignment_diagnostics(vectors, patterns, chunk_size=2)
    b = compute_v_assignment_diagnostics(vectors, patterns, chunk_size=0)
    assert torch.equal(a.l2_assignment, b.l2_assignment)
    assert torch.equal(a.minmax_assignment, b.minmax_assignment)
    assert torch.allclose(a.range_regret, b.range_regret)


def test_k_group_diagnostics_respect_group_projection_and_counts():
    groups = torch.zeros(2, 128, 4, dtype=torch.float32)
    groups[0, :, 0] = 1.0
    patterns = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    diag = compute_k_group_assignment_diagnostics(groups, patterns)
    assert diag.total_count == 2 * 4
    assert 0 <= diag.mismatch_count <= diag.total_count
    assert diag.l2_range.shape[0] == diag.total_count


def test_aggregate_stats_merge_order_independent():
    a = AggregateStats.from_tensor(torch.tensor([1.0, 2.0]))
    b = AggregateStats.from_tensor(torch.tensor([3.0]))
    assert a.merge(b) == merge_stats([a, b])
    assert a.merge(b) == b.merge(a)


def test_nan_inf_fail_fast():
    with pytest.raises(ValueError):
        AggregateStats.from_tensor(torch.tensor([1.0, float("nan")]))
    with pytest.raises(ValueError):
        AggregateStats.from_tensor(torch.tensor([1.0, float("inf")]))


def test_sample_records_disabled_keeps_aggregate_path():
    collector = InsightCollector(
        InsightRuntimeConfig(
            enabled=True,
            sample_records_enabled=False,
            range_aware_aggregates=True,
        )
    )
    collector.add_sample_record({"x": 1})
    stats = AggregateStats.from_tensor(torch.tensor([1.0, 3.0]))
    collector.add_range_aware_aggregate(
        phase="prefill",
        kv_type="v",
        layer=0,
        kv_head=0,
        bucket="all",
        assignment_total_count=2,
        assignment_mismatch_count=1,
        l2_residual_range=stats,
        minmax_residual_range=stats,
        range_gain_absolute=stats,
        range_regret=stats,
    )
    assert collector.records == []
    assert collector.dropped_record_count == 0
    assert len(collector.range_aware_aggregates) == 1
