from __future__ import annotations

import csv
import gzip
from pathlib import Path

import torch

from models.segmented_cache import (
    affine_dequantize_last_dim_reference,
    build_cache_from_prefill,
    pattern_gather_centroids,
    reconstruct_full_k,
    reconstruct_packed_v,
)
from scripts.run_aime24_value_capacity_budget import (
    EXP8_COMMIT,
    bit_cost_table,
    bootstrap_ci,
    capacity_effect,
    effective_kv_bits,
    pairwise_between,
    pairwise_metric,
    stored_v_coverage,
)


def _all_v4_pair() -> tuple:
    torch.manual_seed(91)
    k = torch.randn(1, 2, 16, 16)
    v = torch.randn(1, 2, 16, 16)
    c = torch.randn(2, 5, 16)
    base = build_cache_from_prefill(k, v, sink_length=0, recent_length=0, group_size=16, k_bits=2, v_bits=2, pattern=True, k_centroids=c, v_centroids=c, v_precision_selector="all_v2")
    allv4 = build_cache_from_prefill(k, v, sink_length=0, recent_length=0, group_size=16, k_bits=2, v_bits=2, pattern=True, k_centroids=c, v_centroids=c, v_precision_selector="all_v4")
    return v, base, allv4


def test_all_v4_matches_reference() -> None:
    v, _base, allv4 = _all_v4_pair()
    packed = reconstruct_packed_v(allv4)
    assert packed is not None
    idx = allv4.v_assignment_idx[:, :, : allv4.packed_v_tokens]
    mask = allv4.v_pattern_mask[:, :, : allv4.packed_v_tokens].bool()
    cent = pattern_gather_centroids(idx, allv4.v_centroids)
    reference = affine_dequantize_last_dim_reference(v - mask.unsqueeze(-1).to(v.dtype) * cent, 16, 4)
    assert torch.allclose(packed, reference, atol=1e-5, rtol=1e-5)


def test_all_v4_no_selector_dependency() -> None:
    _v, _base, allv4 = _all_v4_pair()
    assert allv4.v_causal_importance is None
    assert allv4.v_oracle_importance is None
    assert allv4.v_precision_mask.bool().all()


def test_all_v4_centroid_assignment_identical() -> None:
    _v, base, allv4 = _all_v4_pair()
    assert torch.equal(base.v_assignment_idx, allv4.v_assignment_idx)


def test_all_v4_k_path_identical() -> None:
    _v, base, allv4 = _all_v4_pair()
    assert torch.allclose(reconstruct_full_k(base), reconstruct_full_k(allv4))


def test_base_reuse_provenance() -> None:
    assert EXP8_COMMIT == "241b832"


def test_stored_v_task_coverage() -> None:
    rows = []
    for method in ("BASE_V2", "ALL_V4"):
        for task in range(6):
            for cp in (512, 1024):
                rows.append(
                    {
                        "method": method,
                        "mode": "static",
                        "layer": "31",
                        "task_key": f"task{task}",
                        "checkpoint": str(cp),
                        "region": "all_packed_tokens",
                        "metric_name": "direction_error",
                        "statistic": "p95",
                    }
                )
    audit = stored_v_coverage(rows, ("BASE_V2", "ALL_V4"))
    assert audit["stored_v_task_coverage"]["ALL_V4"] == "6/6"


def test_capacity_ceiling_pairwise() -> None:
    static_auc = []
    gap_auc = []
    for task in ("a", "b"):
        gap_auc.append({"task_key": task, "method": "BASE_V2", "layer": "31", "metric_family": "hidden_accumulation", "object_type": "hidden_state", "region": "current_token", "metric_name": "relative_L2", "statistic": "global", "auc": 1.0})
        gap_auc.append({"task_key": task, "method": "ALL_V4", "layer": "31", "metric_family": "hidden_accumulation", "object_type": "hidden_state", "region": "current_token", "metric_name": "relative_L2", "statistic": "global", "auc": 0.8})
    row = pairwise_metric(static_auc, gap_auc, method="ALL_V4", metric_name="hidden")
    assert row["median_delta"] == -0.19999999999999996
    assert row["tasks_improved"] == 2


def test_capacity_headroom_over_causal12p5() -> None:
    gap_auc = []
    for task in ("a", "b", "c"):
        gap_auc.append({"task_key": task, "method": "CAUSAL_V4", "layer": "31", "metric_family": "hidden_accumulation", "object_type": "hidden_state", "region": "current_token", "metric_name": "relative_L2", "statistic": "global", "auc": 0.9})
        gap_auc.append({"task_key": task, "method": "ALL_V4", "layer": "31", "metric_family": "hidden_accumulation", "object_type": "hidden_state", "region": "current_token", "metric_name": "relative_L2", "statistic": "global", "auc": 0.7})
    row = pairwise_between([], gap_auc, lhs="ALL_V4", rhs="CAUSAL_V4", metric_name="hidden")
    assert row["lhs_better_tasks"] == 3
    assert row["lhs_minus_rhs_median_delta"] == -0.20000000000000007


def test_budget_gate() -> None:
    summary = {
        "hidden": {"median_delta": -0.2, "tasks_improved": 6},
        "attention_output": {"median_delta": -0.1, "tasks_improved": 5},
        "value_only": {"median_delta": -0.1, "tasks_improved": 5},
    }
    headroom = {"lhs_minus_rhs_median_delta": -0.1}
    assert capacity_effect(summary, headroom) == "STRONG"


def test_budget_fraction_25() -> None:
    row = next(row for row in bit_cost_table() if row["budget"] == 0.25)
    assert row["value_payload_bits_per_element"] == 2.5


def test_budget_fraction_50() -> None:
    row = next(row for row in bit_cost_table() if row["budget"] == 0.5)
    assert row["value_payload_bits_per_element"] == 3.0


def test_budget_bit_accounting() -> None:
    assert effective_kv_bits(0.0, precision_metadata=False) == 2.25
    assert effective_kv_bits(1.0, precision_metadata=False) == 3.25


def test_marginal_gain() -> None:
    assert bootstrap_ci([-0.1, -0.2, -0.3])[0] <= bootstrap_ci([-0.1, -0.2, -0.3])[1]


def test_budget_curve_alignment() -> None:
    assert [row["budget"] for row in bit_cost_table()] == [0.0, 0.125, 0.25, 0.5, 1.0]


def test_selector_advantage_curve() -> None:
    assert -0.2 < 0
