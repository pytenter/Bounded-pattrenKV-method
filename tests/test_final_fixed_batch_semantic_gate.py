from __future__ import annotations

import torch

from bench.final_fixed_batch_semantic_utils import (
    StructuralLayerState,
    boundary_explosion,
    compare_float_tensors,
    difference_rate,
    final_gate_requires_bi_k_mode,
    forced_replay_step_tokens,
    semantic_gate_bounded,
    structural_cross_batch_equal,
    topk_logit_metrics,
    validate_assignment_index_range,
    validate_logical_counts,
    validate_request_slot_mapping,
    validate_v4_budget,
)


def _state(**kwargs) -> StructuralLayerState:
    base = dict(
        request="A",
        batch_mode="B1",
        step=128,
        layer=0,
        total_tokens=640,
        sink_tokens=16,
        recent_tokens=128,
        pending_tokens=112,
        packed_k_tokens=384,
        packed_v_tokens=384,
        packed_v4_tokens=96,
        k_centroid_count=35,
        v_centroid_count=35,
        k_update_count=3,
        v_update_count=3,
        last_flush_pos=384,
        page_count=3,
        last_page_valid_tokens=128,
        slot_id=0,
    )
    base.update(kwargs)
    return StructuralLayerState(**base)


def test_forced_reference_replay_keeps_same_request_tokens() -> None:
    refs = {"A": [11, 12], "B": [21, 22]}
    assert forced_replay_step_tokens(refs, ["A", "B"], 2) == [12, 22]


def test_forced_replay_batch_rows_use_correct_reference() -> None:
    refs = {"A": [1], "B": [2], "C": [3], "D": [4]}
    assert forced_replay_step_tokens(refs, ["A", "C"], 1) == [1, 3]
    assert forced_replay_step_tokens(refs, ["A", "B", "C", "D"], 1) == [1, 2, 3, 4]


def test_structural_gate_ignores_physical_slot_id() -> None:
    ref = _state(slot_id=0)
    got = _state(batch_mode="B4", slot_id=3)
    assert structural_cross_batch_equal(ref, got)


def test_structural_gate_checks_logical_counts() -> None:
    assert validate_logical_counts(_state())
    assert not validate_logical_counts(_state(pending_tokens=111))


def test_structural_gate_assignment_index_range() -> None:
    assert validate_assignment_index_range(torch.tensor([0, 3, 4]), 5)
    assert not validate_assignment_index_range(torch.tensor([0, 5]), 5)


def test_structural_gate_v4_budget() -> None:
    assert validate_v4_budget(25, 100, 0.25)
    assert not validate_v4_budget(40, 100, 0.25)


def test_boundary_transition_validator() -> None:
    states = [_state(step=127, total_tokens=639, pending_tokens=111), _state(step=128), _state(step=129, total_tokens=641, recent_tokens=128, pending_tokens=113)]
    assert all(validate_logical_counts(state) for state in states)


def test_boundary_explosion_metric() -> None:
    result = boundary_explosion({127: 1e-4, 128: 2e-2, 129: 2e-4}, 128)
    assert result["explosion"] is True
    quiet = boundary_explosion({127: 1e-4, 128: 3e-4, 129: 2e-4}, 128)
    assert quiet["explosion"] is False


def test_semantic_metrics_allow_nonexact_float_state() -> None:
    ref = torch.tensor([1.0, 2.0])
    got = torch.tensor([1.0, 2.001])
    metrics = compare_float_tensors(ref, got)
    assert metrics["exact"] is False
    assert metrics["relative_l2"] is not None
    assert metrics["nan"] is False


def test_semantic_gate_rejects_nan() -> None:
    result = semantic_gate_bounded({"hidden": {127: 0.1, 128: float("nan"), 129: 0.1}})
    assert result["finite"] is False
    assert result["bounded"] is False


def test_semantic_gate_rejects_cross_request_contamination() -> None:
    states = [_state(request="A", batch_mode="B2", slot_id=0), _state(request="B", batch_mode="B2", slot_id=0)]
    assert validate_request_slot_mapping(states) is False


def test_bi_mlp_oracle_disabled_in_final_gate() -> None:
    assert final_gate_requires_bi_k_mode("bi_k", 0)
    assert not final_gate_requires_bi_k_mode("bi_k", 1)


def test_final_gate_requires_bi_k_mode() -> None:
    assert not final_gate_requires_bi_k_mode("bi_kv", 0)


def test_assignment_and_mask_drift_are_semantic_metrics() -> None:
    ref = torch.tensor([0, 1, 1, 2])
    got = torch.tensor([0, 1, 2, 2])
    assert difference_rate(ref, got) == 0.25


def test_top1_metrics_record_margin_without_forcing_failure() -> None:
    ref = torch.tensor([0.1, 1.0, 0.99, 0.0])
    got = torch.tensor([0.1, 0.98, 1.01, 0.0])
    metrics = topk_logit_metrics(ref, got)
    assert metrics["top1_equal"] is False
    assert metrics["reference_top1_margin"] < 0.02
