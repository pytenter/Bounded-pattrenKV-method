from __future__ import annotations

import torch

from bench.p2_first_divergence_utils import (
    canonical_request_state_from_fields,
    compare_canonical_states,
    compare_tensors,
    first_non_exact,
)


def test_tensor_comparator_shape_mismatch_not_100_percent_diff() -> None:
    result = compare_tensors(torch.zeros(2, 3), torch.zeros(2, 4))
    assert result["shape_equal"] is False
    assert result["comparable"] is False
    assert result["difference_rate"] is None
    assert result["relative_l2"] is None


def test_canonical_request_state_different_slot_same_content() -> None:
    fields = {"packed_k": torch.arange(4, dtype=torch.uint8)}
    a = canonical_request_state_from_fields(fields, metadata={"logical_tokens": 4})
    b = canonical_request_state_from_fields(fields, metadata={"logical_tokens": 4})
    a["metadata"]["slot"] = 0
    b["metadata"]["slot"] = 7
    a["metadata"].pop("slot")
    b["metadata"].pop("slot")
    assert compare_canonical_states(a, b)["exact"] is True


def test_canonical_request_state_different_capacity_same_content() -> None:
    a = canonical_request_state_from_fields(
        {"packed_v": torch.tensor([1, 2, 3, 99], dtype=torch.uint8)},
        token_axes={"packed_v": 0},
        logical_lengths={"packed_v": 3},
    )
    b = canonical_request_state_from_fields(
        {"packed_v": torch.tensor([1, 2, 3, 88, 77], dtype=torch.uint8)},
        token_axes={"packed_v": 0},
        logical_lengths={"packed_v": 3},
    )
    assert compare_canonical_states(a, b)["exact"] is True


def test_canonical_request_state_logical_length_mismatch() -> None:
    a = canonical_request_state_from_fields({"packed_v": torch.tensor([1, 2, 3], dtype=torch.uint8)})
    b = canonical_request_state_from_fields({"packed_v": torch.tensor([1, 2], dtype=torch.uint8)})
    result = compare_canonical_states(a, b)
    assert result["exact"] is False
    assert result["field_results"]["packed_v"]["comparable"] is False


def test_canonical_request_state_packed_byte_difference() -> None:
    a = canonical_request_state_from_fields({"packed_k": torch.tensor([1, 2, 3], dtype=torch.uint8)})
    b = canonical_request_state_from_fields({"packed_k": torch.tensor([1, 7, 3], dtype=torch.uint8)})
    result = compare_canonical_states(a, b)
    assert result["exact"] is False
    assert result["field_results"]["packed_k"]["difference_rate"] == 1 / 3


def test_canonical_request_state_assignment_difference() -> None:
    a = canonical_request_state_from_fields({"k_assignments": torch.tensor([0, 1, 2], dtype=torch.long)})
    b = canonical_request_state_from_fields({"k_assignments": torch.tensor([0, 2, 2], dtype=torch.long)})
    result = compare_canonical_states(a, b)
    assert result["exact"] is False
    assert result["field_results"]["k_assignments"]["difference_rate"] == 1 / 3


def test_cache_field_layout_contract() -> None:
    state = canonical_request_state_from_fields(
        {"v_precision_mask": torch.tensor([1, 0, 1, 9], dtype=torch.uint8)},
        token_axes={"v_precision_mask": 0},
        logical_lengths={"v_precision_mask": 3},
        metadata={"packed_v_tokens": 3},
    )
    assert tuple(state["fields"]["v_precision_mask"].shape) == (3,)
    assert state["metadata"]["packed_v_tokens"] == 3


def test_first_divergence_detector() -> None:
    rows = [
        {"layer": 0, "component": "A", "exact": True},
        {"layer": 0, "component": "B", "exact": False},
        {"layer": 1, "component": "A", "exact": False},
    ]
    assert first_non_exact(rows)["component"] == "B"


def test_layerwise_metric_helper() -> None:
    result = compare_tensors(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))
    assert result["shape_equal"] is True
    assert result["exact"] is False
    assert result["difference_rate"] == 0.5
    assert result["relative_l2"] > 0
