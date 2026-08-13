from __future__ import annotations

import torch

from bench.run_actual_model_k_assignment_trace import (
    cross_assignment,
    locate_first_divergence,
    assignment_with_margins,
    snapshot_phase,
    trace_enabled,
)


def test_trace_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_EQUIV_TRACE", raising=False)
    monkeypatch.delenv("PATTERNKV_K_ASSIGNMENT_TRACE", raising=False)

    assert trace_enabled() is False


def test_assignment_margin_helper_uses_minmax_distance() -> None:
    x = torch.tensor([[[[0.0, 1.0], [5.0, 5.0]]]])
    centroids = torch.tensor([[[0.0, 1.0], [5.0, 5.0]]])

    assignment, best, second, margin = assignment_with_margins(x, centroids)

    assert assignment.tolist() == [[[0, 1]]]
    assert torch.allclose(best, torch.tensor([[[0.0, 0.0]]]))
    assert torch.allclose(second, torch.tensor([[[1.0, 1.0]]]))
    assert torch.allclose(margin, torch.tensor([[[1.0, 1.0]]]))


def test_cross_assignment_reports_all_four_combinations() -> None:
    x_b1 = torch.tensor([[[[0.0, 1.0], [5.0, 5.0]]]])
    c_b1 = torch.tensor([[[0.0, 1.0], [5.0, 5.0]]])
    x_b2 = torch.tensor([[[[0.0, 1.0], [8.0, 8.0]]]])
    c_b2 = torch.tensor([[[0.0, 1.0], [8.0, 8.0]]])

    result = cross_assignment(x_b1, c_b1, x_b2, c_b2)

    assert set(result) == {
        "b1_k_b1_centroid",
        "b2_k_b2_centroid",
        "b1_k_b2_centroid",
        "b2_k_b1_centroid",
    }
    assert result["b1_k_b1_centroid"]["shape"] == [1, 1, 2]
    assert result["b1_k_b1_centroid"]["assignment"] == [[[0, 1]]]


def test_first_divergence_locator_uses_pipeline_order() -> None:
    rows = [
        {"component": "K_ASSIGNMENT", "pass": False, "request_row": 1},
        {"component": "K_PROJ", "pass": False, "request_row": 0},
        {"component": "HIDDEN_INPUT", "pass": True, "request_row": 0},
    ]

    assert locate_first_divergence(rows)["component"] == "K_PROJ"


def test_snapshot_phase_labels_prefill_and_decode1() -> None:
    assert snapshot_phase(False) == "PREFILL"
    assert snapshot_phase(True) == "DECODE1"
