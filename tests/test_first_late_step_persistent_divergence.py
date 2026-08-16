from __future__ import annotations

import torch

from scripts.first_late_step_persistent_divergence import (
    compare_cache_snapshots,
    control_summary,
    metric,
    summarize_cache_diff,
    timeline_step_exact,
    transition_events,
)


def layer(component: torch.Tensor, *, recent: int = 1, pending: int = 0, packed_k: int = 0, packed_v: int = 0) -> dict:
    return {
        "layer": 0,
        "lengths": {
            "total": recent + pending + packed_k,
            "sink": 0,
            "recent": recent,
            "pending": pending,
            "packed_k": packed_k,
            "packed_v": packed_v,
            "packed_v4": 0,
        },
        "page": {"present": True, "num_pages": 0, "seq_len": recent + pending + packed_v, "valid_tokens": []},
        "centroid": {"k_count": 0, "v_count": 0, "updates_k": 0, "updates_v": 0, "last_flush_pos": -1, "active": True},
        "hashes": {"recent_k": "same"},
        "_tensors": {"recent_k": component},
    }


def test_multistep_first_bad_step_locator() -> None:
    timeline = [
        {"step": 1, "input_hidden": {"exact_equal": True}, "final_hidden": {"exact_equal": True}, "logits": {"exact_equal": True}, "persistent_state": {"exact_equal": True}},
        {"step": 2, "input_hidden": {"exact_equal": True}, "final_hidden": {"exact_equal": False}, "logits": {"exact_equal": False}, "persistent_state": {"exact_equal": True}},
    ]
    assert timeline_step_exact(timeline, 1) is True
    assert timeline_step_exact(timeline, 2) is False


def test_multistep_active_semantic_state_canonicalization() -> None:
    ref = [layer(torch.tensor([[[[1.0, 2.0]]]]))]
    got = [layer(torch.tensor([[[[1.0, 2.0]]]]))]
    got[0]["page"]["request_indptr"] = 99
    diff = compare_cache_snapshots(got, ref)
    assert diff["exact_equal"] is True


def test_multistep_transition_timeline() -> None:
    before = [layer(torch.ones(1, 1, 1, 1), recent=128, pending=127, packed_k=128, packed_v=128)]
    after = [layer(torch.ones(1, 1, 1, 1), recent=1, pending=0, packed_k=256, packed_v=256)]
    events = transition_events(before, after)
    assert "recent_overflow" in events
    assert "pending_append_or_reset" in events
    assert "pending_to_packed" in events


def test_multistep_persistent_state_first_divergence() -> None:
    ref = [layer(torch.tensor([[[[1.0, 2.0]]]]))]
    got = [layer(torch.tensor([[[[1.0, 3.0]]]]))]
    diff = summarize_cache_diff(compare_cache_snapshots(got, ref))
    assert diff["exact_equal"] is False
    assert diff["first_diff"]["component"] == "recent_k"
    assert diff["first_diff"]["mismatch_count"] == 1


def test_multistep_peer_length_first_divergence() -> None:
    primary = {"step": 8, "state": "pending_k"}
    control = {"step": 10, "state": "pending_k"}
    summary = control_summary(primary, control)
    assert summary["same_first_bad_step"] is False


def test_multistep_peer_content_first_divergence() -> None:
    primary = {"step": 8, "state": "pending_k"}
    control = {"step": 8, "state": "pending_v"}
    summary = control_summary(primary, control)
    assert summary["same_first_bad_state"] is False


def test_multistep_reorder_first_divergence() -> None:
    primary = {"step": 8, "state": "pending_k"}
    control = {"step": 8, "state": "pending_k"}
    summary = control_summary(primary, control)
    assert summary["same_first_bad_step"] is True
    assert summary["same_first_bad_state"] is True


def test_multistep_metric_reports_exact_and_nonexact() -> None:
    exact = metric(torch.tensor([1.0]), torch.tensor([1.0]))
    nonexact = metric(torch.tensor([2.0]), torch.tensor([1.0]))
    assert exact["exact_equal"] is True
    assert nonexact["exact_equal"] is False
    assert nonexact["max_abs"] == 1.0
