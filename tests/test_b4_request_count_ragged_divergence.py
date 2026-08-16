from __future__ import annotations

import torch

from scripts.b4_request_count_ragged_divergence import (
    compare_cases,
    geometry_logical_match,
    input_state_match,
)


def state(value: float) -> list[dict]:
    tensor = torch.tensor([[[[value]]]])
    return [
        {
            "layer": 0,
            "lengths": {"total": 1, "sink": 0, "recent": 1, "pending": 0, "packed_k": 0, "packed_v": 0, "packed_v4": 0},
            "page": {"present": True, "num_pages": 0, "seq_len": 1, "valid_tokens": []},
            "centroid": {"k_count": 0, "v_count": 0, "updates_k": 0, "updates_v": 0, "last_flush_pos": -1, "active": True},
            "hashes": {"recent_k": str(value)},
            "_tensors": {"recent_k": tensor},
        }
    ]


def case(values: list[float], requests: list[str] | None = None) -> dict:
    timeline = []
    states = {}
    for idx, value in enumerate(values, start=1):
        states[str(idx)] = state(value)
        timeline.append(
            {
                "step": idx,
                "input_hidden": torch.tensor([[value]]),
                "final_hidden": torch.tensor([[value]]),
                "logits": torch.tensor([[value]]),
                "token": int(value),
                "state": state(value),
            }
        )
    return {
        "requests": requests or ["A", "B"],
        "target": "B",
        "target_row": 1,
        "timeline": timeline,
        "states": states,
        "transitions": [
            {"step": idx, "events": [], "after": [{"lengths": state(value)[0]["lengths"], "page": state(value)[0]["page"]}]}
            for idx, value in enumerate(values, start=1)
        ],
    }


def test_b4_known_good_b2_control() -> None:
    good = case([1.0, 2.0])
    cmp = compare_cases(good, good)
    assert cmp["first_any"] is None
    assert cmp["first_persistent"] is None


def test_b4_request_b_temporal_timeline() -> None:
    good = case([1.0, 2.0])
    bad = case([1.0, 3.0])
    cmp = compare_cases(good, bad)
    assert len(cmp["timeline"]) == 2


def test_b4_first_bad_step_locator() -> None:
    good = case([1.0, 2.0, 3.0])
    bad = case([1.0, 2.0, 4.0])
    cmp = compare_cases(good, bad)
    assert cmp["first_persistent"]["step"] == 3


def test_b4_request_count_ladder() -> None:
    ladder = {"2": None, "3": None, "4": {"step": 6}}
    min_bad = next(count for count, key in [(3, "3"), (4, "4")] if ladder[key] is not None)
    assert min_bad == 4


def test_b4_request_row_metadata_mapping() -> None:
    assert case([1.0], ["A", "B"])["target_row"] == 1
    assert case([1.0], ["A", "B", "C", "D"])["target_row"] == 1


def test_b4_seq_len_row_mapping() -> None:
    good = case([1.0])
    bad = case([1.0], ["A", "B", "C", "D"])
    assert geometry_logical_match(good, bad, "seq_len") is True


def test_b4_workspace_region_nonoverlap() -> None:
    regions = [(0, 16), (16, 32), (32, 48), (48, 64)]
    assert all(a[1] <= b[0] for a, b in zip(regions, regions[1:]))


def test_b4_peer_content_independence() -> None:
    good = case([1.0])
    bad = case([1.0], ["A", "B", "C", "D"])
    assert compare_cases(good, bad)["first_persistent"] is None


def test_b4_input_state_match_uses_previous_step() -> None:
    good = case([1.0, 2.0])
    bad = case([1.0, 3.0])
    cmp = compare_cases(good, bad)
    assert input_state_match(cmp, 2) is True
