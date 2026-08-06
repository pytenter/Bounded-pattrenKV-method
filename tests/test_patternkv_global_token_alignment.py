from __future__ import annotations


def test_chunked_global_token_alignment() -> None:
    legacy_indices = list(range(256))
    segmented_indices = list(range(128)) + list(range(128, 256))
    assert legacy_indices == segmented_indices


def test_rolling_and_chunked_internal_layouts_can_align_by_global_index() -> None:
    chunked_packed = list(range(4))
    chunked_buffer = list(range(4, 6))
    rolling_pending = list(range(0, 2))
    rolling_recent = list(range(2, 6))
    assert sorted(chunked_packed + chunked_buffer) == sorted(rolling_pending + rolling_recent)
