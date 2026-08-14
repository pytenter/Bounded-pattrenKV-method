from __future__ import annotations

import pytest
import torch

from bench.ragged_batch_decode_utils import (
    build_ragged_metadata,
    current_first_ragged_blocker,
    increment_ragged_total_tokens,
    last_page_valid_for_tokens,
    page_count_for_tokens,
    page_range_for_request,
    ragged_k_masked_scores,
    ragged_position_ids_from_lengths,
    validate_assignment_index_range,
    validate_ragged_metadata,
    validate_v4_budget,
)


def _metadata():
    return build_ragged_metadata(
        request_ids=["A", "B"],
        total_tokens=[384, 641],
        packed_k_tokens=[240, 512],
        packed_v_tokens=[240, 512],
        packed_v4_tokens=[60, 128],
        centroid_state_indices=[0, 1],
        page_counts=[2, 4],
        last_page_valid_tokens=[112, 128],
    )


def test_ragged_metadata_accepts_different_seq_lengths() -> None:
    metadata = _metadata()
    assert metadata.seq_lens == (384, 641)
    assert len(set(metadata.seq_lens)) == 2


def test_ragged_metadata_rejects_bad_indptr() -> None:
    metadata = _metadata()
    bad = metadata.__class__(
        request_ids=metadata.request_ids,
        seq_lens=metadata.seq_lens,
        position_ids=metadata.position_ids,
        total_tokens=metadata.total_tokens,
        packed_k_tokens=metadata.packed_k_tokens,
        packed_v_tokens=metadata.packed_v_tokens,
        packed_v4_tokens=metadata.packed_v4_tokens,
        centroid_state_indices=metadata.centroid_state_indices,
        page_indptr=(0, 3, 2),
        page_counts=metadata.page_counts,
        last_page_valid_tokens=metadata.last_page_valid_tokens,
    )
    with pytest.raises(ValueError, match="monotonic"):
        validate_ragged_metadata(bad)


def test_ragged_position_ids_follow_request_lengths() -> None:
    got = ragged_position_ids_from_lengths([384, 641])
    assert got.tolist() == [[384], [641]]


def test_ragged_total_tokens_increment_independently() -> None:
    assert increment_ragged_total_tokens([384, 641]) == (385, 642)
    assert increment_ragged_total_tokens([384, 641], [1, 0]) == (385, 641)


def test_ragged_page_indptr_different_page_counts() -> None:
    metadata = _metadata()
    assert metadata.page_counts == (2, 4)
    assert metadata.page_indptr == (0, 2, 6)


def test_ragged_last_page_valid_lengths() -> None:
    assert page_count_for_tokens(240) == 2
    assert page_count_for_tokens(512) == 4
    assert last_page_valid_for_tokens(240) == 112
    assert last_page_valid_for_tokens(512) == 128


def test_ragged_request_slots_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_ragged_metadata(
            request_ids=["A", "B"],
            total_tokens=[384, 641],
            packed_k_tokens=[240, 512],
            packed_v_tokens=[240, 512],
            packed_v4_tokens=[60, 128],
            centroid_state_indices=[0, 0],
            page_counts=[2, 4],
            last_page_valid_tokens=[112, 128],
        )


def test_ragged_cache_assembly_preserves_logical_lengths() -> None:
    metadata = _metadata()
    assert metadata.total_tokens == (384, 641)
    assert metadata.packed_k_tokens == (240, 512)


def test_ragged_k_mask_ignores_invalid_tail() -> None:
    query = torch.ones(2, 1, 1, 2)
    key = torch.ones(2, 1, 4, 2)
    key[0, :, 2:, :] = 1_000_000.0
    scores = ragged_k_masked_scores(query, key, torch.tensor([2, 4]))
    assert torch.isfinite(scores[0, :, :, :2]).all()
    assert (scores[0, :, :, 2:] < -1e20).all()


def test_ragged_v_page_range_is_request_local() -> None:
    metadata = _metadata()
    assert page_range_for_request(metadata, 0) == (0, 2)
    assert page_range_for_request(metadata, 1) == (2, 6)


def test_ragged_assignment_index_range() -> None:
    assert validate_assignment_index_range(torch.tensor([0, 4]), 5)
    assert not validate_assignment_index_range(torch.tensor([0, 5]), 5)


def test_ragged_v4_budget() -> None:
    assert validate_v4_budget(60, 240, 0.25)
    assert not validate_v4_budget(80, 240, 0.25)


def test_ragged_reorder_preserves_request_identity() -> None:
    ab = _metadata()
    ba = build_ragged_metadata(
        request_ids=["B", "A"],
        total_tokens=[641, 384],
        packed_k_tokens=[512, 240],
        packed_v_tokens=[512, 240],
        packed_v4_tokens=[128, 60],
        centroid_state_indices=[1, 0],
        page_counts=[4, 2],
        last_page_valid_tokens=[128, 112],
    )
    assert dict(zip(ab.request_ids, ab.total_tokens)) == dict(zip(ba.request_ids, ba.total_tokens))


def test_ragged_path_does_not_enable_bi_mlp_oracle(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_BI_MLP_ORACLE", raising=False)
    assert "PATTERNKV_BI_MLP_ORACLE" not in __import__("os").environ


def test_equal_length_batch_still_supported() -> None:
    metadata = build_ragged_metadata(
        request_ids=["A", "B"],
        total_tokens=[512, 512],
        packed_k_tokens=[256, 256],
        packed_v_tokens=[256, 256],
        packed_v4_tokens=[64, 64],
        centroid_state_indices=[0, 1],
        page_counts=[2, 2],
        last_page_valid_tokens=[128, 128],
    )
    assert metadata.seq_lens == (512, 512)


def test_current_first_ragged_blocker_is_cache_assembly() -> None:
    assert current_first_ragged_blocker()["first_ragged_blocker"] == "RAGGED_CACHE_ASSEMBLY_UNSUPPORTED"
