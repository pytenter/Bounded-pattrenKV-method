from __future__ import annotations

import pytest
import torch

from models.segmented_cache import (
    finalize_segmented_attention_state,
    merge_segmented_attention_state,
    merge_segmented_attention_states,
    request_invariant_segment_attention_state,
)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, heads, tokens, dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, heads, n_rep, tokens, dim)
    return hidden_states.reshape(batch, heads * n_rep, tokens, dim)


def _segment_masks(lengths: torch.Tensor, physical: int) -> torch.Tensor:
    positions = torch.arange(physical, device=lengths.device, dtype=torch.long).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _reference_from_segments(
    score_parts: list[torch.Tensor],
    value_parts: list[torch.Tensor],
    valid_lengths: list[torch.Tensor],
    num_key_value_groups: int,
) -> torch.Tensor:
    masked_scores = []
    repeated_values = []
    for scores, values, lengths in zip(score_parts, value_parts, valid_lengths):
        if scores.shape[-1] == 0:
            continue
        mask = _segment_masks(lengths.to(device=scores.device), int(scores.shape[-1]))
        masked_scores.append(scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min))
        repeated_values.append(_repeat_kv(values, num_key_value_groups))
    full_scores = torch.cat(masked_scores, dim=-1)
    full_values = torch.cat(repeated_values, dim=2)
    return torch.matmul(torch.softmax(full_scores, dim=-1), full_values)


def _build_segments(batch: int, *, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(20260816 + batch)
    score_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    valid_lengths: list[torch.Tensor] = []
    for index, physical in enumerate((3, 4, 2, 5)):
        scores = torch.randn(batch, 4, 1, physical, device=device, dtype=torch.float32, generator=generator) * (index + 1) - float(index)
        values = torch.randn(batch, 2, physical, 6, device=device, dtype=torch.float32, generator=generator) * 0.1 + float(index)
        lengths = torch.tensor([max(physical - row, 1) for row in range(batch)], device=device, dtype=torch.long)
        score_parts.append(scores)
        value_parts.append(values)
        valid_lengths.append(lengths)
    return score_parts, value_parts, valid_lengths


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_segmented_state_merge_matches_concat_softmax(batch: int) -> None:
    device = torch.device("cpu")
    score_parts, value_parts, valid_lengths = _build_segments(batch, device=device)
    states = []
    for scores, values, lengths in zip(score_parts, value_parts, valid_lengths):
        mask = _segment_masks(lengths, int(scores.shape[-1]))
        masked_scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        local_probs = torch.softmax(masked_scores, dim=-1)
        local_output = torch.matmul(local_probs, _repeat_kv(values, 2))
        states.append(request_invariant_segment_attention_state(masked_scores, local_output, lengths))
    merged = merge_segmented_attention_states(states)
    got = finalize_segmented_attention_state(merged, dtype=torch.float32)
    ref = _reference_from_segments(score_parts, value_parts, valid_lengths, 2)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_segmented_state_merge_empty_segment_identity() -> None:
    device = torch.device("cpu")
    empty_scores = torch.empty(2, 4, 1, 0, device=device, dtype=torch.float32)
    empty_lengths = torch.zeros(2, dtype=torch.long, device=device)
    empty_state = request_invariant_segment_attention_state(
        empty_scores,
        torch.zeros(2, 4, 1, 6, device=device, dtype=torch.float32),
        empty_lengths,
    )

    scores = torch.tensor([[[[-2.0, 0.0, 1.0]]]], device=device, dtype=torch.float32).repeat(2, 4, 1, 1)
    values = torch.arange(2 * 2 * 3 * 6, device=device, dtype=torch.float32).reshape(2, 2, 3, 6) / 100.0
    lengths = torch.tensor([3, 2], device=device, dtype=torch.long)
    mask = _segment_masks(lengths, 3)
    local_output = torch.matmul(torch.softmax(scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min), dim=-1), _repeat_kv(values, 2))
    state = request_invariant_segment_attention_state(scores, local_output, lengths)

    left = merge_segmented_attention_state(empty_state, state)
    right = merge_segmented_attention_state(state, empty_state)
    torch.testing.assert_close(finalize_segmented_attention_state(left, dtype=torch.float32), finalize_segmented_attention_state(state, dtype=torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(finalize_segmented_attention_state(right, dtype=torch.float32), finalize_segmented_attention_state(state, dtype=torch.float32), rtol=0, atol=0)


def test_segmented_state_merge_extreme_logits() -> None:
    device = torch.device("cpu")
    scores_a = torch.tensor([[[[-1000.0, -1000.0, -999.0]]]], device=device, dtype=torch.float32).repeat(1, 4, 1, 1)
    scores_b = torch.tensor([[[[1000.0, 999.0, 998.0]]]], device=device, dtype=torch.float32).repeat(1, 4, 1, 1)
    values_a = torch.arange(1 * 2 * 3 * 6, device=device, dtype=torch.float32).reshape(1, 2, 3, 6)
    values_b = values_a + 10.0
    lengths = torch.tensor([3], device=device, dtype=torch.long)
    output_a = torch.matmul(torch.softmax(scores_a, dim=-1), _repeat_kv(values_a, 2))
    output_b = torch.matmul(torch.softmax(scores_b, dim=-1), _repeat_kv(values_b, 2))
    state_a = request_invariant_segment_attention_state(scores_a, output_a, lengths)
    state_b = request_invariant_segment_attention_state(scores_b, output_b, lengths)
    merged = merge_segmented_attention_state(state_a, state_b)
    got = finalize_segmented_attention_state(merged, dtype=torch.float32)
    ref = _reference_from_segments([scores_a, scores_b], [values_a, values_b], [lengths, lengths], 2)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_segmented_state_merge_respects_batch_reorder() -> None:
    device = torch.device("cpu")
    score_parts, value_parts, valid_lengths = _build_segments(2, device=device)
    states_ab = []
    states_ba = []
    for scores, values, lengths in zip(score_parts, value_parts, valid_lengths):
        mask = _segment_masks(lengths, int(scores.shape[-1]))
        masked_scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        local_probs = torch.softmax(masked_scores, dim=-1)
        local_output = torch.matmul(local_probs, _repeat_kv(values, 2))
        states_ab.append(request_invariant_segment_attention_state(masked_scores, local_output, lengths))
        states_ba.append(request_invariant_segment_attention_state(masked_scores[[1, 0]], local_output[[1, 0]], lengths[[1, 0]]))

    merged_ab = finalize_segmented_attention_state(merge_segmented_attention_states(states_ab), dtype=torch.float32)
    merged_ba = finalize_segmented_attention_state(merge_segmented_attention_states(states_ba), dtype=torch.float32)
    torch.testing.assert_close(merged_ab[[1, 0]], merged_ba, rtol=1e-5, atol=1e-5)
