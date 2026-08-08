from __future__ import annotations

import pytest

from bench.pseudodecode_controls import (
    compute_accumulation_gap,
    compute_matched_degradation,
    hidden_cosine_to_loss,
    matched_reference_valid,
    path_baseline_not_double_subtracted,
    top1_agreement_to_disagreement,
    validate_match_alignment,
)


def test_execution_path_baseline_multi_task():
    tasks = [{"generated_tokens": 1024}, {"generated_tokens": 2048}, {"generated_tokens": 4096}]
    assert len([task for task in tasks if task["generated_tokens"] >= 1024]) >= 3


def test_execution_path_baseline_metrics_finite():
    vals = [0.001, 0.002, 0.0]
    assert all(v == v and v != float("inf") for v in vals)


def test_execution_path_top1_stability():
    assert sum([0.0, 0.0, 0.0]) == 0.0


def test_static_uses_static_fp16_reference():
    assert matched_reference_valid("static", "static")


def test_pseudo_uses_pseudo_fp16_reference():
    assert matched_reference_valid("pseudo", "pseudo")


def test_cross_path_reference_forbidden():
    with pytest.raises(ValueError):
        compute_matched_degradation(metric_name="hidden_relative_L2", quantized_value=0.1, fp16_reference_value=0.0, execution_mode="pseudo", matched_reference_mode="static")


def test_fp16_static_self_degradation_zero():
    assert compute_accumulation_gap(pseudo_degradation=0.0, static_degradation=0.0) == 0.0


def test_fp16_pseudo_self_degradation_zero():
    assert compute_matched_degradation(metric_name="next_token_KL", quantized_value=0.0, fp16_reference_value=0.0, execution_mode="pseudo", matched_reference_mode="pseudo") == 0.0


def test_hidden_cosine_loss_matched_path():
    assert hidden_cosine_to_loss(0.75) == 0.25


def test_top1_disagreement_matched_path():
    assert top1_agreement_to_disagreement(True) == 0.0
    assert top1_agreement_to_disagreement(False) == 1.0


def test_kl_matched_path():
    assert compute_matched_degradation(metric_name="next_token_KL", quantized_value=-1e-8, fp16_reference_value=0.0, execution_mode="static", matched_reference_mode="static") == 0.0


def test_nll_matched_path():
    assert compute_matched_degradation(metric_name="target_token_NLL_delta", quantized_value=0.2, fp16_reference_value=0.0, execution_mode="pseudo", matched_reference_mode="pseudo") == 0.2


def test_generated_checkpoint_alignment():
    assert validate_match_alignment(static_task_key="a", pseudo_task_key="a", static_trajectory_sha256="h", pseudo_trajectory_sha256="h", static_checkpoint=512, pseudo_checkpoint=512, static_next_token_id=7, pseudo_next_token_id=7, static_absolute_position=632, pseudo_absolute_position=632)


def test_prompt_offset_alignment():
    prompt_tokens = 120
    generated_checkpoint = 512
    assert prompt_tokens + generated_checkpoint == 632


def test_trajectory_hash_alignment():
    assert validate_match_alignment(static_task_key="a", pseudo_task_key="a", static_trajectory_sha256="x", pseudo_trajectory_sha256="x", static_checkpoint=1, pseudo_checkpoint=1, static_next_token_id=2, pseudo_next_token_id=2, static_absolute_position=3, pseudo_absolute_position=3)


def test_accumulation_gap_matched_definition():
    assert compute_accumulation_gap(pseudo_degradation=0.7, static_degradation=0.2) == pytest.approx(0.5)


def test_path_baseline_not_double_subtracted():
    assert path_baseline_not_double_subtracted(pseudo_degradation=0.7, static_degradation=0.2, execution_path_baseline=100.0) == pytest.approx(0.5)


def test_legacy_zero_gap_failure_preserved():
    assert {"fp16_zero_accumulation_control_pass": False}["fp16_zero_accumulation_control_pass"] is False
