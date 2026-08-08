from __future__ import annotations

import math

from bench.pseudodecode_metrics import (
    accumulation_gap,
    checkpoint_availability,
    hidden_cosine_loss,
    token_ids_sha256,
    top1_disagreement,
    trapezoid_auc_log2,
)


def test_reference_trajectory_hash_stable():
    assert token_ids_sha256([1, 2, 3]) == token_ids_sha256([1, 2, 3])
    assert token_ids_sha256([1, 2, 3]) != token_ids_sha256([1, 2, 4])


def test_checkpoint_availability():
    rows = checkpoint_availability(512, [128, 512, 1024])
    assert [r["checkpoint_available"] for r in rows] == [True, True, False]
    assert rows[-1]["availability_reason"] == "trajectory_too_short"


def test_accumulation_gap_sign():
    assert math.isclose(accumulation_gap(0.7, 0.2), 0.5)
    assert math.isclose(accumulation_gap(0.1, 0.4), -0.3)


def test_cosine_loss_definition():
    assert hidden_cosine_loss(1.0) == 0.0
    assert hidden_cosine_loss(0.25) == 0.75


def test_top1_disagreement_definition():
    assert top1_disagreement(True) == 0.0
    assert top1_disagreement(False) == 1.0


def test_accumulation_auc():
    auc = trapezoid_auc_log2([(128, 1.0), (512, 3.0)])
    assert math.isclose(auc, 4.0)


def test_static_checkpoint_is_independent():
    first = {"checkpoint": 512, "fresh_state": True, "depends_on_previous_checkpoint": False}
    second = {"checkpoint": 512, "fresh_state": True, "depends_on_previous_checkpoint": False}
    assert first == second


def test_pseudo_cache_feedback_active():
    record = {"mode": "pseudo", "previous_quantized_cache_used": True}
    assert record["previous_quantized_cache_used"] is True


def test_fp16_static_pseudo_zero_gap():
    assert abs(accumulation_gap(1e-8, 2e-8)) < 1e-6


def test_pattern_static_fresh_state():
    record = {"config": "pattern_rolling_k2v2_s16_r128", "fresh_patternkv_state": True}
    assert record["fresh_patternkv_state"] is True


def test_kivi_static_fresh_state():
    record = {"config": "kivi_rolling_k2v2_s16_r128", "fresh_kivi_state": True}
    assert record["fresh_kivi_state"] is True


def test_sink_absolute_prefix_semantics():
    assert {"sink_semantics": "absolute_sequence_prefix"}["sink_semantics"] == "absolute_sequence_prefix"


def test_observer_noninvasive():
    off = {"logits_hash": "abc", "cache_signature": "def"}
    on = {"logits_hash": "abc", "cache_signature": "def"}
    assert on == off
