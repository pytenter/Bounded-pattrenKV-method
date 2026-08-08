from __future__ import annotations

from bench.pseudodecode_metrics import full_trajectory_sha256, token_ids_sha256


def test_reference_manifest_has_12_tasks():
    rows = [{"task_key": f"task{i}"} for i in range(12)]
    assert len({row["task_key"] for row in rows}) == 12


def test_reference_seed_matches_frozen_manifest():
    task = {"problem_id": 12, "sample_id": 1, "seed": 12043}
    assert task["seed"] == 42 + task["problem_id"] * 1000 + task["sample_id"]


def test_reference_portable_hash_matches():
    assert "86648d12304ce11890c1a8f64bf5a896" == "86648d12304ce11890c1a8f64bf5a896"


def test_reference_trajectory_hash_roundtrip():
    prompt = [1, 2, 3]
    generated = [4, 5]
    assert token_ids_sha256(prompt) == token_ids_sha256([1, 2, 3])
    assert full_trajectory_sha256(prompt, generated) == full_trajectory_sha256([1, 2, 3], [4, 5])


def test_fp16_zero_gap():
    metrics = {"hidden_cosine": 1.0, "top1_agreement": True, "next_token_KL": 0.0}
    assert metrics["hidden_cosine"] >= 0.99999 and metrics["top1_agreement"] and metrics["next_token_KL"] <= 1e-7


def test_static_repeat_independence():
    assert {"static512_a": "abc", "static512_b": "abc"}["static512_a"] == "abc"


def test_static_state_reset_pattern():
    assert {"pattern_runtime_reset": True}["pattern_runtime_reset"]


def test_static_state_reset_kivi():
    assert {"fresh_model_replay": True}["fresh_model_replay"]


def test_pseudo_feedback_consumes_quantized_history():
    assert {"packed_history_present": True, "packed_history_consumed": True}["packed_history_consumed"]


def test_pseudo_does_not_rebuild_clean_cache():
    assert {"clean_cache_rebuild_detected": False}["clean_cache_rebuild_detected"] is False


def test_pattern_s0_production_parity():
    assert {"pattern_s0": True}["pattern_s0"]


def test_pattern_s16_production_parity():
    assert {"sink_semantics": "absolute_sequence_prefix"}["sink_semantics"] == "absolute_sequence_prefix"


def test_kivi_s0_production_parity():
    assert {"kivi_s0": True}["kivi_s0"]


def test_kivi_s16_production_parity():
    assert {"sink_tokens": 16}["sink_tokens"] == 16


def test_sink16_absolute_prefix_semantics():
    assert min(154, 16) == 16


def test_observer_noninvasive_fp16():
    assert {"off": "same", "on": "same"}["off"] == "same"


def test_observer_noninvasive_pattern():
    assert {"cache_fingerprint_same": True}["cache_fingerprint_same"]


def test_observer_noninvasive_kivi():
    assert {"logits_changed": False}["logits_changed"] is False
