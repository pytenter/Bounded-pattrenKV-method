from bench.gsm8k_utils import compute_stop_state, normalize_eos_token_ids


def test_output_shorter_than_limit_ending_with_eos():
    state = compute_stop_state([10, 11, 2], max_new_tokens=5, eos_token_ids=[2])
    assert state["last_generated_token_id"] == 2
    assert state["ended_with_eos"] is True
    assert state["hit_max_new_tokens"] is False
    assert state["length_truncated"] is False
    assert state["stop_reason"] == "eos"


def test_output_equal_limit_without_eos_is_length():
    state = compute_stop_state([10, 11, 12], max_new_tokens=3, eos_token_ids=[2])
    assert state["ended_with_eos"] is False
    assert state["hit_max_new_tokens"] is True
    assert state["length_truncated"] is True
    assert state["stop_reason"] == "length"


def test_output_equal_limit_with_eos_is_not_truncated():
    state = compute_stop_state([10, 11, 2], max_new_tokens=3, eos_token_ids=[2])
    assert state["ended_with_eos"] is True
    assert state["hit_max_new_tokens"] is True
    assert state["length_truncated"] is False
    assert state["stop_reason"] == "eos"


def test_multiple_eos_token_ids():
    state = compute_stop_state([10, 128009], max_new_tokens=2, eos_token_ids=[2, 128009])
    assert state["eos_token_ids"] == [2, 128009]
    assert state["ended_with_eos"] is True
    assert state["length_truncated"] is False


def test_empty_output():
    state = compute_stop_state([], max_new_tokens=3, eos_token_ids=[2])
    assert state["last_generated_token_id"] is None
    assert state["ended_with_eos"] is False
    assert state["hit_max_new_tokens"] is False
    assert state["length_truncated"] is False
    assert state["stop_reason"] == "unknown"


def test_normalize_eos_token_ids_accepts_int_and_list():
    assert normalize_eos_token_ids(2, [128001, 128009], None) == [2, 128001, 128009]
