from scripts.run_wave1a4_attention_observer import classify_time_series, first_divergence, split_task_keys


def test_first_divergence_detects_common_prefix() -> None:
    out = first_divergence([1, 2, 3, 4], [1, 2, 9, 4])
    assert out["common_prefix_length"] == 2
    assert out["first_divergence_token"] == 3


def test_first_divergence_handles_identical_sequences() -> None:
    out = first_divergence([1, 2, 3], [1, 2, 3])
    assert out["common_prefix_length"] == 3
    assert out["first_divergence_token"] is None


def test_split_task_keys_ignores_empty_fields() -> None:
    assert split_task_keys("a;b;;") == ["a", "b"]


def test_time_series_classification_decay() -> None:
    assert classify_time_series([(512, 0.8), (1024, 0.3), (2048, 0.2)]) == "EARLY_MASS_DECAY"


def test_time_series_classification_persistent() -> None:
    assert classify_time_series([(512, 0.5), (1024, 0.45), (2048, 0.48)]) == "EARLY_MASS_PERSISTENT"


def test_time_series_classification_late_rebound() -> None:
    assert classify_time_series([(512, 0.1), (1024, 0.12), (2048, 0.3)]) == "EARLY_MASS_LATE_REBOUND"
