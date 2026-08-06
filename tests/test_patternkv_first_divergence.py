from __future__ import annotations

from bench.validate_patternkv_legacy_segmented import classify_divergence, first_divergence


def test_first_divergence_identical_sequences() -> None:
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None


def test_first_divergence_first_token() -> None:
    assert first_divergence([1, 2, 3], [9, 2, 3]) == 0


def test_first_divergence_middle_token() -> None:
    assert first_divergence([1, 2, 3], [1, 2, 4]) == 2


def test_first_divergence_early_eos_or_length() -> None:
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def test_near_tie_classification() -> None:
    assert classify_divergence(1e-4, 2e-4, reference_diverged=False, cache_mismatch_before=False) == "near_tie_amplification"
    assert classify_divergence(0.1, 0.2, reference_diverged=True, cache_mismatch_before=False) == "algorithmic_mismatch"
    assert classify_divergence(0.1, 0.2, reference_diverged=False, cache_mismatch_before=True) == "algorithmic_mismatch"
