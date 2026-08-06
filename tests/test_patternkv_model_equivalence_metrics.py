from __future__ import annotations

import math

import torch

from bench.validate_patternkv_legacy_segmented import (
    cosine_similarity,
    disagreement_rate,
    kl_divergence,
    max_abs_error,
    nll_for_target,
    relative_mse,
    topk_overlap,
)


def test_vector_metric_correctness() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 4.0])
    assert math.isclose(cosine_similarity(a, a), 1.0, rel_tol=1e-6)
    assert relative_mse(a, a) == 0.0
    assert max_abs_error(a, b) == 1.0
    assert relative_mse(a, b) > 0.0


def test_probability_and_logits_metrics() -> None:
    logits_a = torch.tensor([0.1, 3.0, 2.0, -1.0])
    logits_b = torch.tensor([0.2, 2.9, 2.1, -1.0])
    probs_a = torch.softmax(logits_a, dim=-1)
    probs_b = torch.softmax(logits_b, dim=-1)
    assert kl_divergence(probs_a, probs_a) < 1e-7
    assert kl_divergence(probs_a, probs_b) >= 0.0
    assert topk_overlap(logits_a, logits_b, 2) == 2
    assert nll_for_target(logits_a, 1) < nll_for_target(logits_a, 3)


def test_assignment_and_gate_disagreement() -> None:
    a = torch.tensor([[[0, 1, 1, 2]]])
    b = torch.tensor([[[0, 0, 1, 3]]])
    assert disagreement_rate(a, a) == 0.0
    assert disagreement_rate(a, b) == 0.5
    assert disagreement_rate(None, b) is None

