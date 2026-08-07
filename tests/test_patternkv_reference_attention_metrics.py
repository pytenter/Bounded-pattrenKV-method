from __future__ import annotations

import torch

from bench.patternkv_equivalence_reference import attention_probability_metrics, tensor_pair_metrics


def test_masked_attention_score_metrics_ignore_infinite_values() -> None:
    left = torch.tensor([0.1, float("-inf"), 0.3, float("-inf")])
    right = torch.tensor([0.1, float("-inf"), 0.4, float("-inf")])

    metrics = tensor_pair_metrics(left, right, finite_only=True)

    assert metrics["count"] == 2
    assert metrics["relative_mse"] > 0.0
    assert torch.isfinite(torch.tensor(metrics["cosine"]))


def test_attention_probability_metrics_identical_distribution_has_zero_kl() -> None:
    probs = torch.tensor([[[[0.25, 0.25, 0.5]]]], dtype=torch.float16)

    metrics = attention_probability_metrics(probs, probs.clone())

    assert metrics["max_abs_error"] == 0.0
    assert abs(metrics["kl_legacy_segmented"]) < 1e-7
    assert abs(metrics["kl_segmented_legacy"]) < 1e-7
    assert abs(metrics["symmetric_kl"]) < 1e-7
