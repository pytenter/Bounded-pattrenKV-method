from __future__ import annotations

import torch

from bench.patternkv_equivalence_reference import tensor_pair_metrics


def test_tensor_pair_metrics_identical_tensors() -> None:
    value = torch.arange(12, dtype=torch.float16).reshape(1, 3, 4)

    metrics = tensor_pair_metrics(value, value.clone())

    assert metrics["cosine"] == 1.0
    assert metrics["relative_mse"] == 0.0
    assert metrics["relative_l2"] == 0.0
    assert metrics["max_abs_error"] == 0.0


def test_tensor_pair_metrics_small_perturbation() -> None:
    left = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
    right = torch.tensor([1.0, 2.5, 3.0], dtype=torch.float16)

    metrics = tensor_pair_metrics(left, right)

    assert metrics["cosine"] < 1.0
    assert metrics["relative_mse"] > 0.0
    assert metrics["relative_l2"] > 0.0
    assert metrics["max_abs_error"] == 0.5
