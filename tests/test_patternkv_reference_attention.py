from __future__ import annotations

import torch

from bench.patternkv_equivalence_reference import reference_attention, reference_logits_metrics


def test_reference_attention_matches_manual_softmax() -> None:
    q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float16)
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float16)
    v = torch.tensor([[[[2.0, 0.0], [0.0, 4.0]]]], dtype=torch.float16)
    out = reference_attention(q, k, v)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / (2**0.5)
    expected = torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(torch.float16)
    assert torch.allclose(out, expected)


def test_reference_logits_metrics() -> None:
    a = torch.tensor([1.0, 3.0, 2.0])
    b = torch.tensor([1.1, 2.9, 2.0])
    metrics = reference_logits_metrics(a, b)
    assert metrics["top1_agreement"] is True
    assert metrics["cosine"] > 0.99
