from __future__ import annotations

import torch

from bench.aime24_int2_wave1 import make_channel_mask, mask_hash, mixed_key_qk_reference, quantize_dequantize_per_channel


def test_mask_selects_twelve_point_five_percent() -> None:
    scores = torch.arange(2 * 3 * 128, dtype=torch.float32).reshape(2, 3, 128)
    mask = make_channel_mask(scores, 0.125)
    assert mask.shape == scores.shape
    assert torch.equal(mask.sum(dim=-1), torch.full((2, 3), 16))


def test_mask_hash_is_reproducible() -> None:
    scores = torch.ones(1, 2, 128)
    mask_a = make_channel_mask(scores, 0.125)
    mask_b = make_channel_mask(scores, 0.125)
    assert mask_hash(mask_a) == mask_hash(mask_b)


def test_mixed_key_qk_matches_manual_split() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 6, 8)
    mask = torch.tensor([True, False, True, False, False, True, False, False])
    got = mixed_key_qk_reference(q, k, mask)
    expected = torch.matmul(q[..., mask], quantize_dequantize_per_channel(k[..., mask], 4).transpose(-2, -1))
    expected = expected + torch.matmul(q[..., ~mask], quantize_dequantize_per_channel(k[..., ~mask], 2).transpose(-2, -1))
    assert torch.allclose(got, expected)
