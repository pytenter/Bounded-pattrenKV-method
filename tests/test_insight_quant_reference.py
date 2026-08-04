import torch

from insight.quant_reference import quantize_dequant_k_token_groups, quantize_dequant_v_head_dim


def test_k_reference_quantizes_token_groups_per_channel():
    k = torch.arange(1 * 2 * 128 * 4, dtype=torch.float16).view(1, 2, 128, 4)
    out = quantize_dequant_k_token_groups(k, bits=2, group_size=128)
    assert out.q.shape == k.shape
    assert out.dequant.shape == k.shape
    assert out.scale.shape == (1, 2, 4, 1)


def test_v_reference_quantizes_head_dim_per_token():
    v = torch.arange(1 * 2 * 3 * 128, dtype=torch.float16).view(1, 2, 3, 128)
    out = quantize_dequant_v_head_dim(v, bits=2, group_size=128)
    assert out.q.shape == v.shape
    assert out.dequant.shape == v.shape
    assert out.scale.shape == (1, 2, 3, 1)


def test_reference_quantizer_handles_constant_group_without_nan():
    v = torch.ones(1, 1, 2, 128, dtype=torch.float16)
    out = quantize_dequant_v_head_dim(v, bits=2, group_size=128)
    assert not torch.isnan(out.dequant).any()
