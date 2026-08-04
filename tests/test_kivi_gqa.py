import torch

from models.llama_kivi import repeat_kv_for_gqa


def test_repeat_kv_gqa_mapping_shapes_and_head_ids():
    bsz, hq, hkv, groups, dim = 1, 32, 8, 4, 128
    for q_len in (1, 7):
        query = torch.empty(bsz, hq, q_len, dim)
        assert query.shape[1] == hq
    for kv_len in (1, 127, 128, 129):
        kv = torch.arange(hkv, dtype=torch.float32).view(1, hkv, 1, 1).expand(bsz, hkv, kv_len, dim)
        repeated = repeat_kv_for_gqa(kv, groups, expected_heads=hq, tensor_name="kv")
        assert repeated.shape == (bsz, hq, kv_len, dim)
        assert torch.equal(repeated[:, 0], kv[:, 0])
        assert torch.equal(repeated[:, 1], kv[:, 0])
        assert torch.equal(repeated[:, 2], kv[:, 0])
        assert torch.equal(repeated[:, 3], kv[:, 0])
        assert torch.equal(repeated[:, 4], kv[:, 1])
        assert torch.equal(repeated[:, 7], kv[:, 1])
        assert torch.equal(repeated[:, 28], kv[:, 7])
        assert torch.equal(repeated[:, 31], kv[:, 7])


def test_repeat_kv_mha_regression_is_identity():
    x = torch.randn(1, 8, 11, 16)
    y = repeat_kv_for_gqa(x, 1, expected_heads=8, tensor_name="mha")
    assert y.shape == x.shape
    assert torch.equal(y, x)
