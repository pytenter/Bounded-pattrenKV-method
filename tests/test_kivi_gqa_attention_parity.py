import math

import torch

from models.llama_kivi import kivi_gqa_attention_reference, repeat_kv_for_gqa


def explicit_attention(query, key, value, groups):
    key_rep = repeat_kv_for_gqa(key, groups, expected_heads=query.shape[1], tensor_name="key")
    value_rep = repeat_kv_for_gqa(value, groups, expected_heads=query.shape[1], tensor_name="value")
    weights = torch.softmax(torch.matmul(query, key_rep.transpose(2, 3)) / math.sqrt(query.shape[-1]), dim=-1)
    return torch.matmul(weights, value_rep), weights


def test_gqa_attention_reference_matches_explicit_repeat_for_qk_and_av():
    torch.manual_seed(7)
    query = torch.randn(1, 4, 2, 8)
    key = torch.randn(1, 2, 5, 8)
    value = torch.randn(1, 2, 5, 8)

    fixed_output, fixed_weights = kivi_gqa_attention_reference(query, key, value, 2)
    ref_output, ref_weights = explicit_attention(query, key, value, 2)

    torch.testing.assert_close(fixed_weights, ref_weights, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(fixed_output, ref_output, rtol=1e-4, atol=1e-4)


def test_gqa_qk_and_av_paths_have_query_head_count():
    query = torch.randn(1, 32, 1, 128)
    key = torch.randn(1, 8, 129, 128)
    value = torch.randn(1, 8, 129, 128)
    output, weights = kivi_gqa_attention_reference(query, key, value, 4)
    assert weights.shape == (1, 32, 1, 129)
    assert output.shape == (1, 32, 1, 128)
