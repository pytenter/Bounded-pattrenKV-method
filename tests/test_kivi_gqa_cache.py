import torch
from transformers import LlamaConfig

from bench.paper_config import cache_storage_summary
from models.llama_kivi import LlamaFlashAttention_KIVI, repeat_kv_for_gqa


def _bytes(tensor):
    return tensor.numel() * tensor.element_size()


def test_persistent_cache_heads_do_not_expand_to_query_heads():
    bsz, hq, hkv, groups, dim = 1, 32, 8, 4, 128
    key_cache = torch.zeros(bsz, hkv, 129, dim, dtype=torch.float16)
    value_cache = torch.zeros(bsz, hkv, 129, dim, dtype=torch.float16)
    key_attn = repeat_kv_for_gqa(key_cache, groups, expected_heads=hq, tensor_name="key_cache")
    value_attn = repeat_kv_for_gqa(value_cache, groups, expected_heads=hq, tensor_name="value_cache")

    assert key_cache.shape[1] == hkv
    assert value_cache.shape[1] == hkv
    assert key_attn.shape[1] == hq
    assert value_attn.shape[1] == hq
    assert _bytes(key_cache) * groups == _bytes(key_attn)
    assert _bytes(value_cache) * groups == _bytes(value_attn)


def test_cache_storage_summary_reports_persistent_kv_heads():
    layer_cache = (
        None,
        torch.zeros(1, 8, 127, 128, dtype=torch.float16),
        None,
        None,
        None,
        torch.zeros(1, 8, 127, 128, dtype=torch.float16),
        None,
        None,
        127,
    )
    stats = cache_storage_summary("kivi_paper_g128", [layer_cache], total_cached_tokens=127, residual_length=128)
    assert stats["persistent_key_heads"] == 8
    assert stats["persistent_value_heads"] == 8


def test_kivi_attention_asserts_bad_persistent_cache_heads():
    cfg = LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=1,
        vocab_size=128,
    )
    cfg.k_bits = 2
    cfg.v_bits = 2
    cfg.group_size = 128
    cfg.residual_length = 128
    cfg.use_flash = True
    attn = LlamaFlashAttention_KIVI(cfg)
    bad_cache = torch.zeros(1, 32, 1, 128)
    try:
        attn._check_persistent_cache_heads(bad_cache, "key_states_full")
    except ValueError as exc:
        assert "num_key_value_heads=8" in str(exc)
        assert "num_attention_heads=32" in str(exc)
    else:
        raise AssertionError("expected bad persistent cache head count to fail")


def test_residual_boundaries_keep_persistent_heads_and_temporary_attention_heads():
    bsz, hq, hkv, groups, dim = 1, 32, 8, 4, 128
    for cached_len in (1, 127, 128, 129, 255, 256, 257):
        residual_len = min(cached_len, 128)
        quant_len = cached_len - residual_len
        residual_k = torch.zeros(bsz, hkv, residual_len, dim)
        residual_v = torch.zeros(bsz, hkv, residual_len, dim)
        assert residual_k.shape[1] == hkv
        assert residual_v.shape[1] == hkv
        assert repeat_kv_for_gqa(residual_k, groups, expected_heads=hq).shape[1] == hq
        assert repeat_kv_for_gqa(residual_v, groups, expected_heads=hq).shape[1] == hq
        if quant_len:
            quant_k = torch.zeros(bsz, hkv, dim, quant_len // 16)
            quant_v = torch.zeros(bsz, hkv, quant_len, dim // 16)
            assert quant_k.shape[1] == hkv
            assert quant_v.shape[1] == hkv
