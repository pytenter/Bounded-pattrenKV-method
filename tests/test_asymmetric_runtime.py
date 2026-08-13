from types import SimpleNamespace

import pytest
import torch

from models.llama_patternkv import patternkv_mixed_value_attention
from models.segmented_cache import (
    build_cache_from_prefill,
    get_capacity_cache_counters,
    reset_capacity_cache_counters,
)
from quant.matmul import (
    get_patternkv_strided_k_reader_counters,
    patternkv_gqa_v_backend,
    patternkv_page_v_reader_backend,
    reset_patternkv_strided_k_reader_counters,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="asymmetric runtime tests require CUDA")

GROUP_SIZE = 128
NH = 32
NH_KV = 8
HEAD_DIM = 128
CENTROIDS = 16


def _states(tokens: int, *, seed: int = 20260813):
    generator = torch.Generator(device="cuda").manual_seed(seed + tokens)
    key = (torch.randn(1, NH_KV, tokens, HEAD_DIM, device="cuda", dtype=torch.float16, generator=generator) * 0.25).contiguous()
    value = (torch.randn(1, NH_KV, tokens, HEAD_DIM, device="cuda", dtype=torch.float16, generator=generator) * 0.25).contiguous()
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device="cuda", dtype=torch.float16, generator=generator) * 0.1).contiguous()
    return key, value, centroids


def _cache(monkeypatch, backend: str, *, tokens: int = 512):
    monkeypatch.setenv("PATTERNKV_CACHE_GROWTH_BACKEND", backend)
    monkeypatch.setenv("PATTERNKV_CACHE_FIXED_CAPACITY_TOKENS", "33792")
    key, value, centroids = _states(tokens)
    return build_cache_from_prefill(
        key,
        value,
        sink_length=16,
        recent_length=128,
        group_size=GROUP_SIZE,
        k_bits=2,
        v_bits=2,
        pattern=True,
        k_centroids=centroids,
        v_centroids=centroids,
        cache_mode="segmented_rolling",
        chunk_length=GROUP_SIZE,
        value_objective="base",
        v_precision_selector="causal_v4",
        v4_budget_fraction=0.25,
        random_selector_seed=44,
        selector_task_key="asymmetric-runtime-test",
        selector_layer_idx=0,
    )


def _run_mixed_value(cache):
    module = SimpleNamespace(group_size=GROUP_SIZE, num_heads=NH, num_key_value_heads=NH_KV, num_key_value_groups=NH // NH_KV)
    torch.manual_seed(123)
    attn = torch.softmax(torch.randn(1, NH, 1, cache.packed_v_tokens, device="cuda", dtype=torch.float16), dim=-1).contiguous()
    return patternkv_mixed_value_attention(module, cache, attn, cache.v_pattern_mask, cache.packed_v_tokens)


def test_asymmetric_backend_keeps_k_tight(monkeypatch):
    cache = _cache(monkeypatch, "chunked_capacity")
    assert torch.is_tensor(cache.packed_k)
    assert cache.packed_k.is_contiguous()
    assert torch.is_tensor(cache.packed_k_scale)
    assert cache.packed_k_scale.is_contiguous()
    assert "packed_k" not in cache.capacity_buffers


def test_asymmetric_fixed_uses_v_capacity(monkeypatch):
    cache = _cache(monkeypatch, "fixed_capacity")
    assert cache.capacity_buffers
    assert "packed_v" in cache.capacity_buffers
    assert "packed_v4" in cache.capacity_buffers
    assert torch.is_tensor(cache.packed_v)
    assert not cache.packed_v.is_contiguous()


def test_asymmetric_chunked_uses_v_capacity(monkeypatch):
    cache = _cache(monkeypatch, "chunked_capacity")
    assert cache.capacity_buffers
    assert "v_precision_mask" in cache.capacity_buffers
    assert torch.is_tensor(cache.packed_v4)
    assert not cache.packed_v4.is_contiguous()


def test_asymmetric_no_strided_k(monkeypatch):
    reset_patternkv_strided_k_reader_counters()
    cache = _cache(monkeypatch, "fixed_capacity")
    _run_mixed_value(cache)
    counters = get_patternkv_strided_k_reader_counters()
    assert counters["strided_k_reader_calls"] == 0
    assert counters["strided_k_materialize_calls"] == 0


def test_asymmetric_no_v_materialization(monkeypatch):
    for backend in ("fixed_capacity", "chunked_capacity"):
        reset_capacity_cache_counters()
        cache = _cache(monkeypatch, backend)
        _run_mixed_value(cache)
        counters = get_capacity_cache_counters()
        assert counters["historical_materialization_calls"] == 0
        assert counters["historical_materialized_bytes"] == 0


def test_asymmetric_selector_identity(monkeypatch):
    caches = [_cache(monkeypatch, backend, tokens=1024) for backend in ("baseline", "fixed_capacity", "chunked_capacity")]
    masks = [cache.v_precision_mask[:, : cache.packed_v_tokens] for cache in caches]
    torch.testing.assert_close(masks[1], masks[0], rtol=0, atol=0)
    torch.testing.assert_close(masks[2], masks[0], rtol=0, atol=0)


def test_asymmetric_page_reader_off(monkeypatch):
    monkeypatch.delenv("PATTERNKV_PAGE_V_READER", raising=False)
    assert patternkv_page_v_reader_backend() == "contiguous"


def test_asymmetric_gqa_experimental_off(monkeypatch):
    monkeypatch.delenv("PATTERNKV_GQA_V_BACKEND", raising=False)
    assert patternkv_gqa_v_backend() == "baseline"
