from types import SimpleNamespace

import pytest
import torch

from models.segmented_cache import (
    CHUNKED_CACHE_MODE,
    CAPACITY_GROWTH_BASELINE,
    build_cache_from_prefill,
    get_capacity_cache_counters,
    normalize_capacity_growth_backend,
    reset_capacity_cache_counters,
    serialize_cache,
    deserialize_cache,
    tensor_tokens,
)
from models.llama_patternkv import patternkv_mixed_value_attention
from quant.matmul import (
    cuda_attn_v_fused_with_base,
    cuda_attn_v_fused_with_base_strided_v4,
    cuda_attn_v_mixed_fused_with_base,
    get_patternkv_strided_v2_reader_counters,
    patternkv_gqa_v_backend,
    patternkv_page_v_reader_backend,
    reset_patternkv_strided_v2_reader_counters,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA capacity integration tests require a GPU")

GROUP_SIZE = 128
NH = 32
NH_KV = 8
HEAD_DIM = 128
CENTROIDS = 16


def _states(tokens: int, *, heads: int = NH_KV, seed: int = 888):
    torch.manual_seed(seed + tokens + heads)
    key = (torch.randn(1, heads, tokens, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.25).contiguous()
    value = (torch.randn(1, heads, tokens, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.25).contiguous()
    centroids = (torch.randn(heads, CENTROIDS, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    return key, value, centroids


def _cache(tokens: int, *, backend: str, selector: str = "causal_v4", fraction: float = 0.25, chunk_length: int = 128, heads: int = NH_KV):
    key, value, centroids = _states(tokens, heads=heads)
    return build_cache_from_prefill(
        key,
        value,
        sink_length=0,
        recent_length=0,
        group_size=GROUP_SIZE,
        k_bits=2,
        v_bits=2,
        pattern=True,
        k_centroids=centroids,
        v_centroids=centroids,
        cache_mode=CHUNKED_CACHE_MODE,
        chunk_length=chunk_length,
        v_precision_selector=selector,
        v4_budget_fraction=fraction,
        selector_layer_idx=0,
    )


def _baseline_and_capacity(monkeypatch, tokens: int = 512):
    monkeypatch.setenv("PATTERNKV_CACHE_GROWTH_BACKEND", "baseline")
    baseline = _cache(tokens, backend="baseline")
    monkeypatch.setenv("PATTERNKV_CACHE_GROWTH_BACKEND", "fixed_capacity")
    reset_capacity_cache_counters()
    capacity = _cache(tokens, backend="fixed_capacity")
    return baseline, capacity


def _assert_same(a: torch.Tensor | None, b: torch.Tensor | None):
    assert torch.is_tensor(a) and torch.is_tensor(b)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_real_capacity_v2_append(monkeypatch):
    base, cap = _baseline_and_capacity(monkeypatch)
    _assert_same(cap.packed_v, base.packed_v)
    assert not cap.packed_v.is_contiguous()


def test_real_capacity_v4_append(monkeypatch):
    base, cap = _baseline_and_capacity(monkeypatch)
    _assert_same(cap.packed_v4, base.packed_v4)
    assert not cap.packed_v4.is_contiguous()


def test_real_capacity_metadata_append(monkeypatch):
    base, cap = _baseline_and_capacity(monkeypatch)
    _assert_same(cap.v_precision_mask, base.v_precision_mask)
    _assert_same(cap.v_pattern_mask, base.v_pattern_mask)
    _assert_same(cap.v_assignment_idx, base.v_assignment_idx)


def test_real_capacity_v2_compact_order(monkeypatch):
    base, cap = _baseline_and_capacity(monkeypatch)
    _assert_same(cap.v2_pattern_mask, base.v2_pattern_mask)
    _assert_same(cap.v2_assignment_idx, base.v2_assignment_idx)


def test_real_capacity_v4_compact_order(monkeypatch):
    base, cap = _baseline_and_capacity(monkeypatch)
    _assert_same(cap.v4_pattern_mask, base.v4_pattern_mask)
    _assert_same(cap.v4_assignment_idx, base.v4_assignment_idx)


def test_real_capacity_v4_identity(monkeypatch):
    base, cap = _baseline_and_capacity(monkeypatch)
    assert torch.equal(cap.v_precision_mask.bool().nonzero(), base.v_precision_mask.bool().nonzero())


def test_real_capacity_scale_zero(monkeypatch):
    base, cap = _baseline_and_capacity(monkeypatch)
    _assert_same(cap.packed_v_scale, base.packed_v_scale)
    _assert_same(cap.packed_v_zero, base.packed_v_zero)
    _assert_same(cap.packed_v4_scale, base.packed_v4_scale)
    _assert_same(cap.packed_v4_zero, base.packed_v4_zero)


def test_real_capacity_pattern_mask(monkeypatch):
    base, cap = _baseline_and_capacity(monkeypatch)
    _assert_same(cap.v_pattern_mask, base.v_pattern_mask)


def test_real_capacity_assignment(monkeypatch):
    base, cap = _baseline_and_capacity(monkeypatch)
    _assert_same(cap.v_assignment_idx, base.v_assignment_idx)


@pytest.mark.parametrize("tokens", [127, 128, 129])
def test_real_capacity_127_128_129(monkeypatch, tokens):
    monkeypatch.setenv("PATTERNKV_CACHE_GROWTH_BACKEND", "chunked_capacity")
    cap = _cache(tokens, backend="chunked_capacity")
    assert cap.packed_v_tokens == (tokens // 128) * 128


@pytest.mark.parametrize("tokens", [255, 256, 257])
def test_real_capacity_255_256_257(monkeypatch, tokens):
    monkeypatch.setenv("PATTERNKV_CACHE_GROWTH_BACKEND", "chunked_capacity")
    cap = _cache(tokens, backend="chunked_capacity")
    assert cap.packed_v_tokens == (tokens // 128) * 128


@pytest.mark.parametrize("tokens", [4095, 4096, 4097])
def test_real_capacity_chunk_4095_4096_4097(monkeypatch, tokens):
    monkeypatch.setenv("PATTERNKV_CACHE_GROWTH_BACKEND", "chunked_capacity")
    cap = _cache(tokens, backend="chunked_capacity", chunk_length=128)
    assert cap.packed_v_tokens == (tokens // 128) * 128
    assert cap.capacity_buffers


def test_real_capacity_unused_slots_not_read(monkeypatch):
    monkeypatch.setenv("PATTERNKV_CACHE_GROWTH_BACKEND", "fixed_capacity")
    cap = _cache(512, backend="fixed_capacity")
    for name in ("packed_v", "packed_v_scale", "packed_v_zero", "packed_v4", "packed_v4_scale", "packed_v4_zero"):
        buf = cap.capacity_buffers[name]
        if buf.length < buf.capacity():
            buf.storage.narrow(buf.token_dim, buf.length, buf.capacity() - buf.length).fill_(float("nan") if buf.storage.is_floating_point() else 2147483647)
    module = SimpleNamespace(group_size=GROUP_SIZE, num_heads=NH, num_key_value_heads=NH_KV, num_key_value_groups=NH // NH_KV)
    attn = torch.softmax(torch.randn(1, NH, 1, cap.packed_v_tokens, device="cuda", dtype=torch.float16), dim=-1).contiguous()
    out = patternkv_mixed_value_attention(module, cap, attn, cap.v_pattern_mask, cap.packed_v_tokens)
    assert int(torch.isnan(out).sum().item()) == 0


def test_real_capacity_zero_materialization(monkeypatch):
    monkeypatch.setenv("PATTERNKV_CACHE_GROWTH_BACKEND", "fixed_capacity")
    reset_capacity_cache_counters()
    _cache(512, backend="fixed_capacity")
    counters = get_capacity_cache_counters()
    assert counters["historical_materialization_calls"] == 0
    assert counters["historical_materialized_bytes"] == 0


def test_real_capacity_zero_historical_cat_when_no_growth(monkeypatch):
    monkeypatch.setenv("PATTERNKV_CACHE_GROWTH_BACKEND", "fixed_capacity")
    reset_capacity_cache_counters()
    _cache(512, backend="fixed_capacity")
    assert get_capacity_cache_counters()["historical_torch_cat_calls"] == 0


def test_real_capacity_attention_matches_baseline(monkeypatch):
    base, cap = _baseline_and_capacity(monkeypatch)
    module = SimpleNamespace(group_size=GROUP_SIZE, num_heads=NH, num_key_value_heads=NH_KV, num_key_value_groups=NH // NH_KV)
    torch.manual_seed(123)
    attn = torch.softmax(torch.randn(1, NH, 1, base.packed_v_tokens, device="cuda", dtype=torch.float16), dim=-1).contiguous()
    monkeypatch.setenv("PATTERNKV_CACHE_GROWTH_BACKEND", "baseline")
    base_out = patternkv_mixed_value_attention(module, base, attn, base.v_pattern_mask, base.packed_v_tokens)
    monkeypatch.setenv("PATTERNKV_CACHE_GROWTH_BACKEND", "fixed_capacity")
    reset_patternkv_strided_v2_reader_counters()
    cap_out = patternkv_mixed_value_attention(module, cap, attn, cap.v_pattern_mask, cap.packed_v_tokens)
    diff = (cap_out.float() - base_out.float()).abs()
    assert float(diff.max().item()) <= 5e-3
    assert get_patternkv_strided_v2_reader_counters()["strided_v2_calls"] >= 1
    assert get_patternkv_strided_v2_reader_counters()["strided_v4_calls"] >= 1


def test_strided_v4_matches_tight():
    from models.segmented_cache import quantize_pack_v_reference

    key, value, centroids = _states(2048)
    vq, scale, zero = quantize_pack_v_reference(value, GROUP_SIZE, 4)
    mask = torch.randint(0, 2, (1, NH_KV, 2048), device="cuda", dtype=torch.uint8)
    idx = torch.randint(0, CENTROIDS, (1, NH_KV, 2048), device="cuda", dtype=torch.int32)
    attn = torch.softmax(torch.randn(1, NH, 1, 2048, device="cuda", dtype=torch.float16), dim=-1).contiguous()
    tight = cuda_attn_v_fused_with_base(GROUP_SIZE, attn, vq, scale, zero, 4, centroids, mask, idx, NH, NH_KV)
    cap = 4096
    vq_cap = torch.empty(1, NH_KV, cap, HEAD_DIM // 8, device="cuda", dtype=torch.int32)
    vq_cap.fill_(0x7FFFFFFF)
    vq_cap[:, :, :2048, :] = vq
    scale_cap = torch.empty(1, NH_KV, cap, HEAD_DIM // GROUP_SIZE, device="cuda", dtype=torch.float16)
    scale_cap.fill_(float("nan"))
    scale_cap[:, :, :2048, :] = scale
    zero_cap = torch.empty_like(scale_cap)
    zero_cap.fill_(float("nan"))
    zero_cap[:, :, :2048, :] = zero
    mask_cap = torch.empty(1, NH_KV, cap, device="cuda", dtype=torch.uint8)
    mask_cap.fill_(255)
    mask_cap[:, :, :2048] = mask
    idx_cap = torch.empty(1, NH_KV, cap, device="cuda", dtype=torch.int32)
    idx_cap.fill_(2147483647)
    idx_cap[:, :, :2048] = idx
    out = cuda_attn_v_fused_with_base_strided_v4(
        GROUP_SIZE, attn, vq_cap[:, :, :2048, :], scale_cap[:, :, :2048, :], zero_cap[:, :, :2048, :], centroids, mask_cap[:, :, :2048], idx_cap[:, :, :2048], NH, NH_KV
    )
    torch.testing.assert_close(out, tight, rtol=5e-3, atol=5e-3)


def test_strided_mixed_v_matches_baseline(monkeypatch):
    base, cap = _baseline_and_capacity(monkeypatch)
    torch.manual_seed(321)
    attn = torch.softmax(torch.randn(1, NH, 1, base.packed_v_tokens, device="cuda", dtype=torch.float16), dim=-1).contiguous()
    monkeypatch.setenv("PATTERNKV_CACHE_GROWTH_BACKEND", "baseline")
    base_out = cuda_attn_v_mixed_fused_with_base(
        GROUP_SIZE, attn, base.packed_v, base.packed_v_scale, base.packed_v_zero, base.packed_v4, base.packed_v4_scale, base.packed_v4_zero,
        base.v_precision_mask, base.v_centroids, base.v_pattern_mask, base.v_assignment_idx, NH, NH_KV
    )
    monkeypatch.setenv("PATTERNKV_CACHE_GROWTH_BACKEND", "fixed_capacity")
    cap_out = cuda_attn_v_mixed_fused_with_base(
        GROUP_SIZE, attn, cap.packed_v, cap.packed_v_scale, cap.packed_v_zero, cap.packed_v4, cap.packed_v4_scale, cap.packed_v4_zero,
        cap.v_precision_mask, cap.v_centroids, cap.v_pattern_mask, cap.v_assignment_idx, NH, NH_KV,
        v2_mask_q=cap.v2_pattern_mask, v2_idx_q=cap.v2_assignment_idx, v4_mask_q=cap.v4_pattern_mask, v4_idx_q=cap.v4_assignment_idx
    )
    torch.testing.assert_close(cap_out, base_out, rtol=5e-3, atol=5e-3)


def test_fixed_capacity_backend_not_default_initially(monkeypatch):
    monkeypatch.delenv("PATTERNKV_CACHE_GROWTH_BACKEND", raising=False)
    assert normalize_capacity_growth_backend(None) == CAPACITY_GROWTH_BASELINE


def test_chunked_capacity_backend_not_default_initially(monkeypatch):
    monkeypatch.delenv("PATTERNKV_CACHE_GROWTH_BACKEND", raising=False)
    assert normalize_capacity_growth_backend(None) == CAPACITY_GROWTH_BASELINE


def test_gqa_experimental_remains_off(monkeypatch):
    monkeypatch.delenv("PATTERNKV_GQA_V_BACKEND", raising=False)
    assert patternkv_gqa_v_backend() == "baseline"


def test_page_native_reader_remains_off(monkeypatch):
    monkeypatch.delenv("PATTERNKV_PAGE_V_READER", raising=False)
    assert patternkv_page_v_reader_backend() == "contiguous"


def test_capacity_buffers_survive_serialization(monkeypatch):
    monkeypatch.setenv("PATTERNKV_CACHE_GROWTH_BACKEND", "fixed_capacity")
    cap = _cache(512, backend="fixed_capacity")
    restored = deserialize_cache(serialize_cache(cap), pattern=True)
    assert restored.capacity_backend == "fixed_capacity"
    assert restored.capacity_buffers
    _assert_same(restored.packed_v, cap.packed_v)
