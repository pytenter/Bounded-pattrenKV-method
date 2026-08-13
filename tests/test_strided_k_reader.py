import math

import pytest
import torch

from models.segmented_cache import dequantize_k_reference, pattern_gather_centroids, quantize_pack_k_reference
from quant.matmul import (
    cuda_attn_v_fused_with_base_strided_v2,
    cuda_bmm_fA_qB_outer_with_base,
    cuda_bmm_fA_qB_outer_with_base_strided_k,
    get_patternkv_strided_k_reader_counters,
    get_patternkv_strided_v2_reader_counters,
    patternkv_cache_growth_backend,
    patternkv_gqa_v_backend,
    patternkv_page_v_reader_backend,
    reset_patternkv_strided_k_reader_counters,
    reset_patternkv_strided_v2_reader_counters,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA strided K reader tests require a GPU")

GROUP_SIZE = 128
BITS = 2
NH = 32
NH_KV = 8
HEAD_DIM = 128
CENTROIDS = 16


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _round_group(tokens: int) -> int:
    return _ceil_div(tokens, GROUP_SIZE) * GROUP_SIZE


def _capacity_for(tokens: int) -> int:
    if tokens >= 32768:
        return 33792
    if tokens >= 8192:
        return 32768
    return max(_round_group(tokens + 257), GROUP_SIZE)


def _assignments(tokens: int, *, padded_tokens: int, mode: str, seed: int) -> torch.Tensor:
    torch.manual_seed(seed + tokens)
    if mode == "uniform":
        ids = torch.arange(padded_tokens, device="cuda", dtype=torch.long) % CENTROIDS
        out = ids.view(1, 1, padded_tokens).expand(1, NH_KV, padded_tokens).contiguous()
    elif mode == "skewed":
        out = torch.randint(1, CENTROIDS, (1, NH_KV, padded_tokens), device="cuda", dtype=torch.long)
        out[torch.rand(1, NH_KV, padded_tokens, device="cuda") < 0.75] = 0
    elif mode == "all_same":
        out = torch.zeros(1, NH_KV, padded_tokens, device="cuda", dtype=torch.long)
    else:
        out = torch.randint(0, CENTROIDS, (1, NH_KV, padded_tokens), device="cuda", dtype=torch.long)
    if padded_tokens > tokens:
        out[:, :, tokens:] = 0
    return out


def _build_case(tokens: int, *, capacity: int | None = None, assignment: str = "normal", seed: int = 515):
    torch.manual_seed(seed + tokens)
    padded_tokens = _round_group(tokens)
    capacity = _round_group(capacity or _capacity_for(tokens))
    assert capacity >= padded_tokens
    key = (torch.randn(1, NH_KV, padded_tokens, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.25).contiguous()
    query = (torch.randn(1, NH, 1, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.25).contiguous()
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    assignments = _assignments(tokens, padded_tokens=padded_tokens, mode=assignment, seed=seed)
    base = pattern_gather_centroids(assignments, centroids).to(key.dtype)
    packed, scale, zero = quantize_pack_k_reference(key - base, GROUP_SIZE, BITS)

    pack = 32 // BITS
    logical_packs = _ceil_div(tokens, pack)
    logical_groups = _ceil_div(tokens, GROUP_SIZE)
    cap_packs = _ceil_div(capacity, pack)
    cap_groups = _ceil_div(capacity, GROUP_SIZE)

    packed_cap = torch.empty(1, NH_KV, HEAD_DIM, cap_packs, device="cuda", dtype=torch.int32)
    packed_cap.fill_(0x7FFFFFFF)
    packed_cap[:, :, :, : packed.shape[-1]] = packed
    scale_cap = torch.empty(1, NH_KV, HEAD_DIM, cap_groups, device="cuda", dtype=torch.float16)
    scale_cap.fill_(float("nan"))
    scale_cap[:, :, :, : scale.shape[-1]] = scale
    zero_cap = torch.empty_like(scale_cap)
    zero_cap.fill_(float("nan"))
    zero_cap[:, :, :, : zero.shape[-1]] = zero
    assignments_cap = torch.empty(1, NH_KV, capacity, device="cuda", dtype=torch.long)
    assignments_cap.fill_(2**62)
    assignments_cap[:, :, :padded_tokens] = assignments

    return {
        "tokens": int(tokens),
        "capacity": int(capacity),
        "padded_tokens": int(padded_tokens),
        "query": query,
        "packed": packed,
        "scale": scale,
        "zero": zero,
        "centroids": centroids,
        "assignments": assignments,
        "packed_cap": packed_cap,
        "scale_cap": scale_cap,
        "zero_cap": zero_cap,
        "assignments_cap": assignments_cap,
        "packed_view": packed_cap[:, :, :, :logical_packs],
        "scale_view": scale_cap[:, :, :, :logical_groups],
        "zero_view": zero_cap[:, :, :, :logical_groups],
        "assignments_view": assignments_cap[:, :, :tokens],
    }


def _tight(data):
    out = cuda_bmm_fA_qB_outer_with_base(
        GROUP_SIZE,
        data["query"],
        data["packed"],
        data["scale"],
        data["zero"],
        BITS,
        data["centroids"],
        data["assignments"],
        NH,
        NH_KV,
    )
    return out[:, :, :, : data["tokens"]]


def _strided(data):
    return cuda_bmm_fA_qB_outer_with_base_strided_k(
        GROUP_SIZE,
        data["query"],
        data["packed_view"],
        data["scale_view"],
        data["zero_view"],
        BITS,
        data["centroids"],
        data["assignments_view"],
        NH,
        NH_KV,
    )


def _assert_close(candidate: torch.Tensor, baseline: torch.Tensor):
    torch.cuda.synchronize()
    diff = (candidate.float() - baseline.float()).abs()
    rel_l2 = torch.linalg.vector_norm((candidate.float() - baseline.float()).flatten()) / torch.linalg.vector_norm(baseline.float().flatten()).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(candidate.float().flatten(), baseline.float().flatten(), dim=0)
    assert int(torch.isnan(candidate).sum().item()) == 0
    assert int(torch.isinf(candidate).sum().item()) == 0
    assert float(diff.max().item()) <= 5e-3
    assert float(rel_l2.item()) <= 5e-3
    assert float(cosine.item()) >= 0.9999
    torch.testing.assert_close(candidate, baseline, rtol=5e-3, atol=5e-3)


def test_strided_k_one_capacity_matches_tight():
    data = _build_case(128, capacity=128)
    _assert_close(_strided(data), _tight(data))


def test_strided_k_slack_capacity_matches_tight():
    data = _build_case(512, capacity=1024)
    assert not data["packed_view"].is_contiguous()
    _assert_close(_strided(data), _tight(data))


@pytest.mark.parametrize("tokens", [127, 128, 129])
def test_strided_k_127_128_129(tokens):
    data = _build_case(tokens)
    _assert_close(_strided(data), _tight(data))


@pytest.mark.parametrize("tokens", [255, 256, 257])
def test_strided_k_255_256_257(tokens):
    data = _build_case(tokens)
    _assert_close(_strided(data), _tight(data))


def test_strided_k_2k():
    data = _build_case(2048, capacity=4096)
    _assert_close(_strided(data), _tight(data))


def test_strided_k_8k():
    data = _build_case(8192, capacity=32768)
    _assert_close(_strided(data), _tight(data))


def test_strided_k_16k():
    data = _build_case(16384, capacity=32768)
    _assert_close(_strided(data), _tight(data))


def test_strided_k_32k():
    data = _build_case(32768, capacity=33792)
    _assert_close(_strided(data), _tight(data))


def test_strided_k_normal_assignment():
    data = _build_case(512, assignment="normal")
    _assert_close(_strided(data), _tight(data))


def test_strided_k_uniform_assignment():
    data = _build_case(512, assignment="uniform")
    _assert_close(_strided(data), _tight(data))


def test_strided_k_skewed_assignment():
    data = _build_case(512, assignment="skewed")
    _assert_close(_strided(data), _tight(data))


def test_strided_k_all_same_assignment():
    data = _build_case(512, assignment="all_same")
    _assert_close(_strided(data), _tight(data))


def test_strided_k_unused_capacity_not_read():
    data = _build_case(129, capacity=512, assignment="uniform")
    _assert_close(_strided(data), _tight(data))


def test_strided_k_zero_materialization():
    reset_patternkv_strided_k_reader_counters()
    data = _build_case(256)
    _strided(data)
    counters = get_patternkv_strided_k_reader_counters()
    assert counters["strided_k_reader_calls"] == 1
    assert counters["strided_k_materialize_calls"] == 0
    assert counters["strided_k_materialized_bytes"] == 0
    assert counters["strided_k_torch_cat_calls"] == 0


def test_strided_k_iterates_logical_tokens_only():
    reset_patternkv_strided_k_reader_counters()
    data = _build_case(129, capacity=4096)
    _strided(data)
    assert get_patternkv_strided_k_reader_counters()["strided_k_tokens_processed"] == 129


def test_strided_k_preserves_quantization_semantics():
    data = _build_case(512)
    torch.testing.assert_close(data["packed_view"][:, :, :, : data["packed"].shape[-1]], data["packed"], rtol=0, atol=0)
    torch.testing.assert_close(data["scale_view"][:, :, :, : data["scale"].shape[-1]], data["scale"], rtol=0, atol=0)
    torch.testing.assert_close(data["zero_view"][:, :, :, : data["zero"].shape[-1]], data["zero"], rtol=0, atol=0)
    _assert_close(_strided(data), _tight(data))


def test_strided_k_preserves_centroid_semantics():
    data = _build_case(512, assignment="uniform")
    restored = dequantize_k_reference(data["packed"], data["scale"], data["zero"], GROUP_SIZE, BITS)
    restored = restored + pattern_gather_centroids(data["assignments"], data["centroids"]).to(restored.dtype)
    ref = torch.matmul(data["query"], restored[:, :, : data["tokens"], :].repeat_interleave(NH // NH_KV, dim=1).transpose(2, 3))
    torch.testing.assert_close(_strided(data).float(), ref.float(), rtol=5e-2, atol=5e-2)


def test_strided_k_reader_not_default():
    reset_patternkv_strided_k_reader_counters()
    data = _build_case(128)
    _tight(data)
    assert get_patternkv_strided_k_reader_counters()["strided_k_reader_calls"] == 0


def test_v_capacity_integration_unchanged(monkeypatch):
    monkeypatch.delenv("PATTERNKV_CACHE_GROWTH_BACKEND", raising=False)
    assert patternkv_cache_growth_backend() == "baseline"
    reset_patternkv_strided_v2_reader_counters()
    assert callable(cuda_attn_v_fused_with_base_strided_v2)
    assert get_patternkv_strided_v2_reader_counters()["strided_reader_calls"] == 0


def test_page_native_reader_remains_off(monkeypatch):
    monkeypatch.delenv("PATTERNKV_PAGE_V_READER", raising=False)
    assert patternkv_page_v_reader_backend() == "contiguous"


def test_gqa_experimental_remains_off(monkeypatch):
    monkeypatch.delenv("PATTERNKV_GQA_V_BACKEND", raising=False)
    assert patternkv_gqa_v_backend() == "baseline"
