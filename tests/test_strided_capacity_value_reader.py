import pytest
import torch

from models.segmented_cache import quantize_pack_v_reference
from quant.matmul import (
    cuda_attn_v_fused_with_base,
    cuda_attn_v_fused_with_base_debug,
    cuda_attn_v_fused_with_base_strided_v2,
    get_patternkv_strided_v2_reader_counters,
    patternkv_gqa_v_backend,
    patternkv_page_v_reader_backend,
    reset_patternkv_strided_v2_reader_counters,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA strided Value reader tests require a GPU")

GROUP_SIZE = 128
NH = 32
NH_KV = 8
HEAD_DIM = 128
CENTROIDS = 16


def _capacity_for_tokens(tokens: int) -> int:
    if tokens >= 32768:
        return 33792
    if tokens >= 16384:
        return 32768
    if tokens >= 8192:
        return 32768
    return max(tokens + 131, 512)


def _build_case(tokens: int, *, capacity: int | None = None, seed: int = 721, assignment: str = "normal", mask_density: float = 0.5):
    device = torch.device("cuda")
    torch.manual_seed(seed + tokens)
    capacity = int(capacity or _capacity_for_tokens(tokens))
    assert capacity >= tokens
    values = (torch.randn(1, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16) * 0.25).contiguous()
    vq, scale, zero = quantize_pack_v_reference(values, GROUP_SIZE, 2)
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1).contiguous()
    if assignment == "uniform":
        ids = torch.arange(tokens, device=device, dtype=torch.int32) % CENTROIDS
        idx = ids.view(1, 1, tokens).expand(1, NH_KV, tokens).contiguous()
    elif assignment == "skewed":
        idx = torch.randint(1, CENTROIDS, (1, NH_KV, tokens), device=device, dtype=torch.int32)
        idx[torch.rand(1, NH_KV, tokens, device=device) < 0.75] = 0
    elif assignment == "all_same":
        idx = torch.zeros(1, NH_KV, tokens, device=device, dtype=torch.int32)
    else:
        idx = torch.randint(0, CENTROIDS, (1, NH_KV, tokens), device=device, dtype=torch.int32)
    if mask_density <= 0:
        mask = torch.zeros(1, NH_KV, tokens, device=device, dtype=torch.uint8)
    elif mask_density >= 1:
        mask = torch.ones(1, NH_KV, tokens, device=device, dtype=torch.uint8)
    else:
        mask = (torch.rand(1, NH_KV, tokens, device=device) < mask_density).to(torch.uint8)
    attn = torch.softmax(torch.randn(1, NH, 1, tokens, device=device, dtype=torch.float16), dim=-1).contiguous()

    vq_cap = torch.empty(1, NH_KV, capacity, HEAD_DIM // 16, device=device, dtype=torch.int32)
    vq_cap.fill_(0x7FFFFFFF)
    vq_cap[:, :, :tokens, :] = vq
    scale_cap = torch.empty(1, NH_KV, capacity, HEAD_DIM // GROUP_SIZE, device=device, dtype=torch.float16)
    scale_cap.fill_(float("nan"))
    scale_cap[:, :, :tokens, :] = scale
    zero_cap = torch.empty_like(scale_cap)
    zero_cap.fill_(float("nan"))
    zero_cap[:, :, :tokens, :] = zero
    mask_cap = torch.empty(1, NH_KV, capacity, device=device, dtype=torch.uint8)
    mask_cap.fill_(255)
    mask_cap[:, :, :tokens] = mask
    idx_cap = torch.empty(1, NH_KV, capacity, device=device, dtype=torch.int32)
    idx_cap.fill_(2147483647)
    idx_cap[:, :, :tokens] = idx

    return {
        "attn": attn,
        "vq": vq,
        "scale": scale,
        "zero": zero,
        "centroids": centroids,
        "mask": mask,
        "idx": idx,
        "vq_cap": vq_cap,
        "scale_cap": scale_cap,
        "zero_cap": zero_cap,
        "mask_cap": mask_cap,
        "idx_cap": idx_cap,
        "tokens": tokens,
        "capacity": capacity,
    }


def _baseline_output(data):
    return cuda_attn_v_fused_with_base(
        GROUP_SIZE,
        data["attn"],
        data["vq"],
        data["scale"],
        data["zero"],
        2,
        data["centroids"],
        data["mask"],
        data["idx"],
        NH,
        NH_KV,
    )


def _strided_output(data):
    T = data["tokens"]
    return cuda_attn_v_fused_with_base_strided_v2(
        GROUP_SIZE,
        data["attn"],
        data["vq_cap"][:, :, :T, :],
        data["scale_cap"][:, :, :T, :],
        data["zero_cap"][:, :, :T, :],
        data["centroids"],
        data["mask_cap"][:, :, :T],
        data["idx_cap"][:, :, :T],
        NH,
        NH_KV,
    )


def _assert_close(candidate: torch.Tensor, baseline: torch.Tensor):
    torch.cuda.synchronize()
    diff = (candidate.float() - baseline.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(candidate.float().flatten(), baseline.float().flatten(), dim=0)
    assert int(torch.isnan(candidate).sum().item()) == 0
    assert int(torch.isinf(candidate).sum().item()) == 0
    assert float(diff.max().item()) <= 5e-3
    assert float(cosine.item()) >= 0.9999
    torch.testing.assert_close(candidate, baseline, rtol=5e-3, atol=5e-3)


def test_strided_v2_one_capacity_matches_contiguous():
    data = _build_case(128, capacity=128)
    _assert_close(_strided_output(data), _baseline_output(data))


def test_strided_v2_slack_capacity_matches_contiguous():
    data = _build_case(512, capacity=1024)
    assert data["vq_cap"][:, :, : data["tokens"], :].stride(2) == data["vq_cap"].stride(2)
    _assert_close(_strided_output(data), _baseline_output(data))


@pytest.mark.parametrize("tokens", [127, 128, 129])
def test_strided_v2_127_128_129(tokens):
    data = _build_case(tokens)
    _assert_close(_strided_output(data), _baseline_output(data))


@pytest.mark.parametrize("tokens", [255, 256, 257])
def test_strided_v2_255_256_257(tokens):
    data = _build_case(tokens)
    _assert_close(_strided_output(data), _baseline_output(data))


def test_strided_v2_8k():
    data = _build_case(8192, capacity=32768)
    _assert_close(_strided_output(data), _baseline_output(data))


def test_strided_v2_16k():
    data = _build_case(16384, capacity=32768)
    _assert_close(_strided_output(data), _baseline_output(data))


def test_strided_v2_32k():
    data = _build_case(32768, capacity=33792)
    _assert_close(_strided_output(data), _baseline_output(data))


def test_strided_v2_uniform_assignment():
    data = _build_case(512, assignment="uniform", mask_density=1.0)
    _assert_close(_strided_output(data), _baseline_output(data))


def test_strided_v2_skewed_assignment():
    data = _build_case(512, assignment="skewed", mask_density=1.0)
    _assert_close(_strided_output(data), _baseline_output(data))


def test_strided_v2_all_same_assignment():
    data = _build_case(512, assignment="all_same", mask_density=1.0)
    _assert_close(_strided_output(data), _baseline_output(data))


def test_strided_v2_mask_zero():
    data = _build_case(512, mask_density=0.0)
    _assert_close(_strided_output(data), _baseline_output(data))


def test_strided_v2_mask_full():
    data = _build_case(512, assignment="uniform", mask_density=1.0)
    _assert_close(_strided_output(data), _baseline_output(data))


def test_strided_v2_unused_capacity_not_read():
    data = _build_case(129, capacity=512, assignment="uniform", mask_density=1.0)
    _assert_close(_strided_output(data), _baseline_output(data))


def test_strided_v2_zero_materialization():
    reset_patternkv_strided_v2_reader_counters()
    data = _build_case(256)
    _strided_output(data)
    counters = get_patternkv_strided_v2_reader_counters()
    assert counters["strided_reader_calls"] == 1
    assert counters["strided_reader_materialize_calls"] == 0
    assert counters["strided_reader_materialized_bytes"] == 0
    assert counters["strided_reader_torch_cat_calls"] == 0


def test_strided_v2_preserves_per_warp_histogram():
    data = _build_case(256)
    production = _baseline_output(data)
    lane0 = cuda_attn_v_fused_with_base_debug(
        GROUP_SIZE,
        data["attn"],
        data["vq"],
        data["scale"],
        data["zero"],
        2,
        data["centroids"],
        data["mask"],
        data["idx"],
        NH,
        NH_KV,
        debug_mode="LANE0_TABLE_FULL",
    )
    _assert_close(production, lane0)


def test_strided_v2_preserves_lane0_centroid():
    data = _build_case(256, mask_density=1.0)
    _assert_close(_strided_output(data), _baseline_output(data))


def test_strided_reader_not_default(monkeypatch):
    monkeypatch.delenv("PATTERNKV_STRIDED_V2_READER", raising=False)
    data = _build_case(128)
    reset_patternkv_strided_v2_reader_counters()
    _baseline_output(data)
    assert get_patternkv_strided_v2_reader_counters()["strided_reader_calls"] == 0


def test_page_native_reader_still_not_default(monkeypatch):
    monkeypatch.delenv("PATTERNKV_PAGE_V_READER", raising=False)
    assert patternkv_page_v_reader_backend() == "contiguous"


def test_gqa_experimental_still_not_default(monkeypatch):
    monkeypatch.delenv("PATTERNKV_GQA_V_BACKEND", raising=False)
    assert patternkv_gqa_v_backend() == "baseline"
