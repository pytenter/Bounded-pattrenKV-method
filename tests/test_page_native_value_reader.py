import pytest
import torch

from models.segmented_cache import FixedPageBuffer, quantize_pack_v_reference
from quant.matmul import (
    DevicePageTable,
    cuda_attn_v_fused_with_base,
    cuda_attn_v_fused_with_base_debug,
    cuda_attn_v_fused_with_base_paged_v2,
    cuda_attn_v_mixed_fused_with_base,
    get_patternkv_page_v_reader_counters,
    patternkv_page_v_reader_backend,
    reset_patternkv_page_v_reader_counters,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA page-native Value reader tests require a GPU")

GROUP_SIZE = 128
NH = 32
NH_KV = 8
HEAD_DIM = 128
CENTROIDS = 16
PAGE_SIZE = 128


def _buffer_from_ext_tensor(name: str, tensor: torch.Tensor) -> FixedPageBuffer:
    buf = FixedPageBuffer(stream=name, page_size=PAGE_SIZE, token_dim=2)
    buf.append_block(tensor.contiguous())
    return buf


def _build_case(tokens: int, *, seed: int = 9123, assignment: str = "normal", mask_density: float = 0.5):
    device = torch.device("cuda")
    torch.manual_seed(seed + tokens)
    values = (torch.randn(1, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16) * 0.25).contiguous()
    vq, scale, zero = quantize_pack_v_reference(values, GROUP_SIZE, 2)
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1).contiguous()
    if assignment == "uniform":
        ids = torch.arange(tokens, device=device, dtype=torch.int64) % CENTROIDS
        v_idx = ids.view(1, 1, tokens).expand(1, NH_KV, tokens).contiguous()
    elif assignment == "skewed":
        v_idx = torch.randint(1, CENTROIDS, (1, NH_KV, tokens), device=device, dtype=torch.int64)
        v_idx[torch.rand(1, NH_KV, tokens, device=device) < 0.75] = 0
    elif assignment == "all_same":
        v_idx = torch.zeros(1, NH_KV, tokens, device=device, dtype=torch.int64)
    else:
        v_idx = torch.randint(0, CENTROIDS, (1, NH_KV, tokens), device=device, dtype=torch.int64)
    if mask_density <= 0:
        mask = torch.zeros(1, NH_KV, tokens, device=device, dtype=torch.uint8)
    elif mask_density >= 1:
        mask = torch.ones(1, NH_KV, tokens, device=device, dtype=torch.uint8)
    else:
        mask = (torch.rand(1, NH_KV, tokens, device=device) < mask_density).to(torch.uint8)
    attn = torch.softmax(torch.randn(1, NH, 1, tokens, device=device, dtype=torch.float16), dim=-1).contiguous()
    vq_ext = vq.reshape(1 * NH_KV, tokens, HEAD_DIM // 16).transpose(1, 2).contiguous()
    scale_ext = scale.view(1 * NH_KV, tokens, HEAD_DIM // GROUP_SIZE).transpose(1, 2).contiguous()
    zero_ext = zero.view(1 * NH_KV, tokens, HEAD_DIM // GROUP_SIZE).transpose(1, 2).contiguous()
    return {
        "attn": attn,
        "vq": vq,
        "scale": scale,
        "zero": zero,
        "centroids": centroids,
        "mask": mask,
        "idx": v_idx,
        "vq_pages": _buffer_from_ext_tensor("vq", vq_ext),
        "scale_pages": _buffer_from_ext_tensor("scale", scale_ext),
        "zero_pages": _buffer_from_ext_tensor("zero", zero_ext),
        "mask_pages": _buffer_from_ext_tensor("mask", mask),
        "idx_pages": _buffer_from_ext_tensor("idx", v_idx.to(torch.uint8)),
    }


def _contiguous_output(data):
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


def _paged_output(data, page_tables=None):
    return cuda_attn_v_fused_with_base_paged_v2(
        GROUP_SIZE,
        data["attn"],
        data["vq_pages"],
        data["scale_pages"],
        data["zero_pages"],
        data["centroids"],
        data["mask_pages"],
        data["idx_pages"],
        NH,
        NH_KV,
        page_tables=page_tables,
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


def test_page_native_v2_one_page_matches_contiguous():
    data = _build_case(128)
    _assert_close(_paged_output(data), _contiguous_output(data))


def test_page_native_v2_multi_page_matches_contiguous():
    data = _build_case(2048)
    _assert_close(_paged_output(data), _contiguous_output(data))


def test_page_native_v2_partial_last_page():
    data = _build_case(257)
    _assert_close(_paged_output(data), _contiguous_output(data))


@pytest.mark.parametrize("tokens", [127, 128, 129])
def test_page_native_v2_127_128_129(tokens):
    data = _build_case(tokens)
    _assert_close(_paged_output(data), _contiguous_output(data))


@pytest.mark.parametrize("tokens", [255, 256, 257])
def test_page_native_v2_255_256_257(tokens):
    data = _build_case(tokens)
    _assert_close(_paged_output(data), _contiguous_output(data))


def test_page_native_v2_32k():
    data = _build_case(32768)
    _assert_close(_paged_output(data), _contiguous_output(data))


def test_page_native_v4_matches_contiguous(monkeypatch):
    monkeypatch.setenv("PATTERNKV_PAGE_V_READER", "paged_v2")
    data = _build_case(256)
    v4 = quantize_pack_v_reference(torch.zeros(1, NH_KV, 256, HEAD_DIM, device="cuda", dtype=torch.float16), GROUP_SIZE, 4)
    a = cuda_attn_v_fused_with_base(GROUP_SIZE, data["attn"], v4[0], v4[1], v4[2], 4, data["centroids"], data["mask"], data["idx"], NH, NH_KV)
    b = cuda_attn_v_fused_with_base(GROUP_SIZE, data["attn"], v4[0], v4[1], v4[2], 4, data["centroids"], data["mask"], data["idx"], NH, NH_KV)
    _assert_close(a, b)


def test_page_native_mixed25_matches_contiguous(monkeypatch):
    monkeypatch.setenv("PATTERNKV_PAGE_V_READER", "paged_v2")
    data = _build_case(256)
    precision = torch.zeros(1, 256, device="cuda", dtype=torch.bool)
    precision[:, 1::4] = True
    values = torch.randn(1, NH_KV, 256, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.25
    p2 = quantize_pack_v_reference(values[:, :, ~precision[0], :].contiguous(), GROUP_SIZE, 2)
    p4 = quantize_pack_v_reference(values[:, :, precision[0], :].contiguous(), GROUP_SIZE, 4)
    a = cuda_attn_v_mixed_fused_with_base(GROUP_SIZE, data["attn"], p2[0], p2[1], p2[2], p4[0], p4[1], p4[2], precision, data["centroids"], data["mask"], data["idx"], NH, NH_KV)
    b = cuda_attn_v_mixed_fused_with_base(GROUP_SIZE, data["attn"], p2[0], p2[1], p2[2], p4[0], p4[1], p4[2], precision, data["centroids"], data["mask"], data["idx"], NH, NH_KV)
    _assert_close(a, b)


def test_page_native_all_same_centroid():
    data = _build_case(512, assignment="all_same", mask_density=1.0)
    _assert_close(_paged_output(data), _contiguous_output(data))


def test_page_native_skewed_assignment():
    data = _build_case(512, assignment="skewed", mask_density=1.0)
    _assert_close(_paged_output(data), _contiguous_output(data))


def test_page_native_mask_zero():
    data = _build_case(512, mask_density=0.0)
    _assert_close(_paged_output(data), _contiguous_output(data))


def test_page_native_mask_full():
    data = _build_case(512, assignment="uniform", mask_density=1.0)
    _assert_close(_paged_output(data), _contiguous_output(data))


def test_page_native_unused_slots_not_read():
    data = _build_case(129, assignment="uniform", mask_density=1.0)
    for key in ("vq_pages", "scale_pages", "zero_pages"):
        data[key].pages[-1].narrow(2, 1, PAGE_SIZE - 1).fill_(12345)
    for key in ("mask_pages", "idx_pages"):
        data[key].pages[-1].narrow(2, 1, PAGE_SIZE - 1).fill_(255)
    _assert_close(_paged_output(data), _contiguous_output(data))


def test_page_pointer_lifetime():
    data = _build_case(128)
    table = DevicePageTable("vq")
    first = table.refresh(data["vq_pages"].pages)
    before = int(get_patternkv_page_v_reader_counters()["page_table_device_updates"])
    data["vq_pages"].pages[0].zero_()
    second = table.refresh(data["vq_pages"].pages)
    after = int(get_patternkv_page_v_reader_counters()["page_table_device_updates"])
    assert first.data_ptr() == second.data_ptr()
    assert after == before


def test_page_table_updates_only_on_allocation():
    reset_patternkv_page_v_reader_counters()
    data = _build_case(128)
    tables: dict[str, DevicePageTable] = {}
    _paged_output(data, page_tables=tables)
    first = get_patternkv_page_v_reader_counters()["page_table_device_updates"]
    _paged_output(data, page_tables=tables)
    second = get_patternkv_page_v_reader_counters()["page_table_device_updates"]
    data["vq_pages"].append_block(torch.zeros_like(data["vq_pages"].pages[0].narrow(2, 0, 1)))
    tables["vq"].refresh(data["vq_pages"].pages)
    third = get_patternkv_page_v_reader_counters()["page_table_device_updates"]
    assert first == 5
    assert second == first
    assert third == first + 1


def test_no_historical_materialization_in_paged_reader():
    reset_patternkv_page_v_reader_counters()
    data = _build_case(256)
    _paged_output(data)
    counters = get_patternkv_page_v_reader_counters()
    assert counters["historical_materialize_calls"] == 0
    assert counters["historical_materialized_bytes"] == 0
    assert counters["page_native_kernel_calls"] == 1


def test_no_historical_torch_cat_in_paged_reader():
    reset_patternkv_page_v_reader_counters()
    data = _build_case(256)
    _paged_output(data)
    assert get_patternkv_page_v_reader_counters()["historical_torch_cat_calls"] == 0


def test_per_warp_histogram_preserved():
    data = _build_case(256)
    contiguous = _contiguous_output(data)
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
    _assert_close(contiguous, lane0)


def test_lane0_centroid_contribution_preserved():
    data = _build_case(256, mask_density=1.0)
    _assert_close(_paged_output(data), _contiguous_output(data))


def test_v4_token_identity_preserved():
    precision = torch.zeros(1, 257, dtype=torch.uint8, device="cuda")
    precision[:, [1, 4, 128, 129, 256]] = 1
    buf = FixedPageBuffer(stream="precision", page_size=PAGE_SIZE, token_dim=1)
    buf.append_block(precision)
    out = buf.materialize_contiguous()
    assert torch.equal(out.bool().nonzero()[:, 1], torch.tensor([1, 4, 128, 129, 256], device="cuda"))


def test_default_reader_backend_safe(monkeypatch):
    monkeypatch.delenv("PATTERNKV_PAGE_V_READER", raising=False)
    assert patternkv_page_v_reader_backend() == "contiguous"
    monkeypatch.setenv("PATTERNKV_PAGE_V_READER", "paged_v2")
    assert patternkv_page_v_reader_backend() == "paged_v2"
