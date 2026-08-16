from __future__ import annotations

import pytest
import torch

from models.segmented_cache import (
    build_cache_from_prefill,
    dequantize_v_reference,
    pattern_gather_request_centroids,
    quantize_pack_v_reference,
)
from quant.page_batch import (
    build_operator_ready_page_pools,
    get_patternkv_page_batch_counters,
    get_patternkv_real_decode_counters,
    pack_mixed_v_pages,
    patternkv_fused_page_batch_decode,
    reset_patternkv_page_batch_counters,
    reset_patternkv_real_decode_counters,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="PatternKV true-batch runtime tests require CUDA")

GROUP_SIZE = 128
NH = 4
NH_KV = 2
HEAD_DIM = 128
CENTROIDS = 48


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_key_value_heads, tokens, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, tokens, head_dim)
    return hidden_states.reshape(bsz, num_key_value_heads * n_rep, tokens, head_dim)


def _case(batch: int, tokens: int, *, seed: int = 20260816) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed + batch * 1000 + tokens)
    rows = torch.arange(batch, dtype=torch.float32, device=device).view(batch, 1, 1, 1)
    heads = torch.arange(NH_KV, dtype=torch.float32, device=device).view(1, NH_KV, 1, 1)
    dims = torch.arange(HEAD_DIM, dtype=torch.float32, device=device).view(1, 1, 1, HEAD_DIM)
    v_adjusted = (torch.randn(batch, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16, generator=generator) * 0.05 + rows * 0.35 + heads * 0.07).contiguous()
    centroids = (torch.randn(batch, NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16, generator=generator) * 0.03 + rows * 1.5 + dims * 0.0005).contiguous()
    assignment = torch.randint(0, CENTROIDS, (batch, NH_KV, tokens), device=device, dtype=torch.int64, generator=generator)
    pattern_mask = (torch.rand(batch, NH_KV, tokens, device=device, generator=generator) > 0.4).to(torch.uint8)
    precision = torch.zeros(batch, tokens, dtype=torch.bool, device=device)
    attn = torch.softmax(torch.randn(batch, NH, 1, tokens, device=device, dtype=torch.float16, generator=generator), dim=-1).contiguous()
    return {
        "v_adjusted": v_adjusted,
        "centroids": centroids,
        "assignment": assignment,
        "pattern_mask": pattern_mask,
        "precision": precision,
        "attn": attn,
    }


def _reference(case: dict[str, torch.Tensor]) -> torch.Tensor:
    packed, scale, zero = quantize_pack_v_reference(case["v_adjusted"], GROUP_SIZE, 2)
    dequant = dequantize_v_reference(packed, scale, zero, GROUP_SIZE, 2)
    restored = dequant + case["pattern_mask"].unsqueeze(-1).to(dequant.dtype) * pattern_gather_request_centroids(
        case["assignment"], case["centroids"]
    ).to(dequant.dtype)
    return torch.matmul(case["attn"].to(restored.dtype), _repeat_kv(restored, NH // NH_KV))


def _fused_page(case: dict[str, torch.Tensor], *, centroids: torch.Tensor | None = None) -> torch.Tensor:
    page_cache = pack_mixed_v_pages(
        case["v_adjusted"],
        case["precision"],
        case["pattern_mask"],
        case["assignment"].to(torch.int32),
        case["centroids"] if centroids is None else centroids,
        group_size=GROUP_SIZE,
        nh=NH,
    )
    pools = build_operator_ready_page_pools(page_cache)
    return patternkv_fused_page_batch_decode(case["attn"], pools)


def _assert_close_with_metrics(got: torch.Tensor, expected: torch.Tensor) -> None:
    assert int(torch.isnan(got).sum().item()) == 0
    assert int(torch.isinf(got).sum().item()) == 0
    cosine = torch.nn.functional.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0)
    assert float(cosine.item()) >= 0.999
    torch.testing.assert_close(got.float(), expected.float(), rtol=7e-3, atol=7e-3)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_patternkv_paper_all_v2_page_pool_matches_request_local_reference(batch: int) -> None:
    case = _case(batch, 256)
    _assert_close_with_metrics(_fused_page(case), _reference(case))


def test_patternkv_paper_b2_reorder_preserves_request_local_centroids() -> None:
    case = _case(2, 256)
    order = torch.tensor([1, 0], device="cuda")
    reordered = {name: tensor[order].contiguous() for name, tensor in case.items()}
    restored = _fused_page(reordered)[order].contiguous()
    _assert_close_with_metrics(restored, _fused_page(case))


def test_patternkv_paper_request_local_centroid_banks_diverge() -> None:
    case = _case(2, 128)
    assert float((case["centroids"][0].float() - case["centroids"][1].float()).abs().max().item()) > 1.0
    assert not torch.equal(case["assignment"][0], case["assignment"][1])


def test_patternkv_paper_shared_row0_centroids_negative_control_fails() -> None:
    case = _case(2, 128)
    case["v_adjusted"].zero_()
    case["pattern_mask"].fill_(1)
    case["assignment"].zero_()
    correct = _fused_page(case)
    shared_row0 = _fused_page(case, centroids=case["centroids"][0].contiguous())
    assert float((correct[1].float() - shared_row0[1].float()).abs().max().item()) > 0.25
    torch.testing.assert_close(correct[0], shared_row0[0], rtol=7e-3, atol=7e-3)


def test_patternkv_paper_page_pool_has_zero_serial_dispatch_counters() -> None:
    reset_patternkv_page_batch_counters()
    reset_patternkv_real_decode_counters()
    case = _case(4, 256)
    _assert_close_with_metrics(_fused_page(case), _reference(case))
    page_counters = get_patternkv_page_batch_counters()
    real_decode_counters = get_patternkv_real_decode_counters()
    assert page_counters["python_serial_b1_dispatches"] == 0
    assert real_decode_counters["serial_b1_dispatches"] == 0
    assert real_decode_counters["fused_page_operator_calls"] == 1


def test_patternkv_paper_prefill_cache_builds_request_local_page_pools() -> None:
    case = _case(2, 512)
    key = case["v_adjusted"].clone()
    cache = build_cache_from_prefill(
        key,
        case["v_adjusted"],
        sink_length=0,
        recent_length=128,
        group_size=GROUP_SIZE,
        k_bits=2,
        v_bits=2,
        pattern=True,
        k_centroids=case["centroids"],
        v_centroids=case["centroids"],
        k_assignments=case["assignment"],
        v_assignment_idx=case["assignment"],
        v_pattern_mask=case["pattern_mask"],
        cache_mode="segmented_rolling",
        chunk_length=GROUP_SIZE,
        v_precision_selector="base_v2",
        v4_budget_fraction=0.0,
    )
    assert cache.v_centroids is not None and tuple(cache.v_centroids.shape[:2]) == (2, NH_KV)
    assert cache.v_assignment_idx is not None and tuple(cache.v_assignment_idx.shape[:2]) == (2, NH_KV)
    assert cache.operator_ready_page_pools is not None
    assert cache.operator_ready_page_pools.centroids.shape == cache.v_centroids.shape
    assert cache.v_precision_mask is not None and int(cache.v_precision_mask.sum().item()) == 0
