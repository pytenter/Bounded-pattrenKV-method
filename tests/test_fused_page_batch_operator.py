import pytest
import torch

from quant.page_batch import (
    PAGE_SIZE,
    build_operator_ready_page_pools,
    correctness_metrics,
    get_patternkv_page_batch_counters,
    pack_mixed_v_pages,
    patternkv_fused_page_batch_decode,
    patternkv_page_batched_v_decode,
    reset_patternkv_page_batch_counters,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused page batch operator tests require a GPU")

GROUP_SIZE = 128
NH = 32
NH_KV = 8
HEAD_DIM = 128
CENTROIDS = 16


def _precision_masks(batch: int, tokens: int, *, device: torch.device) -> torch.Tensor:
    rows = []
    for b in range(batch):
        mask = torch.zeros(tokens, dtype=torch.bool, device=device)
        if b % 4 == 0:
            mask[1::4] = True
        elif b % 4 == 1:
            width = max(1, tokens // 4)
            start = min(tokens // 3, tokens - width)
            mask[start : start + width] = True
        elif b % 4 == 2:
            mask[: max(1, tokens // 4)] = True
        else:
            mask[-max(1, tokens // 4) :] = True
        rows.append(mask)
    return torch.stack(rows, dim=0).contiguous()


def _build_cache(batch: int, tokens: int, *, seed: int = 20260813):
    device = torch.device("cuda")
    torch.manual_seed(seed + batch * 100_000 + tokens)
    v_adjusted = (torch.randn(batch, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16) * 0.25).contiguous()
    precision = _precision_masks(batch, tokens, device=device)
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1).contiguous()
    v_idx = torch.randint(0, CENTROIDS, (batch, NH_KV, tokens), device=device, dtype=torch.int64)
    v_pattern_mask = (torch.rand(batch, NH_KV, tokens, device=device) > 0.55).to(torch.uint8)
    attn = torch.softmax(torch.randn(batch, NH, 1, tokens, device=device, dtype=torch.float16), dim=-1).contiguous()
    cache = pack_mixed_v_pages(v_adjusted, precision, v_pattern_mask, v_idx, centroids, group_size=GROUP_SIZE, nh=NH)
    return attn, cache


def _assert_flat_pool_matches_pages(pool: torch.Tensor, offsets: torch.Tensor, pages: list[torch.Tensor | None]) -> None:
    for page_id, page in enumerate(pages):
        offset = int(offsets[page_id].item())
        if page is None:
            assert offset == -1
            continue
        count = int(page.shape[2])
        assert offset >= 0
        torch.testing.assert_close(pool[:, offset : offset + count], page.squeeze(0))


@pytest.mark.parametrize("batch,tokens", [(1, 513), (2, 2051), (4, 4096), (8, 512)])
def test_operator_ready_page_pools_preserve_page_layout(batch, tokens):
    _attn, cache = _build_cache(batch, tokens)
    pools = build_operator_ready_page_pools(cache)

    _assert_flat_pool_matches_pages(pools.v2_payload_pool, pools.v2_page_offsets, cache.v2_payload)
    _assert_flat_pool_matches_pages(pools.v4_payload_pool, pools.v4_page_offsets, cache.v4_payload)
    _assert_flat_pool_matches_pages(pools.v2_scale_pool, pools.v2_page_offsets, cache.v2_scale)
    _assert_flat_pool_matches_pages(pools.v2_zero_pool, pools.v2_page_offsets, cache.v2_zero)
    _assert_flat_pool_matches_pages(pools.v4_scale_pool, pools.v4_page_offsets, cache.v4_scale)
    _assert_flat_pool_matches_pages(pools.v4_zero_pool, pools.v4_page_offsets, cache.v4_zero)
    _assert_flat_pool_matches_pages(pools.v2_pattern_pool, pools.v2_page_offsets, cache.v2_pattern_mask)
    _assert_flat_pool_matches_pages(pools.v4_pattern_pool, pools.v4_page_offsets, cache.v4_pattern_mask)
    _assert_flat_pool_matches_pages(pools.v2_assignment_pool, pools.v2_page_offsets, cache.v2_assignment_idx)
    _assert_flat_pool_matches_pages(pools.v4_assignment_pool, pools.v4_page_offsets, cache.v4_assignment_idx)

    assert pools.metadata is cache.metadata
    assert pools.centroids.data_ptr() == cache.centroids.data_ptr()
    assert pools.page_value_materialized_bytes == 0
    assert pools.historical_materialized_bytes == 0
    assert pools.gpu_tensor_item_calls == 0


@pytest.mark.parametrize("batch,tokens", [(1, 512), (2, 2048), (4, 4096), (2, 2051)])
def test_fused_page_batch_operator_matches_page_torch_mvp(batch, tokens):
    attn, cache = _build_cache(batch, tokens)
    pools = build_operator_ready_page_pools(cache)

    expected = patternkv_page_batched_v_decode(attn, cache)
    reset_patternkv_page_batch_counters()
    got = patternkv_fused_page_batch_decode(attn, pools)
    torch.cuda.synchronize()
    counters = get_patternkv_page_batch_counters()
    metrics = correctness_metrics(got, expected)

    assert metrics["nan"] == 0
    assert metrics["inf"] == 0
    assert metrics["relative_l2"] <= 2e-3, metrics
    assert metrics["cosine"] >= 0.9999, metrics
    torch.testing.assert_close(got, expected, rtol=6e-3, atol=6e-3)
    assert counters["page_value_materialized_bytes"] == 0
    assert counters["gpu_tensor_item_calls"] == 0
    assert counters["matmul_calls"] == 0
    assert cache.page_size == PAGE_SIZE


def test_fused_page_batch_operator_uses_request_local_seq_lens() -> None:
    device = torch.device("cuda")
    torch.manual_seed(2026081401)
    tokens = 256
    valid = 128
    v_adjusted = (torch.randn(2, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16) * 0.25).contiguous()
    precision = _precision_masks(2, tokens, device=device)
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1).contiguous()
    v_idx = torch.randint(0, CENTROIDS, (2, NH_KV, tokens), device=device, dtype=torch.int64)
    v_pattern_mask = (torch.rand(2, NH_KV, tokens, device=device) > 0.55).to(torch.uint8)
    attn = torch.softmax(torch.randn(2, NH, 1, tokens, device=device, dtype=torch.float16), dim=-1).contiguous()
    attn[0, :, :, valid:] = 1.0
    ragged_cache = pack_mixed_v_pages(v_adjusted, precision, v_pattern_mask, v_idx, centroids, group_size=GROUP_SIZE, nh=NH)
    ragged_cache.metadata.seq_lens[0] = valid
    ref_cache = pack_mixed_v_pages(
        v_adjusted[0:1, :, :valid, :].contiguous(),
        precision[0:1, :valid].contiguous(),
        v_pattern_mask[0:1, :, :valid].contiguous(),
        v_idx[0:1, :, :valid].contiguous(),
        centroids,
        group_size=GROUP_SIZE,
        nh=NH,
    )
    got = patternkv_fused_page_batch_decode(attn, build_operator_ready_page_pools(ragged_cache))[0:1]
    ref = patternkv_fused_page_batch_decode(attn[0:1, :, :, :valid].contiguous(), build_operator_ready_page_pools(ref_cache))
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
