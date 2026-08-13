import pytest
import torch

from quant.page_batch import (
    PAGE_SIZE,
    cache_isolation_summary,
    correctness_metrics,
    get_patternkv_page_batch_counters,
    pack_mixed_v_pages,
    patternkv_page_batched_v_decode,
    reset_patternkv_page_batch_counters,
    selector_isolation_summary,
    validate_page_mapping,
)
from bench.patternkv_page_batch_mvp import reference_batch_mixed_v


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA page batch MVP tests require a GPU")

GROUP_SIZE = 128
NH = 32
NH_KV = 8
HEAD_DIM = 128
CENTROIDS = 16


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, num_key_value_heads * n_rep, slen, head_dim)


def _mask_uniform(tokens: int, *, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    mask[1::4] = True
    return mask


def _mask_clustered(tokens: int, *, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    k = max(1, tokens // 4)
    start = min(tokens // 3, tokens - k)
    mask[start : start + k] = True
    return mask


def _mask_front(tokens: int, *, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    mask[: max(1, tokens // 4)] = True
    return mask


def _mask_back(tokens: int, *, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    mask[-max(1, tokens // 4) :] = True
    return mask


def _precision_masks(batch: int, tokens: int, mode: str, *, device: torch.device) -> torch.Tensor:
    if mode == "same":
        return _mask_uniform(tokens, device=device).view(1, -1).expand(batch, -1).contiguous()
    builders = [_mask_uniform, _mask_clustered, _mask_front, _mask_back]
    rows = [builders[b % len(builders)](tokens, device=device) for b in range(batch)]
    if mode == "strong":
        if batch >= 2:
            rows[1] = _mask_clustered(tokens, device=device)
        if batch >= 3:
            rows[2] = _mask_front(tokens, device=device)
        if batch >= 4:
            rows[3] = _mask_back(tokens, device=device)
    return torch.stack(rows, dim=0).contiguous()


def _build_case(batch: int, tokens: int, mode: str = "different", *, seed: int = 20260813):
    device = torch.device("cuda")
    torch.manual_seed(seed + batch * 100_000 + tokens)
    v_adjusted = (torch.randn(batch, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16) * 0.25).contiguous()
    precision = _precision_masks(batch, tokens, mode, device=device)
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1).contiguous()
    v_idx = torch.randint(0, CENTROIDS, (batch, NH_KV, tokens), device=device, dtype=torch.int64)
    v_pattern_mask = (torch.rand(batch, NH_KV, tokens, device=device) > 0.55).to(torch.uint8)
    attn = torch.softmax(torch.randn(batch, NH, 1, tokens, device=device, dtype=torch.float16), dim=-1).contiguous()
    return attn, v_adjusted, precision, v_pattern_mask, v_idx, centroids


def _page_output(case):
    attn, v_adjusted, precision, v_pattern_mask, v_idx, centroids = case
    cache = pack_mixed_v_pages(v_adjusted, precision, v_pattern_mask, v_idx, centroids, group_size=GROUP_SIZE, nh=NH)
    return patternkv_page_batched_v_decode(attn, cache), cache


def _reference_output(case):
    attn, v_adjusted, precision, v_pattern_mask, v_idx, centroids = case
    return reference_batch_mixed_v(attn, v_adjusted, precision, v_pattern_mask, v_idx, centroids, group_size=GROUP_SIZE, nh=NH, nh_kv=NH_KV)


def _assert_mvp_close(candidate: torch.Tensor, reference: torch.Tensor):
    metrics = correctness_metrics(candidate, reference)
    assert metrics["nan"] == 0
    assert metrics["inf"] == 0
    assert metrics["relative_l2"] <= 1e-3, metrics
    assert metrics["cosine"] >= 0.9999, metrics
    torch.testing.assert_close(candidate, reference, rtol=5e-3, atol=5e-3)


def test_page_batch_b1_matches_production():
    case = _build_case(1, 512, "different")
    candidate, cache = _page_output(case)
    reference = _reference_output(case)
    assert cache.historical_materialized_bytes == 0
    _assert_mvp_close(candidate, reference)


@pytest.mark.parametrize("tokens", [512, 2048, 4096])
def test_page_batch_b2_matches_independent_b1(tokens):
    case = _build_case(2, tokens, "different")
    candidate, _cache = _page_output(case)
    _assert_mvp_close(candidate, _reference_output(case))


@pytest.mark.parametrize("tokens", [512, 2048, 4096])
def test_page_batch_b4_matches_independent_b1(tokens):
    case = _build_case(4, tokens, "strong")
    candidate, _cache = _page_output(case)
    _assert_mvp_close(candidate, _reference_output(case))


def test_page_batch_different_precision_masks():
    case = _build_case(4, 512, "strong")
    _attn, _v_adjusted, precision, _v_pattern_mask, _v_idx, _centroids = case
    assert len({tuple(torch.nonzero(precision[b], as_tuple=False).flatten().detach().cpu().tolist()) for b in range(4)}) == 4
    candidate, _cache = _page_output(case)
    _assert_mvp_close(candidate, _reference_output(case))


def test_page_batch_request_local_v4_counts():
    case = _build_case(4, 512, "strong")
    _candidate, cache = _page_output(case)
    counts_by_request = []
    for b in range(4):
        counts_by_request.append(tuple(int(cache.metadata.v4_counts[int(cache.metadata.metadata_page_table[b, p].item())].item()) for p in range(4)))
    assert len(set(counts_by_request)) > 1
    assert validate_page_mapping(cache)["mapping_valid"]


def test_page_batch_partial_last_page():
    case = _build_case(2, 530, "different")
    candidate, cache = _page_output(case)
    assert int(cache.metadata.valid_tokens.max().item()) == PAGE_SIZE
    assert 0 < int(cache.metadata.valid_tokens.min().item()) < PAGE_SIZE
    _assert_mvp_close(candidate, _reference_output(case))


@pytest.mark.parametrize("batch", [2, 4])
def test_page_batch_cache_isolation(batch):
    case = _build_case(batch, 512, "strong")
    attn, v_adjusted, precision, v_pattern_mask, v_idx, centroids = case
    summary = cache_isolation_summary(attn, v_adjusted, precision, v_pattern_mask, v_idx, centroids, group_size=GROUP_SIZE, nh=NH)
    assert summary["cache_isolation_pass"], summary


def test_page_batch_selector_isolation():
    case = _build_case(4, 512, "strong")
    summary = selector_isolation_summary(case[2])
    assert summary["selector_isolation_pass"], summary


def test_page_batch_pattern_metadata_alignment():
    case = _build_case(2, 512, "strong")
    _attn, _v_adjusted, precision, v_pattern_mask, v_idx, _centroids = case
    _candidate, cache = _page_output(case)
    pages = int(cache.metadata.num_pages[0].item())
    for b in range(2):
        for p in range(pages):
            start = p * PAGE_SIZE
            stop = min(start + PAGE_SIZE, precision.shape[1])
            page_precision = precision[b, start:stop].bool()
            v2_id = int(cache.metadata.v2_page_table[b, p].item())
            v4_id = int(cache.metadata.v4_page_table[b, p].item())
            if v2_id >= 0:
                torch.testing.assert_close(cache.v2_pattern_mask[v2_id], v_pattern_mask[b : b + 1, :, start:stop][:, :, ~page_precision])
                torch.testing.assert_close(cache.v2_assignment_idx[v2_id].to(v_idx.dtype), v_idx[b : b + 1, :, start:stop][:, :, ~page_precision])
            if v4_id >= 0:
                torch.testing.assert_close(cache.v4_pattern_mask[v4_id], v_pattern_mask[b : b + 1, :, start:stop][:, :, page_precision])
                torch.testing.assert_close(cache.v4_assignment_idx[v4_id].to(v_idx.dtype), v_idx[b : b + 1, :, start:stop][:, :, page_precision])


def test_page_batch_scale_zero_alignment():
    case = _build_case(2, 512, "strong")
    _attn, v_adjusted, precision, _v_pattern_mask, _v_idx, _centroids = case
    _candidate, cache = _page_output(case)
    pages = int(cache.metadata.num_pages[0].item())
    for b in range(2):
        for p in range(pages):
            start = p * PAGE_SIZE
            stop = min(start + PAGE_SIZE, precision.shape[1])
            page_precision = precision[b, start:stop].bool()
            v2_id = int(cache.metadata.v2_page_table[b, p].item())
            v4_id = int(cache.metadata.v4_page_table[b, p].item())
            if v2_id >= 0:
                expected = v_adjusted[b : b + 1, :, start:stop][:, :, ~page_precision].contiguous()
                exp_payload, exp_scale, exp_zero = cache.v2_payload[v2_id], cache.v2_scale[v2_id], cache.v2_zero[v2_id]
                got = pack_mixed_v_pages(expected, torch.zeros(1, expected.shape[2], dtype=torch.bool, device=expected.device), torch.zeros(1, NH_KV, expected.shape[2], dtype=torch.uint8, device=expected.device), torch.zeros(1, NH_KV, expected.shape[2], dtype=torch.int64, device=expected.device), cache.centroids, group_size=GROUP_SIZE, nh=NH)
                torch.testing.assert_close(exp_payload, got.v2_payload[0])
                torch.testing.assert_close(exp_scale, got.v2_scale[0])
                torch.testing.assert_close(exp_zero, got.v2_zero[0])
            if v4_id >= 0:
                expected = v_adjusted[b : b + 1, :, start:stop][:, :, page_precision].contiguous()
                exp_payload, exp_scale, exp_zero = cache.v4_payload[v4_id], cache.v4_scale[v4_id], cache.v4_zero[v4_id]
                got = pack_mixed_v_pages(expected, torch.ones(1, expected.shape[2], dtype=torch.bool, device=expected.device), torch.zeros(1, NH_KV, expected.shape[2], dtype=torch.uint8, device=expected.device), torch.zeros(1, NH_KV, expected.shape[2], dtype=torch.int64, device=expected.device), cache.centroids, group_size=GROUP_SIZE, nh=NH)
                torch.testing.assert_close(exp_payload, got.v4_payload[0])
                torch.testing.assert_close(exp_scale, got.v4_scale[0])
                torch.testing.assert_close(exp_zero, got.v4_zero[0])


def test_page_batch_no_v_materialization():
    case = _build_case(4, 512, "strong")
    reset_patternkv_page_batch_counters()
    _candidate, cache = _page_output(case)
    counters = get_patternkv_page_batch_counters()
    assert counters["page_batch_decode_calls"] == 1
    assert counters["python_serial_b1_dispatches"] == 0
    assert counters["historical_v_materialization_bytes"] == 0
    assert counters["page_value_materialized_bytes"] > 0
    assert cache.historical_materialization_calls == 0
    assert cache.historical_materialized_bytes == 0


def test_page_batch_k_remains_tight():
    case = _build_case(2, 512, "different")
    _candidate, cache = _page_output(case)
    assert not hasattr(cache, "k_page_table")
    assert not hasattr(cache, "k_capacity")
