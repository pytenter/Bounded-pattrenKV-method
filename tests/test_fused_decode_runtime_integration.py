from types import SimpleNamespace

import pytest
import torch

from models.segmented_cache import append_decode, build_cache_from_prefill, validate_cache
from models.llama_patternkv import patternkv_mixed_value_attention
from quant.page_batch import (
    correctness_metrics,
    get_patternkv_real_decode_counters,
    reset_patternkv_real_decode_counters,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused decode runtime integration tests require a GPU")

GROUP_SIZE = 128
NH = 4
NH_KV = 2
HEAD_DIM = 128
CENTROIDS = 16


def _case(batch: int, tokens: int, *, seed: int = 20260813):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed + batch * 100_000 + tokens)
    k = torch.randn(batch, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16, generator=generator) * 0.02
    v = torch.randn(batch, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16, generator=generator) * 0.02
    centroids = torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16, generator=generator) * 0.02
    assignment = torch.randint(0, CENTROIDS, (batch, NH_KV, tokens), device=device, dtype=torch.long, generator=generator)
    pattern = (torch.rand(batch, NH_KV, tokens, device=device, generator=generator) > 0.5)
    weights = torch.softmax(torch.randn(batch, NH, 1, tokens, device=device, dtype=torch.float16, generator=generator), dim=-1)
    return k, v, centroids, assignment, pattern, weights


def _cache(k, v, centroids, assignment, pattern):
    return build_cache_from_prefill(
        k,
        v,
        sink_length=0,
        recent_length=0,
        group_size=GROUP_SIZE,
        k_bits=2,
        v_bits=2,
        pattern=True,
        k_centroids=centroids,
        v_centroids=centroids,
        k_assignments=assignment,
        v_assignment_idx=assignment,
        v_pattern_mask=pattern,
        cache_mode="segmented_chunked",
        chunk_length=GROUP_SIZE,
        value_objective="v_dir",
        v_precision_selector="causal_v4",
        v4_budget_fraction=0.25,
    )


def _module():
    return SimpleNamespace(group_size=GROUP_SIZE, num_key_value_groups=NH // NH_KV, num_heads=NH, num_key_value_heads=NH_KV)


def _value(cache, weights):
    return patternkv_mixed_value_attention(_module(), cache, weights, cache.v_pattern_mask, cache.packed_v_tokens)


def _assert_close(got, ref):
    metrics = correctness_metrics(got, ref)
    assert metrics["nan"] == 0
    assert metrics["inf"] == 0
    assert metrics["relative_l2"] <= 1e-6, metrics
    assert metrics["cosine"] >= 0.999999, metrics
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("batch,tokens", [(1, 2048), (2, 2048), (4, 2048), (2, 2051)])
def test_real_decode_fused_page_value_path_matches_independent_b1(batch, tokens, monkeypatch):
    monkeypatch.setenv("PATTERNKV_RUNTIME_NH", str(NH))
    monkeypatch.setenv("PATTERNKV_MIXED_V_BACKEND", "fused_page")
    k, v, centroids, assignment, pattern, weights = _case(batch, tokens)
    cache = _cache(k, v, centroids, assignment, pattern)
    reset_patternkv_real_decode_counters()
    got = _value(cache, weights[:, :, :, : cache.packed_v_tokens])
    refs = []
    for b in range(batch):
        b_cache = _cache(k[b : b + 1], v[b : b + 1], centroids, assignment[b : b + 1], pattern[b : b + 1])
        refs.append(_value(b_cache, weights[b : b + 1, :, :, : b_cache.packed_v_tokens]))
    ref = torch.cat(refs, dim=0)
    torch.cuda.synchronize()
    _assert_close(got, ref)
    counters = get_patternkv_real_decode_counters()
    assert counters["fused_page_operator_calls"] == batch + 1
    assert counters["legacy_mixed_v_operator_calls"] == 0
    assert counters["serial_b1_dispatches"] == 0
    assert counters["operator_ready_pool_full_rebuilds"] == 0
    assert counters["page_value_materialization_bytes"] == 0
    assert counters["historical_v_materialization_bytes"] == 0
    assert counters["gpu_tensor_item_calls_hot_path"] == 0
    assert counters["python_page_dispatches"] == 0


def test_multi_step_cache_append_keeps_incremental_page_pools(monkeypatch):
    monkeypatch.setenv("PATTERNKV_RUNTIME_NH", str(NH))
    monkeypatch.setenv("PATTERNKV_MIXED_V_BACKEND", "fused_page")
    k, v, centroids, assignment, pattern, weights = _case(2, 2048)
    cache = _cache(k, v, centroids, assignment, pattern)
    before_pages = int(cache.operator_ready_page_pools.metadata.v2_page_table.shape[1])
    new_k, new_v, _centroids, _new_assignment, _new_pattern, _weights = _case(2, GROUP_SIZE, seed=20260901)
    append_decode(cache, new_k, new_v)
    validate_cache(cache)
    after_pages = int(cache.operator_ready_page_pools.metadata.v2_page_table.shape[1])
    assert after_pages == before_pages + 1
    assert int(cache.operator_ready_page_pools.metadata.seq_lens.min().item()) == cache.packed_v_tokens
    assert int(cache.operator_ready_page_pools.metadata.seq_lens.max().item()) == cache.packed_v_tokens


def test_selector_and_cache_isolation_with_different_requests(monkeypatch):
    monkeypatch.setenv("PATTERNKV_RUNTIME_NH", str(NH))
    monkeypatch.setenv("PATTERNKV_MIXED_V_BACKEND", "fused_page")
    k, v, centroids, assignment, pattern, weights = _case(2, 2048, seed=11)
    mutated_k = k.clone()
    mutated_v = v.clone()
    mutated_assignment = assignment.clone()
    mutated_pattern = pattern.clone()
    mutated_k[0].mul_(1.7)
    mutated_v[0].mul_(-1.3)
    mutated_assignment[0] = (mutated_assignment[0] + 3) % CENTROIDS
    mutated_pattern[0] = ~mutated_pattern[0]
    cache_a = _cache(k, v, centroids, assignment, pattern)
    cache_b = _cache(mutated_k, mutated_v, centroids, mutated_assignment, mutated_pattern)
    out_a = _value(cache_a, weights[:, :, :, : cache_a.packed_v_tokens])
    out_b = _value(cache_b, weights[:, :, :, : cache_b.packed_v_tokens])
    assert not torch.equal(cache_a.v_precision_mask[0], cache_b.v_precision_mask[0])
    torch.testing.assert_close(cache_a.v_precision_mask[1], cache_b.v_precision_mask[1], rtol=0, atol=0)
    torch.testing.assert_close(out_a[1], out_b[1], rtol=0, atol=0)
