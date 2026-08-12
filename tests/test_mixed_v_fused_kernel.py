import math

import pytest
import torch

from models.segmented_cache import dequantize_v_reference, pattern_gather_centroids, quantize_pack_v_reference
from quant.matmul import (
    cuda_attn_v_fused_with_base,
    cuda_attn_v_fused_with_base_debug,
    cuda_attn_v_fused_with_base_gqa_v2,
    cuda_attn_v_mixed_fused_with_base,
    get_patternkv_mixed_v_counters,
    patternkv_gqa_v_backend,
    reset_patternkv_mixed_v_counters,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA mixed Value kernel tests require a GPU")


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


def _mask_for_case(case: str, tokens: int, *, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    if case == "all_v2":
        return mask
    if case == "all_v4":
        return torch.ones(tokens, dtype=torch.bool, device=device)
    k = max(1, int(round(tokens * 0.25)))
    if case == "first25":
        mask[:k] = True
    elif case == "last25":
        mask[-k:] = True
    elif case == "alternating":
        mask[1::4] = True
    elif case == "causal_like":
        idx = torch.linspace(0, tokens - 1, steps=k, device=device).round().long().unique()
        mask[idx[:k]] = True
        if int(mask.sum().item()) < k:
            mask[torch.arange(tokens - 1, -1, -1, device=device)[: k - int(mask.sum().item())]] = True
    elif case == "random":
        g = torch.Generator(device=device).manual_seed(20260812 + tokens)
        idx = torch.randperm(tokens, generator=g, device=device)[:k]
        mask[idx] = True
    elif case == "mixed25":
        idx = (torch.arange(k, device=device) * 4 + 1).clamp_max(tokens - 1)
        mask[idx.unique()] = True
        cursor = 0
        while int(mask.sum().item()) < k:
            mask[cursor] = True
            cursor += 1
    else:
        raise ValueError(case)
    return mask


def _build_case(case: str, tokens: int, *, seed: int = 1234):
    device = torch.device("cuda")
    torch.manual_seed(seed + tokens)
    precision = _mask_for_case(case, tokens, device=device).unsqueeze(0)
    v_adjusted = (torch.randn(1, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16) * 0.25).contiguous()
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1).contiguous()
    v_idx = torch.randint(0, CENTROIDS, (1, NH_KV, tokens), device=device, dtype=torch.int64)
    v_pattern_mask = (torch.rand(1, NH_KV, tokens, device=device) > 0.55).to(torch.uint8)
    attn = torch.softmax(torch.randn(1, NH, 1, tokens, device=device, dtype=torch.float16), dim=-1).contiguous()

    low = v_adjusted[:, :, ~precision[0], :].contiguous()
    high = v_adjusted[:, :, precision[0], :].contiguous()
    p2 = quantize_pack_v_reference(low, GROUP_SIZE, 2) if low.shape[2] else (None, None, None)
    p4 = quantize_pack_v_reference(high, GROUP_SIZE, 4) if high.shape[2] else (None, None, None)
    return {
        "precision": precision,
        "v_adjusted": v_adjusted,
        "centroids": centroids,
        "v_idx": v_idx,
        "v_pattern_mask": v_pattern_mask,
        "attn": attn,
        "p2": p2,
        "p4": p4,
    }


def _reference_output(case_data):
    precision = case_data["precision"]
    low_mask = ~precision[0].bool()
    high_mask = precision[0].bool()
    low = dequantize_v_reference(*case_data["p2"], GROUP_SIZE, 2) if int(low_mask.sum().item()) else None
    high = dequantize_v_reference(*case_data["p4"], GROUP_SIZE, 4) if int(high_mask.sum().item()) else None
    template = high if high is not None else low
    assert template is not None
    packed_v = torch.empty(
        template.shape[0],
        template.shape[1],
        precision.shape[1],
        template.shape[-1],
        dtype=template.dtype,
        device=template.device,
    )
    if low is not None:
        packed_v[:, :, low_mask, :] = low[:, :, : int(low_mask.sum().item()), :]
    if high is not None:
        packed_v[:, :, high_mask, :] = high[:, :, : int(high_mask.sum().item()), :]
    restored = packed_v + case_data["v_pattern_mask"].unsqueeze(-1).to(packed_v.dtype) * pattern_gather_centroids(
        case_data["v_idx"], case_data["centroids"]
    ).to(packed_v.dtype)
    return torch.matmul(case_data["attn"], _repeat_kv(restored, NH // NH_KV))


def _mixed_output(case_data):
    p2, p4 = case_data["p2"], case_data["p4"]
    return cuda_attn_v_mixed_fused_with_base(
        GROUP_SIZE,
        case_data["attn"],
        p2[0],
        p2[1],
        p2[2],
        p4[0],
        p4[1],
        p4[2],
        case_data["precision"],
        case_data["centroids"],
        case_data["v_pattern_mask"],
        case_data["v_idx"],
        NH,
        NH_KV,
    )


def _assert_close_with_metrics(fused: torch.Tensor, ref: torch.Tensor):
    diff = (fused.float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rel_l2 = float(torch.linalg.vector_norm((fused - ref).float()).item() / torch.linalg.vector_norm(ref.float()).clamp_min(1e-8).item())
    cosine = float(torch.nn.functional.cosine_similarity(fused.float().flatten(), ref.float().flatten(), dim=0).item())
    assert torch.isnan(fused).sum().item() == 0
    assert torch.isinf(fused).sum().item() == 0
    assert cosine >= 0.9999, {"max_abs": max_abs, "mean_abs": mean_abs, "rel_l2": rel_l2, "cosine": cosine}
    torch.testing.assert_close(fused, ref, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("case", ["all_v2", "all_v4", "mixed25", "random", "causal_like", "first25", "last25", "alternating"])
def test_mixed_v_fused_precision_layouts(case):
    reset_patternkv_mixed_v_counters()
    data = _build_case(case, 128)
    fused = _mixed_output(data)
    ref = _reference_output(data)
    _assert_close_with_metrics(fused, ref)
    counters = get_patternkv_mixed_v_counters()
    assert counters["mixed_v_fused_calls"] == 1
    assert counters["mixed_v_reference_calls"] == 0


@pytest.mark.parametrize("tokens", [128, 256, 512, 1024, 2048, 4096, 8192])
def test_mixed_v_fused_decode_lengths(tokens):
    data = _build_case("mixed25", tokens)
    fused = _mixed_output(data)
    ref = _reference_output(data)
    _assert_close_with_metrics(fused, ref)


def test_mixed_v_fused_tiny_mapping_detects_logical_order_errors():
    data = _build_case("mixed25", 8, seed=99)
    data["precision"] = torch.tensor([[False, True, False, False, True, False, False, False]], device="cuda")
    # Rebuild payloads after forcing the adversarial logical positions [1, 4].
    low = data["v_adjusted"][:, :, ~data["precision"][0], :].contiguous()
    high = data["v_adjusted"][:, :, data["precision"][0], :].contiguous()
    data["p2"] = quantize_pack_v_reference(low, GROUP_SIZE, 2)
    data["p4"] = quantize_pack_v_reference(high, GROUP_SIZE, 4)
    data["attn"] = torch.arange(1, 9, device="cuda", dtype=torch.float16).view(1, 1, 1, 8).expand(1, NH, 1, 8).contiguous()
    data["attn"] = (data["attn"] / data["attn"].sum(dim=-1, keepdim=True)).contiguous()
    fused = _mixed_output(data)
    ref = _reference_output(data)
    _assert_close_with_metrics(fused, ref)


def test_mixed_v_fused_all_v2_matches_existing_fused_kernel():
    data = _build_case("all_v2", 128)
    fused = _mixed_output(data)
    p2 = data["p2"]
    base = cuda_attn_v_fused_with_base(
        GROUP_SIZE,
        data["attn"],
        p2[0],
        p2[1],
        p2[2],
        2,
        data["centroids"],
        data["v_pattern_mask"],
        data["v_idx"],
        NH,
        NH_KV,
    )
    _assert_close_with_metrics(fused, base)


def test_mixed_v_fused_all_v4_matches_existing_fused_kernel():
    data = _build_case("all_v4", 128)
    fused = _mixed_output(data)
    p4 = data["p4"]
    base = cuda_attn_v_fused_with_base(
        GROUP_SIZE,
        data["attn"],
        p4[0],
        p4[1],
        p4[2],
        4,
        data["centroids"],
        data["v_pattern_mask"],
        data["v_idx"],
        NH,
        NH_KV,
    )
    _assert_close_with_metrics(fused, base)


def test_full_mode_matches_existing_path():
    data = _build_case("all_v2", 128)
    p2 = data["p2"]
    production = cuda_attn_v_fused_with_base(
        GROUP_SIZE,
        data["attn"],
        p2[0],
        p2[1],
        p2[2],
        2,
        data["centroids"],
        data["v_pattern_mask"],
        data["v_idx"],
        NH,
        NH_KV,
    )
    debug_full = cuda_attn_v_fused_with_base_debug(
        GROUP_SIZE,
        data["attn"],
        p2[0],
        p2[1],
        p2[2],
        2,
        data["centroids"],
        data["v_pattern_mask"],
        data["v_idx"],
        NH,
        NH_KV,
        debug_mode="FULL",
    )
    torch.testing.assert_close(debug_full, production, rtol=1e-5, atol=1e-5)


def test_residual_only_debug_mode_not_default():
    data = _build_case("all_v2", 128)
    p2 = data["p2"]
    production = cuda_attn_v_fused_with_base(
        GROUP_SIZE,
        data["attn"],
        p2[0],
        p2[1],
        p2[2],
        2,
        data["centroids"],
        data["v_pattern_mask"],
        data["v_idx"],
        NH,
        NH_KV,
    )
    residual_only = cuda_attn_v_fused_with_base_debug(
        GROUP_SIZE,
        data["attn"],
        p2[0],
        p2[1],
        p2[2],
        2,
        data["centroids"],
        data["v_pattern_mask"],
        data["v_idx"],
        NH,
        NH_KV,
        debug_mode="RESIDUAL_ONLY",
    )
    assert not torch.allclose(residual_only, production, rtol=1e-5, atol=1e-5)


def test_centroid_debug_mode_not_default():
    data = _build_case("all_v2", 128)
    p2 = data["p2"]
    production = cuda_attn_v_fused_with_base(
        GROUP_SIZE,
        data["attn"],
        p2[0],
        p2[1],
        p2[2],
        2,
        data["centroids"],
        data["v_pattern_mask"],
        data["v_idx"],
        NH,
        NH_KV,
    )
    centroid_only = cuda_attn_v_fused_with_base_debug(
        GROUP_SIZE,
        data["attn"],
        p2[0],
        p2[1],
        p2[2],
        2,
        data["centroids"],
        data["v_pattern_mask"],
        data["v_idx"],
        NH,
        NH_KV,
        debug_mode="CENTROID_ONLY",
    )
    assert not torch.allclose(centroid_only, production, rtol=1e-5, atol=1e-5)


def test_invalid_debug_mode_rejected():
    data = _build_case("all_v2", 128)
    p2 = data["p2"]
    with pytest.raises(ValueError):
        cuda_attn_v_fused_with_base_debug(
            GROUP_SIZE,
            data["attn"],
            p2[0],
            p2[1],
            p2[2],
            2,
            data["centroids"],
            data["v_pattern_mask"],
            data["v_idx"],
            NH,
            NH_KV,
            debug_mode="NOT_A_MODE",
        )


def test_production_backend_unchanged_when_env_unset(monkeypatch):
    data = _build_case("all_v2", 128)
    p2 = data["p2"]
    monkeypatch.delenv("PATTERNKV_ATTENTION_V_DEBUG_MODE", raising=False)
    baseline = cuda_attn_v_fused_with_base(
        GROUP_SIZE,
        data["attn"],
        p2[0],
        p2[1],
        p2[2],
        2,
        data["centroids"],
        data["v_pattern_mask"],
        data["v_idx"],
        NH,
        NH_KV,
    )
    monkeypatch.setenv("PATTERNKV_ATTENTION_V_DEBUG_MODE", "CENTROID_ONLY")
    env_ignored = cuda_attn_v_fused_with_base(
        GROUP_SIZE,
        data["attn"],
        p2[0],
        p2[1],
        p2[2],
        2,
        data["centroids"],
        data["v_pattern_mask"],
        data["v_idx"],
        NH,
        NH_KV,
    )
    torch.testing.assert_close(env_ignored, baseline, rtol=1e-5, atol=1e-5)


def _single_lane_output(data, *, bit: int, mode: str | None = None):
    precision = data["precision"][0].bool()
    token_mask = ~precision if bit == 2 else precision
    payload = data["p2"] if bit == 2 else data["p4"]
    fn = cuda_attn_v_fused_with_base if mode is None else cuda_attn_v_fused_with_base_debug
    kwargs = {} if mode is None else {"debug_mode": mode}
    return fn(
        GROUP_SIZE,
        data["attn"][..., token_mask].contiguous(),
        payload[0],
        payload[1],
        payload[2],
        bit,
        data["centroids"],
        data["v_pattern_mask"][:, :, token_mask].contiguous(),
        data["v_idx"][:, :, token_mask].contiguous(),
        NH,
        NH_KV,
        **kwargs,
    )


def test_histogram_baseline_matches_reference():
    data = _build_case("all_v2", 256)
    fused = _mixed_output(data)
    ref = _reference_output(data)
    _assert_close_with_metrics(fused, ref)


def test_warp_aggregate_histogram_matches_baseline():
    data = _build_case("all_v2", 256)
    baseline = _single_lane_output(data, bit=2, mode="FULL")
    warp_agg = _single_lane_output(data, bit=2, mode="WARP_AGG_FULL")
    torch.testing.assert_close(warp_agg, baseline, rtol=1e-5, atol=1e-5)


def test_skewed_assignment_correctness():
    data = _build_case("all_v2", 256)
    skew = torch.rand_like(data["v_idx"].float()) < 0.5
    data["v_idx"] = torch.randint(1, CENTROIDS, data["v_idx"].shape, device="cuda", dtype=torch.int64)
    data["v_idx"][skew] = 0
    data["v_pattern_mask"].fill_(1)
    fused = _mixed_output(data)
    ref = _reference_output(data)
    _assert_close_with_metrics(fused, ref)


def test_all_same_centroid_correctness():
    data = _build_case("all_v2", 256)
    data["v_idx"].zero_()
    data["v_pattern_mask"].fill_(1)
    fused = _mixed_output(data)
    ref = _reference_output(data)
    _assert_close_with_metrics(fused, ref)


def test_mask_zero_correctness():
    data = _build_case("all_v2", 256)
    data["v_pattern_mask"].zero_()
    fused = _mixed_output(data)
    ref = _reference_output(data)
    _assert_close_with_metrics(fused, ref)


def test_mask_full_correctness():
    data = _build_case("all_v2", 256)
    data["v_pattern_mask"].fill_(1)
    fused = _mixed_output(data)
    ref = _reference_output(data)
    _assert_close_with_metrics(fused, ref)


def test_v4_regression_guard():
    data = _build_case("all_v4", 256)
    production = _single_lane_output(data, bit=4)
    baseline_full = _single_lane_output(data, bit=4, mode="FULL")
    torch.testing.assert_close(production, baseline_full, rtol=1e-5, atol=1e-5)


def test_production_default_path(monkeypatch):
    data = _build_case("all_v2", 256)
    monkeypatch.setenv("PATTERNKV_ATTENTION_V_DEBUG_MODE", "FULL")
    production = _single_lane_output(data, bit=2)
    monkeypatch.setenv("PATTERNKV_ATTENTION_V_DEBUG_MODE", "WARP_AGG_FULL")
    still_production = _single_lane_output(data, bit=2)
    torch.testing.assert_close(still_production, production, rtol=1e-5, atol=1e-5)


def test_current_per_warp_histogram_still_default():
    data = _build_case("all_v2", 256)
    production = _single_lane_output(data, bit=2)
    lane0_table = _single_lane_output(data, bit=2, mode="LANE0_TABLE_FULL")
    per_warp_hist = _single_lane_output(data, bit=2, mode="PER_WARP_HIST_FULL")
    torch.testing.assert_close(production, lane0_table, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lane0_table, per_warp_hist, rtol=1e-5, atol=1e-5)


def test_no_table_debug_mode_not_default():
    data = _build_case("all_v2", 256)
    data["v_pattern_mask"].fill_(1)
    production = _single_lane_output(data, bit=2)
    no_table = _single_lane_output(data, bit=2, mode="NO_TABLE_CONTRIBUTION")
    assert not torch.allclose(no_table, production, rtol=1e-5, atol=1e-5)


def test_centroid_table_candidate_matches_baseline():
    data = _build_case("all_v2", 256)
    baseline = _single_lane_output(data, bit=2, mode="PER_WARP_HIST_FULL")
    candidate = _single_lane_output(data, bit=2, mode="LANE0_TABLE_FULL")
    torch.testing.assert_close(candidate, baseline, rtol=1e-5, atol=1e-5)


def test_single_active_centroid():
    data = _build_case("all_v2", 256)
    data["v_idx"].zero_()
    data["v_pattern_mask"].fill_(1)
    production = _single_lane_output(data, bit=2)
    baseline = _single_lane_output(data, bit=2, mode="PER_WARP_HIST_FULL")
    torch.testing.assert_close(production, baseline, rtol=1e-5, atol=1e-5)
    _assert_close_with_metrics(_mixed_output(data), _reference_output(data))


def test_all_centroids_active():
    data = _build_case("all_v2", 256)
    ids = torch.arange(data["v_idx"].shape[-1], device="cuda", dtype=torch.int64) % CENTROIDS
    data["v_idx"] = ids.view(1, 1, -1).expand_as(data["v_idx"]).contiguous()
    data["v_pattern_mask"].fill_(1)
    production = _single_lane_output(data, bit=2)
    baseline = _single_lane_output(data, bit=2, mode="PER_WARP_HIST_FULL")
    torch.testing.assert_close(production, baseline, rtol=1e-5, atol=1e-5)
    _assert_close_with_metrics(_mixed_output(data), _reference_output(data))


def test_gqa_ratio4_correctness():
    assert NH // NH_KV == 4
    data = _build_case("mixed25", 512)
    fused = _mixed_output(data)
    ref = _reference_output(data)
    _assert_close_with_metrics(fused, ref)


def test_production_mode_unchanged_when_debug_disabled(monkeypatch):
    data = _build_case("all_v2", 256)
    monkeypatch.delenv("PATTERNKV_ATTENTION_V_DEBUG_MODE", raising=False)
    production = _single_lane_output(data, bit=2)
    monkeypatch.setenv("PATTERNKV_ATTENTION_V_DEBUG_MODE", "NO_TABLE_CONTRIBUTION")
    env_ignored = _single_lane_output(data, bit=2)
    torch.testing.assert_close(env_ignored, production, rtol=1e-5, atol=1e-5)


def _gqa_candidate_output(data, *, nh: int = NH, nh_kv: int = NH_KV):
    precision = data["precision"][0].bool()
    token_mask = ~precision
    p2 = data["p2"]
    return cuda_attn_v_fused_with_base_gqa_v2(
        GROUP_SIZE,
        data["attn"][:, :nh, :, token_mask].contiguous(),
        p2[0],
        p2[1],
        p2[2],
        2,
        data["centroids"],
        data["v_pattern_mask"][:, :, token_mask].contiguous(),
        data["v_idx"][:, :, token_mask].contiguous(),
        nh,
        nh_kv,
    )


def _gqa_baseline_output(data, *, nh: int = NH, nh_kv: int = NH_KV):
    precision = data["precision"][0].bool()
    token_mask = ~precision
    p2 = data["p2"]
    return cuda_attn_v_fused_with_base(
        GROUP_SIZE,
        data["attn"][:, :nh, :, token_mask].contiguous(),
        p2[0],
        p2[1],
        p2[2],
        2,
        data["centroids"],
        data["v_pattern_mask"][:, :, token_mask].contiguous(),
        data["v_idx"][:, :, token_mask].contiguous(),
        nh,
        nh_kv,
    )


def test_gqa_qhead_to_kvhead_mapping():
    device = torch.device("cuda")
    tokens = 128
    precision = torch.zeros(1, tokens, dtype=torch.bool, device=device)
    values = torch.zeros(1, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16)
    p2 = quantize_pack_v_reference(values, GROUP_SIZE, 2)
    centroids = torch.zeros(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16)
    for hk in range(NH_KV):
        centroids[hk, 0, :].fill_(float(hk + 1))
    data = {
        "precision": precision,
        "attn": torch.full((1, NH, 1, tokens), 1.0 / tokens, device=device, dtype=torch.float16),
        "p2": p2,
        "centroids": centroids,
        "v_pattern_mask": torch.ones(1, NH_KV, tokens, device=device, dtype=torch.uint8),
        "v_idx": torch.zeros(1, NH_KV, tokens, device=device, dtype=torch.int64),
    }
    out = _gqa_candidate_output(data)
    for hq in range(NH):
        hk = hq // 4
        torch.testing.assert_close(out[0, hq, 0], torch.full((HEAD_DIM,), float(hk + 1), device=device, dtype=torch.float16))


def test_gqa_ratio4_v2_matches_baseline():
    data = _build_case("all_v2", 512)
    torch.testing.assert_close(_gqa_candidate_output(data), _gqa_baseline_output(data), rtol=1e-5, atol=1e-5)


def test_gqa_ratio1_fallback():
    data = _build_case("all_v2", 256)
    candidate = _gqa_candidate_output(data, nh=8, nh_kv=8)
    baseline = _gqa_baseline_output(data, nh=8, nh_kv=8)
    torch.testing.assert_close(candidate, baseline, rtol=1e-5, atol=1e-5)


def test_gqa_ratio2_if_supported():
    data = _build_case("all_v2", 256)
    candidate = _gqa_candidate_output(data, nh=16, nh_kv=8)
    baseline = _gqa_baseline_output(data, nh=16, nh_kv=8)
    torch.testing.assert_close(candidate, baseline, rtol=1e-5, atol=1e-5)


def test_all_same_centroid_gqa():
    data = _build_case("all_v2", 256)
    data["v_idx"].zero_()
    data["v_pattern_mask"].fill_(1)
    torch.testing.assert_close(_gqa_candidate_output(data), _gqa_baseline_output(data), rtol=1e-5, atol=1e-5)


def test_skewed_assignment_gqa():
    data = _build_case("all_v2", 256)
    skew = torch.rand_like(data["v_idx"].float()) < 0.75
    data["v_idx"] = torch.randint(1, CENTROIDS, data["v_idx"].shape, device="cuda", dtype=torch.int64)
    data["v_idx"][skew] = 0
    data["v_pattern_mask"].fill_(1)
    torch.testing.assert_close(_gqa_candidate_output(data), _gqa_baseline_output(data), rtol=1e-5, atol=1e-5)


def test_mask_zero_gqa():
    data = _build_case("all_v2", 256)
    data["v_pattern_mask"].zero_()
    torch.testing.assert_close(_gqa_candidate_output(data), _gqa_baseline_output(data), rtol=1e-5, atol=1e-5)


def test_mask_full_gqa():
    data = _build_case("all_v2", 256)
    data["v_pattern_mask"].fill_(1)
    torch.testing.assert_close(_gqa_candidate_output(data), _gqa_baseline_output(data), rtol=1e-5, atol=1e-5)


def test_per_warp_histogram_preserved():
    data = _build_case("all_v2", 256)
    candidate = _gqa_candidate_output(data)
    baseline = _gqa_baseline_output(data)
    torch.testing.assert_close(candidate, baseline, rtol=1e-5, atol=1e-5)


def test_lane0_table_contribution_preserved():
    data = _build_case("all_v2", 256)
    data["v_pattern_mask"].fill_(1)
    candidate = _gqa_candidate_output(data)
    baseline = _gqa_baseline_output(data)
    torch.testing.assert_close(candidate, baseline, rtol=1e-5, atol=1e-5)


def test_v4_baseline_unchanged(monkeypatch):
    data = _build_case("all_v4", 256)
    monkeypatch.setenv("PATTERNKV_GQA_V_BACKEND", "gqa")
    fused = _mixed_output(data)
    p4 = data["p4"]
    baseline = cuda_attn_v_fused_with_base(
        GROUP_SIZE,
        data["attn"],
        p4[0],
        p4[1],
        p4[2],
        4,
        data["centroids"],
        data["v_pattern_mask"],
        data["v_idx"],
        NH,
        NH_KV,
    )
    _assert_close_with_metrics(fused, baseline)


def test_default_backend_is_safe(monkeypatch):
    monkeypatch.delenv("PATTERNKV_GQA_V_BACKEND", raising=False)
    assert patternkv_gqa_v_backend() == "baseline"
    data = _build_case("mixed25", 256)
    reset_patternkv_mixed_v_counters()
    _mixed_output(data)
    counters = get_patternkv_mixed_v_counters()
    assert counters["gqa_v2_calls"] == 0
    assert counters["gqa_v2_fallbacks"] == 0
