import math

import pytest
import torch

from models.segmented_cache import dequantize_v_reference, pattern_gather_centroids, quantize_pack_v_reference
from quant.matmul import (
    cuda_attn_v_fused_with_base,
    cuda_attn_v_fused_with_base_debug,
    cuda_attn_v_mixed_fused_with_base,
    get_patternkv_mixed_v_counters,
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
