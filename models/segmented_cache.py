from __future__ import annotations

import os
from dataclasses import dataclass
from math import lcm
from typing import Any

import torch

from quant.new_pack import pack_tensor, triton_quantize_and_pack_along_last_dim, unpack_tensor


@dataclass
class QuantizedKVCache:
    sink_k: torch.Tensor | None = None
    sink_v: torch.Tensor | None = None
    packed_k: torch.Tensor | None = None
    packed_k_scale: torch.Tensor | None = None
    packed_k_zero: torch.Tensor | None = None
    packed_v: torch.Tensor | None = None
    packed_v_scale: torch.Tensor | None = None
    packed_v_zero: torch.Tensor | None = None
    pending_k: torch.Tensor | None = None
    pending_v: torch.Tensor | None = None
    recent_k: torch.Tensor | None = None
    recent_v: torch.Tensor | None = None
    total_tokens: int = 0
    packed_k_tokens: int = 0
    packed_v_tokens: int = 0
    sink_length: int = 0
    recent_length: int = 128
    group_size: int = 128
    k_bits: int = 2
    v_bits: int = 2
    pack_count_k: int = 0
    pack_count_v: int = 0
    cache_mode: str = "segmented_rolling"
    chunk_length: int = 128


@dataclass
class PatternQuantizedKVCache(QuantizedKVCache):
    k_assignments: torch.Tensor | None = None
    v_assignments: torch.Tensor | None = None
    v_assignment_idx: torch.Tensor | None = None
    v_pattern_mask: torch.Tensor | None = None
    k_centroids: torch.Tensor | None = None
    v_centroids: torch.Tensor | None = None
    centroid_updates_k: int = 0
    centroid_updates_v: int = 0
    value_objective: str = "base"


CHUNKED_CACHE_MODE = "segmented_chunked"
ROLLING_CACHE_MODE = "segmented_rolling"


def normalize_cache_mode(cache_mode: str | None) -> str:
    mode = str(cache_mode or ROLLING_CACHE_MODE).strip().lower()
    aliases = {
        "segmented": ROLLING_CACHE_MODE,
        "rolling": ROLLING_CACHE_MODE,
        "chunked": CHUNKED_CACHE_MODE,
    }
    mode = aliases.get(mode, mode)
    if mode not in (CHUNKED_CACHE_MODE, ROLLING_CACHE_MODE):
        raise ValueError(f"unsupported segmented cache mode: {cache_mode!r}")
    return mode


def cache_validate_enabled() -> bool:
    return os.environ.get("PATTERNKV_CACHE_VALIDATE") == "1"


def segment_lengths(total_tokens: int, sink_length: int, recent_length: int) -> dict[str, int]:
    if total_tokens < 0 or sink_length < 0 or recent_length < 0:
        raise ValueError("cache token lengths must be non-negative")
    sink_tokens = min(total_tokens, sink_length)
    non_sink_tokens = max(total_tokens - sink_tokens, 0)
    recent_tokens = min(non_sink_tokens, recent_length)
    quantized_history_tokens = max(non_sink_tokens - recent_tokens, 0)
    return {
        "sink_tokens": sink_tokens,
        "quantized_history_tokens": quantized_history_tokens,
        "recent_tokens": recent_tokens,
        "total_tokens": total_tokens,
    }


def tensor_tokens(value: torch.Tensor | None) -> int:
    return int(value.shape[2]) if torch.is_tensor(value) else 0


def packed_last_dim_tokens(value: torch.Tensor | None, bits: int) -> int:
    return int(value.shape[-1] * (32 // bits)) if torch.is_tensor(value) else 0


def _empty_like_tokens(reference: torch.Tensor, tokens: int) -> torch.Tensor | None:
    if tokens == 0:
        return None
    return reference[:, :, :tokens, :].contiguous()


def _cat_token(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
    if a is None:
        return b
    if b is None:
        return a
    return torch.cat([a, b], dim=2).contiguous()


def _cat_packed_k(cache: QuantizedKVCache, packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, tokens: int) -> None:
    cache.packed_k = packed if cache.packed_k is None else torch.cat([cache.packed_k, packed], dim=3)
    cache.packed_k_scale = scale if cache.packed_k_scale is None else torch.cat([cache.packed_k_scale, scale], dim=3)
    cache.packed_k_zero = zero if cache.packed_k_zero is None else torch.cat([cache.packed_k_zero, zero], dim=3)
    cache.packed_k_tokens += int(tokens)
    cache.pack_count_k += 1


def _cat_packed_v(cache: QuantizedKVCache, packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, tokens: int) -> None:
    cache.packed_v = packed if cache.packed_v is None else torch.cat([cache.packed_v, packed], dim=2)
    cache.packed_v_scale = scale if cache.packed_v_scale is None else torch.cat([cache.packed_v_scale, scale], dim=2)
    cache.packed_v_zero = zero if cache.packed_v_zero is None else torch.cat([cache.packed_v_zero, zero], dim=2)
    cache.packed_v_tokens += int(tokens)
    cache.pack_count_v += 1


def _cat_assignment(current: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    return value if current is None else torch.cat([current, value], dim=2).contiguous()


def _assign_minmax_hnk(x: torch.Tensor, centroids: torch.Tensor, block_k: int = 256) -> torch.Tensor:
    heads, tokens, dim = x.shape
    if centroids.shape[0] != heads or centroids.shape[-1] != dim:
        raise ValueError(f"centroid shape mismatch: x={tuple(x.shape)} centroids={tuple(centroids.shape)}")
    best_dist = torch.full((heads, tokens), float("inf"), device=x.device, dtype=x.dtype)
    best_idx = torch.zeros((heads, tokens), device=x.device, dtype=torch.long)
    for start in range(0, centroids.shape[1], block_k):
        stop = min(start + block_k, centroids.shape[1])
        diff = x.unsqueeze(2) - centroids[:, start:stop, :].unsqueeze(1)
        distance = diff.amax(dim=-1) - diff.amin(dim=-1)
        cand, idx = distance.min(dim=-1)
        better = cand < best_dist
        best_dist[better] = cand[better]
        best_idx[better] = (start + idx)[better]
    return best_idx


def pattern_chebyshev_center_per_head(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected [heads, tokens, dim], got {tuple(x.shape)}")
    return (x.amin(dim=1, keepdim=True) + x.amax(dim=1, keepdim=True)) * 0.5


def pattern_gather_centroids(idx: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    if idx.dim() != 3 or centroids.dim() != 3:
        raise ValueError(f"expected idx [B,H,T] and centroids [H,M,D], got {tuple(idx.shape)} {tuple(centroids.shape)}")
    bsz, heads, tokens = idx.shape
    if centroids.shape[0] != heads:
        raise ValueError(f"centroid head mismatch: idx heads={heads}, centroids={centroids.shape[0]}")
    dim = centroids.shape[-1]
    expanded = centroids.unsqueeze(0).expand(bsz, -1, -1, -1)
    return torch.gather(expanded, 2, idx.unsqueeze(-1).expand(-1, -1, -1, dim))


def pattern_nearest_v_centroid(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4 or centroids.dim() != 3:
        raise ValueError(f"expected x [B,H,T,D] and centroids [H,M,D], got {tuple(x.shape)} {tuple(centroids.shape)}")
    bsz, heads, tokens, dim = x.shape
    if centroids.shape[0] != heads or centroids.shape[-1] != dim:
        raise ValueError(f"centroid shape mismatch: x={tuple(x.shape)} centroids={tuple(centroids.shape)}")
    diff = x.unsqueeze(2) - centroids.unsqueeze(0).unsqueeze(3)
    distance = diff.amax(dim=-1) - diff.amin(dim=-1)
    return distance.argmin(dim=2).contiguous()


def normalize_value_objective(value_objective: str | None) -> str:
    value = str(value_objective or "base").strip().lower().replace("-", "_")
    aliases = {
        "baseline": "base",
        "minmax": "base",
        "range": "base",
        "dir": "v_dir",
        "direction": "v_dir",
        "hybrid": "v_hybrid",
    }
    value = aliases.get(value, value)
    if value not in {"base", "v_dir", "v_hybrid"}:
        raise ValueError(f"unsupported PatternKV Value objective: {value_objective!r}")
    return value


def pattern_v_threshold_and_mask(x: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape != base.shape:
        raise ValueError(f"V threshold tensors must match: {tuple(x.shape)} != {tuple(base.shape)}")
    eps = 1e-12
    range_x = (x.amax(dim=-1) - x.amin(dim=-1)).clamp_min(eps)
    diff = x - base
    range_residual = (diff.amax(dim=-1) - diff.amin(dim=-1)).clamp_min(eps)
    rho = (range_residual / range_x).clamp_min(0.0)
    rho4 = rho * rho
    rho4 = rho4 * rho4
    f32 = torch.float32
    z = torch.sqrt(torch.tensor(2.0, dtype=f32, device=x.device)) * torch.erfinv(torch.tensor(0.9, dtype=f32, device=x.device))
    z = z.to(x.dtype)
    lhs = 1.0 - rho * rho
    rhs = (2.0 * z / torch.sqrt(torch.tensor(5.0 * float(x.shape[-1]), dtype=x.dtype, device=x.device))) * torch.sqrt(1.0 + rho4)
    return rho.unsqueeze(-1), lhs >= rhs


def affine_dequantize_last_dim_reference(x: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    if x.shape[-1] % group_size != 0:
        raise ValueError(f"last dim {x.shape[-1]} must be divisible by group_size={group_size}")
    levels = float((1 << bits) - 1)
    grouped = x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
    zero = grouped.amin(dim=-1)
    mx = grouped.amax(dim=-1)
    scale = ((mx - zero) / levels).clamp_min(torch.finfo(x.dtype).eps)
    q = torch.round((grouped - zero.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp_(0, levels)
    out = q.to(x.dtype) * scale.unsqueeze(-1) + zero.unsqueeze(-1)
    return out.reshape_as(x).contiguous()


def pattern_v_candidate_reconstructions(
    x: torch.Tensor,
    centroids: torch.Tensor,
    *,
    group_size: int,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 4 or centroids.dim() != 3:
        raise ValueError(f"expected x [B,H,T,D] and centroids [H,M,D], got {tuple(x.shape)} {tuple(centroids.shape)}")
    bsz, heads, tokens, dim = x.shape
    if centroids.shape[0] != heads or centroids.shape[-1] != dim:
        raise ValueError(f"centroid shape mismatch: x={tuple(x.shape)} centroids={tuple(centroids.shape)}")
    expanded_x = x.unsqueeze(2)
    expanded_c = centroids.unsqueeze(0).unsqueeze(3)
    _, mask = pattern_v_threshold_and_mask(expanded_x.expand(-1, -1, centroids.shape[1], -1, -1), expanded_c.expand(bsz, -1, -1, tokens, -1))
    adjusted = expanded_x - mask.unsqueeze(-1).to(x.dtype) * expanded_c
    dequant = affine_dequantize_last_dim_reference(adjusted.contiguous(), group_size, bits)
    restored = dequant + mask.unsqueeze(-1).to(x.dtype) * expanded_c
    base_score = (expanded_x - expanded_c).amax(dim=-1) - (expanded_x - expanded_c).amin(dim=-1)
    return restored, mask, base_score


def _vector_direction_error(x: torch.Tensor, y: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    xf = x.float()
    yf = y.float()
    x_norm = xf.norm(dim=-1)
    y_norm = yf.norm(dim=-1)
    nre = (xf - yf).pow(2).sum(dim=-1) / xf.pow(2).sum(dim=-1).clamp_min(eps)
    valid = (x_norm >= eps) & (y_norm >= eps)
    cosine = (xf * yf).sum(dim=-1) / (x_norm.clamp_min(eps) * y_norm.clamp_min(eps))
    return torch.where(valid, 1.0 - cosine.clamp(-1.0, 1.0), nre)


def _vector_nre(x: torch.Tensor, y: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    xf = x.float()
    yf = y.float()
    return (xf - yf).pow(2).sum(dim=-1) / xf.pow(2).sum(dim=-1).clamp_min(eps)


def pattern_select_v_candidate(
    x: torch.Tensor,
    centroids: torch.Tensor,
    *,
    value_objective: str,
    group_size: int,
    bits: int,
    tie_atol: float = 1e-7,
    block_tokens: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    objective = normalize_value_objective(value_objective)
    if objective == "base":
        idx = pattern_nearest_v_centroid(x, centroids).to(torch.long)
        selected = pattern_gather_centroids(idx, centroids).to(x.dtype)
        _, mask = pattern_v_threshold_and_mask(x, selected)
        return idx, mask, {"base_score": torch.empty(0, device=x.device, dtype=x.dtype)}
    if x.shape[2] > block_tokens:
        idx_parts = []
        mask_parts = []
        for start in range(0, x.shape[2], block_tokens):
            stop = min(start + block_tokens, x.shape[2])
            idx_part, mask_part, _ = pattern_select_v_candidate(
                x[:, :, start:stop, :].contiguous(),
                centroids,
                value_objective=objective,
                group_size=group_size,
                bits=bits,
                tie_atol=tie_atol,
                block_tokens=block_tokens,
            )
            idx_parts.append(idx_part)
            mask_parts.append(mask_part)
        return torch.cat(idx_parts, dim=2), torch.cat(mask_parts, dim=2), {"score": torch.empty(0, device=x.device, dtype=torch.float32)}
    restored, masks, base_score = pattern_v_candidate_reconstructions(x, centroids, group_size=group_size, bits=bits)
    expanded_x = x.unsqueeze(2).expand_as(restored)
    direction = _vector_direction_error(expanded_x, restored)
    nre = _vector_nre(expanded_x, restored)
    score = direction if objective == "v_dir" else direction + nre
    best = score.min(dim=2).values
    eligible = score <= best.unsqueeze(2) + float(tie_atol)
    masked_base = torch.where(eligible, base_score.float(), torch.full_like(base_score.float(), float("inf")))
    idx = masked_base.argmin(dim=2).contiguous().to(torch.long)
    mask = torch.gather(masks, 2, idx.unsqueeze(2)).squeeze(2).contiguous()
    return idx, mask, {"score": score, "direction": direction, "nre": nre, "base_score": base_score}


def quantize_pack_k_reference(k: torch.Tensor, group_size: int, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if k.shape[2] % group_size != 0:
        raise ValueError(f"K token length {k.shape[2]} must be divisible by group_size={group_size}")
    if k.is_cuda:
        return triton_quantize_and_pack_along_last_dim(k.transpose(2, 3).contiguous(), group_size, bits)
    feat_per_int = 32 // bits
    legal_multiple = lcm(group_size, feat_per_int)
    pad_tokens = (-k.shape[2]) % legal_multiple
    if pad_tokens:
        pad = torch.zeros(k.shape[0], k.shape[1], pad_tokens, k.shape[3], dtype=k.dtype, device=k.device)
        k = torch.cat([k, pad], dim=2)
    transposed = k.transpose(2, 3).contiguous()
    bsz, heads, dim, tokens = transposed.shape
    levels = float((1 << bits) - 1)
    grouped = transposed.reshape(bsz, heads, dim, tokens // group_size, group_size)
    zero = grouped.amin(dim=-1)
    mx = grouped.amax(dim=-1)
    scale = ((mx - zero) / levels).clamp_min(torch.finfo(transposed.dtype).eps)
    q = torch.round((grouped - zero.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp_(0, levels).to(torch.int32)
    q = q.reshape(bsz, heads, dim, tokens)
    return pack_tensor(q, bits, pack_dim=3), scale.contiguous(), zero.contiguous()


def quantize_pack_v_reference(v: torch.Tensor, group_size: int, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if v.shape[-1] % group_size != 0:
        raise ValueError(f"V head_dim {v.shape[-1]} must be divisible by group_size={group_size}")
    if v.is_cuda:
        return triton_quantize_and_pack_along_last_dim(v.contiguous(), group_size, bits)
    bsz, heads, tokens, dim = v.shape
    levels = float((1 << bits) - 1)
    grouped = v.reshape(bsz, heads, tokens, dim // group_size, group_size)
    zero = grouped.amin(dim=-1)
    mx = grouped.amax(dim=-1)
    scale = ((mx - zero) / levels).clamp_min(torch.finfo(v.dtype).eps)
    q = torch.round((grouped - zero.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp_(0, levels).to(torch.int32)
    q = q.reshape(bsz, heads, tokens, dim)
    return pack_tensor(q, bits, pack_dim=3), scale.contiguous(), zero.contiguous()


def dequantize_k_reference(packed: torch.Tensor | None, scale: torch.Tensor | None, zero: torch.Tensor | None, group_size: int, bits: int) -> torch.Tensor | None:
    if packed is None:
        return None
    if scale is None or zero is None:
        raise ValueError("packed K requires scale and zero")
    q = unpack_tensor(packed, bits, pack_dim=3).to(scale.dtype)
    bsz, heads, dim, tokens = q.shape
    grouped = q.reshape(bsz, heads, dim, tokens // group_size, group_size)
    out = grouped * scale.unsqueeze(-1) + zero.unsqueeze(-1)
    return out.reshape(bsz, heads, dim, tokens).transpose(2, 3).contiguous()


def dequantize_v_reference(packed: torch.Tensor | None, scale: torch.Tensor | None, zero: torch.Tensor | None, group_size: int, bits: int) -> torch.Tensor | None:
    if packed is None:
        return None
    if scale is None or zero is None:
        raise ValueError("packed V requires scale and zero")
    q = unpack_tensor(packed, bits, pack_dim=3).to(scale.dtype)
    bsz, heads, tokens, dim = q.shape
    grouped = q.reshape(bsz, heads, tokens, dim // group_size, group_size)
    out = grouped * scale.unsqueeze(-1) + zero.unsqueeze(-1)
    return out.reshape(bsz, heads, tokens, dim).contiguous()


def _pack_raw_pending(cache: QuantizedKVCache, tokens: int) -> None:
    to_pack = cache.pending_k[:, :, :tokens, :].contiguous()
    packed, scale, zero = quantize_pack_k_reference(to_pack, cache.group_size, cache.k_bits)
    _cat_packed_k(cache, packed, scale, zero, tokens)
    cache.pending_k = cache.pending_k[:, :, tokens:, :].contiguous() if tensor_tokens(cache.pending_k) > tokens else None
    if cache.pending_v is None or tensor_tokens(cache.pending_v) < tokens:
        raise ValueError("V pending must cover the same prefix as K pending")
    value_to_pack = cache.pending_v[:, :, :tokens, :].contiguous()
    packed_v, scale_v, zero_v = quantize_pack_v_reference(value_to_pack, cache.group_size, cache.v_bits)
    _cat_packed_v(cache, packed_v, scale_v, zero_v, tokens)
    cache.pending_v = cache.pending_v[:, :, tokens:, :].contiguous() if tensor_tokens(cache.pending_v) > tokens else None


def _append_dynamic_centroids(cache: PatternQuantizedKVCache, k_window: torch.Tensor, v_window: torch.Tensor) -> None:
    bsz, heads, tokens, dim = k_window.shape
    xk = k_window.permute(1, 0, 2, 3).reshape(heads, bsz * tokens, dim).contiguous()
    xv = v_window.permute(1, 0, 2, 3).reshape(heads, bsz * tokens, dim).contiguous()
    k_centroid = pattern_chebyshev_center_per_head(xk).to(cache.k_centroids.dtype)
    v_centroid = pattern_chebyshev_center_per_head(xv).to(cache.v_centroids.dtype)
    cache.k_centroids = torch.cat([cache.k_centroids, k_centroid], dim=1).contiguous()
    cache.v_centroids = torch.cat([cache.v_centroids, v_centroid], dim=1).contiguous()
    cache.centroid_updates_k += 1
    cache.centroid_updates_v += 1


def _pack_pattern_window(
    cache: PatternQuantizedKVCache,
    tokens: int,
    *,
    k_assignments: torch.Tensor | None = None,
    v_assignment_idx: torch.Tensor | None = None,
    v_pattern_mask: torch.Tensor | None = None,
    dynamic_update: bool,
) -> None:
    if cache.pending_k is None or cache.pending_v is None:
        return
    if cache.k_centroids is None or cache.v_centroids is None:
        _pack_raw_pending(cache, tokens)
        return
    k_window = cache.pending_k[:, :, :tokens, :].contiguous()
    v_window = cache.pending_v[:, :, :tokens, :].contiguous()
    if dynamic_update:
        _append_dynamic_centroids(cache, k_window, v_window)
    bsz, heads, window_tokens, dim = k_window.shape
    if k_assignments is None:
        xk = k_window.permute(1, 0, 2, 3).reshape(heads, bsz * window_tokens, dim).contiguous()
        assign_hn = _assign_minmax_hnk(xk, cache.k_centroids)
        k_assignments = assign_hn.view(heads, bsz, window_tokens).permute(1, 0, 2).contiguous().to(torch.long)
    else:
        k_assignments = k_assignments[:, :, :tokens].contiguous().to(torch.long)
    if v_assignment_idx is None:
        v_assignment_idx, inferred_v_pattern_mask, _ = pattern_select_v_candidate(
            v_window,
            cache.v_centroids,
            value_objective=getattr(cache, "value_objective", "base"),
            group_size=cache.group_size,
            bits=cache.v_bits,
        )
        if v_pattern_mask is None:
            v_pattern_mask = inferred_v_pattern_mask
    else:
        v_assignment_idx = v_assignment_idx[:, :, :tokens].contiguous().to(torch.long)
    k_centroid_per_token = pattern_gather_centroids(k_assignments, cache.k_centroids).to(k_window.dtype)
    v_centroid_per_token = pattern_gather_centroids(v_assignment_idx, cache.v_centroids).to(v_window.dtype)
    if v_pattern_mask is None:
        _, v_pattern_mask = pattern_v_threshold_and_mask(v_window, v_centroid_per_token)
    else:
        v_pattern_mask = v_pattern_mask[:, :, :tokens].contiguous().bool()
    k_residual = k_window - k_centroid_per_token
    v_adjusted = v_window - v_pattern_mask.unsqueeze(-1).to(v_window.dtype) * v_centroid_per_token
    if getattr(cache, "trace_layer_idx", None) is not None:
        try:
            from bench.patternkv_equivalence_reference import save_assignment_trace

            save_assignment_trace(
                mode=str(getattr(cache, "cache_mode", "segmented")),
                layer_idx=int(getattr(cache, "trace_layer_idx")),
                decode_position=int(os.environ.get("PATTERNKV_EQUIV_TRACE_DECODE_POS", "-1")),
                k_window=k_window,
                v_window=v_window,
                k_centroids=cache.k_centroids,
                k_assignments=k_assignments,
                v_centroids=cache.v_centroids,
                v_assignment_idx=v_assignment_idx,
                v_gate=v_pattern_mask,
            )
        except Exception:
            if os.environ.get("PATTERNKV_EQUIV_TRACE_STRICT") == "1":
                raise
    packed_k, scale_k, zero_k = quantize_pack_k_reference(k_residual, cache.group_size, cache.k_bits)
    packed_v, scale_v, zero_v = quantize_pack_v_reference(v_adjusted, cache.group_size, cache.v_bits)
    _cat_packed_k(cache, packed_k, scale_k, zero_k, tokens)
    _cat_packed_v(cache, packed_v, scale_v, zero_v, tokens)
    cache.k_assignments = _cat_assignment(cache.k_assignments, k_assignments)
    cache.v_assignment_idx = _cat_assignment(cache.v_assignment_idx, v_assignment_idx)
    mask_u8 = v_pattern_mask.to(torch.uint8)
    cache.v_pattern_mask = _cat_assignment(cache.v_pattern_mask, mask_u8)
    cache.v_assignments = cache.v_pattern_mask
    cache.pending_k = cache.pending_k[:, :, tokens:, :].contiguous() if tensor_tokens(cache.pending_k) > tokens else None
    cache.pending_v = cache.pending_v[:, :, tokens:, :].contiguous() if tensor_tokens(cache.pending_v) > tokens else None


def flush_pending(
    cache: QuantizedKVCache,
    *,
    k_assignments: torch.Tensor | None = None,
    v_assignment_idx: torch.Tensor | None = None,
    v_pattern_mask: torch.Tensor | None = None,
    dynamic_update: bool = True,
) -> None:
    pending_k_tokens = tensor_tokens(cache.pending_k)
    k_pack_tokens = (pending_k_tokens // cache.group_size) * cache.group_size
    if not k_pack_tokens:
        return
    if not isinstance(cache, PatternQuantizedKVCache):
        _pack_raw_pending(cache, k_pack_tokens)
        return
    if k_assignments is not None or v_assignment_idx is not None or v_pattern_mask is not None:
        _pack_pattern_window(
            cache,
            k_pack_tokens,
            k_assignments=k_assignments,
            v_assignment_idx=v_assignment_idx,
            v_pattern_mask=v_pattern_mask,
            dynamic_update=False,
        )
        return
    while tensor_tokens(cache.pending_k) >= cache.group_size:
        _pack_pattern_window(cache, cache.group_size, dynamic_update=dynamic_update)


def flush_chunked_buffer(
    cache: QuantizedKVCache,
    *,
    k_assignments: torch.Tensor | None = None,
    v_assignment_idx: torch.Tensor | None = None,
    v_pattern_mask: torch.Tensor | None = None,
    dynamic_update: bool = True,
) -> None:
    chunk_tokens = int(getattr(cache, "chunk_length", cache.group_size) or cache.group_size)
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_length must be positive, got {chunk_tokens}")
    if chunk_tokens % cache.group_size != 0:
        raise ValueError(f"chunk_length={chunk_tokens} must be divisible by group_size={cache.group_size}")
    while tensor_tokens(cache.pending_k) >= chunk_tokens:
        if isinstance(cache, PatternQuantizedKVCache):
            _pack_pattern_window(
                cache,
                chunk_tokens,
                k_assignments=k_assignments,
                v_assignment_idx=v_assignment_idx,
                v_pattern_mask=v_pattern_mask,
                dynamic_update=dynamic_update and k_assignments is None and v_assignment_idx is None and v_pattern_mask is None,
            )
            if k_assignments is not None:
                k_assignments = k_assignments[:, :, chunk_tokens:].contiguous() if tensor_tokens(k_assignments) > chunk_tokens else None
            if v_assignment_idx is not None:
                v_assignment_idx = v_assignment_idx[:, :, chunk_tokens:].contiguous() if tensor_tokens(v_assignment_idx) > chunk_tokens else None
            if v_pattern_mask is not None:
                v_pattern_mask = v_pattern_mask[:, :, chunk_tokens:].contiguous() if tensor_tokens(v_pattern_mask) > chunk_tokens else None
        else:
            _pack_raw_pending(cache, chunk_tokens)


def build_cache_from_prefill(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    sink_length: int,
    recent_length: int,
    group_size: int,
    k_bits: int,
    v_bits: int,
    pattern: bool = False,
    k_centroids: torch.Tensor | None = None,
    v_centroids: torch.Tensor | None = None,
    k_assignments: torch.Tensor | None = None,
    v_assignment_idx: torch.Tensor | None = None,
    v_pattern_mask: torch.Tensor | None = None,
    cache_mode: str = ROLLING_CACHE_MODE,
    chunk_length: int | None = None,
    value_objective: str = "base",
) -> QuantizedKVCache:
    cache_mode = normalize_cache_mode(cache_mode)
    cache_cls = PatternQuantizedKVCache if pattern else QuantizedKVCache
    cache = cache_cls(
        total_tokens=int(key_states.shape[2]),
        sink_length=int(sink_length),
        recent_length=0 if cache_mode == CHUNKED_CACHE_MODE else int(recent_length),
        group_size=int(group_size),
        k_bits=int(k_bits),
        v_bits=int(v_bits),
        cache_mode=cache_mode,
        chunk_length=int(chunk_length if chunk_length is not None else group_size),
    )
    total = cache.total_tokens
    sink_end = 0 if cache_mode == CHUNKED_CACHE_MODE else min(total, sink_length)
    recent_start = total if cache_mode == CHUNKED_CACHE_MODE else max(sink_end, total - recent_length)
    cache.sink_k = _empty_like_tokens(key_states, sink_end)
    cache.sink_v = _empty_like_tokens(value_states, sink_end)
    history_k = key_states[:, :, sink_end:recent_start, :].contiguous()
    history_v = value_states[:, :, sink_end:recent_start, :].contiguous()
    cache.recent_k = key_states[:, :, recent_start:, :].contiguous() if recent_start < total else None
    cache.recent_v = value_states[:, :, recent_start:, :].contiguous() if recent_start < total else None
    cache.pending_k = history_k if history_k.shape[2] else None
    cache.pending_v = history_v if history_v.shape[2] else None
    if isinstance(cache, PatternQuantizedKVCache):
        cache.k_centroids = k_centroids
        cache.v_centroids = v_centroids
        cache.value_objective = normalize_value_objective(value_objective)
        history_k_assignments = k_assignments[:, :, sink_end:recent_start].contiguous() if k_assignments is not None and recent_start > sink_end else None
        history_v_assignment_idx = v_assignment_idx[:, :, sink_end:recent_start].contiguous() if v_assignment_idx is not None and recent_start > sink_end else None
        history_v_pattern_mask = v_pattern_mask[:, :, sink_end:recent_start].contiguous() if v_pattern_mask is not None and recent_start > sink_end else None
        if cache.cache_mode == CHUNKED_CACHE_MODE:
            flush_chunked_buffer(
                cache,
                k_assignments=history_k_assignments,
                v_assignment_idx=history_v_assignment_idx,
                v_pattern_mask=history_v_pattern_mask,
                dynamic_update=False,
            )
        else:
            flush_pending(
                cache,
                k_assignments=history_k_assignments,
                v_assignment_idx=history_v_assignment_idx,
                v_pattern_mask=history_v_pattern_mask,
                dynamic_update=False,
            )
    else:
        if cache.cache_mode == CHUNKED_CACHE_MODE:
            flush_chunked_buffer(cache)
        else:
            flush_pending(cache)
    validate_cache(cache)
    return cache


def append_decode_rolling(cache: QuantizedKVCache, key_states: torch.Tensor, value_states: torch.Tensor) -> QuantizedKVCache:
    append_tokens = int(key_states.shape[2])
    sink_capacity = max(int(cache.sink_length) - tensor_tokens(cache.sink_k), 0)
    sink_fill = min(sink_capacity, append_tokens)
    if sink_fill:
        cache.sink_k = _cat_token(cache.sink_k, key_states[:, :, :sink_fill, :].contiguous())
        cache.sink_v = _cat_token(cache.sink_v, value_states[:, :, :sink_fill, :].contiguous())
    if sink_fill < append_tokens:
        cache.recent_k = _cat_token(cache.recent_k, key_states[:, :, sink_fill:, :].contiguous())
        cache.recent_v = _cat_token(cache.recent_v, value_states[:, :, sink_fill:, :].contiguous())
    cache.total_tokens += int(key_states.shape[2])
    overflow = max(tensor_tokens(cache.recent_k) - cache.recent_length, 0)
    if overflow:
        cache.pending_k = _cat_token(cache.pending_k, cache.recent_k[:, :, :overflow, :].contiguous())
        cache.pending_v = _cat_token(cache.pending_v, cache.recent_v[:, :, :overflow, :].contiguous())
        cache.recent_k = cache.recent_k[:, :, overflow:, :].contiguous()
        cache.recent_v = cache.recent_v[:, :, overflow:, :].contiguous()
    flush_pending(cache)
    if isinstance(cache, PatternQuantizedKVCache):
        reference = cache.sink_k
        if reference is None:
            reference = cache.recent_k
        if reference is None:
            reference = cache.pending_k
        if reference is not None:
            bsz, heads = reference.shape[0], reference.shape[1]
            device = reference.device
            if cache.packed_k_tokens and (cache.k_assignments is None or cache.k_assignments.shape[2] != cache.packed_k_tokens):
                cache.k_assignments = torch.zeros(bsz, heads, cache.packed_k_tokens, dtype=torch.long, device=device)
            if cache.packed_v_tokens and (cache.v_assignment_idx is None or cache.v_assignment_idx.shape[2] != cache.packed_v_tokens):
                cache.v_assignment_idx = torch.zeros(bsz, heads, cache.packed_v_tokens, dtype=torch.long, device=device)
                cache.v_assignments = torch.zeros(bsz, heads, cache.packed_v_tokens, dtype=torch.uint8, device=device)
    validate_cache(cache)
    return cache


def append_decode_chunked(cache: QuantizedKVCache, key_states: torch.Tensor, value_states: torch.Tensor) -> QuantizedKVCache:
    append_decode_chunked_buffer_only(cache, key_states, value_states)
    flush_chunked_buffer(cache)
    validate_cache(cache)
    return cache


def append_decode_chunked_buffer_only(cache: QuantizedKVCache, key_states: torch.Tensor, value_states: torch.Tensor) -> QuantizedKVCache:
    if tensor_tokens(cache.sink_k) or tensor_tokens(cache.recent_k):
        raise ValueError("chunked cache must not contain sink or rolling recent tokens")
    cache.pending_k = _cat_token(cache.pending_k, key_states)
    cache.pending_v = _cat_token(cache.pending_v, value_states)
    cache.total_tokens += int(key_states.shape[2])
    return cache


def append_decode(cache: QuantizedKVCache, key_states: torch.Tensor, value_states: torch.Tensor) -> QuantizedKVCache:
    mode = normalize_cache_mode(getattr(cache, "cache_mode", ROLLING_CACHE_MODE))
    if mode == CHUNKED_CACHE_MODE:
        return append_decode_chunked(cache, key_states, value_states)
    return append_decode_rolling(cache, key_states, value_states)


def reconstruct_full_k(cache: QuantizedKVCache) -> torch.Tensor | None:
    packed_k = dequantize_k_reference(cache.packed_k, cache.packed_k_scale, cache.packed_k_zero, cache.group_size, cache.k_bits)
    if packed_k is not None:
        packed_k = packed_k[:, :, : cache.packed_k_tokens, :].contiguous()
        if isinstance(cache, PatternQuantizedKVCache) and cache.k_centroids is not None and cache.k_assignments is not None:
            packed_k = packed_k + pattern_gather_centroids(cache.k_assignments[:, :, : cache.packed_k_tokens], cache.k_centroids).to(packed_k.dtype)
    parts = [
        cache.sink_k,
        packed_k,
        cache.pending_k,
        cache.recent_k,
    ]
    parts = [part for part in parts if torch.is_tensor(part)]
    return torch.cat(parts, dim=2).contiguous() if parts else None


def reconstruct_full_v(cache: QuantizedKVCache) -> torch.Tensor | None:
    packed_v = dequantize_v_reference(cache.packed_v, cache.packed_v_scale, cache.packed_v_zero, cache.group_size, cache.v_bits)
    if packed_v is not None:
        packed_v = packed_v[:, :, : cache.packed_v_tokens, :].contiguous()
        if isinstance(cache, PatternQuantizedKVCache) and cache.v_centroids is not None and cache.v_assignment_idx is not None:
            mask = cache.v_pattern_mask if cache.v_pattern_mask is not None else cache.v_assignments
            if mask is not None:
                centroids = pattern_gather_centroids(cache.v_assignment_idx[:, :, : cache.packed_v_tokens], cache.v_centroids).to(packed_v.dtype)
                packed_v = packed_v + mask[:, :, : cache.packed_v_tokens].unsqueeze(-1).to(packed_v.dtype) * centroids
    parts = [
        cache.sink_v,
        packed_v,
        cache.pending_v,
        cache.recent_v,
    ]
    parts = [part for part in parts if torch.is_tensor(part)]
    return torch.cat(parts, dim=2).contiguous() if parts else None


def cache_segment_stats(cache: QuantizedKVCache | None) -> dict[str, int | None]:
    if cache is None:
        return {
            "sink_tokens": 0,
            "packed_history_tokens": 0,
            "pending_history_tokens": 0,
            "recent_tokens": 0,
            "total_tokens": 0,
            "k_assignment_tokens": None,
            "v_assignment_tokens": None,
        }
    k_assignments = getattr(cache, "k_assignments", None)
    v_assignment_idx = getattr(cache, "v_assignment_idx", None)
    v_pattern_mask = getattr(cache, "v_pattern_mask", None)
    if v_pattern_mask is None:
        v_pattern_mask = getattr(cache, "v_assignments", None)
    return {
        "sink_tokens": tensor_tokens(cache.sink_k),
        "packed_history_tokens": int(cache.packed_k_tokens),
        "pending_history_tokens": tensor_tokens(cache.pending_k),
        "recent_tokens": tensor_tokens(cache.recent_k),
        "chunk_tokens": tensor_tokens(cache.pending_k) if getattr(cache, "cache_mode", ROLLING_CACHE_MODE) == CHUNKED_CACHE_MODE else 0,
        "total_tokens": int(cache.total_tokens),
        "cache_mode": getattr(cache, "cache_mode", ROLLING_CACHE_MODE),
        "chunk_length": int(getattr(cache, "chunk_length", cache.group_size)),
        "k_assignment_tokens": tensor_tokens(k_assignments) if torch.is_tensor(k_assignments) else None,
        "v_assignment_tokens": tensor_tokens(v_assignment_idx) if torch.is_tensor(v_assignment_idx) else None,
        "v_pattern_mask_tokens": tensor_tokens(v_pattern_mask) if torch.is_tensor(v_pattern_mask) else None,
    }


def validate_cache(cache: QuantizedKVCache) -> None:
    cache.cache_mode = normalize_cache_mode(getattr(cache, "cache_mode", ROLLING_CACHE_MODE))
    if not int(getattr(cache, "chunk_length", 0) or 0):
        cache.chunk_length = int(cache.group_size)
    sink_tokens = tensor_tokens(cache.sink_k)
    recent_tokens = tensor_tokens(cache.recent_k)
    pending_tokens = tensor_tokens(cache.pending_k)
    if cache.cache_mode == CHUNKED_CACHE_MODE:
        if sink_tokens or recent_tokens or cache.sink_length or cache.recent_length:
            raise ValueError("chunked cache requires empty sink/recent and zero sink/recent lengths")
        if pending_tokens >= cache.chunk_length:
            raise ValueError(f"chunked pending buffer not flushed: {pending_tokens} >= {cache.chunk_length}")
        if cache.packed_k_tokens != (cache.total_tokens // cache.chunk_length) * cache.chunk_length:
            raise ValueError("chunked packed token cadence mismatch")
        if pending_tokens != cache.total_tokens % cache.chunk_length:
            raise ValueError("chunked buffer token cadence mismatch")
    if sink_tokens > cache.sink_length:
        raise ValueError(f"sink exceeds configured length: {sink_tokens} > {cache.sink_length}")
    if recent_tokens > cache.recent_length:
        raise ValueError(f"recent exceeds configured length: {recent_tokens} > {cache.recent_length}")
    if tensor_tokens(cache.sink_v) != sink_tokens:
        raise ValueError("sink K/V token mismatch")
    if tensor_tokens(cache.pending_v) != pending_tokens:
        raise ValueError("pending K/V token mismatch")
    if tensor_tokens(cache.recent_v) != recent_tokens:
        raise ValueError("recent K/V token mismatch")
    if cache.packed_k_tokens != cache.packed_v_tokens:
        raise ValueError(f"packed K/V token mismatch: {cache.packed_k_tokens} != {cache.packed_v_tokens}")
    counted = sink_tokens + cache.packed_k_tokens + pending_tokens + recent_tokens
    if counted != cache.total_tokens:
        raise ValueError(f"cache token conservation failed: counted={counted}, total={cache.total_tokens}")
    if cache.cache_mode != CHUNKED_CACHE_MODE:
        expected = segment_lengths(cache.total_tokens, cache.sink_length, cache.recent_length)
        if sink_tokens != expected["sink_tokens"]:
            raise ValueError(f"sink token count mismatch: {sink_tokens} != {expected['sink_tokens']}")
        if recent_tokens != expected["recent_tokens"]:
            raise ValueError(f"recent token count mismatch: {recent_tokens} != {expected['recent_tokens']}")
        if cache.packed_k_tokens + pending_tokens != expected["quantized_history_tokens"]:
            raise ValueError("history token count mismatch")
    if isinstance(cache, PatternQuantizedKVCache):
        if cache.v_pattern_mask is None and cache.v_assignments is not None:
            cache.v_pattern_mask = cache.v_assignments
        if cache.v_assignments is None and cache.v_pattern_mask is not None:
            cache.v_assignments = cache.v_pattern_mask
        assignment_tokens = tensor_tokens(cache.k_assignments)
        if assignment_tokens not in (0, cache.packed_k_tokens):
            raise ValueError(f"Pattern K assignment tokens must match packed history: {assignment_tokens} != {cache.packed_k_tokens}")
        v_assignment_tokens = tensor_tokens(cache.v_assignment_idx)
        if v_assignment_tokens not in (0, cache.packed_v_tokens):
            raise ValueError(f"Pattern V assignment tokens must match packed history: {v_assignment_tokens} != {cache.packed_v_tokens}")
        v_mask_tokens = tensor_tokens(cache.v_pattern_mask)
        if v_mask_tokens not in (0, cache.packed_v_tokens):
            raise ValueError(f"Pattern V gate tokens must match packed history: {v_mask_tokens} != {cache.packed_v_tokens}")
        if torch.is_tensor(cache.k_centroids):
            if cache.k_centroids.dim() != 3:
                raise ValueError(f"K centroids must be [kv_heads, centroids, head_dim], got {tuple(cache.k_centroids.shape)}")
            if cache.k_assignments is not None and cache.k_assignments.shape[1] != cache.k_centroids.shape[0]:
                raise ValueError("K assignment KV heads must match K centroid heads")
            if cache.k_assignments is not None and cache.k_assignments.numel() and int(cache.k_assignments.max().item()) >= cache.k_centroids.shape[1]:
                raise ValueError("K assignment index exceeds K centroid bank")
        if torch.is_tensor(cache.v_centroids):
            if cache.v_centroids.dim() != 3:
                raise ValueError(f"V centroids must be [kv_heads, centroids, head_dim], got {tuple(cache.v_centroids.shape)}")
            if cache.v_assignment_idx is not None and cache.v_assignment_idx.shape[1] != cache.v_centroids.shape[0]:
                raise ValueError("V assignment KV heads must match V centroid heads")
            if cache.v_assignment_idx is not None and cache.v_assignment_idx.numel() and int(cache.v_assignment_idx.max().item()) >= cache.v_centroids.shape[1]:
                raise ValueError("V assignment index exceeds V centroid bank")


def serialize_cache(cache: QuantizedKVCache) -> tuple[Any, ...]:
    base = (
        "patternkv_segmented_cache_v1" if isinstance(cache, PatternQuantizedKVCache) else "quantized_segmented_cache_v1",
        cache.sink_k,
        cache.sink_v,
        cache.packed_k,
        cache.packed_k_scale,
        cache.packed_k_zero,
        cache.packed_v,
        cache.packed_v_scale,
        cache.packed_v_zero,
        cache.pending_k,
        cache.pending_v,
        cache.recent_k,
        cache.recent_v,
        int(cache.total_tokens),
        int(cache.packed_k_tokens),
        int(cache.packed_v_tokens),
        int(cache.sink_length),
        int(cache.recent_length),
        int(cache.group_size),
        int(cache.k_bits),
        int(cache.v_bits),
        int(cache.pack_count_k),
        int(cache.pack_count_v),
    )
    if isinstance(cache, PatternQuantizedKVCache):
        return base + (
            cache.k_assignments,
            cache.v_assignments,
            cache.v_assignment_idx,
            cache.v_pattern_mask,
            cache.k_centroids,
            cache.v_centroids,
            int(cache.centroid_updates_k),
            int(cache.centroid_updates_v),
            cache.cache_mode,
            int(cache.chunk_length),
            cache.value_objective,
        )
    return base + (cache.cache_mode, int(cache.chunk_length))


def deserialize_cache(value: Any, *, pattern: bool = False) -> QuantizedKVCache:
    if isinstance(value, QuantizedKVCache):
        return value
    if not isinstance(value, tuple) or not value:
        raise TypeError("cache must be a segmented cache tuple")
    tag = value[0]
    if tag not in ("quantized_segmented_cache_v1", "patternkv_segmented_cache_v1"):
        raise TypeError(f"unsupported cache tag: {tag!r}")
    cls = PatternQuantizedKVCache if tag == "patternkv_segmented_cache_v1" or pattern else QuantizedKVCache
    cache = cls(
        sink_k=value[1],
        sink_v=value[2],
        packed_k=value[3],
        packed_k_scale=value[4],
        packed_k_zero=value[5],
        packed_v=value[6],
        packed_v_scale=value[7],
        packed_v_zero=value[8],
        pending_k=value[9],
        pending_v=value[10],
        recent_k=value[11],
        recent_v=value[12],
        total_tokens=int(value[13]),
        packed_k_tokens=int(value[14]),
        packed_v_tokens=int(value[15]),
        sink_length=int(value[16]),
        recent_length=int(value[17]),
        group_size=int(value[18]),
        k_bits=int(value[19]),
        v_bits=int(value[20]),
        pack_count_k=int(value[21]),
        pack_count_v=int(value[22]),
        cache_mode=ROLLING_CACHE_MODE,
        chunk_length=int(value[18]),
    )
    if not isinstance(cache, PatternQuantizedKVCache) and len(value) >= 25 and isinstance(value[23], str):
        cache.cache_mode = normalize_cache_mode(value[23])
        cache.chunk_length = int(value[24])
    pattern_offset = 23
    if isinstance(cache, PatternQuantizedKVCache) and len(value) >= pattern_offset + 7:
        cache.k_assignments = value[pattern_offset]
        cache.v_assignments = value[pattern_offset + 1]
        cache.v_assignment_idx = value[pattern_offset + 2]
        if len(value) >= pattern_offset + 8:
            cache.v_pattern_mask = value[pattern_offset + 3]
            cache.k_centroids = value[pattern_offset + 4]
            cache.v_centroids = value[pattern_offset + 5]
            cache.centroid_updates_k = int(value[pattern_offset + 6])
            cache.centroid_updates_v = int(value[pattern_offset + 7])
        else:
            cache.v_pattern_mask = cache.v_assignments
            cache.k_centroids = value[pattern_offset + 3]
            cache.v_centroids = value[pattern_offset + 4]
            cache.centroid_updates_k = int(value[pattern_offset + 5])
            cache.centroid_updates_v = int(value[pattern_offset + 6])
        if len(value) >= pattern_offset + 10 and isinstance(value[pattern_offset + 8], str):
            cache.cache_mode = normalize_cache_mode(value[pattern_offset + 8])
            cache.chunk_length = int(value[pattern_offset + 9])
        if len(value) >= pattern_offset + 11 and isinstance(value[pattern_offset + 10], str):
            cache.value_objective = normalize_value_objective(value[pattern_offset + 10])
    validate_cache(cache)
    return cache


def maybe_validate_cache(cache: QuantizedKVCache) -> None:
    if cache_validate_enabled():
        validate_cache(cache)
