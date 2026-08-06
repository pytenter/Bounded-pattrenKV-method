from __future__ import annotations

import os
from dataclasses import dataclass
from math import lcm
from typing import Any

import torch

from quant.new_pack import pack_tensor, unpack_tensor


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


@dataclass
class PatternQuantizedKVCache(QuantizedKVCache):
    k_assignments: torch.Tensor | None = None
    v_assignments: torch.Tensor | None = None
    v_assignment_idx: torch.Tensor | None = None
    k_centroids: torch.Tensor | None = None
    v_centroids: torch.Tensor | None = None
    centroid_updates_k: int = 0
    centroid_updates_v: int = 0


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


def quantize_pack_k_reference(k: torch.Tensor, group_size: int, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if k.shape[2] % group_size != 0:
        raise ValueError(f"K token length {k.shape[2]} must be divisible by group_size={group_size}")
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


def flush_pending(cache: QuantizedKVCache) -> None:
    pending_k_tokens = tensor_tokens(cache.pending_k)
    k_pack_tokens = (pending_k_tokens // cache.group_size) * cache.group_size
    if k_pack_tokens:
        to_pack = cache.pending_k[:, :, :k_pack_tokens, :].contiguous()
        packed, scale, zero = quantize_pack_k_reference(to_pack, cache.group_size, cache.k_bits)
        _cat_packed_k(cache, packed, scale, zero, k_pack_tokens)
        cache.pending_k = cache.pending_k[:, :, k_pack_tokens:, :].contiguous() if pending_k_tokens > k_pack_tokens else None
        if cache.pending_v is None or tensor_tokens(cache.pending_v) < k_pack_tokens:
            raise ValueError("V pending must cover the same prefix as K pending")
        value_to_pack = cache.pending_v[:, :, :k_pack_tokens, :].contiguous()
        packed_v, scale_v, zero_v = quantize_pack_v_reference(value_to_pack, cache.group_size, cache.v_bits)
        _cat_packed_v(cache, packed_v, scale_v, zero_v, k_pack_tokens)
        cache.pending_v = cache.pending_v[:, :, k_pack_tokens:, :].contiguous() if tensor_tokens(cache.pending_v) > k_pack_tokens else None


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
) -> QuantizedKVCache:
    cache_cls = PatternQuantizedKVCache if pattern else QuantizedKVCache
    cache = cache_cls(
        total_tokens=int(key_states.shape[2]),
        sink_length=int(sink_length),
        recent_length=int(recent_length),
        group_size=int(group_size),
        k_bits=int(k_bits),
        v_bits=int(v_bits),
    )
    total = cache.total_tokens
    sink_end = min(total, sink_length)
    recent_start = max(sink_end, total - recent_length)
    cache.sink_k = _empty_like_tokens(key_states, sink_end)
    cache.sink_v = _empty_like_tokens(value_states, sink_end)
    history_k = key_states[:, :, sink_end:recent_start, :].contiguous()
    history_v = value_states[:, :, sink_end:recent_start, :].contiguous()
    cache.recent_k = key_states[:, :, recent_start:, :].contiguous() if recent_start < total else None
    cache.recent_v = value_states[:, :, recent_start:, :].contiguous() if recent_start < total else None
    cache.pending_k = history_k if history_k.shape[2] else None
    cache.pending_v = history_v if history_v.shape[2] else None
    flush_pending(cache)
    validate_cache(cache)
    return cache


def append_decode(cache: QuantizedKVCache, key_states: torch.Tensor, value_states: torch.Tensor) -> QuantizedKVCache:
    cache.recent_k = _cat_token(cache.recent_k, key_states)
    cache.recent_v = _cat_token(cache.recent_v, value_states)
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


def reconstruct_full_k(cache: QuantizedKVCache) -> torch.Tensor | None:
    packed_k = dequantize_k_reference(cache.packed_k, cache.packed_k_scale, cache.packed_k_zero, cache.group_size, cache.k_bits)
    if packed_k is not None:
        packed_k = packed_k[:, :, : cache.packed_k_tokens, :].contiguous()
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
    return {
        "sink_tokens": tensor_tokens(cache.sink_k),
        "packed_history_tokens": int(cache.packed_k_tokens),
        "pending_history_tokens": tensor_tokens(cache.pending_k),
        "recent_tokens": tensor_tokens(cache.recent_k),
        "total_tokens": int(cache.total_tokens),
        "k_assignment_tokens": tensor_tokens(k_assignments) if torch.is_tensor(k_assignments) else None,
        "v_assignment_tokens": tensor_tokens(v_assignment_idx) if torch.is_tensor(v_assignment_idx) else None,
    }


def validate_cache(cache: QuantizedKVCache) -> None:
    sink_tokens = tensor_tokens(cache.sink_k)
    recent_tokens = tensor_tokens(cache.recent_k)
    pending_tokens = tensor_tokens(cache.pending_k)
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
    expected = segment_lengths(cache.total_tokens, cache.sink_length, cache.recent_length)
    if sink_tokens != expected["sink_tokens"]:
        raise ValueError(f"sink token count mismatch: {sink_tokens} != {expected['sink_tokens']}")
    if recent_tokens != expected["recent_tokens"]:
        raise ValueError(f"recent token count mismatch: {recent_tokens} != {expected['recent_tokens']}")
    if cache.packed_k_tokens + pending_tokens != expected["quantized_history_tokens"]:
        raise ValueError("history token count mismatch")
    if isinstance(cache, PatternQuantizedKVCache):
        assignment_tokens = tensor_tokens(cache.k_assignments)
        if assignment_tokens not in (0, cache.packed_k_tokens):
            raise ValueError(f"Pattern K assignment tokens must match packed history: {assignment_tokens} != {cache.packed_k_tokens}")
        v_assignment_tokens = tensor_tokens(cache.v_assignment_idx)
        if v_assignment_tokens not in (0, cache.packed_v_tokens):
            raise ValueError(f"Pattern V assignment tokens must match packed history: {v_assignment_tokens} != {cache.packed_v_tokens}")


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
            cache.k_centroids,
            cache.v_centroids,
            int(cache.centroid_updates_k),
            int(cache.centroid_updates_v),
        )
    return base


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
    )
    if isinstance(cache, PatternQuantizedKVCache) and len(value) >= 30:
        cache.k_assignments = value[23]
        cache.v_assignments = value[24]
        cache.v_assignment_idx = value[25]
        cache.k_centroids = value[26]
        cache.v_centroids = value[27]
        cache.centroid_updates_k = int(value[28])
        cache.centroid_updates_v = int(value[29])
    validate_cache(cache)
    return cache


def maybe_validate_cache(cache: QuantizedKVCache) -> None:
    if cache_validate_enabled():
        validate_cache(cache)
