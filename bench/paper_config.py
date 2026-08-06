from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch

from models.segmented_cache import cache_segment_stats, deserialize_cache


PAPER_METHODS = {"kivi_paper_g128", "patternkv_paper"}


@dataclass(frozen=True)
class MethodConfig:
    method: str
    backend_method: str
    k_bits: int
    v_bits: int
    group_size: int
    residual_length: int
    key_quant_axis: str
    value_quant_axis: str
    asym: bool
    sink_length: int = 0
    recent_length: int = 128
    initial_pattern_count: int | None = None
    pattern_group: int | None = None
    pattern_selection_position: str | None = None


def apply_method_defaults(args) -> MethodConfig:
    if not hasattr(args, "sink_length"):
        args.sink_length = 0
    if not hasattr(args, "recent_length") or args.recent_length is None:
        args.recent_length = getattr(args, "residual_length", 128)
    args.residual_length = args.recent_length
    if args.method == "kivi_paper_g128":
        args.k_bits = 2
        args.v_bits = 2
        args.group_size = 128
        args.residual_length = 128
        return MethodConfig(
            method=args.method,
            backend_method="kivi_official",
            k_bits=2,
            v_bits=2,
            group_size=128,
            residual_length=128,
            sink_length=args.sink_length,
            recent_length=args.recent_length,
            key_quant_axis="per-channel: quantize transposed K along token axis",
            value_quant_axis="per-token: quantize V along head_dim axis",
            asym=True,
        )
    if args.method == "kivi_original_g32":
        args.k_bits = 2
        args.v_bits = 2
        args.group_size = 32
        args.residual_length = 128
        return MethodConfig(
            method=args.method,
            backend_method="kivi_official",
            k_bits=2,
            v_bits=2,
            group_size=32,
            residual_length=128,
            sink_length=args.sink_length,
            recent_length=args.recent_length,
            key_quant_axis="per-channel: quantize transposed K along token axis",
            value_quant_axis="per-token: quantize V along head_dim axis",
            asym=True,
        )
    if args.method == "patternkv_paper":
        args.k_bits = 2
        args.v_bits = 2
        args.group_size = 128
        args.residual_length = 128
        args.num_k_base = 32
        args.num_v_base = 32
        return MethodConfig(
            method=args.method,
            backend_method="patternkv",
            k_bits=2,
            v_bits=2,
            group_size=128,
            residual_length=128,
            sink_length=args.sink_length,
            recent_length=args.recent_length,
            key_quant_axis="per-channel: quantize transposed K along token axis",
            value_quant_axis="per-token: quantize V residual/centroids along head_dim axis",
            asym=True,
            initial_pattern_count=32,
            pattern_group=128,
            pattern_selection_position="post-RoPE key/value states",
        )
    if args.method == "kivi_official":
        return MethodConfig(
            method=args.method,
            backend_method="kivi_official",
            k_bits=args.k_bits,
            v_bits=args.v_bits,
            group_size=args.group_size,
            residual_length=args.residual_length,
            sink_length=args.sink_length,
            recent_length=args.recent_length,
            key_quant_axis="per-channel: quantize transposed K along token axis",
            value_quant_axis="per-token: quantize V along head_dim axis",
            asym=True,
        )
    if args.method == "patternkv":
        return MethodConfig(
            method=args.method,
            backend_method="patternkv",
            k_bits=args.k_bits,
            v_bits=args.v_bits,
            group_size=args.group_size,
            residual_length=args.residual_length,
            sink_length=args.sink_length,
            recent_length=args.recent_length,
            key_quant_axis="per-channel: quantize transposed K along token axis",
            value_quant_axis="per-token: quantize V residual/centroids along head_dim axis",
            asym=True,
            initial_pattern_count=args.num_k_base,
            pattern_group=args.residual_length,
            pattern_selection_position="post-RoPE key/value states",
        )
    if args.method == "kivi":
        return MethodConfig(
            method=args.method,
            backend_method="hf_flexible_quantized_cache",
            k_bits=args.k_bits,
            v_bits=args.v_bits,
            group_size=args.group_size,
            residual_length=args.residual_length,
            sink_length=args.sink_length,
            recent_length=args.recent_length,
            key_quant_axis="FlexibleQuantizedCache axis_key=1",
            value_quant_axis="FlexibleQuantizedCache axis_value=0",
            asym=True,
        )
    return MethodConfig(
        method=args.method,
        backend_method="fp16",
        k_bits=16,
        v_bits=16,
        group_size=0,
        residual_length=0,
        sink_length=0,
        recent_length=0,
        key_quant_axis="none",
        value_quant_axis="none",
        asym=False,
    )


def method_config_dict(args) -> dict[str, Any]:
    cfg = getattr(args, "paper_method_config", None) or apply_method_defaults(args)
    out = asdict(cfg)
    out["kivi_quantized_region_theoretical_bits"] = kivi_quantized_region_bits(cfg.group_size, cfg.k_bits) if cfg.backend_method == "kivi_official" else None
    if out["kivi_quantized_region_theoretical_bits"] is not None:
        assert abs(out["kivi_quantized_region_theoretical_bits"] - (cfg.k_bits + 32 / cfg.group_size)) < 1e-9
    out["quantized_region_affine_bits"] = (
        kivi_quantized_region_bits(cfg.group_size, cfg.k_bits)
        if cfg.backend_method in ("kivi_official", "patternkv")
        else None
    )
    out["compact_kernel_storage_note"] = "Packed payload plus FP16 scale/min; Python cache may store indices/masks at wider tensor dtypes."
    out["mixed_key_mask_path"] = str(getattr(args, "mixed_key_mask_path", "") or "")
    out["mixed_key_int4_ratio"] = float(getattr(args, "mixed_key_int4_ratio", 0.0) or 0.0)
    out["patternkv_cache_path"] = str(getattr(args, "patternkv_cache_path", "") or "") if str(getattr(args, "method", "")).startswith("pattern") else None
    return out


def kivi_quantized_region_bits(group_size: int = 128, payload_bits: int = 2) -> float:
    return payload_bits + 32.0 / group_size


def residual_split(total_cached_tokens: int, residual_length: int) -> dict[str, int]:
    residual = min(max(total_cached_tokens, 0), max(residual_length, 0))
    return {
        "total_cached_tokens": max(total_cached_tokens, 0),
        "quantized_tokens": max(total_cached_tokens - residual, 0),
        "fp16_residual_tokens": residual,
    }


def pattern_boundary_events(total_decode_tokens: int, residual_length: int = 128) -> list[int]:
    return [step for step in range(1, total_decode_tokens + 1) if step % residual_length == 0]


def tensor_bytes(value: Any) -> int:
    return int(value.numel() * value.element_size()) if torch.is_tensor(value) else 0


def dtype_name(value: Any) -> str | None:
    return str(value.dtype) if torch.is_tensor(value) else None


def cache_storage_summary(method: str, past_key_values, model=None, total_cached_tokens: int = 0, residual_length: int = 128) -> dict[str, Any]:
    out: dict[str, Any] = residual_split(total_cached_tokens, residual_length)
    segment_totals = {
        "sink_tokens": 0,
        "packed_history_tokens": 0,
        "pending_history_tokens": 0,
        "recent_tokens": 0,
        "total_tokens": max(total_cached_tokens, 0),
        "k_assignment_tokens": None,
        "v_assignment_tokens": None,
    }
    out.update(
        {
            "packed_payload_bytes": 0,
            "scale_min_bytes": 0,
            "fp16_residual_bytes": 0,
            "assignment_bytes": 0,
            "assignment_dtype": None,
            "mask_bytes": 0,
            "mask_dtype": None,
            "centroid_bytes": 0,
            "k_centroid_bytes": 0,
            "v_centroid_bytes": 0,
            "k_assignment_theoretical_compact_bits": 0,
            "v_assignment_theoretical_compact_bits": 0,
            "v_gate_theoretical_compact_bits": 0,
            "assignment_actual_python_tensor_bits": 0,
            "v_gate_actual_python_tensor_bits": 0,
            "initial_pattern_count": None,
            "dynamic_pattern_count_k": None,
            "dynamic_pattern_count_v": None,
            "persistent_key_heads": None,
            "persistent_value_heads": None,
            "cache_segment_stats": segment_totals,
        }
    )
    for layer in past_key_values or []:
        if isinstance(layer, tuple) and layer and layer[0] in ("quantized_segmented_cache_v1", "patternkv_segmented_cache_v1"):
            cache = deserialize_cache(layer, pattern=layer[0] == "patternkv_segmented_cache_v1")
            stats = cache_segment_stats(cache)
            for key in ("sink_tokens", "packed_history_tokens", "pending_history_tokens", "recent_tokens"):
                segment_totals[key] = max(int(segment_totals[key] or 0), int(stats[key] or 0))
            segment_totals["total_tokens"] = int(stats["total_tokens"] or segment_totals["total_tokens"])
            for key in ("k_assignment_tokens", "v_assignment_tokens"):
                if stats[key] is not None:
                    segment_totals[key] = int(stats[key] or 0) if segment_totals[key] is None else max(int(segment_totals[key]), int(stats[key] or 0))
            out["packed_payload_bytes"] += tensor_bytes(cache.packed_k) + tensor_bytes(cache.packed_v)
            out["scale_min_bytes"] += tensor_bytes(cache.packed_k_scale) + tensor_bytes(cache.packed_k_zero) + tensor_bytes(cache.packed_v_scale) + tensor_bytes(cache.packed_v_zero)
            out["fp16_residual_bytes"] += tensor_bytes(cache.sink_k) + tensor_bytes(cache.sink_v) + tensor_bytes(cache.pending_k) + tensor_bytes(cache.pending_v) + tensor_bytes(cache.recent_k) + tensor_bytes(cache.recent_v)
            out["persistent_key_heads"] = out["persistent_key_heads"] or (int(cache.sink_k.shape[1]) if torch.is_tensor(cache.sink_k) else None)
            out["persistent_value_heads"] = out["persistent_value_heads"] or (int(cache.sink_v.shape[1]) if torch.is_tensor(cache.sink_v) else None)
            k_assignments = getattr(cache, "k_assignments", None)
            v_assignment_idx = getattr(cache, "v_assignment_idx", None)
            v_mask = getattr(cache, "v_pattern_mask", None) if getattr(cache, "v_pattern_mask", None) is not None else getattr(cache, "v_assignments", None)
            k_centroids = getattr(cache, "k_centroids", None)
            v_centroids = getattr(cache, "v_centroids", None)
            k_assignment_bytes = tensor_bytes(k_assignments)
            v_assignment_bytes = tensor_bytes(v_assignment_idx)
            v_mask_bytes = tensor_bytes(v_mask)
            out["assignment_bytes"] += k_assignment_bytes + v_assignment_bytes
            out["assignment_actual_python_tensor_bits"] += 8 * (k_assignment_bytes + v_assignment_bytes)
            out["assignment_dtype"] = out["assignment_dtype"] or dtype_name(k_assignments) or dtype_name(v_assignment_idx)
            out["mask_bytes"] += v_mask_bytes
            out["v_gate_actual_python_tensor_bits"] += 8 * v_mask_bytes
            out["mask_dtype"] = out["mask_dtype"] or dtype_name(v_mask)
            out["k_centroid_bytes"] += tensor_bytes(k_centroids)
            out["v_centroid_bytes"] += tensor_bytes(v_centroids)
            out["centroid_bytes"] += tensor_bytes(k_centroids) + tensor_bytes(v_centroids)
            if torch.is_tensor(k_assignments) and torch.is_tensor(k_centroids):
                out["k_assignment_theoretical_compact_bits"] += int(k_assignments.numel()) * max(1, math.ceil(math.log2(max(int(k_centroids.shape[1]), 2))))
            if torch.is_tensor(v_assignment_idx) and torch.is_tensor(v_centroids):
                out["v_assignment_theoretical_compact_bits"] += int(v_assignment_idx.numel()) * max(1, math.ceil(math.log2(max(int(v_centroids.shape[1]), 2))))
            if torch.is_tensor(v_mask):
                out["v_gate_theoretical_compact_bits"] += int(v_mask.numel())
            continue
        names = (
            "key_states_quant_trans",
            "key_states_full",
            "key_scale_trans",
            "key_mn_trans",
            "value_states_quant",
            "value_states_full",
            "value_scale",
            "value_mn",
            "kv_seq_len",
            "k_assignments",
            "v_mask",
            "v_assignments_idx",
        )
        vals = dict(zip(names, layer))
        for name in ("key_states_quant_trans", "value_states_quant"):
            out["packed_payload_bytes"] += tensor_bytes(vals.get(name))
        for name in ("key_scale_trans", "key_mn_trans", "value_scale", "value_mn"):
            out["scale_min_bytes"] += tensor_bytes(vals.get(name))
        for name in ("key_states_full", "value_states_full"):
            out["fp16_residual_bytes"] += tensor_bytes(vals.get(name))
        for name in ("key_states_quant_trans", "key_states_full"):
            value = vals.get(name)
            if torch.is_tensor(value):
                out["persistent_key_heads"] = out["persistent_key_heads"] or int(value.shape[1])
        for name in ("value_states_quant", "value_states_full"):
            value = vals.get(name)
            if torch.is_tensor(value):
                out["persistent_value_heads"] = out["persistent_value_heads"] or int(value.shape[1])
        for name in ("k_assignments", "v_assignments_idx"):
            value = vals.get(name)
            out["assignment_bytes"] += tensor_bytes(value)
            out["assignment_dtype"] = out["assignment_dtype"] or dtype_name(value)
        value = vals.get("v_mask")
        out["mask_bytes"] += tensor_bytes(value)
        out["mask_dtype"] = out["mask_dtype"] or dtype_name(value)
    if model is not None:
        k_counts = []
        v_counts = []
        for layer in getattr(getattr(model, "model", None), "layers", []):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            k_base = getattr(attn, "k_base", None)
            v_centroids = getattr(attn, "v_centroids", None)
            out["centroid_bytes"] += tensor_bytes(k_base) + tensor_bytes(v_centroids)
            if torch.is_tensor(k_base):
                k_counts.append(int(k_base.shape[-2]))
            if torch.is_tensor(v_centroids):
                v_counts.append(int(v_centroids.shape[-2]))
        if k_counts:
            out["initial_pattern_count"] = 32
            out["dynamic_pattern_count_k"] = max(k_counts) - 32
        if v_counts:
            out["dynamic_pattern_count_v"] = max(v_counts) - 32
    payload_bits = 2 if method != "fp16" else 16
    group_size = 128 if method in PAPER_METHODS else residual_length
    out["quantized_region_theoretical_bits_per_scalar"] = payload_bits + (32.0 / group_size if method != "fp16" and group_size else 0.0)
    out["python_tensor_storage_bytes"] = sum(out[k] for k in ("packed_payload_bytes", "scale_min_bytes", "fp16_residual_bytes", "assignment_bytes", "mask_bytes", "centroid_bytes"))
    return out
