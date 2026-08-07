from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from models.segmented_cache import pattern_gather_centroids

_REFERENCE_METRIC_CAPTURES: dict[int, dict[int, dict[str, torch.Tensor]]] = {}


def _parse_int_set(value: str | None) -> set[int]:
    if not value:
        return set()
    return {int(item) for item in value.replace(",", " ").split() if item.strip()}


def reference_metric_capture_enabled(layer_idx: int) -> bool:
    if os.environ.get("PATTERNKV_REFERENCE_METRIC_CAPTURE") != "1":
        return False
    position = int(os.environ.get("PATTERNKV_EQUIV_TRACE_DECODE_POS", "-1"))
    return position in _parse_int_set(os.environ.get("PATTERNKV_REFERENCE_METRIC_CHECKPOINTS")) and int(layer_idx) in _parse_int_set(
        os.environ.get("PATTERNKV_REFERENCE_METRIC_LAYERS")
    )


def record_reference_metric_capture(
    *,
    decode_position: int,
    layer_idx: int,
    cache_mode: str,
    details: dict[str, torch.Tensor],
    post_o_proj: torch.Tensor,
) -> None:
    stored = {
        "cache_mode": cache_mode,
        "query": details["query"].detach().cpu(),
        "reconstructed_k": details["reconstructed_k"].detach().cpu(),
        "reconstructed_v": details["reconstructed_v"].detach().cpu(),
        "attention_scores": details["attention_scores"].detach().cpu(),
        "attention_probs": details["attention_probs"].detach().cpu(),
        "attention_output": details["attention_output"].detach().cpu(),
        "post_o_proj": post_o_proj.detach().cpu(),
    }
    _REFERENCE_METRIC_CAPTURES.setdefault(int(decode_position), {})[int(layer_idx)] = stored


def pop_reference_metric_captures() -> dict[int, dict[int, dict[str, torch.Tensor]]]:
    captures = dict(_REFERENCE_METRIC_CAPTURES)
    _REFERENCE_METRIC_CAPTURES.clear()
    return captures


def tensor_hash(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(cpu.shape)).encode())
    digest.update(str(cpu.dtype).encode())
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def reference_chebyshev_center_fp16(x: torch.Tensor) -> torch.Tensor:
    return ((x.to(torch.float16).amin(dim=1, keepdim=True) + x.to(torch.float16).amax(dim=1, keepdim=True)) * 0.5).to(torch.float16)


def reference_chebyshev_center_fp32(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return (xf.amin(dim=1, keepdim=True) + xf.amax(dim=1, keepdim=True)) * 0.5


def reference_minmax_distances(x: torch.Tensor, centroids: torch.Tensor, *, compute_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    xx = x.to(compute_dtype)
    cc = centroids.to(compute_dtype)
    diff = xx.unsqueeze(2) - cc.unsqueeze(1)
    return diff.amax(dim=-1) - diff.amin(dim=-1)


def reference_minmax_assign(x: torch.Tensor, centroids: torch.Tensor, *, compute_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    distances = reference_minmax_distances(x, centroids, compute_dtype=compute_dtype)
    return torch.argmin(distances, dim=-1)


def reference_top2(x: torch.Tensor, centroids: torch.Tensor, *, compute_dtype: torch.dtype) -> dict[str, Any]:
    distances = reference_minmax_distances(x, centroids, compute_dtype=compute_dtype)
    values, indices = torch.topk(distances, k=2, dim=-1, largest=False, sorted=True)
    return {
        "indices": indices.detach().cpu(),
        "distances": values.detach().cpu(),
        "margin": (values[..., 1] - values[..., 0]).detach().cpu(),
    }


def reference_v_gate(v: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1e-12
    vf = v.float()
    bf = base.float()
    range_v = (vf.amax(dim=-1) - vf.amin(dim=-1)).clamp_min(eps)
    residual = vf - bf
    range_residual = (residual.amax(dim=-1) - residual.amin(dim=-1)).clamp_min(eps)
    rho = (range_residual / range_v).clamp_min(0.0)
    rho4 = rho * rho
    rho4 = rho4 * rho4
    z = torch.sqrt(torch.tensor(2.0, dtype=torch.float32, device=v.device)) * torch.erfinv(torch.tensor(0.9, dtype=torch.float32, device=v.device))
    lhs = 1.0 - rho * rho
    rhs = (2.0 * z / torch.sqrt(torch.tensor(5.0 * float(v.shape[-1]), dtype=torch.float32, device=v.device))) * torch.sqrt(1.0 + rho4)
    return rho, lhs, rhs, lhs >= rhs


def reference_dequant_k(packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    from quant.new_pack import unpack_tensor

    q = unpack_tensor(packed, bits, pack_dim=3).to(scale.dtype)
    bsz, heads, dim, tokens = q.shape
    grouped = q.reshape(bsz, heads, dim, tokens // group_size, group_size)
    out = grouped * scale.unsqueeze(-1) + zero.unsqueeze(-1)
    return out.reshape(bsz, heads, dim, tokens).transpose(2, 3).contiguous()


def reference_dequant_v(packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    from quant.new_pack import unpack_tensor

    q = unpack_tensor(packed, bits, pack_dim=3).to(scale.dtype)
    bsz, heads, tokens, dim = q.shape
    grouped = q.reshape(bsz, heads, tokens, dim // group_size, group_size)
    out = grouped * scale.unsqueeze(-1) + zero.unsqueeze(-1)
    return out.reshape(bsz, heads, tokens, dim).contiguous()


def reference_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / (query.shape[-1] ** 0.5)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, value.float()).to(query.dtype)


def tensor_pair_metrics(a: torch.Tensor, b: torch.Tensor, *, finite_only: bool = False) -> dict[str, float | int | list[int]]:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    if finite_only:
        mask = torch.isfinite(af) & torch.isfinite(bf)
        af = af[mask]
        bf = bf[mask]
    if af.numel() == 0 and bf.numel() == 0:
        return {"shape": list(a.shape), "count": 0, "cosine": 1.0, "relative_mse": 0.0, "relative_l2": 0.0, "max_abs_error": 0.0}
    diff = af - bf
    denom = af.norm().clamp_min(1e-12)
    cosine_denom = (af.norm() * bf.norm()).clamp_min(1e-12)
    return {
        "shape": list(a.shape),
        "count": int(af.numel()),
        "cosine": float(torch.dot(af, bf).div(cosine_denom).item()),
        "relative_mse": float((diff.pow(2).mean() / af.pow(2).mean().clamp_min(1e-12)).item()),
        "relative_l2": float((torch.linalg.vector_norm(diff) / denom).item()),
        "max_abs_error": float(diff.abs().max().item()),
    }


def attention_probability_metrics(legacy_probs: torch.Tensor, segmented_probs: torch.Tensor) -> dict[str, float | int | list[int]]:
    metrics = tensor_pair_metrics(legacy_probs, segmented_probs)
    p = legacy_probs.detach().float().clamp_min(1e-12)
    q = segmented_probs.detach().float().clamp_min(1e-12)
    kl_pq = (p * (p.log() - q.log())).sum(dim=-1).mean()
    kl_qp = (q * (q.log() - p.log())).sum(dim=-1).mean()
    metrics.update(
        {
            "kl_legacy_segmented": float(kl_pq.item()),
            "kl_segmented_legacy": float(kl_qp.item()),
            "symmetric_kl": float((0.5 * (kl_pq + kl_qp)).item()),
        }
    )
    return metrics


@dataclass(frozen=True)
class ReferencePatternKVView:
    packed_k: torch.Tensor | None
    packed_k_scale: torch.Tensor | None
    packed_k_zero: torch.Tensor | None
    packed_v: torch.Tensor | None
    packed_v_scale: torch.Tensor | None
    packed_v_zero: torch.Tensor | None
    chunk_k: torch.Tensor | None
    chunk_v: torch.Tensor | None
    k_centroids: torch.Tensor | None
    v_centroids: torch.Tensor | None
    k_assignments: torch.Tensor | None
    v_assignment_idx: torch.Tensor | None
    v_pattern_mask: torch.Tensor | None
    total_tokens: int
    packed_k_tokens: int
    packed_v_tokens: int
    group_size: int
    k_bits: int
    v_bits: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    cache_mode: str


def repeat_kv_for_gqa(hidden_states: torch.Tensor, num_key_value_groups: int, *, expected_heads: int | None = None, tensor_name: str = "kv") -> torch.Tensor:
    if hidden_states is None:
        return hidden_states
    if hidden_states.dim() != 4:
        raise ValueError(f"{tensor_name} must be 4D [B,H,L,D], got shape={tuple(hidden_states.shape)}")
    repeated = hidden_states.repeat_interleave(num_key_value_groups, dim=1)
    if expected_heads is not None and repeated.shape[1] != expected_heads:
        raise ValueError(
            f"{tensor_name} repeated heads mismatch: got shape={tuple(repeated.shape)}, expected_heads={expected_heads}, num_key_value_groups={num_key_value_groups}"
        )
    return repeated


def reference_view_from_segmented_cache(cache: Any, *, num_attention_heads: int, num_key_value_heads: int, head_dim: int) -> ReferencePatternKVView:
    from models.segmented_cache import CHUNKED_CACHE_MODE, cache_validate_enabled, deserialize_cache, maybe_validate_cache

    if not isinstance(cache, tuple):
        cache = deserialize_cache(cache, pattern=True)
    if cache.cache_mode != CHUNKED_CACHE_MODE:
        raise ValueError("reference equivalence only supports segmented_chunked cache mode")
    if cache.sink_length != 0 or cache.recent_length != 0:
        raise ValueError("reference chunked cache must have zero sink and recent lengths")
    if cache_validate_enabled():
        maybe_validate_cache(cache)
    return ReferencePatternKVView(
        packed_k=cache.packed_k,
        packed_k_scale=cache.packed_k_scale,
        packed_k_zero=cache.packed_k_zero,
        packed_v=cache.packed_v,
        packed_v_scale=cache.packed_v_scale,
        packed_v_zero=cache.packed_v_zero,
        chunk_k=cache.pending_k,
        chunk_v=cache.pending_v,
        k_centroids=cache.k_centroids,
        v_centroids=cache.v_centroids,
        k_assignments=cache.k_assignments,
        v_assignment_idx=cache.v_assignment_idx,
        v_pattern_mask=cache.v_pattern_mask if cache.v_pattern_mask is not None else cache.v_assignments,
        total_tokens=int(cache.total_tokens),
        packed_k_tokens=int(cache.packed_k_tokens),
        packed_v_tokens=int(cache.packed_v_tokens),
        group_size=int(cache.group_size),
        k_bits=int(cache.k_bits),
        v_bits=int(cache.v_bits),
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
        head_dim=int(head_dim),
        cache_mode=str(cache.cache_mode),
    )


def reference_view_from_legacy_tuple(
    legacy_cache: tuple[Any, ...],
    *,
    chunk_k: torch.Tensor | None,
    chunk_v: torch.Tensor | None,
    k_centroids: torch.Tensor | None,
    v_centroids: torch.Tensor | None,
    total_tokens: int | None = None,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> ReferencePatternKVView:
    if not isinstance(legacy_cache, tuple) or len(legacy_cache) < 12:
        raise TypeError("legacy cache must be a PatternKV tuple")
    packed_k = legacy_cache[0]
    packed_v = legacy_cache[4]
    k_assignments = legacy_cache[9]
    v_pattern_mask = legacy_cache[10]
    v_assignment_idx = legacy_cache[11]
    total_tokens_value = int(total_tokens if total_tokens is not None else legacy_cache[8])
    packed_k_tokens = int(k_assignments.shape[-1]) if torch.is_tensor(k_assignments) else int(total_tokens_value - (chunk_k.shape[2] if torch.is_tensor(chunk_k) else 0))
    packed_v_tokens = int(v_assignment_idx.shape[-1]) if torch.is_tensor(v_assignment_idx) else int(total_tokens_value - (chunk_v.shape[2] if torch.is_tensor(chunk_v) else 0))
    return ReferencePatternKVView(
        packed_k=packed_k,
        packed_k_scale=legacy_cache[2],
        packed_k_zero=legacy_cache[3],
        packed_v=packed_v,
        packed_v_scale=legacy_cache[6],
        packed_v_zero=legacy_cache[7],
        chunk_k=chunk_k,
        chunk_v=chunk_v,
        k_centroids=k_centroids,
        v_centroids=v_centroids,
        k_assignments=k_assignments,
        v_assignment_idx=v_assignment_idx,
        v_pattern_mask=v_pattern_mask,
        total_tokens=total_tokens_value,
        packed_k_tokens=packed_k_tokens,
        packed_v_tokens=packed_v_tokens,
        group_size=128,
        k_bits=2,
        v_bits=2,
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
        head_dim=int(head_dim),
        cache_mode="legacy_tuple_chunked",
    )


def _validate_view(view: ReferencePatternKVView) -> None:
    if view.total_tokens < 0 or view.packed_k_tokens < 0 or view.packed_v_tokens < 0:
        raise ValueError("token counts must be non-negative")
    chunk_k_tokens = int(view.chunk_k.shape[2]) if torch.is_tensor(view.chunk_k) else 0
    chunk_v_tokens = int(view.chunk_v.shape[2]) if torch.is_tensor(view.chunk_v) else 0
    if view.packed_k_tokens + chunk_k_tokens != view.total_tokens:
        raise ValueError(
            f"K token count mismatch: packed={view.packed_k_tokens}, chunk={chunk_k_tokens}, total={view.total_tokens}"
        )
    if view.packed_v_tokens + chunk_v_tokens != view.total_tokens:
        raise ValueError(
            f"V token count mismatch: packed={view.packed_v_tokens}, chunk={chunk_v_tokens}, total={view.total_tokens}"
        )


def reference_reconstruct_k(view: ReferencePatternKVView) -> torch.Tensor | None:
    _validate_view(view)
    if view.packed_k is None:
        return view.chunk_k
    packed = reference_dequant_k(view.packed_k, view.packed_k_scale, view.packed_k_zero, view.group_size, view.k_bits)
    if packed is not None:
        packed = packed[:, :, : view.packed_k_tokens, :].contiguous()
        if view.k_centroids is None or view.k_assignments is None:
            raise ValueError("reference K reconstruction requires centroids and assignments")
        packed = packed + pattern_gather_centroids(view.k_assignments[:, :, : view.packed_k_tokens], view.k_centroids).to(packed.dtype)
    parts = [part for part in (packed, view.chunk_k) if torch.is_tensor(part)]
    return torch.cat(parts, dim=2).contiguous() if parts else None


def reference_reconstruct_v(view: ReferencePatternKVView) -> torch.Tensor | None:
    _validate_view(view)
    if view.packed_v is None:
        return view.chunk_v
    packed = reference_dequant_v(view.packed_v, view.packed_v_scale, view.packed_v_zero, view.group_size, view.v_bits)
    if packed is not None:
        packed = packed[:, :, : view.packed_v_tokens, :].contiguous()
        if view.v_centroids is None or view.v_assignment_idx is None or view.v_pattern_mask is None:
            raise ValueError("reference V reconstruction requires centroids, assignments and gate")
        centroids = pattern_gather_centroids(view.v_assignment_idx[:, :, : view.packed_v_tokens], view.v_centroids).to(packed.dtype)
        packed = packed + view.v_pattern_mask[:, :, : view.packed_v_tokens].unsqueeze(-1).to(packed.dtype) * centroids
    parts = [part for part in (packed, view.chunk_v) if torch.is_tensor(part)]
    return torch.cat(parts, dim=2).contiguous() if parts else None


def reference_patternkv_attention(
    query_states: torch.Tensor,
    view: ReferencePatternKVView,
    *,
    attention_mask: torch.Tensor | None = None,
    softmax_dtype: torch.dtype = torch.float32,
    return_details: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    key_states = reference_reconstruct_k(view)
    value_states = reference_reconstruct_v(view)
    if key_states is None or value_states is None:
        raise ValueError("reference PatternKV attention requires K and V states")
    num_key_value_groups = view.num_attention_heads // view.num_key_value_heads
    key_for_attention = repeat_kv_for_gqa(key_states, num_key_value_groups, expected_heads=view.num_attention_heads, tensor_name="key_states")
    value_for_attention = repeat_kv_for_gqa(value_states, num_key_value_groups, expected_heads=view.num_attention_heads, tensor_name="value_states")
    scores = torch.matmul(query_states.float(), key_for_attention.float().transpose(-2, -1)) / (query_states.shape[-1] ** 0.5)
    if attention_mask is not None:
        scores = scores + attention_mask
        scores = torch.max(scores, torch.tensor(torch.finfo(scores.dtype).min, device=scores.device))
    probs = torch.softmax(scores, dim=-1, dtype=softmax_dtype).to(query_states.dtype)
    output = torch.matmul(probs.float(), value_for_attention.float()).to(query_states.dtype)
    if return_details:
        return output, probs, key_for_attention, value_for_attention, {
            "query": query_states,
            "reconstructed_k": key_states,
            "reconstructed_v": value_states,
            "attention_scores": scores,
            "attention_probs": probs,
            "attention_output": output,
        }
    return output, probs, key_for_attention, value_for_attention


def reference_assignment_disagreement(stored: torch.Tensor | None, reference: torch.Tensor | None) -> float | None:
    if not torch.is_tensor(stored) or not torch.is_tensor(reference):
        return None
    tokens = min(stored.shape[-1], reference.shape[-1])
    if tokens == 0:
        return 0.0
    stored = stored[..., :tokens].detach().cpu()
    reference = reference[..., :tokens].detach().cpu()
    return float((stored != reference).float().mean().item())


def reference_logits_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int | bool]:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    denom = af.norm() * bf.norm()
    cosine = float(torch.dot(af, bf).div(denom.clamp_min(1e-12)).item())
    return {
        "cosine": cosine,
        "relative_mse": float(((af - bf) ** 2).mean().div((af**2).mean().clamp_min(1e-12)).item()),
        "max_abs_error": float((af - bf).abs().max().item()),
        "top1_a": int(torch.argmax(af).item()),
        "top1_b": int(torch.argmax(bf).item()),
        "top1_agreement": int(torch.argmax(af).item()) == int(torch.argmax(bf).item()),
    }


def _trace_enabled(layer_idx: int) -> bool:
    if os.environ.get("PATTERNKV_EQUIV_TRACE") != "1":
        return False
    target_layer = os.environ.get("PATTERNKV_EQUIV_TRACE_LAYER")
    return target_layer is None or int(target_layer) == int(layer_idx)


def save_assignment_trace(
    *,
    mode: str,
    layer_idx: int,
    decode_position: int | None,
    k_window: torch.Tensor,
    v_window: torch.Tensor | None,
    k_centroids: torch.Tensor,
    k_assignments: torch.Tensor,
    v_centroids: torch.Tensor | None = None,
    v_assignment_idx: torch.Tensor | None = None,
    v_gate: torch.Tensor | None = None,
) -> None:
    if not _trace_enabled(layer_idx):
        return
    out_dir = Path(os.environ.get("PATTERNKV_EQUIV_TRACE_DIR", "artifacts/aime24_patternkv_equivalence"))
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = os.environ.get("PATTERNKV_EQUIV_TRACE_SAMPLE", "unknown").replace(":", "_")
    pos = int(decode_position or -1)
    stem = f"trace_{sample}_{mode}_ckpt{pos}_layer{layer_idx}_chunk{k_assignments.shape[-1]}"
    bsz, heads, tokens, dim = k_window.shape
    xk = k_window.permute(1, 0, 2, 3).reshape(heads, bsz * tokens, dim).contiguous()
    top2_fp16 = reference_top2(xk, k_centroids, compute_dtype=torch.float16)
    top2_fp32 = reference_top2(xk, k_centroids, compute_dtype=torch.float32)
    payload = {
        "metadata": {
            "mode": mode,
            "layer": int(layer_idx),
            "decode_position": pos,
            "k_window_shape": list(k_window.shape),
            "k_window_dtype": str(k_window.dtype),
            "k_window_stride": list(k_window.stride()),
            "k_window_contiguous": bool(k_window.is_contiguous()),
            "k_window_hash": tensor_hash(k_window),
            "k_centroids_shape": list(k_centroids.shape),
            "k_centroids_dtype": str(k_centroids.dtype),
            "k_centroids_hash": tensor_hash(k_centroids),
            "distance_compute_dtypes": ["torch.float16", "torch.float32"],
            "torch_argmin_tie_rule": "lowest index",
        },
        "k_window": k_window.detach().cpu(),
        "v_window": v_window.detach().cpu() if torch.is_tensor(v_window) else None,
        "k_centroids": k_centroids.detach().cpu(),
        "v_centroids": v_centroids.detach().cpu() if torch.is_tensor(v_centroids) else None,
        "k_assignments": k_assignments.detach().cpu(),
        "v_assignment_idx": v_assignment_idx.detach().cpu() if torch.is_tensor(v_assignment_idx) else None,
        "v_gate": v_gate.detach().cpu() if torch.is_tensor(v_gate) else None,
        "k_top2_fp16": top2_fp16,
        "k_top2_fp32": top2_fp32,
    }
    torch.save(payload, out_dir / f"{stem}.pt")
    (out_dir / f"{stem}.json").write_text(json.dumps(payload["metadata"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
