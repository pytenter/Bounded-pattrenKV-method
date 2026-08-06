from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


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
