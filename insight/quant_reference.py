"""Pure PyTorch reference quantizers for PatternKV/KIVI diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantReferenceResult:
    """Reference affine quantization result."""

    q: torch.Tensor
    dequant: torch.Tensor
    scale: torch.Tensor
    mn: torch.Tensor


def affine_quant_dequant_last_dim(x: torch.Tensor, *, bits: int = 2, group_size: int = 128) -> QuantReferenceResult:
    """Asymmetric affine quantize/dequantize over groups on the last dimension.

    Scale and min are rounded to FP16 to match the storage semantics used by the
    packed cache. Zero-range groups use scale 1 before quantization to avoid
    NaN in diagnostics; their dequantized value is still exactly the group min.
    """
    if x.shape[-1] % group_size != 0:
        raise ValueError(f"last dim {x.shape[-1]} must be divisible by group_size={group_size}")
    max_int = 2**bits - 1
    original_shape = x.shape
    grouped = x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size).float()
    mn = grouped.amin(dim=-1, keepdim=True).to(torch.float16)
    mx = grouped.amax(dim=-1, keepdim=True).to(torch.float16)
    scale = ((mx - mn) / max_int).to(torch.float16)
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = ((grouped.to(torch.float16) - mn) / safe_scale).clamp(0, max_int).round().to(torch.int32)
    dequant = (q.to(torch.float16) * safe_scale + mn).reshape(original_shape)
    return QuantReferenceResult(q=q.reshape(original_shape), dequant=dequant, scale=scale.squeeze(-1), mn=mn.squeeze(-1))


def quantize_dequant_k_token_groups(k: torch.Tensor, *, bits: int = 2, group_size: int = 128) -> QuantReferenceResult:
    """Quantize K using real per-channel 128-token group semantics.

    Input shape is ``[batch, kv_heads, tokens, head_dim]``. The token dimension is
    grouped independently for every layer/head/head-dimension channel.
    """
    if k.dim() != 4:
        raise ValueError(f"K tensor must be [B,H,T,D], got {tuple(k.shape)}")
    transposed = k.transpose(2, 3).contiguous()
    out = affine_quant_dequant_last_dim(transposed, bits=bits, group_size=group_size)
    return QuantReferenceResult(
        q=out.q.transpose(2, 3).contiguous(),
        dequant=out.dequant.transpose(2, 3).contiguous(),
        scale=out.scale,
        mn=out.mn,
    )


def quantize_dequant_v_head_dim(v: torch.Tensor, *, bits: int = 2, group_size: int = 128) -> QuantReferenceResult:
    """Quantize V using per-token head-dimension group semantics.

    Input shape is ``[batch, kv_heads, tokens, head_dim]``. For Llama-3.1-8B the
    head dimension is 128, so each V token vector is one quantization group.
    """
    if v.dim() != 4:
        raise ValueError(f"V tensor must be [B,H,T,D], got {tuple(v.shape)}")
    return affine_quant_dequant_last_dim(v.contiguous(), bits=bits, group_size=group_size)


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return mean squared error as a scalar tensor."""
    return ((a.float() - b.float()) ** 2).mean()
