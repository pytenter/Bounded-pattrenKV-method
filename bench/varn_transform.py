from __future__ import annotations

from dataclasses import dataclass

import torch

from bench.reference_varn import (
    CLIP_STD_MAX,
    CLIP_STD_MIN,
    DEFAULT_ITERATIONS,
    EPS,
    LOG_S_MAX,
    LOG_S_MIN,
    restore_varn_tile,
    variance_normalize_batched_reference,
)


@dataclass
class VarNMetadata:
    s_col: torch.Tensor
    s_row: torch.Tensor
    tile_tokens: int
    axis: str


CANONICAL_VARN_CONFIG = {
    "iterations": DEFAULT_ITERATIONS,
    "clip_std_min": CLIP_STD_MIN,
    "clip_std_max": CLIP_STD_MAX,
    "log_s_min": LOG_S_MIN,
    "log_s_max": LOG_S_MAX,
    "eps": EPS,
    "dtype_behavior": "compute scales in fp32; store production metadata in input dtype",
    "k_axis": {"tile_shape": "[D, group]", "s_col": "[1, group] per token", "s_row": "[D, 1] per channel"},
    "v_axis": {"tile_shape": "[group, D]", "s_col": "[1, D] per channel", "s_row": "[group, 1] per token"},
}


def _check_bhtd(tile: torch.Tensor, group_size: int) -> tuple[int, int, int, int, int]:
    if tile.dim() != 4:
        raise ValueError(f"VarN expects [batch, heads, tokens, dim], got {tuple(tile.shape)}")
    bsz, heads, tokens, dim = tile.shape
    if tokens % group_size != 0:
        raise ValueError(f"VarN token length {tokens} must be divisible by group_size={group_size}")
    return bsz, heads, tokens, dim, tokens // group_size


def varn_balance_k(tile: torch.Tensor, group_size: int, *, iterations: int = DEFAULT_ITERATIONS) -> tuple[torch.Tensor, VarNMetadata]:
    """Apply canonical VarN-K semantics to [B,H,T,D].

    K canonical tiles are [D, group]: s_col is per token and s_row is per
    channel. The returned balanced tensor has the original [B,H,T,D] layout.
    """

    bsz, heads, tokens, dim, n_tiles = _check_bhtd(tile, group_size)
    canonical = tile.reshape(bsz, heads, n_tiles, group_size, dim).permute(0, 1, 2, 4, 3)
    flat = canonical.reshape(bsz * heads * n_tiles, dim, group_size)
    balanced, s_col, s_row = variance_normalize_batched_reference(flat, iterations=iterations)
    balanced_bhtd = balanced.reshape(bsz, heads, n_tiles, dim, group_size).permute(0, 1, 2, 4, 3).reshape(bsz, heads, tokens, dim)
    meta = VarNMetadata(
        s_col=s_col.reshape(bsz, heads, n_tiles, group_size).reshape(bsz, heads, tokens).to(tile.dtype),
        s_row=s_row.reshape(bsz, heads, n_tiles, dim).to(tile.dtype),
        tile_tokens=group_size,
        axis="k",
    )
    return balanced_bhtd.to(tile.dtype), meta


def varn_balance_v(tile: torch.Tensor, group_size: int, *, iterations: int = DEFAULT_ITERATIONS) -> tuple[torch.Tensor, VarNMetadata]:
    """Apply canonical VarN-V semantics to [B,H,T,D].

    V canonical tiles are [group, D]: s_col is per channel and s_row is per
    token. The returned balanced tensor has the original [B,H,T,D] layout.
    """

    bsz, heads, tokens, dim, n_tiles = _check_bhtd(tile, group_size)
    flat = tile.reshape(bsz * heads * n_tiles, group_size, dim)
    balanced, s_col, s_row = variance_normalize_batched_reference(flat, iterations=iterations)
    meta = VarNMetadata(
        s_col=s_col.reshape(bsz, heads, n_tiles, dim).to(tile.dtype),
        s_row=s_row.reshape(bsz, heads, n_tiles, group_size).reshape(bsz, heads, tokens).to(tile.dtype),
        tile_tokens=group_size,
        axis="v",
    )
    return balanced.reshape(bsz, heads, tokens, dim).to(tile.dtype), meta


def varn_restore_k(balanced: torch.Tensor, metadata: VarNMetadata) -> torch.Tensor:
    if metadata.axis != "k":
        raise ValueError(f"expected K VarN metadata, got {metadata.axis!r}")
    bsz, heads, tokens, dim, n_tiles = _check_bhtd(balanced, metadata.tile_tokens)
    canonical = balanced.reshape(bsz, heads, n_tiles, metadata.tile_tokens, dim).permute(0, 1, 2, 4, 3)
    s_col = metadata.s_col.reshape(bsz, heads, n_tiles, 1, metadata.tile_tokens)
    s_row = metadata.s_row.reshape(bsz, heads, n_tiles, dim, 1)
    restored = restore_varn_tile(canonical, s_col, s_row)
    return restored.permute(0, 1, 2, 4, 3).reshape(bsz, heads, tokens, dim).to(balanced.dtype)


def varn_restore_v(balanced: torch.Tensor, metadata: VarNMetadata) -> torch.Tensor:
    if metadata.axis != "v":
        raise ValueError(f"expected V VarN metadata, got {metadata.axis!r}")
    bsz, heads, tokens, dim, n_tiles = _check_bhtd(balanced, metadata.tile_tokens)
    canonical = balanced.reshape(bsz, heads, n_tiles, metadata.tile_tokens, dim)
    s_col = metadata.s_col.reshape(bsz, heads, n_tiles, 1, dim)
    s_row = metadata.s_row.reshape(bsz, heads, n_tiles, metadata.tile_tokens, 1)
    return restore_varn_tile(canonical, s_col, s_row).reshape(bsz, heads, tokens, dim).to(balanced.dtype)


def metadata_stats(*tensors: torch.Tensor | None) -> dict[str, float | bool | int]:
    values = [tensor.detach().float().reshape(-1).cpu() for tensor in tensors if torch.is_tensor(tensor)]
    if not values:
        return {"present": False, "finite": True, "count": 0}
    flat = torch.cat(values)
    finite = torch.isfinite(flat)
    valid = flat[finite]
    if valid.numel() == 0:
        return {"present": True, "finite": False, "count": int(flat.numel())}
    return {
        "present": True,
        "finite": bool(finite.all().item()),
        "count": int(flat.numel()),
        "min": float(valid.min().item()),
        "median": float(valid.median().item()),
        "p95": float(torch.quantile(valid, 0.95).item()),
        "p99": float(torch.quantile(valid, 0.99).item()),
        "max": float(valid.max().item()),
    }
