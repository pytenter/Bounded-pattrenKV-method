"""Read-only oracle metrics for PatternKV diagnostics."""

from __future__ import annotations

import torch

from insight.quant_reference import quantize_dequant_k_token_groups, quantize_dequant_v_head_dim


def l2_assignment(tokens: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Return argmin L2 centroid assignment for ``[N,D]`` tokens."""
    d2 = ((tokens[:, None, :].float() - centroids[None, :, :].float()) ** 2).sum(dim=-1)
    return d2.argmin(dim=-1)


def minmax_assignment(tokens: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Return argmin residual range centroid assignment for ``[N,D]`` tokens."""
    residual = tokens[:, None, :].float() - centroids[None, :, :].float()
    ranges = residual.amax(dim=-1) - residual.amin(dim=-1)
    return ranges.argmin(dim=-1)


def v_mse_oracle_assignment(tokens: torch.Tensor, centroids: torch.Tensor, *, bits: int = 2, group_size: int = 128) -> torch.Tensor:
    """Return V centroid assignment minimizing reference quantization MSE."""
    losses = []
    for idx in range(centroids.shape[0]):
        residual = (tokens - centroids[idx]).view(1, 1, tokens.shape[0], tokens.shape[1])
        dq = quantize_dequant_v_head_dim(residual, bits=bits, group_size=group_size).dequant.view_as(tokens)
        rec = dq + centroids[idx]
        losses.append(((tokens.float() - rec.float()) ** 2).mean(dim=-1))
    return torch.stack(losses, dim=-1).argmin(dim=-1)


def conditional_k_oracle_group_mse(k_group: torch.Tensor, pattern_group: torch.Tensor, *, bits: int = 2, group_size: int = 128) -> float:
    """Compute K group reconstruction MSE under supplied per-token patterns.

    ``k_group`` and ``pattern_group`` are ``[1,H,128,D]`` tensors. This helper is
    intentionally small and read-only; callers can loop over candidate pattern
    substitutions without changing model state.
    """
    residual = k_group - pattern_group
    dq = quantize_dequant_k_token_groups(residual, bits=bits, group_size=group_size).dequant
    return float(((k_group.float() - (dq + pattern_group).float()) ** 2).mean().item())
