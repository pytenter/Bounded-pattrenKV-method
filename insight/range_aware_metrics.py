"""Pure helpers for range-aware PatternKV assignment diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch


RANGE_EPSILON = 1e-12


class QuantizationLayout(str, Enum):
    """Explicit diagnostic layouts matching real quantization units."""

    V_PER_TOKEN = "v_per_token"
    K_TOKEN_GROUP_PER_CHANNEL = "k_token_group_per_channel"


@dataclass(frozen=True)
class AggregateStats:
    """Streaming-style statistics for one scalar series."""

    count: int
    total: float
    sum_sq: float
    min_value: float
    max_value: float

    @classmethod
    def from_tensor(cls, values: torch.Tensor) -> "AggregateStats":
        if values.numel() == 0:
            raise ValueError("aggregate requires at least one value")
        if not torch.isfinite(values).all():
            raise ValueError("aggregate values contain NaN or Inf")
        arr = values.detach().float().reshape(-1)
        return cls(
            count=int(arr.numel()),
            total=float(arr.sum().item()),
            sum_sq=float((arr * arr).sum().item()),
            min_value=float(arr.amin().item()),
            max_value=float(arr.amax().item()),
        )

    def merge(self, other: "AggregateStats") -> "AggregateStats":
        return AggregateStats(
            count=self.count + other.count,
            total=self.total + other.total,
            sum_sq=self.sum_sq + other.sum_sq,
            min_value=min(self.min_value, other.min_value),
            max_value=max(self.max_value, other.max_value),
        )

    @property
    def mean(self) -> float:
        return self.total / self.count

    def to_json(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "sum": self.total,
            "sum_sq": self.sum_sq,
            "min": self.min_value,
            "max": self.max_value,
            "mean": self.mean,
        }


@dataclass(frozen=True)
class AssignmentDiagnostics:
    """Assignment-level diagnostic tensors and aggregate summaries."""

    total_count: int
    mismatch_count: int
    l2_range: torch.Tensor
    minmax_range: torch.Tensor
    range_gain_absolute: torch.Tensor
    range_regret: torch.Tensor
    l2_assignment: torch.Tensor
    minmax_assignment: torch.Tensor

    @property
    def mismatch_rate(self) -> float:
        return float(self.mismatch_count / self.total_count) if self.total_count else 0.0

    def aggregates(self) -> dict[str, AggregateStats]:
        return {
            "l2_residual_range": AggregateStats.from_tensor(self.l2_range),
            "minmax_residual_range": AggregateStats.from_tensor(self.minmax_range),
            "range_gain_absolute": AggregateStats.from_tensor(self.range_gain_absolute),
            "range_regret": AggregateStats.from_tensor(self.range_regret),
        }


def l2_assign_tokens(tokens: torch.Tensor, centroids: torch.Tensor, *, chunk_size: int = 0) -> torch.Tensor:
    """Argmin over squared L2 distance with torch.argmin tie-breaking."""
    return _argmin_over_candidates(tokens, centroids, score="l2", chunk_size=chunk_size)


def minmax_assign_tokens(tokens: torch.Tensor, centroids: torch.Tensor, *, chunk_size: int = 0) -> torch.Tensor:
    """Argmin over residual range with torch.argmin tie-breaking."""
    return _argmin_over_candidates(tokens, centroids, score="range", chunk_size=chunk_size)


def _argmin_over_candidates(tokens: torch.Tensor, centroids: torch.Tensor, *, score: str, chunk_size: int) -> torch.Tensor:
    if tokens.dim() != 2 or centroids.dim() != 2:
        raise ValueError(f"expected [N,D] tokens and [M,D] centroids, got {tuple(tokens.shape)} and {tuple(centroids.shape)}")
    if tokens.shape[-1] != centroids.shape[-1]:
        raise ValueError("token and centroid head_dim must match")
    if chunk_size <= 0:
        chunk_size = int(centroids.shape[0])
    best_score = None
    best_idx = None
    for start in range(0, int(centroids.shape[0]), chunk_size):
        chunk = centroids[start : start + chunk_size]
        diff = tokens[:, None, :].float() - chunk[None, :, :].float()
        if score == "l2":
            values = (diff * diff).sum(dim=-1)
        elif score == "range":
            values = diff.amax(dim=-1) - diff.amin(dim=-1)
        else:
            raise ValueError(f"unsupported score={score}")
        local_score, local_idx = values.min(dim=-1)
        local_idx = local_idx + start
        if best_score is None:
            best_score = local_score
            best_idx = local_idx
            continue
        better = local_score < best_score
        best_score = torch.where(better, local_score, best_score)
        best_idx = torch.where(better, local_idx, best_idx)
    assert best_idx is not None
    return best_idx.to(torch.long)


def compute_v_assignment_diagnostics(
    vectors: torch.Tensor,
    patterns: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    epsilon: float = RANGE_EPSILON,
    chunk_size: int = 0,
) -> AssignmentDiagnostics:
    """Compute V diagnostics on per-token head_dim vectors."""
    if vectors.dim() != 2 or patterns.dim() != 2:
        raise ValueError("V diagnostics expect [N,D] vectors and [M,D] patterns")
    mask = _normalize_mask(valid_mask, vectors.shape[0], vectors.device)
    vectors_valid = vectors[mask]
    if vectors_valid.numel() == 0:
        raise ValueError("no valid V vectors for diagnostics")
    l2_idx = l2_assign_tokens(vectors_valid, patterns, chunk_size=chunk_size)
    mm_idx = minmax_assign_tokens(vectors_valid, patterns, chunk_size=chunk_size)
    l2_range = _token_ranges(vectors_valid, patterns[l2_idx])
    mm_range = _token_ranges(vectors_valid, patterns[mm_idx])
    return _finalize_assignment_diagnostics(l2_idx, mm_idx, l2_range, mm_range, epsilon=epsilon)


def compute_k_group_assignment_diagnostics(
    groups: torch.Tensor,
    patterns: torch.Tensor,
    *,
    valid_group_mask: torch.Tensor | None = None,
    epsilon: float = RANGE_EPSILON,
    chunk_size: int = 0,
) -> AssignmentDiagnostics:
    """Compute K diagnostics on full 128-token groups projected to per-channel units.

    `groups` must be `[G,T,D]` where `T` is the token-group length and `D` is head_dim.
    L2/min-max assignments are computed with the real token-wise PatternKV formulas and
    then projected onto the real quantization unit: per-channel range across the token group.
    """
    if groups.dim() != 3 or patterns.dim() != 2:
        raise ValueError("K diagnostics expect [G,T,D] groups and [M,D] patterns")
    if groups.shape[-1] != patterns.shape[-1]:
        raise ValueError("group head_dim and pattern head_dim must match")
    mask = _normalize_mask(valid_group_mask, groups.shape[0], groups.device)
    groups_valid = groups[mask]
    if groups_valid.numel() == 0:
        raise ValueError("no valid K groups for diagnostics")
    flat = groups_valid.reshape(-1, groups_valid.shape[-1])
    l2_idx = l2_assign_tokens(flat, patterns, chunk_size=chunk_size).view(groups_valid.shape[0], groups_valid.shape[1])
    mm_idx = minmax_assign_tokens(flat, patterns, chunk_size=chunk_size).view(groups_valid.shape[0], groups_valid.shape[1])
    l2_patterns = patterns[l2_idx]
    mm_patterns = patterns[mm_idx]
    l2_residual = groups_valid.float() - l2_patterns.float()
    mm_residual = groups_valid.float() - mm_patterns.float()
    l2_range = l2_residual.amax(dim=1) - l2_residual.amin(dim=1)
    mm_range = mm_residual.amax(dim=1) - mm_residual.amin(dim=1)
    mismatch = (l2_idx != mm_idx).any(dim=1, keepdim=True).expand_as(l2_range)
    return _finalize_assignment_diagnostics(
        mismatch.to(torch.long),
        torch.ones_like(mismatch, dtype=torch.long),
        l2_range.reshape(-1),
        mm_range.reshape(-1),
        epsilon=epsilon,
        precomputed_mismatch=True,
    )


def _finalize_assignment_diagnostics(
    l2_assignment: torch.Tensor,
    minmax_assignment: torch.Tensor,
    l2_range: torch.Tensor,
    minmax_range: torch.Tensor,
    *,
    epsilon: float,
    precomputed_mismatch: bool = False,
) -> AssignmentDiagnostics:
    l2_flat = l2_range.detach().float().reshape(-1)
    mm_flat = minmax_range.detach().float().reshape(-1)
    if not torch.isfinite(l2_flat).all() or not torch.isfinite(mm_flat).all():
        raise ValueError("diagnostic ranges contain NaN or Inf")
    gain = l2_flat - mm_flat
    regret = gain / (l2_flat + float(epsilon))
    if not torch.isfinite(regret).all():
        raise ValueError("diagnostic regret contains NaN or Inf")
    if precomputed_mismatch:
        mismatch_count = int(l2_assignment.detach().reshape(-1).sum().item())
        total_count = int(minmax_assignment.detach().reshape(-1).sum().item())
    else:
        mismatch = l2_assignment.detach().reshape(-1) != minmax_assignment.detach().reshape(-1)
        mismatch_count = int(mismatch.sum().item())
        total_count = int(mismatch.numel())
    return AssignmentDiagnostics(
        total_count=total_count,
        mismatch_count=mismatch_count,
        l2_range=l2_flat,
        minmax_range=mm_flat,
        range_gain_absolute=gain,
        range_regret=regret,
        l2_assignment=l2_assignment.detach().clone(),
        minmax_assignment=minmax_assignment.detach().clone(),
    )


def _token_ranges(vectors: torch.Tensor, patterns: torch.Tensor) -> torch.Tensor:
    residual = vectors.float() - patterns.float()
    return residual.amax(dim=-1) - residual.amin(dim=-1)


def _normalize_mask(mask: torch.Tensor | None, length: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(length, dtype=torch.bool, device=device)
    if mask.dim() != 1 or mask.shape[0] != length:
        raise ValueError(f"valid mask must be [N], got {tuple(mask.shape)} for length={length}")
    return mask.to(device=device, dtype=torch.bool)


def merge_stats(items: Iterable[AggregateStats]) -> AggregateStats:
    items = list(items)
    if not items:
        raise ValueError("merge_stats requires at least one item")
    current = items[0]
    for item in items[1:]:
        current = current.merge(item)
    return current
