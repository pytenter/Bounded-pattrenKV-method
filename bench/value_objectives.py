from __future__ import annotations

from dataclasses import dataclass

import torch


EPS = 1e-8


@dataclass(frozen=True)
class CandidateDecision:
    indices: torch.Tensor
    costs: torch.Tensor
    tie_break_costs: torch.Tensor


def normalized_reconstruction_error(source: torch.Tensor, reconstruction: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    src = source.detach().float()
    rec = reconstruction.detach().float()
    numerator = (src - rec).pow(2).sum(dim=-1)
    denominator = src.pow(2).sum(dim=-1).clamp_min(float(eps))
    return numerator / denominator


def direction_error(source: torch.Tensor, reconstruction: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    src = source.detach().float()
    rec = reconstruction.detach().float()
    src_norm = src.norm(dim=-1)
    rec_norm = rec.norm(dim=-1)
    valid = (src_norm >= float(eps)) & (rec_norm >= float(eps))
    cosine = (src * rec).sum(dim=-1) / (src_norm.clamp_min(float(eps)) * rec_norm.clamp_min(float(eps)))
    return torch.where(valid, 1.0 - cosine.clamp(-1.0, 1.0), normalized_reconstruction_error(src, rec, eps=eps))


def objective_costs(
    source: torch.Tensor,
    candidate_reconstructions: torch.Tensor,
    *,
    objective: str,
    eps: float = EPS,
) -> torch.Tensor:
    """Return per-candidate costs.

    `candidate_reconstructions` is shaped `[C, ..., D]`; `source` is shaped
    `[..., D]`.  Costs are `[C, ...]` and use the same feasible candidate set.
    """
    objective = objective.strip().lower().replace("-", "_")
    if candidate_reconstructions.dim() < 2:
        raise ValueError("candidate_reconstructions must be [C, ..., D]")
    expanded_source = source.unsqueeze(0).expand_as(candidate_reconstructions)
    nre = normalized_reconstruction_error(expanded_source, candidate_reconstructions, eps=eps)
    direction = direction_error(expanded_source, candidate_reconstructions, eps=eps)
    if objective in {"base", "baseline", "mse", "nre"}:
        return nre
    if objective in {"v_dir", "dir", "direction"}:
        return direction
    if objective in {"v_hybrid", "hybrid"}:
        return nre + direction
    raise ValueError(f"unsupported value objective: {objective}")


def select_lowest_cost(costs: torch.Tensor, *, tie_break_costs: torch.Tensor | None = None, atol: float = 0.0) -> CandidateDecision:
    if costs.dim() < 1:
        raise ValueError("costs must include a candidate dimension")
    primary_min = costs.min(dim=0).values
    if tie_break_costs is None:
        tie_break_costs = torch.zeros_like(costs)
    if tie_break_costs.shape != costs.shape:
        raise ValueError("tie_break_costs must have the same shape as costs")
    eligible = costs <= primary_min.unsqueeze(0) + float(atol)
    masked_tie = torch.where(eligible, tie_break_costs, torch.full_like(tie_break_costs, float("inf")))
    indices = masked_tie.argmin(dim=0)
    chosen_costs = torch.gather(costs, 0, indices.unsqueeze(0)).squeeze(0)
    chosen_tie = torch.gather(tie_break_costs, 0, indices.unsqueeze(0)).squeeze(0)
    return CandidateDecision(indices=indices, costs=chosen_costs, tie_break_costs=chosen_tie)


def choose_value_candidate(
    source: torch.Tensor,
    candidate_reconstructions: torch.Tensor,
    *,
    objective: str,
    tie_break: str = "nre",
    eps: float = EPS,
    atol: float = 0.0,
) -> CandidateDecision:
    costs = objective_costs(source, candidate_reconstructions, objective=objective, eps=eps)
    tie_costs = objective_costs(source, candidate_reconstructions, objective=tie_break, eps=eps)
    return select_lowest_cost(costs, tie_break_costs=tie_costs, atol=atol)


def causal_weighted_tile_cost(
    source_vectors: torch.Tensor,
    candidate_reconstructions: torch.Tensor,
    importance: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """Weighted mean of NRE + DIR over a coupled tile.

    This only changes decisions when one candidate jointly affects multiple
    token vectors.  For independent per-token decisions, a token scalar weight
    cancels out of the argmin.
    """
    per_vector = objective_costs(source_vectors, candidate_reconstructions, objective="v_hybrid", eps=eps)
    weights = importance.detach().float().clamp_min(float(eps))
    weights = weights / weights.mean().clamp_min(float(eps))
    while weights.dim() < per_vector.dim():
        weights = weights.unsqueeze(0)
    weighted = per_vector * weights
    denom = weights.sum(dim=tuple(range(1, weights.dim()))).clamp_min(float(eps))
    return weighted.sum(dim=tuple(range(1, weighted.dim()))) / denom


def per_token_causal_weighted_costs(
    source: torch.Tensor,
    candidate_reconstructions: torch.Tensor,
    importance: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    costs = objective_costs(source, candidate_reconstructions, objective="v_hybrid", eps=eps)
    weights = importance.detach().float().clamp_min(float(eps))
    weights = weights / weights.mean().clamp_min(float(eps))
    return costs * weights.unsqueeze(0)
