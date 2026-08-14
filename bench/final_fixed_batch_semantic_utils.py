from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch


CHECKPOINTS = (0, 1, 2, 16, 127, 128, 129, 255, 256, 257)
BOUNDARY_STEPS = (128, 256)


@dataclass(frozen=True)
class StructuralLayerState:
    request: str
    batch_mode: str
    step: int
    layer: int
    total_tokens: int
    sink_tokens: int
    recent_tokens: int
    pending_tokens: int
    packed_k_tokens: int
    packed_v_tokens: int
    packed_v4_tokens: int
    k_centroid_count: int | None
    v_centroid_count: int | None
    k_update_count: int | None
    v_update_count: int | None
    last_flush_pos: int | None
    page_count: int | None
    last_page_valid_tokens: int | None
    slot_id: int | None = None


def forced_replay_step_tokens(reference_continuations: dict[str, list[int]], requests: list[str], step: int) -> list[int]:
    if step <= 0:
        raise ValueError("decode step must be positive")
    tokens = []
    for request in requests:
        continuation = reference_continuations[request]
        if len(continuation) < step:
            raise ValueError(f"request {request} has only {len(continuation)} reference tokens, need step {step}")
        tokens.append(int(continuation[step - 1]))
    return tokens


def compare_float_tensors(ref: torch.Tensor, got: torch.Tensor) -> dict[str, Any]:
    if tuple(ref.shape) != tuple(got.shape):
        return {
            "shape_equal": False,
            "ref_shape": list(ref.shape),
            "got_shape": list(got.shape),
            "comparable": False,
            "exact": None,
            "relative_l2": None,
            "max_abs": None,
            "mean_abs": None,
            "cosine": None,
            "nan": bool(torch.isnan(got).any().item()) if got.is_floating_point() else False,
            "inf": bool(torch.isinf(got).any().item()) if got.is_floating_point() else False,
        }
    ref_f = ref.detach().float().cpu().contiguous()
    got_f = got.detach().float().cpu().contiguous()
    if ref_f.numel() == 0:
        return {
            "shape_equal": True,
            "ref_shape": list(ref.shape),
            "got_shape": list(got.shape),
            "comparable": True,
            "exact": True,
            "relative_l2": 0.0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "cosine": 1.0,
            "nan": False,
            "inf": False,
        }
    diff = got_f - ref_f
    ref_norm = torch.linalg.vector_norm(ref_f).clamp_min(1e-12)
    got_norm = torch.linalg.vector_norm(got_f).clamp_min(1e-12)
    cosine = torch.sum(ref_f * got_f) / (ref_norm * got_norm)
    return {
        "shape_equal": True,
        "ref_shape": list(ref.shape),
        "got_shape": list(got.shape),
        "comparable": True,
        "exact": bool(torch.equal(ref.detach().cpu(), got.detach().cpu())),
        "relative_l2": float((torch.linalg.vector_norm(diff) / ref_norm).item()),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "cosine": float(cosine.item()),
        "nan": bool(torch.isnan(got_f).any().item()),
        "inf": bool(torch.isinf(got_f).any().item()),
    }


def topk_logit_metrics(ref_logits: torch.Tensor, got_logits: torch.Tensor, *, k: int = 5) -> dict[str, Any]:
    ref = ref_logits.detach().float().cpu()
    got = got_logits.detach().float().cpu()
    ref_top = torch.topk(ref, k=min(k, ref.numel()))
    got_top = torch.topk(got, k=min(k, got.numel()))
    ref_top1 = int(ref_top.indices[0].item())
    got_top1 = int(got_top.indices[0].item())
    margin = float((ref_top.values[0] - ref_top.values[1]).item()) if ref_top.values.numel() > 1 else math.inf
    overlap = len(set(int(x) for x in ref_top.indices.tolist()) & set(int(x) for x in got_top.indices.tolist()))
    return {
        "ref_top1": ref_top1,
        "got_top1": got_top1,
        "top1_equal": ref_top1 == got_top1,
        "top5_overlap": overlap,
        "reference_top1_margin": margin,
    }


def difference_rate(ref: torch.Tensor | None, got: torch.Tensor | None) -> float | None:
    if ref is None and got is None:
        return 0.0
    if ref is None or got is None or tuple(ref.shape) != tuple(got.shape):
        return None
    ref_c = ref.detach().cpu().contiguous()
    got_c = got.detach().cpu().contiguous()
    if ref_c.numel() == 0:
        return 0.0
    return float((ref_c != got_c).sum().item()) / float(ref_c.numel())


def validate_assignment_index_range(indices: torch.Tensor | None, centroid_count: int | None) -> bool:
    if indices is None:
        return True
    if centroid_count is None or centroid_count <= 0:
        return False
    logical = indices.detach()
    if logical.numel() == 0:
        return True
    return bool(((logical >= 0) & (logical < int(centroid_count))).all().item())


def validate_v4_budget(packed_v4_tokens: int, packed_v_tokens: int, budget_fraction: float, *, slack: int = 1) -> bool:
    if packed_v4_tokens < 0 or packed_v_tokens < 0:
        return False
    budget = int(math.ceil(float(packed_v_tokens) * float(budget_fraction))) + int(slack)
    return int(packed_v4_tokens) <= budget


def validate_logical_counts(state: StructuralLayerState) -> bool:
    return (
        state.total_tokens >= 0
        and state.sink_tokens >= 0
        and state.recent_tokens >= 0
        and state.pending_tokens >= 0
        and state.packed_k_tokens >= 0
        and state.packed_v_tokens >= 0
        and state.packed_v4_tokens >= 0
        and state.sink_tokens + state.packed_k_tokens + state.pending_tokens + state.recent_tokens == state.total_tokens
        and state.packed_v_tokens == state.packed_k_tokens
    )


def canonical_structural_metadata(state: StructuralLayerState) -> dict[str, Any]:
    return {
        "total_tokens": state.total_tokens,
        "sink_tokens": state.sink_tokens,
        "recent_tokens": state.recent_tokens,
        "pending_tokens": state.pending_tokens,
        "packed_k_tokens": state.packed_k_tokens,
        "packed_v_tokens": state.packed_v_tokens,
        "packed_v4_tokens": state.packed_v4_tokens,
        "k_centroid_count": state.k_centroid_count,
        "v_centroid_count": state.v_centroid_count,
        "k_update_count": state.k_update_count,
        "v_update_count": state.v_update_count,
        "last_flush_pos": state.last_flush_pos,
        "page_count": state.page_count,
        "last_page_valid_tokens": state.last_page_valid_tokens,
    }


def structural_cross_batch_equal(ref: StructuralLayerState, got: StructuralLayerState) -> bool:
    return canonical_structural_metadata(ref) == canonical_structural_metadata(got)


def validate_request_slot_mapping(states: Iterable[StructuralLayerState]) -> bool:
    by_layer_step: dict[tuple[str, int, int], set[int]] = {}
    for state in states:
        if state.slot_id is None:
            continue
        key = (state.batch_mode, state.step, state.layer)
        by_layer_step.setdefault(key, set()).add(int(state.slot_id))
    return all(len(slots) == sum(1 for state in states if (state.batch_mode, state.step, state.layer) == key and state.slot_id is not None) for key, slots in by_layer_step.items())


def boundary_explosion(values_by_step: dict[int, float | None], boundary: int, *, epsilon: float = 1e-6, absolute_delta: float = 1e-3, ratio: float = 5.0) -> dict[str, Any]:
    before = values_by_step.get(boundary - 1)
    at = values_by_step.get(boundary)
    after = values_by_step.get(boundary + 1)
    nearby = max([float(v) for v in (before, after) if v is not None] or [0.0])
    if at is None:
        return {"boundary": boundary, "explosion": False, "value": None, "nearby_baseline": nearby, "absolute_increase": None, "ratio": None}
    denom = max(nearby, epsilon)
    increase = float(at) - nearby
    ratio_value = float(at) / denom
    explosion = bool(float(at) > ratio * denom and increase > absolute_delta)
    return {
        "boundary": boundary,
        "explosion": explosion,
        "value": float(at),
        "nearby_baseline": nearby,
        "absolute_increase": increase,
        "ratio": ratio_value,
    }


def semantic_gate_bounded(metric_series: dict[str, dict[int, float | None]]) -> dict[str, Any]:
    explosions = []
    finite = True
    for name, series in metric_series.items():
        for value in series.values():
            if value is not None and not math.isfinite(float(value)):
                finite = False
        for boundary in BOUNDARY_STEPS:
            item = boundary_explosion(series, boundary)
            item["metric"] = name
            explosions.append(item)
    return {
        "finite": finite,
        "boundary_explosion": any(item["explosion"] for item in explosions),
        "explosions": explosions,
        "bounded": finite and not any(item["explosion"] for item in explosions),
    }


def final_gate_requires_bi_k_mode(mode: str, bi_mlp_oracle_calls: int) -> bool:
    return str(mode) == "bi_k" and int(bi_mlp_oracle_calls) == 0
