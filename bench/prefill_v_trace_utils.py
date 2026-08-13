from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from bench.run_prefill_kmeans_stability import (
    align_centroids_to_ref,
    centroid_hungarian_matching,
    cosine,
    difference_rate,
    max_abs,
    relative_l2,
    remap_assignments_to_ref,
)
from models.llama_patternkv import batched_assign_compiled, batched_kmeans_fast_compiled
from models.segmented_cache import (
    ROLLING_CACHE_MODE,
    build_cache_from_prefill,
    pattern_gather_request_centroids,
    pattern_select_v_candidate,
    quantize_pack_v_reference,
)
from quant.batch_invariant_kproj import batch_invariant_k_projection_v2


TRACE_ITERS = (0, 1, 2, 4, 8, 16, 30)


def tensor_metric_dict(got: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    return {
        "exact": bool(torch.equal(got, ref)),
        "max_abs": max_abs(got, ref),
        "mean_abs": float((got.float() - ref.float()).abs().mean().item()) if got.numel() else 0.0,
        "relative_l2": relative_l2(got, ref),
        "cosine": cosine(got, ref),
    }


def element_difference_rate(got: torch.Tensor | None, ref: torch.Tensor | None) -> float | None:
    if got is None or ref is None:
        return None if got is not ref else 0.0
    if got.shape != ref.shape:
        common = min(got.numel(), ref.numel())
        if common == 0:
            return 1.0
        return 1.0
    return difference_rate(got, ref)


def v_centroid_amplification(centroid_relative_l2: float | None, input_relative_l2: float | None) -> float | str | None:
    if centroid_relative_l2 is None or input_relative_l2 is None:
        return None
    if input_relative_l2 == 0.0:
        return "inf" if centroid_relative_l2 != 0.0 else None
    return centroid_relative_l2 / input_relative_l2


def kmeans_initial_indices(heads: int, tokens: int, k: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    scores = torch.rand(heads, tokens, generator=generator, device=device)
    return scores.topk(k, dim=1).indices


@torch.no_grad()
def trace_reference_kmeans(X: torch.Tensor, k: int, *, iters: int = 30, tol: float = 1e-4, seed: int = 0) -> dict[str, Any]:
    H, N, D = X.shape
    generator = torch.Generator(device=X.device)
    generator.manual_seed(seed)
    scores = torch.rand(H, N, generator=generator, device=X.device)
    init_idx = scores.topk(k, dim=1).indices
    centroids = torch.gather(X, 1, init_idx.unsqueeze(-1).expand(-1, -1, D)).contiguous()
    snapshots = {0: centroids.clone()}
    shifts: dict[int, float] = {}
    populations: dict[int, torch.Tensor] = {}
    assignments: dict[int, torch.Tensor] = {}
    x2 = (X * X).sum(-1, keepdim=True)
    sums = torch.empty(H, k, D, device=X.device, dtype=X.dtype)
    counts = torch.empty(H, k, device=X.device, dtype=X.dtype)
    ones = torch.ones(H, N, device=X.device, dtype=X.dtype)
    last_shift = None
    final_iter = 0
    for iteration in range(1, iters + 1):
        c2 = (centroids * centroids).sum(-1).unsqueeze(1)
        d2 = torch.baddbmm(x2 + c2, X, centroids.transpose(1, 2), beta=1.0, alpha=-2.0)
        assign = d2.argmin(dim=-1)
        sums.zero_()
        counts.zero_()
        sums.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), X)
        counts.scatter_add_(1, assign, ones)
        empty = counts == 0
        counts_safe = counts.clamp_min(1.0).unsqueeze(-1)
        new_centroids = sums / counts_safe
        if empty.any():
            rand_idx = torch.randint(0, N, (H, k), generator=generator, device=X.device)
            repl = torch.gather(X, 1, rand_idx.unsqueeze(-1).expand(-1, -1, D))
            new_centroids = torch.where(empty.unsqueeze(-1), repl, new_centroids)
        shift = (new_centroids - centroids).abs().amax()
        centroids = new_centroids
        final_iter = iteration
        if iteration in TRACE_ITERS:
            snapshots[iteration] = centroids.clone()
            populations[iteration] = counts.clone()
            assignments[iteration] = assign.clone()
            shifts[iteration] = float(shift.item())
        if last_shift is not None and shift <= tol:
            break
        last_shift = shift
    assign_final = batched_assign_compiled(X, centroids).to(torch.long)
    return {
        "initial_indices": init_idx,
        "snapshots": snapshots,
        "populations": populations,
        "assignments": assignments,
        "shifts": shifts,
        "final_centroids": centroids,
        "final_assignments": assign_final,
        "final_iter": final_iter,
    }


def validate_reference_kmeans(X: torch.Tensor, ref_centroids: torch.Tensor) -> dict[str, Any]:
    _unused, prod_centroids = batched_kmeans_fast_compiled(X, k=ref_centroids.shape[1], iters=30, tol=1e-4, seed=0)
    ref_assign = batched_assign_compiled(X, ref_centroids.float()).to(torch.long)
    prod_assign = batched_assign_compiled(X, prod_centroids.float()).to(torch.long)
    return {
        "reference_vs_production_centroid_rel_l2": relative_l2(ref_centroids, prod_centroids),
        "reference_vs_production_assignment_diff": difference_rate(ref_assign, prod_assign),
    }


def centroid_alignment_metrics(ref: torch.Tensor, got: torch.Tensor) -> dict[str, Any]:
    ref_to_got, got_to_ref, match_rows = centroid_hungarian_matching(ref, got)
    aligned = align_centroids_to_ref(got, ref_to_got)
    raw_rel = relative_l2(got, ref)
    aligned_rel = relative_l2(aligned, ref)
    raw_sq = torch.linalg.vector_norm((got - ref).float()).pow(2)
    aligned_sq = torch.linalg.vector_norm((aligned - ref).float()).pow(2)
    explained = None if float(raw_sq.item()) == 0.0 else float((1.0 - aligned_sq / raw_sq).item())
    return {
        "raw_v_centroid_relative_l2": raw_rel,
        "aligned_v_centroid_relative_l2": aligned_rel,
        "raw_v_centroid_max_abs": max_abs(got, ref),
        "aligned_v_centroid_max_abs": max_abs(aligned, ref),
        "permutation_explained_fraction": explained,
        "label_permutation_dominant": bool(explained is not None and explained > 0.5),
        "ref_to_got": ref_to_got,
        "got_to_ref": got_to_ref,
        "aligned_centroids": aligned,
        "match_rows": match_rows,
    }


def assignment_metrics(ref_assign: torch.Tensor, got_assign: torch.Tensor, got_to_ref: torch.Tensor | None = None) -> dict[str, Any]:
    raw = difference_rate(got_assign, ref_assign)
    aligned = None
    if got_to_ref is not None:
        aligned = difference_rate(remap_assignments_to_ref(got_assign, got_to_ref), ref_assign)
    return {"raw_assignment_difference_rate": raw, "aligned_assignment_difference_rate": aligned}


def run_v_state(
    value_states: torch.Tensor,
    centroids: torch.Tensor,
    *,
    value_objective: str,
    group_size: int,
    bits: int,
    v4_budget_fraction: float,
    sink_length: int = 16,
    recent_length: int = 128,
) -> dict[str, Any]:
    assign = batched_assign_compiled(value_states[0].float(), centroids.float()).view(1, value_states.shape[1], value_states.shape[2]).to(torch.long)
    idx, mask, aux = pattern_select_v_candidate(
        value_states,
        centroids.to(value_states.dtype),
        value_objective=value_objective,
        group_size=group_size,
        bits=bits,
    )
    selected = pattern_gather_request_centroids(idx, centroids.to(value_states.dtype).unsqueeze(0))
    residualized = value_states - mask.unsqueeze(-1).to(value_states.dtype) * selected
    dummy_k = torch.zeros_like(value_states)
    dummy_k_assign = torch.zeros_like(idx)
    cache = build_cache_from_prefill(
        dummy_k,
        value_states,
        sink_length=sink_length,
        recent_length=recent_length,
        group_size=group_size,
        k_bits=2,
        v_bits=bits,
        pattern=True,
        k_centroids=torch.zeros_like(centroids),
        v_centroids=centroids.to(value_states.dtype),
        k_assignments=dummy_k_assign,
        v_assignment_idx=idx,
        v_pattern_mask=mask.to(torch.uint8),
        cache_mode=ROLLING_CACHE_MODE,
        chunk_length=group_size,
        value_objective=value_objective,
        v_precision_selector="causal_v4",
        v4_budget_fraction=v4_budget_fraction,
        random_selector_seed=20260809,
        selector_layer_idx=0,
    )
    return {
        "assign": assign,
        "idx": idx,
        "mask": mask,
        "selected": selected,
        "residualized": residualized,
        "aux": aux,
        "cache": cache,
    }


def packed_v_metrics(got_cache: Any, ref_cache: Any) -> dict[str, Any]:
    return {
        "packed_v_payload_difference_rate": element_difference_rate(got_cache.packed_v, ref_cache.packed_v),
        "packed_v_scale_relative_l2": None if got_cache.packed_v_scale is None or ref_cache.packed_v_scale is None else relative_l2(got_cache.packed_v_scale, ref_cache.packed_v_scale),
        "packed_v_zero_relative_l2": None if got_cache.packed_v_zero is None or ref_cache.packed_v_zero is None else relative_l2(got_cache.packed_v_zero, ref_cache.packed_v_zero),
        "packed_v4_payload_difference_rate": element_difference_rate(got_cache.packed_v4, ref_cache.packed_v4),
        "packed_v4_scale_relative_l2": None if got_cache.packed_v4_scale is None or ref_cache.packed_v4_scale is None else relative_l2(got_cache.packed_v4_scale, ref_cache.packed_v4_scale),
        "packed_v4_zero_relative_l2": None if got_cache.packed_v4_zero is None or ref_cache.packed_v4_zero is None else relative_l2(got_cache.packed_v4_zero, ref_cache.packed_v4_zero),
        "v_precision_mask_difference_rate": element_difference_rate(got_cache.v_precision_mask, ref_cache.v_precision_mask),
        "v2_stream_tokens_ref": int(ref_cache.packed_v_tokens - (ref_cache.v_precision_mask.bool().sum().item() if ref_cache.v_precision_mask is not None else 0)),
        "v2_stream_tokens_got": int(got_cache.packed_v_tokens - (got_cache.v_precision_mask.bool().sum().item() if got_cache.v_precision_mask is not None else 0)),
        "v4_stream_tokens_ref": int(ref_cache.v_precision_mask.bool().sum().item()) if ref_cache.v_precision_mask is not None else 0,
        "v4_stream_tokens_got": int(got_cache.v_precision_mask.bool().sum().item()) if got_cache.v_precision_mask is not None else 0,
    }


def fixed_centroid_control(
    ref_value: torch.Tensor,
    got_value: torch.Tensor,
    ref_centroids: torch.Tensor,
    *,
    value_objective: str,
    group_size: int,
    bits: int,
    v4_budget_fraction: float,
    sink_length: int = 16,
    recent_length: int = 128,
) -> dict[str, Any]:
    ref_state = run_v_state(ref_value, ref_centroids, value_objective=value_objective, group_size=group_size, bits=bits, v4_budget_fraction=v4_budget_fraction, sink_length=sink_length, recent_length=recent_length)
    got_state = run_v_state(got_value, ref_centroids, value_objective=value_objective, group_size=group_size, bits=bits, v4_budget_fraction=v4_budget_fraction, sink_length=sink_length, recent_length=recent_length)
    packed = packed_v_metrics(got_state["cache"], ref_state["cache"])
    return {
        "fixed_centroid_assignment_difference_rate": difference_rate(got_state["idx"], ref_state["idx"]),
        "fixed_centroid_mask_difference_rate": difference_rate(got_state["mask"], ref_state["mask"]),
        "fixed_centroid_selected_relative_l2": relative_l2(got_state["selected"], ref_state["selected"]),
        "fixed_centroid_residualized_v_relative_l2": relative_l2(got_state["residualized"], ref_state["residualized"]),
        **packed,
    }


def bi_vproj_control(hidden_b1: torch.Tensor, hidden_b2: torch.Tensor, hidden_b4: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> dict[str, torch.Tensor | bool]:
    out_b1 = batch_invariant_k_projection_v2(hidden_b1, weight, bias)
    out_b2 = batch_invariant_k_projection_v2(hidden_b2, weight, bias)
    out_b4 = batch_invariant_k_projection_v2(hidden_b4, weight, bias)
    return {
        "b1": out_b1,
        "b2_row0": out_b2[0:1],
        "b4_row0": out_b4[0:1],
        "b1_b2_exact": bool(torch.equal(out_b1, out_b2[0:1])),
        "b1_b4_exact": bool(torch.equal(out_b1, out_b4[0:1])),
    }
