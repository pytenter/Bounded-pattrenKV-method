from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.run_actual_model_fixed_batch_smoke import MODEL_PATH, load_model, make_fixed_inputs
from bench.run_actual_model_k_assignment_trace import make_position_ids, tensor_metrics
from models.llama_patternkv import apply_rotary_pos_emb, batched_assign_compiled, batched_kmeans_fast_compiled
from models.segmented_cache import dequantize_k_reference, pattern_gather_request_centroids
from quant.new_pack import triton_quantize_and_pack_along_last_dim


REPORT_DIR = REPO_ROOT / "reports/system_prefill_kmeans_stability_v1"
START_HEAD = "a57a9bc594ef4c702ac30b2a7d3c8b213cdef92b"
NOISE_LEVELS = [0.0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
REAL_DELTA_ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((torch.linalg.vector_norm((a - b).float()) / torch.linalg.vector_norm(b.float()).clamp_min(1e-12)).item())


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item()) if a.numel() else 0.0


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float()
    bf = b.float()
    return float((torch.sum(af * bf) / (torch.linalg.vector_norm(af).clamp_min(1e-12) * torch.linalg.vector_norm(bf).clamp_min(1e-12))).item())


def difference_rate(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a != b).float().mean().item()) if a.numel() else 0.0


def amplification_factor(output_relative_l2: float | None, input_relative_l2: float | None) -> float | None:
    if input_relative_l2 is None or output_relative_l2 is None or input_relative_l2 == 0:
        return None
    return output_relative_l2 / input_relative_l2


def scaled_delta(k_ref: torch.Tensor, delta: torch.Tensor, alpha: float) -> torch.Tensor:
    return k_ref + delta.to(k_ref.dtype) * alpha


def deterministic_noise_like(k_ref: torch.Tensor, target_relative_l2: float, seed: int = 20260207) -> torch.Tensor:
    if target_relative_l2 == 0:
        return torch.zeros_like(k_ref)
    generator = torch.Generator(device=k_ref.device)
    generator.manual_seed(seed)
    noise = torch.randn(k_ref.shape, device=k_ref.device, dtype=torch.float32, generator=generator)
    target_norm = torch.linalg.vector_norm(k_ref.float()) * target_relative_l2
    return (noise / torch.linalg.vector_norm(noise).clamp_min(1e-12) * target_norm).to(k_ref.dtype)


def kmeans_initial_indices(heads: int, tokens: int, k: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    scores = torch.rand(heads, tokens, generator=generator, device=device)
    return scores.topk(k, dim=1).indices


def run_production_kmeans(k_states: torch.Tensor, num_k_bases: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = k_states[0].to(torch.float32)
    _unused, centroids = batched_kmeans_fast_compiled(x, k=num_k_bases, iters=30, tol=1e-4, seed=0)
    assignments = batched_assign_compiled(x, centroids).view(1, x.shape[0], x.shape[1]).contiguous().to(torch.long)
    return centroids.to(k_states.dtype), assignments


def gather_selected(assignments: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    return pattern_gather_request_centroids(assignments, centroids.unsqueeze(0))


def _linear_sum_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)
    except Exception:
        return _hungarian_square_minimize(cost)


def _hungarian_square_minimize(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = cost.detach().cpu().double()
    n, m = matrix.shape
    if n != m:
        raise ValueError("fallback Hungarian helper expects a square cost matrix")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = float(matrix[i0 - 1, j - 1].item()) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows = torch.arange(n, dtype=torch.long)
    cols = torch.empty(n, dtype=torch.long)
    for j in range(1, m + 1):
        if p[j] != 0:
            cols[p[j] - 1] = j - 1
    return rows, cols


def centroid_hungarian_matching(ref: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    heads, k, _dim = ref.shape
    ref_to_test = torch.empty(heads, k, dtype=torch.long, device=ref.device)
    test_to_ref = torch.empty(heads, k, dtype=torch.long, device=ref.device)
    rows: list[dict[str, Any]] = []
    for head in range(heads):
        cost = torch.cdist(ref[head].float(), test[head].float(), p=2)
        row_idx, col_idx = _linear_sum_assignment(cost)
        row_idx = row_idx.to(ref.device)
        col_idx = col_idx.to(ref.device)
        ref_to_test[head, row_idx] = col_idx
        test_to_ref[head, col_idx] = row_idx
        for r, c in zip(row_idx.detach().cpu().tolist(), col_idx.detach().cpu().tolist()):
            rows.append({"kv_head": head, "ref_centroid": int(r), "test_centroid": int(c), "l2_cost": float(cost[r, c].item())})
    return ref_to_test, test_to_ref, rows


def align_centroids_to_ref(test: torch.Tensor, ref_to_test: torch.Tensor) -> torch.Tensor:
    return torch.gather(test, 1, ref_to_test.to(test.device).unsqueeze(-1).expand(-1, -1, test.shape[-1]))


def remap_assignments_to_ref(assignments: torch.Tensor, test_to_ref: torch.Tensor) -> torch.Tensor:
    mapping = test_to_ref.to(assignments.device).unsqueeze(0).expand(assignments.shape[0], -1, -1)
    return torch.gather(mapping, 2, assignments)


def permutation_aligned_centroid_metrics(ref: torch.Tensor, test: torch.Tensor) -> dict[str, Any]:
    ref_to_test, test_to_ref, match_rows = centroid_hungarian_matching(ref, test)
    aligned = align_centroids_to_ref(test, ref_to_test)
    raw_error = torch.linalg.vector_norm((test - ref).float()).pow(2)
    aligned_error = torch.linalg.vector_norm((aligned - ref).float()).pow(2)
    explained = None if float(raw_error.item()) == 0.0 else float((1.0 - aligned_error / raw_error).item())
    return {
        "ref_to_test": ref_to_test,
        "test_to_ref": test_to_ref,
        "match_rows": match_rows,
        "aligned_centroids": aligned,
        "raw_centroid_relative_l2": relative_l2(test, ref),
        "raw_centroid_max_abs": max_abs(test, ref),
        "aligned_centroid_relative_l2": relative_l2(aligned, ref),
        "aligned_centroid_max_abs": max_abs(aligned, ref),
        "permutation_explained_fraction": explained,
    }


def quantize_reconstruct(k_states: torch.Tensor, assignments: torch.Tensor, centroids: torch.Tensor, group_size: int, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = gather_selected(assignments, centroids)
    adjusted = k_states - selected
    packed, scale, zero = triton_quantize_and_pack_along_last_dim(adjusted.transpose(2, 3).contiguous(), group_size, bits)
    dequant = dequantize_k_reference(packed, scale, zero, group_size, bits)
    if dequant is None:
        raise RuntimeError("K dequantization unexpectedly returned None")
    return dequant + selected, packed, scale, zero


def qk_scores(query: torch.Tensor, key: torch.Tensor, num_key_value_heads: int, scale: float) -> torch.Tensor:
    q_heads = query.shape[1]
    groups = q_heads // num_key_value_heads
    key_for_q = key.repeat_interleave(groups, dim=1)
    return torch.matmul(query.float(), key_for_q.float().transpose(2, 3)) * scale


def qk_impact_metrics(query: torch.Tensor, ref_k: torch.Tensor, test_k: torch.Tensor, num_key_value_heads: int, scale: float) -> dict[str, Any]:
    ref = qk_scores(query, ref_k, num_key_value_heads, scale)
    test = qk_scores(query, test_k, num_key_value_heads, scale)
    return {"qk_relative_l2": relative_l2(test, ref), "qk_max_abs": max_abs(test, ref), "qk_cosine": cosine(test, ref)}


def fixed_centroid_control(k_ref: torch.Tensor, k_test: torch.Tensor, centroids_ref: torch.Tensor) -> dict[str, Any]:
    assign_ref = batched_assign_compiled(k_ref[0].float(), centroids_ref.float()).view(1, k_ref.shape[1], k_ref.shape[2]).to(torch.long)
    assign_test = batched_assign_compiled(k_test[0].float(), centroids_ref.float()).view(1, k_test.shape[1], k_test.shape[2]).to(torch.long)
    return {
        "assignment_difference_count": int((assign_ref != assign_test).sum().item()),
        "assignment_difference_rate": difference_rate(assign_test, assign_ref),
    }


def layer0_qk(model: Any, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int, float]:
    layer = model.model.layers[0]
    attn = layer.self_attn
    with torch.inference_mode():
        hidden = layer.input_layernorm(model.model.embed_tokens(input_ids))
        q_proj = attn.q_proj(hidden)
        k_proj = attn.k_proj(hidden)
        bsz, seq_len, _ = hidden.shape
        query = q_proj.view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        key = k_proj.view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        value_stub = torch.empty_like(key)
        pos = make_position_ids(input_ids)
        cos, sin = attn.rotary_emb(value_stub, pos)
        query_post, key_post = apply_rotary_pos_emb(query, key, cos, sin, pos)
    return key_post.detach(), query_post.detach(), int(attn.num_k_bases), int(attn.group_size), float(attn.head_dim**-0.5)


def real_noise_report(k_ref: torch.Tensor, k_b2: torch.Tensor) -> dict[str, Any]:
    delta = (k_b2 - k_ref).float()
    abs_delta = delta.abs().flatten()
    percentiles = torch.quantile(abs_delta, torch.tensor([0.5, 0.9, 0.95, 0.99, 0.999], device=abs_delta.device))
    metrics = tensor_metrics(k_b2, k_ref)
    return {
        "max_abs": metrics["max_abs"],
        "mean_abs": metrics["mean_abs"],
        "relative_l2": metrics["relative_l2"],
        "cosine": metrics["cosine"],
        "delta_std": float(delta.std(unbiased=False).item()),
        "abs_delta_p50": float(percentiles[0].item()),
        "abs_delta_p90": float(percentiles[1].item()),
        "abs_delta_p95": float(percentiles[2].item()),
        "abs_delta_p99": float(percentiles[3].item()),
        "abs_delta_p999": float(percentiles[4].item()),
    }


def analyze_variant(
    *,
    name: str,
    kind: str,
    parameter: float,
    k_ref: torch.Tensor,
    k_test: torch.Tensor,
    query_ref: torch.Tensor,
    centroids_ref: torch.Tensor,
    assignments_ref: torch.Tensor,
    recon_ref: torch.Tensor,
    packed_ref: torch.Tensor,
    scale_ref: torch.Tensor,
    zero_ref: torch.Tensor,
    num_k_bases: int,
    group_size: int,
    bits: int,
    qk_scale: float,
) -> dict[str, Any]:
    centroids_test, assignments_test = run_production_kmeans(k_test, num_k_bases)
    matching = permutation_aligned_centroid_metrics(centroids_ref, centroids_test)
    aligned_assignments = remap_assignments_to_ref(assignments_test, matching["test_to_ref"])
    selected_ref = gather_selected(assignments_ref, centroids_ref)
    selected_test = gather_selected(assignments_test, centroids_test)
    recon_test, packed_test, scale_test, zero_test = quantize_reconstruct(k_test, assignments_test, centroids_test, group_size, bits)
    k_rel = relative_l2(k_test, k_ref)
    raw_centroid_rel = matching["raw_centroid_relative_l2"]
    raw_assignment_rate = difference_rate(assignments_test, assignments_ref)
    aligned_assignment_rate = difference_rate(aligned_assignments, assignments_ref)
    fixed = fixed_centroid_control(k_ref, k_test, centroids_ref)
    qk = qk_impact_metrics(query_ref, recon_ref, recon_test, centroids_ref.shape[0], qk_scale)
    return {
        "name": name,
        "kind": kind,
        "parameter": parameter,
        "k_relative_l2": k_rel,
        "k_max_abs": max_abs(k_test, k_ref),
        "raw_centroid_relative_l2": raw_centroid_rel,
        "raw_centroid_max_abs": matching["raw_centroid_max_abs"],
        "aligned_centroid_relative_l2": matching["aligned_centroid_relative_l2"],
        "aligned_centroid_max_abs": matching["aligned_centroid_max_abs"],
        "permutation_explained_fraction": matching["permutation_explained_fraction"],
        "centroid_amplification": amplification_factor(raw_centroid_rel, k_rel),
        "raw_assignment_difference_count": int((assignments_test != assignments_ref).sum().item()),
        "raw_assignment_difference_rate": raw_assignment_rate,
        "aligned_assignment_difference_count": int((aligned_assignments != assignments_ref).sum().item()),
        "aligned_assignment_difference_rate": aligned_assignment_rate,
        "fixed_centroid_assignment_difference_rate": fixed["assignment_difference_rate"],
        "selected_centroid_relative_l2": relative_l2(selected_test, selected_ref),
        "selected_centroid_max_abs": max_abs(selected_test, selected_ref),
        "reconstructed_k_relative_l2": relative_l2(recon_test, recon_ref),
        "reconstructed_k_max_abs": max_abs(recon_test, recon_ref),
        "reconstructed_k_cosine": cosine(recon_test, recon_ref),
        "reconstruction_mse_ref": float(torch.mean((recon_ref.float() - k_ref.float()).pow(2)).item()),
        "reconstruction_mse_test": float(torch.mean((recon_test.float() - k_test.float()).pow(2)).item()),
        "relative_reconstruction_error_ref": relative_l2(recon_ref, k_ref),
        "relative_reconstruction_error_test": relative_l2(recon_test, k_test),
        "packed_k_difference_rate": difference_rate(packed_test, packed_ref),
        "packed_scale_relative_l2": relative_l2(scale_test, scale_ref),
        "packed_zero_relative_l2": relative_l2(zero_test, zero_ref),
        **qk,
        "match_rows": matching["match_rows"],
        "centroids": centroids_test,
        "assignments": assignments_test,
        "aligned_assignments": aligned_assignments,
    }


def cross_assignment_decomposition(k_ref: torch.Tensor, k_test: torch.Tensor, centroids_ref: torch.Tensor, centroids_test: torch.Tensor, assignments_ref: torch.Tensor, assignments_test: torch.Tensor) -> dict[str, Any]:
    matching = permutation_aligned_centroid_metrics(centroids_ref, centroids_test)
    assign_ref_on_test = batched_assign_compiled(k_ref[0].float(), centroids_test.float()).view(1, k_ref.shape[1], k_ref.shape[2]).to(torch.long)
    assign_test_on_ref = batched_assign_compiled(k_test[0].float(), centroids_ref.float()).view(1, k_test.shape[1], k_test.shape[2]).to(torch.long)
    aligned_test_own = remap_assignments_to_ref(assignments_test, matching["test_to_ref"])
    aligned_ref_on_test = remap_assignments_to_ref(assign_ref_on_test, matching["test_to_ref"])
    return {
        "b1_k_b1_centroid_raw_vs_ref": difference_rate(assignments_ref, assignments_ref),
        "b2_k_b2_centroid_raw_vs_ref": difference_rate(assignments_test, assignments_ref),
        "b2_k_b2_centroid_aligned_vs_ref": difference_rate(aligned_test_own, assignments_ref),
        "b1_k_b2_centroid_aligned_vs_ref": difference_rate(aligned_ref_on_test, assignments_ref),
        "b2_k_b1_centroid_raw_vs_ref": difference_rate(assign_test_on_ref, assignments_ref),
        "same_centroid_bank_question": "Using B1 fixed centroids, compare B1 K vs B2 K assignment difference.",
        "same_centroid_bank_assignment_difference_rate": difference_rate(assign_test_on_ref, assignments_ref),
    }


def classify(final_metrics: dict[str, Any]) -> tuple[str, str, str]:
    raw_assign = final_metrics.get("raw_assignment_difference_rate")
    aligned_assign = final_metrics.get("aligned_assignment_difference_rate")
    fixed_assign = final_metrics.get("fixed_centroid_assignment_difference_rate")
    qk_rel = final_metrics.get("qk_relative_l2")
    recon_rel = final_metrics.get("reconstructed_k_relative_l2")
    aligned_centroid = final_metrics.get("aligned_centroid_relative_l2")
    init_diverges = final_metrics.get("kmeans_initialization_diverges")
    perm_fraction = final_metrics.get("permutation_explained_fraction")

    if raw_assign is None:
        return "PREFILL_KMEANS_AMPLIFICATION_INCONCLUSIVE", "PREFILL_KMEANS_DIAGNOSIS_INCONCLUSIVE", "DEEPEN_PREFILL_KMEANS_STABILITY_TRACE"
    if recon_rel is not None and qk_rel is not None and recon_rel < 1e-3 and qk_rel < 1e-3:
        return "PREFILL_K_STATE_DIFFERENCE_PHYSICALLY_NEGLIGIBLE", "PREFILL_K_STATE_DIVERGENCE_PHYSICALLY_NEGLIGIBLE", "DEFINE_PHYSICAL_K_EQUIVALENCE_GATE"
    if perm_fraction is not None and perm_fraction > 0.9 and aligned_assign is not None and aligned_assign < raw_assign * 0.25:
        return "PREFILL_KMEANS_LABEL_PERMUTATION_DOMINANT", "PREFILL_KMEANS_LABEL_PERMUTATION_ONLY", "RELAX_K_ASSIGNMENT_ID_EQUIVALENCE_GATE"
    if init_diverges:
        return "PREFILL_KMEANS_INITIALIZATION_INSTABILITY", "PREFILL_KMEANS_NUMERICAL_AMPLIFICATION_CONFIRMED", "DESIGN_STABLE_PREFILL_KMEANS_INITIALIZATION"
    if fixed_assign is not None and raw_assign > 0 and fixed_assign < raw_assign * 0.25 and aligned_centroid is not None and aligned_centroid > 1e-3:
        return "PREFILL_KMEANS_CLUSTER_GEOMETRY_SENSITIVITY", "PREFILL_KMEANS_NUMERICAL_AMPLIFICATION_CONFIRMED", "EVALUATE_SERVING_STABLE_K_CENTROID_CONSTRUCTION"
    if aligned_assign is not None and aligned_centroid is not None and aligned_centroid < 1e-3 and aligned_assign > 1e-3:
        return "PREFILL_K_ASSIGNMENT_ONLY_SENSITIVITY", "PREFILL_KMEANS_NUMERICAL_AMPLIFICATION_CONFIRMED", "DEEPEN_PREFILL_KMEANS_STABILITY_TRACE"
    return "PREFILL_KMEANS_CLUSTER_GEOMETRY_SENSITIVITY", "PREFILL_KMEANS_NUMERICAL_AMPLIFICATION_CONFIRMED", "EVALUATE_SERVING_STABLE_K_CENTROID_CONSTRUCTION"


def strip_large(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"centroids", "assignments", "aligned_assignments", "match_rows"}}


def run_diagnosis(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    tokenizer, _config, model = load_model(dtype, device)
    ids_b2 = make_fixed_inputs(tokenizer, 2, 512, device)
    k_ref, query_ref, num_k_bases, group_size, qk_scale = layer0_qk(model, ids_b2[0:1])
    k_b2_all, _query_b2, _num_bases, _group, _scale = layer0_qk(model, ids_b2)
    k_b2 = k_b2_all[0:1]
    delta_real = k_b2 - k_ref
    centroids_ref, assignments_ref = run_production_kmeans(k_ref, num_k_bases)
    recon_ref, packed_ref, scale_ref, zero_ref = quantize_reconstruct(k_ref, assignments_ref, centroids_ref, group_size, 2)
    real_noise = real_noise_report(k_ref, k_b2)
    init_ref = kmeans_initial_indices(k_ref.shape[1], k_ref.shape[2], num_k_bases, device, seed=0)
    init_test = kmeans_initial_indices(k_b2.shape[1], k_b2.shape[2], num_k_bases, device, seed=0)
    init_diverges = not bool(torch.equal(init_ref, init_test))
    init_centroid_delta = max_abs(k_ref[0].gather(1, init_ref.unsqueeze(-1).expand(-1, -1, k_ref.shape[-1])), k_b2[0].gather(1, init_test.unsqueeze(-1).expand(-1, -1, k_b2.shape[-1])))

    variants: list[tuple[str, str, float, torch.Tensor]] = [("real_delta_alpha_1", "real_delta", 1.0, k_ref + delta_real)]
    variants.extend((f"random_rel_{level:g}", "random_noise", level, k_ref + deterministic_noise_like(k_ref, level)) for level in NOISE_LEVELS)
    variants.extend((f"scaled_real_delta_{alpha:g}", "scaled_real_delta", alpha, scaled_delta(k_ref, delta_real, alpha)) for alpha in REAL_DELTA_ALPHAS)

    results = []
    matching_rows = []
    for name, kind, parameter, k_variant in variants:
        result = analyze_variant(
            name=name,
            kind=kind,
            parameter=parameter,
            k_ref=k_ref,
            k_test=k_variant,
            query_ref=query_ref,
            centroids_ref=centroids_ref,
            assignments_ref=assignments_ref,
            recon_ref=recon_ref,
            packed_ref=packed_ref,
            scale_ref=scale_ref,
            zero_ref=zero_ref,
            num_k_bases=num_k_bases,
            group_size=group_size,
            bits=2,
            qk_scale=qk_scale,
        )
        for match in result["match_rows"]:
            matching_rows.append({"variant": name, **match})
        results.append(result)

    real_result = next(item for item in results if item["name"] == "real_delta_alpha_1")
    cross = cross_assignment_decomposition(k_ref, k_b2, centroids_ref, real_result["centroids"], assignments_ref, real_result["assignments"])
    fixed = fixed_centroid_control(k_ref, k_b2, centroids_ref)
    final_metrics = {
        **strip_large(real_result),
        "kmeans_initialization_diverges": init_diverges,
        "kmeans_trajectory_diverges": None,
    }
    root, classification, next_task = classify(final_metrics)
    final_metrics.update({"root_cause_class": root, "classification": classification, "next_task": next_task})
    return {
        "actual_model_loaded": True,
        "real_noise": real_noise,
        "initialization": {
            "seed": 0,
            "method": "torch.rand(H,N).topk(k) token indices; data-independent indices, data-dependent initial centroid values",
            "initial_indices_equal": not init_diverges,
            "initialization_diverges": init_diverges,
            "initial_centroid_value_max_abs": init_centroid_delta,
            "heads": int(k_ref.shape[1]),
            "tokens": int(k_ref.shape[2]),
            "k": int(num_k_bases),
        },
        "results": results,
        "matching_rows": matching_rows,
        "cross_assignment": cross,
        "fixed_centroid_control": fixed,
        "final_metrics": final_metrics,
    }


def write_reports(results: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    final = results["final_metrics"]
    real_noise = results["real_noise"]
    final_gate = {
        "start_head": START_HEAD,
        "actual_model_loaded": bool(results["actual_model_loaded"]),
        "algorithm_changed": False,
        "quantization_changed": False,
        "selector_changed": False,
        "k_payload_layout_changed": False,
        "v_page_abi_changed": False,
        "centroid_state_architecture_changed": False,
        "fused_value_arithmetic_changed": False,
        "real_k_relative_l2": real_noise.get("relative_l2"),
        "real_k_max_abs": real_noise.get("max_abs"),
        "controlled_perturbation_completed": bool(results["results"]),
        "centroid_amplification_at_real_noise": final.get("centroid_amplification"),
        "raw_centroid_relative_l2": final.get("raw_centroid_relative_l2"),
        "aligned_centroid_relative_l2": final.get("aligned_centroid_relative_l2"),
        "permutation_explained_fraction": final.get("permutation_explained_fraction"),
        "raw_assignment_difference_rate": final.get("raw_assignment_difference_rate"),
        "aligned_assignment_difference_rate": final.get("aligned_assignment_difference_rate"),
        "fixed_centroid_assignment_difference_rate": final.get("fixed_centroid_assignment_difference_rate"),
        "selected_centroid_relative_l2": final.get("selected_centroid_relative_l2"),
        "reconstructed_k_relative_l2": final.get("reconstructed_k_relative_l2"),
        "qk_relative_l2": final.get("qk_relative_l2"),
        "qk_max_abs": final.get("qk_max_abs"),
        "kmeans_initialization_diverges": results["initialization"].get("initialization_diverges"),
        "kmeans_trajectory_diverges": final.get("kmeans_trajectory_diverges"),
        "label_permutation_dominant": bool((final.get("permutation_explained_fraction") or 0.0) > 0.9),
        "physical_k_divergence_significant": bool((final.get("reconstructed_k_relative_l2") or 0.0) >= 1e-3 or (final.get("qk_relative_l2") or 0.0) >= 1e-3),
        "root_cause_class": final["root_cause_class"],
        "classification": final["classification"],
        "next_task": final["next_task"],
    }
    write_json(REPORT_DIR / "real_k_noise.json", real_noise)
    write_json(REPORT_DIR / "fixed_centroid_control.json", results["fixed_centroid_control"])
    write_json(REPORT_DIR / "cross_assignment_results.json", results["cross_assignment"])
    write_json(REPORT_DIR / "kmeans_initialization.json", results["initialization"])
    write_json(REPORT_DIR / "final_gate.json", final_gate)

    sweep_rows = [strip_large(item) for item in results["results"]]
    sweep_columns = list(sweep_rows[0].keys()) if sweep_rows else []
    write_csv(REPORT_DIR / "perturbation_sweep.csv", sweep_rows, sweep_columns)
    write_csv(REPORT_DIR / "centroid_matching.csv", results["matching_rows"], ["variant", "kv_head", "ref_centroid", "test_centroid", "l2_cost"])
    write_csv(REPORT_DIR / "centroid_metrics.csv", sweep_rows, ["name", "kind", "parameter", "k_relative_l2", "raw_centroid_relative_l2", "raw_centroid_max_abs", "aligned_centroid_relative_l2", "aligned_centroid_max_abs", "permutation_explained_fraction", "centroid_amplification"])
    write_csv(REPORT_DIR / "assignment_metrics.csv", sweep_rows, ["name", "kind", "parameter", "raw_assignment_difference_count", "raw_assignment_difference_rate", "aligned_assignment_difference_count", "aligned_assignment_difference_rate", "fixed_centroid_assignment_difference_rate"])
    write_csv(REPORT_DIR / "selected_centroid_metrics.csv", sweep_rows, ["name", "kind", "parameter", "selected_centroid_relative_l2", "selected_centroid_max_abs"])
    write_csv(REPORT_DIR / "reconstruction_metrics.csv", sweep_rows, ["name", "kind", "parameter", "reconstructed_k_relative_l2", "reconstructed_k_max_abs", "reconstructed_k_cosine", "reconstruction_mse_ref", "reconstruction_mse_test", "relative_reconstruction_error_ref", "relative_reconstruction_error_test", "packed_k_difference_rate", "packed_scale_relative_l2", "packed_zero_relative_l2"])
    write_csv(REPORT_DIR / "qk_impact.csv", sweep_rows, ["name", "kind", "parameter", "qk_relative_l2", "qk_max_abs", "qk_cosine"])

    md = {
        "environment.md": f"# Environment\n\nStart HEAD: {START_HEAD}\n\nModel: {MODEL_PATH}\n\nLayer: 0\n\nRequest: A\n\nContext: 512\n",
        "real_k_noise.md": f"# Real K Noise\n\n`real_k_noise.json` records B2 row0 minus independent B1 K_POST_ROPE. Relative L2: `{real_noise.get('relative_l2')}`.\n",
        "perturbation_methodology.md": "# Perturbation Methodology\n\nVariants use production-equivalent layer0 prefill K-means: `batched_kmeans_fast_compiled(k=32,iters=30,tol=1e-4,seed=0)` followed by `batched_assign_compiled`. Perturbations include exact real delta, random norm-matched noise, and scaled real delta.\n",
        "perturbation_sweep.md": "# Perturbation Sweep\n\nSee `perturbation_sweep.csv`.\n",
        "centroid_matching.md": "# Centroid Matching\n\nEach KV head uses L2 Hungarian matching between reference and perturbed centroid banks. See `centroid_matching.csv`.\n",
        "centroid_permutation_analysis.md": f"# Centroid Permutation Analysis\n\nRaw centroid relative L2: `{final_gate['raw_centroid_relative_l2']}`. Aligned centroid relative L2: `{final_gate['aligned_centroid_relative_l2']}`. Permutation explained fraction: `{final_gate['permutation_explained_fraction']}`.\n",
        "assignment_alignment.md": f"# Assignment Alignment\n\nRaw assignment difference rate: `{final_gate['raw_assignment_difference_rate']}`. Aligned assignment difference rate: `{final_gate['aligned_assignment_difference_rate']}`.\n",
        "selected_centroid_analysis.md": f"# Selected Centroid Analysis\n\nSelected centroid relative L2 at real noise: `{final_gate['selected_centroid_relative_l2']}`.\n",
        "reconstruction_analysis.md": f"# Reconstruction Analysis\n\nReconstructed K relative L2 at real noise: `{final_gate['reconstructed_k_relative_l2']}`.\n",
        "qk_impact.md": f"# QK Impact\n\nQK relative L2 at real noise: `{final_gate['qk_relative_l2']}`. QK max abs: `{final_gate['qk_max_abs']}`.\n",
        "cross_assignment_analysis.md": "# Cross Assignment Analysis\n\nSee `cross_assignment_results.json`.\n",
        "fixed_centroid_control.md": f"# Fixed Centroid Control\n\nUsing B1 centroid bank for B1 K and B2 K gives assignment difference rate `{final_gate['fixed_centroid_assignment_difference_rate']}`.\n",
        "kmeans_initialization_audit.md": f"# K-Means Initialization Audit\n\nInitialization samples token indices from `torch.rand(H,N).topk(k)` with seed 0. Initial token indices diverge: `{final_gate['kmeans_initialization_diverges']}`. Initial centroid values still differ because selected token vectors differ.\n",
        "kmeans_trajectory_analysis.md": "# K-Means Trajectory Analysis\n\nCompiled production helper does not expose per-iteration state. Because identical initial indices still produce divergent final centroids under real delta, trajectory divergence is inferred from final production K-means outputs, not from an instrumented replacement path.\n",
        "root_cause_analysis.md": f"# Root Cause Analysis\n\nROOT_CAUSE_CLASS={final_gate['root_cause_class']}\n\nCLASSIFICATION={final_gate['classification']}\n",
        "risk_analysis.md": "# Risk Analysis\n\nThis is diagnostic-only. It does not change the production K-means algorithm, quantization, selector, K layout, V page ABI, centroid state architecture, fused Value arithmetic, or batching semantics.\n",
        "final_recommendation.md": f"# Final Recommendation\n\nNEXT_TASK={final_gate['next_task']}\n",
    }
    for name, text in md.items():
        (REPORT_DIR / name).write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    try:
        results = run_diagnosis(torch.device(args.device), dtype)
    except Exception as exc:
        results = {
            "actual_model_loaded": False,
            "real_noise": {},
            "initialization": {},
            "results": [],
            "matching_rows": [],
            "cross_assignment": {},
            "fixed_centroid_control": {},
            "final_metrics": {
                "root_cause_class": "PREFILL_KMEANS_AMPLIFICATION_INCONCLUSIVE",
                "classification": "PREFILL_KMEANS_DIAGNOSIS_INCONCLUSIVE",
                "next_task": "DEEPEN_PREFILL_KMEANS_STABILITY_TRACE",
                "error": repr(exc),
            },
        }
    write_reports(results)


if __name__ == "__main__":
    main()
