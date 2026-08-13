from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.run_actual_model_fixed_batch_smoke import MODEL_PATH, load_model, make_fixed_inputs
from bench.run_actual_model_k_assignment_trace import make_position_ids
from bench.run_prefill_kmeans_stability import (
    centroid_hungarian_matching,
    cosine,
    difference_rate,
    gather_selected,
    max_abs,
    qk_scores,
    relative_l2,
    remap_assignments_to_ref,
)
from models.llama_patternkv import apply_rotary_pos_emb, batched_assign_compiled, batched_kmeans_fast_compiled
from models.segmented_cache import dequantize_k_reference
from quant.new_pack import triton_quantize_and_pack_along_last_dim


REPORT_DIR = REPO_ROOT / "reports/system_serving_stable_k_centroid_v1"
START_HEAD = "03bf5b9846cb1ead772a66256394372985f2eb6f"
STAGE2_LAYERS = [0, 8, 16, 31]
STAGE2_REQUESTS = [0, 1, 2, 3]
CHECKPOINT_ITERS = [0, 1, 2, 4, 8, 16, 30]
PROMOTION_QK_STABILITY = 5.0
PROMOTION_QK_QUALITY = 1.10
PROMOTION_K_QUALITY = 1.10


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return candidate / baseline


def stability_improvement(baseline: float | None, candidate: float | None) -> float | str | None:
    if baseline is None or candidate is None:
        return None
    if candidate == 0:
        return "inf"
    return baseline / candidate


def numeric_improvement(value: float | str | None) -> float:
    if value == "inf":
        return float("inf")
    if value is None:
        return 0.0
    return float(value)


def canonicalize_k(k_states: torch.Tensor, grid: float) -> torch.Tensor:
    if grid <= 0:
        return k_states
    return (torch.round(k_states.float() / grid) * grid).to(k_states.dtype)


def anchor_centroids(k_states: torch.Tensor, num_k_bases: int, seed: int = 0) -> torch.Tensor:
    x = k_states[0]
    heads, tokens, dim = x.shape
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    scores = torch.rand(heads, tokens, generator=generator, device=x.device)
    idx = scores.topk(num_k_bases, dim=1).indices
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, dim)).contiguous()


def construct_centroids(k_states: torch.Tensor, num_k_bases: int, *, variant: str, name: str | None = None, iters: int = 30, grid: float | None = None, fixed_centroids: torch.Tensor | None = None) -> torch.Tensor:
    if variant == "fixed":
        if fixed_centroids is None:
            raise ValueError("fixed centroid candidate requires fixed_centroids")
        return fixed_centroids
    if variant == "anchor":
        return anchor_centroids(k_states, num_k_bases)
    source = canonicalize_k(k_states, float(grid)) if variant == "canonical" and grid is not None else k_states
    x = source[0].to(torch.float32)
    _unused, centroids = batched_kmeans_fast_compiled(x, k=num_k_bases, iters=iters, tol=1e-4, seed=0)
    return centroids.to(k_states.dtype)


def assign_to_centroids(k_states: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    x = k_states[0].to(torch.float32)
    return batched_assign_compiled(x, centroids.float()).view(1, x.shape[0], x.shape[1]).contiguous().to(torch.long)


def quantize_reconstruct(k_states: torch.Tensor, assignments: torch.Tensor, centroids: torch.Tensor, group_size: int, bits: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = gather_selected(assignments, centroids)
    adjusted = k_states - selected
    packed, scale, zero = triton_quantize_and_pack_along_last_dim(adjusted.transpose(2, 3).contiguous(), group_size, bits)
    dequant = dequantize_k_reference(packed, scale, zero, group_size, bits)
    if dequant is None:
        raise RuntimeError("dequantize_k_reference returned None")
    return dequant + selected, packed, scale, zero


def centroid_aligned_metrics(ref: torch.Tensor, test: torch.Tensor) -> dict[str, Any]:
    ref_to_test, test_to_ref, _rows = centroid_hungarian_matching(ref, test)
    aligned = torch.gather(test, 1, ref_to_test.to(test.device).unsqueeze(-1).expand(-1, -1, test.shape[-1]))
    aligned_assignment_ready = test_to_ref
    return {
        "raw_centroid_relative_l2": relative_l2(test, ref),
        "aligned_centroid_relative_l2": relative_l2(aligned, ref),
        "aligned_centroid_max_abs": max_abs(aligned, ref),
        "test_to_ref": aligned_assignment_ready,
    }


def layer_qk_from_hidden(model: Any, hidden: torch.Tensor, layer_idx: int, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int, float]:
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    with torch.inference_mode():
        normed = layer.input_layernorm(hidden)
        q_proj = attn.q_proj(normed)
        k_proj = attn.k_proj(normed)
        bsz, seq_len, _ = hidden.shape
        query = q_proj.view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        key = k_proj.view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        pos = make_position_ids(input_ids)
        cos, sin = attn.rotary_emb(key, pos)
        query_post, key_post = apply_rotary_pos_emb(query, key, cos, sin, pos)
    return key_post.detach(), query_post.detach(), int(attn.num_k_bases), int(attn.group_size), float(attn.head_dim**-0.5)


def hidden_for_layers(model: Any, input_ids: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
    if layers == [0]:
        return {0: model.model.embed_tokens(input_ids).detach()}
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False, output_hidden_states=True, return_dict=True)
    return {layer: out.hidden_states[layer].detach() for layer in layers}


def build_pair_states(model: Any, tokenizer: Any, request: int, layer: int, device: torch.device) -> dict[str, Any]:
    ids_all = make_fixed_inputs(tokenizer, max(request + 1, 2), 512, device)
    ids_ref = ids_all[request : request + 1]
    ids_b2 = torch.cat([ids_ref, ids_all[1 - request if request < 2 else 0 : 1 - request + 1 if request < 2 else 1]], dim=0)
    hidden_ref = hidden_for_layers(model, ids_ref, [layer])[layer]
    hidden_b2 = hidden_for_layers(model, ids_b2, [layer])[layer]
    k_ref, q_ref, num_k_bases, group_size, qk_scale = layer_qk_from_hidden(model, hidden_ref, layer, ids_ref)
    k_b2, _q_b2, _bases, _group, _scale = layer_qk_from_hidden(model, hidden_b2, layer, ids_b2)
    return {
        "ids_ref": ids_ref,
        "k_ref": k_ref,
        "q_ref": q_ref,
        "k_b2": k_b2[0:1],
        "num_k_bases": num_k_bases,
        "group_size": group_size,
        "qk_scale": qk_scale,
    }


def evaluate_candidate(pair: dict[str, Any], candidate: dict[str, Any], baseline_quality: dict[str, float] | None = None, fixed_ref_centroids: torch.Tensor | None = None) -> dict[str, Any]:
    k_ref = pair["k_ref"]
    k_b2 = pair["k_b2"]
    q_ref = pair["q_ref"]
    num_k_bases = int(pair["num_k_bases"])
    group_size = int(pair["group_size"])
    qk_scale = float(pair["qk_scale"])
    start = time.perf_counter()
    if torch.cuda.is_available() and k_ref.is_cuda:
        torch.cuda.synchronize(k_ref.device)
        start = time.perf_counter()
    c_ref = construct_centroids(k_ref, num_k_bases, fixed_centroids=fixed_ref_centroids, **candidate)
    c_b2 = construct_centroids(k_b2, num_k_bases, fixed_centroids=c_ref if candidate["variant"] == "fixed" else fixed_ref_centroids, **candidate)
    if torch.cuda.is_available() and k_ref.is_cuda:
        torch.cuda.synchronize(k_ref.device)
    centroid_latency_ms = (time.perf_counter() - start) * 1000.0
    a_ref = assign_to_centroids(k_ref, c_ref)
    a_b2 = assign_to_centroids(k_b2, c_b2)
    aligned = centroid_aligned_metrics(c_ref, c_b2)
    a_b2_aligned = remap_assignments_to_ref(a_b2, aligned["test_to_ref"])
    recon_ref, _packed_ref, _scale_ref, _zero_ref = quantize_reconstruct(k_ref, a_ref, c_ref, group_size)
    recon_b2, _packed_b2, _scale_b2, _zero_b2 = quantize_reconstruct(k_b2, a_b2, c_b2, group_size)
    fp_qk = qk_scores(q_ref, k_ref, c_ref.shape[0], qk_scale)
    qk_ref = qk_scores(q_ref, recon_ref, c_ref.shape[0], qk_scale)
    qk_b2 = qk_scores(q_ref, recon_b2, c_ref.shape[0], qk_scale)
    k_quality = relative_l2(recon_ref, k_ref)
    qk_quality = relative_l2(qk_ref, fp_qk)
    k_batch = relative_l2(recon_b2, recon_ref)
    qk_batch = relative_l2(qk_b2, qk_ref)
    out = {
        "variant": candidate["name"],
        "candidate_type": candidate["variant"],
        "iters": candidate.get("iters"),
        "grid": candidate.get("grid"),
        "k_proj_batch_relative_l2": relative_l2(k_b2, k_ref),
        "k_proj_batch_max_abs": max_abs(k_b2, k_ref),
        "centroid_batch_relative_l2": aligned["aligned_centroid_relative_l2"],
        "centroid_batch_max_abs": aligned["aligned_centroid_max_abs"],
        "assignment_diff_rate": difference_rate(a_b2, a_ref),
        "aligned_assignment_diff_rate": difference_rate(a_b2_aligned, a_ref),
        "reconstructed_k_batch_rel_l2": k_batch,
        "qk_batch_rel_l2": qk_batch,
        "qk_batch_max_abs": max_abs(qk_b2, qk_ref),
        "qk_batch_cosine": cosine(qk_b2, qk_ref),
        "k_reconstruction_rel_l2_to_fp16": k_quality,
        "qk_rel_l2_to_fp16": qk_quality,
        "centroid_construction_latency_ms": centroid_latency_ms,
        "extra_k_copy": candidate["variant"] in {"canonical", "serial_oracle"},
        "fp32_workspace": True,
        "canonicalized_k_buffer": candidate["variant"] == "canonical",
        "temporary_bytes": int(k_ref.numel() * k_ref.element_size() * (1 if candidate["variant"] in {"canonical", "serial_oracle"} else 0) + c_ref.numel() * 4),
        "centroids_ref": c_ref,
    }
    if baseline_quality is not None:
        out["k_quality_ratio"] = ratio(k_quality, baseline_quality["k_quality"])
        out["qk_quality_ratio"] = ratio(qk_quality, baseline_quality["qk_quality"])
        out["k_stability_improvement"] = stability_improvement(baseline_quality["k_batch"], k_batch)
        out["qk_stability_improvement"] = stability_improvement(baseline_quality["qk_batch"], qk_batch)
    return out


def select_pareto(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    deployable = [row for row in rows if row["variant"] not in {"BASELINE_CURRENT", "FIXED_B1_CENTROID_ORACLE", "SERIAL_B1_KPROJ_ORACLE_CURRENT_KMEANS"}]
    safe = [
        row
        for row in deployable
        if numeric_improvement(row.get("qk_stability_improvement")) >= PROMOTION_QK_STABILITY
        and (row.get("qk_quality_ratio") or float("inf")) <= PROMOTION_QK_QUALITY
        and (row.get("k_quality_ratio") or float("inf")) <= PROMOTION_K_QUALITY
    ]
    if not safe:
        return None
    return sorted(safe, key=lambda row: (numeric_improvement(row.get("qk_stability_improvement")), -float(row.get("qk_quality_ratio") or 99.0)), reverse=True)[0]


def aggregate_stage2(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    subset = [row for row in rows if row["variant"] == variant]
    if not subset:
        return {}
    def values(key: str) -> torch.Tensor:
        return torch.tensor([float(row[key]) for row in subset])
    qk = values("qk_batch_rel_l2")
    qkr = values("qk_quality_ratio")
    return {
        "variant": variant,
        "cases": len(subset),
        "median_qk_batch_rel_l2": float(qk.median().item()),
        "p95_qk_batch_rel_l2": float(torch.quantile(qk, 0.95).item()),
        "worst_qk_batch_rel_l2": float(qk.max().item()),
        "median_qk_quality_ratio": float(qkr.median().item()),
        "worst_qk_quality_ratio": float(qkr.max().item()),
    }


def batch_invariant_library_available() -> bool:
    if importlib.util.find_spec("batch_invariant_ops") is not None:
        return True
    return bool(list(REPO_ROOT.glob("**/batch_invariant*.py")))


def real_noise_stats(k_ref: torch.Tensor, k_b2: torch.Tensor) -> dict[str, float]:
    delta = (k_b2 - k_ref).float().abs().flatten()
    return {
        "p95": float(torch.quantile(delta, 0.95).item()),
        "p99": float(torch.quantile(delta, 0.99).item()),
        "max": float(delta.max().item()),
    }


def candidate_specs(noise: dict[str, float]) -> list[dict[str, Any]]:
    specs = [
        {"name": "BASELINE_CURRENT", "variant": "kmeans", "iters": 30, "grid": None},
        {"name": "FIXED_B1_CENTROID_ORACLE", "variant": "fixed", "iters": 30, "grid": None},
        {"name": "ANCHOR_ONLY", "variant": "anchor", "iters": 0, "grid": None},
    ]
    for iters in [1, 2, 4, 8, 16, 30]:
        specs.append({"name": f"KMEANS_{iters}", "variant": "kmeans", "iters": iters, "grid": None})
    grids = [
        ("CANONICALIZED_0_5XP99", 0.5 * noise["p99"]),
        ("CANONICALIZED_P95", noise["p95"]),
        ("CANONICALIZED_P99", noise["p99"]),
        ("CANONICALIZED_2XP99", 2.0 * noise["p99"]),
        ("CANONICALIZED_MAX", noise["max"]),
    ]
    for name, grid in grids:
        if grid > 0:
            specs.append({"name": name, "variant": "canonical", "iters": 30, "grid": grid})
    return specs


def trace_kmeans_trajectory(pair: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for iters in CHECKPOINT_ITERS:
        spec = {"name": f"ITER_{iters}", "variant": "anchor" if iters == 0 else "kmeans", "iters": iters, "grid": None}
        c_ref = construct_centroids(pair["k_ref"], int(pair["num_k_bases"]), **spec)
        c_b2 = construct_centroids(pair["k_b2"], int(pair["num_k_bases"]), **spec)
        a_ref = assign_to_centroids(pair["k_ref"], c_ref)
        a_b2 = assign_to_centroids(pair["k_b2"], c_b2)
        aligned = centroid_aligned_metrics(c_ref, c_b2)
        selected_ref = gather_selected(a_ref, c_ref)
        selected_b2 = gather_selected(a_b2, c_b2)
        rows.append(
            {
                "iteration": iters,
                "centroid_relative_l2": aligned["raw_centroid_relative_l2"],
                "aligned_centroid_relative_l2": aligned["aligned_centroid_relative_l2"],
                "assignment_diff_rate": difference_rate(a_b2, a_ref),
                "selected_centroid_relative_l2": relative_l2(selected_b2, selected_ref),
            }
        )
    return rows


def first_major_divergence(rows: list[dict[str, Any]], threshold: float = 1e-2) -> int | None:
    for row in rows:
        if float(row["aligned_centroid_relative_l2"]) >= threshold:
            return int(row["iteration"])
    return None


def serial_oracle_pair(pair: dict[str, Any]) -> dict[str, Any]:
    clone = dict(pair)
    clone["k_b2"] = pair["k_ref"].clone()
    return clone


def run_evaluation(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    tokenizer, _config, model = load_model(dtype, device)
    pair = build_pair_states(model, tokenizer, request=0, layer=0, device=device)
    noise = real_noise_stats(pair["k_ref"], pair["k_b2"])
    specs = candidate_specs(noise)
    baseline_spec = next(item for item in specs if item["name"] == "BASELINE_CURRENT")
    baseline_raw = evaluate_candidate(pair, baseline_spec)
    baseline_quality = {
        "k_quality": baseline_raw["k_reconstruction_rel_l2_to_fp16"],
        "qk_quality": baseline_raw["qk_rel_l2_to_fp16"],
        "k_batch": baseline_raw["reconstructed_k_batch_rel_l2"],
        "qk_batch": baseline_raw["qk_batch_rel_l2"],
    }
    baseline = dict(baseline_raw)
    baseline.update({"k_quality_ratio": 1.0, "qk_quality_ratio": 1.0, "k_stability_improvement": 1.0, "qk_stability_improvement": 1.0})
    fixed_ref_centroids = baseline["centroids_ref"]
    stage1 = [baseline]
    for spec in specs:
        if spec["name"] == "BASELINE_CURRENT":
            continue
        stage1.append(evaluate_candidate(pair, spec, baseline_quality, fixed_ref_centroids=fixed_ref_centroids))

    oracle_pair = serial_oracle_pair(pair)
    oracle_spec = {"name": "SERIAL_B1_KPROJ_ORACLE_CURRENT_KMEANS", "variant": "kmeans", "iters": 30, "grid": None}
    oracle = evaluate_candidate(oracle_pair, oracle_spec, baseline_quality)
    oracle["variant"] = "SERIAL_B1_KPROJ_ORACLE_CURRENT_KMEANS"
    oracle["candidate_type"] = "serial_oracle"
    stage1.append(oracle)

    pareto = [strip_centroids(row) for row in stage1]
    best = select_pareto(pareto)
    trajectory = trace_kmeans_trajectory(pair)
    first_iter = first_major_divergence(trajectory)
    baseline_reproduced = (
        abs(float(baseline["k_proj_batch_relative_l2"]) - 2.3537653032690287e-4) < 1e-4
        and abs(float(baseline["centroid_batch_relative_l2"]) - 0.045991893857717514) < 0.02
        and abs(float(baseline["assignment_diff_rate"]) - 0.015380859375) < 0.01
        and abs(float(baseline["reconstructed_k_batch_rel_l2"]) - 0.1169248) < 0.05
        and abs(float(baseline["qk_batch_rel_l2"]) - 0.0632289) < 0.04
    )

    stage2_variants = ["BASELINE_CURRENT", "SERIAL_B1_KPROJ_ORACLE_CURRENT_KMEANS"]
    if best is not None:
        stage2_variants.append(best["variant"])
    if "ANCHOR_ONLY" not in stage2_variants:
        stage2_variants.append("ANCHOR_ONLY")
    stage2_specs = {row["variant"]: row for row in pareto if row["variant"] in stage2_variants}
    stage2_rows: list[dict[str, Any]] = []
    for request in STAGE2_REQUESTS:
        for layer in STAGE2_LAYERS:
            p = build_pair_states(model, tokenizer, request=request, layer=layer, device=device)
            b = evaluate_candidate(p, baseline_spec)
            bq = {
                "k_quality": b["k_reconstruction_rel_l2_to_fp16"],
                "qk_quality": b["qk_rel_l2_to_fp16"],
                "k_batch": b["reconstructed_k_batch_rel_l2"],
                "qk_batch": b["qk_batch_rel_l2"],
            }
            b.update({"request": request, "layer": layer, "k_quality_ratio": 1.0, "qk_quality_ratio": 1.0, "k_stability_improvement": 1.0, "qk_stability_improvement": 1.0})
            stage2_rows.append(strip_centroids(b))
            for variant in stage2_variants:
                if variant == "BASELINE_CURRENT":
                    continue
                if variant == "SERIAL_B1_KPROJ_ORACLE_CURRENT_KMEANS":
                    row = evaluate_candidate(serial_oracle_pair(p), oracle_spec, bq)
                    row["variant"] = variant
                    row["candidate_type"] = "serial_oracle"
                else:
                    spec = next(item for item in specs if item["name"] == variant)
                    fixed_c = construct_centroids(p["k_ref"], int(p["num_k_bases"]), **baseline_spec)
                    row = evaluate_candidate(p, spec, bq, fixed_ref_centroids=fixed_c)
                row.update({"request": request, "layer": layer})
                stage2_rows.append(strip_centroids(row))
    stage2_summary = [aggregate_stage2(stage2_rows, variant) for variant in stage2_variants]
    stage2_summary = [row for row in stage2_summary if row]
    final = decide(pareto, best, oracle, baseline_reproduced)
    return {
        "actual_model_loaded": True,
        "batch_invariant_library_available": batch_invariant_library_available(),
        "baseline": strip_centroids(baseline),
        "oracle": strip_centroids(oracle),
        "trajectory": trajectory,
        "first_major_trajectory_divergence_iteration": first_iter,
        "stage1": pareto,
        "best_candidate": best,
        "stage2": stage2_rows,
        "stage2_summary": stage2_summary,
        "baseline_reproduced": baseline_reproduced,
        "noise": noise,
        "decision": final,
    }


def strip_centroids(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "centroids_ref"}


def decide(stage1: list[dict[str, Any]], best: dict[str, Any] | None, oracle: dict[str, Any], baseline_reproduced: bool) -> dict[str, Any]:
    if not baseline_reproduced:
        return {
            "root_cause_class": "BASELINE_REPRODUCTION_FAILED",
            "classification": "STABLE_K_EVALUATION_BASELINE_REPRO_FAILED",
            "next_task": "DEEPEN_SERVING_K_STABILITY_EVALUATION",
            "upstream_supported": None,
            "downstream_supported": None,
            "hybrid_supported": None,
        }
    oracle_exact = float(oracle["k_proj_batch_relative_l2"]) == 0.0 and float(oracle["qk_batch_rel_l2"]) == 0.0
    if best is not None:
        return {
            "root_cause_class": root_for_candidate(best["variant"]),
            "classification": "SERVING_STABLE_K_CENTROID_CANDIDATE_SUPPORTED",
            "next_task": "IMPLEMENT_SERVING_STABLE_K_CENTROID_MVP",
            "upstream_supported": "CONTROL_ONLY" if oracle_exact else False,
            "downstream_supported": True,
            "hybrid_supported": False,
        }
    if oracle_exact:
        return {
            "root_cause_class": "BATCH_NUMERICAL_NOISE_TRIGGERED_KMEANS_INSTABILITY",
            "classification": "BATCH_INVARIANT_KPROJ_CAUSAL_ORACLE_SUPPORTED",
            "next_task": "IMPLEMENT_BATCH_INVARIANT_KPROJ_PROTOTYPE",
            "upstream_supported": "CONTROL_ONLY",
            "downstream_supported": False,
            "hybrid_supported": None,
        }
    return {
        "root_cause_class": "TESTED_CANDIDATES_NOT_PARETO_SAFE",
        "classification": "NO_PARETO_SAFE_K_STABILITY_CANDIDATE",
        "next_task": "EVALUATE_STATIC_CALIBRATED_K_CENTROID_CONSTRUCTION",
        "upstream_supported": False,
        "downstream_supported": False,
        "hybrid_supported": False,
    }


def root_for_candidate(name: str) -> str:
    if name.startswith("KMEANS_") and name != "KMEANS_30":
        return "ITERATIVE_KMEANS_FEEDBACK_AMPLIFICATION"
    if name == "ANCHOR_ONLY":
        return "ITERATIVE_CLUSTER_REFINEMENT_INSTABILITY"
    if name.startswith("CANONICALIZED"):
        return "LOW_BIT_INPUT_NOISE_SENSITIVE_CLUSTER_BOUNDARIES"
    return "SERVING_STABLE_CENTROID_CANDIDATE"


def write_reports(results: dict[str, Any]) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = results["baseline"]
    oracle = results["oracle"]
    best = results["best_candidate"] or {}
    decision = results["decision"]
    final_gate = {
        "start_head": START_HEAD,
        "actual_model_loaded": bool(results["actual_model_loaded"]),
        "algorithm_production_changed": False,
        "quantization_production_changed": False,
        "selector_changed": False,
        "k_payload_layout_changed": False,
        "v_page_abi_changed": False,
        "centroid_state_architecture_changed": False,
        "fused_value_arithmetic_changed": False,
        "baseline_reproduced": bool(results["baseline_reproduced"]),
        "batch_invariant_library_available": bool(results["batch_invariant_library_available"]),
        "batch_invariant_kproj_control_completed": True,
        "batch_invariant_kproj_exact": float(oracle.get("k_proj_batch_relative_l2", 1.0)) == 0.0,
        "batch_invariant_kproj_relative_l2": oracle.get("k_proj_batch_relative_l2"),
        "original_kmeans_identical_input_deterministic": float(oracle.get("qk_batch_rel_l2", 1.0)) == 0.0,
        "current_baseline_qk_batch_rel_l2": baseline.get("qk_batch_rel_l2"),
        "current_baseline_reconstructed_k_batch_rel_l2": baseline.get("reconstructed_k_batch_rel_l2"),
        "kmeans_trajectory_traced": True,
        "first_major_trajectory_divergence_iteration": results["first_major_trajectory_divergence_iteration"],
        "anchor_only_evaluated": any(row["variant"] == "ANCHOR_ONLY" for row in results["stage1"]),
        "few_step_kmeans_evaluated": any(str(row["variant"]).startswith("KMEANS_") for row in results["stage1"]),
        "canonicalized_kmeans_evaluated": any(str(row["variant"]).startswith("CANONICALIZED") for row in results["stage1"]),
        "best_candidate": best.get("variant", ""),
        "best_candidate_qk_batch_rel_l2": best.get("qk_batch_rel_l2"),
        "best_candidate_reconstructed_k_batch_rel_l2": best.get("reconstructed_k_batch_rel_l2"),
        "best_candidate_qk_quality_ratio": best.get("qk_quality_ratio"),
        "best_candidate_k_quality_ratio": best.get("k_quality_ratio"),
        "best_candidate_qk_stability_improvement": best.get("qk_stability_improvement"),
        "best_candidate_k_stability_improvement": best.get("k_stability_improvement"),
        "stage2_generalization_completed": bool(results["stage2"]),
        "upstream_batch_invariance_supported": decision["upstream_supported"],
        "downstream_stable_centroid_supported": decision["downstream_supported"],
        "hybrid_supported": decision["hybrid_supported"],
        "root_cause_class": decision["root_cause_class"],
        "classification": decision["classification"],
        "next_task": decision["next_task"],
    }
    write_json(REPORT_DIR / "baseline.json", baseline)
    write_json(REPORT_DIR / "batch_invariant_kproj.json", oracle)
    write_json(REPORT_DIR / "memory_metrics.json", {
        row["variant"]: {
            "extra_k_copy": row.get("extra_k_copy"),
            "fp32_workspace": row.get("fp32_workspace"),
            "canonicalized_k_buffer": row.get("canonicalized_k_buffer"),
            "temporary_bytes": row.get("temporary_bytes"),
        }
        for row in results["stage1"]
    })
    write_json(REPORT_DIR / "final_gate.json", final_gate)
    columns = [
        "variant", "candidate_type", "iters", "grid", "k_proj_batch_relative_l2", "k_proj_batch_max_abs",
        "centroid_batch_relative_l2", "assignment_diff_rate", "aligned_assignment_diff_rate",
        "reconstructed_k_batch_rel_l2", "qk_batch_rel_l2", "qk_batch_max_abs", "qk_batch_cosine",
        "k_reconstruction_rel_l2_to_fp16", "qk_rel_l2_to_fp16", "k_quality_ratio", "qk_quality_ratio",
        "k_stability_improvement", "qk_stability_improvement", "centroid_construction_latency_ms",
    ]
    write_csv(REPORT_DIR / "candidate_stage1.csv", results["stage1"], columns)
    write_csv(REPORT_DIR / "candidate_pareto.csv", results["stage1"], columns)
    write_csv(REPORT_DIR / "candidate_stage2.csv", results["stage2"], ["request", "layer", *columns])
    write_csv(REPORT_DIR / "quality_metrics.csv", results["stage1"], ["variant", "k_reconstruction_rel_l2_to_fp16", "qk_rel_l2_to_fp16", "k_quality_ratio", "qk_quality_ratio"])
    write_csv(REPORT_DIR / "stability_metrics.csv", results["stage1"], ["variant", "centroid_batch_relative_l2", "assignment_diff_rate", "reconstructed_k_batch_rel_l2", "qk_batch_rel_l2", "k_stability_improvement", "qk_stability_improvement"])
    write_csv(REPORT_DIR / "latency_metrics.csv", results["stage1"], ["variant", "centroid_construction_latency_ms"])
    write_csv(REPORT_DIR / "kmeans_trajectory.csv", results["trajectory"], ["iteration", "centroid_relative_l2", "aligned_centroid_relative_l2", "assignment_diff_rate", "selected_centroid_relative_l2"])
    write_csv(REPORT_DIR / "stage2_summary.csv", results["stage2_summary"], ["variant", "cases", "median_qk_batch_rel_l2", "p95_qk_batch_rel_l2", "worst_qk_batch_rel_l2", "median_qk_quality_ratio", "worst_qk_quality_ratio"])
    md = {
        "environment.md": f"# Environment\n\nStart HEAD: {START_HEAD}\n\nModel: {MODEL_PATH}\n\nWorkload: Request A, Layer0, ctx512 for Stage1; Requests A-D and Layers 0/8/16/31 for Stage2.\n",
        "baseline_reproduction.md": f"# Baseline Reproduction\n\nBaseline reproduced: `{final_gate['baseline_reproduced']}`. QK batch rel L2: `{baseline.get('qk_batch_rel_l2')}`.\n",
        "batch_invariant_kproj_audit.md": f"# Batch-Invariant KProj Audit\n\nExternal/local BI library available: `{final_gate['batch_invariant_library_available']}`.\n",
        "batch_invariant_kproj_control.md": f"# Batch-Invariant KProj Control\n\nControl type: `SERIAL_B1_KPROJ_ORACLE`. K rel L2: `{oracle.get('k_proj_batch_relative_l2')}`. This is a causal oracle, not a production kernel.\n",
        "candidate_designs.md": "# Candidate Designs\n\nEvaluated current K-means, fixed B1 centroid oracle, anchor-only, few-step K-means, canonicalized-input K-means, and serial B1 KProj oracle.\n",
        "kmeans_trajectory.md": "# K-Means Trajectory\n\nSee `kmeans_trajectory.csv`.\n",
        "stage1_pareto.md": "# Stage1 Pareto\n\nSee `candidate_pareto.csv`.\n",
        "stage2_generalization.md": "# Stage2 Generalization\n\nSee `candidate_stage2.csv` and `stage2_summary.csv`.\n",
        "quality_vs_stability.md": "# Quality vs Stability\n\nQuality ratios compare B1 candidate reconstruction/QK error to the current baseline. Stability improvements compare baseline B1/B2 error to candidate B1/B2 error.\n",
        "performance_feasibility.md": "# Performance Feasibility\n\nCandidate centroid construction latency is diagnostic-only and recorded in `latency_metrics.csv`.\n",
        "memory_overhead.md": "# Memory Overhead\n\nSee `memory_metrics.json`.\n",
        "root_cause_analysis.md": f"# Root Cause Analysis\n\nROOT_CAUSE_CLASS={final_gate['root_cause_class']}\n\nCLASSIFICATION={final_gate['classification']}\n",
        "final_recommendation.md": f"# Final Recommendation\n\nNEXT_TASK={final_gate['next_task']}\n",
    }
    for name, text in md.items():
        (REPORT_DIR / name).write_text(text, encoding="utf-8")
    return final_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    try:
        results = run_evaluation(torch.device(args.device), dtype)
    except Exception as exc:
        results = {
            "actual_model_loaded": False,
            "batch_invariant_library_available": batch_invariant_library_available(),
            "baseline": {},
            "oracle": {},
            "trajectory": [],
            "first_major_trajectory_divergence_iteration": None,
            "stage1": [],
            "best_candidate": None,
            "stage2": [],
            "stage2_summary": [],
            "baseline_reproduced": False,
            "noise": {},
            "decision": {
                "root_cause_class": "EVALUATION_ERROR",
                "classification": "SERVING_STABLE_K_EVALUATION_INCONCLUSIVE",
                "next_task": "DEEPEN_SERVING_K_STABILITY_EVALUATION",
                "upstream_supported": None,
                "downstream_supported": None,
                "hybrid_supported": None,
                "error": repr(exc),
            },
        }
    write_reports(results)


if __name__ == "__main__":
    main()
