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

from bench.prefill_v_trace_utils import (
    TRACE_ITERS,
    assignment_metrics,
    bi_vproj_control,
    centroid_alignment_metrics,
    difference_rate,
    fixed_centroid_control,
    kmeans_initial_indices,
    packed_v_metrics,
    relative_l2,
    run_v_state,
    tensor_metric_dict,
    trace_reference_kmeans,
    validate_reference_kmeans,
    v_centroid_amplification,
)
from bench.run_actual_model_fixed_batch_smoke import MODEL_PATH, load_model, make_fixed_inputs
from bench.run_serving_stable_k_centroid_eval import hidden_for_layers
from models.llama_patternkv import batched_assign_compiled


REPORT_DIR = REPO_ROOT / "reports/system_prefill_v_centroid_trace_v1"
START_HEAD = "7cada5a58f7c841c84d4e8490846458b7559d2f1"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def set_trace_env() -> None:
    os.environ["PATTERNKV_BATCH_INVARIANT_KPROJ"] = "1"
    os.environ["PATTERNKV_BI_KPROJ_BACKEND"] = "v2"
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"


def layer_hidden_pair(model: Any, tokenizer: Any, layer_idx: int, device: torch.device) -> dict[str, torch.Tensor]:
    ids = make_fixed_inputs(tokenizer, 4, 512, device)
    if layer_idx == 0:
        hidden_b1 = model.model.embed_tokens(ids[0:1]).detach()
        hidden_b2 = model.model.embed_tokens(ids[:2]).detach()
        hidden_b4 = model.model.embed_tokens(ids[:4]).detach()
    else:
        hidden_b1 = hidden_for_layers(model, ids[0:1], [layer_idx])[layer_idx]
        hidden_b2 = hidden_for_layers(model, ids[:2], [layer_idx])[layer_idx]
        hidden_b4 = hidden_for_layers(model, ids[:4], [layer_idx])[layer_idx]
    return {"hidden_b1": hidden_b1, "hidden_b2": hidden_b2, "hidden_b4": hidden_b4}


def value_from_hidden(model: Any, layer_idx: int, hidden: torch.Tensor, *, use_bi: bool = False) -> torch.Tensor:
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    normed = layer.input_layernorm(hidden)
    if use_bi:
        proj = bi_vproj_control(normed[0:1], normed, normed, attn.v_proj.weight, getattr(attn.v_proj, "bias", None))["b1"]
    else:
        proj = attn.v_proj(normed)
    bsz, seq_len, _ = proj.shape
    return proj.view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2).contiguous()


def normal_value_pair(model: Any, layer_idx: int, hidden_b1: torch.Tensor, hidden_b2: torch.Tensor, hidden_b4: torch.Tensor) -> dict[str, torch.Tensor]:
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    norm_b1 = layer.input_layernorm(hidden_b1)
    norm_b2 = layer.input_layernorm(hidden_b2)
    norm_b4 = layer.input_layernorm(hidden_b4)
    normal_b1 = attn.v_proj(norm_b1)
    normal_b2 = attn.v_proj(norm_b2)
    normal_b4 = attn.v_proj(norm_b4)
    bi = bi_vproj_control(norm_b1, norm_b2, norm_b4, attn.v_proj.weight, getattr(attn.v_proj, "bias", None))
    def shape(x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2).contiguous()
    return {
        "normal_b1": shape(normal_b1),
        "normal_b2_row0": shape(normal_b2)[0:1],
        "normal_b4_row0": shape(normal_b4)[0:1],
        "bi_b1": shape(bi["b1"]),
        "bi_b2_row0": shape(bi["b2_row0"]),
        "bi_b4_row0": shape(bi["b4_row0"]),
        "bi_b1_b2_exact": bi["b1_b2_exact"],
        "bi_b1_b4_exact": bi["b1_b4_exact"],
    }


def run_pair_trace(model: Any, tokenizer: Any, layer_idx: int, device: torch.device, *, full: bool) -> dict[str, Any]:
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    hidden = layer_hidden_pair(model, tokenizer, layer_idx, device)
    values = normal_value_pair(model, layer_idx, hidden["hidden_b1"], hidden["hidden_b2"], hidden["hidden_b4"])
    hidden_metrics = tensor_metric_dict(hidden["hidden_b2"][0:1], hidden["hidden_b1"])
    vproj_metrics = tensor_metric_dict(values["normal_b2_row0"], values["normal_b1"])
    kproj_metrics = {}
    with torch.inference_mode():
        norm_b1 = layer.input_layernorm(hidden["hidden_b1"])
        norm_b2 = layer.input_layernorm(hidden["hidden_b2"])
        k_b1 = attn.k_proj(norm_b1)
        k_b2 = attn.k_proj(norm_b2)[0:1]
        kproj_metrics = tensor_metric_dict(k_b2, k_b1)

    X1 = values["normal_b1"][0].to(torch.float32)
    X2 = values["normal_b2_row0"][0].to(torch.float32)
    trace1 = trace_reference_kmeans(X1, int(attn.num_v_bases), iters=30, tol=1e-4, seed=0)
    trace2 = trace_reference_kmeans(X2, int(attn.num_v_bases), iters=30, tol=1e-4, seed=0)
    prod_validation = validate_reference_kmeans(X1, trace1["final_centroids"])
    init_indices_1 = kmeans_initial_indices(X1.shape[0], X1.shape[1], int(attn.num_v_bases), X1.device, seed=0)
    init_indices_2 = kmeans_initial_indices(X2.shape[0], X2.shape[1], int(attn.num_v_bases), X2.device, seed=0)
    init_rel = relative_l2(trace2["snapshots"][0], trace1["snapshots"][0])
    alignment = centroid_alignment_metrics(trace1["final_centroids"], trace2["final_centroids"])
    assign1 = batched_assign_compiled(X1, trace1["final_centroids"]).view(1, X1.shape[0], X1.shape[1]).to(torch.long)
    assign2 = batched_assign_compiled(X2, trace2["final_centroids"]).view(1, X2.shape[0], X2.shape[1]).to(torch.long)
    assign = assignment_metrics(assign1, assign2, alignment["got_to_ref"])
    state1 = run_v_state(values["normal_b1"], trace1["final_centroids"].to(values["normal_b1"].dtype), value_objective=attn.value_objective, group_size=int(attn.group_size), bits=int(attn.v_bits), v4_budget_fraction=float(attn.v4_budget_fraction))
    state2 = run_v_state(values["normal_b2_row0"], trace2["final_centroids"].to(values["normal_b1"].dtype), value_objective=attn.value_objective, group_size=int(attn.group_size), bits=int(attn.v_bits), v4_budget_fraction=float(attn.v4_budget_fraction))
    fixed = fixed_centroid_control(values["normal_b1"], values["normal_b2_row0"], trace1["final_centroids"].to(values["normal_b1"].dtype), value_objective=attn.value_objective, group_size=int(attn.group_size), bits=int(attn.v_bits), v4_budget_fraction=float(attn.v4_budget_fraction))
    packed_normal = packed_v_metrics(state2["cache"], state1["cache"])

    bi_X1 = values["bi_b1"][0].to(torch.float32)
    bi_X2 = values["bi_b2_row0"][0].to(torch.float32)
    bi_X4 = values["bi_b4_row0"][0].to(torch.float32)
    bi_trace1 = trace_reference_kmeans(bi_X1, int(attn.num_v_bases), iters=30, tol=1e-4, seed=0)
    bi_trace2 = trace_reference_kmeans(bi_X2, int(attn.num_v_bases), iters=30, tol=1e-4, seed=0)
    bi_trace4 = trace_reference_kmeans(bi_X4, int(attn.num_v_bases), iters=30, tol=1e-4, seed=0)
    bi_alignment = centroid_alignment_metrics(bi_trace1["final_centroids"], bi_trace2["final_centroids"])
    bi_alignment_b4 = centroid_alignment_metrics(bi_trace1["final_centroids"], bi_trace4["final_centroids"])
    bi_state1 = run_v_state(values["bi_b1"], bi_trace1["final_centroids"].to(values["bi_b1"].dtype), value_objective=attn.value_objective, group_size=int(attn.group_size), bits=int(attn.v_bits), v4_budget_fraction=float(attn.v4_budget_fraction))
    bi_state2 = run_v_state(values["bi_b2_row0"], bi_trace2["final_centroids"].to(values["bi_b1"].dtype), value_objective=attn.value_objective, group_size=int(attn.group_size), bits=int(attn.v_bits), v4_budget_fraction=float(attn.v4_budget_fraction))
    bi_state4 = run_v_state(values["bi_b4_row0"], bi_trace4["final_centroids"].to(values["bi_b1"].dtype), value_objective=attn.value_objective, group_size=int(attn.group_size), bits=int(attn.v_bits), v4_budget_fraction=float(attn.v4_budget_fraction))
    bi_packed = packed_v_metrics(bi_state2["cache"], bi_state1["cache"])
    bi_packed_b4 = packed_v_metrics(bi_state4["cache"], bi_state1["cache"])
    bi_assign = assignment_metrics(bi_state1["idx"], bi_state2["idx"], bi_alignment["got_to_ref"])
    bi_assign_b4 = assignment_metrics(bi_state1["idx"], bi_state4["idx"], bi_alignment_b4["got_to_ref"])

    trajectory_rows = []
    first_major = None
    input_rel = relative_l2(X2, X1)
    for iteration in TRACE_ITERS:
        if iteration not in trace1["snapshots"] or iteration not in trace2["snapshots"]:
            continue
        align_i = centroid_alignment_metrics(trace1["snapshots"][iteration], trace2["snapshots"][iteration])
        row = {
            "layer": layer_idx,
            "iteration": iteration,
            "raw_centroid_relative_l2": align_i["raw_v_centroid_relative_l2"],
            "aligned_centroid_relative_l2": align_i["aligned_v_centroid_relative_l2"],
            "permutation_explained_fraction": align_i["permutation_explained_fraction"],
            "assignment_difference_rate": None,
            "cluster_population_difference_l1": None,
            "max_centroid_shift_b1": trace1["shifts"].get(iteration),
            "max_centroid_shift_b2": trace2["shifts"].get(iteration),
        }
        if iteration in trace1["assignments"] and iteration in trace2["assignments"]:
            row["assignment_difference_rate"] = difference_rate(trace2["assignments"][iteration], trace1["assignments"][iteration])
        if iteration in trace1["populations"] and iteration in trace2["populations"]:
            row["cluster_population_difference_l1"] = float((trace2["populations"][iteration] - trace1["populations"][iteration]).abs().sum().item())
        trajectory_rows.append(row)
        if first_major is None and row["aligned_centroid_relative_l2"] > max(input_rel * 10.0, 1e-3):
            first_major = iteration

    normal_residual_rel = relative_l2(state2["residualized"], state1["residualized"])
    bi_residual_rel = relative_l2(bi_state2["residualized"], bi_state1["residualized"])
    result = {
        "layer": layer_idx,
        "hidden_metrics": hidden_metrics,
        "vproj_metrics": vproj_metrics,
        "kproj_normal_metrics": kproj_metrics,
        "v_kmeans_input_metrics": tensor_metric_dict(X2, X1),
        "initialization_indices_equal": bool(torch.equal(init_indices_1, init_indices_2)),
        "initial_centroid_relative_l2": init_rel,
        "reference_validation": prod_validation,
        "trajectory_rows": trajectory_rows,
        "first_major_v_centroid_divergence_iteration": first_major,
        "alignment": {k: v for k, v in alignment.items() if not torch.is_tensor(v)},
        "assignment": assign,
        "mask_metrics": {
            "normal_v_mask_difference_rate": difference_rate(state2["mask"], state1["mask"]),
            "bi_vproj_mask_difference_rate": difference_rate(bi_state2["mask"], bi_state1["mask"]),
        },
        "residualized": {
            "normal_residualized_v_relative_l2": normal_residual_rel,
            "bi_vproj_residualized_v_relative_l2": bi_residual_rel,
        },
        "packed_normal": packed_normal,
        "fixed_centroid": fixed,
        "bi_vproj": {
            "uses_existing_v2_kernel": True,
            "bi_vproj_b1_b2_exact": values["bi_b1_b2_exact"],
            "bi_vproj_b1_b4_exact": values["bi_b1_b4_exact"],
            "bi_vproj_v_centroid_relative_l2": bi_alignment["aligned_v_centroid_relative_l2"],
            "bi_vproj_assignment_difference_rate": bi_assign["aligned_assignment_difference_rate"],
            "bi_vproj_mask_difference_rate": difference_rate(bi_state2["mask"], bi_state1["mask"]),
            "bi_vproj_residualized_v_relative_l2": bi_residual_rel,
            "bi_vproj_packed_v_difference_rate": bi_packed["packed_v_payload_difference_rate"],
            "bi_vproj_b1_b4_v_centroid_relative_l2": bi_alignment_b4["aligned_v_centroid_relative_l2"],
            "bi_vproj_b1_b4_assignment_difference_rate": bi_assign_b4["aligned_assignment_difference_rate"],
            "bi_vproj_b1_b4_mask_difference_rate": difference_rate(bi_state4["mask"], bi_state1["mask"]),
            "bi_vproj_b1_b4_packed_v_difference_rate": bi_packed_b4["packed_v_payload_difference_rate"],
            "bi_vproj_packed": bi_packed,
            "bi_vproj_b4_packed": bi_packed_b4,
        },
        "v_kmeans_identical_input_deterministic": bool(torch.equal(trace1["final_centroids"], trace_reference_kmeans(X1, int(attn.num_v_bases), iters=30, tol=1e-4, seed=0)["final_centroids"])),
        "centroid_amplification": v_centroid_amplification(alignment["aligned_v_centroid_relative_l2"], input_rel),
    }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_trace_env()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = torch.device(args.device)
    tokenizer, config, model = load_model(dtype, device)
    primary = run_pair_trace(model, tokenizer, 0, device, full=True)
    multi_rows = []
    for layer_idx in (0, 8, 16, 31):
        trace = primary if layer_idx == 0 else run_pair_trace(model, tokenizer, layer_idx, device, full=False)
        multi_rows.append(
            {
                "layer": layer_idx,
                "v_proj_relative_l2": trace["vproj_metrics"]["relative_l2"],
                "v_centroid_relative_l2": trace["alignment"]["aligned_v_centroid_relative_l2"],
                "assignment_diff": trace["assignment"]["aligned_assignment_difference_rate"],
                "bi_vproj_recovery_centroid_relative_l2": trace["bi_vproj"]["bi_vproj_v_centroid_relative_l2"],
                "bi_vproj_assignment_diff": trace["bi_vproj"]["bi_vproj_assignment_difference_rate"],
            }
        )
    return {
        "actual_model_loaded": True,
        "model_config": {
            "num_hidden_layers": int(config.num_hidden_layers),
            "hidden_size": int(config.hidden_size),
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(config.num_key_value_heads),
        },
        "primary": primary,
        "multi_layer": multi_rows,
    }


def classify(results: dict[str, Any]) -> dict[str, Any]:
    p = results["primary"]
    hidden_exact = bool(p["hidden_metrics"]["exact"])
    bi = p["bi_vproj"]
    bi_complete = bool(
        bi["bi_vproj_b1_b2_exact"]
        and bi["bi_vproj_b1_b4_exact"]
        and bi["bi_vproj_v_centroid_relative_l2"] == 0.0
        and bi["bi_vproj_assignment_difference_rate"] == 0.0
        and bi["bi_vproj_mask_difference_rate"] == 0.0
        and bi["bi_vproj_residualized_v_relative_l2"] == 0.0
        and bi["bi_vproj_packed_v_difference_rate"] == 0.0
        and bi["bi_vproj_b1_b4_v_centroid_relative_l2"] == 0.0
        and bi["bi_vproj_b1_b4_assignment_difference_rate"] == 0.0
        and bi["bi_vproj_b1_b4_mask_difference_rate"] == 0.0
        and bi["bi_vproj_b1_b4_packed_v_difference_rate"] == 0.0
    )
    if not hidden_exact:
        classification = "PREFILL_V_TRACE_UPSTREAM_HIDDEN_DIVERGENCE"
        root = "UPSTREAM_HIDDEN_DIVERGENCE"
        next_task = "DEEPEN_PREFILL_V_STATE_TRACE"
    elif not (bi["bi_vproj_b1_b2_exact"] and bi["bi_vproj_b1_b4_exact"]):
        classification = "BI_VPROJ_CONTROL_NOT_BATCH_INVARIANT"
        root = "BI_VPROJ_CONTROL_NOT_BATCH_INVARIANT"
        next_task = "DEEPEN_PREFILL_V_STATE_TRACE"
    elif p["vproj_metrics"]["relative_l2"] > 0 and p["alignment"]["aligned_v_centroid_relative_l2"] > p["v_kmeans_input_metrics"]["relative_l2"] and bi_complete:
        classification = "PREFILL_V_KMEANS_NUMERICAL_AMPLIFICATION_CONFIRMED"
        root = "BATCH_NUMERICAL_NOISE_TRIGGERED_V_KMEANS_INSTABILITY"
        next_task = "INTEGRATE_BATCH_INVARIANT_KVPROJ_PREFILL_RUNTIME"
    elif p["assignment"]["aligned_assignment_difference_rate"] > 0:
        classification = "PREFILL_V_ASSIGNMENT_NUMERICAL_INSTABILITY"
        root = "V_ASSIGNMENT_NUMERICAL_INSTABILITY"
        next_task = "TRACE_AND_STABILIZE_V_ASSIGNMENT"
    elif p["mask_metrics"]["normal_v_mask_difference_rate"] > 0:
        classification = "PREFILL_V_PATTERN_SELECTION_NUMERICAL_INSTABILITY"
        root = "V_PATTERN_SELECTION_NUMERICS"
        next_task = "TRACE_V_PATTERN_SELECTION_NUMERICS"
    elif p["packed_normal"]["packed_v_payload_difference_rate"] and p["packed_normal"]["packed_v_payload_difference_rate"] > 0:
        classification = "PREFILL_V_QUANTIZATION_NUMERICAL_DIVERGENCE"
        root = "V_QUANTIZATION_NUMERICAL_DIVERGENCE"
        next_task = "TRACE_V_QUANTIZATION_BATCH_NUMERICS"
    else:
        classification = "PREFILL_V_DIVERGENCE_NOT_EXPLAINED_BY_VPROJ"
        root = "V_STATE_DIVERGENCE_NOT_EXPLAINED_BY_VPROJ"
        next_task = "DEEPEN_PREFILL_V_STATE_TRACE"
    return {"classification": classification, "root_cause_class": root, "next_task": next_task, "bi_complete": bi_complete}


def build_gate(results: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    p = results["primary"]
    return {
        "start_head": START_HEAD,
        "actual_model_loaded": results["actual_model_loaded"],
        "algorithm_changed": False,
        "quantization_changed": False,
        "selector_changed": False,
        "kmeans_changed": False,
        "production_vproj_changed": False,
        "k_payload_layout_changed": False,
        "v_page_abi_changed": False,
        "centroid_state_architecture_changed": False,
        "fused_value_arithmetic_changed": False,
        "hidden_input_exact": p["hidden_metrics"]["exact"],
        "normal_vproj_b1_b2_exact": p["vproj_metrics"]["exact"],
        "normal_vproj_relative_l2": p["vproj_metrics"]["relative_l2"],
        "normal_vproj_max_abs": p["vproj_metrics"]["max_abs"],
        "normal_vproj_cosine": p["vproj_metrics"]["cosine"],
        "normal_kproj_relative_l2": p["kproj_normal_metrics"]["relative_l2"],
        "v_kmeans_input_relative_l2": p["v_kmeans_input_metrics"]["relative_l2"],
        "v_kmeans_initialization_indices_equal": p["initialization_indices_equal"],
        "initial_v_centroid_relative_l2": p["initial_centroid_relative_l2"],
        "v_kmeans_trajectory_traced": True,
        "first_major_v_centroid_divergence_iteration": p["first_major_v_centroid_divergence_iteration"],
        "raw_v_centroid_relative_l2": p["alignment"]["raw_v_centroid_relative_l2"],
        "aligned_v_centroid_relative_l2": p["alignment"]["aligned_v_centroid_relative_l2"],
        "v_centroid_amplification": p["centroid_amplification"],
        "label_permutation_dominant": p["alignment"]["label_permutation_dominant"],
        "permutation_explained_fraction": p["alignment"]["permutation_explained_fraction"],
        "raw_v_assignment_difference_rate": p["assignment"]["raw_assignment_difference_rate"],
        "aligned_v_assignment_difference_rate": p["assignment"]["aligned_assignment_difference_rate"],
        "fixed_centroid_assignment_difference_rate": p["fixed_centroid"]["fixed_centroid_assignment_difference_rate"],
        "fixed_centroid_mask_difference_rate": p["fixed_centroid"]["fixed_centroid_mask_difference_rate"],
        "normal_v_mask_difference_rate": p["mask_metrics"]["normal_v_mask_difference_rate"],
        "normal_residualized_v_relative_l2": p["residualized"]["normal_residualized_v_relative_l2"],
        "packed_v_payload_difference_rate": p["packed_normal"]["packed_v_payload_difference_rate"],
        "packed_v_scale_relative_l2": p["packed_normal"]["packed_v_scale_relative_l2"],
        "packed_v_zero_relative_l2": p["packed_normal"]["packed_v_zero_relative_l2"],
        "bi_vproj_control_completed": True,
        "bi_vproj_b1_b2_exact": p["bi_vproj"]["bi_vproj_b1_b2_exact"],
        "bi_vproj_b1_b4_exact": p["bi_vproj"]["bi_vproj_b1_b4_exact"],
        "bi_vproj_v_centroid_relative_l2": p["bi_vproj"]["bi_vproj_v_centroid_relative_l2"],
        "bi_vproj_assignment_difference_rate": p["bi_vproj"]["bi_vproj_assignment_difference_rate"],
        "bi_vproj_mask_difference_rate": p["bi_vproj"]["bi_vproj_mask_difference_rate"],
        "bi_vproj_residualized_v_relative_l2": p["bi_vproj"]["bi_vproj_residualized_v_relative_l2"],
        "bi_vproj_packed_v_difference_rate": p["bi_vproj"]["bi_vproj_packed_v_difference_rate"],
        "bi_vproj_b1_b4_v_centroid_relative_l2": p["bi_vproj"]["bi_vproj_b1_b4_v_centroid_relative_l2"],
        "bi_vproj_b1_b4_assignment_difference_rate": p["bi_vproj"]["bi_vproj_b1_b4_assignment_difference_rate"],
        "bi_vproj_b1_b4_mask_difference_rate": p["bi_vproj"]["bi_vproj_b1_b4_mask_difference_rate"],
        "bi_vproj_b1_b4_packed_v_difference_rate": p["bi_vproj"]["bi_vproj_b1_b4_packed_v_difference_rate"],
        "v_kmeans_identical_input_deterministic": p["v_kmeans_identical_input_deterministic"],
        "multi_layer_sanity_completed": True,
        "root_cause_class": decision["root_cause_class"],
        "classification": decision["classification"],
        "next_task": decision["next_task"],
    }


def write_reports(results: dict[str, Any], gate: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p = results["primary"]
    write_json(REPORT_DIR / "v_projection_metrics.json", {"hidden": p["hidden_metrics"], "normal_vproj": p["vproj_metrics"], "normal_kproj": p["kproj_normal_metrics"], "v_kmeans_input": p["v_kmeans_input_metrics"]})
    write_csv(REPORT_DIR / "v_kmeans_trajectory.csv", p["trajectory_rows"], list(p["trajectory_rows"][0].keys()))
    write_json(REPORT_DIR / "centroid_alignment.json", p["alignment"])
    write_json(REPORT_DIR / "assignment_metrics.json", p["assignment"])
    write_json(REPORT_DIR / "mask_metrics.json", p["mask_metrics"])
    write_json(REPORT_DIR / "residualized_v_metrics.json", p["residualized"])
    write_json(REPORT_DIR / "packed_v_metrics.json", p["packed_normal"])
    write_json(REPORT_DIR / "fixed_centroid_control.json", p["fixed_centroid"])
    write_json(REPORT_DIR / "bi_vproj_control.json", p["bi_vproj"])
    write_csv(REPORT_DIR / "multi_layer.csv", results["multi_layer"], list(results["multi_layer"][0].keys()))
    write_json(REPORT_DIR / "final_gate.json", gate)
    docs = {
        "environment.md": f"# Environment\n\nStart HEAD: {START_HEAD}\n\nRepo: pytenter/Bounded-pattrenKV-method\n\nLocal: {REPO_ROOT}\n\nModel: {MODEL_PATH}\n\nCUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}\n",
        "current_blocker.md": "# Current Blocker\n\nS6-B.2.11 first structural divergence is prefill step0 layer0 `v_centroid_hash`.\n",
        "trace_methodology.md": "# Trace Methodology\n\nThe runner traces Request A ctx512, independent B1 versus B2 row0, with BI KProj enabled. Production forward is not modified.\n",
        "hidden_input.md": f"# Hidden Input\n\nExact: `{gate['hidden_input_exact']}`.\n",
        "v_projection_divergence.md": f"# V Projection Divergence\n\nExact: `{gate['normal_vproj_b1_b2_exact']}`. Relative L2: `{gate['normal_vproj_relative_l2']}`. K normal relative L2 control: `{gate['normal_kproj_relative_l2']}`.\n",
        "v_kmeans_initialization.md": f"# V K-Means Initialization\n\nIndices equal: `{gate['v_kmeans_initialization_indices_equal']}`. Initial centroid rel L2: `{gate['initial_v_centroid_relative_l2']}`.\n",
        "v_kmeans_trajectory.md": "# V K-Means Trajectory\n\nSee `v_kmeans_trajectory.csv`.\n",
        "centroid_alignment.md": "# Centroid Alignment\n\nSee `centroid_alignment.json`.\n",
        "fixed_centroid_control.md": "# Fixed Centroid Control\n\nSee `fixed_centroid_control.json`.\n",
        "v_assignment.md": "# V Assignment\n\nSee `assignment_metrics.json`.\n",
        "v_pattern_mask.md": "# V Pattern Mask\n\nSee `mask_metrics.json`.\n",
        "v_residualized.md": "# V Residualized\n\nSee `residualized_v_metrics.json`.\n",
        "packed_v_divergence.md": "# Packed V Divergence\n\nSee `packed_v_metrics.json`.\n",
        "bi_vproj_control.md": "# BI VProj Control\n\nDiagnostic only; uses existing V2 linear kernel with `v_proj.weight`. See `bi_vproj_control.json`.\n",
        "multi_layer_sanity.md": "# Multi Layer Sanity\n\nSee `multi_layer.csv`.\n",
        "root_cause_analysis.md": f"# Root Cause Analysis\n\nROOT_CAUSE={gate['root_cause_class']}\n\nCLASSIFICATION={gate['classification']}\n",
        "final_recommendation.md": f"# Final Recommendation\n\nCLASSIFICATION={gate['classification']}\n\nNEXT_TASK={gate['next_task']}\n",
    }
    for name, text in docs.items():
        (REPORT_DIR / name).write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    return parser.parse_args()


def main() -> None:
    results = run(parse_args())
    decision = classify(results)
    gate = build_gate(results, decision)
    write_reports(results, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
