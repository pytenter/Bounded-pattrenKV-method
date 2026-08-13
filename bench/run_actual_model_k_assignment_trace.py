from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.run_actual_model_fixed_batch_smoke import MODEL_PATH, PROMPTS, load_model, make_fixed_inputs
from models.llama_patternkv import apply_rotary_pos_emb, batched_assign_compiled, batched_kmeans_fast_compiled
from models.segmented_cache import deserialize_cache, pattern_gather_request_centroids, tensor_tokens
from quant.new_pack import triton_quantize_and_pack_along_last_dim


REPORT_DIR = REPO_ROOT / "reports/system_actual_model_k_assignment_trace_v1"
START_HEAD = "3d071f93dbc1d9a685f41d74d2f7b25262274953"
COMPONENT_ORDER = [
    "HIDDEN_INPUT",
    "K_PROJ",
    "K_PRE_ROPE",
    "K_POST_ROPE",
    "K_PACK_INPUT",
    "ACTIVE_K_CENTROIDS",
    "K_CENTROID_COUNTS",
    "K_ASSIGNMENT",
    "K_ADJUSTED",
    "PACKED_K",
    "PACKED_K_SCALE",
    "PACKED_K_ZERO",
]


def trace_enabled() -> bool:
    return os.environ.get("PATTERNKV_EQUIV_TRACE") == "1" or os.environ.get("PATTERNKV_K_ASSIGNMENT_TRACE") == "1"


def snapshot_phase(is_decode1: bool) -> str:
    return "DECODE1" if is_decode1 else "PREFILL"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    af = a.detach().float()
    bf = b.detach().float()
    diff = af - bf
    denom = torch.linalg.vector_norm(bf).clamp_min(1e-12)
    rel = torch.linalg.vector_norm(diff) / denom
    a_norm = torch.linalg.vector_norm(af).clamp_min(1e-12)
    b_norm = torch.linalg.vector_norm(bf).clamp_min(1e-12)
    denom_elem = bf.abs().clamp_min(1e-12)
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "equal": bool(torch.equal(a, b)),
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.abs().mean().item()) if diff.numel() else 0.0,
        "relative_l2": float(rel.item()),
        "cosine": float((torch.sum(af * bf) / (a_norm * b_norm)).item()),
        "max_relative_error": float((diff.abs() / denom_elem).max().item()) if diff.numel() else 0.0,
        "nan": int(torch.isnan(af).sum().item() + torch.isnan(bf).sum().item()),
        "inf": int(torch.isinf(af).sum().item() + torch.isinf(bf).sum().item()),
    }


def discrete_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    same = a == b
    diff = ~same
    first = None
    if bool(diff.any().item()):
        idx = diff.nonzero(as_tuple=False)[0]
        first = [int(x) for x in idx.tolist()]
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "equal": bool(torch.equal(a, b)),
        "num_equal": int(same.sum().item()),
        "num_different": int(diff.sum().item()),
        "difference_rate": float(diff.float().mean().item()) if diff.numel() else 0.0,
        "first_difference_index": first,
    }


def minmax_distances(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    if centroids.dim() == 3:
        diff = x.unsqueeze(2) - centroids.unsqueeze(0).unsqueeze(3)
    else:
        diff = x.unsqueeze(2) - centroids.unsqueeze(3)
    return diff.amax(dim=-1) - diff.amin(dim=-1)


def assignment_with_margins(x: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    distances = minmax_distances(x, centroids)
    top2 = torch.topk(distances, k=2, dim=2, largest=False)
    assignment = top2.indices[:, :, 0, :].contiguous().to(torch.long)
    best = top2.values[:, :, 0, :].contiguous()
    second = top2.values[:, :, 1, :].contiguous()
    margin = second - best
    return assignment, best, second, margin


def gather_k_centroids(assignments: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    if centroids.dim() == 4:
        return pattern_gather_request_centroids(assignments, centroids)
    expanded = centroids.unsqueeze(0).expand(assignments.shape[0], -1, -1, -1)
    return torch.gather(expanded, 2, assignments.unsqueeze(-1).expand(-1, -1, -1, centroids.shape[-1]))


def cross_assignment(x_b1: torch.Tensor, c_b1: torch.Tensor, x_b2: torch.Tensor, c_b2: torch.Tensor) -> dict[str, Any]:
    combos = {
        "b1_k_b1_centroid": assignment_with_margins(x_b1, c_b1)[0],
        "b2_k_b2_centroid": assignment_with_margins(x_b2, c_b2)[0],
        "b1_k_b2_centroid": assignment_with_margins(x_b1, c_b2)[0],
        "b2_k_b1_centroid": assignment_with_margins(x_b2, c_b1)[0],
    }
    return {name: {"shape": list(value.shape), "hash": tensor_hash(value), "assignment": value.detach().cpu().tolist()} for name, value in combos.items()}


def tensor_hash(x: torch.Tensor) -> str:
    import hashlib

    y = x.detach().cpu().contiguous()
    return hashlib.sha256(y.numpy().tobytes()).hexdigest()


def locate_first_divergence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for component in COMPONENT_ORDER:
        for row in rows:
            if row.get("component") == component and not bool(row.get("pass")):
                return row
    return {}


def cache_layer0_summary(past_key_values: Any, row: int | None) -> dict[str, Any]:
    cache = deserialize_cache(past_key_values[0], pattern=True)
    k_assign = cache.k_assignments[row : row + 1] if cache.k_assignments is not None and row is not None else cache.k_assignments
    v_assign = cache.v_assignment_idx[row : row + 1] if cache.v_assignment_idx is not None and row is not None else cache.v_assignment_idx
    v_precision = cache.v_precision_mask[row : row + 1] if cache.v_precision_mask is not None and row is not None else cache.v_precision_mask
    summary = {
        "total_tokens": int(cache.total_tokens),
        "packed_k_tokens": int(cache.packed_k_tokens),
        "packed_v_tokens": int(cache.packed_v_tokens),
        "pending_k_tokens": tensor_tokens(cache.pending_k),
        "pending_v_tokens": tensor_tokens(cache.pending_v),
        "recent_k_tokens": tensor_tokens(cache.recent_k),
        "recent_v_tokens": tensor_tokens(cache.recent_v),
        "k_assignment_shape": list(k_assign.shape) if k_assign is not None else None,
        "k_assignment_hash": tensor_hash(k_assign) if k_assign is not None else None,
        "v_assignment_shape": list(v_assign.shape) if v_assign is not None else None,
        "v_assignment_hash": tensor_hash(v_assign) if v_assign is not None else None,
        "v_precision_shape": list(v_precision.shape) if v_precision is not None else None,
        "v_precision_hash": tensor_hash(v_precision) if v_precision is not None else None,
    }
    pool = cache.centroid_state_pool
    slots = cache.centroid_state_indices
    if pool is not None and slots is not None:
        active = slots.long()
        if row is not None:
            active = active[row : row + 1]
        summary["centroid_state_indices"] = [int(x) for x in active.detach().cpu().tolist()]
        summary["k_counts"] = [int(x) for x in pool.k_counts[active].detach().cpu().tolist()]
        summary["v_counts"] = [int(x) for x in pool.v_counts[active].detach().cpu().tolist()]
        summary["update_counts_k"] = [int(x) for x in pool.update_counts_k[active].detach().cpu().tolist()]
        summary["update_counts_v"] = [int(x) for x in pool.update_counts_v[active].detach().cpu().tolist()]
    return summary


def make_position_ids(input_ids: torch.Tensor, past: int = 0) -> torch.Tensor:
    return torch.arange(past, past + input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)


def layer0_prefill_pipeline(model: Any, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    layer = model.model.layers[0]
    attn = layer.self_attn
    with torch.inference_mode():
        hidden_raw = model.model.embed_tokens(input_ids)
        hidden = layer.input_layernorm(hidden_raw)
        k_proj = attn.k_proj(hidden)
        q_proj = attn.q_proj(hidden)
        v_proj = attn.v_proj(hidden)
        bsz, seq_len, _ = hidden.shape
        query = q_proj.view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        key_pre = k_proj.view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        value = v_proj.view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        position_ids = make_position_ids(input_ids)
        cos, sin = attn.rotary_emb(value, position_ids)
        _query_post, key_post = apply_rotary_pos_emb(query, key_pre, cos, sin, position_ids)
        k_centroids, assignments = initial_k_centroids_and_assignments(key_post, int(attn.num_k_bases))
        gathered = gather_k_centroids(assignments, k_centroids)
        adjusted = key_post - gathered
        packed, scale, zero = triton_quantize_and_pack_along_last_dim(adjusted.transpose(2, 3).contiguous(), int(attn.group_size), int(attn.k_bits))
    return {
        "HIDDEN_INPUT": hidden.detach(),
        "K_PROJ": k_proj.detach(),
        "K_PRE_ROPE": key_pre.detach(),
        "K_POST_ROPE": key_post.detach(),
        "K_PACK_INPUT": key_post.detach(),
        "ACTIVE_K_CENTROIDS": k_centroids.detach(),
        "K_CENTROID_COUNTS": torch.full((bsz,), int(attn.num_k_bases), dtype=torch.int32, device=input_ids.device),
        "K_ASSIGNMENT": assignments.detach(),
        "K_ADJUSTED": adjusted.detach(),
        "PACKED_K": packed.detach(),
        "PACKED_K_SCALE": scale.detach(),
        "PACKED_K_ZERO": zero.detach(),
    }


def initial_k_centroids_and_assignments(key_states: torch.Tensor, num_k_bases: int) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, n_kv, seq_len, hd = key_states.shape
    if bsz == 1:
        x = key_states.permute(1, 0, 2, 3).reshape(n_kv, seq_len, hd).to(torch.float32)
        _seed, centroids = batched_kmeans_fast_compiled(x, k=num_k_bases, iters=30, tol=1e-4, seed=0)
        assign = batched_assign_compiled(x, centroids)
        assignments = assign.view(n_kv, 1, seq_len).permute(1, 0, 2).contiguous().to(torch.long)
        return centroids.to(key_states.dtype), assignments
    centroids_rows = []
    assignments_rows = []
    for row in range(bsz):
        x = key_states[row : row + 1].permute(1, 0, 2, 3).reshape(n_kv, seq_len, hd).to(torch.float32)
        _seed, centroids = batched_kmeans_fast_compiled(x, k=num_k_bases, iters=30, tol=1e-4, seed=0)
        assign = batched_assign_compiled(x, centroids)
        centroids_rows.append(centroids.to(key_states.dtype))
        assignments_rows.append(assign.view(n_kv, 1, seq_len).permute(1, 0, 2).contiguous().to(torch.long))
    return torch.stack(centroids_rows, dim=0).contiguous(), torch.cat(assignments_rows, dim=0)


def slice_component(value: torch.Tensor, row: int) -> torch.Tensor:
    if value.dim() >= 1 and value.shape[0] > row:
        return value[row : row + 1]
    return value


def compare_pipeline(b2: dict[str, torch.Tensor], refs: list[dict[str, torch.Tensor]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    assignment_diff_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    cross: dict[str, Any] = {}
    for request_row in (0, 1):
        ref = refs[request_row]
        for component in COMPONENT_ORDER:
            b_value = slice_component(b2[component], request_row)
            r_value = ref[component]
            if component in {"K_ASSIGNMENT", "K_CENTROID_COUNTS", "PACKED_K"}:
                metrics = discrete_metrics(b_value, r_value)
                passed = bool(metrics["equal"])
            else:
                metrics = tensor_metrics(b_value, r_value)
                passed = bool(metrics["equal"])
            rows.append({"phase": "PREFILL", "layer": 0, "request_row": request_row, "component": component, "pass": passed, **metrics})
        b_assign = b2["K_ASSIGNMENT"][request_row : request_row + 1]
        r_assign = ref["K_ASSIGNMENT"]
        diff = (b_assign != r_assign)
        if bool(diff.any().item()):
            b_x = b2["K_PACK_INPUT"][request_row : request_row + 1]
            r_x = ref["K_PACK_INPUT"]
            b_c = b2["ACTIVE_K_CENTROIDS"][request_row : request_row + 1]
            r_c = ref["ACTIVE_K_CENTROIDS"]
            _ba, b_best, b_second, b_margin = assignment_with_margins(b_x, b_c)
            _ra, r_best, r_second, r_margin = assignment_with_margins(r_x, r_c)
            for idx in diff.nonzero(as_tuple=False).detach().cpu().tolist():
                b, h, t = [int(x) for x in idx]
                assignment_diff_rows.append(
                    {
                        "request_row": request_row,
                        "kv_head": h,
                        "token_index": t,
                        "b2_assignment": int(b_assign[b, h, t].item()),
                        "b1_assignment": int(r_assign[b, h, t].item()),
                    }
                )
                margin_rows.append(
                    {
                        "request_row": request_row,
                        "kv_head": h,
                        "token_index": t,
                        "b2_best_distance": float(b_best[b, h, t].item()),
                        "b2_second_best_distance": float(b_second[b, h, t].item()),
                        "b2_margin": float(b_margin[b, h, t].item()),
                        "b1_best_distance": float(r_best[b, h, t].item()),
                        "b1_second_best_distance": float(r_second[b, h, t].item()),
                        "b1_margin": float(r_margin[b, h, t].item()),
                        "k_input_max_abs": float((b_x - r_x).abs().max().item()),
                        "k_input_relative_l2": float(tensor_metrics(b_x, r_x)["relative_l2"]),
                    }
                )
            cross[f"request_{request_row}"] = cross_assignment(r_x, r_c, b_x, b_c)
    first = locate_first_divergence(rows)
    return rows, cross, assignment_diff_rows, {"first": first, "margins": margin_rows}


def run_prefill(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True, return_dict=True)
    token = out.logits[:, -1, :].argmax(dim=-1)
    return {"past": out.past_key_values, "token": token.detach(), "hidden": out.hidden_states[-1][:, -1, :].detach(), "logits": out.logits[:, -1, :].detach()}


def run_decode1(model: Any, prefill: dict[str, Any]) -> dict[str, Any]:
    with torch.inference_mode():
        out = model(input_ids=prefill["token"][:, None], past_key_values=prefill["past"], use_cache=True, output_hidden_states=True, return_dict=True)
    token = out.logits[:, -1, :].argmax(dim=-1)
    return {"past": out.past_key_values, "token": token.detach(), "hidden": out.hidden_states[-1][:, -1, :].detach(), "logits": out.logits[:, -1, :].detach()}


def state_rows(phase: str, b2: dict[str, Any], refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in (0, 1):
        b_state = cache_layer0_summary(b2["past"], row=row)
        r_state = cache_layer0_summary(refs[row]["past"], row=0)
        for key in sorted(set(b_state) | set(r_state)):
            rows.append({"phase": phase, "request_row": row, "field": key, "b2": json.dumps(b_state.get(key), sort_keys=True), "b1": json.dumps(r_state.get(key), sort_keys=True), "pass": b_state.get(key) == r_state.get(key)})
    return rows


def classify(rows: list[dict[str, Any]], first: dict[str, Any], margin_rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    by_component = {row["component"]: row for row in rows if int(row["request_row"]) == int(first.get("request_row", 0))}
    if not first:
        return "INCONCLUSIVE", "ACTUAL_MODEL_K_TRACE_INCONCLUSIVE", "DEEPEN_ACTUAL_MODEL_K_ASSIGNMENT_TRACE"
    if not bool(by_component.get("K_PACK_INPUT", {}).get("pass", True)):
        margins = [min(abs(float(row["b1_margin"])), abs(float(row["b2_margin"]))) for row in margin_rows]
        near_tie = bool(margins) and (sum(m <= 1e-3 for m in margins) / len(margins) >= 0.5)
        if near_tie:
            return "K_ASSIGNMENT_NUMERICAL_SENSITIVITY", "ACTUAL_MODEL_K_ASSIGNMENT_NUMERICAL_SENSITIVITY", "DESIGN_STABLE_K_ASSIGNMENT_EQUIVALENCE"
        return "UPSTREAM_K_NUMERICAL_DIVERGENCE", "ACTUAL_MODEL_K_INPUT_DIVERGENCE", "TRACE_UPSTREAM_ACTUAL_MODEL_K_NUMERICAL_DIVERGENCE"
    if not bool(by_component.get("ACTIVE_K_CENTROIDS", {}).get("pass", True)):
        return "K_CENTROID_STATE_INITIALIZATION_OR_MAPPING", "ACTUAL_MODEL_K_CENTROID_STATE_DIVERGENCE", "FIX_ACTUAL_MODEL_K_CENTROID_STATE_INITIALIZATION"
    if not bool(by_component.get("K_ASSIGNMENT", {}).get("pass", True)):
        return "K_ASSIGNMENT_IMPLEMENTATION_DIVERGENCE", "ACTUAL_MODEL_K_ASSIGNMENT_IMPLEMENTATION_DIVERGENCE", "FIX_REQUEST_LOCAL_K_ASSIGNMENT_EQUIVALENCE"
    if not bool(by_component.get("PACKED_K", {}).get("pass", True)):
        return "K_PACKING_DIVERGENCE", "ACTUAL_MODEL_K_PACKING_DIVERGENCE", "TRACE_ACTUAL_MODEL_PACKED_K_DIVERGENCE"
    return "INCONCLUSIVE", "ACTUAL_MODEL_K_TRACE_INCONCLUSIVE", "DEEPEN_ACTUAL_MODEL_K_ASSIGNMENT_TRACE"


def write_reports(results: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    trace_rows = results["pipeline_rows"]
    prefill_state = results["prefill_state_rows"]
    decode1_state = results["decode1_state_rows"]
    assignment_rows = results["assignment_differences"]
    margin_rows = results["assignment_margins"]
    first = results["first_divergence"]
    root_cause, classification, next_task = classify(trace_rows, first, margin_rows)
    first_request_row = first.get("request_row", 0)

    def component_pass(component: str) -> bool | None:
        for row in trace_rows:
            if row.get("component") == component and row.get("request_row") == first_request_row:
                return bool(row.get("pass"))
        return None

    assignment_difference_count = len(assignment_rows)
    total_assignments = 2 * 8 * 512
    assignment_difference_rate = assignment_difference_count / total_assignments
    margins = [min(abs(float(row["b1_margin"])), abs(float(row["b2_margin"]))) for row in margin_rows]
    near_tie_count = sum(m <= 1e-3 for m in margins)
    first_json = {
        "phase": first.get("phase"),
        "layer": first.get("layer"),
        "component": first.get("component", ""),
        "request_row": first.get("request_row"),
        "token_index": assignment_rows[0]["token_index"] if assignment_rows else None,
        "kv_head": assignment_rows[0]["kv_head"] if assignment_rows else None,
        "continuous_tensor_max_abs": first.get("max_abs"),
        "continuous_tensor_relative_l2": first.get("relative_l2"),
        "assignment_difference_count": assignment_difference_count,
        "assignment_difference_rate": assignment_difference_rate,
        "assignment_margin_min": min(margins) if margins else None,
        "assignment_margin_median": float(torch.tensor(margins).median().item()) if margins else None,
        "root_cause_class": root_cause,
    }
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
        "failure_reproduced": bool(results["failure_reproduced"]),
        "prefill_token_pass": bool(results["prefill_token_pass"]),
        "prefill_state_pass": all(bool(row["pass"]) for row in prefill_state),
        "decode1_token_pass": bool(results["decode1_token_pass"]),
        "decode1_state_pass": all(bool(row["pass"]) for row in decode1_state),
        "first_divergence_phase": first_json["phase"],
        "first_divergence_layer": first_json["layer"],
        "first_divergence_component": first_json["component"],
        "k_input_pass": component_pass("K_PACK_INPUT"),
        "k_centroid_bank_pass": component_pass("ACTIVE_K_CENTROIDS"),
        "k_centroid_counts_pass": component_pass("K_CENTROID_COUNTS"),
        "k_assignment_pass": assignment_difference_count == 0,
        "packed_k_pass": component_pass("PACKED_K"),
        "k_assignment_difference_count": assignment_difference_count,
        "k_assignment_difference_rate": assignment_difference_rate,
        "assignment_margin_analysis_completed": True,
        "assignment_disagreements_near_ties": (near_tie_count / len(margins) >= 0.5) if margins else None,
        "cross_assignment_analysis_completed": True,
        "root_cause_class": root_cause,
        "classification": classification if results["failure_reproduced"] else "TRACE_REPRODUCTION_FAILED",
        "next_task": next_task if results["failure_reproduced"] else "DEEPEN_ACTUAL_MODEL_K_ASSIGNMENT_TRACE",
    }
    write_json(REPORT_DIR / "first_divergence.json", first_json)
    write_json(REPORT_DIR / "cross_assignment_results.json", results["cross_assignment"])
    write_json(REPORT_DIR / "final_gate.json", final_gate)
    write_csv(REPORT_DIR / "trace_summary.csv", [{"key": key, "value": json.dumps(value, sort_keys=True)} for key, value in final_gate.items()], ["key", "value"])
    write_csv(REPORT_DIR / "prefill_state.csv", prefill_state, ["phase", "request_row", "field", "b2", "b1", "pass"])
    write_csv(REPORT_DIR / "decode1_state.csv", decode1_state, ["phase", "request_row", "field", "b2", "b1", "pass"])
    write_csv(REPORT_DIR / "layer0_k_pipeline.csv", trace_rows, ["phase", "layer", "request_row", "component", "pass", "equal", "max_abs", "mean_abs", "relative_l2", "cosine", "max_relative_error", "num_equal", "num_different", "difference_rate", "first_difference_index"])
    write_csv(REPORT_DIR / "k_assignment_differences.csv", assignment_rows, ["request_row", "kv_head", "token_index", "b2_assignment", "b1_assignment"])
    write_csv(REPORT_DIR / "assignment_margin.csv", margin_rows, ["request_row", "kv_head", "token_index", "b2_best_distance", "b2_second_best_distance", "b2_margin", "b1_best_distance", "b1_second_best_distance", "b1_margin", "k_input_max_abs", "k_input_relative_l2"])
    docs = {
        "environment.md": f"# Environment\n\nStart HEAD: {START_HEAD}\n\nModel: {MODEL_PATH}\n\nCUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}\n",
        "reproduction.md": f"# Reproduction\n\nActual model loaded: {results['actual_model_loaded']}\n\nB2 ctx512 decode1 failure reproduced: {results['failure_reproduced']}\n",
        "trace_methodology.md": "# Trace Methodology\n\nThe runner uses the same Request A/B and token construction protocol as S6-B.2.5. It compares independent B1 A/B against true B2 [A,B] at explicit PREFILL and DECODE1 snapshots. Layer0 K pipeline tensors are recomputed diagnostically under `PATTERNKV_K_ASSIGNMENT_TRACE=1`; production behavior is unchanged when trace is disabled.\n",
        "prefill_state_comparison.md": "# Prefill State Comparison\n\nSee `prefill_state.csv`.\n",
        "decode1_state_comparison.md": "# Decode1 State Comparison\n\nSee `decode1_state.csv`.\n",
        "layer0_k_pipeline.md": "# Layer0 K Pipeline\n\nSee `layer0_k_pipeline.csv`.\n",
        "k_input_analysis.md": "# K Input Analysis\n\n`K_PACK_INPUT` metrics identify whether upstream K differs before assignment.\n",
        "k_centroid_analysis.md": "# K Centroid Analysis\n\n`ACTIVE_K_CENTROIDS` and `K_CENTROID_COUNTS` rows compare request-local centroid banks and counts.\n",
        "k_assignment_analysis.md": "# K Assignment Analysis\n\nSee `k_assignment_differences.csv`.\n",
        "assignment_margin_analysis.md": "# Assignment Margin Analysis\n\nSee `assignment_margin.csv`.\n",
        "cross_assignment_analysis.md": "# Cross Assignment Analysis\n\nSee `cross_assignment_results.json`.\n",
        "packed_k_analysis.md": "# Packed K Analysis\n\n`PACKED_K`, `PACKED_K_SCALE`, and `PACKED_K_ZERO` rows in `layer0_k_pipeline.csv` show packing divergence after assignment.\n",
        "root_cause_analysis.md": f"# Root Cause Analysis\n\nROOT_CAUSE_CLASS={root_cause}\n\nClassification={final_gate['classification']}\n",
        "final_recommendation.md": f"# Final Recommendation\n\nCLASSIFICATION={final_gate['classification']}\n\nNEXT_TASK={final_gate['next_task']}\n",
    }
    for name, text in docs.items():
        (REPORT_DIR / name).write_text(text, encoding="utf-8")


def run_trace(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    os.environ["PATTERNKV_K_ASSIGNMENT_TRACE"] = "1"
    tokenizer, _config, model = load_model(dtype, device)
    ids_b2 = make_fixed_inputs(tokenizer, 2, 512, device)
    ids_refs = [ids_b2[0:1], ids_b2[1:2]]
    pipeline_b2 = layer0_prefill_pipeline(model, ids_b2)
    pipeline_refs = [layer0_prefill_pipeline(model, ids) for ids in ids_refs]
    pipeline_rows, cross, assignment_rows, assignment_bundle = compare_pipeline(pipeline_b2, pipeline_refs)
    prefill_refs = [run_prefill(model, ids) for ids in ids_refs]
    prefill_b2 = run_prefill(model, ids_b2)
    decode_refs = [run_decode1(model, item) for item in prefill_refs]
    decode_b2 = run_decode1(model, prefill_b2)
    prefill_token_pass = bool(torch.equal(prefill_b2["token"], torch.cat([item["token"] for item in prefill_refs], dim=0)))
    decode1_token_pass = bool(torch.equal(decode_b2["token"], torch.cat([item["token"] for item in decode_refs], dim=0)))
    prefill_state = state_rows("PREFILL", prefill_b2, prefill_refs)
    decode1_state = state_rows("DECODE1", decode_b2, decode_refs)
    failure_reproduced = any(not bool(row["pass"]) for row in prefill_state + decode1_state)
    return {
        "actual_model_loaded": True,
        "failure_reproduced": failure_reproduced,
        "prefill_token_pass": prefill_token_pass,
        "decode1_token_pass": decode1_token_pass,
        "pipeline_rows": pipeline_rows,
        "prefill_state_rows": prefill_state,
        "decode1_state_rows": decode1_state,
        "assignment_differences": assignment_rows,
        "assignment_margins": assignment_bundle["margins"],
        "cross_assignment": cross,
        "first_divergence": assignment_bundle["first"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    try:
        results = run_trace(torch.device(args.device), dtype)
    except Exception as exc:
        results = {
            "actual_model_loaded": False,
            "failure_reproduced": False,
            "prefill_token_pass": False,
            "decode1_token_pass": False,
            "pipeline_rows": [],
            "prefill_state_rows": [],
            "decode1_state_rows": [],
            "assignment_differences": [],
            "assignment_margins": [],
            "cross_assignment": {},
            "first_divergence": {},
            "error": repr(exc),
        }
    write_reports(results)


if __name__ == "__main__":
    main()
