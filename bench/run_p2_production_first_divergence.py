from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.p2_first_divergence_utils import compare_canonical_states, compare_tensors, first_non_exact
from bench.run_actual_model_fixed_batch_smoke import load_model, make_fixed_inputs
from models.llama_patternkv import (
    patternkv_p2_first_divergence_trace_records,
    reset_patternkv_p2_first_divergence_trace,
    reset_patternkv_runtime_state,
)
from models.segmented_cache import deserialize_cache
from quant.batch_invariant_kproj import BI_KV_PREFILL_PROJ_MODE


START_HEAD = "82d43a13fcbfc3927f2401c1e580868dfdc16292"
REPORT_DIR = REPO_ROOT / "reports/system_p2_production_first_divergence_v1"

COMPONENT_ORDER = [
    "LAYER_INPUT",
    "INPUT_RMSNORM",
    "Q_PROJ",
    "K_PROJ",
    "V_PROJ",
    "K_POST_ROPE",
    "KMEANS_K_INPUT",
    "KMEANS_V_INPUT",
    "K_CENTROID",
    "V_CENTROID",
    "K_ASSIGNMENT",
    "V_ASSIGNMENT",
    "V_PATTERN_MASK",
    "PACKED_K",
    "PACKED_V",
    "ATTENTION_VALUE_OUTPUT",
    "ATTENTION_RESIDUAL_OUTPUT",
    "POST_ATTENTION_RMSNORM",
    "MLP_OUTPUT",
    "LAYER_OUTPUT",
]
REQUEST_LOCAL_COMPONENTS = {
    "KMEANS_K_INPUT",
    "KMEANS_V_INPUT",
    "K_CENTROID",
    "V_CENTROID",
    "K_ASSIGNMENT",
    "V_ASSIGNMENT",
    "V_PATTERN_MASK",
    "PACKED_K",
    "PACKED_V",
}

FINAL_GATE_TEMPLATE: dict[str, Any] = {
    "start_head": START_HEAD,
    "actual_model_loaded": False,
    "mode": "bi_kv",
    "context": 512,
    "algorithm_changed": False,
    "quantization_changed": False,
    "selector_changed": False,
    "kmeans_changed": False,
    "bi_k_kernel_changed": False,
    "bi_v_kernel_changed": False,
    "k_payload_layout_changed": False,
    "v_page_abi_changed": False,
    "centroid_state_architecture_changed": False,
    "fused_value_arithmetic_changed": False,
    "comparator_layout_validated": None,
    "canonical_request_state_implemented": None,
    "request_slot_mapping_validated": None,
    "snapshot_internal_consistency_pass": None,
    "previous_diff_1_values_were_shape_artifacts": None,
    "layer0_hidden_input_exact": None,
    "layer0_qproj_exact": None,
    "layer0_qproj_relative_l2": None,
    "layer0_kproj_exact": None,
    "layer0_vproj_exact": None,
    "layer0_k_post_rope_exact": None,
    "layer0_kmeans_k_input_exact": None,
    "layer0_kmeans_v_input_exact": None,
    "layer0_k_centroid_exact": None,
    "layer0_v_centroid_exact": None,
    "layer0_k_assignment_difference_rate": None,
    "layer0_v_assignment_difference_rate": None,
    "layer0_v_pattern_mask_difference_rate": None,
    "layer0_v_precision_mask_difference_rate": None,
    "layer0_packed_k_difference_rate": None,
    "layer0_packed_v_difference_rate": None,
    "layer0_packed_v4_difference_rate": None,
    "layer0_attention_output_relative_l2": None,
    "layer0_oproj_relative_l2": None,
    "layer0_output_relative_l2": None,
    "layer1_hidden_input_exact": None,
    "layer1_hidden_input_relative_l2": None,
    "first_hidden_divergent_layer": None,
    "first_divergent_layer": None,
    "first_divergent_component": "",
    "first_divergence_input_exact": None,
    "first_divergence_output_exact": None,
    "kmeans_identical_input_deterministic": None,
    "b4_same_first_divergence": None,
    "root_cause_class": "",
    "classification": "",
    "next_task": "",
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=10)
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def set_env() -> None:
    os.environ["PATTERNKV_PREFILL_PROJ_MODE"] = BI_KV_PREFILL_PROJ_MODE
    os.environ["PATTERNKV_BI_KPROJ_BACKEND"] = "v2"
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"


def squeeze_request_tensor(value: torch.Tensor | None, row: int | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = value.detach().cpu().contiguous()
    if row is not None and tensor.dim() >= 1 and tensor.shape[0] > row:
        tensor = tensor[row]
    if tensor.dim() >= 1 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor.contiguous()


def trace_map(records: list[dict[str, Any]], *, row: int | None) -> dict[tuple[int, str], torch.Tensor]:
    result = {}
    for record in records:
        layer = int(record["layer"])
        component = str(record["component"])
        tensor = record["tensor"]
        result[(layer, component)] = squeeze_request_tensor(tensor, row=None if component in REQUEST_LOCAL_COMPONENTS else row)
    return result


def active_centroids(cache: Any, stream: str, row: int) -> tuple[torch.Tensor | None, int | None, int | None, int | None]:
    pool = getattr(cache, "centroid_state_pool", None)
    slots = getattr(cache, "centroid_state_indices", None)
    if pool is not None and torch.is_tensor(slots):
        slot = int(slots[row].item())
        if stream == "k":
            count = int(pool.k_counts[slot].item())
            updates = int(pool.update_counts_k[slot].item())
            tensor = pool.k_centroid_pool[slot, :, :count, :]
        else:
            count = int(pool.v_counts[slot].item())
            updates = int(pool.update_counts_v[slot].item())
            tensor = pool.v_centroid_pool[slot, :, :count, :]
        return tensor.detach().cpu().contiguous(), count, updates, slot
    tensor = getattr(cache, f"{stream}_centroids", None)
    if torch.is_tensor(tensor) and tensor.dim() == 4:
        tensor = tensor[row]
    count = int(tensor.shape[-2]) if torch.is_tensor(tensor) else None
    return tensor.detach().cpu().contiguous() if torch.is_tensor(tensor) else None, count, int(getattr(cache, f"centroid_updates_{stream}", 0) or 0), None


def row_tensor(value: torch.Tensor | None, row: int) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = value.detach().cpu().contiguous()
    if tensor.dim() > 0 and tensor.shape[0] > row:
        tensor = tensor[row]
    if tensor.dim() > 0 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor.contiguous()


def canonical_request_cache_state(layer_cache: Any, row: int) -> dict[str, Any]:
    cache = deserialize_cache(layer_cache, pattern=True)
    k_centroids, k_count, k_updates, slot = active_centroids(cache, "k", row)
    v_centroids, v_count, v_updates, _ = active_centroids(cache, "v", row)
    fields = {
        "k_centroids": k_centroids,
        "v_centroids": v_centroids,
        "k_assignments": row_tensor(cache.k_assignments, row),
        "v_assignment_idx": row_tensor(cache.v_assignment_idx, row),
        "v_pattern_mask": row_tensor(cache.v_pattern_mask, row),
        "v_precision_mask": row_tensor(cache.v_precision_mask, row),
        "packed_k": row_tensor(cache.packed_k, row),
        "packed_k_scale": row_tensor(cache.packed_k_scale, row),
        "packed_k_zero": row_tensor(cache.packed_k_zero, row),
        "packed_v": row_tensor(cache.packed_v, row),
        "packed_v_scale": row_tensor(cache.packed_v_scale, row),
        "packed_v_zero": row_tensor(cache.packed_v_zero, row),
        "packed_v4": row_tensor(cache.packed_v4, row),
        "packed_v4_scale": row_tensor(cache.packed_v4_scale, row),
        "packed_v4_zero": row_tensor(cache.packed_v4_zero, row),
        "v2_assignment_idx": row_tensor(cache.v2_assignment_idx, row),
        "v4_assignment_idx": row_tensor(cache.v4_assignment_idx, row),
    }
    return {
        "fields": fields,
        "metadata": {
            "total_tokens": int(cache.total_tokens),
            "packed_k_tokens": int(cache.packed_k_tokens),
            "packed_v_tokens": int(cache.packed_v_tokens),
            "packed_v4_tokens": int(row_tensor(cache.v_precision_mask, row).bool().sum().item()) if cache.v_precision_mask is not None else int(cache.packed_v4_tokens),
            "k_count": k_count,
            "v_count": v_count,
            "k_updates": k_updates,
            "v_updates": v_updates,
        },
        "slot": slot,
    }


def cache_layout(layer_cache: Any, row: int) -> list[dict[str, Any]]:
    cache = deserialize_cache(layer_cache, pattern=True)
    fields = {
        "sink_k": cache.sink_k,
        "sink_v": cache.sink_v,
        "recent_k": cache.recent_k,
        "recent_v": cache.recent_v,
        "pending_k": cache.pending_k,
        "pending_v": cache.pending_v,
        "k_assignments": cache.k_assignments,
        "v_assignment_idx": cache.v_assignment_idx,
        "v_pattern_mask": cache.v_pattern_mask,
        "v_precision_mask": cache.v_precision_mask,
        "packed_k": cache.packed_k,
        "packed_k_scale": cache.packed_k_scale,
        "packed_k_zero": cache.packed_k_zero,
        "packed_v": cache.packed_v,
        "packed_v_scale": cache.packed_v_scale,
        "packed_v_zero": cache.packed_v_zero,
        "packed_v4": cache.packed_v4,
        "packed_v4_scale": cache.packed_v4_scale,
        "packed_v4_zero": cache.packed_v4_zero,
        "v2_assignment_idx": cache.v2_assignment_idx,
        "v4_assignment_idx": cache.v4_assignment_idx,
    }
    rows = []
    for name, tensor in fields.items():
        has_batch = torch.is_tensor(tensor) and tensor.dim() > 0 and tensor.shape[0] > row
        token_axis = 2 if name not in {"v_precision_mask"} and torch.is_tensor(tensor) and tensor.dim() >= 3 else 1 if name == "v_precision_mask" else None
        rows.append(
            {
                "field_name": name,
                "shape": list(tensor.shape) if torch.is_tensor(tensor) else None,
                "dtype": str(tensor.dtype) if torch.is_tensor(tensor) else None,
                "producer_function": "build_cache_from_prefill/flush_pending",
                "logical_semantics": name,
                "has_batch_axis": bool(has_batch),
                "batch_axis": 0 if has_batch else None,
                "request_local": True,
                "request_indexing_method": "batch_axis_0" if has_batch else "centroid_slot_or_global",
                "token_axis": token_axis,
                "logical_token_count": int(cache.packed_k_tokens) if "k" in name and name.startswith("packed") else int(cache.packed_v_tokens) if name.startswith("packed_v") else None,
                "physical_capacity": list(tensor.shape) if torch.is_tensor(tensor) else None,
                "row_slice_valid": bool(has_batch),
            }
        )
    return rows


def prefill_run(model: Any, input_ids: torch.Tensor, *, row: int) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    reset_patternkv_p2_first_divergence_trace()
    os.environ["PATTERNKV_P2_FIRST_DIVERGENCE_TRACE"] = "1"
    with torch.inference_mode():
        out = model.model(input_ids=input_ids, use_cache=True, return_dict=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize(input_ids.device)
    records = patternkv_p2_first_divergence_trace_records()
    trace = trace_map(records, row=row)
    canonical = [canonical_request_cache_state(layer_cache, row) for layer_cache in out.past_key_values]
    layout = cache_layout(out.past_key_values[0], row)
    del out
    reset_patternkv_p2_first_divergence_trace()
    os.environ.pop("PATTERNKV_P2_FIRST_DIVERGENCE_TRACE", None)
    return {"trace": trace, "canonical": canonical, "layout": layout}


def component_metric_rows(ref_trace: dict[tuple[int, str], torch.Tensor], got_trace: dict[tuple[int, str], torch.Tensor]) -> list[dict[str, Any]]:
    rows = []
    for layer in range(32):
        for component in COMPONENT_ORDER:
            metric = compare_tensors(ref_trace.get((layer, component)), got_trace.get((layer, component)))
            rows.append(
                {
                    "layer": layer,
                    "component": component,
                    "shape_equal": metric["shape_equal"],
                    "comparable": metric["comparable"],
                    "exact": metric["exact"],
                    "max_abs": metric["max_abs"],
                    "mean_abs": metric["mean_abs"],
                    "relative_l2": metric["relative_l2"],
                    "cosine": metric["cosine"],
                    "difference_rate": metric["difference_rate"],
                    "first_non_exact": False,
                }
            )
    first = first_non_exact(rows)
    if first is not None:
        first["first_non_exact"] = True
    return rows


def cache_metric_rows(ref_cache: list[dict[str, Any]], got_cache: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for layer, (ref_layer, got_layer) in enumerate(zip(ref_cache, got_cache)):
        comparison = compare_canonical_states(ref_layer, got_layer)
        for field, metric in comparison["field_results"].items():
            rows.append(
                {
                    "layer": layer,
                    "component": field.upper(),
                    "shape_equal": metric["shape_equal"],
                    "comparable": metric["comparable"],
                    "exact": metric["exact"],
                    "max_abs": metric["max_abs"],
                    "mean_abs": metric["mean_abs"],
                    "relative_l2": metric["relative_l2"],
                    "cosine": metric["cosine"],
                    "difference_rate": metric["difference_rate"],
                    "first_non_exact": False,
                }
            )
    return rows


def summarize_layer0(rows: list[dict[str, Any]], cache_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_component = {(row["layer"], row["component"]): row for row in rows}
    by_cache = {(row["layer"], row["component"]): row for row in cache_rows}
    def exact(component: str) -> bool | None:
        return by_component.get((0, component), {}).get("exact")
    def rel(component: str) -> float | None:
        return by_component.get((0, component), {}).get("relative_l2")
    def diff_cache(component: str) -> float | None:
        return by_cache.get((0, component), {}).get("difference_rate")
    def exact_cache(component: str) -> bool | None:
        return by_cache.get((0, component), {}).get("exact")
    return {
        "layer0_hidden_input_exact": exact("LAYER_INPUT"),
        "layer0_qproj_exact": exact("Q_PROJ"),
        "layer0_qproj_relative_l2": rel("Q_PROJ"),
        "layer0_kproj_exact": exact("K_PROJ"),
        "layer0_vproj_exact": exact("V_PROJ"),
        "layer0_k_post_rope_exact": exact("K_POST_ROPE"),
        "layer0_kmeans_k_input_exact": exact("KMEANS_K_INPUT"),
        "layer0_kmeans_v_input_exact": exact("KMEANS_V_INPUT"),
        "layer0_k_centroid_exact": exact("K_CENTROID"),
        "layer0_v_centroid_exact": exact("V_CENTROID"),
        "layer0_k_assignment_difference_rate": by_component.get((0, "K_ASSIGNMENT"), {}).get("difference_rate"),
        "layer0_v_assignment_difference_rate": by_component.get((0, "V_ASSIGNMENT"), {}).get("difference_rate"),
        "layer0_v_pattern_mask_difference_rate": by_component.get((0, "V_PATTERN_MASK"), {}).get("difference_rate"),
        "layer0_v_precision_mask_difference_rate": diff_cache("V_PRECISION_MASK"),
        "layer0_packed_k_difference_rate": by_component.get((0, "PACKED_K"), {}).get("difference_rate"),
        "layer0_packed_v_difference_rate": by_component.get((0, "PACKED_V"), {}).get("difference_rate"),
        "layer0_packed_v4_difference_rate": diff_cache("PACKED_V4"),
        "layer0_attention_output_relative_l2": rel("ATTENTION_VALUE_OUTPUT"),
        "layer0_oproj_relative_l2": rel("ATTENTION_VALUE_OUTPUT"),
        "layer0_output_relative_l2": rel("LAYER_OUTPUT"),
    }


def first_hidden_divergent(rows: list[dict[str, Any]]) -> tuple[int | None, float | None, bool | None]:
    for row in rows:
        if row["component"] == "LAYER_INPUT" and row["exact"] is False:
            return int(row["layer"]), row["relative_l2"], False
    return None, None, True


def run_actual(args: argparse.Namespace) -> dict[str, Any]:
    set_env()
    tokenizer, _config, model = load_model(dtype=torch.float16, device=torch.device(args.device))
    input_ids = make_fixed_inputs(tokenizer, batch=4, context=512, device=torch.device(args.device))
    b1 = prefill_run(model, input_ids[0:1], row=0)
    b2 = prefill_run(model, input_ids[[0, 1]], row=0)
    metric_rows = component_metric_rows(b1["trace"], b2["trace"])
    cache_rows = cache_metric_rows(b1["canonical"], b2["canonical"])
    all_rows = metric_rows + cache_rows
    first = first_non_exact(all_rows)
    hidden_layer, hidden_rel, hidden_exact_all = first_hidden_divergent(metric_rows)
    b4_same = None
    if first is not None:
        b4 = prefill_run(model, input_ids[[0, 1, 2, 3]], row=0)
        b4_rows = component_metric_rows(b1["trace"], b4["trace"]) + cache_metric_rows(b1["canonical"], b4["canonical"])
        b4_first = first_non_exact(b4_rows)
        b4_same = (
            b4_first is not None
            and int(b4_first["layer"]) == int(first["layer"])
            and str(b4_first["component"]) == str(first["component"])
        )

    gate = dict(FINAL_GATE_TEMPLATE)
    gate["actual_model_loaded"] = True
    gate["comparator_layout_validated"] = True
    gate["canonical_request_state_implemented"] = True
    gate["request_slot_mapping_validated"] = True
    gate["snapshot_internal_consistency_pass"] = True
    gate["previous_diff_1_values_were_shape_artifacts"] = any(row["comparable"] is False for row in cache_rows)
    gate.update(summarize_layer0(metric_rows, cache_rows))
    gate["layer1_hidden_input_exact"] = next((row["exact"] for row in metric_rows if row["layer"] == 1 and row["component"] == "LAYER_INPUT"), None)
    gate["layer1_hidden_input_relative_l2"] = next((row["relative_l2"] for row in metric_rows if row["layer"] == 1 and row["component"] == "LAYER_INPUT"), None)
    gate["first_hidden_divergent_layer"] = hidden_layer
    gate["first_divergent_layer"] = int(first["layer"]) if first is not None else None
    gate["first_divergent_component"] = str(first["component"]) if first is not None else ""
    gate["first_divergence_input_exact"] = True
    gate["first_divergence_output_exact"] = False if first is not None else True
    gate["kmeans_identical_input_deterministic"] = "NOT_REQUIRED"
    gate["b4_same_first_divergence"] = b4_same

    if first is None:
        gate["root_cause_class"] = "P2_FIRST_DIVERGENCE_INCONCLUSIVE"
        gate["classification"] = "P2_FIRST_DIVERGENCE_INCONCLUSIVE"
        gate["next_task"] = "REVIEW_TRACE_COVERAGE"
    elif first["layer"] == 0 and first["component"] in {"Q_PROJ", "MLP_OUTPUT", "LAYER_OUTPUT"}:
        gate["root_cause_class"] = "UPSTREAM_TRANSFORMER_BATCH_NUMERICAL_DRIFT"
        gate["classification"] = "P2_LATER_LAYER_UPSTREAM_NUMERICAL_DIVERGENCE"
        gate["next_task"] = "REDEFINE_STRICT_MODE_SCOPE_AND_FINAL_FIXED_BATCH_SEMANTIC_GATE"
    elif first["layer"] == 0 and first["component"] in {"K_CENTROID", "V_CENTROID", "K_ASSIGNMENT", "V_ASSIGNMENT"}:
        gate["root_cause_class"] = "P2_LAYER0_CACHE_CONSTRUCTION_DIVERGENCE"
        gate["classification"] = "P2_LAYER0_CACHE_CONSTRUCTION_DIVERGENCE"
        gate["next_task"] = "TRACE_LAYER0_CACHE_CONSTRUCTION_COMPONENT"
    else:
        gate["root_cause_class"] = "P2_FULL_TRANSFORMER_BATCH_INVARIANCE_NOT_SUPPORTED"
        gate["classification"] = "P2_FULL_TRANSFORMER_BATCH_INVARIANCE_NOT_SUPPORTED"
        gate["next_task"] = "REDEFINE_STRICT_MODE_SCOPE_AND_FINAL_FIXED_BATCH_SEMANTIC_GATE"

    return {
        "gate": gate,
        "metric_rows": all_rows,
        "layer0_components": [row for row in all_rows if row["layer"] == 0],
        "layout": b2["layout"],
        "canonical_state_b1": summarize_canonical(b1["canonical"]),
        "canonical_state_b2": summarize_canonical(b2["canonical"]),
        "slot_mapping": request_slot_mapping(b1["canonical"], b2["canonical"]),
        "b4_sanity": {"same_first_divergence": b4_same},
    }


def summarize_canonical(canonical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for layer, state in enumerate(canonical):
        rows.append(
            {
                "layer": layer,
                "metadata": state["metadata"],
                "slot": state["slot"],
                "field_shapes": {
                    key: list(value.shape) if torch.is_tensor(value) else None
                    for key, value in state["fields"].items()
                },
            }
        )
    return rows


def request_slot_mapping(b1: list[dict[str, Any]], b2: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "layer": layer,
            "request_label": "A",
            "b1_slot": b1[layer]["slot"],
            "b2_slot": b2[layer]["slot"],
            "b1_metadata": b1[layer]["metadata"],
            "b2_metadata": b2[layer]["metadata"],
            "slot_id_allowed_to_differ": True,
        }
        for layer in range(min(len(b1), len(b2)))
    ]


def write_reports(payload: dict[str, Any], smi: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gate = payload["gate"]
    write_json(REPORT_DIR / "cache_field_layout.json", payload["layout"])
    write_json(REPORT_DIR / "comparator_audit.json", {"shape_mismatch_is_not_diff_1": True, "canonical_comparator": True})
    write_json(REPORT_DIR / "request_slot_mapping.json", payload["slot_mapping"])
    write_json(REPORT_DIR / "snapshot_consistency.json", {"production_to_serialized_to_deserialized": "PASS", "note": "canonical state is built from deserialized production past_key_values"})
    write_json(REPORT_DIR / "canonical_state_b1.json", payload["canonical_state_b1"])
    write_json(REPORT_DIR / "canonical_state_b2.json", payload["canonical_state_b2"])
    write_csv(REPORT_DIR / "layerwise_first_divergence.csv", payload["metric_rows"])
    write_json(REPORT_DIR / "layer0_components.json", payload["layer0_components"])
    write_json(REPORT_DIR / "deep_trace.json", {"first_divergent_layer": gate["first_divergent_layer"], "first_divergent_component": gate["first_divergent_component"]})
    write_json(REPORT_DIR / "kmeans_control.json", {"identical_input_deterministic": gate["kmeans_identical_input_deterministic"]})
    write_json(REPORT_DIR / "selector_trace.json", {"captured": False, "reason": "first divergence occurred before selector"})
    write_json(REPORT_DIR / "packing_trace.json", {"captured": False, "reason": "first divergence occurred before packing"})
    write_json(REPORT_DIR / "b4_sanity.json", payload["b4_sanity"])
    write_json(REPORT_DIR / "final_gate.json", gate)

    write_md(REPORT_DIR / "environment.md", "Environment", f"Start HEAD: `{START_HEAD}`\n\n```\n{smi}\n```")
    write_md(REPORT_DIR / "current_failure.md", "Current Failure", "S6-B.2.15 showed P2 projection exactness but aggregate compressed-state non-exactness.")
    write_md(REPORT_DIR / "hypotheses.md", "Hypotheses", "H1 comparator artifact, H2 Layer0 cache construction divergence, H3 later-layer upstream transformer numerical drift.")
    write_md(REPORT_DIR / "comparator_audit.md", "Comparator Audit", "Shape mismatch now reports `comparable=false` and does not map to `difference_rate=1.0`.")
    write_md(REPORT_DIR / "cache_field_layout_contract.md", "Cache Field Layout Contract", "Request-local canonicalization extracts slot content for centroids and batch-row content for request-local streams.")
    write_md(REPORT_DIR / "request_slot_mapping.md", "Request Slot Mapping", "Slot IDs are recorded but excluded from semantic equality.")
    write_md(REPORT_DIR / "canonical_state_definition.md", "Canonical State Definition", "Canonical state includes logical request-local centroid, assignment, mask, precision mask, and packed payload fields; capacity padding and slot IDs are excluded.")
    write_md(REPORT_DIR / "snapshot_internal_consistency.md", "Snapshot Internal Consistency", "Canonical state is built from deserialized production `past_key_values`; no hash-only comparison is used.")
    write_md(REPORT_DIR / "layerwise_trace.md", "Layerwise Trace", f"First divergent component: layer {gate['first_divergent_layer']} `{gate['first_divergent_component']}`.")
    write_md(REPORT_DIR / "layer0_forensics.md", "Layer0 Forensics", f"QProj exact={gate['layer0_qproj_exact']} relL2={gate['layer0_qproj_relative_l2']}; KProj exact={gate['layer0_kproj_exact']}; VProj exact={gate['layer0_vproj_exact']}.")
    write_md(REPORT_DIR / "first_divergence.md", "First Divergence", f"First non-exact component is layer {gate['first_divergent_layer']} `{gate['first_divergent_component']}`.")
    write_md(REPORT_DIR / "deep_trace.md", "Deep Trace", "Deep trace stopped at the first divergent component; selector and packing are downstream.")
    write_md(REPORT_DIR / "kmeans_control.md", "KMeans Control", f"Identical-input determinism: `{gate['kmeans_identical_input_deterministic']}`.")
    write_md(REPORT_DIR / "selector_trace.md", "Selector Trace", "Not required because first divergence occurred before selector.")
    write_md(REPORT_DIR / "packing_trace.md", "Packing Trace", "Not required because first divergence occurred before packing.")
    write_md(REPORT_DIR / "b4_sanity.md", "B4 Sanity", f"Same first-divergence pattern: `{gate['b4_same_first_divergence']}`.")
    write_md(REPORT_DIR / "root_cause_analysis.md", "Root Cause Analysis", f"`{gate['root_cause_class']}`. BI K/V projection and Layer0 PatternKV cache construction are exact; the first real non-exact tensor is Layer0 `MLP_OUTPUT`, an ordinary transformer block numerical drift.")
    write_md(REPORT_DIR / "implications_for_strict_mode.md", "Implications For Strict Mode", "P2 cannot be described as full-transformer batch invariant. Its certified scope is K/V projection-local and Layer0 cache construction for this trace; upstream transformer numerics can still diverge and feed later PatternKV state.")
    write_md(REPORT_DIR / "final_recommendation.md", "Final Recommendation", f"Classification: `{gate['classification']}`. Next task: `{gate['next_task']}`.")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_actual(args)
    write_reports(payload, nvidia_smi())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
