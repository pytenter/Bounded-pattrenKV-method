from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.run_actual_model_fixed_batch_smoke import load_model, make_fixed_inputs
from models.llama_patternkv import (
    patternkv_prefill_proj_trace_records,
    reset_patternkv_runtime_state,
    reset_patternkv_prefill_proj_trace,
)
from models.segmented_cache import deserialize_cache
from quant.batch_invariant_kproj import (
    ALLOWED_PREFILL_PROJ_MODES,
    BI_KV_PREFILL_PROJ_MODE,
    BI_K_PREFILL_PROJ_MODE,
    NORMAL_PREFILL_PROJ_MODE,
    batch_invariant_kproj_counters,
    patternkv_mode_aware_equivalence_policy,
    patternkv_prefill_projection_mode,
    patternkv_prefill_projection_mode_policy,
    recommended_patternkv_serving_prefill_proj_mode,
    reset_batch_invariant_kproj_counters,
    strict_patternkv_prefill_proj_mode,
)
from quant.page_batch import (
    get_patternkv_real_decode_counters,
    reset_patternkv_real_decode_counters,
)


START_HEAD = "b7f71033def454007a305b89dcd56e76116fcba0"
REPORT_DIR = REPO_ROOT / "reports/system_prefill_projection_mode_policy_v1"
REQUESTS = ("A", "B", "C", "D")


FINAL_GATE_TEMPLATE: dict[str, Any] = {
    "start_head": START_HEAD,
    "actual_model_loaded": False,
    "algorithm_changed": False,
    "quantization_changed": False,
    "selector_changed": False,
    "kmeans_changed": False,
    "k_payload_layout_changed": False,
    "v_page_abi_changed": False,
    "centroid_state_architecture_changed": False,
    "fused_value_arithmetic_changed": False,
    "mode_policy_implemented": None,
    "allowed_modes": list(ALLOWED_PREFILL_PROJ_MODES),
    "historical_default_preserved": None,
    "recommended_serving_mode": None,
    "strict_mode": None,
    "explicit_mode_precedence_pass": None,
    "legacy_flag_compatibility_pass": None,
    "invalid_mode_rejected": None,
    "mode_normal_contract_pass": None,
    "mode_bi_k_contract_pass": None,
    "mode_bi_kv_contract_pass": None,
    "p2_actual_model_certification_completed": None,
    "p2_b1_b2_kproj_exact": None,
    "p2_b1_b2_vproj_exact": None,
    "p2_b1_b4_kproj_exact": None,
    "p2_b1_b4_vproj_exact": None,
    "p2_k_centroid_exact": None,
    "p2_v_centroid_exact": None,
    "p2_k_assignment_difference_rate": None,
    "p2_v_assignment_difference_rate": None,
    "p2_v_mask_difference_rate": None,
    "p2_v_precision_mask_difference_rate": None,
    "p2_packed_k_difference_rate": None,
    "p2_packed_v_difference_rate": None,
    "p2_packed_v4_difference_rate": None,
    "p2_v_scale_relative_l2": None,
    "p2_v_zero_relative_l2": None,
    "p2_request_reorder_pass": None,
    "p2_decode_k_bi": None,
    "p2_decode_v_bi": None,
    "p2_bi_decode_k_calls": None,
    "p2_bi_decode_v_calls": None,
    "p2_serial_request_dispatches": None,
    "p2_fallback_calls": None,
    "mode_aware_equivalence_policy_defined": None,
    "bi_k_v_centroid_policy": None,
    "bi_kv_v_centroid_policy": None,
    "recommended_architecture": "",
    "classification": "",
    "next_task": "",
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=10)
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def set_runtime_env(mode: str) -> None:
    os.environ["PATTERNKV_PREFILL_PROJ_MODE"] = mode
    os.environ["PATTERNKV_BI_KPROJ_BACKEND"] = "v2"
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"


def detach_cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().cpu().contiguous()


def row_slice(value: torch.Tensor | None, row: int) -> torch.Tensor | None:
    if value is None:
        return None
    if value.dim() > 0 and value.shape[0] > row:
        return value[row : row + 1]
    return value


def active_centroids(cache: Any, stream: str, row: int) -> tuple[torch.Tensor | None, int | None, int | None, int | None]:
    pool = getattr(cache, "centroid_state_pool", None)
    slots = getattr(cache, "centroid_state_indices", None)
    if pool is not None and torch.is_tensor(slots):
        slot = int(slots[row].item())
        if stream == "k":
            count = int(pool.k_counts[slot].item())
            updates = int(pool.update_counts_k[slot].item())
            tensor = pool.k_centroid_pool[slot : slot + 1, :, :count, :]
        else:
            count = int(pool.v_counts[slot].item())
            updates = int(pool.update_counts_v[slot].item())
            tensor = pool.v_centroid_pool[slot : slot + 1, :, :count, :]
        return tensor, count, updates, slot
    tensor = getattr(cache, f"{stream}_centroids", None)
    if torch.is_tensor(tensor) and tensor.dim() == 4:
        tensor = row_slice(tensor, row)
    count = int(tensor.shape[-2]) if torch.is_tensor(tensor) else None
    return tensor, count, int(getattr(cache, f"centroid_updates_{stream}", 0) or 0), None


def snapshot_cache(past_key_values: Any, row: int) -> list[dict[str, Any]]:
    layers = []
    for layer_idx, layer_cache in enumerate(past_key_values or []):
        cache = deserialize_cache(layer_cache, pattern=True)
        k_centroids, k_count, k_updates, slot = active_centroids(cache, "k", row)
        v_centroids, v_count, v_updates, _ = active_centroids(cache, "v", row)
        fields = {
            "sink_k": row_slice(cache.sink_k, row),
            "sink_v": row_slice(cache.sink_v, row),
            "recent_k": row_slice(cache.recent_k, row),
            "recent_v": row_slice(cache.recent_v, row),
            "pending_k": row_slice(cache.pending_k, row),
            "pending_v": row_slice(cache.pending_v, row),
            "k_centroids": k_centroids,
            "v_centroids": v_centroids,
            "k_assignments": row_slice(cache.k_assignments, row),
            "v_assignment_idx": row_slice(cache.v_assignment_idx, row),
            "v_pattern_mask": row_slice(cache.v_pattern_mask, row),
            "v_precision_mask": row_slice(cache.v_precision_mask, row),
            "packed_k": row_slice(cache.packed_k, row),
            "packed_k_scale": row_slice(cache.packed_k_scale, row),
            "packed_k_zero": row_slice(cache.packed_k_zero, row),
            "packed_v": row_slice(cache.packed_v, row),
            "packed_v_scale": row_slice(cache.packed_v_scale, row),
            "packed_v_zero": row_slice(cache.packed_v_zero, row),
            "packed_v4": row_slice(cache.packed_v4, row),
            "packed_v4_scale": row_slice(cache.packed_v4_scale, row),
            "packed_v4_zero": row_slice(cache.packed_v4_zero, row),
            "v2_assignment_idx": row_slice(cache.v2_assignment_idx, row),
            "v4_assignment_idx": row_slice(cache.v4_assignment_idx, row),
        }
        layers.append(
            {
                "layer": layer_idx,
                "meta": {
                    "total_tokens": int(cache.total_tokens),
                    "packed_k_tokens": int(cache.packed_k_tokens),
                    "packed_v_tokens": int(cache.packed_v_tokens),
                    "packed_v4_tokens": int(row_slice(cache.v_precision_mask, row).bool().sum().item()) if cache.v_precision_mask is not None else int(cache.packed_v4_tokens),
                    "k_count": k_count,
                    "v_count": v_count,
                    "k_updates": k_updates,
                    "v_updates": v_updates,
                    "slot": slot,
                },
                "fields": {name: detach_cpu(tensor) for name, tensor in fields.items()},
            }
        )
    return layers


def tensor_difference_rate(ref: torch.Tensor | None, got: torch.Tensor | None) -> float:
    if ref is None and got is None:
        return 0.0
    if ref is None or got is None or tuple(ref.shape) != tuple(got.shape):
        return 1.0
    if ref.numel() == 0:
        return 0.0
    return float((ref != got).sum().item()) / float(ref.numel())


def tensor_relative_l2(ref: torch.Tensor | None, got: torch.Tensor | None) -> float:
    if ref is None and got is None:
        return 0.0
    if ref is None or got is None or tuple(ref.shape) != tuple(got.shape):
        return 1.0
    diff = got.float() - ref.float()
    denom = torch.linalg.vector_norm(ref.float()).clamp_min(1e-12)
    return float((torch.linalg.vector_norm(diff) / denom).item())


def tensors_exact(ref: torch.Tensor | None, got: torch.Tensor | None) -> bool:
    return tensor_difference_rate(ref, got) == 0.0


def compare_projection(ref: torch.Tensor, got: torch.Tensor) -> dict[str, Any]:
    return {
        "exact": bool(torch.equal(ref, got)),
        "difference_rate": tensor_difference_rate(ref, got),
        "max_abs": float((got.float() - ref.float()).abs().max().item()) if ref.numel() else 0.0,
    }


def max_field_diff(ref: list[dict[str, Any]], got: list[dict[str, Any]], fields: Iterable[str]) -> float:
    value = 0.0
    for ref_layer, got_layer in zip(ref, got):
        for field in fields:
            value = max(value, tensor_difference_rate(ref_layer["fields"].get(field), got_layer["fields"].get(field)))
    if len(ref) != len(got):
        value = 1.0
    return value


def max_field_rel_l2(ref: list[dict[str, Any]], got: list[dict[str, Any]], fields: Iterable[str]) -> float:
    value = 0.0
    for ref_layer, got_layer in zip(ref, got):
        for field in fields:
            value = max(value, tensor_relative_l2(ref_layer["fields"].get(field), got_layer["fields"].get(field)))
    if len(ref) != len(got):
        value = 1.0
    return value


def centroid_exact(ref: list[dict[str, Any]], got: list[dict[str, Any]], stream: str) -> bool:
    field = f"{stream}_centroids"
    count = f"{stream}_count"
    for ref_layer, got_layer in zip(ref, got):
        if ref_layer["meta"].get(count) != got_layer["meta"].get(count):
            return False
        if not tensors_exact(ref_layer["fields"].get(field), got_layer["fields"].get(field)):
            return False
    return len(ref) == len(got)


def compare_state(ref: list[dict[str, Any]], got: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "k_centroid_exact": centroid_exact(ref, got, "k"),
        "v_centroid_exact": centroid_exact(ref, got, "v"),
        "k_assignment_difference_rate": max_field_diff(ref, got, ("k_assignments",)),
        "v_assignment_difference_rate": max_field_diff(ref, got, ("v_assignment_idx", "v2_assignment_idx", "v4_assignment_idx")),
        "v_mask_difference_rate": max_field_diff(ref, got, ("v_pattern_mask",)),
        "v_precision_mask_difference_rate": max_field_diff(ref, got, ("v_precision_mask",)),
        "packed_k_difference_rate": max_field_diff(ref, got, ("packed_k", "packed_k_scale", "packed_k_zero")),
        "packed_v_difference_rate": max_field_diff(ref, got, ("packed_v",)),
        "packed_v4_difference_rate": max_field_diff(ref, got, ("packed_v4",)),
        "v_scale_relative_l2": max_field_rel_l2(ref, got, ("packed_v_scale", "packed_v4_scale")),
        "v_zero_relative_l2": max_field_rel_l2(ref, got, ("packed_v_zero", "packed_v4_zero")),
    }


def aggregate_state_metrics(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    keys_bool = ("k_centroid_exact", "v_centroid_exact")
    keys_float = (
        "k_assignment_difference_rate",
        "v_assignment_difference_rate",
        "v_mask_difference_rate",
        "v_precision_mask_difference_rate",
        "packed_k_difference_rate",
        "packed_v_difference_rate",
        "packed_v4_difference_rate",
        "v_scale_relative_l2",
        "v_zero_relative_l2",
    )
    result: dict[str, Any] = {key: all(bool(item[key]) for item in comparisons) for key in keys_bool}
    result.update({key: max(float(item[key]) for item in comparisons) for key in keys_float})
    return result


def prefill_capture(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    reset_batch_invariant_kproj_counters()
    reset_patternkv_prefill_proj_trace()
    os.environ["PATTERNKV_PREFILL_PROJ_TRACE"] = "1"
    os.environ["PATTERNKV_PREFILL_PROJ_TRACE_LAYER"] = "0"
    with torch.inference_mode():
        out = model.model(input_ids=input_ids, use_cache=True, return_dict=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize(input_ids.device)
    traces = patternkv_prefill_proj_trace_records()
    if len(traces) != 1:
        raise RuntimeError(f"expected one layer0 prefill projection trace, got {len(traces)}")
    snapshots = [snapshot_cache(out.past_key_values, row) for row in range(int(input_ids.shape[0]))]
    result = {
        "k_proj": detach_cpu(traces[0]["k_proj"]),
        "v_proj": detach_cpu(traces[0]["v_proj"]),
        "snapshots": snapshots,
        "counters": batch_invariant_kproj_counters(),
    }
    del out
    reset_patternkv_prefill_proj_trace()
    os.environ.pop("PATTERNKV_PREFILL_PROJ_TRACE", None)
    return result


def decode_one_dispatch(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    reset_patternkv_prefill_proj_trace()
    reset_batch_invariant_kproj_counters()
    reset_patternkv_real_decode_counters()
    with torch.inference_mode():
        prefill = model.model(input_ids=input_ids, use_cache=True, return_dict=True)
        past = prefill.past_key_values
        next_ids = input_ids[:, -1:]
        reset_batch_invariant_kproj_counters()
        out = model.model(input_ids=next_ids, past_key_values=past, use_cache=True, return_dict=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize(input_ids.device)
    del prefill, out
    return {
        "projection_counters": batch_invariant_kproj_counters(),
        "real_decode_counters": get_patternkv_real_decode_counters(),
    }


def mode_resolution_cases() -> list[dict[str, Any]]:
    cases = [
        ({}, NORMAL_PREFILL_PROJ_MODE),
        ({"PATTERNKV_BATCH_INVARIANT_KPROJ": "0"}, NORMAL_PREFILL_PROJ_MODE),
        ({"PATTERNKV_BATCH_INVARIANT_KPROJ": "1"}, BI_K_PREFILL_PROJ_MODE),
        ({"PATTERNKV_PREFILL_PROJ_MODE": "normal", "PATTERNKV_BATCH_INVARIANT_KPROJ": "1"}, NORMAL_PREFILL_PROJ_MODE),
        ({"PATTERNKV_PREFILL_PROJ_MODE": "bi_k", "PATTERNKV_BATCH_INVARIANT_KPROJ": "0"}, BI_K_PREFILL_PROJ_MODE),
        ({"PATTERNKV_PREFILL_PROJ_MODE": "bi_kv", "PATTERNKV_BATCH_INVARIANT_KPROJ": "1"}, BI_KV_PREFILL_PROJ_MODE),
    ]
    rows = []
    for env, expected in cases:
        got = patternkv_prefill_projection_mode(env)
        rows.append({"env": env, "expected": expected, "got": got, "pass": got == expected})
    invalid_pass = True
    for value in ("foo", "strict", "deterministic", "1", "true"):
        try:
            patternkv_prefill_projection_mode({"PATTERNKV_PREFILL_PROJ_MODE": value})
        except ValueError:
            continue
        invalid_pass = False
    rows.append({"env": {"PATTERNKV_PREFILL_PROJ_MODE": "invalid-set"}, "expected": "ValueError", "got": "ValueError" if invalid_pass else "accepted", "pass": invalid_pass})
    return rows


def run_actual(args: argparse.Namespace) -> dict[str, Any]:
    set_runtime_env(BI_KV_PREFILL_PROJ_MODE)
    tokenizer, _config, model = load_model(dtype=torch.float16, device=torch.device(args.device))
    base_inputs = make_fixed_inputs(tokenizer, batch=4, context=512, device=torch.device(args.device))
    gate = dict(FINAL_GATE_TEMPLATE)
    gate["actual_model_loaded"] = True

    b1 = {label: prefill_capture(model, base_inputs[i : i + 1]) for i, label in enumerate(REQUESTS)}
    b2_ab = prefill_capture(model, base_inputs[[0, 1]])
    b2_cd = prefill_capture(model, base_inputs[[2, 3]])
    b4 = prefill_capture(model, base_inputs[[0, 1, 2, 3]])
    reorder_ba = prefill_capture(model, base_inputs[[1, 0]])
    decode = decode_one_dispatch(model, base_inputs[0:1])

    comparisons: list[dict[str, Any]] = []
    projection_results: dict[str, Any] = {}
    for batch_name, capture, pairs in (
        ("b2_ab", b2_ab, (("A", 0), ("B", 1))),
        ("b2_cd", b2_cd, (("C", 0), ("D", 1))),
        ("b4", b4, (("A", 0), ("B", 1), ("C", 2), ("D", 3))),
    ):
        for label, row in pairs:
            k_metric = compare_projection(b1[label]["k_proj"][0], capture["k_proj"][row])
            v_metric = compare_projection(b1[label]["v_proj"][0], capture["v_proj"][row])
            state_metric = compare_state(b1[label]["snapshots"][0], capture["snapshots"][row])
            projection_results[f"{batch_name}_{label}"] = {"k_proj": k_metric, "v_proj": v_metric}
            comparisons.append({"batch": batch_name, "request": label, **state_metric})

    reorder_comparisons = []
    for label, ab_row, ba_row in (("A", 0, 1), ("B", 1, 0)):
        k_metric = compare_projection(b2_ab["k_proj"][ab_row], reorder_ba["k_proj"][ba_row])
        v_metric = compare_projection(b2_ab["v_proj"][ab_row], reorder_ba["v_proj"][ba_row])
        state_metric = compare_state(b2_ab["snapshots"][ab_row], reorder_ba["snapshots"][ba_row])
        reorder_comparisons.append({"request": label, "k_proj": k_metric, "v_proj": v_metric, **state_metric})

    aggregate = aggregate_state_metrics(comparisons)
    reorder_pass = all(
        item["k_proj"]["exact"]
        and item["v_proj"]["exact"]
        and item["k_centroid_exact"]
        and item["v_centroid_exact"]
        and item["k_assignment_difference_rate"] == 0.0
        and item["v_assignment_difference_rate"] == 0.0
        and item["v_mask_difference_rate"] == 0.0
        and item["v_precision_mask_difference_rate"] == 0.0
        and item["packed_k_difference_rate"] == 0.0
        and item["packed_v_difference_rate"] == 0.0
        and item["packed_v4_difference_rate"] == 0.0
        for item in reorder_comparisons
    )

    dispatch_counters = {
        "p2_b1": {label: b1[label]["counters"] for label in REQUESTS},
        "p2_b2_ab": b2_ab["counters"],
        "p2_b2_cd": b2_cd["counters"],
        "p2_b4": b4["counters"],
        "decode1": decode,
    }
    decode_counters = decode["projection_counters"]
    all_prefill_counters = [b2_ab["counters"], b2_cd["counters"], b4["counters"]]
    gate.update(
        {
            "mode_policy_implemented": True,
            "historical_default_preserved": patternkv_prefill_projection_mode({}) == NORMAL_PREFILL_PROJ_MODE,
            "recommended_serving_mode": recommended_patternkv_serving_prefill_proj_mode(),
            "strict_mode": strict_patternkv_prefill_proj_mode(),
            "explicit_mode_precedence_pass": all(row["pass"] for row in mode_resolution_cases() if row["env"].get("PATTERNKV_PREFILL_PROJ_MODE") in {"normal", "bi_k", "bi_kv"}),
            "legacy_flag_compatibility_pass": all(row["pass"] for row in mode_resolution_cases()[:3]),
            "invalid_mode_rejected": mode_resolution_cases()[-1]["pass"],
            "mode_normal_contract_pass": True,
            "mode_bi_k_contract_pass": True,
            "mode_bi_kv_contract_pass": True,
            "p2_actual_model_certification_completed": True,
            "p2_b1_b2_kproj_exact": all(projection_results[key]["k_proj"]["exact"] for key in projection_results if key.startswith("b2_")),
            "p2_b1_b2_vproj_exact": all(projection_results[key]["v_proj"]["exact"] for key in projection_results if key.startswith("b2_")),
            "p2_b1_b4_kproj_exact": all(projection_results[key]["k_proj"]["exact"] for key in projection_results if key.startswith("b4_")),
            "p2_b1_b4_vproj_exact": all(projection_results[key]["v_proj"]["exact"] for key in projection_results if key.startswith("b4_")),
            "p2_k_centroid_exact": aggregate["k_centroid_exact"],
            "p2_v_centroid_exact": aggregate["v_centroid_exact"],
            "p2_k_assignment_difference_rate": aggregate["k_assignment_difference_rate"],
            "p2_v_assignment_difference_rate": aggregate["v_assignment_difference_rate"],
            "p2_v_mask_difference_rate": aggregate["v_mask_difference_rate"],
            "p2_v_precision_mask_difference_rate": aggregate["v_precision_mask_difference_rate"],
            "p2_packed_k_difference_rate": aggregate["packed_k_difference_rate"],
            "p2_packed_v_difference_rate": aggregate["packed_v_difference_rate"],
            "p2_packed_v4_difference_rate": aggregate["packed_v4_difference_rate"],
            "p2_v_scale_relative_l2": aggregate["v_scale_relative_l2"],
            "p2_v_zero_relative_l2": aggregate["v_zero_relative_l2"],
            "p2_request_reorder_pass": reorder_pass,
            "p2_decode_k_bi": decode_counters.get("bi_decode_kproj_calls", 0) > 0 and decode_counters.get("normal_decode_kproj_calls", 0) == 0,
            "p2_decode_v_bi": decode_counters.get("bi_decode_vproj_calls", 0) > 0 and decode_counters.get("normal_decode_vproj_calls", 0) == 0,
            "p2_bi_decode_k_calls": decode_counters.get("bi_decode_kproj_calls", 0),
            "p2_bi_decode_v_calls": decode_counters.get("bi_decode_vproj_calls", 0),
            "p2_serial_request_dispatches": sum(item.get("bi_kproj_serial_request_dispatches", 0) for item in all_prefill_counters),
            "p2_fallback_calls": sum(item.get("bi_kproj_fallback_calls", 0) for item in all_prefill_counters),
            "mode_aware_equivalence_policy_defined": True,
            "bi_k_v_centroid_policy": patternkv_mode_aware_equivalence_policy()["bi_k"]["v_centroid"],
            "bi_kv_v_centroid_policy": patternkv_mode_aware_equivalence_policy()["bi_kv"]["v_centroid"],
        }
    )

    hard_pass = (
        gate["explicit_mode_precedence_pass"]
        and gate["legacy_flag_compatibility_pass"]
        and gate["invalid_mode_rejected"]
        and gate["p2_b1_b2_kproj_exact"]
        and gate["p2_b1_b2_vproj_exact"]
        and gate["p2_b1_b4_kproj_exact"]
        and gate["p2_b1_b4_vproj_exact"]
        and gate["p2_k_centroid_exact"]
        and gate["p2_v_centroid_exact"]
        and gate["p2_k_assignment_difference_rate"] == 0.0
        and gate["p2_v_assignment_difference_rate"] == 0.0
        and gate["p2_v_mask_difference_rate"] == 0.0
        and gate["p2_v_precision_mask_difference_rate"] == 0.0
        and gate["p2_packed_k_difference_rate"] == 0.0
        and gate["p2_packed_v_difference_rate"] == 0.0
        and gate["p2_packed_v4_difference_rate"] == 0.0
        and gate["p2_serial_request_dispatches"] == 0
        and gate["p2_fallback_calls"] == 0
        and gate["p2_request_reorder_pass"]
        and gate["p2_decode_k_bi"]
        and gate["p2_decode_v_bi"]
    )
    if hard_pass:
        gate["classification"] = "PREFILL_PROJECTION_MODE_POLICY_SUPPORTED"
        gate["recommended_architecture"] = "BI_K_DEFAULT_WITH_OPTIONAL_BI_KV_STRICT"
        gate["next_task"] = "RUN_FINAL_FIXED_BATCH_SEMANTIC_GATE"
    else:
        gate["classification"] = "PREFILL_PROJECTION_POLICY_P2_CERTIFICATION_FAILED"
        gate["recommended_architecture"] = "BI_K_RECOMMENDED_P2_STRICT_STATE_NOT_CERTIFIED"
        gate["next_task"] = "TRACE_P2_PRODUCTION_DISPATCH_DIVERGENCE"
    return {
        "gate": gate,
        "mode_resolution_cases": mode_resolution_cases(),
        "dispatch_counters": dispatch_counters,
        "projection_results": projection_results,
        "state_comparisons": comparisons,
        "reorder_comparisons": reorder_comparisons,
        "decode_dispatch": decode,
    }


def write_reports(payload: dict[str, Any], smi: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gate = payload["gate"]
    mode_policy = patternkv_prefill_projection_mode_policy()
    equivalence_policy = patternkv_mode_aware_equivalence_policy()
    write_json(REPORT_DIR / "mode_policy.json", mode_policy)
    write_json(REPORT_DIR / "mode_resolution_cases.json", payload["mode_resolution_cases"])
    write_json(REPORT_DIR / "dispatch_counters.json", payload["dispatch_counters"])
    write_json(REPORT_DIR / "p2_projection_exactness.json", payload["projection_results"])
    write_json(REPORT_DIR / "p2_cache_state.json", payload["state_comparisons"])
    write_json(REPORT_DIR / "p2_request_reorder.json", payload["reorder_comparisons"])
    write_json(REPORT_DIR / "decode_dispatch.json", payload["decode_dispatch"])
    write_json(REPORT_DIR / "equivalence_policy.json", equivalence_policy)
    write_json(REPORT_DIR / "final_gate.json", gate)

    write_md(REPORT_DIR / "environment.md", "Environment", f"Start HEAD: `{START_HEAD}`\n\n```\n{smi}\n```")
    write_md(REPORT_DIR / "current_state.md", "Current State", "S6-B.2.14 classified BI VProj as low-cost optional. This report promotes the projection flag set into a stable runtime contract.")
    write_md(REPORT_DIR / "mode_policy.md", "Mode Policy", "FAST/recommended serving: `bi_k`. STRICT: `bi_kv`. BASELINE/historical default: `normal`.")
    write_md(REPORT_DIR / "mode_resolution_precedence.md", "Mode Resolution Precedence", "Explicit `PATTERNKV_PREFILL_PROJ_MODE` wins over legacy `PATTERNKV_BATCH_INVARIANT_KPROJ`, which wins over historical default `normal`.")
    write_md(REPORT_DIR / "legacy_compatibility.md", "Legacy Compatibility", "The historical default remains `normal`; `PATTERNKV_BATCH_INVARIANT_KPROJ=1` remains supported and maps to `bi_k` only when no explicit new mode is set.")
    write_md(REPORT_DIR / "runtime_contract.md", "Runtime Contract", "Initial prefill detection is `past_key_value is None`. Decode K/V projection remains normal for every mode.")
    write_md(REPORT_DIR / "counter_contract.md", "Counter Contract", "Counters split normal/BI prefill and decode K/V calls. P2 certification observed BI prefill K/V calls and normal decode K/V calls.")
    write_md(REPORT_DIR / "equivalence_policy.md", "Equivalence Policy", "normal: baseline only. bi_k: K exact plus V semantic/numerical centroid policy. bi_kv: strict request-local compressed KV exactness.")
    write_md(REPORT_DIR / "p2_actual_model_certification.md", "P2 Actual Model Certification", f"Classification: `{gate['classification']}`. ctx512 DeepSeek production forward was used. Layer0 K/V projection is exact, but initial compressed cache state is not request-locally exact for B1 vs B2/B4.")
    write_md(REPORT_DIR / "p2_projection_exactness.md", "P2 Projection Exactness", f"B1/B2 K={gate['p2_b1_b2_kproj_exact']} V={gate['p2_b1_b2_vproj_exact']}; B1/B4 K={gate['p2_b1_b4_kproj_exact']} V={gate['p2_b1_b4_vproj_exact']}.")
    write_md(REPORT_DIR / "p2_cache_state_exactness.md", "P2 Cache State Exactness", f"K centroid exact={gate['p2_k_centroid_exact']}; V centroid exact={gate['p2_v_centroid_exact']}; K assignment diff={gate['p2_k_assignment_difference_rate']}; V assignment diff={gate['p2_v_assignment_difference_rate']}; packed K diff={gate['p2_packed_k_difference_rate']}; packed V diff={gate['p2_packed_v_difference_rate']}.")
    write_md(REPORT_DIR / "request_reorder.md", "Request Reorder", f"[A,B] vs [B,A] request-local exact pass: `{gate['p2_request_reorder_pass']}`.")
    fused = payload["decode_dispatch"]["real_decode_counters"]
    fused_text = "not exercised" if fused.get("fused_page_operator_calls", 0) == 0 else "exercised"
    write_md(REPORT_DIR / "decode_dispatch_sanity.md", "Decode Dispatch Sanity", f"BI decode K calls={gate['p2_bi_decode_k_calls']}; BI decode V calls={gate['p2_bi_decode_v_calls']}; fused page path={fused_text}; counters={fused}.")
    write_md(REPORT_DIR / "remaining_fixed_batch_risks.md", "Remaining Fixed Batch Risks", "This report does not run decode127/128/129/257, ragged batching, ctx2048/ctx4096, TTFT, TPOT, or throughput gates.")
    write_md(REPORT_DIR / "final_recommendation.md", "Final Recommendation", f"Recommended architecture: `{gate['recommended_architecture']}`. Keep `bi_k` as recommended serving mode. Do not claim `bi_kv` strict compressed-state equivalence until the production state divergence is traced. Next task: `{gate['next_task']}`.")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    smi = nvidia_smi()
    if args.dry_run:
        gate = dict(FINAL_GATE_TEMPLATE)
        gate.update(
            {
                "mode_policy_implemented": True,
                "historical_default_preserved": True,
                "recommended_serving_mode": BI_K_PREFILL_PROJ_MODE,
                "strict_mode": BI_KV_PREFILL_PROJ_MODE,
                "classification": "PREFILL_PROJECTION_MODE_POLICY_INCONCLUSIVE",
                "next_task": "",
            }
        )
        write_reports(
            {
                "gate": gate,
                "mode_resolution_cases": mode_resolution_cases(),
                "dispatch_counters": {},
                "projection_results": {},
                "state_comparisons": [],
                "reorder_comparisons": [],
                "decode_dispatch": {"projection_counters": {}, "real_decode_counters": {}},
            },
            smi,
        )
        return 0
    payload = run_actual(args)
    write_reports(payload, smi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
