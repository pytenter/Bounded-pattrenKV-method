from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.run_actual_model_fixed_batch_smoke import load_model, make_fixed_inputs, tensor_metrics
from bench.run_ragged_decode1_semantic_gate import compare_logits, nvidia_smi
from models.llama_patternkv import reset_patternkv_runtime_state
from models.segmented_cache import (
    assemble_ragged_patternkv_cache,
    deserialize_cache,
    get_packed_k_tokens_per_request,
    get_ragged_k_counters,
    get_total_tokens_per_request,
    k_segment_valid_lengths,
    reset_ragged_k_counters,
    serialize_cache,
)
from quant.page_batch import get_patternkv_real_decode_counters, reset_patternkv_real_decode_counters


START_HEAD = "70996333219bdfd6ab9f5331a4e8c1c4b98dd801"
REPORT_DIR = REPO_ROOT / "reports/system_ragged_multistep_failure_forensic_v1"
STEPS = 16
TRACE_REQUEST = "A"
TRACE_STEP = 5


def set_env() -> None:
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    os.environ["PATTERNKV_CENTROID_MAX_SLOTS"] = "8"
    os.environ.pop("PATTERNKV_BI_MLP_ORACLE", None)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def git_output(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def tensor_hash(value: torch.Tensor | None) -> str | None:
    if value is None:
        return None
    cpu = value.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tuple(cpu.shape)).encode())
    h.update(str(cpu.dtype).encode())
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def tensor_metric_summary(got: torch.Tensor | None, ref: torch.Tensor | None) -> dict[str, Any]:
    if got is None and ref is None:
        return {"exact": True, "relative_l2": 0.0, "max_abs": 0.0}
    if got is None or ref is None:
        return {"exact": False, "relative_l2": None, "max_abs": None, "shape_mismatch": True}
    if tuple(got.shape) != tuple(ref.shape):
        return {"exact": False, "relative_l2": None, "max_abs": None, "got_shape": list(got.shape), "ref_shape": list(ref.shape)}
    metrics = {k: v for k, v in tensor_metrics(got, ref).items() if k not in {"got", "ref"}}
    metrics["exact"] = bool(torch.equal(got, ref))
    return metrics


def slice_or_none(value: torch.Tensor | None, row: int | None, dim: int | None, length: int | None) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    out = value
    if row is not None and out.dim() > 0 and int(out.shape[0]) > row:
        out = out[row : row + 1]
    if dim is not None and length is not None:
        dim = dim if dim >= 0 else out.dim() + dim
        length = min(int(length), int(out.shape[dim]))
        out = out.narrow(dim, 0, length)
    return out.detach().contiguous()


def centroid_tensor(cache: Any, row: int, name: str) -> torch.Tensor | None:
    pool = getattr(cache, "centroid_state_pool", None)
    indices = getattr(cache, "centroid_state_indices", None)
    if pool is not None and torch.is_tensor(indices) and int(indices.numel()) > row:
        slot = int(indices[row].item())
        return getattr(pool, f"{name}_centroid_pool")[slot : slot + 1].detach().contiguous()
    value = getattr(cache, f"{name}_centroids", None)
    if not torch.is_tensor(value):
        return None
    if value.dim() == 4 and int(value.shape[0]) > row:
        return value[row : row + 1].detach().contiguous()
    return value.unsqueeze(0).detach().contiguous()


def page_state(cache: Any, row: int) -> dict[str, Any]:
    pools = getattr(cache, "operator_ready_page_pools", None)
    if pools is None:
        return {"present": False}
    meta = pools.metadata
    pages = int(meta.num_pages[row].item())
    page_ids = meta.metadata_page_table[row, :pages].detach().contiguous() if pages else None
    valid = meta.valid_tokens[page_ids.long()].detach().contiguous() if pages else None
    return {
        "present": True,
        "num_pages": pages,
        "seq_len": int(meta.seq_lens[row].item()),
        "request_indptr": int(meta.request_indptr[row].item()),
        "page_ids_hash": tensor_hash(page_ids),
        "valid_tokens_hash": tensor_hash(valid),
    }


def cache_component_tensors(cache: Any, row: int) -> dict[str, torch.Tensor | None]:
    lengths = k_segment_valid_lengths(cache)
    total = int(get_total_tokens_per_request(cache)[row].item())
    packed_k = int(get_packed_k_tokens_per_request(cache)[row].item())
    packed_v_tensor = getattr(cache, "request_packed_v_tokens", None)
    packed_v4_tensor = getattr(cache, "request_packed_v4_tokens", None)
    packed_v = int(packed_v_tensor[row].item()) if torch.is_tensor(packed_v_tensor) else int(cache.packed_v_tokens)
    packed_v4 = int(packed_v4_tensor[row].item()) if torch.is_tensor(packed_v4_tensor) else int(getattr(cache, "packed_v4_tokens", 0) or 0)
    packed_v2 = max(packed_v - packed_v4, 0)
    packed_k_payload = ceil_div(packed_k, 32 // int(cache.k_bits))
    packed_k_groups = ceil_div(packed_k, int(cache.group_size))
    return {
        "sink_k": slice_or_none(cache.sink_k, row, 2, int(lengths["sink"][row].item())),
        "sink_v": slice_or_none(cache.sink_v, row, 2, int(lengths["sink"][row].item())),
        "packed_k_payload": slice_or_none(cache.packed_k, row, 3, packed_k_payload),
        "packed_k_scale": slice_or_none(cache.packed_k_scale, row, 3, packed_k_groups),
        "packed_k_zero": slice_or_none(cache.packed_k_zero, row, 3, packed_k_groups),
        "k_assignments": slice_or_none(cache.k_assignments, row, 2, packed_k),
        "pending_k": slice_or_none(cache.pending_k, row, 2, int(lengths["pending"][row].item())),
        "pending_v": slice_or_none(cache.pending_v, row, 2, int(lengths["pending"][row].item())),
        "recent_k": slice_or_none(cache.recent_k, row, 2, int(lengths["recent"][row].item())),
        "recent_v": slice_or_none(cache.recent_v, row, 2, int(lengths["recent"][row].item())),
        "packed_v2_payload": slice_or_none(cache.packed_v, row, 2, packed_v2),
        "packed_v2_scale": slice_or_none(cache.packed_v_scale, row, 2, packed_v2),
        "packed_v2_zero": slice_or_none(cache.packed_v_zero, row, 2, packed_v2),
        "packed_v4_payload": slice_or_none(cache.packed_v4, row, 2, packed_v4),
        "packed_v4_scale": slice_or_none(cache.packed_v4_scale, row, 2, packed_v4),
        "packed_v4_zero": slice_or_none(cache.packed_v4_zero, row, 2, packed_v4),
        "v_assignment_idx": slice_or_none(cache.v_assignment_idx, row, 2, packed_v),
        "v_pattern_mask": slice_or_none(cache.v_pattern_mask, row, 2, packed_v),
        "v_precision_mask": slice_or_none(cache.v_precision_mask, row, 1, packed_v),
        "v2_assignment_idx": slice_or_none(cache.v2_assignment_idx, row, 2, packed_v2),
        "v2_pattern_mask": slice_or_none(cache.v2_pattern_mask, row, 2, packed_v2),
        "v4_assignment_idx": slice_or_none(cache.v4_assignment_idx, row, 2, packed_v4),
        "v4_pattern_mask": slice_or_none(cache.v4_pattern_mask, row, 2, packed_v4),
        "v_causal_importance": slice_or_none(cache.v_causal_importance, row, 1, total),
        "v_oracle_importance": slice_or_none(cache.v_oracle_importance, row, 1, total),
        "k_centroid_values": centroid_tensor(cache, row, "k"),
        "v_centroid_values": centroid_tensor(cache, row, "v"),
    }


def cache_layer_snapshot(past: Any, row: int) -> list[dict[str, Any]]:
    snapshots = []
    for layer_idx, layer in enumerate(past):
        cache = deserialize_cache(layer, pattern=True)
        lengths = k_segment_valid_lengths(cache)
        total = int(get_total_tokens_per_request(cache)[row].item())
        packed_k = int(get_packed_k_tokens_per_request(cache)[row].item())
        packed_v_tensor = getattr(cache, "request_packed_v_tokens", None)
        packed_v4_tensor = getattr(cache, "request_packed_v4_tokens", None)
        packed_v = int(packed_v_tensor[row].item()) if torch.is_tensor(packed_v_tensor) else int(cache.packed_v_tokens)
        packed_v4 = int(packed_v4_tensor[row].item()) if torch.is_tensor(packed_v4_tensor) else int(getattr(cache, "packed_v4_tokens", 0) or 0)
        tensors = cache_component_tensors(cache, row)
        pool = getattr(cache, "centroid_state_pool", None)
        indices = getattr(cache, "centroid_state_indices", None)
        slot = int(indices[row].item()) if torch.is_tensor(indices) and int(indices.numel()) > row else None
        centroid_meta = {}
        if pool is not None and slot is not None:
            centroid_meta = {
                "slot": slot,
                "k_count": int(pool.k_counts[slot].item()),
                "v_count": int(pool.v_counts[slot].item()),
                "updates_k": int(pool.update_counts_k[slot].item()),
                "updates_v": int(pool.update_counts_v[slot].item()),
                "last_flush_pos": int(pool.last_flush_pos[slot].item()),
                "active": bool(pool.active[slot].item()),
            }
        snapshots.append(
            {
                "layer": layer_idx,
                "lengths": {
                    "total": total,
                    "sink": int(lengths["sink"][row].item()),
                    "packed_k": packed_k,
                    "pending": int(lengths["pending"][row].item()),
                    "recent": int(lengths["recent"][row].item()),
                    "packed_v": packed_v,
                    "packed_v4": packed_v4,
                },
                "component_hashes": {name: tensor_hash(value) for name, value in tensors.items()},
                "centroid_meta": centroid_meta,
                "page_state": page_state(cache, row),
            }
        )
    return snapshots


def first_snapshot_diff(ref: list[dict[str, Any]], got: list[dict[str, Any]]) -> dict[str, Any] | None:
    for ref_layer, got_layer in zip(ref, got):
        layer = int(ref_layer["layer"])
        if ref_layer["lengths"] != got_layer["lengths"]:
            return {"layer": layer, "component": "segment_lengths", "ref": ref_layer["lengths"], "got": got_layer["lengths"]}
        for name, ref_hash in ref_layer["component_hashes"].items():
            if ref_hash != got_layer["component_hashes"].get(name):
                return {"layer": layer, "component": name, "ref_hash": ref_hash, "got_hash": got_layer["component_hashes"].get(name)}
        if ref_layer["centroid_meta"] != got_layer["centroid_meta"]:
            return {"layer": layer, "component": "centroid_meta", "ref": ref_layer["centroid_meta"], "got": got_layer["centroid_meta"]}
        if ref_layer["page_state"] != got_layer["page_state"]:
            return {"layer": layer, "component": "page_state", "ref": ref_layer["page_state"], "got": got_layer["page_state"]}
    if len(ref) != len(got):
        return {"layer": min(len(ref), len(got)), "component": "layer_count", "ref_layers": len(ref), "got_layers": len(got)}
    return None


def prefill_once(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    next_token = out.logits[:, -1, :].argmax(dim=-1)
    return {"past": out.past_key_values, "next_token": next_token, "logits": out.logits.detach()}


def install_projection_hooks(model: Any) -> tuple[dict[int, dict[str, torch.Tensor]], list[Any]]:
    traces: dict[int, dict[str, torch.Tensor]] = {}
    handles = []
    layers = list(getattr(getattr(model, "model", None), "layers", []))

    def make_hook(layer_idx: int, name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            traces.setdefault(layer_idx, {})[name] = output.detach()

        return hook

    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = getattr(attn, name, None)
            if module is not None:
                handles.append(module.register_forward_hook(make_hook(layer_idx, name)))
    return traces, handles


def decode_once(model: Any, token: torch.Tensor, past: Any, *, trace: bool = False) -> dict[str, Any]:
    traces: dict[int, dict[str, torch.Tensor]] = {}
    handles: list[Any] = []
    if trace:
        traces, handles = install_projection_hooks(model)
    try:
        with torch.inference_mode():
            out = model(input_ids=token[:, None], past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "logits": out.logits[:, -1, :].detach(),
        "past": out.past_key_values,
        "hidden_states": [hidden[:, -1, :].detach() for hidden in out.hidden_states] if getattr(out, "hidden_states", None) is not None else [],
        "projection_traces": traces,
    }


def request_row(request: str, *, same_content: bool = False) -> int:
    if same_content:
        return 0
    return ord(request) - ord("A") if request in "ABCD" else 0


def build_reference_trajectory(
    model: Any,
    inputs: torch.Tensor,
    *,
    request: str,
    context: int,
    input_row: int,
    steps: int = STEPS,
) -> dict[str, Any]:
    prefill = prefill_once(model, inputs[input_row : input_row + 1, :context])
    past = prefill["past"]
    current = prefill["next_token"]
    trajectory: dict[str, Any] = {
        "request": request,
        "context": context,
        "input_row": input_row,
        "prefill_next_token": int(current.item()),
        "snapshots": {"0": cache_layer_snapshot(past, 0)},
        "input_tokens": {},
        "logits": {},
        "step5_trace": None,
    }
    for step in range(1, steps + 1):
        trajectory["input_tokens"][str(step)] = int(current.item())
        out = decode_once(model, current, past, trace=(request == TRACE_REQUEST and step == TRACE_STEP))
        past = out["past"]
        trajectory["logits"][str(step)] = out["logits"][0].detach().cpu()
        if step <= TRACE_STEP - 1:
            trajectory["snapshots"][str(step)] = cache_layer_snapshot(past, 0)
        if request == TRACE_REQUEST and step == TRACE_STEP:
            trajectory["step5_trace"] = trace_to_cpu(out, row=0)
        current = out["logits"].argmax(dim=-1)
    return trajectory


def trace_to_cpu(out: dict[str, Any], *, row: int) -> dict[str, Any]:
    projections = {}
    for layer_idx, values in out.get("projection_traces", {}).items():
        projections[str(layer_idx)] = {name: value[row : row + 1].detach().cpu() for name, value in values.items()}
    return {
        "hidden_states": [hidden[row : row + 1].detach().cpu() for hidden in out.get("hidden_states", [])],
        "projections": projections,
        "logits": out["logits"][row : row + 1].detach().cpu(),
    }


def projection_metrics(ref_trace: dict[str, Any] | None, got_trace: dict[str, Any] | None) -> dict[str, Any]:
    if not ref_trace or not got_trace:
        return {"available": False}
    rows: list[dict[str, Any]] = []
    first = None
    for layer, ref_values in sorted(ref_trace.get("projections", {}).items(), key=lambda item: int(item[0])):
        got_values = got_trace.get("projections", {}).get(layer, {})
        for name, ref_tensor in ref_values.items():
            got_tensor = got_values.get(name)
            metrics = tensor_metric_summary(got_tensor, ref_tensor)
            row = {"layer": int(layer), "component": name, **metrics}
            rows.append(row)
            rel = metrics.get("relative_l2")
            max_abs = metrics.get("max_abs")
            if first is None and rel is not None and max_abs is not None and (float(rel) > 0.0 or float(max_abs) > 0.0):
                first = row
    hidden_rows = []
    for idx, ref_hidden in enumerate(ref_trace.get("hidden_states", [])):
        if idx >= len(got_trace.get("hidden_states", [])):
            break
        metrics = tensor_metric_summary(got_trace["hidden_states"][idx], ref_hidden)
        hidden_rows.append({"hidden_index": idx, **metrics})
    logits = tensor_metric_summary(got_trace.get("logits"), ref_trace.get("logits"))
    return {"available": True, "projections": rows, "first_projection_drift": first, "hidden_states": hidden_rows, "logits": logits}


def build_clean_references(model: Any, inputs: torch.Tensor, case: dict[str, Any]) -> dict[str, Any]:
    refs = {}
    for request, spec in case["requests"].items():
        refs[request] = build_reference_trajectory(
            model,
            inputs,
            request=request,
            context=int(spec["context"]),
            input_row=int(spec["input_row"]),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return refs


def repeated_prefill_determinism(model: Any, inputs: torch.Tensor, spec: dict[str, Any]) -> dict[str, Any]:
    first = prefill_once(model, inputs[int(spec["input_row"]) : int(spec["input_row"]) + 1, : int(spec["context"])])
    first_snapshot = cache_layer_snapshot(first["past"], 0)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    second = prefill_once(model, inputs[int(spec["input_row"]) : int(spec["input_row"]) + 1, : int(spec["context"])])
    second_snapshot = cache_layer_snapshot(second["past"], 0)
    diff = first_snapshot_diff(first_snapshot, second_snapshot)
    return {"match": diff is None, "first_diff": diff}


def run_ragged_forced(model: Any, inputs: torch.Tensor, case: dict[str, Any], refs: dict[str, Any]) -> dict[str, Any]:
    requests = list(case["requests"])
    ragged_prefills = []
    for request in requests:
        spec = case["requests"][request]
        prefill = prefill_once(model, inputs[int(spec["input_row"]) : int(spec["input_row"]) + 1, : int(spec["context"])])
        ragged_prefills.append(prefill["past"])
    assembled = [assemble_ragged_patternkv_cache([past[layer] for past in ragged_prefills]) for layer in range(len(ragged_prefills[0]))]
    ragged_past = tuple(serialize_cache(cache) for cache in assembled)
    current_tokens = {request: torch.tensor([refs[request]["input_tokens"]["1"]], dtype=torch.long, device=inputs.device) for request in requests}
    steps = []
    snapshots = {"0": {request: cache_layer_snapshot(ragged_past, idx) for idx, request in enumerate(requests)}}
    first_failure = None
    reset_ragged_k_counters()
    reset_patternkv_real_decode_counters()
    step5_trace = None
    for step in range(1, STEPS + 1):
        ragged_input = torch.stack([current_tokens[request] for request in requests]).view(len(requests))
        out = decode_once(model, ragged_input, ragged_past, trace=(TRACE_REQUEST in requests and step == TRACE_STEP))
        ragged_past = out["past"]
        step_metrics = {}
        for idx, request in enumerate(requests):
            metrics = compare_logits(out["logits"][idx], refs[request]["logits"][str(step)].to(device=out["logits"].device))
            metrics["step"] = step
            metrics["request"] = request
            step_metrics[request] = metrics
            passed = bool(metrics["top1_equal"] and int(metrics["top5_overlap"]) >= 4 and float(metrics["relative_l2"]) <= 1e-2)
            if first_failure is None and not passed:
                first_failure = {"step": step, "request": request, "metrics": metrics}
        if step <= TRACE_STEP - 1:
            snapshots[str(step)] = {request: cache_layer_snapshot(ragged_past, idx) for idx, request in enumerate(requests)}
        if TRACE_REQUEST in requests and step == TRACE_STEP:
            step5_trace = trace_to_cpu(out, row=requests.index(TRACE_REQUEST))
        steps.append({"step": step, "metrics": step_metrics})
        current_tokens = {request: torch.tensor([refs[request]["input_tokens"].get(str(step + 1), refs[request]["logits"][str(step)].argmax().item())], dtype=torch.long, device=inputs.device) for request in requests}
    return {
        "requests": requests,
        "steps": steps,
        "first_failure": first_failure,
        "snapshots": snapshots,
        "step5_trace": step5_trace,
        "runtime_counters": {"ragged_k": get_ragged_k_counters(), "real_decode": get_patternkv_real_decode_counters()},
    }


def compare_pre_step_state(refs: dict[str, Any], ragged: dict[str, Any], request: str) -> dict[str, Any]:
    diffs = []
    earliest = None
    for step in range(0, TRACE_STEP):
        ref_snapshot = refs[request]["snapshots"][str(step)]
        got_snapshot = ragged["snapshots"][str(step)][request]
        diff = first_snapshot_diff(ref_snapshot, got_snapshot)
        row = {"step": step, "match": diff is None, "first_diff": diff}
        diffs.append(row)
        if earliest is None and diff is not None:
            earliest = row
    return {"request": request, "steps": diffs, "earliest": earliest}


def case_pass(case_result: dict[str, Any]) -> bool:
    return case_result["first_failure"] is None and len(case_result["steps"]) == STEPS


def summarize_case(case_result: dict[str, Any]) -> dict[str, Any]:
    rows = [metric for step in case_result["steps"] for metric in step["metrics"].values()]
    max_row = max(rows, key=lambda item: float(item["relative_l2"])) if rows else {}
    return {
        "pass": case_pass(case_result),
        "first_failure": case_result["first_failure"],
        "max_relative_l2": float(max_row.get("relative_l2", 0.0)),
        "max_relative_l2_request": max_row.get("request", ""),
        "max_relative_l2_step": max_row.get("step"),
        "all_top1": all(bool(row["top1_equal"]) for row in rows),
        "min_top5": min((int(row["top5_overlap"]) for row in rows), default=0),
    }


def make_cases() -> dict[str, dict[str, Any]]:
    return {
        "C0_different_content_different_length": {
            "requests": {"A": {"context": 384, "input_row": 0}, "B": {"context": 513, "input_row": 1}},
        },
        "C1_different_content_same_length": {
            "requests": {"A": {"context": 512, "input_row": 0}, "B": {"context": 512, "input_row": 1}},
        },
        "C2_same_content_different_length": {
            "requests": {"A": {"context": 384, "input_row": 0}, "B": {"context": 513, "input_row": 0}},
        },
        "C3_same_content_same_length": {
            "requests": {"A": {"context": 512, "input_row": 0}, "B": {"context": 512, "input_row": 0}},
        },
    }


def write_static_audit() -> dict[str, Any]:
    audit = {
        "v_causal_importance": "CACHE_LOCAL",
        "v_oracle_importance": "CACHE_LOCAL",
        "selector_task_key": "DEBUG_ONLY",
        "attention_layer_k_base": "REQUEST_LOCAL_MODEL_STATE_RESET_ON_PREFILL",
        "attention_layer_v_centroids": "REQUEST_LOCAL_MODEL_STATE_RESET_ON_PREFILL",
        "centroid_state_pool": "CACHE_LOCAL",
        "operator_ready_page_pools": "CACHE_LOCAL",
        "profiling_counters": "DEBUG_ONLY",
        "ragged_decode_counters": "DEBUG_ONLY",
        "model_global_math_state_detected": False,
    }
    write_md(
        REPORT_DIR / "model_mutable_state_audit.md",
        "Model Mutable State Audit",
        "\n".join(f"- `{key}`: `{value}`" for key, value in audit.items()),
    )
    return audit


def run(device: torch.device) -> dict[str, Any]:
    set_env()
    tokenizer, _config, model = load_model(dtype=torch.float16, device=device)
    inputs = make_fixed_inputs(tokenizer, batch=4, context=771, device=device)
    cases = make_cases()
    prefill_determinism = repeated_prefill_determinism(model, inputs, cases["C0_different_content_different_length"]["requests"][TRACE_REQUEST])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    results = {}
    for name, case in cases.items():
        refs = build_clean_references(model, inputs, case)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ragged = run_ragged_forced(model, inputs, case, refs)
        state_diff = compare_pre_step_state(refs, ragged, TRACE_REQUEST) if TRACE_REQUEST in case["requests"] else None
        trace = projection_metrics(refs[TRACE_REQUEST]["step5_trace"], ragged["step5_trace"]) if TRACE_REQUEST in refs else {"available": False}
        results[name] = {"case": case, "summary": summarize_case(ragged), "ragged": ragged, "state_diff": state_diff, "step5_trace": trace}
        del refs, ragged
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "start_head": git_output(["rev-parse", "HEAD"]),
        "cases": results,
        "prefill_determinism": prefill_determinism,
    }


def build_final_gate(payload: dict[str, Any], *, compileall_pass: bool | None = None, pytest_result: str = "") -> dict[str, Any]:
    c0 = payload["cases"]["C0_different_content_different_length"]
    c1 = payload["cases"]["C1_different_content_same_length"]
    c2 = payload["cases"]["C2_same_content_different_length"]
    c3 = payload["cases"]["C3_same_content_same_length"]
    state = c0["state_diff"] or {}
    earliest = state.get("earliest")
    trace = c0["step5_trace"]
    first_projection = trace.get("first_projection_drift") if trace.get("available") else None
    q_row = next((row for row in trace.get("projections", []) if row["component"] == "q_proj"), {})
    k_row = next((row for row in trace.get("projections", []) if row["component"] == "k_proj"), {})
    v_row = next((row for row in trace.get("projections", []) if row["component"] == "v_proj"), {})
    o_row = next((row for row in trace.get("projections", []) if row["component"] == "o_proj"), {})
    model_audit = payload.get("model_mutable_state_audit", {})
    c0_pass = bool(c0["summary"]["pass"])
    prefill_diff = payload.get("prefill_determinism", {}).get("first_diff")
    if c0_pass:
        classification = "RAGGED_MULTISTEP_REFERENCE_ORACLE_ARTIFACT"
        root_cause = "Clean independent B1 oracle removes the previous C0 step5 drift; the earlier failure was caused by reference/runtime interleaving in the formal runner."
        next_task = "FIX_RAGGED_MULTISTEP_REFERENCE_ORACLE"
    elif earliest and "v_causal_importance" in str(earliest.get("first_diff", {}).get("component", "")):
        classification = "RAGGED_CAUSAL_IMPORTANCE_STATE_DIVERGENCE"
        root_cause = "Clean oracle still fails and the earliest valid-prefix state mismatch is v_causal_importance."
        next_task = "FIX_RAGGED_CAUSAL_IMPORTANCE_STATE_DIVERGENCE"
    elif prefill_diff and "centroid" in str(prefill_diff.get("component", "")):
        classification = "RAGGED_CENTROID_STATE_ACCUMULATION_DIVERGENCE"
        root_cause = f"Clean oracle still fails and repeated independent B1 prefill is not bit-stable at `{prefill_diff.get('component')}`; ragged C0 first differs at `{earliest.get('first_diff', {}).get('component') if earliest else ''}` step {earliest.get('step') if earliest else None}."
        next_task = "FIX_RAGGED_CENTROID_STATE_DETERMINISM"
    elif earliest:
        component = str(earliest.get("first_diff", {}).get("component", ""))
        if "page" in component or "packed_v" in component or component.startswith("v_"):
            classification = "RAGGED_V_PAGE_STATE_ACCUMULATION_DIVERGENCE"
        elif "centroid" in component:
            classification = "RAGGED_CENTROID_STATE_ACCUMULATION_DIVERGENCE"
        else:
            classification = "RAGGED_NONBOUNDARY_SEGMENT_STATE_DIVERGENCE"
        root_cause = f"Clean oracle still fails; earliest request-A valid-prefix cache mismatch is `{component}` at step {earliest.get('step')}."
        next_task = f"FIX_{classification}"
    elif bool(c1["summary"]["pass"]):
        classification = "RAGGED_MULTI_STEP_SEMANTIC_DRIFT_UNEXPLAINED"
        root_cause = "Clean oracle still fails, pre-step5 cache hashes match, and equal-length control passes; drift must be localized inside step5 math."
        next_task = "TRACE_RAGGED_STEP5_ATTENTION_MATH"
    else:
        classification = "GENERIC_MULTI_STEP_BATCH_NUMERICAL_DRIFT_CONFIRMED"
        root_cause = "Different-content equal-length control also exceeds the relL2 gate under the clean oracle."
        next_task = "QUANTIFY_GENERIC_BATCH_NUMERICAL_DRIFT"
    return {
        "start_head": START_HEAD,
        "actual_start_head": payload["start_head"],
        "previous_classification": "RAGGED_MULTI_STEP_SEMANTIC_DRIFT_UNEXPLAINED",
        "previous_first_failure": {"request": "A", "step": 5, "boundary": False, "relative_l2": 0.027149327099323273},
        "reference_interleaving_removed": True,
        "reference_runtime_interleaving_artifact": c0_pass,
        "model_global_math_state_detected": bool(model_audit.get("model_global_math_state_detected", False)),
        "repeated_prefill_deterministic": bool(payload.get("prefill_determinism", {}).get("match", False)),
        "repeated_prefill_first_diff": payload.get("prefill_determinism", {}).get("first_diff"),
        "different_content_equal_length_control_pass": bool(c1["summary"]["pass"]),
        "same_content_different_length_control_pass": bool(c2["summary"]["pass"]),
        "same_content_same_length_control_pass": bool(c3["summary"]["pass"]),
        "earliest_state_divergence_step": None if earliest is None else earliest.get("step"),
        "earliest_state_divergence_component": "" if earliest is None else str(earliest.get("first_diff", {}).get("component", "")),
        "pre_step5_cache_semantics_match": earliest is None,
        "first_divergent_layer": None if first_projection is None else first_projection.get("layer"),
        "first_divergent_component": "" if first_projection is None else str(first_projection.get("component", "")),
        "causal_importance_semantics_match": not (earliest and "v_causal_importance" in str(earliest.get("first_diff", {}).get("component", ""))),
        "cross_request_causal_importance_leakage": bool(earliest and "v_causal_importance" in str(earliest.get("first_diff", {}).get("component", ""))),
        "current_q_drift_step5": q_row.get("relative_l2"),
        "current_k_drift_step5": k_row.get("relative_l2"),
        "current_v_drift_step5": v_row.get("relative_l2"),
        "attention_output_drift_step5": o_row.get("relative_l2"),
        "generic_equal_length_drift_step5": c1["summary"]["max_relative_l2"],
        "causal_oracle_confirmed": earliest is not None or first_projection is not None or c0_pass,
        "root_cause": root_cause,
        "production_fix_applied": False,
        "runner_fix_applied": False,
        "b2_16step_after_fix_pass": None,
        "b2_reorder_after_fix_pass": None,
        "b4_after_fix_pass": None,
        "independent_flush_schedule_after_fix_pass": None,
        "all_top1_after_fix": c0["summary"]["all_top1"],
        "min_top5_after_fix": c0["summary"]["min_top5"],
        "max_rel_l2_after_fix": c0["summary"]["max_relative_l2"],
        "serial_request_dispatches": int(c0["ragged"]["runtime_counters"]["ragged_k"]["serial_request_dispatches"])
        + int(c0["ragged"]["runtime_counters"]["real_decode"]["serial_b1_dispatches"]),
        "historical_fp16_k_materialization": int(c0["ragged"]["runtime_counters"]["ragged_k"]["historical_fp16_k_materialization"]),
        "historical_fp16_v_materialization": int(c0["ragged"]["runtime_counters"]["real_decode"]["historical_v_materialization_bytes"]),
        "compileall_pass": compileall_pass,
        "pytest_result": pytest_result,
        "classification": classification,
        "next_task": next_task,
    }


def strip_tensors(payload: Any) -> Any:
    if torch.is_tensor(payload):
        return {"shape": list(payload.shape), "dtype": str(payload.dtype), "hash": tensor_hash(payload)}
    if isinstance(payload, dict):
        return {key: strip_tensors(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [strip_tensors(value) for value in payload]
    return payload


def write_reports(payload: dict[str, Any]) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload["model_mutable_state_audit"] = write_static_audit()
    gate = build_final_gate(payload)
    write_json(REPORT_DIR / "reproduction.json", strip_tensors(payload["cases"]["C0_different_content_different_length"]["summary"]))
    write_json(REPORT_DIR / "clean_b1_reference.json", {"reference_interleaving_removed": True})
    write_json(REPORT_DIR / "prefill_determinism.json", strip_tensors(payload["prefill_determinism"]))
    write_json(REPORT_DIR / "control_matrix.json", {name: case["summary"] for name, case in payload["cases"].items()})
    write_json(REPORT_DIR / "state_diff_by_step.json", strip_tensors(payload["cases"]["C0_different_content_different_length"]["state_diff"]))
    write_json(REPORT_DIR / "adaptive_state_by_step.json", strip_tensors(payload["cases"]["C0_different_content_different_length"]["state_diff"]))
    write_json(REPORT_DIR / "temporal_drift.json", {name: [step["metrics"] for step in case["ragged"]["steps"]] for name, case in payload["cases"].items()})
    write_json(REPORT_DIR / "first_state_divergence.json", strip_tensors(payload["cases"]["C0_different_content_different_length"]["state_diff"]["earliest"]))
    write_json(REPORT_DIR / "step5_first_divergence.json", strip_tensors(payload["cases"]["C0_different_content_different_length"]["step5_trace"].get("first_projection_drift")))
    write_json(REPORT_DIR / "causal_oracle_results.json", {"clean_oracle": payload["cases"]["C0_different_content_different_length"]["summary"], "step5_trace": strip_tensors(payload["cases"]["C0_different_content_different_length"]["step5_trace"])})
    write_json(REPORT_DIR / "post_fix_multistep.json", {"not_run": True, "reason": "forensic pass only; no production fix applied"})
    write_json(REPORT_DIR / "final_gate.json", gate)
    write_md(REPORT_DIR / "reproduction.md", "Reproduction", json.dumps(payload["cases"]["C0_different_content_different_length"]["summary"], indent=2, sort_keys=True))
    write_md(REPORT_DIR / "reference_oracle_audit.md", "Reference Oracle Audit", "Clean B1 references are built independently: A full prefill+16 decode, reset, B full prefill+16 decode, reset, then ragged forced replay with saved tokens/logits.\n\n`REFERENCE_INTERLEAVING_REMOVED=true`\n\nRepeated prefill determinism:\n\n```json\n" + json.dumps(strip_tensors(payload["prefill_determinism"]), indent=2, sort_keys=True) + "\n```")
    write_md(REPORT_DIR / "control_matrix.md", "Control Matrix", json.dumps({name: case["summary"] for name, case in payload["cases"].items()}, indent=2, sort_keys=True))
    write_md(REPORT_DIR / "stepwise_state_equivalence.md", "Stepwise State Equivalence", json.dumps(strip_tensors(payload["cases"]["C0_different_content_different_length"]["state_diff"]), indent=2, sort_keys=True))
    write_md(REPORT_DIR / "adaptive_state_audit.md", "Adaptive State Audit", "Adaptive state is included in the valid-prefix cache hash comparator: `v_causal_importance`, `v_oracle_importance`, centroid pool values/counts/update counters, and page state metadata.")
    write_md(REPORT_DIR / "temporal_drift.md", "Temporal Drift", "See `temporal_drift.json` for per-step metrics across C0-C3.")
    write_md(REPORT_DIR / "step5_layerwise_trace.md", "Step5 Layerwise Trace", json.dumps(strip_tensors(payload["cases"]["C0_different_content_different_length"]["step5_trace"]), indent=2, sort_keys=True))
    write_md(REPORT_DIR / "causal_oracles.md", "Causal Oracles", json.dumps({"clean_oracle": payload["cases"]["C0_different_content_different_length"]["summary"], "control_matrix": {name: case["summary"] for name, case in payload["cases"].items()}}, indent=2, sort_keys=True))
    write_md(REPORT_DIR / "root_cause.md", "Root Cause", gate["root_cause"])
    write_md(REPORT_DIR / "post_fix_gate.md", "Post-Fix Gate", "No production or formal-runner fix was applied in this forensic run.")
    write_md(REPORT_DIR / "pytest.md", "Pytest", "Validation results are filled after explicit test runs.")
    write_md(REPORT_DIR / "environment.md", "Environment", f"HEAD: `{payload['start_head']}`\n\n```text\n{nvidia_smi().strip()}\n```")
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    payload = run(torch.device(args.device))
    payload["elapsed_s"] = time.perf_counter() - started
    gate = write_reports(payload)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
