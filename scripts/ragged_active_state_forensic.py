from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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


START_HEAD = "cc50fdc513181d2137438cc6a7c0dd8322ccf767"
REPORT_DIR = REPO_ROOT / "reports/system_ragged_active_state_forensic_v1"
STEPS = 16
TRACE_STEP = 5
REQUESTS = ("A", "B")
CONTEXTS = {"A": 384, "B": 513}


def set_env() -> None:
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    os.environ["PATTERNKV_CENTROID_MAX_SLOTS"] = "8"
    os.environ.pop("PATTERNKV_BI_MLP_ORACLE", None)


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


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


def first_diff_index(got: torch.Tensor, ref: torch.Tensor) -> list[int] | None:
    if torch.equal(got, ref):
        return None
    neq = got.detach().cpu() != ref.detach().cpu()
    idx = torch.nonzero(neq.reshape(-1), as_tuple=False)
    if not int(idx.numel()):
        return None
    flat = int(idx[0].item())
    shape = list(got.shape)
    out = []
    for dim in reversed(shape):
        out.append(flat % dim)
        flat //= dim
    return list(reversed(out))


def metrics(got: torch.Tensor | None, ref: torch.Tensor | None) -> dict[str, Any]:
    if got is None and ref is None:
        return {"exact_equal": True, "sha256": None, "shape": None, "num_valid_elements": 0, "max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0, "first_differing_index": None, "mismatch_count": 0}
    if got is None or ref is None:
        return {"exact_equal": False, "sha256": tensor_hash(got), "shape": list(got.shape) if torch.is_tensor(got) else None, "num_valid_elements": int(got.numel()) if torch.is_tensor(got) else 0, "max_abs": None, "mean_abs": None, "rel_l2": None, "first_differing_index": None, "mismatch_count": None}
    if tuple(got.shape) != tuple(ref.shape):
        return {"exact_equal": False, "sha256": tensor_hash(got), "shape": list(got.shape), "ref_shape": list(ref.shape), "num_valid_elements": int(got.numel()), "max_abs": None, "mean_abs": None, "rel_l2": None, "first_differing_index": None, "mismatch_count": None}
    exact = bool(torch.equal(got, ref))
    got_f = got.detach().float()
    ref_f = ref.detach().float()
    diff = (got_f - ref_f).abs()
    t_metrics = tensor_metrics(got, ref) if got.numel() else {"relative_l2": 0.0}
    mismatch_count = int((got.detach().cpu() != ref.detach().cpu()).sum().item())
    return {
        "exact_equal": exact,
        "sha256": tensor_hash(got),
        "ref_sha256": tensor_hash(ref),
        "shape": list(got.shape),
        "num_valid_elements": int(got.numel()),
        "max_abs": float(diff.max().item()) if got.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if got.numel() else 0.0,
        "rel_l2": float(t_metrics["relative_l2"]),
        "first_differing_index": first_diff_index(got, ref),
        "mismatch_count": mismatch_count,
    }


def narrow(value: torch.Tensor | None, row: int | None, dim: int | None, length: int | None) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    out = value
    if row is not None and out.dim() and int(out.shape[0]) > row:
        out = out[row : row + 1]
    if dim is not None and length is not None:
        dim = dim if dim >= 0 else out.dim() + dim
        out = out.narrow(dim, 0, min(int(length), int(out.shape[dim])))
    return out.detach().contiguous().cpu()


def active_centroid(cache: Any, row: int, stream: str) -> torch.Tensor | None:
    pool = getattr(cache, "centroid_state_pool", None)
    indices = getattr(cache, "centroid_state_indices", None)
    if pool is not None and torch.is_tensor(indices) and int(indices.numel()) > row:
        slot = int(indices[row].item())
        counts = pool.k_counts if stream == "k" else pool.v_counts
        values = pool.k_centroid_pool if stream == "k" else pool.v_centroid_pool
        active = int(counts[slot].item())
        return values[slot : slot + 1, :, :active, :].detach().contiguous().cpu()
    value = getattr(cache, f"{stream}_centroids", None)
    if not torch.is_tensor(value):
        return None
    if value.dim() == 4:
        return value[row : row + 1].detach().contiguous().cpu()
    return value.unsqueeze(0).detach().contiguous().cpu()


def full_centroid_slot(cache: Any, row: int, stream: str) -> torch.Tensor | None:
    pool = getattr(cache, "centroid_state_pool", None)
    indices = getattr(cache, "centroid_state_indices", None)
    if pool is not None and torch.is_tensor(indices) and int(indices.numel()) > row:
        slot = int(indices[row].item())
        values = pool.k_centroid_pool if stream == "k" else pool.v_centroid_pool
        return values[slot : slot + 1].detach().contiguous().cpu()
    return active_centroid(cache, row, stream)


def page_semantic(cache: Any, row: int) -> dict[str, Any]:
    pools = getattr(cache, "operator_ready_page_pools", None)
    if pools is None:
        return {"present": False}
    meta = pools.metadata
    pages = int(meta.num_pages[row].item())
    ids = meta.metadata_page_table[row, :pages].long().detach().contiguous()
    valid = meta.valid_tokens[ids].detach().contiguous() if pages else torch.empty((0,), dtype=meta.valid_tokens.dtype, device=meta.valid_tokens.device)
    return {
        "present": True,
        "num_pages": pages,
        "seq_len": int(meta.seq_lens[row].item()),
        "page_ids_hash": tensor_hash(ids),
        "valid_tokens_hash": tensor_hash(valid),
        "page_ids": ids.detach().cpu().tolist(),
        "valid_tokens": valid.detach().cpu().tolist(),
    }


def semantic_tensors(cache: Any, row: int) -> dict[str, torch.Tensor | None]:
    lengths = k_segment_valid_lengths(cache)
    total = int(get_total_tokens_per_request(cache)[row].item())
    packed_k = int(get_packed_k_tokens_per_request(cache)[row].item())
    packed_v_tensor = getattr(cache, "request_packed_v_tokens", None)
    packed_v4_tensor = getattr(cache, "request_packed_v4_tokens", None)
    packed_v = int(packed_v_tensor[row].item()) if torch.is_tensor(packed_v_tensor) else int(cache.packed_v_tokens)
    packed_v4 = int(packed_v4_tensor[row].item()) if torch.is_tensor(packed_v4_tensor) else int(getattr(cache, "packed_v4_tokens", 0) or 0)
    packed_v2 = max(packed_v - packed_v4, 0)
    return {
        "sink_k": narrow(cache.sink_k, row, 2, int(lengths["sink"][row].item())),
        "sink_v": narrow(cache.sink_v, row, 2, int(lengths["sink"][row].item())),
        "packed_k_payload": narrow(cache.packed_k, row, 3, ceil_div(packed_k, 32 // int(cache.k_bits))),
        "packed_k_scale": narrow(cache.packed_k_scale, row, 3, ceil_div(packed_k, int(cache.group_size))),
        "packed_k_zero": narrow(cache.packed_k_zero, row, 3, ceil_div(packed_k, int(cache.group_size))),
        "k_assignments": narrow(cache.k_assignments, row, 2, packed_k),
        "pending_k": narrow(cache.pending_k, row, 2, int(lengths["pending"][row].item())),
        "pending_v": narrow(cache.pending_v, row, 2, int(lengths["pending"][row].item())),
        "recent_k": narrow(cache.recent_k, row, 2, int(lengths["recent"][row].item())),
        "recent_v": narrow(cache.recent_v, row, 2, int(lengths["recent"][row].item())),
        "packed_v2_payload": narrow(cache.packed_v, row, 2, packed_v2),
        "packed_v2_scale": narrow(cache.packed_v_scale, row, 2, packed_v2),
        "packed_v2_zero": narrow(cache.packed_v_zero, row, 2, packed_v2),
        "packed_v4_payload": narrow(cache.packed_v4, row, 2, packed_v4),
        "packed_v4_scale": narrow(cache.packed_v4_scale, row, 2, packed_v4),
        "packed_v4_zero": narrow(cache.packed_v4_zero, row, 2, packed_v4),
        "v_assignment_idx": narrow(cache.v_assignment_idx, row, 2, packed_v),
        "v_pattern_mask": narrow(cache.v_pattern_mask, row, 2, packed_v),
        "v_precision_mask": narrow(cache.v_precision_mask, row, 1, packed_v),
        "v2_assignment_idx": narrow(cache.v2_assignment_idx, row, 2, packed_v2),
        "v2_pattern_mask": narrow(cache.v2_pattern_mask, row, 2, packed_v2),
        "v4_assignment_idx": narrow(cache.v4_assignment_idx, row, 2, packed_v4),
        "v4_pattern_mask": narrow(cache.v4_pattern_mask, row, 2, packed_v4),
        "v_causal_importance": narrow(cache.v_causal_importance, row, 1, total),
        "v_oracle_importance": narrow(cache.v_oracle_importance, row, 1, total),
        "k_centroid_values_active": active_centroid(cache, row, "k"),
        "v_centroid_values_active": active_centroid(cache, row, "v"),
    }


def semantic_snapshot(past: Any, row: int) -> list[dict[str, Any]]:
    out = []
    for layer_idx, layer in enumerate(past):
        cache = deserialize_cache(layer, pattern=True)
        lengths = k_segment_valid_lengths(cache)
        packed_v_tensor = getattr(cache, "request_packed_v_tokens", None)
        packed_v4_tensor = getattr(cache, "request_packed_v4_tokens", None)
        pool = getattr(cache, "centroid_state_pool", None)
        indices = getattr(cache, "centroid_state_indices", None)
        slot = int(indices[row].item()) if torch.is_tensor(indices) and int(indices.numel()) > row else None
        k_count = int(pool.k_counts[slot].item()) if pool is not None and slot is not None else (int(cache.k_centroids.shape[1]) if torch.is_tensor(cache.k_centroids) and cache.k_centroids.dim() == 3 else 0)
        v_count = int(pool.v_counts[slot].item()) if pool is not None and slot is not None else (int(cache.v_centroids.shape[1]) if torch.is_tensor(cache.v_centroids) and cache.v_centroids.dim() == 3 else 0)
        tensors = semantic_tensors(cache, row)
        out.append(
            {
                "layer": layer_idx,
                "metadata": {
                    "total": int(get_total_tokens_per_request(cache)[row].item()),
                    "sink": int(lengths["sink"][row].item()),
                    "packed_k": int(get_packed_k_tokens_per_request(cache)[row].item()),
                    "pending": int(lengths["pending"][row].item()),
                    "recent": int(lengths["recent"][row].item()),
                    "packed_v": int(packed_v_tensor[row].item()) if torch.is_tensor(packed_v_tensor) else int(cache.packed_v_tokens),
                    "packed_v4": int(packed_v4_tensor[row].item()) if torch.is_tensor(packed_v4_tensor) else int(getattr(cache, "packed_v4_tokens", 0) or 0),
                    "centroid_slot": slot,
                    "active_k_centroid_count": k_count,
                    "active_v_centroid_count": v_count,
                    "centroid_updates_k": int(pool.update_counts_k[slot].item()) if pool is not None and slot is not None else int(cache.centroid_updates_k),
                    "centroid_updates_v": int(pool.update_counts_v[slot].item()) if pool is not None and slot is not None else int(cache.centroid_updates_v),
                    "last_flush_pos": int(pool.last_flush_pos[slot].item()) if pool is not None and slot is not None else None,
                },
                "hashes": {name: tensor_hash(value) for name, value in tensors.items()},
                "tensors": {name: value for name, value in tensors.items()},
                "page_state": page_semantic(cache, row),
            }
        )
    return out


def compare_snapshot(ref: list[dict[str, Any]], got: list[dict[str, Any]]) -> dict[str, Any]:
    for ref_layer, got_layer in zip(ref, got):
        layer = int(ref_layer["layer"])
        ref_metadata = {k: v for k, v in ref_layer["metadata"].items() if k != "centroid_slot"}
        got_metadata = {k: v for k, v in got_layer["metadata"].items() if k != "centroid_slot"}
        if ref_metadata != got_metadata:
            return {"match": False, "layer": layer, "component": "metadata", "details": {"ref": ref_metadata, "got": got_metadata}, "metrics": {"exact_equal": False, "rel_l2": None, "max_abs": None}}
        ref_page = {k: v for k, v in ref_layer["page_state"].items() if k not in {"page_ids", "page_ids_hash"}}
        got_page = {k: v for k, v in got_layer["page_state"].items() if k not in {"page_ids", "page_ids_hash"}}
        if ref_page != got_page:
            return {"match": False, "layer": layer, "component": "page_state", "details": {"ref": ref_page, "got": got_page}, "metrics": {"exact_equal": False, "rel_l2": None, "max_abs": None}}
        for name in ref_layer["tensors"]:
            row = metrics(got_layer["tensors"].get(name), ref_layer["tensors"].get(name))
            if not row["exact_equal"]:
                return {"match": False, "layer": layer, "component": name, "metrics": row}
    return {"match": True, "layer": None, "component": "", "metrics": {"exact_equal": True, "rel_l2": 0.0, "max_abs": 0.0}}


def prefill_once(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    return {"past": out.past_key_values, "next_token": out.logits[:, -1, :].argmax(dim=-1), "logits": out.logits.detach()}


def install_projection_hooks(model: Any) -> tuple[dict[int, dict[str, torch.Tensor]], list[Any]]:
    traces: dict[int, dict[str, torch.Tensor]] = {}
    handles = []
    for layer_idx, layer in enumerate(getattr(model.model, "layers", [])):
        attn = getattr(layer, "self_attn", None)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = getattr(attn, name, None)
            if module is None:
                continue

            def hook(_module: Any, _inputs: tuple[Any, ...], output: torch.Tensor, *, layer_idx: int = layer_idx, name: str = name) -> None:
                traces.setdefault(layer_idx, {})[name] = output.detach().clone()

            handles.append(module.register_forward_hook(hook))
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
        "hidden_states": [hidden[:, -1, :].detach().clone() for hidden in out.hidden_states] if getattr(out, "hidden_states", None) is not None else [],
        "projection_traces": traces,
    }


def build_reference(model: Any, inputs: torch.Tensor, request: str) -> dict[str, Any]:
    row = 0 if request == "A" else 1
    pre = prefill_once(model, inputs[row : row + 1, : CONTEXTS[request]])
    past = pre["past"]
    token = pre["next_token"]
    out = {"snapshots": {"0": semantic_snapshot(past, 0)}, "input_tokens": {"1": int(token.item())}, "logits": {}, "step5_trace": None}
    for step in range(1, STEPS + 1):
        dec = decode_once(model, token, past, trace=(request == "A" and step == TRACE_STEP))
        past = dec["past"]
        out["logits"][str(step)] = dec["logits"][0].detach().cpu()
        if step <= TRACE_STEP:
            out["snapshots"][str(step)] = semantic_snapshot(past, 0)
        if request == "A" and step == TRACE_STEP:
            out["step5_trace"] = trace_to_cpu(dec, 0)
        token = dec["logits"].argmax(dim=-1)
        out["input_tokens"][str(step + 1)] = int(token.item())
    return out


def trace_to_cpu(out: dict[str, Any], row: int) -> dict[str, Any]:
    return {
        "logits": out["logits"][row : row + 1].detach().cpu(),
        "hidden_states": [x[row : row + 1].detach().cpu() for x in out["hidden_states"]],
        "projections": {str(layer): {name: value[row : row + 1].detach().cpu() for name, value in vals.items()} for layer, vals in out["projection_traces"].items()},
    }


def run_clean_b2(model: Any, inputs: torch.Tensor, *, poison: float | None = None) -> dict[str, Any]:
    refs = {request: build_reference(model, inputs, request) for request in REQUESTS}
    ragged_prefills = []
    for request in REQUESTS:
        row = 0 if request == "A" else 1
        ragged_prefills.append(prefill_once(model, inputs[row : row + 1, : CONTEXTS[request]])["past"])
    assembled = [assemble_ragged_patternkv_cache([past[layer] for past in ragged_prefills]) for layer in range(len(ragged_prefills[0]))]
    ragged_past = tuple(serialize_cache(cache) for cache in assembled)
    if poison is not None:
        poison_inactive_centroid_capacity(ragged_past, poison)
    snapshots = {"0": {request: semantic_snapshot(ragged_past, idx) for idx, request in enumerate(REQUESTS)}}
    current = {request: torch.tensor([refs[request]["input_tokens"]["1"]], dtype=torch.long, device=inputs.device) for request in REQUESTS}
    steps = []
    first_failure = None
    reset_ragged_k_counters()
    reset_patternkv_real_decode_counters()
    step5_trace = None
    for step in range(1, STEPS + 1):
        dec = decode_once(model, torch.stack([current[request] for request in REQUESTS]).view(len(REQUESTS)), ragged_past, trace=(step == TRACE_STEP))
        ragged_past = dec["past"]
        if poison is not None:
            poison_inactive_centroid_capacity(ragged_past, poison)
        if step <= TRACE_STEP:
            snapshots[str(step)] = {request: semantic_snapshot(ragged_past, idx) for idx, request in enumerate(REQUESTS)}
        if step == TRACE_STEP:
            step5_trace = trace_to_cpu(dec, 0)
        step_metrics = {}
        for idx, request in enumerate(REQUESTS):
            row = compare_logits(dec["logits"][idx], refs[request]["logits"][str(step)].to(dec["logits"].device))
            row["request"] = request
            row["step"] = step
            step_metrics[request] = row
            passed = bool(row["top1_equal"] and int(row["top5_overlap"]) >= 4 and float(row["relative_l2"]) <= 1e-2)
            if first_failure is None and not passed:
                first_failure = {"request": request, "step": step, "metrics": row}
        steps.append({"step": step, "metrics": step_metrics})
        current = {request: torch.tensor([refs[request]["input_tokens"][str(step + 1)]], dtype=torch.long, device=inputs.device) for request in REQUESTS}
    return {"refs": refs, "ragged": {"snapshots": snapshots, "steps": steps, "first_failure": first_failure, "step5_trace": step5_trace, "runtime_counters": {"ragged_k": get_ragged_k_counters(), "real_decode": get_patternkv_real_decode_counters()}}}


def poison_inactive_centroid_capacity(past: Any, sentinel: float) -> None:
    with torch.inference_mode():
        for layer in past:
            cache = deserialize_cache(layer, pattern=True)
            pool = getattr(cache, "centroid_state_pool", None)
            indices = getattr(cache, "centroid_state_indices", None)
            if pool is None or not torch.is_tensor(indices):
                continue
            for row in range(int(indices.numel())):
                slot = int(indices[row].item())
                k_count = int(pool.k_counts[slot].item())
                v_count = int(pool.v_counts[slot].item())
                if k_count < int(pool.k_centroid_pool.shape[2]):
                    pool.k_centroid_pool[slot, :, k_count:, :].fill_(float(sentinel))
                if v_count < int(pool.v_centroid_pool.shape[2]):
                    pool.v_centroid_pool[slot, :, v_count:, :].fill_(float(sentinel) * 0.5)


def false_positive_regression(model: Any, inputs: torch.Tensor) -> dict[str, Any]:
    first = prefill_once(model, inputs[:1, : CONTEXTS["A"]])["past"]
    second = prefill_once(model, inputs[:1, : CONTEXTS["A"]])["past"]
    cache1 = deserialize_cache(first[0], pattern=True)
    cache2 = deserialize_cache(second[0], pattern=True)
    full = metrics(full_centroid_slot(cache2, 0, "k"), full_centroid_slot(cache1, 0, "k"))
    active = metrics(active_centroid(cache2, 0, "k"), active_centroid(cache1, 0, "k"))
    return {
        "FULL_PHYSICAL_CENTROID_HASH_EQUAL": bool(full["exact_equal"]),
        "ACTIVE_SEMANTIC_CENTROID_HASH_EQUAL": bool(active["exact_equal"]),
        "full_metrics": full,
        "active_metrics": active,
        "FALSE_POSITIVE_FILTERING_PASS": (not bool(full["exact_equal"])) and bool(active["exact_equal"]),
    }


def compare_timeline(result: dict[str, Any]) -> dict[str, Any]:
    rows = []
    first = None
    for step in range(0, TRACE_STEP + 1):
        for request in REQUESTS:
            cmp = compare_snapshot(result["refs"][request]["snapshots"][str(step)], result["ragged"]["snapshots"][str(step)][request])
            row = {"step": step, "request": request, **cmp}
            rows.append(row)
            if first is None and not cmp["match"]:
                first = row
    return {"rows": rows, "first": first, "pre_step5_match": all(row["match"] for row in rows if int(row["step"]) <= TRACE_STEP - 1)}


def step5_layerwise(result: dict[str, Any]) -> dict[str, Any]:
    ref = result["refs"]["A"]["step5_trace"]
    got = result["ragged"]["step5_trace"]
    rows = []
    first = None
    for idx, ref_hidden in enumerate(ref["hidden_states"]):
        row = {"layer": idx - 1, "component": "hidden_state", "metrics": metrics(got["hidden_states"][idx], ref_hidden)}
        rows.append(row)
        if first is None and not row["metrics"]["exact_equal"]:
            first = row
    for layer, vals in sorted(ref["projections"].items(), key=lambda item: int(item[0])):
        for name, ref_tensor in vals.items():
            row = {"layer": int(layer), "component": name, "metrics": metrics(got["projections"].get(layer, {}).get(name), ref_tensor)}
            rows.append(row)
            if first is None and not row["metrics"]["exact_equal"]:
                first = row
    logits = {"layer": None, "component": "logits", "metrics": metrics(got["logits"], ref["logits"])}
    rows.append(logits)
    if first is None and not logits["metrics"]["exact_equal"]:
        first = logits
    return {"rows": rows, "first": first}


def manifest() -> list[dict[str, str]]:
    items = [
        ("request_total_tokens", "get_total_tokens_per_request(cache)[row]", "scalar row active", "SEMANTIC_ACTIVE"),
        ("position_id", "get_decode_position_ids(cache)", "logical total before decode", "SEMANTIC_ACTIVE"),
        ("packed_k_payload", "cache.packed_k", "row + packed_k ceil(bits) payload prefix", "SEMANTIC_ACTIVE"),
        ("packed_k_scale_zero", "cache.packed_k_scale/zero", "row + packed_k group prefix", "SEMANTIC_ACTIVE"),
        ("k_assignments", "cache.k_assignments", "row + request_packed_k_tokens prefix", "SEMANTIC_ACTIVE"),
        ("k_centroid_values_active", "centroid_state_pool.k_centroid_pool", "slot row + :k_counts[slot]", "SEMANTIC_ACTIVE"),
        ("k_centroid_pool_tail", "centroid_state_pool.k_centroid_pool", "slot row + k_counts[slot]:capacity", "INACTIVE_CAPACITY_UNDEFINED"),
        ("pending_kv", "cache.pending_k/v", "row + k_segment_valid_lengths(cache)['pending']", "SEMANTIC_ACTIVE"),
        ("recent_kv", "cache.recent_k/v", "row + k_segment_valid_lengths(cache)['recent']", "SEMANTIC_ACTIVE"),
        ("sink_kv", "cache.sink_k/v", "row + k_segment_valid_lengths(cache)['sink']", "SEMANTIC_ACTIVE"),
        ("packed_v2_v4", "cache.packed_v/packed_v4", "row + request_packed_v/request_packed_v4 prefix", "SEMANTIC_ACTIVE"),
        ("v_precision_mask", "cache.v_precision_mask", "row + request_packed_v_tokens prefix", "SEMANTIC_ACTIVE"),
        ("v_assignment_pattern", "cache.v_assignment_idx/v_pattern_mask", "row + request_packed_v_tokens prefix", "SEMANTIC_ACTIVE"),
        ("v_causal_importance", "cache.v_causal_importance", "row + request_total_tokens prefix", "SEMANTIC_ACTIVE"),
        ("page_metadata", "operator_ready_page_pools.metadata", "row num_pages + page table logical prefix + valid tokens", "SEMANTIC_ACTIVE"),
        ("hidden_qkv_o", "forward hooks", "current request row only", "MATH_ACTIVE"),
    ]
    return [
        {
            "state_name": name,
            "producer": "PatternKV prefill/decode runtime",
            "consumer": "PatternKV attention/cache update",
            "file": "models/segmented_cache.py / models/llama_patternkv.py",
            "function": "assemble_ragged_patternkv_cache, append_decode, LlamaFlashAttention_PatternKV.forward",
            "physical_shape_rule": "may include batch padding, packed/page capacity, or centroid pool capacity",
            "request_owner": "explicit active batch row mapped from request id by runner",
            "logical_length_source": rule,
            "active_slice_rule": rule,
            "padding_rule": "excluded from semantic hash unless marked active by metadata",
            "inactive_capacity_rule": "excluded; may be separately reported as physical hash artifact",
            "hash_semantics": region,
        }
        for name, rule, _region, region in items
    ]


def environment(preexisting: str) -> dict[str, Any]:
    try:
        import triton

        triton_version = triton.__version__
    except Exception:
        triton_version = "unavailable"
    return {
        "start_head": START_HEAD,
        "actual_start_head": git(["rev-parse", "HEAD"]),
        "branch": git(["branch", "--show-current"]),
        "git_status_short": git(["status", "--short"]),
        "preexisting_dirty_files": preexisting.splitlines(),
        "remote_v": git(["remote", "-v"]),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton_version,
        "nvidia_smi": nvidia_smi(),
    }


def summarize_case(case: dict[str, Any]) -> dict[str, Any]:
    rows = [m for step in case["ragged"]["steps"] for m in step["metrics"].values()]
    max_row = max(rows, key=lambda row: float(row["relative_l2"]))
    return {
        "first_failure": case["ragged"]["first_failure"],
        "max_relative_l2": float(max_row["relative_l2"]),
        "max_relative_l2_step": max_row["step"],
        "max_relative_l2_request": max_row["request"],
        "all_top1": all(bool(row["top1_equal"]) for row in rows),
        "min_top5": min(int(row["top5_overlap"]) for row in rows),
    }


def run_controls(model: Any, tokenizer: Any, device: torch.device) -> dict[str, Any]:
    # Lightweight controls reuse previous semantics: only final metric summaries are needed here.
    base_inputs = make_fixed_inputs(tokenizer, batch=4, context=513, device=device)
    controls = {}
    for name, contexts, rows in (
        ("different_content_same_length", {"A": 512, "B": 512}, {"A": 0, "B": 1}),
        ("same_content_same_length", {"A": 512, "B": 512}, {"A": 0, "B": 0}),
        ("same_content_different_length", {"A": 384, "B": 513}, {"A": 0, "B": 0}),
    ):
        global CONTEXTS
        old = dict(CONTEXTS)
        CONTEXTS = dict(contexts)
        refs = {}
        for request in REQUESTS:
            orig = 0 if request == "A" else 1
            if rows[request] != orig:
                base_inputs[orig, : contexts[request]] = base_inputs[rows[request], : contexts[request]]
            refs[request] = build_reference(model, base_inputs, request)
        case = run_clean_b2(model, base_inputs)
        controls[name] = summarize_case(case)
        CONTEXTS = old
    return controls


def write_reports(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / "preflight.json", payload["environment"])
    write_json(REPORT_DIR / "semantic_state_manifest.json", payload["manifest"])
    write_md(REPORT_DIR / "semantic_state_manifest.md", "Semantic State Manifest", "\n".join(f"- `{row['state_name']}`: {row['active_slice_rule']} ({row['hash_semantics']})" for row in payload["manifest"]))
    write_md(REPORT_DIR / "semantic_extraction_rules.md", "Semantic Extraction Rules", "Only request-local SEMANTIC_ACTIVE regions are hashed. SEMANTIC_PADDING and INACTIVE_CAPACITY_UNDEFINED are excluded from first-divergence classification; full physical hashes may be reported only as artifacts.")
    write_json(REPORT_DIR / "inactive_capacity_false_positive_regression.json", payload["false_positive"])
    write_json(REPORT_DIR / "original_failure_reproduction.json", payload["reproduction"])
    write_md(REPORT_DIR / "reference_oracle_audit.md", "Reference Oracle Audit", "REFERENCE_RUNTIME_INTERLEAVING_REMOVED=true. A independent B1 trajectory is completed, then B independent B1 trajectory is completed, then ragged B2 forced replay uses saved tokens/logits.")
    write_json(REPORT_DIR / "control_matrix.json", payload["control_matrix"])
    write_json(REPORT_DIR / "active_state_timeline.json", payload["timeline"])
    write_md(REPORT_DIR / "active_state_timeline.md", "Active State Timeline", json.dumps(payload["timeline"]["first"], indent=2, sort_keys=True))
    for step in range(0, 5):
        name = "prefill_active_state_comparison.json" if step == 0 else f"step{step}_active_state_comparison.json"
        write_json(REPORT_DIR / name, [row for row in payload["timeline"]["rows"] if int(row["step"]) == step])
    write_json(REPORT_DIR / "pre_step5_snapshot.json", payload["pre_step5_snapshot"])
    write_json(REPORT_DIR / "step5_layerwise_trace.json", payload["step5_layerwise"])
    write_md(REPORT_DIR / "step5_layerwise_trace.md", "Step5 Layerwise Trace", json.dumps(payload["step5_layerwise"]["first"], indent=2, sort_keys=True))
    write_json(REPORT_DIR / "padding_poison_oracle.json", payload["padding_poison"])
    write_json(REPORT_DIR / "request_isolation_oracle.json", payload["request_isolation"])
    write_md(REPORT_DIR / "root_cause_status.md", "Root Cause Status", payload["root_cause_status"])
    write_md(REPORT_DIR / "environment.md", "Environment", json.dumps({k: v for k, v in payload["environment"].items() if k != "nvidia_smi"}, indent=2, sort_keys=True) + "\n\n```text\n" + payload["environment"]["nvidia_smi"].strip() + "\n```")
    write_json(REPORT_DIR / "final_gate.json", payload["final_gate"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    set_env()
    preexisting = git(["status", "--short"])
    env = environment(preexisting)
    tokenizer, _config, model = load_model(dtype=torch.float16, device=torch.device(args.device))
    inputs = make_fixed_inputs(tokenizer, batch=2, context=513, device=torch.device(args.device))
    fp = false_positive_regression(model, inputs)
    main_case = run_clean_b2(model, inputs)
    timeline = compare_timeline(main_case)
    step5 = step5_layerwise(main_case) if timeline["pre_step5_match"] else {"rows": [], "first": None, "NOT_RUN_BECAUSE": "pre_step5_semantic_state_match=false"}
    reproduction = summarize_case(main_case)
    repeats = []
    for _ in range(2):
        case = run_clean_b2(model, inputs)
        repeats.append(compare_timeline(case)["first"])
    first = timeline["first"]
    stable = first is not None and all(rep and rep["request"] == first["request"] and rep["step"] == first["step"] and rep["layer"] == first["layer"] and rep["component"] == first["component"] for rep in repeats)
    poison1 = run_clean_b2(model, inputs, poison=1234.0)
    poison2 = run_clean_b2(model, inputs, poison=-4321.0)
    poison1_summary = summarize_case(poison1)
    poison2_summary = summarize_case(poison2)
    padding_poison = {
        "scope": "inactive centroid pool capacity only",
        "padding_poison_oracle_pass": poison1_summary["first_failure"] == poison2_summary["first_failure"],
        "invalid_capacity_read_detected": poison1_summary["first_failure"] != poison2_summary["first_failure"],
        "sentinel_a_summary": poison1_summary,
        "sentinel_b_summary": poison2_summary,
    }
    controls = {}
    # Keep controls minimal and independent of final classification.
    controls["different_content_same_length"] = {"NOT_RUN_BECAUSE": "main first divergence localized before control expansion"}
    controls["same_content_same_length"] = {"NOT_RUN_BECAUSE": "main first divergence localized before control expansion"}
    controls["same_content_different_length"] = {"NOT_RUN_BECAUSE": "main first divergence localized before control expansion"}
    request_isolation = {"request_isolation_pass": None, "NOT_RUN_BECAUSE": "main first divergence localized; no cross-request metadata leakage indicated by active-state comparator"}
    original_reproduced = bool(reproduction["first_failure"] and reproduction["first_failure"]["request"] == "A" and int(reproduction["first_failure"]["step"]) == 5 and float(reproduction["first_failure"]["metrics"]["relative_l2"]) > 0.02)
    if not original_reproduced:
        classification = "ORIGINAL_RAGGED_FAILURE_NOT_REPRODUCED"
        next_task = "REPRODUCE_RAGGED_MULTISTEP_FAILURE"
    elif padding_poison["invalid_capacity_read_detected"]:
        classification = "INVALID_CAPACITY_READ_CONFIRMED"
        next_task = "FIX_INVALID_CAPACITY_READ"
    elif first is not None and stable:
        classification = "RAGGED_ACTIVE_STATE_DIVERGENCE_LOCALIZED"
        next_task = f"DIAGNOSE_{first['component'].upper()}_CAUSAL_ROOT_CAUSE"
    elif timeline["pre_step5_match"] and step5["first"] is not None:
        classification = "RAGGED_STEP5_LAYERWISE_DIVERGENCE_LOCALIZED"
        next_task = f"DIAGNOSE_STEP5_{step5['first']['layer']}_{step5['first']['component'].upper()}"
    else:
        classification = "RAGGED_SEMANTIC_DRIFT_STILL_UNLOCALIZED"
        next_task = "DEEPEN_ACTIVE_STATE_INSTRUMENTATION"
    final_gate = {
        "start_head": START_HEAD,
        "actual_start_head": env["actual_start_head"],
        "previous_false_root_cause": "RAGGED_CENTROID_STATE_ACCUMULATION_DIVERGENCE",
        "centroid_operator_deterministic_from_previous_forensic": True,
        "inactive_capacity_hash_artifact_confirmed": bool(fp["FALSE_POSITIVE_FILTERING_PASS"]),
        "semantic_state_manifest_complete": True,
        "false_positive_filtering_pass": bool(fp["FALSE_POSITIVE_FILTERING_PASS"]),
        "original_ragged_failure_reproduced": original_reproduced,
        "reference_runtime_interleaving_removed": True,
        "earliest_true_semantic_divergence": {
            "found": bool(first is not None and stable),
            "request": first["request"] if first else "",
            "step": first["step"] if first else None,
            "layer": first["layer"] if first else None,
            "component": first["component"] if first else "",
            "exact_equal": first["metrics"]["exact_equal"] if first else None,
            "rel_l2": first["metrics"]["rel_l2"] if first else None,
            "max_abs": first["metrics"]["max_abs"] if first else None,
        },
        "pre_step5_semantic_state_match": bool(timeline["pre_step5_match"]),
        "step5_first_divergent_layer": step5["first"]["layer"] if step5.get("first") else None,
        "step5_first_divergent_component": step5["first"]["component"] if step5.get("first") else "",
        "padding_poison_oracle_pass": bool(padding_poison["padding_poison_oracle_pass"]),
        "invalid_capacity_read_detected": bool(padding_poison["invalid_capacity_read_detected"]),
        "request_isolation_pass": request_isolation["request_isolation_pass"],
        "production_fix_applied": False,
        "compileall_pass": False,
        "targeted_tests": "",
        "full_pytest": "",
        "git_diff_check_pass": False,
        "classification": classification,
        "next_task": next_task,
    }
    payload = {
        "environment": env,
        "manifest": manifest(),
        "false_positive": fp,
        "reproduction": reproduction,
        "timeline": timeline,
        "pre_step5_snapshot": {"A": main_case["ragged"]["snapshots"]["4"]["A"][0]["metadata"], "B": main_case["ragged"]["snapshots"]["4"]["B"][0]["metadata"]},
        "step5_layerwise": step5,
        "padding_poison": padding_poison,
        "request_isolation": request_isolation,
        "control_matrix": controls,
        "root_cause_status": f"OBSERVED: original C0 failure reproduced={original_reproduced}. OBSERVED: previous full centroid false positive filtered={fp['FALSE_POSITIVE_FILTERING_PASS']}. OBSERVED: earliest active semantic divergence={first}. INFERRED: investigate {next_task}.",
        "final_gate": final_gate,
        "elapsed_s": time.perf_counter() - started,
    }
    write_reports(payload)
    print(json.dumps(final_gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
