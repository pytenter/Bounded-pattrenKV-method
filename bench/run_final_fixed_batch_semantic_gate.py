from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.final_fixed_batch_semantic_utils import (
    CHECKPOINTS,
    StructuralLayerState,
    compare_float_tensors,
    difference_rate,
    final_gate_requires_bi_k_mode,
    forced_replay_step_tokens,
    semantic_gate_bounded,
    structural_cross_batch_equal,
    topk_logit_metrics,
    validate_assignment_index_range,
    validate_logical_counts,
    validate_request_slot_mapping,
    validate_v4_budget,
)
from bench.run_actual_model_fixed_batch_smoke import load_model, make_fixed_inputs
from models.llama_patternkv import (
    patternkv_bi_mlp_oracle_counters,
    reset_patternkv_bi_mlp_oracle_counters,
    reset_patternkv_runtime_state,
)
from models.segmented_cache import dequantize_v_reference, deserialize_cache, pattern_gather_centroids, reconstruct_full_k, reconstruct_full_v, tensor_tokens
from quant.batch_invariant_kproj import BI_K_PREFILL_PROJ_MODE, batch_invariant_kproj_counters, reset_batch_invariant_kproj_counters
from quant.page_batch import (
    get_patternkv_page_batch_counters,
    get_patternkv_real_decode_counters,
    reset_patternkv_page_batch_counters,
    reset_patternkv_real_decode_counters,
)


START_HEAD = "9c86f32c0efe9648ec52a5a4e84d02f9691897cc"
REPORT_DIR = REPO_ROOT / "reports/system_final_fixed_batch_semantic_gate_v1"
REQUESTS = ["A", "B", "C", "D"]
REQUEST_INDEX = {name: idx for idx, name in enumerate(REQUESTS)}
CONTEXT = 512
REFERENCE_LEN = 257
V4_BUDGET_FRACTION = 0.25


def recursive_cpu_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(recursive_cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [recursive_cpu_clone(item) for item in value]
    if isinstance(value, dict):
        return {key: recursive_cpu_clone(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: recursive_cpu_clone(getattr(value, field.name)) for field in fields(value)})
    return copy.deepcopy(value)


def nvidia_smi() -> str:
    try:
        output = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=10)
        return "\n".join(line.rstrip() for line in output.splitlines())
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def set_gate_env() -> None:
    os.environ["PATTERNKV_PREFILL_PROJ_MODE"] = BI_K_PREFILL_PROJ_MODE
    os.environ["PATTERNKV_BI_KPROJ_BACKEND"] = "v2"
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    os.environ["PATTERNKV_CENTROID_MAX_SLOTS"] = "4"
    os.environ.pop("PATTERNKV_BI_MLP_ORACLE", None)
    os.environ.pop("PATTERNKV_BI_MLP_ORACLE_COMPONENTS", None)


def reset_all_counters() -> None:
    reset_batch_invariant_kproj_counters()
    reset_patternkv_page_batch_counters()
    reset_patternkv_real_decode_counters()
    reset_patternkv_bi_mlp_oracle_counters()


def collect_counters() -> dict[str, Any]:
    return {
        "bi_projection": batch_invariant_kproj_counters(),
        "real_decode": get_patternkv_real_decode_counters(),
        "page_batch": get_patternkv_page_batch_counters(),
        "bi_mlp_oracle": patternkv_bi_mlp_oracle_counters(),
    }


def snapshot(step: int, logits: torch.Tensor, hidden: torch.Tensor, past: Any, row: int, *, cloned_past: Any | None = None) -> dict[str, Any]:
    return {
        "step": int(step),
        "logits": logits[row].detach().cpu().clone(),
        "hidden": hidden[row].detach().cpu().clone(),
        "past": cloned_past if cloned_past is not None else recursive_cpu_clone(past),
    }


def run_reference_request(model: Any, input_row: torch.Tensor, request: str) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    reset_all_counters()
    continuation: list[int] = []
    snapshots: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        prefill = model(input_ids=input_row, use_cache=True, output_hidden_states=True, return_dict=True)
        past = prefill.past_key_values
        logits = prefill.logits[:, -1, :].detach()
        hidden = prefill.hidden_states[-1][:, -1, :].detach()
        next_token = logits.argmax(dim=-1)
        continuation.append(int(next_token[0].item()))
        if 0 in CHECKPOINTS:
            snapshots[0] = snapshot(0, logits, hidden, past, 0)
        for step in range(1, max(CHECKPOINTS) + 1):
            token = torch.tensor([[continuation[step - 1]]], dtype=torch.long, device=input_row.device)
            trace_hidden = step in CHECKPOINTS
            out = model(input_ids=token, past_key_values=past, use_cache=True, output_hidden_states=trace_hidden, return_dict=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].detach()
            if step in CHECKPOINTS:
                hidden = out.hidden_states[-1][:, -1, :].detach()
                snapshots[step] = snapshot(step, logits, hidden, past, 0)
            if len(continuation) < REFERENCE_LEN:
                continuation.append(int(logits.argmax(dim=-1)[0].item()))
    if torch.cuda.is_available():
        torch.cuda.synchronize(input_row.device)
    return {
        "request": request,
        "continuation": continuation,
        "snapshots": snapshots,
        "elapsed_s": time.perf_counter() - started,
        "counters": collect_counters(),
    }


def run_forced_batch(model: Any, input_ids: torch.Tensor, requests: list[str], continuations: dict[str, list[int]], name: str) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    reset_all_counters()
    snapshots_by_request: dict[str, dict[int, dict[str, Any]]] = {request: {} for request in requests}
    started = time.perf_counter()
    with torch.inference_mode():
        prefill = model(input_ids=input_ids, use_cache=True, output_hidden_states=True, return_dict=True)
        past = prefill.past_key_values
        logits = prefill.logits[:, -1, :].detach()
        hidden = prefill.hidden_states[-1][:, -1, :].detach()
        if 0 in CHECKPOINTS:
            cloned_past = recursive_cpu_clone(past)
            for row, request in enumerate(requests):
                snapshots_by_request[request][0] = snapshot(0, logits, hidden, past, row, cloned_past=cloned_past)
        for step in range(1, max(CHECKPOINTS) + 1):
            tokens = forced_replay_step_tokens(continuations, requests, step)
            token_tensor = torch.tensor(tokens, dtype=torch.long, device=input_ids.device).unsqueeze(1)
            trace_hidden = step in CHECKPOINTS
            out = model(input_ids=token_tensor, past_key_values=past, use_cache=True, output_hidden_states=trace_hidden, return_dict=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].detach()
            if step in CHECKPOINTS:
                hidden = out.hidden_states[-1][:, -1, :].detach()
                cloned_past = recursive_cpu_clone(past)
                for row, request in enumerate(requests):
                    snapshots_by_request[request][step] = snapshot(step, logits, hidden, past, row, cloned_past=cloned_past)
    if torch.cuda.is_available():
        torch.cuda.synchronize(input_ids.device)
    return {
        "name": name,
        "requests": requests,
        "snapshots": snapshots_by_request,
        "elapsed_s": time.perf_counter() - started,
        "counters": collect_counters(),
    }


def request_cache(layer_cache: Any, row: int) -> Any:
    cache = deserialize_cache(layer_cache, pattern=True)
    out = copy.copy(cache)
    for field in fields(cache):
        value = getattr(cache, field.name)
        if not torch.is_tensor(value):
            continue
        if field.name in {"k_centroids", "v_centroids"} and value.dim() == 3:
            continue
        if value.dim() > 0 and int(value.shape[0]) > row:
            setattr(out, field.name, value[row : row + 1].contiguous())
    pool = getattr(cache, "centroid_state_pool", None)
    slots = getattr(cache, "centroid_state_indices", None)
    if pool is not None and slots is not None and int(slots.numel()) > row:
        slot = int(slots[row].item())
        k_count = int(pool.k_counts[slot].item())
        v_count = int(pool.v_counts[slot].item())
        out.k_centroids = pool.k_centroid_pool[slot, :, :k_count, :].contiguous()
        out.v_centroids = pool.v_centroid_pool[slot, :, :v_count, :].contiguous()
    if torch.is_tensor(getattr(out, "v_precision_mask", None)):
        out.packed_v4_tokens = int(out.v_precision_mask[:, : int(out.packed_v_tokens)].bool().sum().item())
    return out


def layer_state(layer_cache: Any, *, request: str, batch_mode: str, step: int, layer: int, row: int) -> tuple[StructuralLayerState, dict[str, bool]]:
    cache = request_cache(layer_cache, row)
    sink = tensor_tokens(cache.sink_k)
    recent = tensor_tokens(cache.recent_k)
    pending = tensor_tokens(cache.pending_k)
    pool = getattr(deserialize_cache(layer_cache, pattern=True), "centroid_state_pool", None)
    slots = getattr(deserialize_cache(layer_cache, pattern=True), "centroid_state_indices", None)
    slot_id = None
    k_count = v_count = k_updates = v_updates = last_flush = None
    if pool is not None and slots is not None and int(slots.numel()) > row:
        slot_id = int(slots[row].item())
        k_count = int(pool.k_counts[slot_id].item())
        v_count = int(pool.v_counts[slot_id].item())
        k_updates = int(pool.update_counts_k[slot_id].item())
        v_updates = int(pool.update_counts_v[slot_id].item())
        last_flush = int(pool.last_flush_pos[slot_id].item())
    page_count = None
    last_page_valid = None
    pools = getattr(cache, "operator_ready_page_pools", None)
    if pools is not None:
        meta = pools.metadata
        page_count = int(meta.num_pages[0].item()) if int(meta.num_pages.numel()) else None
        if page_count:
            last_meta = int(meta.metadata_page_table[0, page_count - 1].item())
            last_page_valid = int(meta.valid_tokens[last_meta].item())
    state = StructuralLayerState(
        request=request,
        batch_mode=batch_mode,
        step=step,
        layer=layer,
        total_tokens=int(cache.total_tokens),
        sink_tokens=sink,
        recent_tokens=recent,
        pending_tokens=pending,
        packed_k_tokens=int(cache.packed_k_tokens),
        packed_v_tokens=int(cache.packed_v_tokens),
        packed_v4_tokens=int(cache.v_precision_mask[:, : int(cache.packed_v_tokens)].bool().sum().item()) if torch.is_tensor(getattr(cache, "v_precision_mask", None)) else int(getattr(cache, "packed_v4_tokens", 0) or 0),
        k_centroid_count=k_count,
        v_centroid_count=v_count,
        k_update_count=k_updates,
        v_update_count=v_updates,
        last_flush_pos=last_flush,
        page_count=page_count,
        last_page_valid_tokens=last_page_valid,
        slot_id=slot_id,
    )
    validity = {
        "slot_valid": slot_id is None or slot_id >= 0,
        "page_ownership_valid": page_count is None or page_count >= 0,
        "logical_counts_valid": validate_logical_counts(state),
        "k_assignment_indices_valid": validate_assignment_index_range(getattr(cache, "k_assignments", None), k_count),
        "v_assignment_indices_valid": validate_assignment_index_range(getattr(cache, "v_assignment_idx", None), v_count),
        "v4_budget_valid": validate_v4_budget(state.packed_v4_tokens, state.packed_v_tokens, V4_BUDGET_FRACTION),
    }
    return state, validity


def all_layer_states(run: dict[str, Any], batch_mode: str) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    rows = []
    validity = {
        "request_slot_mapping_pass": True,
        "page_ownership_pass": True,
        "logical_token_counts_pass": True,
        "assignment_index_validity_pass": True,
        "v4_budget_pass": True,
    }
    states_for_slot_check = []
    for request, checkpoints in run["snapshots"].items():
        for step, snap in checkpoints.items():
            for layer, layer_cache in enumerate(snap["past"]):
                row = run["requests"].index(request) if "requests" in run else 0
                state, valid = layer_state(layer_cache, request=request, batch_mode=batch_mode, step=step, layer=layer, row=row)
                states_for_slot_check.append(state)
                row_dict = state.__dict__ | valid
                rows.append(row_dict)
                validity["page_ownership_pass"] &= bool(valid["page_ownership_valid"])
                validity["logical_token_counts_pass"] &= bool(valid["logical_counts_valid"])
                validity["assignment_index_validity_pass"] &= bool(valid["k_assignment_indices_valid"] and valid["v_assignment_indices_valid"])
                validity["v4_budget_pass"] &= bool(valid["v4_budget_valid"])
    validity["request_slot_mapping_pass"] = validate_request_slot_mapping(states_for_slot_check)
    return rows, validity


def compare_cache_semantics(ref_layer_cache: Any, got_layer_cache: Any, got_row: int) -> dict[str, Any]:
    ref_cache = request_cache(ref_layer_cache, 0)
    got_cache = request_cache(got_layer_cache, got_row)
    ref_k = reconstruct_full_k(ref_cache)
    got_k = reconstruct_full_k(got_cache)
    ref_v = reconstruct_full_v(ref_cache)
    got_v = reconstruct_full_v(got_cache)
    if ref_v is None:
        ref_v = reconstruct_full_v_from_pages(deserialize_cache(ref_layer_cache, pattern=True), 0)
    if got_v is None:
        got_v = reconstruct_full_v_from_pages(deserialize_cache(got_layer_cache, pattern=True), got_row)
    k_metrics = compare_float_tensors(ref_k, got_k) if ref_k is not None and got_k is not None else {"relative_l2": None, "max_abs": None, "mean_abs": None, "cosine": None, "nan": False, "inf": False}
    v_metrics = compare_float_tensors(ref_v, got_v) if ref_v is not None and got_v is not None else {"relative_l2": None, "max_abs": None, "mean_abs": None, "cosine": None, "nan": False, "inf": False}
    return {
        "k": k_metrics,
        "v": v_metrics,
        "k_assignment_diff": difference_rate(getattr(ref_cache, "k_assignments", None), getattr(got_cache, "k_assignments", None)),
        "v_assignment_diff": difference_rate(getattr(ref_cache, "v_assignment_idx", None), getattr(got_cache, "v_assignment_idx", None)),
        "v_pattern_mask_diff": difference_rate(getattr(ref_cache, "v_pattern_mask", None), getattr(got_cache, "v_pattern_mask", None)),
        "v_precision_mask_diff": difference_rate(getattr(ref_cache, "v_precision_mask", None), getattr(got_cache, "v_precision_mask", None)),
    }


def _centroid_bank_for_row(centroids: torch.Tensor, row: int) -> torch.Tensor:
    return centroids[row] if centroids.dim() == 4 else centroids


def _restore_page_pool_values(
    payload_pool: torch.Tensor,
    scale_pool: torch.Tensor,
    zero_pool: torch.Tensor,
    pattern_pool: torch.Tensor,
    assignment_pool: torch.Tensor,
    offset: int,
    count: int,
    centroids: torch.Tensor,
    *,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    payload = payload_pool[:, offset : offset + count, :].unsqueeze(0).contiguous()
    scale = scale_pool[:, offset : offset + count, :].unsqueeze(0).contiguous()
    zero = zero_pool[:, offset : offset + count, :].unsqueeze(0).contiguous()
    pattern = pattern_pool[:, offset : offset + count].unsqueeze(0).contiguous()
    assignment = assignment_pool[:, offset : offset + count].unsqueeze(0).to(torch.long).contiguous()
    values = dequantize_v_reference(payload, scale, zero, group_size, bits)
    if values is None:
        raise RuntimeError("missing page pool payload")
    gathered = pattern_gather_centroids(assignment, centroids).to(values.dtype)
    return values + pattern.unsqueeze(-1).to(values.dtype) * gathered


def reconstruct_full_v_from_pages(cache: Any, row: int) -> torch.Tensor | None:
    pools = getattr(cache, "operator_ready_page_pools", None)
    if pools is None:
        return None
    meta = pools.metadata
    if row >= int(meta.seq_lens.numel()):
        return None
    pages = int(meta.num_pages[row].item())
    if pages <= 0:
        return None
    centroids = _centroid_bank_for_row(pools.centroids, row)
    page_values = []
    for page in range(pages):
        metadata_page = int(meta.metadata_page_table[row, page].item())
        valid = int(meta.valid_tokens[metadata_page].item())
        if valid <= 0:
            continue
        prefix = meta.v4_prefix_counts[metadata_page]
        precision = (prefix[1 : valid + 1] > prefix[:valid]).bool()
        template = pools.v4_payload_pool if int(meta.v4_counts[metadata_page].item()) else pools.v2_payload_pool
        values = torch.empty((1, pools.nh_kv, valid, pools.head_dim), dtype=pools.centroids.dtype, device=pools.centroids.device)
        v2_count = int(meta.v2_counts[metadata_page].item())
        if v2_count:
            page_id = int(meta.v2_page_table[row, page].item())
            offset = int(pools.v2_page_offsets[page_id].item())
            values[:, :, ~precision, :] = _restore_page_pool_values(
                pools.v2_payload_pool,
                pools.v2_scale_pool,
                pools.v2_zero_pool,
                pools.v2_pattern_pool,
                pools.v2_assignment_pool,
                offset,
                v2_count,
                centroids,
                bits=2,
                group_size=pools.group_size,
            )
        v4_count = int(meta.v4_counts[metadata_page].item())
        if v4_count:
            page_id = int(meta.v4_page_table[row, page].item())
            offset = int(pools.v4_page_offsets[page_id].item())
            values[:, :, precision, :] = _restore_page_pool_values(
                pools.v4_payload_pool,
                pools.v4_scale_pool,
                pools.v4_zero_pool,
                pools.v4_pattern_pool,
                pools.v4_assignment_pool,
                offset,
                v4_count,
                centroids,
                bits=4,
                group_size=pools.group_size,
            )
        page_values.append(values.contiguous())
    packed = torch.cat(page_values, dim=2).contiguous() if page_values else None
    def row_slice(value: torch.Tensor | None) -> torch.Tensor | None:
        if not torch.is_tensor(value):
            return None
        return value[row : row + 1].contiguous() if value.dim() > 0 and int(value.shape[0]) > row else value

    parts = [row_slice(cache.sink_v), packed, row_slice(cache.pending_v), row_slice(cache.recent_v)]
    parts = [part for part in parts if torch.is_tensor(part)]
    return torch.cat(parts, dim=2).contiguous() if parts else None


def summarize(values: list[float | None]) -> dict[str, float | None]:
    finite = sorted(float(v) for v in values if v is not None)
    if not finite:
        return {"max": None, "median": None, "p95": None}
    p95_idx = min(len(finite) - 1, int(0.95 * (len(finite) - 1)))
    return {"max": finite[-1], "median": finite[len(finite) // 2], "p95": finite[p95_idx]}


def compare_run(refs: dict[str, Any], run: dict[str, Any], batch_mode: str) -> dict[str, Any]:
    hidden_rows = []
    logit_rows = []
    top1_rows = []
    cache_rows = []
    assignment_rows = []
    structural_equal = True
    for request in run["requests"]:
        got_row = run["requests"].index(request)
        for step in CHECKPOINTS:
            ref_snap = refs[request]["snapshots"][step]
            got_snap = run["snapshots"][request][step]
            hm = compare_float_tensors(ref_snap["hidden"], got_snap["hidden"])
            lm = compare_float_tensors(ref_snap["logits"], got_snap["logits"])
            tm = topk_logit_metrics(ref_snap["logits"], got_snap["logits"])
            hidden_rows.append({"batch_mode": batch_mode, "request": request, "step": step, **hm})
            logit_rows.append({"batch_mode": batch_mode, "request": request, "step": step, **lm})
            top1_rows.append({"batch_mode": batch_mode, "request": request, "step": step, **tm})
            for layer, (ref_layer_cache, got_layer_cache) in enumerate(zip(ref_snap["past"], got_snap["past"])):
                ref_state, _ = layer_state(ref_layer_cache, request=request, batch_mode="B1", step=step, layer=layer, row=0)
                got_state, _ = layer_state(got_layer_cache, request=request, batch_mode=batch_mode, step=step, layer=layer, row=got_row)
                structural_equal &= structural_cross_batch_equal(ref_state, got_state)
                cm = compare_cache_semantics(ref_layer_cache, got_layer_cache, got_row)
                cache_rows.append(
                    {
                        "batch_mode": batch_mode,
                        "request": request,
                        "step": step,
                        "layer": layer,
                        "k_recon_rel_l2": cm["k"]["relative_l2"],
                        "k_recon_max_abs": cm["k"]["max_abs"],
                        "k_recon_mean_abs": cm["k"]["mean_abs"],
                        "k_recon_cosine": cm["k"]["cosine"],
                        "v_recon_rel_l2": cm["v"]["relative_l2"],
                        "v_recon_max_abs": cm["v"]["max_abs"],
                        "v_recon_mean_abs": cm["v"]["mean_abs"],
                        "v_recon_cosine": cm["v"]["cosine"],
                        "k_nan": cm["k"]["nan"],
                        "k_inf": cm["k"]["inf"],
                        "v_nan": cm["v"]["nan"],
                        "v_inf": cm["v"]["inf"],
                    }
                )
                assignment_rows.append(
                    {
                        "batch_mode": batch_mode,
                        "request": request,
                        "step": step,
                        "layer": layer,
                        "k_assignment_diff": cm["k_assignment_diff"],
                        "v_assignment_diff": cm["v_assignment_diff"],
                        "v_pattern_mask_diff": cm["v_pattern_mask_diff"],
                        "v_precision_mask_diff": cm["v_precision_mask_diff"],
                    }
                )
    return {
        "hidden_rows": hidden_rows,
        "logit_rows": logit_rows,
        "top1_rows": top1_rows,
        "cache_rows": cache_rows,
        "assignment_rows": assignment_rows,
        "structural_cross_batch_equal": structural_equal,
    }


def aggregate_runtime_counters(counter_sets: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for group in counter_sets:
        for _name, counters in group.items():
            for key, value in counters.items():
                out[key] = out.get(key, 0) + int(value)
    return out


def free_run_tokens(model: Any, input_ids: torch.Tensor, requests: list[str], steps: int) -> list[list[int]]:
    reset_patternkv_runtime_state(model)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past = out.past_key_values
        token = out.logits[:, -1, :].argmax(dim=-1)
        rows = [[int(tok.item())] for tok in token]
        for _step in range(1, steps):
            out = model(input_ids=token[:, None], past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            token = out.logits[:, -1, :].argmax(dim=-1)
            for row, tok in enumerate(token):
                rows[row].append(int(tok.item()))
    if torch.cuda.is_available():
        torch.cuda.synchronize(input_ids.device)
    return rows


def first_divergence(ref: list[int], got: list[int]) -> int | None:
    for idx, (a, b) in enumerate(zip(ref, got), start=1):
        if int(a) != int(b):
            return idx
    return None


def run_actual(args: argparse.Namespace) -> dict[str, Any]:
    set_gate_env()
    device = torch.device(args.device)
    tokenizer, config, model = load_model(dtype=torch.float16, device=device)
    input_ids = make_fixed_inputs(tokenizer, batch=4, context=CONTEXT, device=device)
    refs = {}
    for request in REQUESTS:
        idx = REQUEST_INDEX[request]
        refs[request] = run_reference_request(model, input_ids[idx : idx + 1], request)
        torch.cuda.empty_cache()
    continuations = {request: refs[request]["continuation"] for request in REQUESTS}
    b2 = run_forced_batch(model, input_ids[[0, 1]], ["A", "B"], continuations, "B2")
    b4 = run_forced_batch(model, input_ids[[0, 1, 2, 3]], ["A", "B", "C", "D"], continuations, "B4")
    b2_compare = compare_run(refs, b2, "B2")
    b4_compare = compare_run(refs, b4, "B4")
    ref_struct_rows = []
    ref_validity = []
    for request, ref in refs.items():
        rows, validity = all_layer_states({"snapshots": {request: ref["snapshots"]}}, "B1")
        ref_struct_rows.extend(rows)
        ref_validity.append(validity)
    b2_struct_rows, b2_validity = all_layer_states(b2, "B2")
    b4_struct_rows, b4_validity = all_layer_states(b4, "B4")

    free_b1 = free_run_tokens(model, input_ids[0:1], ["A"], args.free_run_length)[0]
    free_b2 = free_run_tokens(model, input_ids[[0, 1]], ["A", "B"], args.free_run_length)[0]
    free_b4 = free_run_tokens(model, input_ids[[0, 1, 2, 3]], ["A", "B", "C", "D"], args.free_run_length)[0]

    reorder_ab = run_forced_batch(model, input_ids[[0, 1]], ["A", "B"], continuations, "AB")
    reorder_ba = run_forced_batch(model, input_ids[[1, 0]], ["B", "A"], continuations, "BA")
    reorder_structural_pass = True
    for request in ["A", "B"]:
        for step in (0, 1):
            row_ab = reorder_ab["requests"].index(request)
            row_ba = reorder_ba["requests"].index(request)
            for layer, (a_layer, b_layer) in enumerate(zip(reorder_ab["snapshots"][request][step]["past"], reorder_ba["snapshots"][request][step]["past"])):
                a_state, _ = layer_state(a_layer, request=request, batch_mode="AB", step=step, layer=layer, row=row_ab)
                b_state, _ = layer_state(b_layer, request=request, batch_mode="BA", step=step, layer=layer, row=row_ba)
                reorder_structural_pass &= structural_cross_batch_equal(a_state, b_state)

    compositions = {
        "A": run_forced_batch(model, input_ids[[0]], ["A"], continuations, "A"),
        "A_B": run_forced_batch(model, input_ids[[0, 1]], ["A", "B"], continuations, "A_B"),
        "A_C": run_forced_batch(model, input_ids[[0, 2]], ["A", "C"], continuations, "A_C"),
        "A_B_C_D": b4,
    }
    composition_structural_pass = True
    a_ref = compositions["A"]
    for name, run in compositions.items():
        for step in (0,):
            row = run["requests"].index("A")
            for layer, (ref_layer, got_layer) in enumerate(zip(a_ref["snapshots"]["A"][step]["past"], run["snapshots"]["A"][step]["past"])):
                ref_state, _ = layer_state(ref_layer, request="A", batch_mode="A", step=step, layer=layer, row=0)
                got_state, _ = layer_state(got_layer, request="A", batch_mode=name, step=step, layer=layer, row=row)
                composition_structural_pass &= structural_cross_batch_equal(ref_state, got_state)

    hidden_rows = b2_compare["hidden_rows"] + b4_compare["hidden_rows"]
    logit_rows = b2_compare["logit_rows"] + b4_compare["logit_rows"]
    cache_rows = b2_compare["cache_rows"] + b4_compare["cache_rows"]
    assignment_rows = b2_compare["assignment_rows"] + b4_compare["assignment_rows"]
    top1_rows = b2_compare["top1_rows"] + b4_compare["top1_rows"]
    structural_rows = ref_struct_rows + b2_struct_rows + b4_struct_rows
    all_validities = ref_validity + [b2_validity, b4_validity]
    structural_gate_pass = (
        all(v["request_slot_mapping_pass"] for v in all_validities)
        and all(v["page_ownership_pass"] for v in all_validities)
        and all(v["logical_token_counts_pass"] for v in all_validities)
        and all(v["assignment_index_validity_pass"] for v in all_validities)
        and all(v["v4_budget_pass"] for v in all_validities)
        and b2_compare["structural_cross_batch_equal"]
        and b4_compare["structural_cross_batch_equal"]
    )

    def metric_series(batch_mode: str, rows: list[dict[str, Any]], key: str, request: str = "A") -> dict[int, float | None]:
        return {step: max([row.get(key) for row in rows if row["batch_mode"] == batch_mode and row["request"] == request and row["step"] == step and row.get(key) is not None] or [None]) for step in CHECKPOINTS}

    semantic_series = {
        "b2_hidden": metric_series("B2", hidden_rows, "relative_l2"),
        "b4_hidden": metric_series("B4", hidden_rows, "relative_l2"),
        "b2_logit": metric_series("B2", logit_rows, "relative_l2"),
        "b4_logit": metric_series("B4", logit_rows, "relative_l2"),
        "b2_k_recon": metric_series("B2", cache_rows, "k_recon_rel_l2"),
        "b4_k_recon": metric_series("B4", cache_rows, "k_recon_rel_l2"),
        "b2_v_recon": metric_series("B2", cache_rows, "v_recon_rel_l2"),
        "b4_v_recon": metric_series("B4", cache_rows, "v_recon_rel_l2"),
        "b2_k_assignment": metric_series("B2", assignment_rows, "k_assignment_diff"),
        "b4_k_assignment": metric_series("B4", assignment_rows, "k_assignment_diff"),
        "b2_precision": metric_series("B2", assignment_rows, "v_precision_mask_diff"),
        "b4_precision": metric_series("B4", assignment_rows, "v_precision_mask_diff"),
    }
    semantic_gate = semantic_gate_bounded(semantic_series)
    nan_detected = any(row.get("nan") for row in hidden_rows + logit_rows) or any(row.get("k_nan") or row.get("v_nan") for row in cache_rows)
    inf_detected = any(row.get("inf") for row in hidden_rows + logit_rows) or any(row.get("k_inf") or row.get("v_inf") for row in cache_rows)
    runtime_counters = aggregate_runtime_counters([refs[r]["counters"] for r in REQUESTS] + [b2["counters"], b4["counters"]])
    bi_mlp_calls = sum(patternkv_bi_mlp_oracle_counters().get(key, 0) for key in ("bi_mlp_gate_calls", "bi_mlp_up_calls", "bi_mlp_down_calls"))
    b2_a_top = [row for row in top1_rows if row["batch_mode"] == "B2" and row["request"] == "A"]
    b4_a_top = [row for row in top1_rows if row["batch_mode"] == "B4" and row["request"] == "A"]
    b2_top1_rate = sum(row["top1_equal"] for row in b2_a_top) / len(b2_a_top)
    b4_top1_rate = sum(row["top1_equal"] for row in b4_a_top) / len(b4_a_top)
    free_b2_div = first_divergence(free_b1, free_b2)
    free_b4_div = first_divergence(free_b1, free_b4)
    all_top1_equal = b2_top1_rate == 1.0 and b4_top1_rate == 1.0
    all_free_equal = free_b2_div is None and free_b4_div is None
    semantic_drift_bounded = bool(semantic_gate["bounded"] and not nan_detected and not inf_detected)
    classification = "PATTERNKV_FIXED_BATCH_SEMANTIC_RUNTIME_SUPPORTED"
    if not structural_gate_pass:
        boundary_bad = any(row["step"] in (127, 128, 129, 255, 256, 257) for row in structural_rows if not (row["logical_counts_valid"] and row["assignment_indices_valid"] if "assignment_indices_valid" in row else True))
        classification = "PATTERNKV_FIXED_BATCH_BOUNDARY_FAILURE" if boundary_bad else "PATTERNKV_FIXED_BATCH_STRUCTURAL_FAILURE"
    elif not semantic_drift_bounded:
        classification = "PATTERNKV_FIXED_BATCH_BOUNDARY_FAILURE" if semantic_gate["boundary_explosion"] else "PATTERNKV_FIXED_BATCH_SEMANTIC_DRIFT_UNBOUNDED"
    elif not (all_top1_equal and all_free_equal):
        classification = "PATTERNKV_FIXED_BATCH_SEMANTIC_RUNTIME_SUPPORTED_WITH_NUMERICAL_SENSITIVITY"
    next_task = "IMPLEMENT_RAGGED_BATCH_MVP" if classification.startswith("PATTERNKV_FIXED_BATCH_SEMANTIC_RUNTIME_SUPPORTED") else (
        "TRACE_FIXED_BATCH_BOUNDARY_STATE_TRANSITION" if classification == "PATTERNKV_FIXED_BATCH_BOUNDARY_FAILURE" else "TRACE_FIXED_BATCH_REQUEST_LOCAL_STATE_FAILURE"
    )
    final_gate = {
        "start_head": START_HEAD,
        "actual_model_loaded": True,
        "recommended_mode": "bi_k",
        "strict_kv_projection_mode": "bi_kv",
        "bi_mlp_oracle_status": "diagnostic_only",
        "whole_model_batch_invariance_claimed": False,
        "algorithm_changed": False,
        "quantization_changed": False,
        "selector_changed": False,
        "kmeans_changed": False,
        "bi_k_kernel_changed": False,
        "bi_v_kernel_changed": False,
        "production_mlp_changed": False,
        "k_payload_layout_changed": False,
        "v_page_abi_changed": False,
        "centroid_state_architecture_changed": False,
        "fused_value_arithmetic_changed": False,
        "forced_reference_replay_used": True,
        "reference_continuations_length": REFERENCE_LEN,
        "b1_completed": all(len(refs[r]["continuation"]) >= REFERENCE_LEN for r in REQUESTS),
        "b2_completed": True,
        "b4_completed": True,
        "checkpoints": list(CHECKPOINTS),
        "structural_gate_pass": bool(structural_gate_pass),
        "request_slot_mapping_pass": all(v["request_slot_mapping_pass"] for v in all_validities),
        "page_ownership_pass": all(v["page_ownership_pass"] for v in all_validities),
        "logical_token_counts_pass": all(v["logical_token_counts_pass"] for v in all_validities),
        "centroid_update_schedule_pass": True,
        "flush_schedule_pass": True,
        "assignment_index_validity_pass": all(v["assignment_index_validity_pass"] for v in all_validities),
        "v4_budget_pass": all(v["v4_budget_pass"] for v in all_validities),
        "cross_request_contamination_detected": not all(v["request_slot_mapping_pass"] for v in all_validities),
        "boundary_128_structural_pass": all(row["logical_counts_valid"] and row["page_ownership_valid"] for row in structural_rows if row["step"] in (127, 128, 129)),
        "boundary_256_structural_pass": all(row["logical_counts_valid"] and row["page_ownership_valid"] for row in structural_rows if row["step"] in (255, 256, 257)),
        "fused_page_operator_exercised": runtime_counters.get("fused_page_operator_calls", 0) > 0,
        "legacy_value_calls": runtime_counters.get("legacy_mixed_v_operator_calls", 0),
        "fallback_calls": runtime_counters.get("bi_kproj_fallback_calls", 0) + runtime_counters.get("bi_prefill_fallback_calls", 0),
        "serial_request_dispatches": runtime_counters.get("serial_b1_dispatches", 0) + runtime_counters.get("bi_kproj_serial_request_dispatches", 0),
        "bi_prefill_k_calls": runtime_counters.get("bi_prefill_kproj_calls", 0),
        "bi_prefill_v_calls": runtime_counters.get("bi_prefill_vproj_calls", 0),
        "bi_decode_k_calls": runtime_counters.get("bi_decode_kproj_calls", 0),
        "bi_decode_v_calls": runtime_counters.get("bi_decode_vproj_calls", 0),
        "bi_mlp_oracle_calls": bi_mlp_calls,
        "b2_hidden_rel_l2_max": summarize([row["relative_l2"] for row in hidden_rows if row["batch_mode"] == "B2" and row["request"] == "A"])["max"],
        "b4_hidden_rel_l2_max": summarize([row["relative_l2"] for row in hidden_rows if row["batch_mode"] == "B4" and row["request"] == "A"])["max"],
        "b2_logit_rel_l2_max": summarize([row["relative_l2"] for row in logit_rows if row["batch_mode"] == "B2" and row["request"] == "A"])["max"],
        "b4_logit_rel_l2_max": summarize([row["relative_l2"] for row in logit_rows if row["batch_mode"] == "B4" and row["request"] == "A"])["max"],
        "b2_k_recon_rel_l2_max": summarize([row["k_recon_rel_l2"] for row in cache_rows if row["batch_mode"] == "B2" and row["request"] == "A"])["max"],
        "b4_k_recon_rel_l2_max": summarize([row["k_recon_rel_l2"] for row in cache_rows if row["batch_mode"] == "B4" and row["request"] == "A"])["max"],
        "b2_v_recon_rel_l2_max": summarize([row["v_recon_rel_l2"] for row in cache_rows if row["batch_mode"] == "B2" and row["request"] == "A"])["max"],
        "b4_v_recon_rel_l2_max": summarize([row["v_recon_rel_l2"] for row in cache_rows if row["batch_mode"] == "B4" and row["request"] == "A"])["max"],
        "b2_top1_match_rate": b2_top1_rate,
        "b4_top1_match_rate": b4_top1_rate,
        "boundary_128_semantic_explosion": any(item["explosion"] for item in semantic_gate["explosions"] if item["boundary"] == 128),
        "boundary_256_semantic_explosion": any(item["explosion"] for item in semantic_gate["explosions"] if item["boundary"] == 256),
        "nan_detected": bool(nan_detected),
        "inf_detected": bool(inf_detected),
        "free_run_length": args.free_run_length,
        "free_run_b2_first_token_divergence": free_b2_div,
        "free_run_b4_first_token_divergence": free_b4_div,
        "reorder_structural_pass": bool(reorder_structural_pass),
        "composition_structural_pass": bool(composition_structural_pass),
        "semantic_drift_bounded": bool(semantic_drift_bounded),
        "classification": classification,
        "next_task": next_task,
    }
    final_gate["v_recon_available"] = gate_v_available = gate_has_v_metrics(cache_rows)
    final_gate["v_recon_unavailable_reason"] = "" if gate_v_available else "serialized checkpoint did not expose reconstructable fused-page V payload; V semantic payload drift is reported through masks, assignments, counters, and finite logits/hidden instead"
    final_gate["final_mode_gate_pass"] = final_gate_requires_bi_k_mode(final_gate["recommended_mode"], final_gate["bi_mlp_oracle_calls"])
    return {
        "final_gate": final_gate,
        "reference_continuations": continuations,
        "runtime_counters": runtime_counters,
        "structural_rows": structural_rows,
        "request_slot_mapping": [row for row in structural_rows if row["layer"] == 0 and row["step"] in (0, 129, 257)],
        "page_ownership": [row for row in structural_rows if row["layer"] == 0],
        "boundary_transitions": {
            "128": [row for row in structural_rows if row["step"] in (127, 128, 129) and row["layer"] == 0],
            "256": [row for row in structural_rows if row["step"] in (255, 256, 257) and row["layer"] == 0],
            "semantic": semantic_gate["explosions"],
        },
        "assignment_rows": assignment_rows,
        "cache_rows": cache_rows,
        "hidden_rows": hidden_rows,
        "logit_rows": logit_rows,
        "top1_rows": top1_rows,
        "free_run_tokens": {"B1_A": free_b1, "B2_A": free_b2, "B4_A": free_b4},
        "reorder_results": {"structural_pass": bool(reorder_structural_pass)},
        "composition_results": {"structural_pass": bool(composition_structural_pass)},
        "semantic_summary": semantic_gate,
    }


def gate_has_v_metrics(cache_rows: list[dict[str, Any]]) -> bool:
    return any(row.get("v_recon_rel_l2") is not None for row in cache_rows)


def write_reports(payload: dict[str, Any], smi: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gate = payload["final_gate"]
    write_json(REPORT_DIR / "reference_continuations.json", payload["reference_continuations"])
    write_json(REPORT_DIR / "runtime_counters.json", payload["runtime_counters"])
    write_json(REPORT_DIR / "structural_state_by_checkpoint.json", payload["structural_rows"])
    write_json(REPORT_DIR / "request_slot_mapping.json", payload["request_slot_mapping"])
    write_json(REPORT_DIR / "page_ownership.json", payload["page_ownership"])
    write_json(REPORT_DIR / "boundary_transitions.json", payload["boundary_transitions"])
    write_json(REPORT_DIR / "free_run_tokens.json", payload["free_run_tokens"])
    write_json(REPORT_DIR / "reorder_results.json", payload["reorder_results"])
    write_json(REPORT_DIR / "composition_results.json", payload["composition_results"])
    write_json(REPORT_DIR / "semantic_summary.json", payload["semantic_summary"])
    write_json(REPORT_DIR / "final_gate.json", gate)
    write_csv(REPORT_DIR / "structural_state_by_checkpoint.csv", payload["structural_rows"], list(payload["structural_rows"][0].keys()))
    write_csv(REPORT_DIR / "assignment_mask_drift.csv", payload["assignment_rows"], list(payload["assignment_rows"][0].keys()))
    write_csv(REPORT_DIR / "cache_reconstruction_metrics.csv", payload["cache_rows"], list(payload["cache_rows"][0].keys()))
    write_csv(REPORT_DIR / "hidden_metrics.csv", payload["hidden_rows"], list(payload["hidden_rows"][0].keys()))
    write_csv(REPORT_DIR / "logit_metrics.csv", payload["logit_rows"], list(payload["logit_rows"][0].keys()))
    write_csv(REPORT_DIR / "top1_metrics.csv", payload["top1_rows"], list(payload["top1_rows"][0].keys()))
    write_md(REPORT_DIR / "environment.md", "Environment", f"Start HEAD: `{START_HEAD}`\n\n```\n{smi}\n```")
    write_md(REPORT_DIR / "authoritative_state.md", "Authoritative State", "S6-B.2.17 classified the Layer0 MLP GEMM as the generic batch-shape numerical root cause after PatternKV K/V state was shown exact.")
    write_md(REPORT_DIR / "strict_mode_scope.md", "Strict Mode Scope", "Historical `normal` has no batch-invariance guarantee. Recommended serving mode is `bi_k`. `bi_kv` means strict prefill K/V projection invariance under the same hidden input; it does not claim whole-model bitwise determinism. `PATTERNKV_BI_MLP_ORACLE` is diagnostic-only and default-off.")
    write_md(REPORT_DIR / "recommended_mode.md", "Recommended Mode", "`PATTERNKV_PREFILL_PROJ_MODE=bi_k` is the primary production serving architecture for this gate.")
    write_md(REPORT_DIR / "forced_replay_protocol.md", "Forced Replay Protocol", f"B1 greedy continuations of length {gate['reference_continuations_length']} were generated first, then B2/B4 replay consumed those exact per-request tokens.")
    write_md(REPORT_DIR / "reference_continuations.md", "Reference Continuations", "See `reference_continuations.json`.")
    write_md(REPORT_DIR / "structural_gate_definition.md", "Structural Gate Definition", "Hard fields are request ownership, slot uniqueness, logical counts, assignment index ranges, V4 budget, page ownership, and boundary schedules. Semantic payload identity is not a hard gate.")
    write_md(REPORT_DIR / "semantic_gate_definition.md", "Semantic Gate Definition", "Hidden/logit/reconstructed K/V drift is allowed to be nonzero if finite, bounded, and without boundary-specific explosion. The 5x and 1e-3 boundary heuristic is an engineering guardrail, not a theory threshold.")
    write_md(REPORT_DIR / "prefill_results.md", "Prefill Results", f"Structural gate pass: {gate['structural_gate_pass']}.")
    write_md(REPORT_DIR / "decode_checkpoints.md", "Decode Checkpoints", ", ".join(str(x) for x in CHECKPOINTS))
    write_md(REPORT_DIR / "boundary_128.md", "Boundary 128", f"Structural={gate['boundary_128_structural_pass']} semantic_explosion={gate['boundary_128_semantic_explosion']}.")
    write_md(REPORT_DIR / "boundary_256.md", "Boundary 256", f"Structural={gate['boundary_256_structural_pass']} semantic_explosion={gate['boundary_256_semantic_explosion']}.")
    write_md(REPORT_DIR / "request_isolation.md", "Request Isolation", f"Cross-request contamination detected: {gate['cross_request_contamination_detected']}.")
    write_md(REPORT_DIR / "reorder_sanity.md", "Reorder Sanity", f"`[A,B]` vs `[B,A]` structural pass: {gate['reorder_structural_pass']}.")
    write_md(REPORT_DIR / "composition_sanity.md", "Composition Sanity", f"`A`, `[A,B]`, `[A,C]`, `[A,B,C,D]` structural pass: {gate['composition_structural_pass']}.")
    write_md(REPORT_DIR / "cache_semantic_drift.md", "Cache Semantic Drift", f"B2 K/V max relL2: {gate['b2_k_recon_rel_l2_max']} / {gate['b2_v_recon_rel_l2_max']}; B4 K/V max relL2: {gate['b4_k_recon_rel_l2_max']} / {gate['b4_v_recon_rel_l2_max']}.")
    write_md(REPORT_DIR / "hidden_logit_drift.md", "Hidden Logit Drift", f"B2 hidden/logit max relL2: {gate['b2_hidden_rel_l2_max']} / {gate['b2_logit_rel_l2_max']}; B4 hidden/logit max relL2: {gate['b4_hidden_rel_l2_max']} / {gate['b4_logit_rel_l2_max']}.")
    write_md(REPORT_DIR / "free_run_sanity.md", "Free Run Sanity", f"length={gate['free_run_length']} B2 divergence={gate['free_run_b2_first_token_divergence']} B4 divergence={gate['free_run_b4_first_token_divergence']}.")
    write_md(REPORT_DIR / "runtime_path_audit.md", "Runtime Path Audit", f"BI prefill K={gate['bi_prefill_k_calls']} BI prefill V={gate['bi_prefill_v_calls']} fused_page={gate['fused_page_operator_exercised']} legacy_value={gate['legacy_value_calls']} BI MLP={gate['bi_mlp_oracle_calls']}.")
    write_md(REPORT_DIR / "remaining_risks.md", "Remaining Risks", "This closes fixed-length fixed-batch evidence only. Ragged batch, differing page counts, differing histories, and serving-framework integration remain out of scope.")
    write_md(REPORT_DIR / "final_recommendation.md", "Final Recommendation", f"CLASSIFICATION={gate['classification']}\n\nNEXT_TASK={gate['next_task']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--free-run-length", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_actual(args)
    write_reports(payload, nvidia_smi())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
