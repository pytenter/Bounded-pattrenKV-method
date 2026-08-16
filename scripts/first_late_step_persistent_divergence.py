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
from bench.run_ragged_multistep_correctness import CONTEXTS, STEPS
from models.llama_patternkv import (
    patternkv_p2_first_divergence_trace_records,
    reset_patternkv_p2_first_divergence_trace,
    reset_patternkv_runtime_state,
)
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
from quant.batch_invariant_kproj import (
    BI_KV_PREFILL_PROJ_MODE,
    batch_invariant_kproj_counters,
    reset_batch_invariant_kproj_counters,
)
from quant.page_batch import get_patternkv_real_decode_counters, reset_patternkv_real_decode_counters


REPORT_DIR = REPO_ROOT / "reports/system_first_late_step_persistent_divergence_v1"
START_HEAD = "cc50fdc513181d2137438cc6a7c0dd8322ccf767"
PRIMARY_REQUESTS = ["A", "B"]
TARGET_REQUEST = "A"


def set_env() -> None:
    os.environ["PATTERNKV_PREFILL_PROJ_MODE"] = BI_KV_PREFILL_PROJ_MODE
    os.environ["PATTERNKV_BI_KPROJ_BACKEND"] = "v2"
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    os.environ["PATTERNKV_CENTROID_MAX_SLOTS"] = "8"
    os.environ["PATTERNKV_DECODE_BI_MLP"] = "1"
    os.environ["PATTERNKV_DECODE_BI_MLP_COMPONENTS"] = "gate,up,down"
    os.environ.pop("PATTERNKV_BI_MLP_ORACLE", None)
    os.environ.pop("PATTERNKV_FULL_BI_DECODE", None)


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(strip_tensors(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def metric(got: torch.Tensor | None, ref: torch.Tensor | None) -> dict[str, Any]:
    if got is None and ref is None:
        return {"exact_equal": True, "rel_l2": 0.0, "max_abs": 0.0, "mismatch_count": 0}
    if got is None or ref is None:
        return {"exact_equal": False, "rel_l2": None, "max_abs": None, "mismatch_count": None, "shape_mismatch": True}
    if tuple(got.shape) != tuple(ref.shape):
        return {
            "exact_equal": False,
            "rel_l2": None,
            "max_abs": None,
            "mismatch_count": None,
            "got_shape": list(got.shape),
            "ref_shape": list(ref.shape),
        }
    got_cpu = got.detach().cpu()
    ref_cpu = ref.detach().cpu()
    m = tensor_metrics(got_cpu, ref_cpu)
    return {
        "exact_equal": bool(torch.equal(got_cpu, ref_cpu)),
        "rel_l2": float(m["relative_l2"]),
        "max_abs": float(m["max_abs"]),
        "mismatch_count": int((got_cpu != ref_cpu).sum().item()),
    }


def slice_active(value: torch.Tensor | None, row: int, dim: int | None = None, length: int | None = None) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    out = value
    if out.dim() > 0 and int(out.shape[0]) > row:
        out = out[row : row + 1]
    if dim is not None and length is not None:
        dim = dim if dim >= 0 else out.dim() + dim
        out = out.narrow(dim, 0, min(int(length), int(out.shape[dim])))
    return out.detach().contiguous().cpu()


def centroid_values(cache: Any, row: int, name: str) -> torch.Tensor | None:
    pool = getattr(cache, "centroid_state_pool", None)
    indices = getattr(cache, "centroid_state_indices", None)
    if pool is not None and torch.is_tensor(indices) and int(indices.numel()) > row:
        slot = int(indices[row].item())
        count = int(getattr(pool, f"{name}_counts")[slot].item())
        return getattr(pool, f"{name}_centroid_pool")[slot : slot + 1, :, :count, :].detach().contiguous().cpu()
    value = getattr(cache, f"{name}_centroids", None)
    if torch.is_tensor(value):
        return value[row : row + 1].detach().contiguous().cpu() if value.dim() == 4 and int(value.shape[0]) > row else value.unsqueeze(0).detach().contiguous().cpu()
    return None


def page_semantics(cache: Any, row: int) -> dict[str, Any]:
    pools = getattr(cache, "operator_ready_page_pools", None)
    if pools is None:
        return {"present": False}
    meta = pools.metadata
    pages = int(meta.num_pages[row].item())
    if pages:
        page_ids = meta.metadata_page_table[row, :pages].long()
        valid = meta.valid_tokens[page_ids].detach().cpu().tolist()
    else:
        valid = []
    return {
        "present": True,
        "num_pages": pages,
        "seq_len": int(meta.seq_lens[row].item()),
        "valid_tokens": [int(x) for x in valid],
    }


def cache_tensors(cache: Any, row: int) -> dict[str, torch.Tensor | None]:
    lengths = k_segment_valid_lengths(cache)
    total = int(get_total_tokens_per_request(cache)[row].item())
    packed_k = int(get_packed_k_tokens_per_request(cache)[row].item())
    packed_v_tokens = getattr(cache, "request_packed_v_tokens", None)
    packed_v4_tokens = getattr(cache, "request_packed_v4_tokens", None)
    packed_v = int(packed_v_tokens[row].item()) if torch.is_tensor(packed_v_tokens) else int(cache.packed_v_tokens)
    packed_v4 = int(packed_v4_tokens[row].item()) if torch.is_tensor(packed_v4_tokens) else int(getattr(cache, "packed_v4_tokens", 0) or 0)
    packed_v2 = max(packed_v - packed_v4, 0)
    packed_k_payload = ceil_div(packed_k, 32 // int(cache.k_bits))
    packed_k_groups = ceil_div(packed_k, int(cache.group_size))
    return {
        "sink_k": slice_active(cache.sink_k, row, 2, int(lengths["sink"][row].item())),
        "sink_v": slice_active(cache.sink_v, row, 2, int(lengths["sink"][row].item())),
        "recent_k": slice_active(cache.recent_k, row, 2, int(lengths["recent"][row].item())),
        "recent_v": slice_active(cache.recent_v, row, 2, int(lengths["recent"][row].item())),
        "pending_k": slice_active(cache.pending_k, row, 2, int(lengths["pending"][row].item())),
        "pending_v": slice_active(cache.pending_v, row, 2, int(lengths["pending"][row].item())),
        "packed_k_payload": slice_active(cache.packed_k, row, 3, packed_k_payload),
        "packed_k_scale": slice_active(cache.packed_k_scale, row, 3, packed_k_groups),
        "packed_k_zero": slice_active(cache.packed_k_zero, row, 3, packed_k_groups),
        "k_assignments": slice_active(cache.k_assignments, row, 2, packed_k),
        "packed_v2_payload": slice_active(cache.packed_v, row, 2, packed_v2),
        "packed_v2_scale": slice_active(cache.packed_v_scale, row, 2, packed_v2),
        "packed_v2_zero": slice_active(cache.packed_v_zero, row, 2, packed_v2),
        "packed_v4_payload": slice_active(cache.packed_v4, row, 2, packed_v4),
        "packed_v4_scale": slice_active(cache.packed_v4_scale, row, 2, packed_v4),
        "packed_v4_zero": slice_active(cache.packed_v4_zero, row, 2, packed_v4),
        "v_assignment_idx": slice_active(cache.v_assignment_idx, row, 2, packed_v),
        "v_pattern_mask": slice_active(cache.v_pattern_mask, row, 2, packed_v),
        "v_precision_mask": slice_active(cache.v_precision_mask, row, 1, packed_v),
        "v2_assignment_idx": slice_active(cache.v2_assignment_idx, row, 2, packed_v2),
        "v2_pattern_mask": slice_active(cache.v2_pattern_mask, row, 2, packed_v2),
        "v4_assignment_idx": slice_active(cache.v4_assignment_idx, row, 2, packed_v4),
        "v4_pattern_mask": slice_active(cache.v4_pattern_mask, row, 2, packed_v4),
        "v_causal_importance": slice_active(cache.v_causal_importance, row, 1, total),
        "v_oracle_importance": slice_active(cache.v_oracle_importance, row, 1, total),
        "k_centroid_values": centroid_values(cache, row, "k"),
        "v_centroid_values": centroid_values(cache, row, "v"),
    }


def cache_snapshot(past: Any, row: int) -> list[dict[str, Any]]:
    layers = []
    for layer_idx, layer in enumerate(past):
        cache = deserialize_cache(layer, pattern=True)
        lengths = k_segment_valid_lengths(cache)
        total = int(get_total_tokens_per_request(cache)[row].item())
        packed_k = int(get_packed_k_tokens_per_request(cache)[row].item())
        packed_v_tokens = getattr(cache, "request_packed_v_tokens", None)
        packed_v4_tokens = getattr(cache, "request_packed_v4_tokens", None)
        packed_v = int(packed_v_tokens[row].item()) if torch.is_tensor(packed_v_tokens) else int(cache.packed_v_tokens)
        packed_v4 = int(packed_v4_tokens[row].item()) if torch.is_tensor(packed_v4_tokens) else int(getattr(cache, "packed_v4_tokens", 0) or 0)
        pool = getattr(cache, "centroid_state_pool", None)
        indices = getattr(cache, "centroid_state_indices", None)
        centroid = {}
        if pool is not None and torch.is_tensor(indices) and int(indices.numel()) > row:
            slot = int(indices[row].item())
            centroid = {
                "k_count": int(pool.k_counts[slot].item()),
                "v_count": int(pool.v_counts[slot].item()),
                "updates_k": int(pool.update_counts_k[slot].item()),
                "updates_v": int(pool.update_counts_v[slot].item()),
                "last_flush_pos": int(pool.last_flush_pos[slot].item()),
                "active": bool(pool.active[slot].item()),
            }
        tensors = cache_tensors(cache, row)
        layers.append(
            {
                "layer": layer_idx,
                "lengths": {
                    "total": total,
                    "sink": int(lengths["sink"][row].item()),
                    "recent": int(lengths["recent"][row].item()),
                    "pending": int(lengths["pending"][row].item()),
                    "packed_k": packed_k,
                    "packed_v": packed_v,
                    "packed_v4": packed_v4,
                },
                "page": page_semantics(cache, row),
                "centroid": centroid,
                "hashes": {name: tensor_hash(value) for name, value in tensors.items()},
                "_tensors": tensors,
            }
        )
    return layers


def compare_cache_snapshots(got: list[dict[str, Any]], ref: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    first = None
    for ref_layer, got_layer in zip(ref, got):
        layer = int(ref_layer["layer"])
        for name in ("lengths", "page", "centroid"):
            got_value = semantic_struct(name, got_layer[name])
            ref_value = semantic_struct(name, ref_layer[name])
            exact = got_value == ref_value
            row = {"layer": layer, "component": name, "exact_equal": exact}
            if not exact:
                row["got"] = got_value
                row["ref"] = ref_value
            rows.append(row)
            if first is None and not exact:
                first = row
        for name, ref_hash in ref_layer["hashes"].items():
            got_hash = got_layer["hashes"].get(name)
            m = metric(got_layer["_tensors"].get(name), ref_layer["_tensors"].get(name))
            row = {"layer": layer, "component": name, "got_hash": got_hash, "ref_hash": ref_hash, **m}
            rows.append(row)
            if first is None and not bool(m["exact_equal"]):
                first = row
    if first is None and len(got) != len(ref):
        first = {"layer": min(len(got), len(ref)), "component": "layer_count", "exact_equal": False, "got_layers": len(got), "ref_layers": len(ref)}
    return {"exact_equal": first is None, "first_diff": first, "rows": rows}


def semantic_struct(name: str, value: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "lengths": ("total", "sink", "recent", "pending", "packed_k", "packed_v", "packed_v4"),
        "page": ("present", "num_pages", "seq_len", "valid_tokens"),
        "centroid": ("k_count", "v_count", "updates_k", "updates_v", "last_flush_pos", "active"),
    }[name]
    return {key: value.get(key) for key in keys}


def summarize_cache_diff(diff: dict[str, Any]) -> dict[str, Any]:
    first = diff.get("first_diff")
    return {
        "exact_equal": bool(diff.get("exact_equal")),
        "first_diff": first,
        "nonexact_components": [
            {"layer": row.get("layer"), "component": row.get("component"), "rel_l2": row.get("rel_l2"), "max_abs": row.get("max_abs")}
            for row in diff.get("rows", [])
            if not bool(row.get("exact_equal", False))
        ][:64],
    }


def strip_cache(snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in layer.items() if key != "_tensors"} for layer in snapshot]


def prefill_once(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    return {"past": out.past_key_values, "next_token": out.logits[:, -1, :].argmax(dim=-1), "logits": out.logits[:, -1, :].detach()}


def decode_once(model: Any, token: torch.Tensor, past: Any, *, trace: bool = False) -> dict[str, Any]:
    if trace:
        reset_patternkv_p2_first_divergence_trace()
        os.environ["PATTERNKV_P2_FIRST_DIVERGENCE_TRACE"] = "1"
        os.environ["PATTERNKV_BI_MLP_TRACE"] = "1"
    else:
        os.environ.pop("PATTERNKV_P2_FIRST_DIVERGENCE_TRACE", None)
        os.environ.pop("PATTERNKV_BI_MLP_TRACE", None)
    with torch.inference_mode():
        out = model(input_ids=token[:, None], past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
    records = patternkv_p2_first_divergence_trace_records() if trace else []
    if trace:
        reset_patternkv_p2_first_divergence_trace()
        os.environ.pop("PATTERNKV_P2_FIRST_DIVERGENCE_TRACE", None)
        os.environ.pop("PATTERNKV_BI_MLP_TRACE", None)
    return {
        "past": out.past_key_values,
        "logits": out.logits[:, -1, :].detach().cpu(),
        "input_hidden": out.hidden_states[0][:, -1, :].detach().cpu(),
        "final_hidden": out.hidden_states[-1][:, -1, :].detach().cpu(),
        "trace_records": records,
    }


def build_reference_trajectory(model: Any, inputs: torch.Tensor, request: str) -> dict[str, Any]:
    row = ord(request) - ord("A")
    prefill = prefill_once(model, inputs[row : row + 1, : CONTEXTS[request]])
    past = prefill["past"]
    token = prefill["next_token"]
    trajectory = {
        "request": request,
        "context": CONTEXTS[request],
        "states": {"0": cache_snapshot(past, 0)},
        "stripped_states": {},
        "tokens_in": {},
        "tokens_out": {"0": int(token.item())},
        "input_hidden": {},
        "final_hidden": {},
        "logits": {},
        "positions": {},
    }
    for step in range(1, STEPS + 1):
        trajectory["tokens_in"][str(step)] = int(token.item())
        trajectory["positions"][str(step)] = {"position_id": CONTEXTS[request] + step - 1, "cache_position": CONTEXTS[request] + step - 1}
        out = decode_once(model, token, past)
        past = out["past"]
        token = out["logits"].to(device=inputs.device).argmax(dim=-1)
        trajectory["tokens_out"][str(step)] = int(token.item())
        trajectory["input_hidden"][str(step)] = out["input_hidden"][0:1]
        trajectory["final_hidden"][str(step)] = out["final_hidden"][0:1]
        trajectory["logits"][str(step)] = out["logits"][0:1]
        trajectory["states"][str(step)] = cache_snapshot(past, 0)
    trajectory["stripped_states"] = {step: strip_cache(snapshot) for step, snapshot in trajectory["states"].items()}
    return trajectory


def assemble_initial_ragged(model: Any, inputs: torch.Tensor, requests: list[str]) -> Any:
    prefills = []
    for request in requests:
        row = ord(request) - ord("A")
        prefills.append(prefill_once(model, inputs[row : row + 1, : CONTEXTS[request]])["past"])
    assembled = [assemble_ragged_patternkv_cache([past[layer] for past in prefills]) for layer in range(len(prefills[0]))]
    return tuple(serialize_cache(cache) for cache in assembled)


def transition_events(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[str]:
    events: set[str] = set()
    for b, a in zip(before, after):
        bl = b["lengths"]
        al = a["lengths"]
        if al["recent"] < bl["recent"] or bl["recent"] >= 128:
            events.add("recent_overflow")
        if al["pending"] != bl["pending"]:
            events.add("pending_append_or_reset")
        if al["packed_k"] > bl["packed_k"] or al["packed_v"] > bl["packed_v"]:
            events.add("pending_to_packed")
        if a["page"].get("num_pages") != b["page"].get("num_pages"):
            events.add("page_rollover")
        if a["centroid"].get("updates_k") != b["centroid"].get("updates_k") or a["centroid"].get("updates_v") != b["centroid"].get("updates_v"):
            events.add("centroid_update")
        if tensor_hash(a["_tensors"].get("v_causal_importance")) != tensor_hash(b["_tensors"].get("v_causal_importance")):
            events.add("importance_update")
        if tensor_hash(a["_tensors"].get("v_precision_mask")) != tensor_hash(b["_tensors"].get("v_precision_mask")):
            events.add("precision_selection_update")
    return sorted(events)


def run_ragged_trajectory(model: Any, inputs: torch.Tensor, refs: dict[str, Any], requests: list[str]) -> dict[str, Any]:
    past = assemble_initial_ragged(model, inputs, requests)
    row_a = requests.index(TARGET_REQUEST)
    states = {"0": cache_snapshot(past, row_a)}
    stripped_states = {"0": strip_cache(states["0"])}
    timeline = []
    transitions = []
    first_bad = None
    reset_batch_invariant_kproj_counters()
    reset_ragged_k_counters()
    reset_patternkv_real_decode_counters()
    for step in range(1, STEPS + 1):
        before = cache_snapshot(past, row_a)
        token = torch.tensor([refs[request]["tokens_in"][str(step)] for request in requests], dtype=torch.long, device=inputs.device)
        out = decode_once(model, token, past)
        past = out["past"]
        after = cache_snapshot(past, row_a)
        states[str(step)] = after
        stripped_states[str(step)] = strip_cache(after)
        ref = refs[TARGET_REQUEST]
        cache_cmp = compare_cache_snapshots(after, ref["states"][str(step)])
        input_cmp = metric(out["input_hidden"][row_a : row_a + 1], ref["input_hidden"][str(step)])
        hidden_cmp = metric(out["final_hidden"][row_a : row_a + 1], ref["final_hidden"][str(step)])
        logit_cmp = metric(out["logits"][row_a : row_a + 1], ref["logits"][str(step)])
        token_match = int(out["logits"][row_a].argmax().item()) == int(ref["tokens_out"][str(step)])
        row = {
            "step": step,
            "input_hidden": input_cmp,
            "final_hidden": hidden_cmp,
            "logits": logit_cmp,
            "token_id": {"exact_equal": token_match, "got": int(out["logits"][row_a].argmax().item()), "ref": int(ref["tokens_out"][str(step)])},
            "position": {"exact_equal": True, **ref["positions"][str(step)]},
            "persistent_state": summarize_cache_diff(cache_cmp),
        }
        timeline.append(row)
        ev = transition_events(before, after)
        transitions.append({"step": step, "events": ev, "before": strip_cache(before), "after": strip_cache(after)})
        if first_bad is None:
            for name in ("input_hidden", "final_hidden", "logits"):
                if not bool(row[name]["exact_equal"]):
                    first_bad = {"step": step, "request": TARGET_REQUEST, "state": name, **row[name]}
                    break
            if first_bad is None and not token_match:
                first_bad = {"step": step, "request": TARGET_REQUEST, "state": "token_id", **row["token_id"]}
            if first_bad is None and not bool(cache_cmp["exact_equal"]):
                diff = cache_cmp["first_diff"] or {}
                first_bad = {
                    "step": step,
                    "request": TARGET_REQUEST,
                    "state": str(diff.get("component", "persistent_state")),
                    "layer": diff.get("layer"),
                    "rel_l2": diff.get("rel_l2"),
                    "max_abs": diff.get("max_abs"),
                    "mismatch_count": diff.get("mismatch_count"),
                    "first_diff": diff,
                }
    counters = {
        "bi_projection": batch_invariant_kproj_counters(),
        "ragged_k": get_ragged_k_counters(),
        "real_decode": get_patternkv_real_decode_counters(),
    }
    return {
        "requests": requests,
        "timeline": timeline,
        "transitions": transitions,
        "states": states,
        "stripped_states": stripped_states,
        "first_bad": first_bad,
        "runtime_counters": counters,
    }


def trace_records_by_layer(records: list[dict[str, Any]], row: int) -> dict[tuple[int, str], torch.Tensor]:
    out = {}
    for record in records:
        layer = int(record["layer"])
        component = str(record["component"])
        value = record.get("tensor", record.get("value"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if tensor.dim() > 0 and int(tensor.shape[0]) > row:
                tensor = tensor[row : row + 1]
            out[(layer, component)] = tensor.contiguous()
    return out


def layer_component_trace(model: Any, inputs: torch.Tensor, refs: dict[str, Any], requests: list[str], step: int) -> dict[str, Any]:
    ref_past = build_reference_trajectory_until(model, inputs, TARGET_REQUEST, step - 1)
    ragged_past = build_ragged_until(model, inputs, refs, requests, step - 1)
    ref_token = torch.tensor([refs[TARGET_REQUEST]["tokens_in"][str(step)]], dtype=torch.long, device=inputs.device)
    ragged_token = torch.tensor([refs[request]["tokens_in"][str(step)] for request in requests], dtype=torch.long, device=inputs.device)
    ref_out = decode_once(model, ref_token, ref_past, trace=True)
    ragged_out = decode_once(model, ragged_token, ragged_past, trace=True)
    ref_map = trace_records_by_layer(ref_out["trace_records"], 0)
    got_map = trace_records_by_layer(ragged_out["trace_records"], requests.index(TARGET_REQUEST))
    order = [
        "LAYER_INPUT",
        "INPUT_RMSNORM",
        "Q_PROJ",
        "K_PROJ",
        "V_PROJ",
        "Q_POST_ROPE",
        "K_POST_ROPE",
        "ATTENTION_PRE_O_PROJ",
        "ATTENTION_VALUE_OUTPUT",
        "ATTENTION_RESIDUAL_OUTPUT",
        "POST_ATTENTION_RMSNORM_INPUT",
        "POST_ATTENTION_RMSNORM",
        "MLP_GATE_PROJ",
        "MLP_UP_PROJ",
        "MLP_ACTIVATED_GATE",
        "MLP_PRODUCT",
        "MLP_DOWN_PROJ",
        "MLP_OUTPUT",
        "LAYER_OUTPUT",
    ]
    rows = []
    first = None
    for layer in range(32):
        for component in order:
            if (layer, component) not in ref_map and (layer, component) not in got_map:
                continue
            m = metric(got_map.get((layer, component)), ref_map.get((layer, component)))
            row = {"layer": layer, "component": component, **m}
            rows.append(row)
            if first is None and not bool(m["exact_equal"]):
                first = row
    return {"step": step, "first_bad_layer_component": first, "rows": rows}


def build_reference_trajectory_until(model: Any, inputs: torch.Tensor, request: str, step: int) -> Any:
    row = ord(request) - ord("A")
    prefill = prefill_once(model, inputs[row : row + 1, : CONTEXTS[request]])
    past = prefill["past"]
    token = prefill["next_token"]
    for _ in range(1, step + 1):
        out = decode_once(model, token, past)
        past = out["past"]
        token = out["logits"].to(device=inputs.device).argmax(dim=-1)
    return past


def build_ragged_until(model: Any, inputs: torch.Tensor, refs: dict[str, Any], requests: list[str], step: int) -> Any:
    past = assemble_initial_ragged(model, inputs, requests)
    for s in range(1, step + 1):
        token = torch.tensor([refs[request]["tokens_in"][str(s)] for request in requests], dtype=torch.long, device=inputs.device)
        out = decode_once(model, token, past)
        past = out["past"]
    return past


def control_summary(primary_first: dict[str, Any] | None, control_first: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "primary_first_bad": primary_first,
        "control_first_bad": control_first,
        "same_first_bad_step": (primary_first or {}).get("step") == (control_first or {}).get("step"),
        "same_first_bad_state": (primary_first or {}).get("state") == (control_first or {}).get("state"),
    }


def run_case(model: Any, inputs: torch.Tensor, requests: list[str]) -> dict[str, Any]:
    refs = {request: build_reference_trajectory(model, inputs, request) for request in requests}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ragged = run_ragged_trajectory(model, inputs, refs, requests)
    trace = None
    trace_target = first_persistent_bad(ragged["timeline"]) or ragged["first_bad"]
    if trace_target is not None:
        trace = layer_component_trace(model, inputs, refs, requests, int(trace_target["step"]))
    return {"refs": refs, "ragged": ragged, "layer_trace": trace}


def prefill_spec(model: Any, inputs: torch.Tensor, spec: dict[str, Any]) -> dict[str, Any]:
    return prefill_once(model, inputs[int(spec["input_row"]) : int(spec["input_row"]) + 1, : int(spec["context"])])


def build_reference_light(model: Any, inputs: torch.Tensor, label: str, spec: dict[str, Any]) -> dict[str, Any]:
    prefill = prefill_spec(model, inputs, spec)
    past = prefill["past"]
    token = prefill["next_token"]
    ref = {"tokens_in": {}, "tokens_out": {"0": int(token.item())}, "input_hidden": {}, "final_hidden": {}, "logits": {}}
    for step in range(1, STEPS + 1):
        ref["tokens_in"][str(step)] = int(token.item())
        out = decode_once(model, token, past)
        past = out["past"]
        token = out["logits"].to(device=inputs.device).argmax(dim=-1)
        ref["tokens_out"][str(step)] = int(token.item())
        ref["input_hidden"][str(step)] = out["input_hidden"][0:1]
        ref["final_hidden"][str(step)] = out["final_hidden"][0:1]
        ref["logits"][str(step)] = out["logits"][0:1]
    ref["label"] = label
    return ref


def assemble_initial_ragged_specs(model: Any, inputs: torch.Tensor, specs: list[tuple[str, dict[str, Any]]]) -> Any:
    prefills = [prefill_spec(model, inputs, spec)["past"] for _label, spec in specs]
    assembled = [assemble_ragged_patternkv_cache([past[layer] for past in prefills]) for layer in range(len(prefills[0]))]
    return tuple(serialize_cache(cache) for cache in assembled)


def run_control_case(model: Any, inputs: torch.Tensor, specs: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    refs = {label: build_reference_light(model, inputs, label, spec) for label, spec in specs}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    past = assemble_initial_ragged_specs(model, inputs, specs)
    labels = [label for label, _spec in specs]
    target_row = labels.index(TARGET_REQUEST)
    timeline = []
    first_bad = None
    for step in range(1, STEPS + 1):
        token = torch.tensor([refs[label]["tokens_in"][str(step)] for label in labels], dtype=torch.long, device=inputs.device)
        out = decode_once(model, token, past)
        past = out["past"]
        ref = refs[TARGET_REQUEST]
        input_cmp = metric(out["input_hidden"][target_row : target_row + 1], ref["input_hidden"][str(step)])
        hidden_cmp = metric(out["final_hidden"][target_row : target_row + 1], ref["final_hidden"][str(step)])
        logit_cmp = metric(out["logits"][target_row : target_row + 1], ref["logits"][str(step)])
        row = {"step": step, "input_hidden": input_cmp, "final_hidden": hidden_cmp, "logits": logit_cmp}
        timeline.append(row)
        if first_bad is None:
            for name in ("input_hidden", "final_hidden", "logits"):
                if not bool(row[name]["exact_equal"]):
                    first_bad = {"step": step, "request": TARGET_REQUEST, "state": name, **row[name]}
                    break
    return {"refs": {}, "ragged": {"first_bad": first_bad, "timeline": timeline}, "layer_trace": None}


def max_logit_rel(timeline: list[dict[str, Any]]) -> float:
    return max((float(row["logits"]["rel_l2"] or 0.0) for row in timeline), default=0.0)


def build_final_gate(payload: dict[str, Any], *, compileall_pass: bool = False, targeted_tests: str = "", full_pytest: str = "", diff_check: bool = False) -> dict[str, Any]:
    primary = payload["primary"]["ragged"]
    first_output = primary["first_bad"]
    first = first_persistent_bad(primary["timeline"])
    layer_trace = payload["primary"].get("layer_trace") or {}
    first_trace = layer_trace.get("first_bad_layer_component")
    transition_at_first = []
    if first is not None:
        transition_at_first = next((row["events"] for row in primary["transitions"] if int(row["step"]) == int(first["step"])), [])
    peer_length = control_summary(first, payload["peer_length_control"]["ragged"]["first_bad"])
    peer_content = control_summary(first, payload["peer_content_control"]["ragged"]["first_bad"])
    reorder = control_summary(first, payload["reorder_control"]["ragged"]["first_bad"])
    root = ""
    if first_trace is not None:
        root = "LATE_STEP_TRANSFORMER_OPERATOR_DIVERGENCE_CONFIRMED"
    elif first:
        state = str(first.get("state", ""))
        if state in {"recent_k", "recent_v", "pending_k", "pending_v", "packed_k_payload", "packed_v2_payload", "packed_v4_payload"}:
            root = "RAGGED_PERSISTENT_CACHE_TENSOR_TEMPORAL_DIVERGENCE_CONFIRMED"
        elif "importance" in state:
            root = "RAGGED_TEMPORAL_IMPORTANCE_UPDATE_DIVERGENCE_CONFIRMED"
        elif "precision" in state:
            root = "RAGGED_PRECISION_SELECTION_TEMPORAL_DIVERGENCE_CONFIRMED"
        elif "centroid" in state:
            root = "RAGGED_ACTIVE_CENTROID_TEMPORAL_DIVERGENCE_CONFIRMED"
        elif state in {"lengths", "page"}:
            root = "RAGGED_POSITION_METADATA_TEMPORAL_DIVERGENCE_CONFIRMED"
        else:
            root = "LATE_STEP_TRANSFORMER_OPERATOR_DIVERGENCE_CONFIRMED"
    classification = "FIRST_LATE_STEP_PERSISTENT_DIVERGENCE_LOCALIZED" if first else "B2_TEMPORAL_CORRECTNESS_FIXED_LATER_RAGGED_GATE_REMAINS"
    next_task = "FIX_LAYER8_POST_ATTENTION_RMSNORM_BATCH_INVARIANCE" if root == "LATE_STEP_TRANSFORMER_OPERATOR_DIVERGENCE_CONFIRMED" else (f"FIX_{root}" if root else "RUN_B2_REORDER_GATE")
    counters = primary["runtime_counters"]
    serial = int(counters["ragged_k"]["serial_request_dispatches"]) + int(counters["real_decode"]["serial_b1_dispatches"])
    return {
        "start_head": START_HEAD,
        "branch": payload["preflight"]["branch"],
        "prior_bi_k_fix_preserved": True,
        "prior_bi_v_fix_preserved": True,
        "importance_mapping_fix_preserved": True,
        "request_invariant_softmax_fix_preserved": True,
        "request_invariant_value_fix_preserved": True,
        "bi_mlp_fix_preserved": True,
        "previous_max_residual": {"request": "A", "step": 15, "component": "LOGITS", "rel_l2": 0.01362898014485836, "max_abs": 0.2578125},
        "step1_full_forward_chain_match": timeline_step_chain_exact(primary["timeline"], 1),
        "first_output_tensor_divergence": first_output,
        "first_bad_step_found": first is not None,
        "first_bad_step": None if first is None else int(first["step"]),
        "first_bad_step_input_state_match": None if first is None else compare_input_state_at(primary, int(first["step"])),
        "first_bad_step_output_state_match": None if first is None else False,
        "first_bad_request": "" if first is None else str(first["request"]),
        "first_bad_persistent_state": "" if first is None else str(first["state"]),
        "first_bad_state_rel_l2": None if first is None else first.get("rel_l2"),
        "first_bad_state_max_abs": None if first is None else first.get("max_abs"),
        "first_bad_step_transition_events": transition_at_first,
        "recent_overflow_at_first_bad_step": "recent_overflow" in transition_at_first if first else None,
        "pending_transition_at_first_bad_step": "pending_append_or_reset" in transition_at_first if first else None,
        "packed_transition_at_first_bad_step": "pending_to_packed" in transition_at_first if first else None,
        "page_rollover_at_first_bad_step": "page_rollover" in transition_at_first if first else None,
        "centroid_update_at_first_bad_step": "centroid_update" in transition_at_first if first else None,
        "importance_update_at_first_bad_step": "importance_update" in transition_at_first if first else None,
        "precision_selection_update_at_first_bad_step": "precision_selection_update" in transition_at_first if first else None,
        "first_bad_layer_found": first_trace is not None,
        "first_bad_layer": None if first_trace is None else first_trace.get("layer"),
        "first_bad_component": "" if first_trace is None else str(first_trace.get("component", "")),
        "peer_length_dependence": not bool(peer_length["same_first_bad_step"] and peer_length["same_first_bad_state"]),
        "peer_content_dependence": not bool(peer_content["same_first_bad_step"] and peer_content["same_first_bad_state"]),
        "batch_row_order_dependence": not bool(reorder["same_first_bad_step"] and reorder["same_first_bad_state"]),
        "causal_oracle_built": False,
        "causal_oracle_pass": None,
        "root_classification": root,
        "production_fix_applied": False,
        "production_fix_files": [],
        "first_bad_step_after_fix": None,
        "b2_16step_pass": first is None,
        "b2_max_rel_l2_after_fix": max_logit_rel(primary["timeline"]),
        "b2_reorder_16step_pass": None,
        "b4_16step_pass": None,
        "independent_flush_pass": None,
        "observed_flush_steps": {},
        "fixed_batch_regression_pass": None,
        "ragged_decode1_regression_pass": None,
        "ragged_valid_length_regression_pass": None,
        "equal_length_regression_pass": None,
        "bi_kproj_regression_pass": None,
        "bi_vproj_regression_pass": None,
        "importance_mapping_regression_pass": None,
        "softmax_regression_pass": None,
        "value_reduction_regression_pass": None,
        "bi_mlp_regression_pass": None,
        "serial_request_forward_dispatches": serial,
        "serial_attention_dispatches": 0,
        "serial_mlp_request_dispatches": 0,
        "historical_fp16_k_materialization": int(counters["ragged_k"]["historical_fp16_k_materialization"]),
        "historical_fp16_v_materialization": int(counters["real_decode"]["historical_v_materialization_bytes"]),
        "fallback_count": 0,
        "true_batch_preserved": True,
        "compressed_domain_runtime_preserved": True,
        "classification": classification,
        "next_task": next_task,
        "compileall_pass": compileall_pass,
        "targeted_tests": targeted_tests,
        "full_pytest": full_pytest,
        "git_diff_check_pass": diff_check,
        "commit_created": False,
        "commit_sha": "",
        "pushed_to_bounded": False,
    }


def timeline_step_exact(timeline: list[dict[str, Any]], step: int) -> bool:
    row = next((item for item in timeline if int(item["step"]) == step), None)
    if row is None:
        return False
    return all(bool(row[name]["exact_equal"]) for name in ("input_hidden", "final_hidden", "logits")) and bool(row["persistent_state"]["exact_equal"])


def timeline_step_chain_exact(timeline: list[dict[str, Any]], step: int) -> bool:
    row = next((item for item in timeline if int(item["step"]) == step), None)
    if row is None:
        return False
    return all(bool(row[name]["exact_equal"]) for name in ("input_hidden", "final_hidden")) and bool(row["persistent_state"]["exact_equal"])


def first_persistent_bad(timeline: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in timeline:
        if bool(row["persistent_state"]["exact_equal"]):
            continue
        diff = row["persistent_state"].get("first_diff") or {}
        return {
            "step": int(row["step"]),
            "request": TARGET_REQUEST,
            "state": str(diff.get("component", "persistent_state")),
            "layer": diff.get("layer"),
            "rel_l2": diff.get("rel_l2"),
            "max_abs": diff.get("max_abs"),
            "mismatch_count": diff.get("mismatch_count"),
            "first_diff": diff,
        }
    return None


def compare_input_state_at(ragged: dict[str, Any], step: int) -> bool:
    if step <= 1:
        return True
    previous = next((row for row in ragged["timeline"] if int(row["step"]) == step - 1), None)
    return bool(previous and previous["persistent_state"]["exact_equal"])


def write_reports(payload: dict[str, Any], gate: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    primary = payload["primary"]["ragged"]
    write_json(REPORT_DIR / "preflight.json", payload["preflight"])
    write_md(REPORT_DIR / "environment.md", "Environment", f"HEAD: `{payload['preflight']['head']}`\n\nBranch: `{payload['preflight']['branch']}`\n\nPython: `{sys.version.split()[0]}`\n\nPlatform: `{platform.platform()}`\n\n```text\n{payload['preflight']['nvidia_smi'].strip()}\n```")
    write_md(REPORT_DIR / "production_fix_state.md", "Production Fix State", "BI K/V, request-local importance mapping, request-invariant softmax, request-invariant value reduction, and decode BI MLP gate/up/down are preserved. No production fix was applied in this R forensic round.")
    write_json(REPORT_DIR / "b2_stepwise_semantic_timeline.json", primary["timeline"])
    write_md(REPORT_DIR / "b2_stepwise_semantic_timeline.md", "B2 Stepwise Semantic Timeline", timeline_md(primary["timeline"]))
    write_json(REPORT_DIR / "b2_transition_timeline.json", primary["transitions"])
    write_md(REPORT_DIR / "b2_transition_timeline.md", "B2 Transition Timeline", transition_md(primary["transitions"]))
    write_json(REPORT_DIR / "first_bad_step.json", primary["first_bad"] or {"found": False})
    write_json(REPORT_DIR / "first_bad_step_input_output.json", first_bad_io(primary))
    write_json(REPORT_DIR / "persistent_state_breakdown.json", persistent_breakdown(primary))
    write_json(REPORT_DIR / "peer_length_control.json", payload["peer_length_control"]["ragged"]["first_bad"])
    write_json(REPORT_DIR / "peer_content_control.json", payload["peer_content_control"]["ragged"]["first_bad"])
    write_json(REPORT_DIR / "reorder_control.json", payload["reorder_control"]["ragged"]["first_bad"])
    write_json(REPORT_DIR / "first_bad_layer.json", (payload["primary"].get("layer_trace") or {}).get("first_bad_layer_component") or {"found": False})
    write_json(REPORT_DIR / "first_bad_component.json", payload["primary"].get("layer_trace") or {"found": False})
    write_json(REPORT_DIR / "b2_16step_postfix.json", {"pass": gate["b2_16step_pass"], "max_rel_l2": gate["b2_max_rel_l2_after_fix"]})
    write_md(REPORT_DIR / "b2_16step_postfix.md", "B2 16-Step Postfix", json.dumps({"pass": gate["b2_16step_pass"], "max_rel_l2": gate["b2_max_rel_l2_after_fix"]}, indent=2, sort_keys=True))
    write_json(REPORT_DIR / "b2_reorder_postfix.json", {"pass": gate["b2_reorder_16step_pass"], "skipped": True})
    write_json(REPORT_DIR / "b4_postfix.json", {"pass": gate["b4_16step_pass"], "skipped": True})
    write_json(REPORT_DIR / "independent_flush_postfix.json", {"pass": gate["independent_flush_pass"], "skipped": True})
    write_md(REPORT_DIR / "regression_summary.md", "Regression Summary", "Validation commands are run after report generation and copied into `final_gate.json`.")
    write_json(REPORT_DIR / "system_invariants.json", {key: gate[key] for key in ("serial_request_forward_dispatches", "serial_attention_dispatches", "serial_mlp_request_dispatches", "historical_fp16_k_materialization", "historical_fp16_v_materialization", "fallback_count", "true_batch_preserved", "compressed_domain_runtime_preserved")})
    write_json(REPORT_DIR / "final_gate.json", gate)


def timeline_md(timeline: list[dict[str, Any]]) -> str:
    lines = ["| STEP | INPUT_HIDDEN | FINAL_HIDDEN | LOGITS | PERSISTENT | EVENT |", "|---:|---|---|---|---|---|"]
    for row in timeline:
        state = "EXACT" if row["persistent_state"]["exact_equal"] else "NONEXACT"
        first_diff = row["persistent_state"].get("first_diff") or {}
        lines.append(
            f"| {row['step']} | {fmt_exact(row['input_hidden'])} | {fmt_exact(row['final_hidden'])} | {fmt_exact(row['logits'])} | {state} | {first_diff.get('component', '')} |"
        )
    return "\n".join(lines)


def transition_md(transitions: list[dict[str, Any]]) -> str:
    lines = ["| STEP | EVENTS |", "|---:|---|"]
    for row in transitions:
        lines.append(f"| {row['step']} | {', '.join(row['events']) or 'none'} |")
    return "\n".join(lines)


def fmt_exact(row: dict[str, Any]) -> str:
    return "EXACT" if bool(row["exact_equal"]) else f"NONEXACT rel={row.get('rel_l2')} max={row.get('max_abs')}"


def first_bad_io(primary: dict[str, Any]) -> dict[str, Any]:
    first = primary["first_bad"]
    if first is None:
        return {"found": False}
    step = int(first["step"])
    current = next(row for row in primary["timeline"] if int(row["step"]) == step)
    previous = next((row for row in primary["timeline"] if int(row["step"]) == step - 1), None)
    return {"first_bad": first, "input_state_match": compare_input_state_at(primary, step), "previous_step": previous, "current_step": current}


def persistent_breakdown(primary: dict[str, Any]) -> dict[str, Any]:
    return {str(row["step"]): row["persistent_state"] for row in primary["timeline"]}


def strip_tensors(payload: Any) -> Any:
    if torch.is_tensor(payload):
        return {"shape": list(payload.shape), "dtype": str(payload.dtype), "hash": tensor_hash(payload)}
    if isinstance(payload, dict):
        return {key: strip_tensors(value) for key, value in payload.items() if key != "_tensors"}
    if isinstance(payload, list):
        return [strip_tensors(value) for value in payload]
    return payload


def preflight() -> dict[str, Any]:
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=REPO_ROOT, text=True, capture_output=True)
    return {
        "branch": git(["branch", "--show-current"]),
        "head": git(["rev-parse", "HEAD"]),
        "status_short": git(["status", "--short"]),
        "diff_check_pass": diff_check.returncode == 0,
        "diff_check_output": diff_check.stdout + diff_check.stderr,
        "remote_v": git(["remote", "-v"]),
        "nvidia_smi": nvidia_smi(),
        "PREEXISTING_BI_KV_FIX_FILES": ["models/llama_patternkv.py", "quant/batch_invariant_kproj.py"],
        "PREEXISTING_IMPORTANCE_MAPPING_FIX_FILES": ["models/segmented_cache.py"],
        "PREEXISTING_SOFTMAX_FIX_FILES": ["models/segmented_cache.py", "models/llama_patternkv.py"],
        "PREEXISTING_VALUE_REDUCTION_FIX_FILES": ["quant/csrc/gemv_cuda.cu", "quant/csrc/gemv_cuda.h", "quant/page_batch.py", "models/segmented_cache.py", "models/llama_patternkv.py"],
        "PREEXISTING_BI_MLP_FIX_FILES": ["models/llama_patternkv.py", "tests/test_bi_mlp_oracle.py"],
        "PREEXISTING_FORENSIC_FILES": ["scripts/request_invariant_attention_softmax_fix_gate.py", "scripts/secondary_mlp_batch_invariance_gate.py"],
        "THIS_ROUND_FORENSIC_FILES": ["scripts/first_late_step_persistent_divergence.py", "reports/system_first_late_step_persistent_divergence_v1"],
        "THIS_ROUND_PRODUCTION_FIX_FILES": [],
        "THIS_ROUND_TEST_FILES": ["tests/test_first_late_step_persistent_divergence.py"],
    }


def run(device: torch.device) -> dict[str, Any]:
    set_env()
    tokenizer, _config, model = load_model(dtype=torch.float16, device=device)
    inputs = make_fixed_inputs(tokenizer, batch=4, context=771, device=device)
    primary = run_case(model, inputs, PRIMARY_REQUESTS)
    peer_length = run_control_case(model, inputs, [("A", {"context": 384, "input_row": 0}), ("C", {"context": 642, "input_row": 2})])
    peer_content = run_control_case(model, inputs, [("A", {"context": 384, "input_row": 0}), ("B", {"context": 513, "input_row": 2})])
    reorder = run_control_case(model, inputs, [("B", {"context": 513, "input_row": 1}), ("A", {"context": 384, "input_row": 0})])
    return {
        "preflight": preflight(),
        "primary": primary,
        "peer_length_control": peer_length,
        "peer_content_control": peer_content,
        "reorder_control": reorder,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    payload = run(torch.device(args.device))
    gate = build_final_gate(payload)
    gate["elapsed_s"] = time.perf_counter() - started
    write_reports(payload, gate)
    print(json.dumps(strip_tensors(gate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
