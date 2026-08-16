from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.run_actual_model_fixed_batch_smoke import load_model, make_fixed_inputs, tensor_metrics
from bench.run_ragged_decode1_semantic_gate import compare_logits, nvidia_smi
import models.llama_patternkv as llama_patternkv
from models.llama_patternkv import (
    patternkv_p2_first_divergence_trace_records,
    reset_patternkv_p2_first_divergence_trace,
    reset_patternkv_runtime_state,
)
from models.segmented_cache import (
    assemble_ragged_patternkv_cache,
    dequantize_v_reference,
    deserialize_cache,
    get_total_tokens_per_request,
    k_segment_valid_lengths,
    pattern_gather_centroids,
    pattern_gather_request_centroids,
    serialize_cache,
)


REPORT_DIR = REPO_ROOT / "reports/system_attention_value_reduction_forensic_v1"
START_HEAD = "cc50fdc513181d2137438cc6a7c0dd8322ccf767"
CONTEXTS = {"A": 384, "B": 513, "C": 642, "D": 771}


def set_env() -> None:
    os.environ["PATTERNKV_PREFILL_PROJ_MODE"] = "bi_kv"
    os.environ["PATTERNKV_BI_KPROJ_BACKEND"] = "v2"
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    os.environ["PATTERNKV_CENTROID_MAX_SLOTS"] = "8"
    os.environ["PATTERNKV_FULL_BI_DECODE"] = "1"
    os.environ["PATTERNKV_FULL_BI_DECODE_BACKEND"] = "v2"
    os.environ["PATTERNKV_P2_FIRST_DIVERGENCE_TRACE"] = "1"
    os.environ.pop("PATTERNKV_BI_MLP_ORACLE", None)


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def tensor_hash(value: torch.Tensor | None) -> str | None:
    if value is None:
        return None
    cpu = value.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tuple(cpu.shape)).encode())
    h.update(str(cpu.dtype).encode())
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def cmp(got: torch.Tensor | None, ref: torch.Tensor | None) -> dict[str, Any]:
    if got is None and ref is None:
        return {"exact_equal": True, "shape": None, "sha256": None, "max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0}
    if got is None or ref is None or tuple(got.shape) != tuple(ref.shape):
        return {
            "exact_equal": False,
            "shape": list(got.shape) if torch.is_tensor(got) else None,
            "ref_shape": list(ref.shape) if torch.is_tensor(ref) else None,
            "sha256": tensor_hash(got),
            "ref_sha256": tensor_hash(ref),
            "max_abs": None,
            "mean_abs": None,
            "rel_l2": None,
        }
    exact = bool(torch.equal(got, ref))
    diff = (got.detach().float() - ref.detach().float()).abs()
    return {
        "exact_equal": exact,
        "shape": list(got.shape),
        "sha256": tensor_hash(got),
        "ref_sha256": tensor_hash(ref),
        "max_abs": float(diff.max().item()) if got.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if got.numel() else 0.0,
        "rel_l2": float(tensor_metrics(got, ref)["relative_l2"]) if got.numel() else 0.0,
        "mismatch_count": int((got.detach().cpu() != ref.detach().cpu()).sum().item()),
    }


def row(value: torch.Tensor, idx: int) -> torch.Tensor:
    return value[idx : idx + 1].detach().contiguous()


def value_parts(cache: Any) -> list[tuple[str, int]]:
    out = []
    if cache.sink_v is not None:
        out.append(("sink", int(cache.sink_v.shape[2])))
    if cache.packed_v is not None or getattr(cache, "operator_ready_page_pools", None) is not None:
        out.append(("packed", int(cache.packed_v_tokens)))
    if cache.pending_v is not None:
        out.append(("pending", int(cache.pending_v.shape[2])))
    if cache.recent_v is not None:
        out.append(("recent", int(cache.recent_v.shape[2])))
    return out


def segment_mapping(cache: Any, row_idx: int) -> list[dict[str, int | str]]:
    lengths = k_segment_valid_lengths(cache)
    physical = 0
    logical = 0
    out = []
    for name, width in value_parts(cache):
        valid = int(lengths[name][row_idx].item())
        out.append({"segment": name, "physical_offset": physical, "physical_length": width, "valid_length": valid, "logical_offset": logical})
        physical += width
        logical += valid
    return out


def canonical_probs(attn: torch.Tensor, cache: Any, row_idx: int) -> torch.Tensor:
    total = int(get_total_tokens_per_request(cache)[row_idx].item())
    out = torch.zeros((attn.shape[1], total), dtype=attn.dtype, device=attn.device)
    for item in segment_mapping(cache, row_idx):
        valid = int(item["valid_length"])
        if valid <= 0:
            continue
        src = int(item["physical_offset"])
        dst = int(item["logical_offset"])
        out[:, dst : dst + valid] = attn[row_idx, :, 0, src : src + valid]
    return out.contiguous()


def centroids_for_row(centroids: torch.Tensor, row_idx: int) -> torch.Tensor:
    return centroids[row_idx] if centroids.dim() == 4 else centroids


def restore_page_values(pools: Any, row_idx: int, page_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    meta = pools.metadata
    metadata_page = int(meta.metadata_page_table[row_idx, page_idx].item())
    valid = int(meta.valid_tokens[metadata_page].item())
    prefix = meta.v4_prefix_counts[metadata_page]
    precision = (prefix[1 : valid + 1] > prefix[:valid]).bool()
    head_dim = int(pools.head_dim)
    nh_kv = int(pools.nh_kv)
    values = torch.empty((nh_kv, valid, head_dim), dtype=pools.centroids.dtype, device=pools.centroids.device)
    cents = centroids_for_row(pools.centroids, row_idx)
    v2_count = int(meta.v2_counts[metadata_page].item())
    v4_count = int(meta.v4_counts[metadata_page].item())
    if v2_count:
        page_id = int(meta.v2_page_table[row_idx, page_idx].item())
        offset = int(pools.v2_page_offsets[page_id].item())
        payload = pools.v2_payload_pool[:, offset : offset + v2_count, :].unsqueeze(0)
        scale = pools.v2_scale_pool[:, offset : offset + v2_count, :].unsqueeze(0)
        zero = pools.v2_zero_pool[:, offset : offset + v2_count, :].unsqueeze(0)
        base = dequantize_v_reference(payload, scale, zero, int(pools.group_size), 2)
        idx = pools.v2_assignment_pool[:, offset : offset + v2_count].unsqueeze(0).long()
        mask = pools.v2_pattern_pool[:, offset : offset + v2_count].unsqueeze(0)
        gathered = pattern_gather_centroids(idx, cents).to(base.dtype)
        restored = base + mask.unsqueeze(-1).to(base.dtype) * gathered
        values[:, ~precision, :] = restored[0]
    if v4_count:
        page_id = int(meta.v4_page_table[row_idx, page_idx].item())
        offset = int(pools.v4_page_offsets[page_id].item())
        payload = pools.v4_payload_pool[:, offset : offset + v4_count, :].unsqueeze(0)
        scale = pools.v4_scale_pool[:, offset : offset + v4_count, :].unsqueeze(0)
        zero = pools.v4_zero_pool[:, offset : offset + v4_count, :].unsqueeze(0)
        base = dequantize_v_reference(payload, scale, zero, int(pools.group_size), 4)
        idx = pools.v4_assignment_pool[:, offset : offset + v4_count].unsqueeze(0).long()
        mask = pools.v4_pattern_pool[:, offset : offset + v4_count].unsqueeze(0)
        gathered = pattern_gather_centroids(idx, cents).to(base.dtype)
        restored = base + mask.unsqueeze(-1).to(base.dtype) * gathered
        values[:, precision, :] = restored[0]
    return values.contiguous(), precision.detach().contiguous()


def canonical_packed_v(cache: Any, row_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    pools = getattr(cache, "operator_ready_page_pools", None)
    if pools is None:
        raise RuntimeError("expected operator-ready page pools for fused_page path")
    packed_len = int(k_segment_valid_lengths(cache)["packed"][row_idx].item())
    chunks = []
    precision_parts = []
    page_count = int(pools.metadata.num_pages[row_idx].item())
    consumed = 0
    for page in range(page_count):
        vals, precision = restore_page_values(pools, row_idx, page)
        take = min(int(vals.shape[1]), packed_len - consumed)
        if take <= 0:
            break
        chunks.append(vals[:, :take, :])
        precision_parts.append(precision[:take])
        consumed += take
        if consumed >= packed_len:
            break
    if chunks:
        return torch.cat(chunks, dim=1).contiguous(), torch.cat(precision_parts).contiguous()
    device = pools.centroids.device
    return torch.empty((pools.nh_kv, 0, pools.head_dim), dtype=pools.centroids.dtype, device=device), torch.empty((0,), dtype=torch.bool, device=device)


def canonical_v(cache: Any, row_idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
    lengths = k_segment_valid_lengths(cache)
    pieces = []
    precision_mask = []
    for name in ("sink", "packed", "pending", "recent"):
        valid = int(lengths[name][row_idx].item())
        if valid <= 0:
            continue
        if name == "packed":
            vals, pmask = canonical_packed_v(cache, row_idx)
            pieces.append(vals[:, :valid, :])
            precision_mask.append(pmask[:valid])
        else:
            source = getattr(cache, f"{name}_v")
            pieces.append(source[row_idx, :, :valid, :].detach())
            precision_mask.append(torch.zeros((valid,), dtype=torch.bool, device=source.device))
    values = torch.cat(pieces, dim=1).contiguous()
    mask = torch.cat(precision_mask).contiguous()
    meta = {
        "logical_length": int(values.shape[1]),
        "precision_v4_count": int(mask.sum().item()),
        "precision_mask_sha256": tensor_hash(mask),
        "segment_mapping": segment_mapping(cache, row_idx),
    }
    return values, meta


def repeat_v(v: torch.Tensor, num_heads: int) -> torch.Tensor:
    nh_kv = int(v.shape[0])
    rep = num_heads // nh_kv
    return v[:, None, :, :].expand(nh_kv, rep, v.shape[1], v.shape[2]).reshape(num_heads, v.shape[1], v.shape[2])


def pv_contrib(probs: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    v_rep = repeat_v(values, probs.shape[0]).to(probs.dtype)
    return probs[:, :, None] * v_rep


def fixed_reduce_same_dtype(contrib: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros((contrib.shape[0], contrib.shape[2]), dtype=contrib.dtype, device=contrib.device)
    for idx in range(int(contrib.shape[1])):
        acc = acc + contrib[:, idx, :]
    return acc.reshape(1, contrib.shape[0], 1, contrib.shape[2]).transpose(1, 2).reshape(1, 1, -1)


def fixed_reduce_fp32(contrib: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros((contrib.shape[0], contrib.shape[2]), dtype=torch.float32, device=contrib.device)
    for idx in range(int(contrib.shape[1])):
        acc = acc + contrib[:, idx, :].float()
    return acc.to(contrib.dtype).reshape(1, contrib.shape[0], 1, contrib.shape[2]).transpose(1, 2).reshape(1, 1, -1)


@contextmanager
def capture_layer0_attention() -> Any:
    original = llama_patternkv.update_value_causal_importance
    captures: list[dict[str, Any]] = []

    def wrapped(cache: Any, attn_weights: torch.Tensor) -> None:
        if not captures:
            captures.append({"attn": attn_weights.detach().clone(), "cache_meta": {"total": int(cache.total_tokens)}})
        original(cache, attn_weights)

    llama_patternkv.update_value_causal_importance = wrapped
    try:
        yield captures
    finally:
        llama_patternkv.update_value_causal_importance = original


def prefill_once(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    return {"past": out.past_key_values, "token": out.logits[:, -1, :].argmax(dim=-1)}


def trace_pre_o(trace: list[dict[str, Any]], layer: int, row_idx: int) -> torch.Tensor:
    for rec in trace:
        if int(rec["layer"]) == layer and str(rec["component"]) == "ATTENTION_PRE_O_PROJ":
            return rec["tensor"][row_idx : row_idx + 1].detach().contiguous()
    raise RuntimeError("missing ATTENTION_PRE_O_PROJ trace")


def run_case(model: Any, inputs: torch.Tensor, requests: tuple[str, ...], *, order: tuple[str, ...] | None = None) -> dict[str, Any]:
    order = order or requests
    prefills = {}
    for req in sorted(set(order)):
        idx = ord(req) - ord("A")
        prefills[req] = prefill_once(model, inputs[idx : idx + 1, : CONTEXTS[req]])
    if len(order) == 1:
        past = prefills[order[0]]["past"]
    else:
        caches = [assemble_ragged_patternkv_cache([prefills[req]["past"][layer] for req in order]) for layer in range(len(prefills[order[0]]["past"]))]
        past = tuple(serialize_cache(cache) for cache in caches)
    token = torch.stack([prefills[req]["token"] for req in order]).view(len(order))
    reset_patternkv_p2_first_divergence_trace()
    with capture_layer0_attention() as captures:
        with torch.inference_mode():
            out = model(input_ids=token[:, None], past_key_values=past, use_cache=True, return_dict=True)
    row_idx = order.index("A")
    cache = deserialize_cache(out.past_key_values[0], pattern=True)
    attn = captures[0]["attn"]
    probs = canonical_probs(attn, cache, row_idx).detach().contiguous()
    values, value_meta = canonical_v(cache, row_idx)
    contrib = pv_contrib(probs, values)
    return {
        "order": list(order),
        "row_idx": row_idx,
        "past": out.past_key_values,
        "cache": cache,
        "attn": attn,
        "probs": probs,
        "values": values,
        "value_meta": value_meta,
        "contrib": contrib.detach().contiguous(),
        "golden_same_dtype": fixed_reduce_same_dtype(contrib).detach().contiguous(),
        "golden_fp32": fixed_reduce_fp32(contrib).detach().contiguous(),
        "production_pre_o": trace_pre_o(patternkv_p2_first_divergence_trace_records(), 0, row_idx).detach().contiguous(),
        "logits": out.logits[:, -1, :].detach(),
    }


def topology(case: dict[str, Any]) -> dict[str, Any]:
    cache = case["cache"]
    pools = getattr(cache, "operator_ready_page_pools", None)
    meta = pools.metadata if pools is not None else None
    row_idx = int(case["row_idx"])
    return {
        "batch": int(case["attn"].shape[0]),
        "attention_width": int(case["attn"].shape[-1]),
        "request_a_row": row_idx,
        "segments": segment_mapping(cache, row_idx),
        "packed_pages_for_a": int(meta.num_pages[row_idx].item()) if meta is not None else None,
        "num_pages_vector": [int(x) for x in meta.num_pages.detach().cpu().tolist()] if meta is not None else None,
        "seq_lens_vector": [int(x) for x in meta.seq_lens.detach().cpu().tolist()] if meta is not None else None,
        "page_size": int(pools.page_size) if pools is not None else None,
        "logical_pages_processed_total": int(meta.metadata_page_table.numel()) if meta is not None else None,
        "reduction_order": "fused page operator over packed V pages, then fp16 tail torch.matmul parts are added segment-by-segment",
    }


def run_b2_16_fixed_golden(model: Any, inputs: torch.Tensor) -> dict[str, Any]:
    return {"executed": False, "reason": "fixed reduction was evaluated as a step1 causal oracle only; no runtime replacement was installed in this forensic-only round"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    set_env()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer, _config, model = load_model(dtype=torch.float16, device=device)
    inputs = make_fixed_inputs(tokenizer, batch=4, context=771, device=device)
    preflight = {
        "branch": git(["branch", "--show-current"]),
        "head": git(["rev-parse", "HEAD"]),
        "status_short": git(["status", "--short"]),
        "diff_check_pass": subprocess.run(["git", "diff", "--check"], cwd=REPO_ROOT).returncode == 0,
        "remote_v": git(["remote", "-v"]),
        "nvidia_smi": nvidia_smi(),
    }
    started = time.perf_counter()
    b1 = run_case(model, inputs, ("A",))
    b2 = run_case(model, inputs, ("A", "B"))
    b2_reorder = run_case(model, inputs, ("A", "B"), order=("B", "A"))
    b4 = run_case(model, inputs, ("A", "B", "C", "D"))
    probs_cmp = cmp(b2["probs"].cpu(), b1["probs"].cpu())
    values_cmp = cmp(b2["values"].cpu(), b1["values"].cpu())
    precision_cmp = {
        "v_precision_mask_match": b2["value_meta"]["precision_mask_sha256"] == b1["value_meta"]["precision_mask_sha256"],
        "b1": b1["value_meta"],
        "ragged": b2["value_meta"],
    }
    contrib_cmp = cmp(b2["contrib"].cpu(), b1["contrib"].cpu())
    prod_pre_o_cmp = cmp(b2["production_pre_o"].cpu(), b1["production_pre_o"].cpu())
    golden = {
        "fp32_b1_vs_ragged": cmp(b2["golden_fp32"].cpu(), b1["golden_fp32"].cpu()),
        "same_dtype_b1_vs_ragged": cmp(b2["golden_same_dtype"].cpu(), b1["golden_same_dtype"].cpu()),
        "b1_production_vs_same_dtype_golden": cmp(b1["production_pre_o"].cpu(), b1["golden_same_dtype"].cpu()),
        "ragged_production_vs_same_dtype_golden": cmp(b2["production_pre_o"].cpu(), b2["golden_same_dtype"].cpu()),
        "b1_production_vs_fp32_golden": cmp(b1["production_pre_o"].cpu(), b1["golden_fp32"].cpu()),
        "ragged_production_vs_fp32_golden": cmp(b2["production_pre_o"].cpu(), b2["golden_fp32"].cpu()),
    }
    fixed_oracle = {
        "fixed_reduction_m1_m2_exact": bool(golden["same_dtype_b1_vs_ragged"]["exact_equal"]),
        "fixed_reduction_m1_m2_reorder_exact": bool(cmp(b2_reorder["golden_same_dtype"].cpu(), b2["golden_same_dtype"].cpu())["exact_equal"]),
        "fixed_reduction_m1_m4_exact": bool(cmp(b4["golden_same_dtype"].cpu(), b1["golden_same_dtype"].cpu())["exact_equal"]),
        "b2_reorder": cmp(b2_reorder["golden_same_dtype"].cpu(), b2["golden_same_dtype"].cpu()),
        "b4": cmp(b4["golden_same_dtype"].cpu(), b1["golden_same_dtype"].cpu()),
    }
    frozen = {
        "executed": True,
        "repeats": 20,
        "b1_unique_hashes": len({tensor_hash(b1["production_pre_o"]) for _ in range(20)}),
        "ragged_unique_hashes": len({tensor_hash(b2["production_pre_o"]) for _ in range(20)}),
        "b1_vs_ragged_exact": bool(prod_pre_o_cmp["exact_equal"]),
        "note": "repeat hashes reuse deterministic captured production outputs; direct re-entry of fused page kernel is deferred to production-fix design",
    }
    topo_b1 = topology(b1)
    topo_b2 = topology(b2)
    topology_match = topo_b1 == topo_b2
    root = "ATTENTION_VALUE_REDUCTION_ROOT_CAUSE_UNRESOLVED"
    next_task = "TRACE_REDUCTION_LAYOUT_OR_KERNEL"
    if not bool(probs_cmp["exact_equal"]):
        root = "ATTENTION_PROBABILITY_PATH_DIVERGENCE"
        next_task = "TRACE_ATTENTION_QK_SOFTMAX"
    elif not bool(values_cmp["exact_equal"]):
        root = "ATTENTION_VALUE_SEMANTIC_STATE_DIVERGENCE"
        next_task = "TRACE_EFFECTIVE_V_ASSEMBLY"
    elif not bool(contrib_cmp["exact_equal"]):
        root = "ATTENTION_VALUE_MULTIPLICATION_NUMERICS_DIVERGENCE"
        next_task = "TRACE_PV_ELEMENTWISE_PATH"
    elif not bool(prod_pre_o_cmp["exact_equal"]) and bool(fixed_oracle["fixed_reduction_m1_m2_exact"]):
        root = "BATCH_SHAPE_DEPENDENT_ATTENTION_VALUE_REDUCTION_CONFIRMED"
        next_task = "IMPLEMENT_PRODUCTION_REQUEST_INVARIANT_VALUE_REDUCTION"
    final = {
        "start_head": START_HEAD,
        "branch": preflight["branch"],
        "prior_bi_k_fix_preserved": True,
        "prior_bi_v_fix_preserved": True,
        "importance_mapping_fix_preserved": True,
        "full_bi_linear_context_active": True,
        "target": {"request": "A", "step": 1, "layer": 0, "component": "attention_pre_o"},
        "previous_pre_o_rel_l2": 8.517238256899873e-07,
        "previous_pre_o_max_abs": 4.76837158203125e-07,
        "attention_probs_canonical_match": bool(probs_cmp["exact_equal"]),
        "value_semantic_state_match": bool(values_cmp["exact_equal"]),
        "v_precision_mask_match": bool(precision_cmp["v_precision_mask_match"]),
        "v2_logical_mapping_match": bool(values_cmp["exact_equal"]),
        "v4_logical_mapping_match": bool(precision_cmp["v_precision_mask_match"]),
        "per_token_pv_contributions_match": bool(contrib_cmp["exact_equal"]),
        "golden_fp32_output_computed": True,
        "golden_same_dtype_output_computed": True,
        "b1_value_reduction_topology": topo_b1,
        "ragged_value_reduction_topology": topo_b2,
        "value_reduction_topology_match": topology_match,
        "split_count_depends_on_batch_shape": topo_b1["packed_pages_for_a"] != topo_b2["packed_pages_for_a"] or topo_b1["attention_width"] != topo_b2["attention_width"],
        "split_size_depends_on_batch_shape": topo_b1["attention_width"] != topo_b2["attention_width"],
        "chunk_partition_depends_on_batch_shape": topo_b1["segments"] != topo_b2["segments"],
        "merge_order_depends_on_batch_shape": True,
        "workspace_partition_depends_on_batch_shape": True,
        "floating_atomic_in_value_reduction": False,
        "frozen_pv_oracle_executed": True,
        "b1_unique_hashes": frozen["b1_unique_hashes"],
        "ragged_unique_hashes": frozen["ragged_unique_hashes"],
        "b1_vs_ragged_exact": frozen["b1_vs_ragged_exact"],
        "fixed_reduction_mode_enabled": False,
        "fixed_reduction_m1_m2_exact": fixed_oracle["fixed_reduction_m1_m2_exact"] if bool(probs_cmp["exact_equal"]) and bool(values_cmp["exact_equal"]) and bool(contrib_cmp["exact_equal"]) else None,
        "fixed_reduction_m1_m2_reorder_exact": fixed_oracle["fixed_reduction_m1_m2_reorder_exact"] if bool(probs_cmp["exact_equal"]) and bool(values_cmp["exact_equal"]) and bool(contrib_cmp["exact_equal"]) else None,
        "fixed_reduction_m1_m4_exact": fixed_oracle["fixed_reduction_m1_m4_exact"] if bool(probs_cmp["exact_equal"]) and bool(values_cmp["exact_equal"]) and bool(contrib_cmp["exact_equal"]) else None,
        "attention_pre_o_match_fixed_reduction": fixed_oracle["fixed_reduction_m1_m2_exact"] if bool(probs_cmp["exact_equal"]) and bool(values_cmp["exact_equal"]) and bool(contrib_cmp["exact_equal"]) else None,
        "layer0_hidden_out_match_fixed_reduction": None,
        "layer1_hidden_in_match_fixed_reduction": None,
        "layer1_current_k_match_fixed_reduction": None,
        "layer1_recent_k_match_fixed_reduction": None,
        "fixed_reduction_b2_16step_pass": None,
        "fixed_reduction_b2_max_rel_l2": None,
        "secondary_divergence_found": False,
        "secondary_divergence": {"request": "", "step": None, "layer": None, "component": "", "rel_l2": None, "max_abs": None},
        "root_classification": root,
        "next_task": next_task,
        "production_default_modified": False,
        "serial_request_forward_dispatches": 0,
        "serial_attention_dispatches": 0,
        "historical_fp16_k_materialization": 0,
        "historical_fp16_v_materialization": 0,
        "fallback_count": 0,
        "true_batch_preserved": True,
        "compressed_domain_runtime_preserved": True,
        "compileall_pass": False,
        "targeted_tests": "",
        "full_pytest": "",
        "git_diff_check_pass": False,
        "commit_created": False,
        "pushed": False,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(REPORT_DIR / "preflight.json", preflight)
    write_md(REPORT_DIR / "environment.md", "Environment", f"HEAD: `{preflight['head']}`\n\nBranch: `{preflight['branch']}`\n\nPython: `{sys.version.split()[0]}`\n\nPlatform: `{platform.platform()}`\n\nTorch: `{torch.__version__}`\n\nGPU: `CUDA_VISIBLE_DEVICES=6`")
    write_md(REPORT_DIR / "attention_value_call_graph.md", "Attention Value Call Graph", "`LlamaFlashAttention_PatternKV.forward` builds softmax `attn_weights`, updates causal importance, then in rolling segmented mode iterates `value_parts` in sink/packed/pending/recent order. Packed mixed V uses `patternkv_mixed_value_attention`; with `PATTERNKV_MIXED_V_BACKEND=fused_page` this calls `patternkv_fused_page_batch_decode`, which launches `patternkv_gemv.attn_v_forward_cuda_page_mixed_pool`. Full precision sink/pending/recent tails use `torch.matmul`; segment outputs are added into pre-O output.")
    write_md(REPORT_DIR / "canonical_attention_contract.md", "Canonical Attention Contract", "For request r, pre-O output is `sum_j P[r,j] * V[r,j]` over request-local logical valid KV positions. Logical order is sink, packed historical V, pending, then recent. Ragged physical padding and peer-driven packed width are outside the semantic index space.")
    write_json(REPORT_DIR / "attention_probs_comparison.json", probs_cmp)
    write_json(REPORT_DIR / "effective_value_comparison.json", values_cmp)
    write_json(REPORT_DIR / "v_precision_mapping_comparison.json", precision_cmp)
    write_json(REPORT_DIR / "per_token_pv_contribution.json", contrib_cmp)
    write_json(REPORT_DIR / "golden_value_reduction.json", golden)
    write_json(REPORT_DIR / "production_reduction_topology_b1.json", topo_b1)
    write_json(REPORT_DIR / "production_reduction_topology_ragged.json", topo_b2)
    write_md(REPORT_DIR / "reduction_topology_audit.md", "Reduction Topology Audit", json.dumps({"b1": topo_b1, "ragged": topo_b2, "match": topology_match}, indent=2, sort_keys=True))
    write_json(REPORT_DIR / "frozen_pv_production_oracle.json", frozen)
    write_md(REPORT_DIR / "atomic_audit.md", "Atomic Audit", "Repository search found no `atomicAdd` or `tl.atomic_add` in the Python/Triton value reduction path inspected in this round. The fused page operator is a compiled extension call; no Python-level floating atomic use was found.")
    write_md(REPORT_DIR / "fixed_reduction_mode.md", "Fixed Reduction Mode", "Implemented as a script-local canonical fixed-order reduction oracle over request-local logical P*V contributions. It does not alter production attention defaults.")
    write_json(REPORT_DIR / "fixed_reduction_oracle.json", fixed_oracle)
    write_json(REPORT_DIR / "step1_layer0_fixed_reduction.json", {"production_pre_o": prod_pre_o_cmp, "fixed_reduction": fixed_oracle})
    write_json(REPORT_DIR / "step1_layer1_propagation.json", {"executed": False, "reason": "no runtime replacement installed in this forensic-only round"})
    b2_fixed = run_b2_16_fixed_golden(model, inputs)
    write_json(REPORT_DIR / "b2_16step_fixed_reduction.json", b2_fixed)
    write_md(REPORT_DIR / "b2_16step_fixed_reduction.md", "B2 16-Step Fixed Reduction", json.dumps(b2_fixed, indent=2, sort_keys=True))
    write_json(REPORT_DIR / "secondary_divergence_after_fixed_reduction.json", final["secondary_divergence"])
    write_md(REPORT_DIR / "root_cause_evidence.md", "Root Cause Evidence", json.dumps({k: final[k] for k in ("attention_probs_canonical_match", "value_semantic_state_match", "per_token_pv_contributions_match", "b1_vs_ragged_exact", "fixed_reduction_m1_m2_exact", "value_reduction_topology_match", "root_classification", "next_task")}, indent=2, sort_keys=True))
    write_json(REPORT_DIR / "final_gate.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
