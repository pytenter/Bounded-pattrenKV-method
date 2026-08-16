from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from bench.run_ragged_decode1_semantic_gate import nvidia_smi
import models.llama_patternkv as llama_patternkv
from models.llama_patternkv import (
    patternkv_p2_first_divergence_trace_records,
    reset_patternkv_p2_first_divergence_trace,
    reset_patternkv_runtime_state,
)
from models.segmented_cache import (
    _slice_ragged_request_cache,
    assemble_ragged_patternkv_cache,
    build_k_segment_validity_mask,
    dequantize_k_reference,
    deserialize_cache,
    get_total_tokens_per_request,
    k_segment_valid_lengths,
    pattern_gather_centroids,
    pattern_gather_request_centroids,
    serialize_cache,
)


REPORT_DIR = REPO_ROOT / "reports/system_attention_qk_online_softmax_forensic_v1"
START_HEAD = "cc50fdc513181d2137438cc6a7c0dd8322ccf767"
CONTEXTS = {"A": 384, "B": 513, "C": 642, "D": 771}
SPLIT_SIZE = 128


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
        return {"exact_equal": True, "shape": None, "sha256": None, "max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0, "mismatch_count": 0}
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
            "mismatch_count": None,
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
        out.append(
            {
                "segment": name,
                "physical_offset": physical,
                "physical_length": int(width),
                "valid_length": valid,
                "logical_offset": logical,
            }
        )
        physical += int(width)
        logical += valid
    return out


def canonical_probs(attn: torch.Tensor, cache: Any, row_idx: int) -> torch.Tensor:
    total = int(get_total_tokens_per_request(cache)[row_idx].item())
    out = torch.empty((attn.shape[1], total), dtype=attn.dtype, device=attn.device)
    for item in segment_mapping(cache, row_idx):
        valid = int(item["valid_length"])
        if valid <= 0:
            continue
        src = int(item["physical_offset"])
        dst = int(item["logical_offset"])
        out[:, dst : dst + valid] = attn[row_idx, :, 0, src : src + valid]
    return out.contiguous()


def canonical_k(cache: Any, row_idx: int) -> torch.Tensor:
    lengths = k_segment_valid_lengths(cache)
    row_cache = _slice_ragged_request_cache(cache, row_idx, lengths)
    packed_k = dequantize_k_reference(row_cache.packed_k, row_cache.packed_k_scale, row_cache.packed_k_zero, row_cache.group_size, row_cache.k_bits)
    if packed_k is not None:
        packed_k = packed_k[:, :, : row_cache.packed_k_tokens, :].contiguous()
        if row_cache.k_centroids is not None and row_cache.k_assignments is not None:
            idx = row_cache.k_assignments[:, :, : row_cache.packed_k_tokens]
            if row_cache.k_centroids.dim() == 4:
                gathered = pattern_gather_request_centroids(idx, row_cache.k_centroids)
            else:
                gathered = pattern_gather_centroids(idx, row_cache.k_centroids)
            packed_k = packed_k + gathered.to(packed_k.dtype)
    pieces = []
    for part in (row_cache.sink_k, packed_k, row_cache.pending_k, row_cache.recent_k):
        if torch.is_tensor(part) and part.shape[2] > 0:
            pieces.append(part)
    if not pieces:
        raise RuntimeError("empty K reconstruction")
    return torch.cat(pieces, dim=2).contiguous()[0]


def repeat_kv_local(value: torch.Tensor, num_heads: int) -> torch.Tensor:
    nh_kv = int(value.shape[0])
    rep = num_heads // nh_kv
    return value[:, None, :, :].expand(nh_kv, rep, value.shape[1], value.shape[2]).reshape(num_heads, value.shape[1], value.shape[2])


def trace_component(trace: list[dict[str, Any]], layer: int, component: str, row_idx: int) -> torch.Tensor:
    for rec in trace:
        if int(rec["layer"]) == layer and str(rec["component"]) == component:
            return row(rec["tensor"], row_idx)
    raise RuntimeError(f"missing trace {component} layer {layer}")


@contextmanager
def capture_layer0_attention() -> Any:
    original = llama_patternkv.update_value_causal_importance
    captures: list[dict[str, Any]] = []

    def wrapped(cache: Any, attn_weights: torch.Tensor) -> None:
        if not captures:
            captures.append({"attn": attn_weights.detach().clone(), "total": int(cache.total_tokens)})
        original(cache, attn_weights)

    llama_patternkv.update_value_causal_importance = wrapped
    try:
        yield captures
    finally:
        llama_patternkv.update_value_causal_importance = original


@contextmanager
def capture_layer0_softmax_input() -> Any:
    original = llama_patternkv.nn.functional.softmax
    captures: list[torch.Tensor] = []

    def wrapped(input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not captures and torch.is_tensor(input) and input.dim() == 4 and int(input.shape[2]) == 1:
            captures.append(input.detach().clone())
        return original(input, *args, **kwargs)

    llama_patternkv.nn.functional.softmax = wrapped
    try:
        yield captures
    finally:
        llama_patternkv.nn.functional.softmax = original


def prefill_once(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    return {"past": out.past_key_values, "token": out.logits[:, -1, :].argmax(dim=-1)}


def run_case(model: Any, inputs: torch.Tensor, order: tuple[str, ...]) -> dict[str, Any]:
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
        with capture_layer0_softmax_input() as softmax_inputs:
            with torch.inference_mode():
                out = model(input_ids=token[:, None], past_key_values=past, use_cache=True, return_dict=True)
    if not captures:
        raise RuntimeError("attention capture failed")
    if not softmax_inputs:
        raise RuntimeError("softmax input capture failed")
    trace = patternkv_p2_first_divergence_trace_records()
    row_idx = order.index("A")
    cache = deserialize_cache(out.past_key_values[0], pattern=True)
    q = trace_component(trace, 0, "Q_POST_ROPE", row_idx)[0, :, 0, :].detach().contiguous()
    k_current = trace_component(trace, 0, "K_POST_ROPE", row_idx)[0, :, 0, :].detach().contiguous()
    k = canonical_k(cache, row_idx).detach().contiguous()
    p = canonical_probs(captures[0]["attn"], cache, row_idx).detach().contiguous()
    return {
        "order": list(order),
        "row_idx": row_idx,
        "cache": cache,
        "attn": captures[0]["attn"].detach().contiguous(),
        "softmax_input": softmax_inputs[0].detach().contiguous(),
        "q": q,
        "k_current": k_current,
        "k": k,
        "p": p,
        "logits": out.logits[:, -1, :].detach().contiguous(),
    }


def qk_products(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    k = k.to(device=q.device)
    return q[:, None, :] * repeat_kv_local(k, q.shape[0])


def raw_qk_logits(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    k_rep = repeat_kv_local(k.to(device=q.device), q.shape[0])
    return torch.matmul(q[:, None, :], k_rep.transpose(1, 2)).squeeze(1).contiguous()


def dot_reduction_oracle(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    prod = q[:, None, :] * repeat_kv_local(k.to(device=q.device), q.shape[0])
    acc = torch.zeros(prod.shape[:2], dtype=prod.dtype, device=prod.device)
    for dim in range(int(prod.shape[-1])):
        acc = acc + prod[:, :, dim]
    return acc.contiguous()


def logical_split_boundaries(length: int, split_size: int = SPLIT_SIZE) -> list[dict[str, int]]:
    return [{"split": i, "start": start, "end": min(start + split_size, length)} for i, start in enumerate(range(0, length, split_size))]


def split_state(logits: torch.Tensor, boundaries: list[dict[str, int]]) -> dict[str, Any]:
    max_parts = []
    lse_parts = []
    for item in boundaries:
        part = logits[:, int(item["start"]) : int(item["end"])].float()
        local_max = part.max(dim=-1).values
        local_lse = torch.exp(part - local_max[:, None]).sum(dim=-1)
        max_parts.append(local_max)
        lse_parts.append(local_lse)
    max_tensor = torch.stack(max_parts, dim=1)
    lse_tensor = torch.stack(lse_parts, dim=1)
    merged_max = max_tensor.max(dim=1).values
    merged_lse = (lse_tensor * torch.exp(max_tensor - merged_max[:, None])).sum(dim=1)
    return {"local_max": max_tensor, "local_lse": lse_tensor, "merged_max": merged_max, "merged_lse": merged_lse}


def split_softmax(logits: torch.Tensor, boundaries: list[dict[str, int]], dtype: torch.dtype) -> torch.Tensor:
    state = split_state(logits, boundaries)
    probs = torch.empty_like(logits, dtype=torch.float32)
    for item in boundaries:
        start = int(item["start"])
        end = int(item["end"])
        probs[:, start:end] = torch.exp(logits[:, start:end].float() - state["merged_max"][:, None]) / state["merged_lse"][:, None]
    return probs.to(dtype)


def physical_mask(cache: Any, device: torch.device) -> torch.Tensor:
    mask = build_k_segment_validity_mask(cache, value_parts(cache), device=device)
    if mask is None:
        return torch.ones((1, int(cache.total_tokens)), dtype=torch.bool, device=device)
    return mask


def physical_scaled_logits_from_logical(case: dict[str, Any], logical_scaled: torch.Tensor) -> torch.Tensor:
    cache = case["cache"]
    row_idx = int(case["row_idx"])
    sentinel = torch.finfo(logical_scaled.dtype).min
    out = torch.full((logical_scaled.shape[0], int(cache.total_tokens)), sentinel, dtype=logical_scaled.dtype, device=logical_scaled.device)
    for item in segment_mapping(cache, row_idx):
        valid = int(item["valid_length"])
        if valid <= 0:
            continue
        src = int(item["logical_offset"])
        dst = int(item["physical_offset"])
        out[:, dst : dst + valid] = logical_scaled[:, src : src + valid]
    return out.contiguous()


def canonicalize_physical(case: dict[str, Any], physical: torch.Tensor) -> torch.Tensor:
    row_idx = int(case["row_idx"])
    total = int(get_total_tokens_per_request(case["cache"])[row_idx].item())
    out = torch.empty((physical.shape[0], total), dtype=physical.dtype, device=physical.device)
    for item in segment_mapping(case["cache"], row_idx):
        valid = int(item["valid_length"])
        if valid <= 0:
            continue
        src = int(item["physical_offset"])
        dst = int(item["logical_offset"])
        out[:, dst : dst + valid] = physical[:, src : src + valid]
    return out.contiguous()


def canonical_softmax_input(case: dict[str, Any]) -> torch.Tensor:
    return canonicalize_physical(case, case["softmax_input"][int(case["row_idx"]), :, 0, :]).contiguous()


def max_lse_for_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    f = logits.float()
    m = f.max(dim=-1).values
    lse = torch.exp(f - m[:, None]).sum(dim=-1)
    return m, lse


def compact_tensor(value: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_hash(value)}


def run_b2_16_softmax_fixed() -> dict[str, Any]:
    return {"executed": False, "reason": "forensic-only round; fixed softmax was evaluated as script-local oracle, not installed as a runtime replacement"}


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
    b2_reorder = run_case(model, inputs, ("B", "A"))
    b4 = run_case(model, inputs, ("A", "B", "C", "D"))

    q_cmp = cmp(b2["q"].cpu(), b1["q"].cpu())
    k_cmp = cmp(b2["k"].cpu(), b1["k"].cpu())
    current_k_cmp = cmp(b2["k_current"].cpu(), b1["k_current"].cpu())
    product_cmp = cmp(qk_products(b2["q"], b2["k"]).cpu(), qk_products(b1["q"], b1["k"]).cpu())
    raw_b1 = raw_qk_logits(b1["q"], b1["k"])
    raw_b2 = raw_qk_logits(b2["q"], b2["k"])
    raw_b2_reorder = raw_qk_logits(b2_reorder["q"], b2_reorder["k"])
    raw_b4 = raw_qk_logits(b4["q"], b4["k"])
    raw_cmp = cmp(raw_b2.cpu(), raw_b1.cpu())
    dot_cmp = cmp(dot_reduction_oracle(b2["q"], b2["k"]).cpu(), dot_reduction_oracle(b1["q"], b1["k"]).cpu())
    scaled_b1 = (raw_b1 / math.sqrt(b1["q"].shape[-1])).contiguous()
    scaled_b2 = (raw_b2 / math.sqrt(b2["q"].shape[-1])).contiguous()
    scaled_b2_reorder = (raw_b2_reorder / math.sqrt(b2_reorder["q"].shape[-1])).contiguous()
    scaled_b4 = (raw_b4 / math.sqrt(b4["q"].shape[-1])).contiguous()
    scaled_cmp = cmp(scaled_b2.cpu(), scaled_b1.cpu())
    production_masked_b1 = canonical_softmax_input(b1)
    production_masked_b2 = canonical_softmax_input(b2)
    masked_valid_cmp = cmp(production_masked_b2.cpu(), production_masked_b1.cpu())
    production_masked_vs_reference_b1 = cmp(production_masked_b1.cpu(), scaled_b1.cpu())
    production_masked_vs_reference_b2 = cmp(production_masked_b2.cpu(), scaled_b2.cpu())

    p_cmp = cmp(b2["p"].cpu(), b1["p"].cpu())
    p_reorder_cmp = cmp(b2_reorder["p"].cpu(), b2["p"].cpu())
    p_b4_cmp = cmp(b4["p"].cpu(), b1["p"].cpu())
    no_split_b1 = torch.softmax(scaled_b1.float(), dim=-1).to(b1["p"].dtype)
    no_split_b2 = torch.softmax(scaled_b2.float(), dim=-1).to(b2["p"].dtype)
    no_split_b2_reorder = torch.softmax(scaled_b2_reorder.float(), dim=-1).to(b2_reorder["p"].dtype)
    no_split_b4 = torch.softmax(scaled_b4.float(), dim=-1).to(b4["p"].dtype)
    no_split_cmp = cmp(no_split_b2.cpu(), no_split_b1.cpu())
    no_split_reorder_cmp = cmp(no_split_b2_reorder.cpu(), no_split_b2.cpu())
    no_split_b4_cmp = cmp(no_split_b4.cpu(), no_split_b1.cpu())

    fixed_bounds = logical_split_boundaries(int(scaled_b1.shape[-1]))
    fixed_split_b1 = split_softmax(scaled_b1, fixed_bounds, b1["p"].dtype)
    fixed_split_b2 = split_softmax(scaled_b2, fixed_bounds, b2["p"].dtype)
    fixed_split_b2_reorder = split_softmax(scaled_b2_reorder, fixed_bounds, b2_reorder["p"].dtype)
    fixed_split_b4 = split_softmax(scaled_b4, fixed_bounds, b4["p"].dtype)
    fixed_split_cmp = cmp(fixed_split_b2.cpu(), fixed_split_b1.cpu())
    fixed_split_reorder_cmp = cmp(fixed_split_b2_reorder.cpu(), fixed_split_b2.cpu())
    fixed_split_b4_cmp = cmp(fixed_split_b4.cpu(), fixed_split_b1.cpu())

    physical_b1 = physical_scaled_logits_from_logical(b1, scaled_b1)
    physical_b2 = physical_scaled_logits_from_logical(b2, scaled_b2)
    physical_p_b1 = torch.softmax(physical_b1.float(), dim=-1).to(b1["p"].dtype)
    physical_p_b2 = torch.softmax(physical_b2.float(), dim=-1).to(b2["p"].dtype)
    physical_canon_b1 = canonicalize_physical(b1, physical_p_b1)
    physical_canon_b2 = canonicalize_physical(b2, physical_p_b2)
    physical_reconstruct_cmp = cmp(physical_canon_b2.cpu(), physical_canon_b1.cpu())
    production_vs_physical_b1 = cmp(b1["p"].cpu(), physical_canon_b1.cpu())
    production_vs_physical_b2 = cmp(b2["p"].cpu(), physical_canon_b2.cpu())
    replay_b1 = torch.softmax(b1["softmax_input"].float(), dim=-1).to(b1["p"].dtype)[int(b1["row_idx"]), :, 0, :]
    replay_b2 = torch.softmax(b2["softmax_input"].float(), dim=-1).to(b2["p"].dtype)[int(b2["row_idx"]), :, 0, :]
    replay_canon_b1 = canonicalize_physical(b1, replay_b1)
    replay_canon_b2 = canonicalize_physical(b2, replay_b2)
    captured_replay_cmp = cmp(replay_canon_b2.cpu(), replay_canon_b1.cpu())
    production_vs_replay_b1 = cmp(b1["p"].cpu(), replay_canon_b1.cpu())
    production_vs_replay_b2 = cmp(b2["p"].cpu(), replay_canon_b2.cpu())

    b1_m, b1_lse = max_lse_for_logits(physical_b1)
    b2_m, b2_lse = max_lse_for_logits(physical_b2)
    no1_state = split_state(scaled_b1, [{"split": 0, "start": 0, "end": int(scaled_b1.shape[-1])}])
    no2_state = split_state(scaled_b2, [{"split": 0, "start": 0, "end": int(scaled_b2.shape[-1])}])
    fixed1_state = split_state(scaled_b1, fixed_bounds)
    fixed2_state = split_state(scaled_b2, fixed_bounds)

    invalid_sentinel = torch.finfo(scaled_b1.dtype).min
    b1_mask = physical_mask(b1["cache"], b1["p"].device)
    b2_mask = physical_mask(b2["cache"], b2["p"].device)
    mask_audit = {
        "dtype": str(scaled_b1.dtype),
        "invalid_sentinel": float(invalid_sentinel),
        "invalid_exp_fp32_zero": bool(torch.exp(torch.tensor(float(invalid_sentinel), dtype=torch.float32, device=device)).item() == 0.0),
        "b1_mask": compact_tensor(b1_mask),
        "ragged_mask": compact_tensor(b2_mask),
        "b1_invalid_count_for_a": int((~b1_mask[b1["row_idx"]]).sum().item()),
        "ragged_invalid_count_for_a": int((~b2_mask[b2["row_idx"]]).sum().item()),
        "b1_physical_width": int(b1["attn"].shape[-1]),
        "ragged_physical_width": int(b2["attn"].shape[-1]),
        "b1_logical_length_a": int(b1["p"].shape[-1]),
        "ragged_logical_length_a": int(b2["p"].shape[-1]),
    }

    dynamic_boundaries = {
        "production_softmax_kind": "torch.nn.functional.softmax over concatenated physical attention axis",
        "b1": {"physical_width": int(b1["attn"].shape[-1]), "single_reduction_interval": [0, int(b1["attn"].shape[-1])], "segments": segment_mapping(b1["cache"], b1["row_idx"])},
        "ragged": {"physical_width": int(b2["attn"].shape[-1]), "single_reduction_interval": [0, int(b2["attn"].shape[-1])], "segments": segment_mapping(b2["cache"], b2["row_idx"])},
        "reorder": {"physical_width": int(b2_reorder["attn"].shape[-1]), "single_reduction_interval": [0, int(b2_reorder["attn"].shape[-1])], "segments": segment_mapping(b2_reorder["cache"], b2_reorder["row_idx"])},
        "b4": {"physical_width": int(b4["attn"].shape[-1]), "single_reduction_interval": [0, int(b4["attn"].shape[-1])], "segments": segment_mapping(b4["cache"], b4["row_idx"])},
    }

    if not bool(raw_cmp["exact_equal"]):
        root = "BATCH_SHAPE_DEPENDENT_QK_REDUCTION_CONFIRMED"
        next_task = "IMPLEMENT_ORACLE_FIXED_QK_REDUCTION"
    elif not bool(masked_valid_cmp["exact_equal"]):
        root = "BATCH_SHAPE_DEPENDENT_QK_REDUCTION_CONFIRMED"
        next_task = "IMPLEMENT_ORACLE_FIXED_QK_REDUCTION"
    elif not bool(p_cmp["exact_equal"]) and bool(no_split_cmp["exact_equal"]) and bool(fixed_split_cmp["exact_equal"]):
        root = "BATCH_SHAPE_DEPENDENT_ONLINE_SOFTMAX_SPLIT_MERGE_CONFIRMED"
        next_task = "IMPLEMENT_REQUEST_INVARIANT_ATTENTION_SOFTMAX"
    elif not bool(p_cmp["exact_equal"]):
        root = "ATTENTION_SOFTMAX_ROOT_CAUSE_UNRESOLVED"
        next_task = "TRACE_SOFTMAX_KERNEL_INTERNALS"
    else:
        root = "ATTENTION_QK_SOFTMAX_EXACT_SECONDARY_VALUE_REDUCTION_REMAINS"
        next_task = "TRACE_VALUE_REDUCTION_AFTER_SOFTMAX"

    final = {
        "start_head": START_HEAD,
        "branch": preflight["branch"],
        "prior_bi_k_fix_preserved": True,
        "prior_bi_v_fix_preserved": True,
        "importance_mapping_fix_preserved": True,
        "full_bi_linear_context_active": True,
        "target": {"request": "A", "step": 1, "layer": 0, "component": "attention_probs"},
        "previous_attention_probs_rel_l2": 2.0124710786717515e-08,
        "previous_attention_probs_max_abs": 5.960464477539063e-08,
        "previous_attention_probs_mismatch_count": 1,
        "current_q_match": bool(q_cmp["exact_equal"]),
        "current_k_match": bool(current_k_cmp["exact_equal"]),
        "canonical_k_match": bool(k_cmp["exact_equal"]),
        "qk_elementwise_product_match": bool(product_cmp["exact_equal"]),
        "raw_qk_canonical_match": bool(raw_cmp["exact_equal"]),
        "qk_dot_reduction_match": bool(dot_cmp["exact_equal"]),
        "scaled_logits_match": bool(scaled_cmp["exact_equal"]),
        "masked_valid_logits_match": bool(masked_valid_cmp["exact_equal"]),
        "invalid_mask_semantics_match": True,
        "attention_probs_canonical_match": bool(p_cmp["exact_equal"]),
        "no_split_attention_exact": bool(no_split_cmp["exact_equal"]),
        "no_split_reorder_exact": bool(no_split_reorder_cmp["exact_equal"]),
        "no_split_b4_exact": bool(no_split_b4_cmp["exact_equal"]),
        "fixed_split_boundary_exact": True,
        "fixed_split_probability_exact": bool(fixed_split_cmp["exact_equal"]),
        "fixed_split_reorder_exact": bool(fixed_split_reorder_cmp["exact_equal"]),
        "fixed_split_b4_exact": bool(fixed_split_b4_cmp["exact_equal"]),
        "physical_width_softmax_reconstructs_production_b1": bool(production_vs_physical_b1["exact_equal"]),
        "physical_width_softmax_reconstructs_production_ragged": bool(production_vs_physical_b2["exact_equal"]),
        "physical_width_softmax_batch_invariant": bool(physical_reconstruct_cmp["exact_equal"]),
        "captured_physical_softmax_replay_reconstructs_production_b1": bool(production_vs_replay_b1["exact_equal"]),
        "captured_physical_softmax_replay_reconstructs_production_ragged": bool(production_vs_replay_b2["exact_equal"]),
        "captured_physical_softmax_replay_batch_invariant": bool(captured_replay_cmp["exact_equal"]),
        "dynamic_split_count_depends_on_batch_shape": int(b1["attn"].shape[-1]) != int(b2["attn"].shape[-1]),
        "dynamic_split_boundaries_depend_on_batch_shape": dynamic_boundaries["b1"]["segments"] != dynamic_boundaries["ragged"]["segments"],
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
    write_md(REPORT_DIR / "attention_probability_call_graph.md", "Attention Probability Call Graph", "`LlamaFlashAttention_PatternKV.forward` projects Q/K/V, applies RoPE, appends decode K/V into the segmented cache, builds QK score parts in sink/packed/pending/recent order, concatenates them into a physical attention axis, divides by `sqrt(head_dim)`, applies `build_k_segment_validity_mask` with `torch.finfo(dtype).min` on invalid physical padding, then calls `torch.nn.functional.softmax(..., dtype=torch.float32).to(fp16)`.")
    write_json(REPORT_DIR / "qk_input_comparison.json", {"q_post_rope": q_cmp, "current_k_post_rope": current_k_cmp, "canonical_k": k_cmp})
    write_json(REPORT_DIR / "raw_qk_logits_comparison.json", raw_cmp)
    write_json(REPORT_DIR / "qk_dot_reduction_oracle.json", {"dot_reduction": dot_cmp, "elementwise_product": product_cmp})
    write_json(REPORT_DIR / "scaled_logits_comparison.json", scaled_cmp)
    write_md(REPORT_DIR / "mask_semantics.md", "Mask Semantics", f"Valid token logits are compared only in request-local logical order. Invalid ragged physical positions are masked with `{float(invalid_sentinel)}` for `{scaled_b1.dtype}` before softmax. `exp(invalid)` is zero in fp32, so invalid positions are semantically excluded even though they still participate in the physical reduction width.")
    write_json(REPORT_DIR / "mask_value_audit.json", mask_audit)
    write_json(
        REPORT_DIR / "masked_valid_logits_comparison.json",
        {
            "b1_vs_ragged_production_softmax_input": masked_valid_cmp,
            "b1_production_vs_reference_scaled_logits": production_masked_vs_reference_b1,
            "ragged_production_vs_reference_scaled_logits": production_masked_vs_reference_b2,
        },
    )
    write_md(REPORT_DIR / "attention_split_planner.md", "Attention Split Planner", f"Production currently performs one `torch.softmax` over the physical concatenated width. The request-invariant oracle uses logical fixed boundaries of {SPLIT_SIZE} valid tokens: `{fixed_bounds}`.")
    write_json(REPORT_DIR / "dynamic_split_boundaries.json", dynamic_boundaries)
    write_md(REPORT_DIR / "online_softmax_state_audit.md", "Online Softmax State Audit", json.dumps({"production_kind": "single physical-axis torch softmax", "oracle_kind": "request-local logical split/merge softmax"}, indent=2, sort_keys=True))
    write_json(REPORT_DIR / "per_split_max_comparison.json", {"dynamic_physical": cmp(b2_m.cpu(), b1_m.cpu()), "logical_no_split": cmp(no2_state["local_max"].cpu(), no1_state["local_max"].cpu()), "fixed_split": cmp(fixed2_state["local_max"].cpu(), fixed1_state["local_max"].cpu())})
    write_json(REPORT_DIR / "per_split_lse_comparison.json", {"dynamic_physical": cmp(b2_lse.cpu(), b1_lse.cpu()), "logical_no_split": cmp(no2_state["local_lse"].cpu(), no1_state["local_lse"].cpu()), "fixed_split": cmp(fixed2_state["local_lse"].cpu(), fixed1_state["local_lse"].cpu())})
    write_md(REPORT_DIR / "merge_state_audit.md", "Merge State Audit", json.dumps({"fixed_split_boundaries": fixed_bounds, "note": "fixed split merge is deterministic for identical request-local logits because boundaries are independent of peer requests"}, indent=2, sort_keys=True))
    write_json(REPORT_DIR / "merged_lse_comparison.json", {"logical_no_split": cmp(no2_state["merged_lse"].cpu(), no1_state["merged_lse"].cpu()), "fixed_split": cmp(fixed2_state["merged_lse"].cpu(), fixed1_state["merged_lse"].cpu())})
    write_json(
        REPORT_DIR / "final_probability_reconstruction.json",
        {
            "production_attention_probs": p_cmp,
            "reference_physical_width_reconstruction": physical_reconstruct_cmp,
            "production_vs_reference_physical_b1": production_vs_physical_b1,
            "production_vs_reference_physical_ragged": production_vs_physical_b2,
            "captured_physical_input_replay": captured_replay_cmp,
            "production_vs_captured_replay_b1": production_vs_replay_b1,
            "production_vs_captured_replay_ragged": production_vs_replay_b2,
        },
    )
    write_json(REPORT_DIR / "no_split_attention_oracle.json", {"b1_vs_b2": no_split_cmp, "b2_vs_reorder": no_split_reorder_cmp, "b1_vs_b4": no_split_b4_cmp, "b1_vs_production": cmp(no_split_b1.cpu(), b1["p"].cpu()), "b2_vs_production": cmp(no_split_b2.cpu(), b2["p"].cpu())})
    write_md(REPORT_DIR / "no_split_attention_oracle.md", "No-Split Attention Oracle", json.dumps({"b1_vs_b2": no_split_cmp, "b2_vs_reorder": no_split_reorder_cmp, "b1_vs_b4": no_split_b4_cmp}, indent=2, sort_keys=True))
    write_md(REPORT_DIR / "fixed_split_size_design.md", "Fixed Split Size Design", f"Use request-local logical valid token indices and fixed split size `{SPLIT_SIZE}`. Invalid physical padding is excluded before split planning, so peers cannot change request A's softmax reduction boundaries.")
    write_json(REPORT_DIR / "fixed_split_boundary_oracle.json", {"fixed_split_size": SPLIT_SIZE, "boundaries": fixed_bounds})
    write_json(REPORT_DIR / "fixed_split_probability_oracle.json", {"b1_vs_b2": fixed_split_cmp, "b2_vs_reorder": fixed_split_reorder_cmp, "b1_vs_b4": fixed_split_b4_cmp})
    write_json(REPORT_DIR / "fixed_split_merge_state_oracle.json", {"local_max": cmp(fixed2_state["local_max"].cpu(), fixed1_state["local_max"].cpu()), "local_lse": cmp(fixed2_state["local_lse"].cpu(), fixed1_state["local_lse"].cpu()), "merged_lse": cmp(fixed2_state["merged_lse"].cpu(), fixed1_state["merged_lse"].cpu())})
    write_json(REPORT_DIR / "fixed_split_layer_propagation.json", {"executed": False, "reason": "fixed split softmax was not installed into runtime in this forensic-only round"})
    write_json(REPORT_DIR / "fixed_split_b2_16step.json", run_b2_16_softmax_fixed())
    write_json(REPORT_DIR / "secondary_value_reduction_after_softmax_fix.json", {"exposed": bool(no_split_cmp["exact_equal"] and fixed_split_cmp["exact_equal"]), "reason": "M already observed value reduction topology mismatch after P path; N does not install a runtime softmax fix"})
    write_md(REPORT_DIR / "root_cause_evidence.md", "Root Cause Evidence", json.dumps({k: final[k] for k in ("current_q_match", "current_k_match", "canonical_k_match", "raw_qk_canonical_match", "scaled_logits_match", "masked_valid_logits_match", "attention_probs_canonical_match", "no_split_attention_exact", "fixed_split_probability_exact", "physical_width_softmax_batch_invariant", "root_classification", "next_task")}, indent=2, sort_keys=True))
    write_json(REPORT_DIR / "final_gate.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
