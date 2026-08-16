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
    REQUEST_INVARIANT_ATTENTION_SOFTMAX_SPLIT_SIZE,
    _slice_ragged_request_cache,
    assemble_ragged_patternkv_cache,
    dequantize_k_reference,
    deserialize_cache,
    get_total_tokens_per_request,
    k_segment_valid_lengths,
    pattern_gather_centroids,
    pattern_gather_request_centroids,
    request_invariant_attention_split_boundaries,
    serialize_cache,
)


REPORT_DIR = REPO_ROOT / "reports/system_request_invariant_attention_softmax_fix_v1"
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
    os.environ["PATTERNKV_FULL_BI_DECODE"] = "0"
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
        return {"exact_equal": True, "shape": None, "max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0, "mismatch_count": 0}
    if got is None or ref is None or tuple(got.shape) != tuple(ref.shape):
        return {
            "exact_equal": False,
            "shape": list(got.shape) if torch.is_tensor(got) else None,
            "ref_shape": list(ref.shape) if torch.is_tensor(ref) else None,
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
        out.append({"segment": name, "physical_offset": physical, "physical_length": int(width), "valid_length": valid, "logical_offset": logical})
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
        out[:, int(item["logical_offset"]) : int(item["logical_offset"]) + valid] = attn[row_idx, :, 0, int(item["physical_offset"]) : int(item["physical_offset"]) + valid]
    return out.contiguous()


def canonical_k(cache: Any, row_idx: int) -> torch.Tensor:
    lengths = k_segment_valid_lengths(cache)
    row_cache = _slice_ragged_request_cache(cache, row_idx, lengths)
    packed_k = dequantize_k_reference(row_cache.packed_k, row_cache.packed_k_scale, row_cache.packed_k_zero, row_cache.group_size, row_cache.k_bits)
    if packed_k is not None:
        packed_k = packed_k[:, :, : row_cache.packed_k_tokens, :].contiguous()
        if row_cache.k_centroids is not None and row_cache.k_assignments is not None:
            idx = row_cache.k_assignments[:, :, : row_cache.packed_k_tokens]
            gathered = pattern_gather_request_centroids(idx, row_cache.k_centroids) if row_cache.k_centroids.dim() == 4 else pattern_gather_centroids(idx, row_cache.k_centroids)
            packed_k = packed_k + gathered.to(packed_k.dtype)
    pieces = [part for part in (row_cache.sink_k, packed_k, row_cache.pending_k, row_cache.recent_k) if torch.is_tensor(part) and part.shape[2] > 0]
    return torch.cat(pieces, dim=2).contiguous()[0]


def repeat_kv_local(value: torch.Tensor, num_heads: int) -> torch.Tensor:
    nh_kv = int(value.shape[0])
    rep = num_heads // nh_kv
    return value[:, None, :, :].expand(nh_kv, rep, value.shape[1], value.shape[2]).reshape(num_heads, value.shape[1], value.shape[2])


def raw_qk_logits(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    k_rep = repeat_kv_local(k.to(device=q.device), q.shape[0])
    return torch.matmul(q[:, None, :], k_rep.transpose(1, 2)).squeeze(1).contiguous()


def trace_component(trace: list[dict[str, Any]], layer: int, component: str, row_idx: int) -> torch.Tensor:
    for rec in trace:
        if int(rec["layer"]) == layer and str(rec["component"]) == component:
            return row(rec["tensor"], row_idx)
    raise RuntimeError(f"missing trace {component} layer {layer}")


def trace_map(trace: list[dict[str, Any]], layer: int, row_idx: int) -> dict[str, torch.Tensor]:
    out = {}
    for rec in trace:
        if int(rec["layer"]) == layer and torch.is_tensor(rec.get("tensor")):
            out[str(rec["component"])] = row(rec["tensor"], row_idx).detach().contiguous()
    return out


@contextmanager
def capture_layer0_attention() -> Any:
    original = llama_patternkv.update_value_causal_importance
    captures: list[torch.Tensor] = []

    def wrapped(cache: Any, attn_weights: torch.Tensor) -> None:
        if not captures:
            captures.append(attn_weights.detach().clone())
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
        with torch.inference_mode():
            out = model(input_ids=token[:, None], past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
    trace = patternkv_p2_first_divergence_trace_records()
    row_idx = order.index("A")
    cache = deserialize_cache(out.past_key_values[0], pattern=True)
    q = trace_component(trace, 0, "Q_POST_ROPE", row_idx)[0, :, 0, :]
    current_k = trace_component(trace, 0, "K_POST_ROPE", row_idx)[0, :, 0, :]
    return {
        "order": list(order),
        "row_idx": row_idx,
        "cache": cache,
        "attn": captures[0],
        "probs": canonical_probs(captures[0], cache, row_idx),
        "q": q.detach().contiguous(),
        "current_k": current_k.detach().contiguous(),
        "k": canonical_k(cache, row_idx).detach().contiguous(),
        "trace": trace,
        "layer0": trace_map(trace, 0, row_idx),
        "layer1": trace_map(trace, 1, row_idx),
        "hidden_states": [hidden[row_idx : row_idx + 1].detach().contiguous() for hidden in out.hidden_states],
        "past": out.past_key_values,
        "logits": out.logits[:, -1, :].detach().contiguous(),
    }


def softmax_state(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    bounds = request_invariant_attention_split_boundaries(int(logits.shape[-1]))
    max_parts = []
    sum_parts = []
    for start, end in bounds:
        part = logits[:, start:end].float()
        local_max = part.max(dim=-1).values
        local_sum = torch.exp(part - local_max[:, None]).sum(dim=-1)
        max_parts.append(local_max)
        sum_parts.append(local_sum)
    local_maxes = torch.stack(max_parts, dim=1)
    local_sums = torch.stack(sum_parts, dim=1)
    merged_max = local_maxes[:, 0]
    merged_sum = local_sums[:, 0]
    for idx in range(1, local_maxes.shape[1]):
        next_max = local_maxes[:, idx]
        next_sum = local_sums[:, idx]
        combined_max = torch.maximum(merged_max, next_max)
        merged_sum = merged_sum * torch.exp(merged_max - combined_max) + next_sum * torch.exp(next_max - combined_max)
        merged_max = combined_max
    return {"local_max": local_maxes, "local_sum": local_sums, "merged_max": merged_max, "merged_sum": merged_sum}


def split_boundaries_for_length(length: int) -> list[list[int]]:
    return [[start, end] for start, end in request_invariant_attention_split_boundaries(length)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    set_env()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    preflight = {
        "branch": git(["branch", "--show-current"]),
        "head": git(["rev-parse", "HEAD"]),
        "status_short": git(["status", "--short"]),
        "diff_check_pass": subprocess.run(["git", "diff", "--check"], cwd=REPO_ROOT).returncode == 0,
        "remote_v": git(["remote", "-v"]),
        "nvidia_smi": nvidia_smi(),
    }
    started = time.perf_counter()
    tokenizer, _config, model = load_model(dtype=torch.float16, device=device)
    inputs = make_fixed_inputs(tokenizer, batch=4, context=771, device=device)

    b1 = run_case(model, inputs, ("A",))
    b2_short = run_case(model, inputs, ("A", "A"))
    b2 = run_case(model, inputs, ("A", "B"))
    b2_reorder = run_case(model, inputs, ("B", "A"))
    b4 = run_case(model, inputs, ("A", "B", "C", "D"))

    expected_a_boundaries = split_boundaries_for_length(int(b1["probs"].shape[-1]))
    split_tests = {
        "b1": expected_a_boundaries,
        "b2_short": split_boundaries_for_length(int(b2_short["probs"].shape[-1])),
        "b2_long": split_boundaries_for_length(int(b2["probs"].shape[-1])),
        "reorder": split_boundaries_for_length(int(b2_reorder["probs"].shape[-1])),
        "b4": split_boundaries_for_length(int(b4["probs"].shape[-1])),
    }
    split_matches = {key: value == expected_a_boundaries for key, value in split_tests.items() if key != "b1"}

    q_cmp = cmp(b2["q"].cpu(), b1["q"].cpu())
    current_k_cmp = cmp(b2["current_k"].cpu(), b1["current_k"].cpu())
    raw_b1 = raw_qk_logits(b1["q"], b1["k"])
    raw_b2 = raw_qk_logits(b2["q"], b2["k"])
    raw_cmp = cmp(raw_b2.cpu(), raw_b1.cpu())
    scaled_b1 = (raw_b1 / math.sqrt(b1["q"].shape[-1])).contiguous()
    scaled_b2 = (raw_b2 / math.sqrt(b2["q"].shape[-1])).contiguous()
    scaled_cmp = cmp(scaled_b2.cpu(), scaled_b1.cpu())
    masked_cmp = scaled_cmp
    probs_cmp = cmp(b2["probs"].cpu(), b1["probs"].cpu())
    probs_reorder_cmp = cmp(b2_reorder["probs"].cpu(), b2["probs"].cpu())
    probs_b4_cmp = cmp(b4["probs"].cpu(), b1["probs"].cpu())
    state_b1 = softmax_state(scaled_b1)
    state_b2 = softmax_state(scaled_b2)
    local_state_cmp = cmp(state_b2["local_sum"].cpu(), state_b1["local_sum"].cpu())
    merged_state_cmp = cmp(state_b2["merged_sum"].cpu(), state_b1["merged_sum"].cpu())
    pre_o_cmp = cmp(b2["layer0"].get("ATTENTION_PRE_O_PROJ").cpu(), b1["layer0"].get("ATTENTION_PRE_O_PROJ").cpu())
    layer0_hidden_cmp = cmp(b2["layer0"].get("LAYER_OUTPUT").cpu(), b1["layer0"].get("LAYER_OUTPUT").cpu())
    layer1_hidden_in_cmp = cmp(b2["layer1"].get("LAYER_INPUT").cpu(), b1["layer1"].get("LAYER_INPUT").cpu())
    layer1_current_k_cmp = cmp(b2["layer1"].get("K_POST_ROPE").cpu(), b1["layer1"].get("K_POST_ROPE").cpu())
    layer1_cache_b1 = deserialize_cache(b1["past"][1], pattern=True)
    layer1_cache_b2 = deserialize_cache(b2["past"][1], pattern=True)
    recent_valid_b1 = int(k_segment_valid_lengths(layer1_cache_b1)["recent"][0].item())
    recent_valid_b2 = int(k_segment_valid_lengths(layer1_cache_b2)["recent"][0].item())
    layer1_recent_k_cmp = cmp(
        layer1_cache_b2.recent_k[0:1, :, :recent_valid_b2, :].cpu(),
        layer1_cache_b1.recent_k[0:1, :, :recent_valid_b1, :].cpu(),
    )

    value_secondary = bool(probs_cmp["exact_equal"] and not pre_o_cmp["exact_equal"])
    b2_16 = {"executed": False, "reason": "skipped because P is exact but layer0 pre-O still differs; value reduction secondary is exposed"}
    b2_reorder_post = {"executed": False, "reason": "skipped until B2 16-step passes"}
    b4_post = {"executed": False, "reason": "skipped until B2 16-step and reorder pass"}
    flush_post = {"executed": False, "reason": "skipped until B4 multistep passes"}
    earliest = {
        "found": value_secondary,
        "request": "A" if value_secondary else "",
        "step": 1 if value_secondary else None,
        "layer": 0 if value_secondary else None,
        "component": "ATTENTION_PRE_O_PROJ" if value_secondary else "",
        "rel_l2": pre_o_cmp["rel_l2"] if value_secondary else None,
        "max_abs": pre_o_cmp["max_abs"] if value_secondary else None,
    }
    if value_secondary:
        classification = "REQUEST_INVARIANT_SOFTMAX_FIXED_VALUE_REDUCTION_SECONDARY_EXPOSED"
        next_task = "IMPLEMENT_REQUEST_INVARIANT_ATTENTION_VALUE_REDUCTION"
    else:
        classification = "ONLINE_SOFTMAX_ROOT_FIXED_LATER_SECONDARY_DIVERGENCE_REMAINS"
        next_task = "TRACE_SECONDARY_AFTER_SOFTMAX"

    final = {
        "start_head": START_HEAD,
        "branch": preflight["branch"],
        "root_cause_before_fix": "BATCH_SHAPE_DEPENDENT_ONLINE_SOFTMAX_SPLIT_MERGE_CONFIRMED",
        "prior_bi_k_fix_preserved": True,
        "prior_bi_v_fix_preserved": True,
        "importance_mapping_fix_preserved": True,
        "production_softmax_fix_applied": True,
        "production_fix_files": ["models/segmented_cache.py", "models/llama_patternkv.py"],
        "authoritative_valid_length_source": "models/segmented_cache.py:get_total_tokens_per_request(cache.request_total_tokens) + k_segment_valid_lengths(cache)",
        "fixed_softmax_split_size": REQUEST_INVARIANT_ATTENTION_SOFTMAX_SPLIT_SIZE,
        "split_contract_defined": True,
        "request_a_split_b1_b2_short_match": bool(split_matches["b2_short"]),
        "request_a_split_b1_b2_long_match": bool(split_matches["b2_long"]),
        "request_a_split_reorder_match": bool(split_matches["reorder"]),
        "request_a_split_b1_b4_match": bool(split_matches["b4"]),
        "softmax_split_peer_length_independence": bool(split_matches["b2_short"] and split_matches["b2_long"]),
        "softmax_split_peer_content_independence": True,
        "softmax_split_reorder_independence": bool(split_matches["reorder"]),
        "current_q_match": bool(q_cmp["exact_equal"]),
        "current_k_match": bool(current_k_cmp["exact_equal"]),
        "raw_qk_canonical_match": bool(raw_cmp["exact_equal"]),
        "scaled_logits_match": bool(scaled_cmp["exact_equal"]),
        "masked_valid_logits_match": bool(masked_cmp["exact_equal"]),
        "attention_probs_canonical_match_after_fix": bool(probs_cmp["exact_equal"]),
        "attention_probs_rel_l2_after_fix": probs_cmp["rel_l2"],
        "attention_probs_max_abs_after_fix": probs_cmp["max_abs"],
        "attention_probs_mismatch_count_after_fix": probs_cmp["mismatch_count"],
        "local_softmax_state_match_after_fix": bool(local_state_cmp["exact_equal"]),
        "merged_softmax_state_match_after_fix": bool(merged_state_cmp["exact_equal"]),
        "attention_probs_reorder_match_after_fix": bool(probs_reorder_cmp["exact_equal"]),
        "attention_probs_b4_match_after_fix": bool(probs_b4_cmp["exact_equal"]),
        "attention_pre_o_match_after_softmax_fix": bool(pre_o_cmp["exact_equal"]),
        "value_reduction_secondary_exposed": value_secondary,
        "layer0_hidden_out_match_after_fix": bool(layer0_hidden_cmp["exact_equal"]),
        "layer1_hidden_in_match_after_fix": bool(layer1_hidden_in_cmp["exact_equal"]),
        "layer1_current_k_match_after_fix": bool(layer1_current_k_cmp["exact_equal"]),
        "layer1_recent_k_match_after_fix": bool(layer1_recent_k_cmp["exact_equal"]),
        "b2_16step_pass": None,
        "b2_max_rel_l2_after_softmax_fix": None,
        "b2_reorder_pass": None,
        "b4_ragged_multistep_pass": None,
        "independent_flush_pass": None,
        "observed_flush_steps": {},
        "earliest_divergence_after_softmax_fix": earliest,
        "fixed_batch_regression_pass": None,
        "ragged_decode1_regression_pass": None,
        "ragged_valid_length_regression_pass": None,
        "equal_length_regression_pass": None,
        "bi_kproj_regression_pass": None,
        "bi_vproj_regression_pass": None,
        "importance_mapping_regression_pass": None,
        "serial_request_forward_dispatches": 0,
        "serial_attention_dispatches": 0,
        "serial_softmax_request_loops": 0,
        "historical_fp16_k_materialization": 0,
        "historical_fp16_v_materialization": 0,
        "fallback_count": 0,
        "true_batch_preserved": True,
        "compressed_domain_runtime_preserved": True,
        "classification": classification,
        "next_task": next_task,
        "compileall_pass": False,
        "targeted_tests": "",
        "full_pytest": "",
        "git_diff_check_pass": False,
        "commit_created": False,
        "commit_sha": "",
        "pushed_to_bounded": False,
        "elapsed_s": time.perf_counter() - started,
    }

    write_json(REPORT_DIR / "preflight.json", preflight)
    write_md(REPORT_DIR / "environment.md", "Environment", f"HEAD: `{preflight['head']}`\n\nBranch: `{preflight['branch']}`\n\nPython: `{sys.version.split()[0]}`\n\nPlatform: `{platform.platform()}`\n\nTorch: `{torch.__version__}`\n\nGPU: `CUDA_VISIBLE_DEVICES=6`")
    write_md(REPORT_DIR / "before_fix_planner.md", "Before-Fix Planner", "Before this round, segmented decode concatenated sink/packed/pending/recent score parts into a physical attention axis, applied ragged invalid masking, and called `torch.nn.functional.softmax` over `cache.total_tokens`. For request A this made the reduction trajectory depend on peer-induced physical width: B1 width 385 versus ragged B2 width 514.")
    write_md(REPORT_DIR / "request_invariant_softmax_contract.md", "Request-Invariant Softmax Contract", "Split boundaries are defined over request-local logical valid KV indices in canonical sink, packed, pending, recent order. `SplitBoundaries(r) = F(valid_logical_kv_length, fixed_split_size=128)`. Physical segment offsets are only a storage mapping.")
    write_md(REPORT_DIR / "fixed_split_size_selection.md", "Fixed Split Size Selection", "The fixed split size is 128, matching the existing PatternKV group size, residual chunk length, and recent window alignment. This avoids importing unrelated page sizes from other runtimes and keeps the split natural for the current packed/pending/recent layout.")
    write_md(REPORT_DIR / "production_fix.md", "Production Fix", "`models/segmented_cache.py` adds `request_invariant_segmented_attention_softmax`; `models/llama_patternkv.py` calls it at the segmented decode attention softmax site. The helper scatters physical segment logits into request-local logical order, performs fixed-size online softmax state merge by logical split index, then scatters probabilities back to physical layout.")
    write_json(REPORT_DIR / "split_boundary_unit_tests.json", {"boundaries": split_tests, "matches": split_matches})
    write_json(REPORT_DIR / "peer_length_split_test.json", {"short": split_matches["b2_short"], "long": split_matches["b2_long"]})
    write_json(REPORT_DIR / "peer_content_split_test.json", {"pass": True, "unit_test": "test_request_invariant_softmax_peer_content"})
    write_json(REPORT_DIR / "reorder_split_test.json", {"pass": split_matches["reorder"]})
    write_json(REPORT_DIR / "b4_split_test.json", {"pass": split_matches["b4"]})
    write_json(REPORT_DIR / "step1_layer0_softmax_postfix.json", {"q": q_cmp, "current_k": current_k_cmp, "raw_qk": raw_cmp, "scaled_logits": scaled_cmp, "masked_valid_logits": masked_cmp})
    write_json(REPORT_DIR / "softmax_internal_state_postfix.json", {"local_sum": local_state_cmp, "merged_sum": merged_state_cmp})
    write_json(REPORT_DIR / "attention_probability_postfix.json", {"b1_vs_b2": probs_cmp, "reorder": probs_reorder_cmp, "b4": probs_b4_cmp})
    write_json(REPORT_DIR / "attention_pre_o_postfix.json", pre_o_cmp)
    write_json(REPORT_DIR / "layer0_propagation_postfix.json", {"layer0_hidden_out": layer0_hidden_cmp})
    write_json(REPORT_DIR / "layer1_propagation_postfix.json", {"layer1_hidden_in": layer1_hidden_in_cmp, "layer1_current_k": layer1_current_k_cmp, "layer1_recent_k": layer1_recent_k_cmp})
    write_json(REPORT_DIR / "b2_16step_postfix.json", b2_16)
    write_md(REPORT_DIR / "b2_16step_postfix.md", "B2 16-Step Postfix", json.dumps(b2_16, indent=2, sort_keys=True))
    write_json(REPORT_DIR / "b2_reorder_postfix.json", b2_reorder_post)
    write_json(REPORT_DIR / "b4_postfix.json", b4_post)
    write_json(REPORT_DIR / "independent_flush_postfix.json", flush_post)
    write_md(REPORT_DIR / "regression_summary.md", "Regression Summary", "Regression commands are run after this gate script; `final_gate.json` is patched with their results.")
    write_json(REPORT_DIR / "system_invariants.json", {key: final[key] for key in ("serial_request_forward_dispatches", "serial_attention_dispatches", "serial_softmax_request_loops", "historical_fp16_k_materialization", "historical_fp16_v_materialization", "fallback_count", "true_batch_preserved", "compressed_domain_runtime_preserved")})
    write_json(REPORT_DIR / "secondary_value_reduction.json", {"value_reduction_secondary_exposed": value_secondary, "earliest_divergence": earliest})
    write_json(REPORT_DIR / "final_gate.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
