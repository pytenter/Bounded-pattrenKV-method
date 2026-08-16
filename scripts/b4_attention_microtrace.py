from __future__ import annotations

import argparse
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

import models.llama_patternkv as llama_patternkv
from bench.run_actual_model_fixed_batch_smoke import load_model, make_fixed_inputs, tensor_metrics
from bench.run_ragged_decode1_semantic_gate import nvidia_smi
from models.llama_patternkv import (
    patternkv_p2_first_divergence_trace_records,
    reset_patternkv_p2_first_divergence_trace,
    reset_patternkv_runtime_state,
)
from models.segmented_cache import (
    assemble_ragged_patternkv_cache,
    build_k_segment_validity_mask,
    deserialize_cache,
    get_total_tokens_per_request,
    k_segment_valid_lengths,
    serialize_cache,
)
from scripts.attention_qk_online_softmax_forensic import (
    canonical_k,
    canonicalize_physical,
    raw_qk_logits,
    row,
    segment_mapping,
    tensor_hash,
    trace_component,
    value_parts,
)
from scripts.attention_value_reduction_forensic import (
    canonical_v,
    fixed_reduce_fp32,
    fixed_reduce_same_dtype,
    pv_contrib,
)


REPORT_DIR = REPO_ROOT / "reports/system_b4_request_count_kernel_geometry_fix_v1"
CONTEXTS = {"A": 384, "B": 513, "C": 642, "D": 771}
TARGET = "B"


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


def tensor_info(value: torch.Tensor | None) -> dict[str, Any]:
    if value is None:
        return {"shape": None, "stride": None, "dtype": None, "device": None, "contiguous": None, "storage_offset": None}
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "contiguous": bool(value.is_contiguous()),
        "storage_offset": int(value.storage_offset()),
        "sha256": tensor_hash(value),
    }


def cmp(got: torch.Tensor | None, ref: torch.Tensor | None) -> dict[str, Any]:
    info = tensor_info(got)
    if got is None and ref is None:
        return {"exact_equal": True, "max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0, "mismatch_count": 0, **info}
    if got is None or ref is None or tuple(got.shape) != tuple(ref.shape):
        return {
            "exact_equal": False,
            "max_abs": None,
            "mean_abs": None,
            "rel_l2": None,
            "mismatch_count": None,
            "ref": tensor_info(ref),
            **info,
        }
    exact = bool(torch.equal(got, ref))
    diff = (got.detach().float() - ref.detach().float()).abs()
    return {
        "exact_equal": exact,
        "max_abs": float(diff.max().item()) if got.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if got.numel() else 0.0,
        "rel_l2": float(tensor_metrics(got, ref)["relative_l2"]) if got.numel() else 0.0,
        "mismatch_count": int((got.detach().cpu() != ref.detach().cpu()).sum().item()),
        "ref": tensor_info(ref),
        **info,
    }


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


@contextmanager
def capture_attention_path() -> Any:
    original_softmax = llama_patternkv.request_invariant_segmented_attention_softmax
    original_importance = llama_patternkv.update_value_causal_importance
    original_page = llama_patternkv.patternkv_fused_page_batch_decode
    original_qbmm_base = llama_patternkv.cuda_bmm_fA_qB_outer_with_base
    original_qbmm = llama_patternkv.cuda_bmm_fA_qB_outer
    captures: dict[str, Any] = {"masked_scores": [], "probs": [], "importance_probs": [], "packed_value": [], "packed_k_scores": []}

    def wrapped_softmax(attn_weights: torch.Tensor, cache: Any, parts: list[tuple[str, int]], *args: Any, **kwargs: Any) -> torch.Tensor:
        if not captures["masked_scores"]:
            captures["masked_scores"].append(attn_weights.detach().clone())
        out = original_softmax(attn_weights, cache, parts, *args, **kwargs)
        if not captures["probs"]:
            captures["probs"].append(out.detach().clone())
        return out

    def wrapped_importance(cache: Any, attn_weights: torch.Tensor) -> None:
        if not captures["importance_probs"]:
            captures["importance_probs"].append(attn_weights.detach().clone())
        original_importance(cache, attn_weights)

    def wrapped_qbmm_base(*args: Any, **kwargs: Any) -> torch.Tensor:
        out = original_qbmm_base(*args, **kwargs)
        if not captures["packed_k_scores"] and torch.is_tensor(out) and out.dim() == 4 and int(out.shape[2]) == 1:
            captures["packed_k_scores"].append(out.detach().clone())
        return out

    def wrapped_qbmm(*args: Any, **kwargs: Any) -> torch.Tensor:
        out = original_qbmm(*args, **kwargs)
        if not captures["packed_k_scores"] and torch.is_tensor(out) and out.dim() == 4 and int(out.shape[2]) == 1:
            captures["packed_k_scores"].append(out.detach().clone())
        return out

    def wrapped_page(attn: torch.Tensor, pools: Any) -> torch.Tensor:
        out = original_page(attn, pools)
        if not captures["packed_value"]:
            captures["packed_value"].append(
                {
                    "attn": attn.detach().clone(),
                    "out": out.detach().clone(),
                    "seq_lens": pools.metadata.seq_lens.detach().clone(),
                    "num_pages": pools.metadata.num_pages.detach().clone(),
                    "pages_per_request": int(pools.metadata.v2_page_table.shape[1]),
                    "v2_tokens": int(pools.v2_payload_pool.shape[1]),
                    "v4_tokens": int(pools.v4_payload_pool.shape[1]),
                }
            )
        return out

    llama_patternkv.request_invariant_segmented_attention_softmax = wrapped_softmax
    llama_patternkv.update_value_causal_importance = wrapped_importance
    llama_patternkv.patternkv_fused_page_batch_decode = wrapped_page
    llama_patternkv.cuda_bmm_fA_qB_outer_with_base = wrapped_qbmm_base
    llama_patternkv.cuda_bmm_fA_qB_outer = wrapped_qbmm
    try:
        yield captures
    finally:
        llama_patternkv.request_invariant_segmented_attention_softmax = original_softmax
        llama_patternkv.update_value_causal_importance = original_importance
        llama_patternkv.patternkv_fused_page_batch_decode = original_page
        llama_patternkv.cuda_bmm_fA_qB_outer_with_base = original_qbmm_base
        llama_patternkv.cuda_bmm_fA_qB_outer = original_qbmm


def run_case(model: Any, inputs: torch.Tensor, order: tuple[str, ...]) -> dict[str, Any]:
    prefills = {}
    for req in sorted(set(order)):
        idx = ord(req) - ord("A")
        prefills[req] = prefill_once(model, inputs[idx : idx + 1, : CONTEXTS[req]])
    caches = [assemble_ragged_patternkv_cache([prefills[req]["past"][layer] for req in order]) for layer in range(len(prefills[order[0]]["past"]))]
    past = tuple(serialize_cache(cache) for cache in caches)
    token = torch.stack([prefills[req]["token"] for req in order]).view(len(order))
    reset_patternkv_p2_first_divergence_trace()
    with capture_attention_path() as captures:
        with torch.inference_mode():
            out = model(input_ids=token[:, None], past_key_values=past, use_cache=True, return_dict=True)
    if not captures["masked_scores"] or not captures["probs"] or not captures["packed_value"] or not captures["packed_k_scores"]:
        raise RuntimeError("attention microtrace capture failed")
    row_idx = order.index(TARGET)
    cache = deserialize_cache(out.past_key_values[0], pattern=True)
    trace = patternkv_p2_first_divergence_trace_records()
    q = trace_component(trace, 0, "Q_POST_ROPE", row_idx)[0, :, 0, :].detach().contiguous()
    k_current = trace_component(trace, 0, "K_POST_ROPE", row_idx)[0, :, 0, :].detach().contiguous()
    v_current = trace_component(trace, 0, "V_PROJ", row_idx).view(1, 1, cache.recent_v.shape[1], cache.recent_v.shape[-1])[0, 0].detach().contiguous()
    k = canonical_k(cache, row_idx).detach().contiguous()
    values, value_meta = canonical_v(cache, row_idx)
    probs = canonical_probs(captures["probs"][0], cache, row_idx).detach().contiguous()
    masked_scores = canonicalize_physical(
        {"cache": cache, "row_idx": row_idx},
        captures["masked_scores"][0][row_idx, :, 0, :],
    ).detach().contiguous()
    raw_scores = raw_qk_logits(q, k).detach().contiguous()
    scaled_scores = (raw_scores / math.sqrt(q.shape[-1])).detach().contiguous()
    packed_len = int(k_segment_valid_lengths(cache)["packed"][row_idx].item())
    packed_scores = captures["packed_k_scores"][0][row_idx, :, 0, :packed_len].detach().contiguous()
    contrib = pv_contrib(probs, values)
    packed_capture = captures["packed_value"][0]
    packed_out = packed_capture["out"][row_idx : row_idx + 1].detach().contiguous()
    return {
        "order": list(order),
        "target": TARGET,
        "row_idx": row_idx,
        "cache": cache,
        "segment_lengths": {key: value.detach().cpu().tolist() for key, value in k_segment_valid_lengths(cache).items()},
        "total_tokens": get_total_tokens_per_request(cache).detach().cpu().tolist(),
        "mask": build_k_segment_validity_mask(cache, value_parts(cache), device=probs.device).detach().contiguous(),
        "q": q,
        "k_current": k_current,
        "v_current": v_current,
        "k": k,
        "values": values.detach().contiguous(),
        "raw_scores": raw_scores,
        "scaled_scores": scaled_scores,
        "production_packed_k_scores": packed_scores,
        "masked_scores": masked_scores,
        "probs": probs,
        "pv_contrib": contrib.detach().contiguous(),
        "reference_pre_o_same_dtype": fixed_reduce_same_dtype(contrib).detach().contiguous(),
        "reference_pre_o_fp32": fixed_reduce_fp32(contrib).detach().contiguous(),
        "production_packed_value": packed_out,
        "production_pre_o": trace_pre_o(trace, 0, row_idx).detach().contiguous(),
        "packed_kernel_geometry": {
            "batch": int(packed_capture["attn"].shape[0]),
            "heads": int(packed_capture["attn"].shape[1]),
            "attention_width": int(packed_capture["attn"].shape[-1]),
            "seq_lens": packed_capture["seq_lens"].detach().cpu().tolist(),
            "num_pages": packed_capture["num_pages"].detach().cpu().tolist(),
            "pages_per_request": int(packed_capture["pages_per_request"]),
            "v2_tokens": int(packed_capture["v2_tokens"]),
            "v4_tokens": int(packed_capture["v4_tokens"]),
            "grid_blocks": [int(packed_capture["attn"].shape[0] * packed_capture["attn"].shape[1]), int(cache.recent_v.shape[-1]), 1],
            "threads": [256, 1, 1],
        },
        "value_meta": value_meta,
    }


def compare(good: dict[str, Any], bad: dict[str, Any]) -> dict[str, Any]:
    components = [
        ("request_local_q", "q"),
        ("request_local_current_k", "k_current"),
        ("request_local_current_v", "v_current"),
        ("request_local_k_cache", "k"),
        ("request_local_v_cache", "values"),
        ("qk_raw_scores", "raw_scores"),
        ("scaled_scores", "scaled_scores"),
        ("production_packed_k_scores", "production_packed_k_scores"),
        ("masked_scores", "masked_scores"),
        ("softmax_probabilities", "probs"),
        ("probability_times_v", "pv_contrib"),
        ("reference_pre_o_same_dtype", "reference_pre_o_same_dtype"),
        ("reference_pre_o_fp32", "reference_pre_o_fp32"),
        ("production_packed_value_kernel", "production_packed_value"),
        ("merged_head_pre_o_proj", "production_pre_o"),
    ]
    rows = []
    first_bad = None
    for label, key in components:
        row_cmp = {"component": label, **cmp(bad[key].cpu(), good[key].cpu())}
        rows.append(row_cmp)
        if first_bad is None and not bool(row_cmp["exact_equal"]):
            first_bad = row_cmp
    return {
        "first_bad_attention_subcomponent": first_bad,
        "rows": rows,
        "valid_length_match": good["segment_lengths"] == bad["segment_lengths"],
        "total_tokens_match": good["total_tokens"][good["row_idx"]] == bad["total_tokens"][bad["row_idx"]],
        "mask_request_local_match": cmp(bad["mask"][bad["row_idx"]].cpu(), good["mask"][good["row_idx"]].cpu()),
        "good_kernel_geometry": good["packed_kernel_geometry"],
        "bad_kernel_geometry": bad["packed_kernel_geometry"],
    }


def parse_order(value: str) -> tuple[str, ...]:
    out = tuple(item.strip() for item in value.split(",") if item.strip())
    if TARGET not in out:
        raise ValueError(f"order must include target request {TARGET}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--good", default="A,B")
    parser.add_argument("--bad", default="A,B,C")
    args = parser.parse_args()
    set_env()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device(args.device)
    tokenizer, _config, model = load_model(dtype=torch.float16, device=device)
    inputs = make_fixed_inputs(tokenizer, batch=4, context=771, device=device)
    good_order = parse_order(args.good)
    bad_order = parse_order(args.bad)
    good = run_case(model, inputs, good_order)
    bad = run_case(model, inputs, bad_order)
    result = compare(good, bad)
    payload = {
        "target": {"request": TARGET, "step": 1, "layer": 0, "good": list(good_order), "bad": list(bad_order)},
        "environment": {
            "branch": git(["branch", "--show-current"]),
            "head": git(["rev-parse", "HEAD"]),
            "status_short": git(["status", "--short"]),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "nvidia_smi": nvidia_smi(),
        },
        "comparison": result,
        "good_request_b_geometry": {
            "segments": segment_mapping(good["cache"], good["row_idx"]),
            "total_tokens": good["total_tokens"],
            "segment_lengths": good["segment_lengths"],
            "value_meta": good["value_meta"],
        },
        "bad_request_b_geometry": {
            "segments": segment_mapping(bad["cache"], bad["row_idx"]),
            "total_tokens": bad["total_tokens"],
            "segment_lengths": bad["segment_lengths"],
            "value_meta": bad["value_meta"],
        },
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(REPORT_DIR / "attention_microtrace_b_request_step1_layer0.json", payload)
    write_md(
        REPORT_DIR / "attention_microtrace_summary.md",
        "B4 Attention Microtrace",
        json.dumps(
            {
                "FIRST_BAD_ATTENTION_SUBCOMPONENT": result["first_bad_attention_subcomponent"],
                "B2_KERNEL_GEOMETRY": result["good_kernel_geometry"],
                "B3_KERNEL_GEOMETRY": result["bad_kernel_geometry"],
            },
            indent=2,
            sort_keys=True,
        ),
    )
    print(json.dumps(payload["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
