from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.run_actual_model_fixed_batch_smoke import load_model, make_fixed_inputs, tensor_metrics
from bench.run_ragged_decode1_semantic_gate import nvidia_smi
from models.llama_patternkv import apply_rotary_pos_emb, patternkv_use_bi_prefill_kproj, reset_patternkv_runtime_state
from models.segmented_cache import (
    assemble_ragged_patternkv_cache,
    deserialize_cache,
    get_total_tokens_per_request,
    serialize_cache,
)
from quant.batch_invariant_kproj import (
    batch_invariant_k_projection,
    batch_invariant_kproj_available,
    batch_invariant_kproj_counters,
    prefill_proj_mode,
    reset_batch_invariant_kproj_counters,
    selected_backend as bi_kproj_selected_backend,
)


START_HEAD = "cc50fdc513181d2137438cc6a7c0dd8322ccf767"
REPORT_DIR = REPO_ROOT / "reports/system_step1_layer0_kpath_forensic_v1"
FORENSICS_DIR = REPO_ROOT / "forensics/step1_layer0_k_path"


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


def sha(t: torch.Tensor | None) -> str | None:
    if t is None:
        return None
    cpu = t.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tuple(cpu.shape)).encode())
    h.update(str(cpu.dtype).encode())
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def first_diff_index(got: torch.Tensor, ref: torch.Tensor) -> list[int] | None:
    if torch.equal(got, ref):
        return None
    idx = torch.nonzero((got.detach().cpu() != ref.detach().cpu()).reshape(-1), as_tuple=False)
    if not int(idx.numel()):
        return None
    flat = int(idx[0].item())
    out = []
    for dim in reversed(list(got.shape)):
        out.append(flat % dim)
        flat //= dim
    return list(reversed(out))


def compare_tensors(got: torch.Tensor | None, ref: torch.Tensor | None) -> dict[str, Any]:
    if got is None and ref is None:
        return {
            "exact_equal": True,
            "sha256": None,
            "ref_sha256": None,
            "shape": None,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "rel_l2": 0.0,
            "first_diff_index": None,
            "mismatch_count": 0,
        }
    if got is None or ref is None or tuple(got.shape) != tuple(ref.shape):
        return {
            "exact_equal": False,
            "sha256": sha(got),
            "ref_sha256": sha(ref),
            "shape": list(got.shape) if torch.is_tensor(got) else None,
            "ref_shape": list(ref.shape) if torch.is_tensor(ref) else None,
            "max_abs": None,
            "mean_abs": None,
            "rel_l2": None,
            "first_diff_index": None,
            "mismatch_count": None,
        }
    exact = bool(torch.equal(got, ref))
    diff = (got.float() - ref.float()).abs()
    return {
        "exact_equal": exact,
        "sha256": sha(got),
        "ref_sha256": sha(ref),
        "shape": list(got.shape),
        "stride": list(got.stride()),
        "dtype": str(got.dtype),
        "numel": int(got.numel()),
        "max_abs": float(diff.max().item()) if got.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if got.numel() else 0.0,
        "rel_l2": float(tensor_metrics(got, ref)["relative_l2"]) if got.numel() else 0.0,
        "first_diff_index": first_diff_index(got, ref),
        "mismatch_count": int((got.detach().cpu() != ref.detach().cpu()).sum().item()),
    }


def tensor_summary(t: torch.Tensor | None) -> dict[str, Any]:
    if t is None:
        return {"present": False}
    return {
        "present": True,
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "sha256": sha(t),
    }


def row(t: torch.Tensor | None, idx: int) -> torch.Tensor | None:
    if t is None:
        return None
    return t[idx : idx + 1].detach().contiguous().cpu()


def prefill_once(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    reset_batch_invariant_kproj_counters()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    return {
        "past": out.past_key_values,
        "next_token": out.logits[:, -1, :].argmax(dim=-1),
        "kproj_counters": batch_invariant_kproj_counters(),
    }


def install_kpath_hooks(model: Any) -> tuple[dict[str, Any], list[Any]]:
    traces: dict[str, Any] = {}
    handles = []
    layer0 = model.model.layers[0]
    attn = layer0.self_attn

    def embed_hook(_module: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        traces["tokens_at_embedding"] = inputs[0].detach().clone() if inputs and torch.is_tensor(inputs[0]) else None
        traces["embedding"] = output.detach().clone()

    def layer_pre_hook(_module: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden = inputs[0] if inputs else kwargs.get("hidden_states")
        if torch.is_tensor(hidden):
            traces["layer0_hidden_in"] = hidden.detach().clone()
        position_ids = kwargs.get("position_ids")
        if torch.is_tensor(position_ids):
            traces["layer0_position_ids"] = position_ids.detach().clone()
        attention_mask = kwargs.get("attention_mask")
        traces["layer0_attention_mask"] = attention_mask.detach().clone() if torch.is_tensor(attention_mask) else None

    def norm_hook(_module: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        if inputs and torch.is_tensor(inputs[0]):
            traces["rmsnorm_input"] = inputs[0].detach().clone()
        traces["rmsnorm_output"] = output.detach().clone()

    def attn_pre_hook(_module: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden = inputs[0] if inputs else kwargs.get("hidden_states")
        if torch.is_tensor(hidden):
            traces["rope_input_hidden_states"] = hidden.detach().clone()
        position_ids = kwargs.get("position_ids")
        if torch.is_tensor(position_ids):
            traces["attn_position_ids"] = position_ids.detach().clone()
        attention_mask = kwargs.get("attention_mask")
        traces["attn_attention_mask"] = attention_mask.detach().clone() if torch.is_tensor(attention_mask) else None

    handles.append(model.model.embed_tokens.register_forward_hook(embed_hook))
    handles.append(layer0.register_forward_pre_hook(layer_pre_hook, with_kwargs=True))
    handles.append(layer0.input_layernorm.register_forward_hook(norm_hook))
    handles.append(attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True))
    for name in ("q_proj", "k_proj", "v_proj"):
        module = getattr(attn, name)

        def proj_hook(module: Any, inputs: tuple[Any, ...], output: torch.Tensor, *, name: str = name) -> None:
            traces[f"{name}_input"] = inputs[0].detach().clone() if inputs and torch.is_tensor(inputs[0]) else None
            traces[name] = output.detach().clone()
            traces[f"{name}_module"] = {
                "class": type(module).__name__,
                "weight_shape": list(module.weight.shape),
                "weight_stride": list(module.weight.stride()),
                "weight_dtype": str(module.weight.dtype),
                "bias": getattr(module, "bias", None) is not None,
            }

        handles.append(module.register_forward_hook(proj_hook))
    return traces, handles


def post_rope_outputs(model: Any, traces: dict[str, Any]) -> dict[str, torch.Tensor]:
    attn = model.model.layers[0].self_attn
    if "k_proj" not in traces:
        traces["k_proj_input"] = traces["rope_input_hidden_states"]
        traces["k_proj"] = batch_invariant_k_projection(
            traces["rope_input_hidden_states"],
            attn.k_proj.weight,
            getattr(attn.k_proj, "bias", None),
            backend=bi_kproj_selected_backend(),
        ).detach().clone()
        traces["k_proj_module"] = {
            "class": type(attn.k_proj).__name__,
            "weight_shape": list(attn.k_proj.weight.shape),
            "weight_stride": list(attn.k_proj.weight.stride()),
            "weight_dtype": str(attn.k_proj.weight.dtype),
            "bias": getattr(attn.k_proj, "bias", None) is not None,
            "captured_from_bi_projection": True,
        }
    if "v_proj" not in traces:
        traces["v_proj_input"] = traces["rope_input_hidden_states"]
        traces["v_proj"] = batch_invariant_k_projection(
            traces["rope_input_hidden_states"],
            attn.v_proj.weight,
            getattr(attn.v_proj, "bias", None),
            backend=bi_kproj_selected_backend(),
        ).detach().clone()
        traces["v_proj_module"] = {
            "class": type(attn.v_proj).__name__,
            "weight_shape": list(attn.v_proj.weight.shape),
            "weight_stride": list(attn.v_proj.weight.stride()),
            "weight_dtype": str(attn.v_proj.weight.dtype),
            "bias": getattr(attn.v_proj, "bias", None) is not None,
            "captured_from_bi_projection": True,
        }
    position_ids = traces["attn_position_ids"]
    bsz, q_len, _ = traces["k_proj"].shape
    key = traces["k_proj"].view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    query = traces["q_proj"].view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    value = traces["v_proj"].view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    cos, sin = attn.rotary_emb(value, position_ids)
    query_rope, key_rope = apply_rotary_pos_emb(query, key, cos, sin, position_ids)
    return {"query_rope": query_rope, "key_rope": key_rope, "cos": cos, "sin": sin}


def decode_with_trace(model: Any, input_ids: torch.Tensor, past: Any, *, active_row: int, label: str) -> dict[str, Any]:
    reset_batch_invariant_kproj_counters()
    cache = deserialize_cache(past[0], pattern=True)
    inferred_positions = get_total_tokens_per_request(cache, device=input_ids.device).view(-1, 1).detach().clone()
    traces, handles = install_kpath_hooks(model)
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids[:, None], past_key_values=past, use_cache=True, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    counters = batch_invariant_kproj_counters()
    rope = post_rope_outputs(model, traces)
    return {
        "label": label,
        "active_row": active_row,
        "full_batch_size": int(input_ids.shape[0]),
        "input_ids_full": input_ids.detach().contiguous().cpu(),
        "input_ids_row": row(input_ids, active_row),
        "inferred_position_ids_full": inferred_positions.detach().contiguous().cpu(),
        "tokens_at_embedding": row(traces.get("tokens_at_embedding"), active_row),
        "embedding": row(traces.get("embedding"), active_row),
        "layer0_hidden_in": row(traces.get("layer0_hidden_in"), active_row),
        "rmsnorm_input": row(traces.get("rmsnorm_input"), active_row),
        "rmsnorm_output": row(traces.get("rmsnorm_output"), active_row),
        "raw_qproj": row(traces.get("q_proj"), active_row),
        "raw_kproj": row(traces.get("k_proj"), active_row),
        "raw_vproj": row(traces.get("v_proj"), active_row),
        "qproj_input": row(traces.get("q_proj_input"), active_row),
        "kproj_input": row(traces.get("k_proj_input"), active_row),
        "vproj_input": row(traces.get("v_proj_input"), active_row),
        "attn_position_ids": row(traces.get("attn_position_ids"), active_row),
        "layer0_position_ids": row(traces.get("layer0_position_ids"), active_row),
        "attn_attention_mask": row(traces.get("attn_attention_mask"), active_row),
        "layer0_attention_mask": row(traces.get("layer0_attention_mask"), active_row),
        "rope_input_k": row(
            traces.get("k_proj").view(
                traces["k_proj"].shape[0],
                traces["k_proj"].shape[1],
                model.model.layers[0].self_attn.num_key_value_heads,
                model.model.layers[0].self_attn.head_dim,
            ).transpose(1, 2),
            active_row,
        ),
        "rope_output_current_k": row(rope["key_rope"], active_row),
        "rope_output_current_q": row(rope["query_rope"], active_row),
        "cos": row(rope["cos"], active_row),
        "sin": row(rope["sin"], active_row),
        "kproj_runtime": {
            "counters_after_decode": counters,
            "used_bi_kproj": counters.get("bi_kproj_calls", 0) > 0 or counters.get("bi_decode_kproj_calls", 0) > 0,
            "normal_decode_kproj_calls": counters.get("normal_decode_kproj_calls", 0),
            "input_shape_full": list(traces["k_proj_input"].shape),
            "input_stride_full": list(traces["k_proj_input"].stride()),
            "output_shape_full": list(traces["k_proj"].shape),
            "output_stride_full": list(traces["k_proj"].stride()),
            "module": traces["k_proj_module"],
        },
        "attention_mask_summary": tensor_summary(traces.get("attn_attention_mask")),
        "cache_position": None,
        "cache_position_note": "LlamaModel_PatternKV.forward does not expose/use a cache_position argument; decode position comes from segmented cache total_tokens.",
        "past_after": out.past_key_values,
    }


def build_cases(model: Any, inputs: torch.Tensor) -> dict[str, Any]:
    a_prefill = prefill_once(model, inputs[0:1, :384])
    b_prefill = prefill_once(model, inputs[1:2, :513])
    b1_trace = decode_with_trace(model, a_prefill["next_token"], a_prefill["past"], active_row=0, label="B1_A_step1")
    prefills = [prefill_once(model, inputs[0:1, :384]), prefill_once(model, inputs[1:2, :513])]
    ragged_past = tuple(serialize_cache(assemble_ragged_patternkv_cache([prefills[0]["past"][i], prefills[1]["past"][i]])) for i in range(len(prefills[0]["past"])))
    ragged_tokens = torch.stack([prefills[0]["next_token"], prefills[1]["next_token"]]).view(2)
    ragged_trace = decode_with_trace(model, ragged_tokens, ragged_past, active_row=0, label="RAGGED_AB_A_step1")
    return {
        "b1": b1_trace,
        "ragged": ragged_trace,
        "prefill_counters_a": a_prefill["kproj_counters"],
        "prefill_counters_b": b_prefill["kproj_counters"],
    }


def compare_boundaries(b1: dict[str, Any], ragged: dict[str, Any]) -> dict[str, Any]:
    return {
        "TOKEN_MATCH": compare_tensors(ragged["input_ids_row"], b1["input_ids_row"]),
        "POSITION_INPUT_MATCH": compare_tensors(ragged["attn_position_ids"], b1["attn_position_ids"]),
        "INFERRED_POSITION_MATCH": compare_tensors(ragged["inferred_position_ids_full"][0:1], b1["inferred_position_ids_full"][0:1]),
        "EMBEDDING_MATCH": compare_tensors(ragged["embedding"], b1["embedding"]),
        "LAYER0_HIDDEN_IN_MATCH": compare_tensors(ragged["layer0_hidden_in"], b1["layer0_hidden_in"]),
        "RMSNORM_INPUT_MATCH": compare_tensors(ragged["rmsnorm_input"], b1["rmsnorm_input"]),
        "RMSNORM_OUTPUT_MATCH": compare_tensors(ragged["rmsnorm_output"], b1["rmsnorm_output"]),
        "QPROJ_INPUT_MATCH": compare_tensors(ragged["qproj_input"], b1["qproj_input"]),
        "RAW_QPROJ_MATCH": compare_tensors(ragged["raw_qproj"], b1["raw_qproj"]),
        "RAW_KPROJ_MATCH": compare_tensors(ragged["raw_kproj"], b1["raw_kproj"]),
        "RAW_VPROJ_MATCH": compare_tensors(ragged["raw_vproj"], b1["raw_vproj"]),
        "ROPE_INPUT_MATCH": compare_tensors(ragged["rope_input_k"], b1["rope_input_k"]),
        "COS_MATCH": compare_tensors(ragged["cos"], b1["cos"]),
        "SIN_MATCH": compare_tensors(ragged["sin"], b1["sin"]),
        "ROPE_OUTPUT_CURRENT_K_MATCH": compare_tensors(ragged["rope_output_current_k"], b1["rope_output_current_k"]),
        "ATTENTION_MASK_MATCH": compare_tensors(ragged["attn_attention_mask"], b1["attn_attention_mask"]),
    }


def repeat_hashes(fn: Any, repeats: int = 20) -> dict[str, Any]:
    hashes = []
    outputs = []
    for _ in range(repeats):
        out = fn().detach().contiguous().cpu()
        hashes.append(sha(out))
        outputs.append(out)
    return {
        "repeats": repeats,
        "unique_hash_count": len(set(hashes)),
        "hashes": hashes,
        "first_output": outputs[0],
    }


def frozen_kproj_oracles(model: Any, b1: dict[str, Any], ragged: dict[str, Any]) -> dict[str, Any]:
    FORENSICS_DIR.mkdir(parents=True, exist_ok=True)
    attn = model.model.layers[0].self_attn
    h_a = b1["rmsnorm_output"].to(device=attn.k_proj.weight.device, dtype=attn.k_proj.weight.dtype).contiguous()
    h_b = ragged["rmsnorm_output"].to(device=attn.k_proj.weight.device, dtype=attn.k_proj.weight.dtype).contiguous()
    h_b2 = torch.flip(h_b, dims=[-1]).contiguous()
    torch.save({"H_A": h_a.detach().cpu(), "dtype": str(h_a.dtype)}, FORENSICS_DIR / "fixed_layer0_step1_norm_A.pt")
    h_ab = torch.cat([h_a, h_b], dim=0).contiguous()
    h_ab2 = torch.cat([h_a, h_b2], dim=0).contiguous()

    with torch.inference_mode():
        m1 = repeat_hashes(lambda: attn.k_proj(h_a))
        m2 = repeat_hashes(lambda: attn.k_proj(h_ab)[0:1])
        peer_a = attn.k_proj(h_ab)[0:1].detach().contiguous().cpu()
        peer_a2 = attn.k_proj(h_ab2)[0:1].detach().contiguous().cpu()
        dummy = torch.zeros_like(h_b)
        h_adummy = torch.cat([h_a, dummy], dim=0).contiguous()
        padded_a = attn.k_proj(h_adummy)[0:1].detach().contiguous().cpu()
        prod_b1 = b1["raw_kproj"]
        prod_ragged = ragged["raw_kproj"]
        fp32_m1 = torch.nn.functional.linear(h_a.float(), attn.k_proj.weight.float(), None).detach().contiguous().cpu()
        fp32_m2 = torch.nn.functional.linear(h_ab.float(), attn.k_proj.weight.float(), None)[0:1].detach().contiguous().cpu()

    low_m1_m2 = compare_tensors(m2["first_output"], m1["first_output"])
    fp32_cmp = compare_tensors(fp32_m2, fp32_m1)
    return {
        "frozen_kproj_batch_shape_oracle": {
            "input_H_A_path": str(FORENSICS_DIR / "fixed_layer0_step1_norm_A.pt"),
            "production_dtype": str(h_a.dtype),
            "weight_dtype": str(attn.k_proj.weight.dtype),
            "weight_shape": list(attn.k_proj.weight.shape),
            "bias": getattr(attn.k_proj, "bias", None) is not None,
            "M1_unique_hash_count": m1["unique_hash_count"],
            "M2_unique_hash_count": m2["unique_hash_count"],
            "M1_vs_M2_A_row": low_m1_m2,
            "M1_matches_B1_production_raw_k": compare_tensors(m1["first_output"], prod_b1),
            "M2_matches_RAGGED_production_raw_k": compare_tensors(m2["first_output"], prod_ragged),
        },
        "peer_content_fixed_shape_oracle": {
            "same_M_different_peer_A_row": compare_tensors(peer_a2, peer_a),
            "classification_if_false": "CROSS_ROW_KPROJ_LEAKAGE_OR_LAYOUT_BUG",
        },
        "fixed_shape_padding_oracle": {
            "ordinary_B1_M1_vs_padded_M2_A_row": compare_tensors(padded_a, m1["first_output"]),
            "padded_M2_A_row_vs_ragged_real_M2_A_row": compare_tensors(padded_a, peer_a),
            "padded_M2_A_row_vs_ragged_production_raw_k": compare_tensors(padded_a, prod_ragged),
        },
        "fp32_diagnostic_oracle": {
            "LOW_PRECISION_M1_M2_REL_L2": low_m1_m2["rel_l2"],
            "LOW_PRECISION_M1_M2_EXACT": low_m1_m2["exact_equal"],
            "FP32_M1_M2_REL_L2": fp32_cmp["rel_l2"],
            "FP32_M1_M2_EXACT": fp32_cmp["exact_equal"],
            "FP32_STABILITY_ORACLE_SUPPORTS_REDUCTION_NUMERICS": (
                low_m1_m2["rel_l2"] is not None
                and fp32_cmp["rel_l2"] is not None
                and float(fp32_cmp["rel_l2"]) < float(low_m1_m2["rel_l2"])
            ),
        },
    }


def existing_bi_oracle(model: Any, b1: dict[str, Any], ragged: dict[str, Any]) -> dict[str, Any]:
    attn = model.model.layers[0].self_attn
    h_a = b1["rmsnorm_output"].to(device=attn.k_proj.weight.device, dtype=attn.k_proj.weight.dtype).contiguous()
    h_b = ragged["rmsnorm_output"].to(device=attn.k_proj.weight.device, dtype=attn.k_proj.weight.dtype).contiguous()
    h_ab = torch.cat([h_a, h_b], dim=0).contiguous()
    result: dict[str, Any] = {
        "available": bool(batch_invariant_kproj_available()),
        "backend": bi_kproj_selected_backend(),
    }
    if not result["available"]:
        result["status"] = "EXTERNAL_BATCH_INVARIANT_OP_NOT_AVAILABLE"
        return result
    reset_batch_invariant_kproj_counters()
    with torch.inference_mode():
        bi_m1 = batch_invariant_k_projection(h_a, attn.k_proj.weight, getattr(attn.k_proj, "bias", None), backend=bi_kproj_selected_backend()).detach().contiguous().cpu()
        bi_m2 = batch_invariant_k_projection(h_ab, attn.k_proj.weight, getattr(attn.k_proj, "bias", None), backend=bi_kproj_selected_backend())[0:1].detach().contiguous().cpu()
    result.update(
        {
            "BI_KPROJ_M1_VS_M2_EXACT": compare_tensors(bi_m2, bi_m1),
            "counters_after_direct_oracle": batch_invariant_kproj_counters(),
        }
    )
    return result


def profile_kproj_kernel(model: Any, b1: dict[str, Any], ragged: dict[str, Any]) -> dict[str, Any]:
    attn = model.model.layers[0].self_attn
    h_a = b1["rmsnorm_output"].to(device=attn.k_proj.weight.device, dtype=attn.k_proj.weight.dtype).contiguous()
    h_b = ragged["rmsnorm_output"].to(device=attn.k_proj.weight.device, dtype=attn.k_proj.weight.dtype).contiguous()
    h_ab = torch.cat([h_a, h_b], dim=0).contiguous()
    try:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            with torch.inference_mode():
                _ = attn.k_proj(h_a)
                _ = attn.k_proj(h_ab)
                torch.cuda.synchronize(h_a.device)
        events = []
        for evt in prof.key_averages(group_by_input_shape=True):
            name = evt.key.lower()
            if "gemm" in name or "mm" in name or "linear" in name or "matmul" in name:
                events.append(
                    {
                        "key": evt.key,
                        "input_shapes": str(evt.input_shapes),
                        "cpu_time_total_us": float(evt.cpu_time_total),
                        "cuda_time_total_us": float(getattr(evt, "cuda_time_total", 0.0)),
                    }
                )
        return {"profile_succeeded": True, "events": events[:30]}
    except Exception as exc:
        return {"profile_succeeded": False, "error": repr(exc)}


def environment(preexisting: str) -> dict[str, Any]:
    try:
        import triton

        triton_version = triton.__version__
    except Exception:
        triton_version = "unavailable"
    return {
        "start_head": START_HEAD,
        "actual_head": git(["rev-parse", "HEAD"]),
        "branch": git(["branch", "--show-current"]),
        "preexisting_dirty_files": preexisting.splitlines(),
        "git_status_short": git(["status", "--short"]),
        "remote_v": git(["remote", "-v"]),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton_version,
        "nvidia_smi": nvidia_smi(),
    }


def serialize_inputs(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": trace["label"],
        "active_row": trace["active_row"],
        "full_batch_size": trace["full_batch_size"],
        "input_ids_full": trace["input_ids_full"].tolist(),
        "input_ids_row": trace["input_ids_row"].view(-1).tolist(),
        "tokens_at_embedding": trace["tokens_at_embedding"].view(-1).tolist() if torch.is_tensor(trace["tokens_at_embedding"]) else None,
        "inferred_position_ids_full": trace["inferred_position_ids_full"].tolist(),
        "attn_position_ids": trace["attn_position_ids"].tolist() if torch.is_tensor(trace["attn_position_ids"]) else None,
        "layer0_position_ids": trace["layer0_position_ids"].tolist() if torch.is_tensor(trace["layer0_position_ids"]) else None,
        "attention_mask_summary": trace["attention_mask_summary"],
        "cache_position": trace["cache_position"],
        "cache_position_note": trace["cache_position_note"],
    }


def boundary_table(comparison: dict[str, Any]) -> str:
    lines = ["| Boundary | Exact | Max abs | Rel L2 | First diff |", "|---|---:|---:|---:|---|"]
    for name, cmp in comparison.items():
        lines.append(
            f"| `{name}` | {cmp.get('exact_equal')} | {cmp.get('max_abs')} | {cmp.get('rel_l2')} | {cmp.get('first_diff_index')} |"
        )
    return "\n".join(lines)


def static_bi_audit(model: Any, b1: dict[str, Any], ragged: dict[str, Any]) -> dict[str, Any]:
    return {
        "BI_KPROJ_STATIC_AUDIT": {
            "code_location": "models/llama_patternkv.py:1100-1110 calls batch_invariant_k_projection only when patternkv_use_bi_prefill_kproj(past_key_value) is true; quant/batch_invariant_kproj.py:238 returns true only for past_key_value is None and prefill mode in {bi_k, bi_kv}.",
            "helper_condition": "patternkv_use_bi_prefill_kproj(past_key_value) == past_key_value is None and prefill_proj_mode in {bi_k, bi_kv}",
            "prefill_proj_mode": prefill_proj_mode(),
            "decode_supported_by_helper": False,
            "b1_step1_is_prefill": False,
            "ragged_step1_is_prefill": False,
            "b1_helper_result": bool(patternkv_use_bi_prefill_kproj(object())),
            "ragged_helper_result": bool(patternkv_use_bi_prefill_kproj(object())),
            "same_function_kernel_expected": "Both decode paths use self.k_proj(hidden_states) rather than BI KProj.",
            "b1_kproj_input_shape": b1["kproj_runtime"]["input_shape_full"],
            "ragged_kproj_input_shape": ragged["kproj_runtime"]["input_shape_full"],
        },
        "BI_KPROJ_RUNTIME_PATH_B1": b1["kproj_runtime"],
        "BI_KPROJ_RUNTIME_PATH_RAGGED": ragged["kproj_runtime"],
    }


def classify(comparison: dict[str, Any], frozen: dict[str, Any], bi: dict[str, Any]) -> tuple[str, str, str]:
    if not comparison["TOKEN_MATCH"]["exact_equal"] or not comparison["POSITION_INPUT_MATCH"]["exact_equal"]:
        return "STEP1_INPUT_TOKEN_OR_POSITION_DIVERGENCE", "TRACE_STEP1_INPUT_TOKEN_OR_POSITION", "token_or_position"
    if not comparison["EMBEDDING_MATCH"]["exact_equal"]:
        return "STEP1_EMBEDDING_INPUT_DIVERGENCE", "TRACE_STEP1_EMBEDDING_INPUT", "embedding"
    if not comparison["LAYER0_HIDDEN_IN_MATCH"]["exact_equal"]:
        return "STEP1_LAYER0_HIDDEN_INPUT_DIVERGENCE", "TRACE_LAYER0_HIDDEN_INPUT", "layer0_hidden_in"
    if not comparison["RMSNORM_OUTPUT_MATCH"]["exact_equal"]:
        return "BATCH_SHAPE_DEPENDENT_RMSNORM_NUMERICS_CONFIRMED", "IMPLEMENT_RAGGED_DECODE_BATCH_INVARIANT_RMSNORM", "rmsnorm_output"
    if not comparison["RAW_KPROJ_MATCH"]["exact_equal"]:
        oracle = frozen["frozen_kproj_batch_shape_oracle"]
        peer = frozen["peer_content_fixed_shape_oracle"]["same_M_different_peer_A_row"]
        bi_cmp = bi.get("BI_KPROJ_M1_VS_M2_EXACT")
        if peer["exact_equal"] is False:
            return "CROSS_ROW_KPROJ_LEAKAGE_OR_LAYOUT_BUG", "DIAGNOSE_KPROJ_LAYOUT_OR_ROW_ISOLATION", "raw_kproj_peer_content"
        if (
            oracle["M1_unique_hash_count"] == 1
            and oracle["M2_unique_hash_count"] == 1
            and not oracle["M1_vs_M2_A_row"]["exact_equal"]
            and isinstance(bi_cmp, dict)
            and bi_cmp["exact_equal"]
        ):
            return "BATCH_SHAPE_DEPENDENT_K_PROJECTION_NUMERICS_CONFIRMED", "EXTEND_OR_ENFORCE_BI_KPROJ_FOR_RAGGED_DECODE", "raw_kproj"
    if comparison["RAW_KPROJ_MATCH"]["exact_equal"] and not comparison["ROPE_OUTPUT_CURRENT_K_MATCH"]["exact_equal"]:
        if not comparison["POSITION_INPUT_MATCH"]["exact_equal"] or not comparison["COS_MATCH"]["exact_equal"] or not comparison["SIN_MATCH"]["exact_equal"]:
            return "ROPE_INPUT_METADATA_DIVERGENCE", "FIX_RAGGED_ROPE_POSITION_SEMANTICS", "rope_metadata"
        return "BATCH_SHAPE_DEPENDENT_ROPE_PATH_CONFIRMED", "FIX_RAGGED_ROPE_BATCH_INVARIANCE", "rope_output_current_k"
    if comparison["RAW_KPROJ_MATCH"]["exact_equal"] and comparison["ROPE_OUTPUT_CURRENT_K_MATCH"]["exact_equal"]:
        return "CURRENT_K_EXTRACTION_OR_INSTRUMENTATION_ARTIFACT", "RECHECK_CURRENT_K_CAPTURE_SITE", "instrumentation"
    return "STEP1_LAYER0_K_PATH_ROOT_CAUSE_UNRESOLVED", "DEEPEN_STEP1_LAYER0_K_PATH_FORENSIC", "unresolved"


def write_reports(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / "preflight.json", payload["environment"])
    write_md(
        REPORT_DIR / "environment.md",
        "Environment",
        json.dumps({k: v for k, v in payload["environment"].items() if k != "nvidia_smi"}, indent=2, sort_keys=True)
        + "\n\n```text\n"
        + payload["environment"]["nvidia_smi"].strip()
        + "\n```",
    )
    write_md(
        REPORT_DIR / "kpath_call_graph.md",
        "K Path Call Graph",
        "- `LlamaForCausalLM_PatternKV.forward` calls `LlamaModel_PatternKV.forward`.\n"
        "- `LlamaModel_PatternKV.forward` creates `position_ids` from segmented cache total tokens for decode and runs `embed_tokens(input_ids)`.\n"
        "- `LlamaDecoderLayer_PatternKV.forward` records layer input, then applies `input_layernorm`.\n"
        "- `LlamaFlashAttention_PatternKV.forward` receives RMSNorm output, runs Q/K/V projections, reshapes K to `[B, Hkv, T, D]`, applies RoPE, then appends current K to segmented cache.",
    )
    write_json(REPORT_DIR / "model_input_comparison.json", payload["model_input_comparison"])
    write_json(REPORT_DIR / "operator_boundary_comparison.json", payload["operator_boundary_comparison"])
    write_md(REPORT_DIR / "operator_boundary_comparison.md", "Operator Boundary Comparison", boundary_table(payload["operator_boundary_comparison"]))
    write_json(REPORT_DIR / "rmsnorm_batch_shape_oracle.json", payload["rmsnorm_batch_shape_oracle"])
    write_json(REPORT_DIR / "raw_projection_comparison.json", payload["raw_projection_comparison"])
    write_json(REPORT_DIR / "kproj_runtime_dispatch.json", payload["kproj_runtime_dispatch"])
    write_md(REPORT_DIR / "existing_bi_kproj_audit.md", "Existing BI KProj Audit", payload["existing_bi_kproj_audit_md"])
    write_json(REPORT_DIR / "frozen_kproj_batch_shape_oracle.json", payload["frozen"]["frozen_kproj_batch_shape_oracle"])
    write_json(REPORT_DIR / "peer_content_fixed_shape_oracle.json", payload["frozen"]["peer_content_fixed_shape_oracle"])
    write_json(REPORT_DIR / "fixed_shape_padding_oracle.json", payload["frozen"]["fixed_shape_padding_oracle"])
    write_json(REPORT_DIR / "existing_bi_kproj_oracle.json", payload["existing_bi_kproj_oracle"])
    write_json(REPORT_DIR / "fp32_diagnostic_oracle.json", payload["frozen"]["fp32_diagnostic_oracle"])
    write_md(REPORT_DIR / "kproj_kernel_audit.md", "KProj Kernel Audit", json.dumps(payload["kproj_kernel_audit"], indent=2, sort_keys=True))
    write_md(REPORT_DIR / "root_cause_evidence.md", "Root Cause Evidence", payload["root_cause_evidence"])
    write_json(REPORT_DIR / "final_gate.json", payload["final_gate"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    set_env()
    preexisting = git(["status", "--short"])
    env = environment(preexisting)
    tokenizer, _config, model = load_model(dtype=torch.float16, device=torch.device(args.device))
    inputs = make_fixed_inputs(tokenizer, batch=4, context=513, device=torch.device(args.device))
    cases = build_cases(model, inputs)
    b1 = cases["b1"]
    ragged = cases["ragged"]
    comparison = compare_boundaries(b1, ragged)
    frozen = frozen_kproj_oracles(model, b1, ragged)
    bi = existing_bi_oracle(model, b1, ragged)
    kernel = profile_kproj_kernel(model, b1, ragged)
    audit = static_bi_audit(model, b1, ragged)
    classification, next_task, first_field = classify(comparison, frozen, bi)
    root_cause_evidence = (
        f"TOKEN_MATCH={comparison['TOKEN_MATCH']['exact_equal']}; "
        f"POSITION_INPUT_MATCH={comparison['POSITION_INPUT_MATCH']['exact_equal']}; "
        f"EMBEDDING_MATCH={comparison['EMBEDDING_MATCH']['exact_equal']}; "
        f"LAYER0_HIDDEN_IN_MATCH={comparison['LAYER0_HIDDEN_IN_MATCH']['exact_equal']}; "
        f"RMSNORM_OUTPUT_MATCH={comparison['RMSNORM_OUTPUT_MATCH']['exact_equal']}; "
        f"RAW_KPROJ_MATCH={comparison['RAW_KPROJ_MATCH']['exact_equal']} "
        f"(max_abs={comparison['RAW_KPROJ_MATCH']['max_abs']}, rel_l2={comparison['RAW_KPROJ_MATCH']['rel_l2']}); "
        f"ROPE_OUTPUT_CURRENT_K_MATCH={comparison['ROPE_OUTPUT_CURRENT_K_MATCH']['exact_equal']}; "
        f"production frozen M1 unique={frozen['frozen_kproj_batch_shape_oracle']['M1_unique_hash_count']}, "
        f"M2 unique={frozen['frozen_kproj_batch_shape_oracle']['M2_unique_hash_count']}, "
        f"M1_vs_M2={frozen['frozen_kproj_batch_shape_oracle']['M1_vs_M2_A_row']['exact_equal']}; "
        f"existing BI M1_vs_M2={bi.get('BI_KPROJ_M1_VS_M2_EXACT', {}).get('exact_equal') if isinstance(bi.get('BI_KPROJ_M1_VS_M2_EXACT'), dict) else None}; "
        f"classification={classification}."
    )
    kproj_runtime_dispatch = {
        **audit,
        "prefill_counters_a": cases["prefill_counters_a"],
        "prefill_counters_b": cases["prefill_counters_b"],
    }
    model_input_comparison = {
        "B1": serialize_inputs(b1),
        "RAGGED": serialize_inputs(ragged),
        "comparisons": {
            "token": comparison["TOKEN_MATCH"],
            "position_ids": comparison["POSITION_INPUT_MATCH"],
            "inferred_position_ids": comparison["INFERRED_POSITION_MATCH"],
            "attention_mask": comparison["ATTENTION_MASK_MATCH"],
            "cache_position": {"match": True, "b1": None, "ragged": None, "note": b1["cache_position_note"]},
        },
    }
    rmsnorm_batch_shape_oracle = {
        "executed": True,
        "rmsnorm_input_b1_vs_ragged": comparison["RMSNORM_INPUT_MATCH"],
        "rmsnorm_output_b1_vs_ragged": comparison["RMSNORM_OUTPUT_MATCH"],
        "classification_if_output_diff": "BATCH_SHAPE_DEPENDENT_RMSNORM_NUMERICS_CONFIRMED",
    }
    raw_projection_comparison = {
        "q_proj": comparison["RAW_QPROJ_MATCH"],
        "k_proj": comparison["RAW_KPROJ_MATCH"],
        "v_proj": comparison["RAW_VPROJ_MATCH"],
        "control_interpretation": "Q/K/V all use normal nn.Linear decode path; K is root-relevant because recent_k consumes post-RoPE K.",
    }
    existing_bi_md = (
        "Static dispatch shows existing BI KProj is guarded by `patternkv_use_bi_prefill_kproj(past_key_value)`, "
        "which requires `past_key_value is None`; step1 decode has non-None segmented cache for both independent B1 and ragged B2, "
        "so both runtime paths bypass BI KProj and call `self.k_proj(hidden_states)`. "
        f"B1 normal decode KProj calls: {b1['kproj_runtime']['normal_decode_kproj_calls']}; "
        f"ragged normal decode KProj calls: {ragged['kproj_runtime']['normal_decode_kproj_calls']}."
    )
    final_gate = {
        "start_head": START_HEAD,
        "actual_head": env["actual_head"],
        "branch": env["branch"],
        "forensic_only": True,
        "production_code_modified": False,
        "request": "A",
        "step": 1,
        "layer": 0,
        "TOKEN_MATCH": comparison["TOKEN_MATCH"]["exact_equal"],
        "POSITION_INPUT_MATCH": comparison["POSITION_INPUT_MATCH"]["exact_equal"],
        "EMBEDDING_MATCH": comparison["EMBEDDING_MATCH"]["exact_equal"],
        "LAYER0_HIDDEN_IN_MATCH": comparison["LAYER0_HIDDEN_IN_MATCH"]["exact_equal"],
        "RMSNORM_OUTPUT_MATCH": comparison["RMSNORM_OUTPUT_MATCH"]["exact_equal"],
        "RAW_KPROJ_MATCH": comparison["RAW_KPROJ_MATCH"]["exact_equal"],
        "ROPE_INPUT_MATCH": comparison["ROPE_INPUT_MATCH"]["exact_equal"],
        "ROPE_OUTPUT_CURRENT_K_MATCH": comparison["ROPE_OUTPUT_CURRENT_K_MATCH"]["exact_equal"],
        "RAW_QPROJ_MATCH": comparison["RAW_QPROJ_MATCH"]["exact_equal"],
        "RAW_VPROJ_MATCH": comparison["RAW_VPROJ_MATCH"]["exact_equal"],
        "BI_KPROJ_STATIC_AUDIT": audit["BI_KPROJ_STATIC_AUDIT"],
        "BI_KPROJ_RUNTIME_PATH_B1_USED_BI": b1["kproj_runtime"]["used_bi_kproj"],
        "BI_KPROJ_RUNTIME_PATH_RAGGED_USED_BI": ragged["kproj_runtime"]["used_bi_kproj"],
        "NORMAL_KPROJ_M1_UNIQUE_HASH_COUNT": frozen["frozen_kproj_batch_shape_oracle"]["M1_unique_hash_count"],
        "NORMAL_KPROJ_M2_UNIQUE_HASH_COUNT": frozen["frozen_kproj_batch_shape_oracle"]["M2_unique_hash_count"],
        "NORMAL_KPROJ_M1_VS_M2_EXACT": frozen["frozen_kproj_batch_shape_oracle"]["M1_vs_M2_A_row"]["exact_equal"],
        "BI_KPROJ_M1_VS_M2_EXACT": bi.get("BI_KPROJ_M1_VS_M2_EXACT", {}).get("exact_equal")
        if isinstance(bi.get("BI_KPROJ_M1_VS_M2_EXACT"), dict)
        else None,
        "PEER_CONTENT_A_ROW_INVARIANT": frozen["peer_content_fixed_shape_oracle"]["same_M_different_peer_A_row"]["exact_equal"],
        "FP32_STABILITY_ORACLE_SUPPORTS_REDUCTION_NUMERICS": frozen["fp32_diagnostic_oracle"]["FP32_STABILITY_ORACLE_SUPPORTS_REDUCTION_NUMERICS"],
        "first_divergent_kpath_field": first_field,
        "root_classification": classification,
        "next_task": next_task,
        "compileall_pass": False,
        "targeted_tests": "",
        "full_pytest": "",
        "git_diff_check_pass": False,
    }
    payload = {
        "environment": env,
        "model_input_comparison": model_input_comparison,
        "operator_boundary_comparison": comparison,
        "rmsnorm_batch_shape_oracle": rmsnorm_batch_shape_oracle,
        "raw_projection_comparison": raw_projection_comparison,
        "kproj_runtime_dispatch": kproj_runtime_dispatch,
        "existing_bi_kproj_audit_md": existing_bi_md,
        "frozen": frozen,
        "existing_bi_kproj_oracle": bi,
        "kproj_kernel_audit": kernel,
        "root_cause_evidence": root_cause_evidence,
        "final_gate": final_gate,
    }
    write_reports(payload)
    print(json.dumps(final_gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
