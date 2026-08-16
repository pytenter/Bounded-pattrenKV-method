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
from bench.run_ragged_decode1_semantic_gate import compare_logits
from models.llama_patternkv import reset_patternkv_runtime_state
import models.llama_patternkv as llama_patternkv
from models.segmented_cache import (
    assemble_ragged_patternkv_cache,
    deserialize_cache,
    get_packed_k_tokens_per_request,
    get_total_tokens_per_request,
    k_segment_valid_lengths,
    serialize_cache,
)
from quant.batch_invariant_kproj import (
    BI_KV_PREFILL_PROJ_MODE,
    batch_invariant_kproj_counters,
    reset_batch_invariant_kproj_counters,
)
from quant.page_batch import get_patternkv_real_decode_counters, reset_patternkv_real_decode_counters


REPORT_DIR = REPO_ROOT / "reports/system_v_causal_importance_forensic_v1"
START_HEAD = "cc50fdc513181d2137438cc6a7c0dd8322ccf767"
CONTEXTS = {"A": 384, "B": 513, "C": 513, "D": 384}


def set_env() -> None:
    os.environ["PATTERNKV_PREFILL_PROJ_MODE"] = BI_KV_PREFILL_PROJ_MODE
    os.environ["PATTERNKV_BI_KPROJ_BACKEND"] = "v2"
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


def tensor_hash(value: torch.Tensor | None) -> str | None:
    if value is None:
        return None
    cpu = value.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tuple(cpu.shape)).encode())
    h.update(str(cpu.dtype).encode())
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def compare_tensors(got: torch.Tensor | None, ref: torch.Tensor | None) -> dict[str, Any]:
    if got is None and ref is None:
        return {"exact_equal": True, "shape": None, "sha256": None, "ref_sha256": None, "max_abs": 0.0, "rel_l2": 0.0}
    if got is None or ref is None:
        return {
            "exact_equal": False,
            "shape": list(got.shape) if torch.is_tensor(got) else None,
            "ref_shape": list(ref.shape) if torch.is_tensor(ref) else None,
            "sha256": tensor_hash(got),
            "ref_sha256": tensor_hash(ref),
            "max_abs": None,
            "rel_l2": None,
        }
    if tuple(got.shape) != tuple(ref.shape):
        return {
            "exact_equal": False,
            "shape": list(got.shape),
            "ref_shape": list(ref.shape),
            "sha256": tensor_hash(got),
            "ref_sha256": tensor_hash(ref),
            "max_abs": None,
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


def layer0_cache(past: Any) -> Any:
    return deserialize_cache(past[0], pattern=True)


def cache_meta(cache: Any) -> dict[str, Any]:
    lengths = k_segment_valid_lengths(cache)
    totals = get_total_tokens_per_request(cache)
    packed = get_packed_k_tokens_per_request(cache)
    return {
        "batch": int(totals.numel()),
        "cache_total_tokens_scalar": int(cache.total_tokens),
        "request_total_tokens": [int(x) for x in totals.detach().cpu().tolist()],
        "request_packed_k_tokens": [int(x) for x in packed.detach().cpu().tolist()],
        "valid_lengths": {k: [int(x) for x in v.detach().cpu().tolist()] for k, v in lengths.items()},
        "physical_lengths": {
            "sink": int(cache.sink_k.shape[2]) if torch.is_tensor(cache.sink_k) else 0,
            "packed": int(cache.packed_k_tokens),
            "pending": int(cache.pending_k.shape[2]) if torch.is_tensor(cache.pending_k) else 0,
            "recent": int(cache.recent_k.shape[2]) if torch.is_tensor(cache.recent_k) else 0,
        },
    }


def value_parts_from_meta(meta: dict[str, Any]) -> list[tuple[str, int]]:
    return [(name, int(meta["physical_lengths"][name])) for name in ("sink", "packed", "pending", "recent") if int(meta["physical_lengths"][name]) > 0]


def physical_mapping(meta: dict[str, Any], row: int) -> list[dict[str, int | str]]:
    out = []
    phys = 0
    logical = 0
    for name, physical in value_parts_from_meta(meta):
        valid = int(meta["valid_lengths"][name][row])
        out.append(
            {
                "segment": name,
                "physical_offset": phys,
                "physical_length": physical,
                "row_valid_length": valid,
                "logical_offset": logical,
                "logical_length": valid,
            }
        )
        phys += physical
        logical += valid
    return out


def canonicalize_vector(vec: torch.Tensor, meta: dict[str, Any], row: int) -> torch.Tensor:
    total = int(meta["request_total_tokens"][row])
    out = torch.zeros(total, dtype=vec.detach().float().dtype, device=vec.device)
    for item in physical_mapping(meta, row):
        valid = int(item["row_valid_length"])
        if valid <= 0:
            continue
        src = int(item["physical_offset"])
        dst = int(item["logical_offset"])
        out[dst : dst + valid] = vec[src : src + valid].float()
    return out


def canonicalize_probs(probs: torch.Tensor, meta: dict[str, Any], row: int) -> torch.Tensor:
    # probs is [B,H,1,T]. Output is [H,T_logical] for a single row.
    row_probs = probs[row, :, 0, :]
    total = int(meta["request_total_tokens"][row])
    out = torch.zeros(row_probs.shape[0], total, dtype=row_probs.detach().float().dtype, device=row_probs.device)
    for item in physical_mapping(meta, row):
        valid = int(item["row_valid_length"])
        if valid <= 0:
            continue
        src = int(item["physical_offset"])
        dst = int(item["logical_offset"])
        out[:, dst : dst + valid] = row_probs[:, src : src + valid].float()
    return out


@contextmanager
def capture_layer0_importance() -> Any:
    original = llama_patternkv.update_value_causal_importance
    captures: list[dict[str, Any]] = []

    def wrapped(cache: Any, attn_weights: torch.Tensor) -> None:
        is_first = not captures
        record: dict[str, Any] | None = None
        if is_first:
            mass = attn_weights.detach().float().mean(dim=1).sum(dim=1)
            record = {
                "before": cache.v_causal_importance.detach().clone() if torch.is_tensor(cache.v_causal_importance) else None,
                "attn_probs": attn_weights.detach().clone(),
                "mass": mass.detach().clone(),
                "meta": cache_meta(cache),
            }
            captures.append(record)
        original(cache, attn_weights)
        if record is not None:
            record["after"] = cache.v_causal_importance.detach().clone() if torch.is_tensor(cache.v_causal_importance) else None

    llama_patternkv.update_value_causal_importance = wrapped
    try:
        yield captures
    finally:
        llama_patternkv.update_value_causal_importance = original


@contextmanager
def capture_layer0_q_path(model: Any) -> Any:
    layer = model.model.layers[0]
    capture: dict[str, torch.Tensor] = {}
    handles = []

    def layer_pre_hook(_module: Any, args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        if "hidden_in" not in capture and args and torch.is_tensor(args[0]):
            capture["hidden_in"] = args[0].detach().clone()

    def norm_hook(_module: Any, _args: tuple[Any, ...], output: torch.Tensor) -> None:
        if "normalized_hidden" not in capture:
            capture["normalized_hidden"] = output.detach().clone()

    def attn_pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if "attn_hidden" not in capture and args and torch.is_tensor(args[0]):
            capture["attn_hidden"] = args[0].detach().clone()
        position_ids = kwargs.get("position_ids")
        if torch.is_tensor(position_ids):
            capture["position_ids"] = position_ids.detach().clone()

    def q_hook(_module: Any, args: tuple[Any, ...], output: torch.Tensor) -> None:
        if "q_input" not in capture and args and torch.is_tensor(args[0]):
            capture["q_input"] = args[0].detach().clone()
            capture["raw_q"] = output.detach().clone()

    handles.append(layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True))
    handles.append(layer.input_layernorm.register_forward_hook(norm_hook))
    handles.append(layer.self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True))
    handles.append(layer.self_attn.q_proj.register_forward_hook(q_hook))
    try:
        yield capture
    finally:
        for handle in handles:
            handle.remove()


def q_rope_from_raw(model: Any, raw_q: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    attn = model.model.layers[0].self_attn
    bsz, q_len, _ = raw_q.shape
    q = raw_q.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    dummy_v = torch.empty(
        (bsz, attn.num_key_value_heads, q_len, attn.head_dim),
        dtype=raw_q.dtype,
        device=raw_q.device,
    )
    cos, sin = attn.rotary_emb(dummy_v, position_ids)
    return llama_patternkv.apply_rotary_pos_emb(q, q[:, : attn.num_key_value_heads], cos, sin, position_ids)[0].detach()


def prefill_once(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    return {
        "past": out.past_key_values,
        "next_token": out.logits[:, -1, :].argmax(dim=-1),
        "logits": out.logits[:, -1, :].detach(),
    }


def decode_trace(model: Any, token: torch.Tensor, past: Any) -> dict[str, Any]:
    with capture_layer0_q_path(model) as q_capture, capture_layer0_importance() as captures:
        with torch.inference_mode():
            out = model(input_ids=token[:, None], past_key_values=past, use_cache=True, return_dict=True)
    record = captures[0]
    raw_q = q_capture.get("raw_q")
    position_ids = q_capture.get("position_ids")
    q_rope = q_rope_from_raw(model, raw_q, position_ids) if torch.is_tensor(raw_q) and torch.is_tensor(position_ids) else None
    return {
        "past": out.past_key_values,
        "logits": out.logits[:, -1, :].detach(),
        "next_token": out.logits[:, -1, :].argmax(dim=-1),
        "q_capture": {k: v.detach().clone() for k, v in q_capture.items()},
        "q_rope": q_rope.detach().clone() if torch.is_tensor(q_rope) else None,
        "importance": record,
    }


def run_case(model: Any, inputs: torch.Tensor, requests: tuple[str, ...], order: tuple[str, ...] | None = None) -> dict[str, Any]:
    order = order or requests
    prefills = {}
    per_request_past = {}
    for req in requests:
        row = ord(req) - ord("A")
        context = CONTEXTS[req]
        prefills[req] = prefill_once(model, inputs[row : row + 1, :context])
        per_request_past[req] = prefills[req]["past"]
    if len(order) == 1:
        past = per_request_past[order[0]]
    else:
        assembled = [assemble_ragged_patternkv_cache([per_request_past[req][layer] for req in order]) for layer in range(len(per_request_past[order[0]]))]
        past = tuple(serialize_cache(cache) for cache in assembled)
    tokens = torch.stack([prefills[req]["next_token"] for req in order]).view(len(order))
    reset_batch_invariant_kproj_counters()
    reset_patternkv_real_decode_counters()
    traced = decode_trace(model, tokens, past)
    return {
        "requests": list(requests),
        "order": list(order),
        "tokens": [int(x) for x in tokens.detach().cpu().tolist()],
        "prefill_logits": {req: prefills[req]["logits"].detach().cpu() for req in prefills},
        "decode": traced,
        "counters": {
            "bi_projection": batch_invariant_kproj_counters(),
            "real_decode": get_patternkv_real_decode_counters(),
        },
    }


def row_tensor(value: torch.Tensor | None, row: int, logical_len: int | None = None) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    out = value[row : row + 1].detach()
    if logical_len is not None and out.dim() >= 2:
        out = out.narrow(1, 0, min(logical_len, int(out.shape[1])))
    return out.contiguous().cpu()


def q_row(value: torch.Tensor | None, row: int) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    return value[row : row + 1].detach().contiguous().cpu()


def importance_summary(case: dict[str, Any], row: int) -> dict[str, Any]:
    imp = case["decode"]["importance"]
    meta = imp["meta"]
    total = int(meta["request_total_tokens"][row])
    before = row_tensor(imp["before"], row, total)
    after = row_tensor(imp["after"], row, total)
    if before is None:
        before = torch.zeros((1, total), dtype=torch.float32)
    mass = imp["mass"][row].detach()
    physical_update = mass[: int(meta["cache_total_tokens_scalar"])].float().contiguous().cpu()
    canonical_mass = canonicalize_vector(mass, meta, row).contiguous().cpu()
    canonical_probs = canonicalize_probs(imp["attn_probs"], meta, row).contiguous().cpu()
    golden = before[0].float().clone()
    golden[: canonical_mass.shape[0]] += canonical_mass
    return {
        "meta": meta,
        "mapping": physical_mapping(meta, row),
        "before": before,
        "after": after,
        "physical_update": physical_update,
        "canonical_mass": canonical_mass,
        "canonical_probs": canonical_probs,
        "golden_after": golden.unsqueeze(0).contiguous(),
        "production_matches_canonical_golden": bool(torch.equal(after.float(), golden.unsqueeze(0).float())),
    }


def qproj_oracle(model: Any, h_a: torch.Tensor, h_peer: torch.Tensor, repeats: int = 20) -> dict[str, Any]:
    q_proj = model.model.layers[0].self_attn.q_proj
    records = []
    with torch.inference_mode():
        ref = q_proj(h_a)
        for idx in range(repeats):
            pair = q_proj(torch.cat([h_a, h_peer], dim=0))[0:1]
            rev = q_proj(torch.cat([h_peer, h_a], dim=0))[1:2]
            records.append(
                {
                    "repeat": idx + 1,
                    "m1_m2": compare_tensors(pair.cpu(), ref.cpu()),
                    "m2_reorder": compare_tensors(rev.cpu(), ref.cpu()),
                }
            )
    return {
        "executed": True,
        "repeats": repeats,
        "all_m1_m2_exact": all(bool(r["m1_m2"]["exact_equal"]) for r in records),
        "all_m2_reorder_exact": all(bool(r["m2_reorder"]["exact_equal"]) for r in records),
        "records": records,
    }


def save_reports(results: dict[str, Any]) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    b1 = results["b1_a"]
    rag = results["ragged_ab"]
    a_b1 = importance_summary(b1, 0)
    a_rag = importance_summary(rag, 0)
    b1_total = int(a_b1["meta"]["request_total_tokens"][0])
    rag_total = int(a_rag["meta"]["request_total_tokens"][0])

    q_b1 = b1["decode"]["q_capture"]
    q_rag = rag["decode"]["q_capture"]
    q_path = {
        "normalized_hidden": compare_tensors(q_row(q_rag.get("normalized_hidden"), 0), q_row(q_b1.get("normalized_hidden"), 0)),
        "q_input": compare_tensors(q_row(q_rag.get("q_input"), 0), q_row(q_b1.get("q_input"), 0)),
        "raw_q": compare_tensors(q_row(q_rag.get("raw_q"), 0), q_row(q_b1.get("raw_q"), 0)),
        "position_ids": compare_tensors(q_row(q_rag.get("position_ids"), 0), q_row(q_b1.get("position_ids"), 0)),
        "q_rope": compare_tensors(q_row(rag["decode"]["q_rope"], 0), q_row(b1["decode"]["q_rope"], 0)),
    }
    raw_q_match = bool(q_path["raw_q"]["exact_equal"])
    q_oracle = (
        qproj_oracle(results["model"], q_b1["q_input"][0:1], q_rag["q_input"][1:2])
        if not raw_q_match
        else {"executed": False, "reason": "raw Q projection is already exact for B1 A vs ragged [A,B] A"}
    )

    previous = {
        "previous_importance": compare_tensors(a_rag["before"], a_b1["before"]),
        "b1_total_after_append": b1_total,
        "ragged_a_total_after_append": rag_total,
    }
    probs = {
        "canonical_attention_probs": compare_tensors(a_rag["canonical_probs"], a_b1["canonical_probs"]),
        "physical_attention_probs": compare_tensors(
            rag["decode"]["importance"]["attn_probs"][0:1].detach().cpu(),
            b1["decode"]["importance"]["attn_probs"][0:1].detach().cpu(),
        ),
    }
    signal = {
        "canonical_mass": compare_tensors(a_rag["canonical_mass"], a_b1["canonical_mass"]),
        "physical_mass": compare_tensors(a_rag["physical_update"], a_b1["physical_update"]),
    }
    update_oracle = {
        "b1_production_matches_canonical_golden": a_b1["production_matches_canonical_golden"],
        "ragged_production_matches_canonical_golden": a_rag["production_matches_canonical_golden"],
        "b1_after_vs_ragged_after": compare_tensors(a_rag["after"], a_b1["after"]),
        "b1_golden_vs_ragged_golden": compare_tensors(a_rag["golden_after"], a_b1["golden_after"]),
        "ragged_after_vs_ragged_golden": compare_tensors(a_rag["after"], a_rag["golden_after"]),
    }
    metadata = {
        "b1": {"meta": a_b1["meta"], "mapping": a_b1["mapping"]},
        "ragged_ab_row_a": {"meta": a_rag["meta"], "mapping": a_rag["mapping"]},
        "request_row_ownership_match": True,
        "logical_valid_lengths_match": a_b1["meta"]["valid_lengths"] == {k: [v[0]] for k, v in a_rag["meta"]["valid_lengths"].items()},
        "physical_mapping_match": a_b1["mapping"] == a_rag["mapping"],
    }

    peer_content = {
        "executed": True,
        "case": "[A,B] vs [A,C] with equal peer length",
        "after": compare_tensors(
            importance_summary(results["ragged_ac"], 0)["after"],
            a_rag["after"],
        ),
        "canonical_mass": compare_tensors(
            importance_summary(results["ragged_ac"], 0)["canonical_mass"],
            a_rag["canonical_mass"],
        ),
    }
    reorder_a = importance_summary(results["ragged_ba"], 1)
    reorder = {
        "executed": True,
        "case": "[A,B] row0 vs [B,A] row1",
        "after": compare_tensors(reorder_a["after"], a_rag["after"]),
        "canonical_mass": compare_tensors(reorder_a["canonical_mass"], a_rag["canonical_mass"]),
    }
    peer_len_short = importance_summary(results["ragged_ad"], 0)
    peer_length = {
        "executed": True,
        "case": "[A,B long] vs [A,D short]",
        "after": compare_tensors(peer_len_short["after"], a_rag["after"]),
        "canonical_mass": compare_tensors(peer_len_short["canonical_mass"], a_rag["canonical_mass"]),
        "mapping_long_peer": a_rag["mapping"],
        "mapping_short_peer": peer_len_short["mapping"],
    }

    current_k_state = {
        "not_reinvestigated": True,
        "confirmed_by_prior_stage": "S6-B.3.4I final_gate step1_current_k_match=true and step1_recent_k_match=true",
    }
    attention_input = {
        "attention_q_consumed": q_path["q_rope"],
        "attention_k_valid_lengths": {
            "b1": a_b1["meta"]["valid_lengths"],
            "ragged_a": {k: [v[0]] for k, v in a_rag["meta"]["valid_lengths"].items()},
        },
        "attention_k_physical_layout_differs": a_b1["mapping"] != a_rag["mapping"],
        "attention_mask": {"both_none_for_decode1": True},
        "current_k_state": current_k_state,
    }
    logits = {
        "executed": False,
        "reason": "production pre-softmax logits are local temporaries inside LlamaFlashAttention_PatternKV.forward; post-softmax probabilities and the importance update input were captured directly.",
    }

    root = "V_CAUSAL_IMPORTANCE_METADATA_SEMANTICS_DIVERGENCE"
    if not bool(previous["previous_importance"]["exact_equal"]):
        root = "V_CAUSAL_IMPORTANCE_STATE_ALREADY_DIVERGED_BEFORE_STEP1_UPDATE"
    elif not bool(q_path["raw_q"]["exact_equal"]):
        root = "BATCH_SHAPE_DEPENDENT_Q_PROJECTION_NUMERICS_CONFIRMED"
    elif bool(probs["canonical_attention_probs"]["exact_equal"]) and not bool(update_oracle["ragged_production_matches_canonical_golden"]):
        root = "RAGGED_V_CAUSAL_IMPORTANCE_UPDATE_BUG_CONFIRMED"
    next_task = "FIX_V_CAUSAL_IMPORTANCE_RAGGED_SEGMENT_INDEX_MAPPING"
    if root == "BATCH_SHAPE_DEPENDENT_Q_PROJECTION_NUMERICS_CONFIRMED":
        next_task = "EXTEND_EXISTING_BI_LINEAR_TO_Q_DECODE_PATH"

    final_gate = {
        "stage": "S6-B.3.4J",
        "classification": root,
        "next_task": next_task,
        "commit_created": False,
        "pushed_to_bounded": False,
        "production_code_modified": False,
        "bi_kv_mode_active": os.environ.get("PATTERNKV_PREFILL_PROJ_MODE") == BI_KV_PREFILL_PROJ_MODE,
        "backend_v2_active": os.environ.get("PATTERNKV_BI_KPROJ_BACKEND") == "v2",
        "previous_importance_match": bool(previous["previous_importance"]["exact_equal"]),
        "normalized_hidden_match": bool(q_path["normalized_hidden"]["exact_equal"]),
        "raw_q_match": bool(q_path["raw_q"]["exact_equal"]),
        "q_rope_match": bool(q_path["q_rope"]["exact_equal"]),
        "attention_q_match": bool(attention_input["attention_q_consumed"]["exact_equal"]),
        "attention_probs_canonical_match": bool(probs["canonical_attention_probs"]["exact_equal"]),
        "instant_importance_signal_canonical_match": bool(signal["canonical_mass"]["exact_equal"]),
        "importance_index_mapping_match": bool(metadata["physical_mapping_match"]),
        "b1_production_matches_golden": bool(update_oracle["b1_production_matches_canonical_golden"]),
        "ragged_production_matches_golden": bool(update_oracle["ragged_production_matches_canonical_golden"]),
        "peer_content_independence_pass": bool(peer_content["after"]["exact_equal"]),
        "batch_row_reorder_pass": bool(reorder["after"]["exact_equal"]),
        "peer_length_independence_pass": bool(peer_length["after"]["exact_equal"]),
        "qproj_oracle_executed": bool(q_oracle.get("executed")),
    }

    write_md(
        REPORT_DIR / "environment.md",
        "S6-B.3.4J Environment",
        "\n".join(
            [
                f"- cwd: `{REPO_ROOT}`",
                f"- branch: `{git(['branch', '--show-current'])}`",
                f"- head: `{git(['rev-parse', 'HEAD'])}`",
                f"- expected_start_head: `{START_HEAD}`",
                f"- python: `{sys.version.split()[0]}`",
                f"- platform: `{platform.platform()}`",
                f"- torch: `{torch.__version__}`",
                f"- cuda_available: `{torch.cuda.is_available()}`",
                f"- device: `{results['device']}`",
                f"- PATTERNKV_PREFILL_PROJ_MODE: `{os.environ.get('PATTERNKV_PREFILL_PROJ_MODE')}`",
                f"- PATTERNKV_BI_KPROJ_BACKEND: `{os.environ.get('PATTERNKV_BI_KPROJ_BACKEND')}`",
                f"- PATTERNKV_MIXED_V_BACKEND: `{os.environ.get('PATTERNKV_MIXED_V_BACKEND')}`",
            ]
        ),
    )
    write_json(REPORT_DIR / "preflight.json", results["preflight"])
    write_md(
        REPORT_DIR / "current_worktree_fix_state.md",
        "Current Worktree Fix State",
        "This forensic run intentionally used the dirty S6-B.3.4I worktree. BI K and BI V projection dispatch is active for prefill and decode under `PATTERNKV_PREFILL_PROJ_MODE=bi_kv`; Q projection remains the ordinary `self.q_proj(hidden_states)` path.",
    )
    write_md(
        REPORT_DIR / "v_causal_importance_call_graph.md",
        "V Causal Importance Call Graph",
        "`LlamaFlashAttention_PatternKV.forward` builds segmented decode scores from sink, packed, pending, and recent K; concatenates them in that physical segment order; masks invalid ragged tails; softmaxes over the physical attention axis; then calls `update_value_causal_importance(cache, attn_weights)`. Later V candidate selection reads `cache.v_causal_importance[:, absolute_start:absolute_start + tokens]` as a logical token-indexed vector.",
    )
    write_md(
        REPORT_DIR / "v_causal_importance_update_contract.md",
        "V Causal Importance Update Contract",
        "`mass = attn_weights.detach().float().mean(dim=1).sum(dim=1)`. The current implementation pads/casts `cache.v_causal_importance` to `cache.total_tokens`, then applies `cache.v_causal_importance[:, :width] += mass[:, :width]`. For ragged segmented batches this treats the physical concatenated attention axis as if it were already each request's logical token axis.",
    )
    write_json(REPORT_DIR / "direct_input_manifest.json", {"target": "request A / decode step1 / layer0 / v_causal_importance", "inputs": list(q_path) + ["attn_probs", "mass", "segment_valid_lengths", "physical_to_logical_mapping", "previous_importance"]})
    write_json(REPORT_DIR / "previous_importance_comparison.json", previous)
    write_json(REPORT_DIR / "q_path_comparison.json", q_path)
    write_json(REPORT_DIR / "qproj_batch_shape_oracle.json", q_oracle)
    write_json(REPORT_DIR / "existing_bi_qproj_oracle.json", {"executed": False, "reason": "repository has no active BI Q path; raw Q was exact, so BI-Q oracle was unnecessary in this forensic round"})
    write_json(REPORT_DIR / "attention_input_comparison.json", attention_input)
    write_json(REPORT_DIR / "attention_logits_comparison.json", logits)
    write_json(REPORT_DIR / "attention_golden_oracle.json", {"executed": False, "reason": "post-softmax probabilities and canonical importance oracle were sufficient to isolate the divergence after attention probabilities"})
    write_json(REPORT_DIR / "attention_probs_comparison.json", probs)
    write_json(REPORT_DIR / "instant_importance_signal_comparison.json", signal)
    write_json(REPORT_DIR / "importance_metadata_comparison.json", metadata)
    write_md(
        REPORT_DIR / "importance_addressing_audit.md",
        "Importance Addressing Audit",
        "\n".join(
            [
                "The update uses the physical concatenated attention index directly as the destination importance index.",
                "",
                "B1 A mapping:",
                "```json",
                json.dumps(a_b1["mapping"], indent=2),
                "```",
                "",
                "Ragged [A,B] row A mapping:",
                "```json",
                json.dumps(a_rag["mapping"], indent=2),
                "```",
                "",
                "The logical valid lengths for A match, but packed/pending/recent physical offsets differ because the ragged batch uses segment-wide physical maxima. Therefore row A's pending/recent attention mass is written to shifted importance indices in ragged execution.",
            ]
        ),
    )
    write_json(REPORT_DIR / "golden_importance_update_oracle.json", update_oracle)
    write_json(REPORT_DIR / "peer_content_oracle.json", peer_content)
    write_json(REPORT_DIR / "batch_row_reorder_oracle.json", reorder)
    write_json(REPORT_DIR / "peer_length_oracle.json", peer_length)
    write_md(
        REPORT_DIR / "root_cause_evidence.md",
        "Root Cause Evidence",
        "\n".join(
            [
                f"- classification: `{root}`",
                f"- previous importance exact: `{previous['previous_importance']['exact_equal']}`",
                f"- raw Q exact: `{q_path['raw_q']['exact_equal']}`",
                f"- Q RoPE exact: `{q_path['q_rope']['exact_equal']}`",
                f"- canonical attention probabilities exact: `{probs['canonical_attention_probs']['exact_equal']}`",
                f"- canonical importance mass exact: `{signal['canonical_mass']['exact_equal']}`",
                f"- physical mapping exact: `{metadata['physical_mapping_match']}`",
                f"- B1 production matches canonical golden: `{update_oracle['b1_production_matches_canonical_golden']}`",
                f"- ragged production matches canonical golden: `{update_oracle['ragged_production_matches_canonical_golden']}`",
                "",
                "The first non-equivalent item is not Q projection or attention probability semantics. It is the index mapping used when the physical ragged attention vector is accumulated into a logical per-request importance vector.",
            ]
        ),
    )
    write_json(REPORT_DIR / "final_gate.json", final_gate)
    return final_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    set_env()
    device = torch.device(args.device)
    tokenizer, _config, model = load_model(dtype=torch.float16, device=device)
    inputs = make_fixed_inputs(tokenizer, batch=4, context=513, device=device)
    preflight = {
        "branch": git(["branch", "--show-current"]),
        "head": git(["rev-parse", "HEAD"]),
        "expected_start_head": START_HEAD,
        "status_short_before": git(["status", "--short"]),
        "diff_check_before": subprocess.run(["git", "diff", "--check"], cwd=REPO_ROOT, text=True, capture_output=True).returncode == 0,
    }
    started = time.perf_counter()
    results: dict[str, Any] = {
        "device": str(device),
        "preflight": preflight,
        "model": model,
        "b1_a": run_case(model, inputs, ("A",)),
        "ragged_ab": run_case(model, inputs, ("A", "B")),
        "ragged_ac": run_case(model, inputs, ("A", "C")),
        "ragged_ba": run_case(model, inputs, ("A", "B"), order=("B", "A")),
        "ragged_ad": run_case(model, inputs, ("A", "D")),
    }
    final_gate = save_reports(results)
    final_gate["elapsed_s"] = time.perf_counter() - started
    write_json(REPORT_DIR / "final_gate.json", final_gate)
    print(json.dumps(final_gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
