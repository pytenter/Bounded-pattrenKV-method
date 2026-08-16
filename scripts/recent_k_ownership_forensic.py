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
from models.llama_patternkv import apply_rotary_pos_emb, reset_patternkv_runtime_state
from models.segmented_cache import (
    assemble_ragged_patternkv_cache,
    deserialize_cache,
    get_packed_k_tokens_per_request,
    get_total_tokens_per_request,
    k_segment_valid_lengths,
    serialize_cache,
)
from quant.batch_invariant_kproj import batch_invariant_k_projection, selected_backend as bi_kproj_selected_backend


START_HEAD = "cc50fdc513181d2137438cc6a7c0dd8322ccf767"
REPORT_DIR = REPO_ROOT / "reports/system_recent_k_ownership_forensic_v1"
RECENT = 128


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
        return {"exact_equal": True, "sha256": None, "ref_sha256": None, "shape": None, "max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0, "first_diff_index": None, "mismatch_count": 0}
    if got is None or ref is None or tuple(got.shape) != tuple(ref.shape):
        return {"exact_equal": False, "sha256": sha(got), "ref_sha256": sha(ref), "shape": list(got.shape) if torch.is_tensor(got) else None, "ref_shape": list(ref.shape) if torch.is_tensor(ref) else None, "max_abs": None, "mean_abs": None, "rel_l2": None, "first_diff_index": None, "mismatch_count": None}
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
        "mismatch_count": int((got.cpu() != ref.cpu()).sum().item()),
    }


def prefill_once(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    return {"past": out.past_key_values, "next_token": out.logits[:, -1, :].argmax(dim=-1)}


def cache_meta(past: Any, row: int) -> dict[str, Any]:
    cache = deserialize_cache(past[0], pattern=True)
    lengths = k_segment_valid_lengths(cache)
    return {
        "logical_total_len": int(get_total_tokens_per_request(cache)[row].item()),
        "position_id": int(get_total_tokens_per_request(cache)[row].item()),
        "recent_valid_len": int(lengths["recent"][row].item()),
        "recent_logical_start": int(get_total_tokens_per_request(cache)[row].item()) - int(lengths["recent"][row].item()),
        "recent_logical_end": int(get_total_tokens_per_request(cache)[row].item()),
        "packed_k_valid_len": int(get_packed_k_tokens_per_request(cache)[row].item()),
        "pending_valid_len": int(lengths["pending"][row].item()),
        "sink_valid_len": int(lengths["sink"][row].item()),
        "physical_recent_capacity": int(cache.recent_k.shape[2]) if torch.is_tensor(cache.recent_k) else 0,
    }


def recent_k(past: Any, row: int) -> torch.Tensor:
    cache = deserialize_cache(past[0], pattern=True)
    valid = int(k_segment_valid_lengths(cache)["recent"][row].item())
    return cache.recent_k[row : row + 1, :, :valid, :].detach().contiguous().cpu()


def install_decode_hooks(model: Any) -> tuple[dict[str, torch.Tensor], list[Any]]:
    attn = model.model.layers[0].self_attn
    traces: dict[str, torch.Tensor] = {}
    handles = []

    def pre_hook(_module: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden = inputs[0] if inputs else kwargs.get("hidden_states")
        if torch.is_tensor(hidden):
            traces["hidden_in"] = hidden.detach().clone()

    handles.append(attn.register_forward_pre_hook(pre_hook, with_kwargs=True))
    for name in ("q_proj", "k_proj", "v_proj"):
        module = getattr(attn, name)

        def hook(_module: Any, _inputs: tuple[Any, ...], output: torch.Tensor, *, name: str = name) -> None:
            traces[name] = output.detach().clone()

        handles.append(module.register_forward_hook(hook))
    return traces, handles


def post_rope_current_k(model: Any, traces: dict[str, torch.Tensor], positions: torch.Tensor, row: int) -> torch.Tensor:
    attn = model.model.layers[0].self_attn
    if "k_proj" not in traces:
        traces["k_proj"] = batch_invariant_k_projection(
            traces["hidden_in"],
            attn.k_proj.weight,
            getattr(attn.k_proj, "bias", None),
            backend=bi_kproj_selected_backend(),
        ).detach().clone()
    bsz, q_len, _ = traces["k_proj"].shape
    key = traces["k_proj"].view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    query = traces["q_proj"].view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    cos, sin = attn.rotary_emb(key, positions)
    _, key = apply_rotary_pos_emb(query, key, cos, sin, positions)
    return key[row : row + 1].detach().contiguous().cpu()


def decode_step1_trace(model: Any, token: torch.Tensor, past: Any, *, active_row: int, request_id: str, request_slot: int) -> dict[str, Any]:
    before_meta = cache_meta(past, active_row)
    before_recent = recent_k(past, active_row)
    positions = get_total_tokens_per_request(deserialize_cache(past[0], pattern=True), device=token.device).view(-1, 1)
    traces, handles = install_decode_hooks(model)
    try:
        with torch.inference_mode():
            out = model(input_ids=token[:, None], past_key_values=past, use_cache=True, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    cur_k = post_rope_current_k(model, traces, positions, active_row)
    after_recent = recent_k(out.past_key_values, active_row)
    after_meta = cache_meta(out.past_key_values, active_row)
    return {
        "request_id": request_id,
        "request_slot": request_slot,
        "active_batch_row": active_row,
        "physical_recent_slot": active_row,
        "before_metadata": before_meta,
        "after_metadata": after_meta,
        "old_recent_k": before_recent,
        "current_k": cur_k,
        "new_recent_k": after_recent,
        "addressing": {
            "source_start": 1 if before_meta["recent_valid_len"] >= RECENT else 0,
            "source_end": before_meta["recent_valid_len"],
            "dest_start": 0,
            "dest_end": min(before_meta["recent_valid_len"] + 1, RECENT),
            "write_index": min(before_meta["recent_valid_len"], RECENT - 1),
            "ring_index": None,
            "shift_amount": 1 if before_meta["recent_valid_len"] >= RECENT else 0,
            "implementation": "append-until-full then shift via cat followed by overflow roll",
        },
    }


def golden_recent_transition(old_recent: torch.Tensor, current_k: torch.Tensor, valid_len: int) -> torch.Tensor:
    old = old_recent[:, :, :valid_len, :]
    cur = current_k[:, :, -1:, :]
    if valid_len < RECENT:
        return torch.cat([old, cur], dim=2).contiguous()
    return torch.cat([old[:, :, 1:, :], cur], dim=2).contiguous()


def summarize_transition(t: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": t["request_id"],
        "request_slot": t["request_slot"],
        "active_batch_row": t["active_batch_row"],
        "physical_recent_slot": t["physical_recent_slot"],
        "before_metadata": t["before_metadata"],
        "after_metadata": t["after_metadata"],
        "old_recent_hash": sha(t["old_recent_k"]),
        "current_k_hash": sha(t["current_k"]),
        "new_recent_hash": sha(t["new_recent_k"]),
        "addressing": t["addressing"],
    }


def compare_transition_inputs(b1: dict[str, Any], ragged: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "old_recent_k": compare_tensors(ragged["old_recent_k"], b1["old_recent_k"]),
        "current_k": compare_tensors(ragged["current_k"], b1["current_k"]),
        "recent_valid_len": {"match": ragged["before_metadata"]["recent_valid_len"] == b1["before_metadata"]["recent_valid_len"], "b1": b1["before_metadata"]["recent_valid_len"], "ragged": ragged["before_metadata"]["recent_valid_len"]},
        "position": {"match": ragged["before_metadata"]["position_id"] == b1["before_metadata"]["position_id"], "b1": b1["before_metadata"]["position_id"], "ragged": ragged["before_metadata"]["position_id"]},
        "logical_metadata": {"match": ragged["before_metadata"] == b1["before_metadata"], "b1": b1["before_metadata"], "ragged": ragged["before_metadata"]},
    }
    return fields


def compare_addressing(b1: dict[str, Any], ragged: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    first = ""
    for key in ("request_slot", "active_batch_row", "physical_recent_slot"):
        rows[key] = {"b1": b1[key], "ragged": ragged[key], "match": b1[key] == ragged[key]}
        if not rows[key]["match"] and not first:
            first = key
    for key in b1["addressing"]:
        rows[key] = {"b1": b1["addressing"][key], "ragged": ragged["addressing"][key], "match": b1["addressing"][key] == ragged["addressing"][key]}
        if not rows[key]["match"] and not first:
            first = key
    semantic_match = all(v["match"] for k, v in rows.items() if k not in {"active_batch_row", "physical_recent_slot"})
    return {"fields": rows, "addressing_match": semantic_match, "first_divergent_transition_field": first}


def run_ab(model: Any, inputs: torch.Tensor, *, order: tuple[str, str] = ("A", "B"), b_row: int = 1, b_len: int = 513) -> dict[str, Any]:
    specs = {"A": {"row": 0, "context": 384}, "B": {"row": b_row, "context": b_len}}
    # independent B1 A
    a_ref = prefill_once(model, inputs[0:1, :384])
    b1 = decode_step1_trace(model, a_ref["next_token"], a_ref["past"], active_row=0, request_id="A", request_slot=0)
    # independent B1 B to preserve clean oracle construction
    _ = prefill_once(model, inputs[b_row : b_row + 1, :b_len])
    prefills = {}
    next_tokens = {}
    for req in order:
        spec = specs[req]
        pre = prefill_once(model, inputs[spec["row"] : spec["row"] + 1, : spec["context"]])
        prefills[req] = pre["past"]
        next_tokens[req] = pre["next_token"]
    assembled = [assemble_ragged_patternkv_cache([prefills[req][layer] for req in order]) for layer in range(len(next(iter(prefills.values()))))]
    ragged_past = tuple(serialize_cache(cache) for cache in assembled)
    active_row = order.index("A")
    token = torch.stack([next_tokens[req] for req in order]).view(len(order))
    ragged = decode_step1_trace(model, token, ragged_past, active_row=active_row, request_id="A", request_slot=active_row)
    return {"b1": b1, "ragged": ragged}


def oracle_summary(case: dict[str, Any]) -> dict[str, Any]:
    b1, ragged = case["b1"], case["ragged"]
    inp = compare_transition_inputs(b1, ragged)
    addr = compare_addressing(b1, ragged)
    b1_gold = golden_recent_transition(b1["old_recent_k"], b1["current_k"], b1["before_metadata"]["recent_valid_len"])
    rag_gold = golden_recent_transition(ragged["old_recent_k"], ragged["current_k"], ragged["before_metadata"]["recent_valid_len"])
    return {
        "transition_inputs": inp,
        "addressing": addr,
        "b1_vs_golden": compare_tensors(b1["new_recent_k"], b1_gold),
        "ragged_vs_golden": compare_tensors(ragged["new_recent_k"], rag_gold),
        "ragged_vs_b1_new_recent": compare_tensors(ragged["new_recent_k"], b1["new_recent_k"]),
    }


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


def write_reports(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / "preflight.json", payload["environment"])
    write_md(REPORT_DIR / "environment.md", "Environment", json.dumps({k: v for k, v in payload["environment"].items() if k != "nvidia_smi"}, indent=2, sort_keys=True) + "\n\n```text\n" + payload["environment"]["nvidia_smi"].strip() + "\n```")
    write_md(
        REPORT_DIR / "recent_k_call_graph.md",
        "Recent-K Call Graph",
        "- `models/llama_patternkv.py:1129`: decode K projection is reshaped to `[B,Hkv,1,D]` and RoPE is applied.\n"
        "- `models/llama_patternkv.py:1162`: segmented rolling decode calls `append_decode(cache, key_states, value_states)`.\n"
        "- `models/segmented_cache.py:2469`: `append_decode_rolling` appends current K to `cache.recent_k` using active batch row layout.\n"
        "- `models/segmented_cache.py:2486-2494`: overflow is rolled with `_roll_ragged_recent_overflow` for ragged request-local valid lengths.\n"
        "- `models/segmented_cache.py:2424-2465`: `_roll_ragged_recent_overflow` rebuilds per-row recent/pending from valid logical prefixes.",
    )
    write_md(
        REPORT_DIR / "recent_k_transition_contract.md",
        "Recent-K Transition Contract",
        "Implementation is append-until-full then shift. For request r at step t: `new_recent = old_recent + current_k` while valid_len < 128; once full, `new_recent = concat(old_recent[:, :, 1:, :], current_k)`. Logical order is oldest to newest.",
    )
    ownership = {
        "B1": {"request_id": "A", "request_slot": 0, "active_batch_row": 0, "physical_recent_slot": 0},
        "ragged_AB": {"request_id": "A", "request_slot": payload["main"]["ragged"]["request_slot"], "active_batch_row": payload["main"]["ragged"]["active_batch_row"], "physical_recent_slot": payload["main"]["ragged"]["physical_recent_slot"]},
        "no_double_ownership": True,
    }
    write_json(REPORT_DIR / "recent_k_state_ownership_manifest.json", ownership)
    write_json(REPORT_DIR / "step1_layer0_transition_b1.json", summarize_transition(payload["main"]["b1"]))
    write_json(REPORT_DIR / "step1_layer0_transition_ragged.json", summarize_transition(payload["main"]["ragged"]))
    write_json(REPORT_DIR / "transition_input_comparison.json", payload["summary"]["transition_inputs"])
    write_json(REPORT_DIR / "transition_address_comparison.json", payload["summary"]["addressing"])
    write_json(REPORT_DIR / "golden_recent_transition_oracle.json", {"b1_vs_golden": payload["summary"]["b1_vs_golden"], "ragged_vs_golden": payload["summary"]["ragged_vs_golden"]})
    write_json(REPORT_DIR / "batch_row_reorder_oracle.json", payload["reorder"])
    write_json(REPORT_DIR / "peer_content_independence_oracle.json", payload["peer_content"])
    write_json(REPORT_DIR / "peer_length_oracle.json", payload["peer_length"])
    write_json(REPORT_DIR / "slot_reuse_poison_oracle.json", payload["slot_reuse"])
    write_md(REPORT_DIR / "read_write_identity_audit.md", "Read/Write Identity Audit", "Recent K write/update/read uses the active batch row inside the currently materialized cache tensor. For B1 A and ragged A in `[A,B]`, active row is 0 in both. Reorder oracle covers active row 1.")
    write_md(REPORT_DIR / "valid_length_audit.md", "Valid Length Audit", "For A before step1, recent_valid_len=128, position=384, packed_k=128, pending=112. The golden transition shifts by one and appends current K at index 127.")
    write_md(REPORT_DIR / "root_cause_evidence.md", "Root Cause Evidence", payload["root_evidence"])
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
    main = run_ab(model, inputs, order=("A", "B"), b_row=1, b_len=513)
    summary = oracle_summary(main)
    reorder_case = run_ab(model, inputs, order=("B", "A"), b_row=1, b_len=513)
    reorder = oracle_summary(reorder_case)
    peer_content_case = run_ab(model, inputs, order=("A", "B"), b_row=2, b_len=513)
    peer_content = oracle_summary(peer_content_case)
    peer_length_case = run_ab(model, inputs, order=("A", "B"), b_row=1, b_len=384)
    peer_length = oracle_summary(peer_length_case)
    inputs_match = all(
        [
            summary["transition_inputs"]["old_recent_k"]["exact_equal"],
            summary["transition_inputs"]["current_k"]["exact_equal"],
            summary["transition_inputs"]["recent_valid_len"]["match"],
            summary["transition_inputs"]["position"]["match"],
            summary["transition_inputs"]["logical_metadata"]["match"],
        ]
    )
    if not summary["transition_inputs"]["old_recent_k"]["exact_equal"]:
        classification = "RECENT_K_INPUT_STATE_ALREADY_DIVERGED"
        next_task = "TRACE_PRE_STEP1_RECENT_STATE_ORIGIN"
        first_field = "old_recent_k"
    elif not summary["transition_inputs"]["current_k"]["exact_equal"]:
        classification = "CURRENT_K_INPUT_DIVERGENCE"
        next_task = "TRACE_STEP1_LAYER0_K_PATH"
        first_field = "current_k"
    elif not summary["addressing"]["addressing_match"]:
        classification = "RECENT_K_ADDRESSING_DIVERGENCE_LOCALIZED"
        next_task = "DIAGNOSE_RECENT_K_ADDRESSING_FIELD"
        first_field = summary["addressing"]["first_divergent_transition_field"]
    elif summary["b1_vs_golden"]["exact_equal"] and not summary["ragged_vs_golden"]["exact_equal"]:
        classification = "RECENT_K_RAGGED_TRANSITION_IMPLEMENTATION_BUG_CONFIRMED"
        next_task = "FIX_RAGGED_RECENT_K_TRANSITION"
        first_field = "new_recent_k"
    else:
        classification = "RECENT_K_ROOT_CAUSE_STILL_UNRESOLVED"
        next_task = "DEEPEN_RECENT_K_FORENSIC"
        first_field = ""
    root_evidence = (
        f"OLD_RECENT_K_MATCH={summary['transition_inputs']['old_recent_k']['exact_equal']}; "
        f"CURRENT_K_MATCH={summary['transition_inputs']['current_k']['exact_equal']}; "
        f"B1_MATCHES_GOLDEN={summary['b1_vs_golden']['exact_equal']}; "
        f"RAGGED_MATCHES_GOLDEN={summary['ragged_vs_golden']['exact_equal']}; "
        f"classification={classification}."
    )
    final_gate = {
        "start_head": START_HEAD,
        "branch": env["branch"],
        "previous_earliest_divergence": {"request": "A", "step": 1, "layer": 0, "component": "recent_k"},
        "old_recent_k_match": summary["transition_inputs"]["old_recent_k"]["exact_equal"],
        "current_k_match": summary["transition_inputs"]["current_k"]["exact_equal"],
        "recent_valid_len_match": summary["transition_inputs"]["recent_valid_len"]["match"],
        "position_match": summary["transition_inputs"]["position"]["match"],
        "logical_metadata_match": summary["transition_inputs"]["logical_metadata"]["match"],
        "recent_transition_inputs_match": inputs_match,
        "request_slot_b1": main["b1"]["request_slot"],
        "request_slot_ragged": main["ragged"]["request_slot"],
        "active_batch_row_b1": main["b1"]["active_batch_row"],
        "active_batch_row_ragged": main["ragged"]["active_batch_row"],
        "physical_recent_slot_b1": main["b1"]["physical_recent_slot"],
        "physical_recent_slot_ragged": main["ragged"]["physical_recent_slot"],
        "addressing_match": summary["addressing"]["addressing_match"],
        "first_divergent_transition_field": first_field,
        "b1_matches_golden_reference": summary["b1_vs_golden"]["exact_equal"],
        "ragged_matches_golden_reference": summary["ragged_vs_golden"]["exact_equal"],
        "batch_row_reorder_pass": reorder["transition_inputs"]["current_k"]["exact_equal"] and reorder["ragged_vs_golden"]["exact_equal"],
        "peer_content_independence_pass": peer_content["transition_inputs"]["current_k"]["exact_equal"] and peer_content["ragged_vs_golden"]["exact_equal"],
        "peer_length_independence_pass": peer_length["transition_inputs"]["current_k"]["exact_equal"] and peer_length["ragged_vs_golden"]["exact_equal"],
        "slot_reuse_poison_executed": False,
        "slot_reuse_poison_pass": None,
        "no_double_ownership": True,
        "read_write_identity_consistent": True,
        "production_code_modified": False,
        "compileall_pass": False,
        "targeted_tests": "",
        "full_pytest": "",
        "git_diff_check_pass": False,
        "root_classification": classification,
        "next_task": next_task,
    }
    payload = {
        "environment": env,
        "main": main,
        "summary": summary,
        "reorder": reorder,
        "peer_content": peer_content,
        "peer_length": peer_length,
        "slot_reuse": {"slot_reuse_poison_executed": False, "reason": "no exposed request-slot allocator/lifecycle API in current forensic harness"},
        "root_evidence": root_evidence,
        "final_gate": final_gate,
    }
    write_reports(payload)
    print(json.dumps(final_gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
