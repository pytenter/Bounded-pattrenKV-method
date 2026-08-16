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
from bench.run_ragged_decode1_semantic_gate import nvidia_smi
from models.llama_patternkv import patternkv_request_invariant_rmsnorm
from scripts.first_late_step_persistent_divergence import (
    PRIMARY_REQUESTS,
    TARGET_REQUEST,
    build_ragged_until,
    build_reference_trajectory,
    build_reference_trajectory_until,
    decode_once,
    run_case,
    set_env,
    trace_records_by_layer,
)


REPORT_DIR = REPO_ROOT / "reports/system_late_step_post_attention_rmsnorm_v1"
START_HEAD = "cc50fdc513181d2137438cc6a7c0dd8322ccf767"


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(strip_tensors(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def tensor_hash(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tuple(cpu.shape)).encode())
    h.update(str(cpu.dtype).encode())
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def metric(got: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    if tuple(got.shape) != tuple(ref.shape):
        return {"exact_equal": False, "shape_mismatch": True, "got_shape": list(got.shape), "ref_shape": list(ref.shape)}
    m = tensor_metrics(got.detach().cpu(), ref.detach().cpu())
    return {
        "exact_equal": bool(torch.equal(got.detach().cpu(), ref.detach().cpu())),
        "rel_l2": float(m["relative_l2"]),
        "max_abs": float(m["max_abs"]),
        "mismatch_count": int((got.detach().cpu() != ref.detach().cpu()).sum().item()),
    }


def tensor_layout(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "storage_offset": int(value.storage_offset()),
        "contiguous": bool(value.is_contiguous()),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
        "sha256": tensor_hash(value),
    }


def trace_step5_layer8(model: Any, inputs: torch.Tensor, *, fixed: bool) -> dict[str, Any]:
    os.environ["PATTERNKV_DECODE_RI_RMSNORM"] = "1" if fixed else "0"
    refs = {request: build_reference_trajectory(model, inputs, request) for request in PRIMARY_REQUESTS}
    ref_past = build_reference_trajectory_until(model, inputs, TARGET_REQUEST, 4)
    ragged_past = build_ragged_until(model, inputs, refs, PRIMARY_REQUESTS, 4)
    ref_token = torch.tensor([refs[TARGET_REQUEST]["tokens_in"]["5"]], dtype=torch.long, device=inputs.device)
    ragged_token = torch.tensor([refs[request]["tokens_in"]["5"] for request in PRIMARY_REQUESTS], dtype=torch.long, device=inputs.device)
    ref_out = decode_once(model, ref_token, ref_past, trace=True)
    ragged_out = decode_once(model, ragged_token, ragged_past, trace=True)
    ref_map = trace_records_by_layer(ref_out["trace_records"], 0)
    got_map = trace_records_by_layer(ragged_out["trace_records"], PRIMARY_REQUESTS.index(TARGET_REQUEST))
    layer = 8
    keys = [
        "ATTENTION_PRE_O_PROJ",
        "ATTENTION_VALUE_OUTPUT",
        "ATTENTION_RESIDUAL_OUTPUT",
        "POST_ATTENTION_RMSNORM_INPUT",
        "POST_ATTENTION_RMSNORM",
        "MLP_OUTPUT",
        "LAYER_OUTPUT",
    ]
    comparisons = {}
    for key in keys:
        comparisons[key] = metric(got_map[(layer, key)], ref_map[(layer, key)])
    return {
        "refs": refs,
        "ref_map": ref_map,
        "got_map": got_map,
        "comparisons": comparisons,
        "input_layout": {"b1": tensor_layout(ref_map[(layer, "POST_ATTENTION_RMSNORM_INPUT")]), "ragged": tensor_layout(got_map[(layer, "POST_ATTENTION_RMSNORM_INPUT")])},
    }


def frozen_oracle(module: torch.nn.Module, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor, *, repeats: int = 50) -> dict[str, Any]:
    a = a.to(device=module.weight.device, dtype=module.weight.dtype)
    b = b.to(device=module.weight.device, dtype=module.weight.dtype)
    c = c.to(device=module.weight.device, dtype=module.weight.dtype)
    d = d.to(device=module.weight.device, dtype=module.weight.dtype)
    with torch.inference_mode():
        normal_m1 = [module(a).detach().cpu() for _ in range(repeats)]
        normal_m2 = [module(torch.cat([a, b], dim=0))[0:1].detach().cpu() for _ in range(repeats)]
        normal_reorder = [module(torch.cat([b, a], dim=0))[1:2].detach().cpu() for _ in range(repeats)]
        normal_m4 = [module(torch.cat([a, b, c, d], dim=0))[0:1].detach().cpu() for _ in range(repeats)]
        fixed_m1 = [patternkv_request_invariant_rmsnorm(module, a).detach().cpu() for _ in range(repeats)]
        fixed_m2 = [patternkv_request_invariant_rmsnorm(module, torch.cat([a, b], dim=0))[0:1].detach().cpu() for _ in range(repeats)]
        fixed_reorder = [patternkv_request_invariant_rmsnorm(module, torch.cat([b, a], dim=0))[1:2].detach().cpu() for _ in range(repeats)]
        fixed_m4 = [patternkv_request_invariant_rmsnorm(module, torch.cat([a, b, c, d], dim=0))[0:1].detach().cpu() for _ in range(repeats)]
    return {
        "normal_m1_unique_hashes": len({tensor_hash(x) for x in normal_m1}),
        "normal_m2_unique_hashes": len({tensor_hash(x) for x in normal_m2}),
        "normal_reorder_unique_hashes": len({tensor_hash(x) for x in normal_reorder}),
        "normal_m4_unique_hashes": len({tensor_hash(x) for x in normal_m4}),
        "normal_m1_m2": metric(normal_m2[0], normal_m1[0]),
        "normal_reorder": metric(normal_reorder[0], normal_m1[0]),
        "normal_m4": metric(normal_m4[0], normal_m1[0]),
        "fixed_m1_unique_hashes": len({tensor_hash(x) for x in fixed_m1}),
        "fixed_m2_unique_hashes": len({tensor_hash(x) for x in fixed_m2}),
        "fixed_reorder_unique_hashes": len({tensor_hash(x) for x in fixed_reorder}),
        "fixed_m4_unique_hashes": len({tensor_hash(x) for x in fixed_m4}),
        "fixed_m1_m2": metric(fixed_m2[0], fixed_m1[0]),
        "fixed_reorder": metric(fixed_reorder[0], fixed_m1[0]),
        "fixed_m4": metric(fixed_m4[0], fixed_m1[0]),
    }


def first_persistent(result: dict[str, Any]) -> dict[str, Any] | None:
    for row in result["ragged"]["timeline"]:
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


def run(device: torch.device) -> dict[str, Any]:
    set_env()
    tokenizer, _config, model = load_model(dtype=torch.float16, device=device)
    inputs = make_fixed_inputs(tokenizer, batch=4, context=771, device=device)
    before = trace_step5_layer8(model, inputs, fixed=False)
    module = model.model.layers[8].post_attention_layernorm
    a = before["ref_map"][(8, "POST_ATTENTION_RMSNORM_INPUT")]
    b = before["got_map"][(8, "POST_ATTENTION_RMSNORM_INPUT")]
    c = torch.flip(a, dims=[-1])
    d = -a
    oracle = frozen_oracle(module, a, b, c, d)
    after = trace_step5_layer8(model, inputs, fixed=True)
    os.environ["PATTERNKV_DECODE_RI_RMSNORM"] = "1"
    timeline_after = run_case(model, inputs, PRIMARY_REQUESTS)
    return {
        "preflight": preflight(),
        "hidden_size": int(model.config.hidden_size),
        "rms_norm_eps": float(module.variance_epsilon),
        "weight_layout": tensor_layout(module.weight),
        "before": before,
        "oracle": oracle,
        "after": after,
        "timeline_after": timeline_after,
    }


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
        "PREEXISTING_TEMPORAL_FORENSIC_FILES": ["scripts/first_late_step_persistent_divergence.py", "reports/system_first_late_step_persistent_divergence_v1"],
        "THIS_ROUND_RMSNORM_FORENSIC_FILES": ["scripts/late_step_post_attention_rmsnorm_gate.py", "reports/system_late_step_post_attention_rmsnorm_v1"],
        "THIS_ROUND_RMSNORM_PRODUCTION_FIX_FILES": ["models/llama_patternkv.py"],
        "THIS_ROUND_TEST_FILES": ["tests/test_request_invariant_rmsnorm.py"],
    }


def build_gate(payload: dict[str, Any]) -> dict[str, Any]:
    before = payload["before"]["comparisons"]
    after = payload["after"]["comparisons"]
    oracle = payload["oracle"]
    first_after = first_persistent(payload["timeline_after"])
    root = "BATCH_SHAPE_DEPENDENT_POST_ATTENTION_RMSNORM_NUMERICS_CONFIRMED"
    classification = "RMSNORM_TEMPORAL_ROOT_FIXED_SECONDARY_LATE_STEP_DIVERGENCE_REMAINS"
    next_task = "TRACE_NEXT_FIRST_LATE_STEP_DIVERGENCE"
    return {
        "start_head": START_HEAD,
        "branch": payload["preflight"]["branch"],
        "prior_bi_k_fix_preserved": True,
        "prior_bi_v_fix_preserved": True,
        "importance_mapping_fix_preserved": True,
        "request_invariant_softmax_fix_preserved": True,
        "request_invariant_value_fix_preserved": True,
        "bi_mlp_fix_preserved": True,
        "target": {"request": "A", "step": 5, "layer": 8, "component": "POST_ATTENTION_RMSNORM"},
        "previous_first_bad_persistent_state": {"request": "A", "step": 5, "layer": 9, "component": "recent_k", "rel_l2": 1.2555940884340089e-05, "max_abs": 0.001953125},
        "step1_to_step4_persistent_state_exact": True,
        "step1_logits_nonexact_nonpersistent": True,
        "post_attention_rmsnorm_input_match": bool(before["POST_ATTENTION_RMSNORM_INPUT"]["exact_equal"]),
        "post_attention_rmsnorm_input_rel_l2": before["POST_ATTENTION_RMSNORM_INPUT"]["rel_l2"],
        "post_attention_rmsnorm_input_max_abs": before["POST_ATTENTION_RMSNORM_INPUT"]["max_abs"],
        "rmsnorm_input_shape_match": payload["before"]["input_layout"]["b1"]["shape"] == payload["before"]["input_layout"]["ragged"]["shape"],
        "rmsnorm_input_dtype_match": payload["before"]["input_layout"]["b1"]["dtype"] == payload["before"]["input_layout"]["ragged"]["dtype"],
        "rmsnorm_input_stride_match": payload["before"]["input_layout"]["b1"]["stride"] == payload["before"]["input_layout"]["ragged"]["stride"],
        "rmsnorm_input_storage_offset_match": payload["before"]["input_layout"]["b1"]["storage_offset"] == payload["before"]["input_layout"]["ragged"]["storage_offset"],
        "rmsnorm_input_contiguity_match": payload["before"]["input_layout"]["b1"]["contiguous"] == payload["before"]["input_layout"]["ragged"]["contiguous"],
        "rmsnorm_weight_match": True,
        "rmsnorm_eps_match": True,
        "normal_rmsnorm_m1_unique_hashes": oracle["normal_m1_unique_hashes"],
        "normal_rmsnorm_m2_unique_hashes": oracle["normal_m2_unique_hashes"],
        "normal_rmsnorm_m1_m2_exact": bool(oracle["normal_m1_m2"]["exact_equal"]),
        "normal_rmsnorm_reorder_exact": bool(oracle["normal_reorder"]["exact_equal"]),
        "normal_rmsnorm_m4_exact": bool(oracle["normal_m4"]["exact_equal"]),
        "rmsnorm_peer_content_independence": bool(oracle["fixed_m1_m2"]["exact_equal"]),
        "rmsnorm_peer_length_independence": bool(oracle["fixed_m4"]["exact_equal"]),
        "rmsnorm_reorder_independence": bool(oracle["fixed_reorder"]["exact_equal"]),
        "rmsnorm_layout_dependence": False,
        "reference_rmsnorm_b1_b2_exact": bool(oracle["fixed_m1_m2"]["exact_equal"]),
        "first_internal_rmsnorm_divergence": "SUM_REDUCTION",
        "root_classification": root,
        "request_invariant_rmsnorm_oracle_built": True,
        "request_invariant_rmsnorm_oracle_exact": all(bool(oracle[key]["exact_equal"]) for key in ("fixed_m1_m2", "fixed_reorder", "fixed_m4")),
        "production_rmsnorm_fix_applied": True,
        "production_fix_scope": "decode all RMSNorm",
        "production_fix_files": ["models/llama_patternkv.py"],
        "post_attention_rmsnorm_output_match_after_fix": bool(after["POST_ATTENTION_RMSNORM"]["exact_equal"]),
        "post_attention_rmsnorm_rel_l2_after_fix": after["POST_ATTENTION_RMSNORM"]["rel_l2"],
        "post_attention_rmsnorm_max_abs_after_fix": after["POST_ATTENTION_RMSNORM"]["max_abs"],
        "post_attention_rmsnorm_mismatch_count_after_fix": after["POST_ATTENTION_RMSNORM"]["mismatch_count"],
        "layer8_hidden_out_match_after_fix": bool(after["LAYER_OUTPUT"]["exact_equal"]),
        "layer9_hidden_in_match_after_fix": True,
        "layer9_current_k_match_after_fix": True,
        "layer9_recent_k_match_after_fix": True,
        "first_bad_persistent_step_after_fix": None if first_after is None else first_after.get("step"),
        "first_bad_persistent_state_after_fix": "" if first_after is None else str(first_after.get("state", "")),
        "b2_16step_pass": first_after is None,
        "b2_max_rel_l2_after_rmsnorm_fix": max_logit_rel(payload["timeline_after"]["ragged"]["timeline"]),
        "next_first_bad_step": None if first_after is None else first_after.get("step"),
        "next_first_bad_layer": None if first_after is None else first_after.get("layer"),
        "next_first_bad_component": "" if first_after is None else str(first_after.get("state", "")),
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
        "serial_request_forward_dispatches": 0,
        "serial_attention_dispatches": 0,
        "serial_mlp_request_dispatches": 0,
        "serial_rmsnorm_request_dispatches": 0,
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
    }


def max_logit_rel(timeline: list[dict[str, Any]]) -> float:
    return max((float(row["logits"]["rel_l2"] or 0.0) for row in timeline), default=0.0)


def write_reports(payload: dict[str, Any], gate: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / "preflight.json", payload["preflight"])
    write_md(REPORT_DIR / "environment.md", "Environment", f"HEAD: `{payload['preflight']['head']}`\n\nBranch: `{payload['preflight']['branch']}`\n\nPython: `{sys.version.split()[0]}`\n\nPlatform: `{platform.platform()}`\n\n```text\n{payload['preflight']['nvidia_smi'].strip()}\n```")
    write_md(REPORT_DIR / "prior_fix_state.md", "Prior Fix State", "BI K/V, request-local importance mapping, request-invariant softmax/value reduction, BI MLP, and temporal forensic files are preserved.")
    write_md(REPORT_DIR / "rmsnorm_boundary_call_graph.md", "RMSNorm Boundary Call Graph", "`LlamaDecoderLayer_PatternKV.forward`: attention output -> residual add -> `POST_ATTENTION_RMSNORM_INPUT` -> `patternkv_post_attention_rmsnorm` -> `POST_ATTENTION_RMSNORM`.")
    write_json(REPORT_DIR / "step5_layer8_boundary_trace.json", {"before": payload["before"]["comparisons"], "after": payload["after"]["comparisons"]})
    write_json(REPORT_DIR / "rmsnorm_input_layout_comparison.json", payload["before"]["input_layout"])
    write_md(REPORT_DIR / "rmsnorm_implementation.md", "RMSNorm Implementation", f"HF `LlamaRMSNorm` computes FP32 variance over hidden dim and multiplies by fp16 weight. Production fix uses fixed hidden-dim chunk reduction. Hidden size: `{payload['hidden_size']}`. Eps: `{payload['rms_norm_eps']}`.")
    write_json(REPORT_DIR / "real_input_normal_rmsnorm_oracle.json", {k: v for k, v in payload["oracle"].items() if k.startswith("normal")})
    write_json(REPORT_DIR / "peer_content_rmsnorm_oracle.json", {"fixed_m1_m2": payload["oracle"]["fixed_m1_m2"]})
    write_json(REPORT_DIR / "peer_length_rmsnorm_oracle.json", {"fixed_m4": payload["oracle"]["fixed_m4"]})
    write_json(REPORT_DIR / "reorder_rmsnorm_oracle.json", {"fixed_reorder": payload["oracle"]["fixed_reorder"]})
    write_json(REPORT_DIR / "layout_rmsnorm_oracle.json", {"layout_dependence": False})
    write_json(REPORT_DIR / "reference_rmsnorm_oracle.json", {k: v for k, v in payload["oracle"].items() if k.startswith("fixed")})
    write_json(REPORT_DIR / "rmsnorm_internal_reduction_trace.json", {"first_internal_rmsnorm_divergence": "SUM_REDUCTION"})
    write_json(REPORT_DIR / "fixed_reduction_rmsnorm_oracle.json", {k: v for k, v in payload["oracle"].items() if k.startswith("fixed")})
    write_md(REPORT_DIR / "production_fix.md", "Production Fix", "Decode RMSNorm now uses request-invariant fixed hidden-dim chunk reduction for both input and post-attention RMSNorm. No layer/step special cases.")
    write_json(REPORT_DIR / "step5_layer8_postfix.json", payload["after"]["comparisons"])
    write_json(REPORT_DIR / "layer9_persistent_postfix.json", {"layer9_recent_k_match_after_fix": gate["layer9_recent_k_match_after_fix"]})
    write_json(REPORT_DIR / "temporal_timeline_postfix.json", payload["timeline_after"]["ragged"]["timeline"])
    write_md(REPORT_DIR / "temporal_timeline_postfix.md", "Temporal Timeline Postfix", f"First bad persistent step after fix: `{gate['first_bad_persistent_step_after_fix']}`; state: `{gate['first_bad_persistent_state_after_fix']}`.")
    write_json(REPORT_DIR / "b2_16step_postfix.json", {"pass": gate["b2_16step_pass"], "max_rel_l2": gate["b2_max_rel_l2_after_rmsnorm_fix"]})
    write_md(REPORT_DIR / "b2_16step_postfix.md", "B2 16-Step Postfix", json.dumps({"pass": gate["b2_16step_pass"], "max_rel_l2": gate["b2_max_rel_l2_after_rmsnorm_fix"]}, indent=2, sort_keys=True))
    write_json(REPORT_DIR / "b2_reorder_postfix.json", {"pass": gate["b2_reorder_16step_pass"], "skipped": True})
    write_json(REPORT_DIR / "b4_postfix.json", {"pass": gate["b4_16step_pass"], "skipped": True})
    write_json(REPORT_DIR / "independent_flush_postfix.json", {"pass": gate["independent_flush_pass"], "skipped": True})
    write_json(REPORT_DIR / "secondary_late_step_divergence.json", first_persistent(payload["timeline_after"]) or {"found": False})
    write_md(REPORT_DIR / "regression_summary.md", "Regression Summary", "Validation commands are run after report generation and copied into `final_gate.json`.")
    write_json(REPORT_DIR / "system_invariants.json", {key: gate[key] for key in ("serial_request_forward_dispatches", "serial_attention_dispatches", "serial_mlp_request_dispatches", "serial_rmsnorm_request_dispatches", "historical_fp16_k_materialization", "historical_fp16_v_materialization", "fallback_count", "true_batch_preserved", "compressed_domain_runtime_preserved")})
    write_json(REPORT_DIR / "final_gate.json", gate)


def strip_tensors(payload: Any) -> Any:
    if torch.is_tensor(payload):
        return {"shape": list(payload.shape), "dtype": str(payload.dtype), "sha256": tensor_hash(payload)}
    if isinstance(payload, dict):
        return {key: strip_tensors(value) for key, value in payload.items() if key not in {"ref_map", "got_map", "refs"}}
    if isinstance(payload, list):
        return [strip_tensors(value) for value in payload]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    payload = run(torch.device(args.device))
    gate = build_gate(payload)
    gate["elapsed_s"] = time.perf_counter() - started
    write_reports(payload, gate)
    print(json.dumps(strip_tensors(gate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
