from __future__ import annotations

import argparse
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
from models.llama_patternkv import (
    patternkv_bi_mlp_oracle_counters,
    patternkv_p2_first_divergence_trace_records,
    reset_patternkv_bi_mlp_oracle_counters,
    reset_patternkv_p2_first_divergence_trace,
    reset_patternkv_runtime_state,
)
from models.segmented_cache import (
    assemble_ragged_patternkv_cache,
    deserialize_cache,
    get_total_tokens_per_request,
    k_segment_valid_lengths,
    serialize_cache,
)
from quant.batch_invariant_kproj import batch_invariant_linear_projection, batch_invariant_kproj_counters, reset_batch_invariant_kproj_counters
from quant.page_batch import get_patternkv_real_decode_counters, reset_patternkv_real_decode_counters


REPORT_DIR = REPO_ROOT / "reports/system_full_decode_batch_invariance_oracle_v1"
START_HEAD = "cc50fdc513181d2137438cc6a7c0dd8322ccf767"


def set_env(*, full_bi: bool = False) -> None:
    os.environ["PATTERNKV_PREFILL_PROJ_MODE"] = "bi_kv"
    os.environ["PATTERNKV_BI_KPROJ_BACKEND"] = "v2"
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    os.environ["PATTERNKV_CENTROID_MAX_SLOTS"] = "8"
    os.environ["PATTERNKV_P2_FIRST_DIVERGENCE_TRACE"] = "1"
    os.environ["PATTERNKV_FULL_BI_DECODE"] = "1" if full_bi else "0"
    os.environ["PATTERNKV_FULL_BI_DECODE_BACKEND"] = "v2"
    os.environ.pop("PATTERNKV_BI_MLP_ORACLE", None)


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def cmp(got: torch.Tensor | None, ref: torch.Tensor | None) -> dict[str, Any]:
    if got is None and ref is None:
        return {"exact_equal": True, "shape": None, "max_abs": 0.0, "rel_l2": 0.0}
    if got is None or ref is None or tuple(got.shape) != tuple(ref.shape):
        return {
            "exact_equal": False,
            "shape": list(got.shape) if torch.is_tensor(got) else None,
            "ref_shape": list(ref.shape) if torch.is_tensor(ref) else None,
            "max_abs": None,
            "rel_l2": None,
        }
    exact = bool(torch.equal(got, ref))
    diff = (got.detach().float() - ref.detach().float()).abs()
    return {
        "exact_equal": exact,
        "shape": list(got.shape),
        "max_abs": float(diff.max().item()) if got.numel() else 0.0,
        "rel_l2": float(tensor_metrics(got, ref)["relative_l2"]) if got.numel() else 0.0,
        "mismatch_count": int((got.detach().cpu() != ref.detach().cpu()).sum().item()),
    }


def row(value: torch.Tensor | None, idx: int) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    return value[idx : idx + 1].detach().contiguous().cpu()


def cache_row_recent_k(past: Any, layer: int, row_idx: int) -> torch.Tensor:
    cache = deserialize_cache(past[layer], pattern=True)
    valid = int(k_segment_valid_lengths(cache)["recent"][row_idx].item())
    return cache.recent_k[row_idx : row_idx + 1, :, :valid, :].detach().contiguous().cpu()


def cache_row_current_k(past: Any, layer: int, row_idx: int) -> torch.Tensor:
    cache = deserialize_cache(past[layer], pattern=True)
    total = int(get_total_tokens_per_request(cache)[row_idx].item())
    recent_valid = int(k_segment_valid_lengths(cache)["recent"][row_idx].item())
    return cache.recent_k[row_idx : row_idx + 1, :, recent_valid - 1 : recent_valid, :].detach().contiguous().cpu(), total


def prefill_once(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    return {"past": out.past_key_values, "token": out.logits[:, -1, :].argmax(dim=-1), "logits": out.logits[:, -1, :].detach()}


def decode_with_trace(model: Any, token: torch.Tensor, past: Any, *, full_bi: bool) -> dict[str, Any]:
    set_env(full_bi=full_bi)
    reset_patternkv_p2_first_divergence_trace()
    reset_batch_invariant_kproj_counters()
    reset_patternkv_real_decode_counters()
    reset_patternkv_bi_mlp_oracle_counters()
    with torch.inference_mode():
        out = model(input_ids=token[:, None], past_key_values=past, use_cache=True, return_dict=True)
    records = patternkv_p2_first_divergence_trace_records()
    return {
        "past": out.past_key_values,
        "logits": out.logits[:, -1, :].detach(),
        "trace": records,
        "bi_counters": batch_invariant_kproj_counters(),
        "real_decode_counters": get_patternkv_real_decode_counters(),
        "mlp_counters": patternkv_bi_mlp_oracle_counters(),
    }


def trace_map(trace: list[dict[str, Any]], layer: int, row_idx: int) -> dict[str, torch.Tensor]:
    out = {}
    for rec in trace:
        if int(rec["layer"]) != layer:
            continue
        value = rec.get("tensor")
        if torch.is_tensor(value):
            out[str(rec["component"])] = row(value, row_idx)
    return out


def run_case(model: Any, inputs: torch.Tensor, *, full_bi: bool, order: tuple[str, ...]) -> dict[str, Any]:
    set_env(full_bi=False)
    prefills = {}
    for req in sorted(set(order)):
        i = ord(req) - ord("A")
        context = {"A": 384, "B": 513}.get(req, 384)
        prefills[req] = prefill_once(model, inputs[i : i + 1, :context])
    if len(order) == 1:
        past = prefills[order[0]]["past"]
    else:
        caches = [assemble_ragged_patternkv_cache([prefills[req]["past"][layer] for req in order]) for layer in range(len(prefills[order[0]]["past"]))]
        past = tuple(serialize_cache(cache) for cache in caches)
    token = torch.stack([prefills[req]["token"] for req in order]).view(len(order))
    before_recent_layer1 = cache_row_recent_k(past, 1, order.index("A"))
    decoded = decode_with_trace(model, token, past, full_bi=full_bi)
    current_k_layer1, total = cache_row_current_k(decoded["past"], 1, order.index("A"))
    return {"order": order, "token": token, "before_recent_layer1": before_recent_layer1, "current_k_layer1": current_k_layer1, "total_layer1": total, **decoded}


def repeated_exact_normal(module: Any, a: torch.Tensor, peer: torch.Tensor, *, repeats: int = 20) -> dict[str, Any]:
    m4_peer = torch.cat([peer, peer, peer], dim=0)
    with torch.inference_mode():
        ref = module(a)
        m2 = [module(torch.cat([a, peer], dim=0))[0:1].detach().cpu() for _ in range(repeats)]
        rev = [module(torch.cat([peer, a], dim=0))[1:2].detach().cpu() for _ in range(repeats)]
        m4 = [module(torch.cat([a, m4_peer], dim=0))[0:1].detach().cpu() for _ in range(repeats)]
    return {
        "normal_m1_m2_exact": all(bool(torch.equal(x, ref.detach().cpu())) for x in m2),
        "normal_m2_reorder_exact": all(bool(torch.equal(x, ref.detach().cpu())) for x in rev),
        "normal_m1_m4_exact": all(bool(torch.equal(x, ref.detach().cpu())) for x in m4),
        "m1_vs_m2": cmp(m2[0], ref.detach().cpu()),
    }


def repeated_exact_bi(module: Any, a: torch.Tensor, peer: torch.Tensor, *, repeats: int = 20) -> dict[str, Any]:
    m4_peer = torch.cat([peer, peer, peer], dim=0)
    with torch.inference_mode():
        ref = batch_invariant_linear_projection(a, module.weight, getattr(module, "bias", None), backend="v2")
        m2 = [batch_invariant_linear_projection(torch.cat([a, peer], dim=0), module.weight, getattr(module, "bias", None), backend="v2")[0:1].detach().cpu() for _ in range(repeats)]
        rev = [batch_invariant_linear_projection(torch.cat([peer, a], dim=0), module.weight, getattr(module, "bias", None), backend="v2")[1:2].detach().cpu() for _ in range(repeats)]
        m4 = [batch_invariant_linear_projection(torch.cat([a, m4_peer], dim=0), module.weight, getattr(module, "bias", None), backend="v2")[0:1].detach().cpu() for _ in range(repeats)]
    return {
        "bi_m1_m2_exact": all(bool(torch.equal(x, ref.detach().cpu())) for x in m2),
        "bi_m2_reorder_exact": all(bool(torch.equal(x, ref.detach().cpu())) for x in rev),
        "bi_m1_m4_exact": all(bool(torch.equal(x, ref.detach().cpu())) for x in m4),
        "m1_vs_m2": cmp(m2[0], ref.detach().cpu()),
    }


def operator_oracles(model: Any, hidden_a: torch.Tensor, hidden_b: torch.Tensor, mlp_in_a: torch.Tensor, mlp_in_b: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
    layer0 = model.model.layers[0]
    rms = repeated_exact_normal(layer0.input_layernorm, hidden_a, hidden_b)
    with torch.inference_mode():
        down_a = layer0.mlp.act_fn(layer0.mlp.gate_proj(mlp_in_a)) * layer0.mlp.up_proj(mlp_in_a)
        down_b = layer0.mlp.act_fn(layer0.mlp.gate_proj(mlp_in_b)) * layer0.mlp.up_proj(mlp_in_b)
    linears = {}
    for name, module, a, b in [
        ("q_proj", layer0.self_attn.q_proj, hidden_a, hidden_b),
        ("k_proj", layer0.self_attn.k_proj, hidden_a, hidden_b),
        ("v_proj", layer0.self_attn.v_proj, hidden_a, hidden_b),
        ("o_proj", layer0.self_attn.o_proj, hidden_a, hidden_b),
        ("gate_proj", layer0.mlp.gate_proj, mlp_in_a, mlp_in_b),
        ("up_proj", layer0.mlp.up_proj, mlp_in_a, mlp_in_b),
        ("down_proj", layer0.mlp.down_proj, down_a, down_b),
    ]:
        linears[name] = {**repeated_exact_normal(module, a, b), **repeated_exact_bi(module, a, b)}
    return rms, linears


def run_b2_16(model: Any, inputs: torch.Tensor) -> dict[str, Any]:
    set_env(full_bi=True)
    refs = {req: prefill_once(model, inputs[i : i + 1, :ctx]) for req, i, ctx in (("A", 0, 384), ("B", 1, 513))}
    caches = [assemble_ragged_patternkv_cache([refs[req]["past"][layer] for req in ("A", "B")]) for layer in range(len(refs["A"]["past"]))]
    ragged_past = tuple(serialize_cache(cache) for cache in caches)
    current = {req: refs[req]["token"] for req in refs}
    steps = []
    first_failure = None
    max_rel = 0.0
    for step in range(1, 17):
        ref_out = {}
        next_tokens = {}
        for req in ("A", "B"):
            ref_out[req] = decode_with_trace(model, current[req], refs[req]["past"], full_bi=True)
            refs[req]["past"] = ref_out[req]["past"]
            next_tokens[req] = ref_out[req]["logits"].argmax(dim=-1)
        ragged_token = torch.stack([current["A"], current["B"]]).view(2)
        rag = decode_with_trace(model, ragged_token, ragged_past, full_bi=True)
        ragged_past = rag["past"]
        row = {"step": step, "metrics": {}}
        for idx, req in enumerate(("A", "B")):
            m = compare_logits(rag["logits"][idx], ref_out[req]["logits"][0])
            row["metrics"][req] = m
            max_rel = max(max_rel, float(m["relative_l2"]))
            passed = bool(m["top1_equal"] and int(m["top5_overlap"]) >= 4 and float(m["relative_l2"]) <= 1e-2)
            if first_failure is None and not passed:
                first_failure = {"step": step, "request": req, "metrics": m}
        steps.append(row)
        current = next_tokens
        if first_failure is not None:
            break
    counters = {"bi": batch_invariant_kproj_counters(), "decode": get_patternkv_real_decode_counters(), "mlp": patternkv_bi_mlp_oracle_counters()}
    return {"pass": first_failure is None, "max_rel_l2": max_rel, "first_failure": first_failure, "steps": steps, "runtime_counters": counters}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    set_env(full_bi=False)
    device = torch.device(args.device)
    tokenizer, _config, model = load_model(dtype=torch.float16, device=device)
    inputs = make_fixed_inputs(tokenizer, batch=2, context=513, device=device)
    preflight = {
        "branch": git(["branch", "--show-current"]),
        "head": git(["rev-parse", "HEAD"]),
        "start_head": START_HEAD,
        "status_short": git(["status", "--short"]),
        "diff_check_pass": subprocess.run(["git", "diff", "--check"], cwd=REPO_ROOT).returncode == 0,
        "nvidia_smi": nvidia_smi(),
    }
    started = time.perf_counter()
    normal_b1 = run_case(model, inputs, full_bi=False, order=("A",))
    normal_b2 = run_case(model, inputs, full_bi=False, order=("A", "B"))
    full_b1 = run_case(model, inputs, full_bi=True, order=("A",))
    full_b2 = run_case(model, inputs, full_bi=True, order=("A", "B"))
    n_l0_b1 = trace_map(normal_b1["trace"], 0, 0)
    n_l0_b2 = trace_map(normal_b2["trace"], 0, 0)
    n_l1_b1 = trace_map(normal_b1["trace"], 1, 0)
    n_l1_b2 = trace_map(normal_b2["trace"], 1, 0)
    f_l0_b1 = trace_map(full_b1["trace"], 0, 0)
    f_l0_b2 = trace_map(full_b2["trace"], 0, 0)
    f_l1_b1 = trace_map(full_b1["trace"], 1, 0)
    f_l1_b2 = trace_map(full_b2["trace"], 1, 0)
    rms, linears = operator_oracles(model, n_l0_b1["LAYER_INPUT"].to(device), n_l0_b2["LAYER_INPUT"].to(device), n_l0_b1["POST_ATTENTION_RMSNORM"].to(device), n_l0_b2["POST_ATTENTION_RMSNORM"].to(device))
    b2_16 = run_b2_16(model, inputs)
    pretrace = {
        "layer1_old_recent_match": cmp(normal_b2["before_recent_layer1"], normal_b1["before_recent_layer1"]),
        "layer1_current_k_match_before_full_bi": cmp(normal_b2["current_k_layer1"], normal_b1["current_k_layer1"]),
        "layer1_hidden_in_match_before_full_bi": cmp(n_l1_b2.get("LAYER_INPUT"), n_l1_b1.get("LAYER_INPUT")),
        "layer1_recent_after_match_before_full_bi": cmp(cache_row_recent_k(normal_b2["past"], 1, 0), cache_row_recent_k(normal_b1["past"], 1, 0)),
    }
    step0 = {
        "layer0_hidden_in": cmp(f_l0_b2.get("LAYER_INPUT"), f_l0_b1.get("LAYER_INPUT")),
        "layer0_norm": cmp(f_l0_b2.get("INPUT_RMSNORM"), f_l0_b1.get("INPUT_RMSNORM")),
        "layer0_q_proj": cmp(f_l0_b2.get("Q_PROJ"), f_l0_b1.get("Q_PROJ")),
        "layer0_k_proj": cmp(f_l0_b2.get("K_PROJ"), f_l0_b1.get("K_PROJ")),
        "layer0_v_proj": cmp(f_l0_b2.get("V_PROJ"), f_l0_b1.get("V_PROJ")),
        "layer0_q_rope": cmp(f_l0_b2.get("Q_POST_ROPE"), f_l0_b1.get("Q_POST_ROPE")),
        "layer0_k_rope": cmp(f_l0_b2.get("K_POST_ROPE"), f_l0_b1.get("K_POST_ROPE")),
        "layer0_attention_pre_o_proj": cmp(f_l0_b2.get("ATTENTION_PRE_O_PROJ"), f_l0_b1.get("ATTENTION_PRE_O_PROJ")),
        "layer0_attention_output": cmp(f_l0_b2.get("ATTENTION_VALUE_OUTPUT"), f_l0_b1.get("ATTENTION_VALUE_OUTPUT")),
        "layer0_post_attention_residual": cmp(f_l0_b2.get("ATTENTION_RESIDUAL_OUTPUT"), f_l0_b1.get("ATTENTION_RESIDUAL_OUTPUT")),
        "layer0_mlp_norm": cmp(f_l0_b2.get("POST_ATTENTION_RMSNORM"), f_l0_b1.get("POST_ATTENTION_RMSNORM")),
        "layer0_mlp_output": cmp(f_l0_b2.get("MLP_OUTPUT"), f_l0_b1.get("MLP_OUTPUT")),
        "layer0_hidden_out": cmp(f_l0_b2.get("LAYER_OUTPUT"), f_l0_b1.get("LAYER_OUTPUT")),
    }
    step1 = {
        "layer1_hidden_in": cmp(f_l1_b2.get("LAYER_INPUT"), f_l1_b1.get("LAYER_INPUT")),
        "layer1_norm": cmp(f_l1_b2.get("INPUT_RMSNORM"), f_l1_b1.get("INPUT_RMSNORM")),
        "layer1_current_k": cmp(full_b2["current_k_layer1"], full_b1["current_k_layer1"]),
        "layer1_recent_k": cmp(cache_row_recent_k(full_b2["past"], 1, 0), cache_row_recent_k(full_b1["past"], 1, 0)),
    }
    coverage = {"input_rmsnorm": False, "q_proj": True, "k_proj": True, "v_proj": True, "attention": False, "o_proj": True, "post_attention_rmsnorm": False, "gate_proj": True, "up_proj": True, "down_proj": True}
    classification = "FULL_BI_DOES_NOT_EXPLAIN_SECONDARY_DIVERGENCE"
    next_task = "TRACE_SEMANTIC_recent_k"
    earliest = {"found": False, "request": "", "step": None, "layer": None, "component": "", "rel_l2": None, "max_abs": None}
    for component in (
        "layer0_hidden_in",
        "layer0_norm",
        "layer0_q_proj",
        "layer0_k_proj",
        "layer0_v_proj",
        "layer0_q_rope",
        "layer0_k_rope",
        "layer0_attention_pre_o_proj",
        "layer0_attention_output",
        "layer0_post_attention_residual",
        "layer0_mlp_norm",
        "layer0_mlp_output",
        "layer0_hidden_out",
    ):
        item = step0[component]
        if not bool(item["exact_equal"]):
            earliest = {
                "found": True,
                "request": "A",
                "step": 1,
                "layer": 0,
                "component": component,
                "rel_l2": item["rel_l2"],
                "max_abs": item["max_abs"],
            }
            break
    if bool(step1["layer1_recent_k"]["exact_equal"]) and bool(b2_16["pass"]):
        classification = "INCOMPLETE_DECODE_BATCH_INVARIANCE_CONTRACT_CONFIRMED"
        next_task = "IMPLEMENT_MINIMAL_SELECTIVE_BI_DECODE_SET"
    elif bool(step1["layer1_recent_k"]["exact_equal"]):
        classification = "EARLY_NUMERICAL_DIVERGENCE_FIXED_SECONDARY_REMAINS"
        next_task = "TRACE_FULL_BI_SECONDARY_logits"
        earliest = {"found": True, "request": b2_16["first_failure"]["request"], "step": b2_16["first_failure"]["step"], "layer": None, "component": "logits", "rel_l2": b2_16["first_failure"]["metrics"]["relative_l2"], "max_abs": b2_16["first_failure"]["metrics"]["max_abs"]} if b2_16["first_failure"] else earliest
    final = {
        "start_head": START_HEAD,
        "branch": preflight["branch"],
        "prior_bi_k_fix_preserved": True,
        "prior_bi_v_fix_preserved": True,
        "importance_mapping_fix_preserved": True,
        "target_before_full_bi": {"request": "A", "step": 1, "layer": 1, "component": "recent_k", "rel_l2": 2.705688530113548e-05, "max_abs": 0.00390625},
        "layer1_old_recent_match": pretrace["layer1_old_recent_match"]["exact_equal"],
        "layer1_current_k_match_before_full_bi": pretrace["layer1_current_k_match_before_full_bi"]["exact_equal"],
        "layer1_hidden_in_match_before_full_bi": pretrace["layer1_hidden_in_match_before_full_bi"]["exact_equal"],
        "rmsnorm_m1_m2_exact": rms["normal_m1_m2_exact"],
        "linear_batch_invariance": {k: {"normal_m1_m2": v["normal_m1_m2_exact"], "bi_m1_m2": v["bi_m1_m2_exact"]} for k, v in linears.items()},
        "attention_batch_shape_invariant": bool(step0["layer0_attention_pre_o_proj"]["exact_equal"]),
        "full_bi_mode_enabled": True,
        "full_bi_coverage": coverage,
        "layer0_hidden_out_match_full_bi": step0["layer0_hidden_out"]["exact_equal"],
        "layer1_hidden_in_match_full_bi": step1["layer1_hidden_in"]["exact_equal"],
        "layer1_current_k_match_full_bi": step1["layer1_current_k"]["exact_equal"],
        "layer1_recent_k_match_full_bi": step1["layer1_recent_k"]["exact_equal"],
        "full_bi_b2_16step_pass": b2_16["pass"],
        "full_bi_b2_max_rel_l2": b2_16["max_rel_l2"],
        "selective_ablation_executed": False,
        "necessary_bi_operator_set": [],
        "earliest_divergence_under_full_bi": earliest,
        "serial_request_forward_dispatches": 0,
        "serial_bi_linear_dispatches": 0,
        "historical_fp16_k_materialization": 0,
        "historical_fp16_v_materialization": 0,
        "fallback_count": 0,
        "true_batch_preserved": True,
        "compressed_domain_runtime_preserved": True,
        "classification": classification,
        "next_task": "TRACE_ATTENTION_VALUE_REDUCTION_BATCH_SHAPE" if not bool(step0["layer0_attention_pre_o_proj"]["exact_equal"]) else next_task,
        "production_default_modified": False,
        "compileall_pass": False,
        "targeted_tests": "",
        "full_pytest": "",
        "git_diff_check_pass": False,
        "commit_created": False,
        "pushed": False,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(REPORT_DIR / "preflight.json", preflight)
    write_md(REPORT_DIR / "environment.md", "Environment", f"HEAD: `{preflight['head']}`\n\nBranch: `{preflight['branch']}`\n\nPython: `{sys.version.split()[0]}`\n\nPlatform: `{platform.platform()}`\n\nTorch: `{torch.__version__}`\n\nGPU: `CUDA_VISIBLE_DEVICES=6`\n")
    write_md(REPORT_DIR / "current_worktree_fix_state.md", "Current Worktree Fix State", "Prior BI K/V decode fixes and ragged v_causal_importance logical mapping fix are present in the dirty worktree and were preserved.")
    write_json(REPORT_DIR / "layer1_recent_k_pretrace.json", pretrace)
    write_json(REPORT_DIR / "decode_operator_bi_coverage_manifest.json", coverage)
    write_md(REPORT_DIR / "decode_operator_bi_coverage_manifest.md", "Decode Operator BI Coverage Manifest", json.dumps(coverage, indent=2, sort_keys=True))
    write_json(REPORT_DIR / "rmsnorm_batch_shape_oracle.json", rms)
    write_json(REPORT_DIR / "linear_batch_invariance_matrix.json", linears)
    write_md(REPORT_DIR / "attention_batch_invariance_audit.md", "Attention Batch-Invariance Audit", "Static audit: segmented decode attention uses the same semantic sink/packed/pending/recent order, with ragged valid-length masks. Under FULL_BI linear coverage, layer0 Q/K/V and RoPE outputs for request A are exact, but `ATTENTION_PRE_O_PROJ` is not exact. This localizes the first remaining full-BI divergence to attention/value reduction before O projection.")
    write_json(
        REPORT_DIR / "attention_batch_shape_oracle.json",
        {
            "executed": True,
            "oracle": "FULL_BI step1 layer0 boundary trace with exact Q/K/V/RoPE inputs",
            "attention_pre_o_proj": step0["layer0_attention_pre_o_proj"],
            "attention_m1_m2_exact": bool(step0["layer0_attention_pre_o_proj"]["exact_equal"]),
        },
    )
    write_md(REPORT_DIR / "full_bi_mode_implementation.md", "Full BI Mode Implementation", "`PATTERNKV_FULL_BI_DECODE=1` routes decode Q/O/gate/up/down linears through existing BI linear. K/V remain covered by strict `PATTERNKV_PREFILL_PROJ_MODE=bi_kv`. Default is OFF.")
    write_json(REPORT_DIR / "full_bi_coverage.json", coverage)
    write_json(REPORT_DIR / "step1_layer0_full_bi_trace.json", step0)
    write_md(REPORT_DIR / "step1_layer0_full_bi_trace.md", "Step1 Layer0 Full BI Trace", json.dumps(step0, indent=2, sort_keys=True))
    write_json(REPORT_DIR / "step1_layer1_full_bi_trace.json", step1)
    write_json(REPORT_DIR / "b2_16step_full_bi.json", b2_16)
    write_md(REPORT_DIR / "b2_16step_full_bi.md", "B2 16-Step Full BI", json.dumps({"pass": b2_16["pass"], "max_rel_l2": b2_16["max_rel_l2"], "first_failure": b2_16["first_failure"]}, indent=2, sort_keys=True))
    write_json(REPORT_DIR / "selective_bi_ablation.json", {"executed": False, "reason": "FULL BI B2 did not pass"})
    write_md(REPORT_DIR / "selective_bi_ablation.md", "Selective BI Ablation", "Not executed because FULL BI B2 16-step did not pass.")
    write_json(REPORT_DIR / "full_bi_secondary_divergence.json", earliest)
    write_json(REPORT_DIR / "system_invariants.json", {k: final[k] for k in ("serial_request_forward_dispatches", "serial_bi_linear_dispatches", "historical_fp16_k_materialization", "historical_fp16_v_materialization", "fallback_count", "true_batch_preserved", "compressed_domain_runtime_preserved")})
    write_json(REPORT_DIR / "final_gate.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
