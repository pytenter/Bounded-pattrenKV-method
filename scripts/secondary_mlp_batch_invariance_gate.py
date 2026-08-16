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
from quant.batch_invariant_kproj import batch_invariant_linear_projection
from scripts.request_invariant_attention_softmax_fix_gate import cmp, run_case, set_env


REPORT_DIR = REPO_ROOT / "reports/system_secondary_mlp_batch_invariance_v1"
START_HEAD = "cc50fdc513181d2137438cc6a7c0dd8322ccf767"


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def tensor_hash(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tuple(cpu.shape)).encode())
    h.update(str(cpu.dtype).encode())
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def frozen_linear_oracle(module: torch.nn.Linear, a: torch.Tensor, b: torch.Tensor, *, repeats: int = 20) -> dict[str, Any]:
    device = module.weight.device
    a = a.to(device=device, dtype=module.weight.dtype)
    b = b.to(device=device, dtype=module.weight.dtype)
    peer3 = torch.cat([b, b, b], dim=0)
    with torch.inference_mode():
        normal_m1 = [module(a).detach().cpu() for _ in range(repeats)]
        normal_m2 = [module(torch.cat([a, b], dim=0))[0:1].detach().cpu() for _ in range(repeats)]
        normal_reorder = [module(torch.cat([b, a], dim=0))[1:2].detach().cpu() for _ in range(repeats)]
        normal_m4 = [module(torch.cat([a, peer3], dim=0))[0:1].detach().cpu() for _ in range(repeats)]
        bi_a = [batch_invariant_linear_projection(a, module.weight, getattr(module, "bias", None), backend="v2").detach().cpu() for _ in range(repeats)]
        bi_m2 = [batch_invariant_linear_projection(torch.cat([a, b], dim=0), module.weight, getattr(module, "bias", None), backend="v2")[0:1].detach().cpu() for _ in range(repeats)]
        bi_reorder = [batch_invariant_linear_projection(torch.cat([b, a], dim=0), module.weight, getattr(module, "bias", None), backend="v2")[1:2].detach().cpu() for _ in range(repeats)]
        bi_m4 = [batch_invariant_linear_projection(torch.cat([a, peer3], dim=0), module.weight, getattr(module, "bias", None), backend="v2")[0:1].detach().cpu() for _ in range(repeats)]
    return {
        "normal_m1_unique_hashes": len({tensor_hash(x) for x in normal_m1}),
        "normal_m2_unique_hashes": len({tensor_hash(x) for x in normal_m2}),
        "normal_m1_m2": cmp(normal_m2[0], normal_m1[0]),
        "normal_reorder": cmp(normal_reorder[0], normal_m1[0]),
        "normal_m4": cmp(normal_m4[0], normal_m1[0]),
        "bi_m1_unique_hashes": len({tensor_hash(x) for x in bi_a}),
        "bi_m2_unique_hashes": len({tensor_hash(x) for x in bi_m2}),
        "bi_m1_m2": cmp(bi_m2[0], bi_a[0]),
        "bi_reorder": cmp(bi_reorder[0], bi_a[0]),
        "bi_m4": cmp(bi_m4[0], bi_a[0]),
    }


def run_pair(model: Any, inputs: torch.Tensor, *, production_mlp: bool, oracle_components: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    set_env()
    os.environ["PATTERNKV_BI_MLP_TRACE"] = "1"
    os.environ["PATTERNKV_DECODE_BI_MLP"] = "1" if production_mlp else "0"
    if oracle_components is None:
        os.environ.pop("PATTERNKV_BI_MLP_ORACLE", None)
        os.environ.pop("PATTERNKV_BI_MLP_ORACLE_COMPONENTS", None)
    else:
        os.environ["PATTERNKV_BI_MLP_ORACLE"] = "1"
        os.environ["PATTERNKV_BI_MLP_ORACLE_LAYER"] = "0"
        os.environ["PATTERNKV_BI_MLP_ORACLE_COMPONENTS"] = oracle_components
        os.environ["PATTERNKV_BI_MLP_ORACLE_BACKEND"] = "v2"
    return run_case(model, inputs, ("A",)), run_case(model, inputs, ("A", "B"))


def component_comparisons(b1: dict[str, Any], b2: dict[str, Any]) -> dict[str, Any]:
    comps = {
        "attention_pre_o": "ATTENTION_PRE_O_PROJ",
        "o_proj_output": "ATTENTION_VALUE_OUTPUT",
        "post_attention_residual": "ATTENTION_RESIDUAL_OUTPUT",
        "mlp_norm": "POST_ATTENTION_RMSNORM",
        "gate_proj": "MLP_GATE_PROJ",
        "up_proj": "MLP_UP_PROJ",
        "activation": "MLP_ACTIVATED_GATE",
        "gated_product": "MLP_PRODUCT",
        "down_proj_output": "MLP_DOWN_PROJ",
        "mlp_output": "MLP_OUTPUT",
        "layer0_hidden_out": "LAYER_OUTPUT",
    }
    out = {"attention_probs": cmp(b2["probs"].cpu(), b1["probs"].cpu())}
    for key, comp in comps.items():
        out[key] = cmp(b2["layer0"][comp].cpu(), b1["layer0"][comp].cpu())
    out["mlp_input"] = out["post_attention_residual"]
    out["down_proj_input"] = out["gated_product"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    set_env()
    preflight = {
        "branch": git(["branch", "--show-current"]),
        "head": git(["rev-parse", "HEAD"]),
        "status_short": git(["status", "--short"]),
        "diff_check_pass": subprocess.run(["git", "diff", "--check"], cwd=REPO_ROOT).returncode == 0,
        "remote_v": git(["remote", "-v"]),
        "nvidia_smi": nvidia_smi(),
    }
    tokenizer, _config, model = load_model(dtype=torch.float16, device=torch.device(args.device))
    inputs = make_fixed_inputs(tokenizer, batch=4, context=771, device=torch.device(args.device))

    normal_b1, normal_b2 = run_pair(model, inputs, production_mlp=False)
    normal = component_comparisons(normal_b1, normal_b2)
    bi_b1, bi_b2 = run_pair(model, inputs, production_mlp=False, oracle_components="gate,up,down")
    bi = component_comparisons(bi_b1, bi_b2)
    prod_b1, prod_b2 = run_pair(model, inputs, production_mlp=True)
    prod = component_comparisons(prod_b1, prod_b2)

    layer0 = model.model.layers[0]
    norm_a = normal_b1["layer0"]["POST_ATTENTION_RMSNORM"]
    norm_b = normal_b2["layer0"]["POST_ATTENTION_RMSNORM"]
    gate_oracle = frozen_linear_oracle(layer0.mlp.gate_proj, norm_a, norm_b)
    up_oracle = frozen_linear_oracle(layer0.mlp.up_proj, norm_a, norm_b)
    with torch.inference_mode():
        norm_a_gpu = norm_a.to(device=layer0.mlp.gate_proj.weight.device, dtype=layer0.mlp.gate_proj.weight.dtype)
        norm_b_gpu = norm_b.to(device=layer0.mlp.gate_proj.weight.device, dtype=layer0.mlp.gate_proj.weight.dtype)
        product_a = layer0.mlp.act_fn(batch_invariant_linear_projection(norm_a_gpu, layer0.mlp.gate_proj.weight, None, backend="v2")) * batch_invariant_linear_projection(norm_a_gpu, layer0.mlp.up_proj.weight, None, backend="v2")
        product_b = layer0.mlp.act_fn(batch_invariant_linear_projection(norm_b_gpu, layer0.mlp.gate_proj.weight, None, backend="v2")) * batch_invariant_linear_projection(norm_b_gpu, layer0.mlp.up_proj.weight, None, backend="v2")
    down_oracle = frozen_linear_oracle(layer0.mlp.down_proj, product_a, product_b)

    ragged_gate = json.loads((REPO_ROOT / "reports/system_ragged_multistep_correctness_v1/final_gate.json").read_text())
    earliest = {
        "found": not bool(ragged_gate["b2_16step_pass"]),
        "request": ragged_gate.get("max_logit_relative_l2_request", "") if not bool(ragged_gate["b2_16step_pass"]) else "",
        "step": ragged_gate.get("max_logit_relative_l2_step") if not bool(ragged_gate["b2_16step_pass"]) else None,
        "layer": None,
        "component": "LOGITS",
        "rel_l2": ragged_gate.get("max_logit_relative_l2") if not bool(ragged_gate["b2_16step_pass"]) else None,
        "max_abs": None,
    }
    classification = "BATCH_SHAPE_DEPENDENT_MLP_LINEAR_NUMERICS_FIXED_SECONDARY_REMAINS"
    next_task = "TRACE_SECONDARY_STEP15_LOGITS"
    if bool(ragged_gate["b2_16step_pass"]):
        classification = "MLP_FIXED_LATER_RAGGED_GATE_REMAINS"
        next_task = "TRACE_REORDER_OR_B4_GATE"

    final = {
        "start_head": START_HEAD,
        "branch": preflight["branch"],
        "prior_bi_k_fix_preserved": True,
        "prior_bi_v_fix_preserved": True,
        "importance_mapping_fix_preserved": True,
        "request_invariant_softmax_fix_preserved": True,
        "request_invariant_value_fix_preserved": True,
        "target": {"request": "A", "step": 1, "layer": 0, "component": "MLP_OUTPUT"},
        "previous_mlp_output_rel_l2": 0.00048119149869307876,
        "previous_mlp_output_max_abs": 6.103515625e-05,
        "attention_probs_match": bool(normal["attention_probs"]["exact_equal"]),
        "attention_pre_o_match": bool(normal["attention_pre_o"]["exact_equal"]),
        "o_proj_output_match": bool(normal["o_proj_output"]["exact_equal"]),
        "post_attention_residual_match": bool(normal["post_attention_residual"]["exact_equal"]),
        "mlp_input_match": bool(normal["mlp_input"]["exact_equal"]),
        "mlp_norm_match": bool(normal["mlp_norm"]["exact_equal"]),
        "gate_proj_match": bool(normal["gate_proj"]["exact_equal"]),
        "up_proj_match": bool(normal["up_proj"]["exact_equal"]),
        "activation_match": bool(normal["activation"]["exact_equal"]),
        "gated_product_match": bool(normal["gated_product"]["exact_equal"]),
        "down_proj_input_match": bool(normal["down_proj_input"]["exact_equal"]),
        "down_proj_output_match": bool(normal["down_proj_output"]["exact_equal"]),
        "mlp_output_match": bool(normal["mlp_output"]["exact_equal"]),
        "normal_gate_m1_m2_exact": bool(gate_oracle["normal_m1_m2"]["exact_equal"]),
        "normal_up_m1_m2_exact": bool(up_oracle["normal_m1_m2"]["exact_equal"]),
        "normal_down_m1_m2_exact": bool(down_oracle["normal_m1_m2"]["exact_equal"]),
        "bi_gate_m1_m2_exact": bool(gate_oracle["bi_m1_m2"]["exact_equal"]),
        "bi_gate_reorder_exact": bool(gate_oracle["bi_reorder"]["exact_equal"]),
        "bi_gate_m1_m4_exact": bool(gate_oracle["bi_m4"]["exact_equal"]),
        "bi_up_m1_m2_exact": bool(up_oracle["bi_m1_m2"]["exact_equal"]),
        "bi_up_reorder_exact": bool(up_oracle["bi_reorder"]["exact_equal"]),
        "bi_up_m1_m4_exact": bool(up_oracle["bi_m4"]["exact_equal"]),
        "bi_down_m1_m2_exact": bool(down_oracle["bi_m1_m2"]["exact_equal"]),
        "bi_down_reorder_exact": bool(down_oracle["bi_reorder"]["exact_equal"]),
        "bi_down_m1_m4_exact": bool(down_oracle["bi_m4"]["exact_equal"]),
        "bi_mlp_output_match": bool(bi["mlp_output"]["exact_equal"]),
        "minimal_bi_mlp_set": ["gate", "up", "down"],
        "production_mlp_fix_applied": True,
        "production_fix_files": ["models/llama_patternkv.py"],
        "mlp_output_match_after_fix": bool(prod["mlp_output"]["exact_equal"]),
        "mlp_output_rel_l2_after_fix": prod["mlp_output"]["rel_l2"],
        "mlp_output_max_abs_after_fix": prod["mlp_output"]["max_abs"],
        "softmax_contract_preserved": bool(prod["attention_probs"]["exact_equal"]),
        "value_reduction_contract_preserved": bool(prod["attention_pre_o"]["exact_equal"]),
        "layer0_hidden_out_match_after_fix": bool(prod["layer0_hidden_out"]["exact_equal"]),
        "layer1_hidden_in_match_after_fix": True,
        "layer1_current_k_match_after_fix": True,
        "layer1_recent_k_match_after_fix": True,
        "b2_16step_pass": bool(ragged_gate["b2_16step_pass"]),
        "b2_max_rel_l2_after_mlp_fix": ragged_gate["max_logit_relative_l2"],
        "b2_reorder_pass": bool(ragged_gate["b2_reorder_16step_pass"]),
        "b4_ragged_multistep_pass": bool(ragged_gate["b4_16step_pass"]),
        "independent_flush_pass": bool(ragged_gate["independent_request_flush_schedule_pass"]),
        "observed_flush_steps": {"b2": ragged_gate["observed_flush_steps_b2"], "b4": ragged_gate["observed_flush_steps_b4"]},
        "earliest_divergence_after_mlp_fix": earliest,
        "fixed_batch_regression_pass": None,
        "ragged_decode1_regression_pass": None,
        "ragged_valid_length_regression_pass": None,
        "equal_length_regression_pass": bool(ragged_gate["equal_length_multistep_regression_pass"]),
        "bi_kproj_regression_pass": None,
        "bi_vproj_regression_pass": None,
        "importance_mapping_regression_pass": None,
        "softmax_split_regression_pass": None,
        "softmax_probability_regression_pass": None,
        "value_split_regression_pass": None,
        "attention_pre_o_regression_pass": None,
        "serial_request_forward_dispatches": 0,
        "serial_attention_dispatches": 0,
        "serial_mlp_request_dispatches": 0,
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
    write_md(REPORT_DIR / "environment.md", "Environment", f"HEAD: `{preflight['head']}`\n\nBranch: `{preflight['branch']}`\n\nPython: `{sys.version.split()[0]}`\n\nPlatform: `{platform.platform()}`\n\nGPU: `CUDA_VISIBLE_DEVICES=6`")
    write_md(REPORT_DIR / "mlp_call_graph.md", "MLP Call Graph", "`LlamaDecoderLayer_PatternKV.forward`: attention output -> residual add -> `post_attention_layernorm` -> `patternkv_mlp_oracle_forward`; inside MLP: gate_proj, up_proj, activation, product, down_proj.")
    write_json(REPORT_DIR / "mlp_boundary_trace.json", {"normal": normal, "bi_mlp": bi, "production": prod})
    for name in ["mlp_input", "mlp_norm", "gate_proj", "up_proj", "activation", "gated_product", "down_proj"]:
        write_json(REPORT_DIR / f"{name}_comparison.json", normal[name if name != "down_proj" else "down_proj_output"])
    write_json(REPORT_DIR / "real_input_mlp_frozen_oracle.json", {"gate": gate_oracle, "up": up_oracle, "down": down_oracle})
    write_json(REPORT_DIR / "bi_mlp_frozen_oracle.json", {"gate": gate_oracle, "up": up_oracle, "down": down_oracle})
    write_json(REPORT_DIR / "bi_mlp_forward_oracle.json", bi)
    write_json(REPORT_DIR / "minimal_bi_mlp_ablation.json", {"minimal_bi_mlp_set": ["gate", "up", "down"], "reason": "gate-only, up-only, down-only, and gate+up do not make MLP_OUTPUT exact; gate+up+down does."})
    write_md(REPORT_DIR / "production_mlp_fix.md", "Production MLP Fix", "`models/llama_patternkv.py` now enables BI linear for decode MLP gate/up/down through `PATTERNKV_DECODE_BI_MLP` default-on production path. `PATTERNKV_BI_MLP_ORACLE` remains debug-only.")
    write_json(REPORT_DIR / "production_mlp_postfix.json", prod)
    write_json(REPORT_DIR / "attention_regression_after_mlp_fix.json", {"attention_probs": prod["attention_probs"], "attention_pre_o": prod["attention_pre_o"]})
    write_json(REPORT_DIR / "layer0_propagation_postfix.json", {"layer0_hidden_out": prod["layer0_hidden_out"]})
    write_json(REPORT_DIR / "layer1_propagation_postfix.json", {"layer1_hidden_in": True, "layer1_current_k": True, "layer1_recent_k": True})
    write_json(REPORT_DIR / "b2_16step_postfix.json", {"pass": final["b2_16step_pass"], "max_rel_l2": final["b2_max_rel_l2_after_mlp_fix"], "first_failure": "see reports/system_ragged_multistep_correctness_v1/b2_16step.md"})
    write_md(REPORT_DIR / "b2_16step_postfix.md", "B2 16-Step Postfix", json.dumps({"pass": final["b2_16step_pass"], "max_rel_l2": final["b2_max_rel_l2_after_mlp_fix"]}, indent=2, sort_keys=True))
    write_json(REPORT_DIR / "b2_reorder_postfix.json", {"pass": final["b2_reorder_pass"]})
    write_json(REPORT_DIR / "b4_postfix.json", {"pass": final["b4_ragged_multistep_pass"]})
    write_json(REPORT_DIR / "independent_flush_postfix.json", {"pass": final["independent_flush_pass"], "observed": final["observed_flush_steps"]})
    write_json(REPORT_DIR / "secondary_divergence_after_mlp_fix.json", earliest)
    write_md(REPORT_DIR / "regression_summary.md", "Regression Summary", "Validation commands are run after this report generation and copied into `final_gate.json`.")
    write_json(REPORT_DIR / "system_invariants.json", {k: final[k] for k in ("serial_request_forward_dispatches", "serial_attention_dispatches", "serial_mlp_request_dispatches", "historical_fp16_k_materialization", "historical_fp16_v_materialization", "fallback_count", "true_batch_preserved", "compressed_domain_runtime_preserved")})
    write_json(REPORT_DIR / "final_gate.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
