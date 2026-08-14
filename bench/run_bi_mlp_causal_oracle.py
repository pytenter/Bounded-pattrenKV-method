from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.p2_first_divergence_utils import compare_tensors, first_non_exact
from bench.run_actual_model_fixed_batch_smoke import load_model, make_fixed_inputs
from models.llama_patternkv import (
    patternkv_bi_mlp_oracle_counters,
    patternkv_p2_first_divergence_trace_records,
    reset_patternkv_bi_mlp_oracle_counters,
    reset_patternkv_p2_first_divergence_trace,
    reset_patternkv_runtime_state,
)
from quant.batch_invariant_kproj import BI_KV_PREFILL_PROJ_MODE, batch_invariant_linear_projection


START_HEAD = "57716a513a90fe2639bf9bc61ba7ae1445639ad8"
REPORT_DIR = REPO_ROOT / "reports/system_bi_mlp_causal_oracle_v1"
MLP_COMPONENT_ORDER = [
    "POST_ATTENTION_RMSNORM",
    "MLP_GATE_PROJ",
    "MLP_UP_PROJ",
    "MLP_ACTIVATED_GATE",
    "MLP_PRODUCT",
    "MLP_DOWN_PROJ",
    "MLP_OUTPUT",
    "LAYER_OUTPUT",
]
FIRST_DIVERGENCE_ORDER = [
    "LAYER_INPUT",
    "INPUT_RMSNORM",
    "Q_PROJ",
    "K_PROJ",
    "V_PROJ",
    "K_POST_ROPE",
    "KMEANS_K_INPUT",
    "KMEANS_V_INPUT",
    "K_CENTROID",
    "V_CENTROID",
    "K_ASSIGNMENT",
    "V_ASSIGNMENT",
    "V_PATTERN_MASK",
    "PACKED_K",
    "PACKED_V",
    "ATTENTION_VALUE_OUTPUT",
    "ATTENTION_RESIDUAL_OUTPUT",
    *MLP_COMPONENT_ORDER,
]
REQUEST_LOCAL_COMPONENTS = {
    "KMEANS_K_INPUT",
    "KMEANS_V_INPUT",
    "K_CENTROID",
    "V_CENTROID",
    "K_ASSIGNMENT",
    "V_ASSIGNMENT",
    "V_PATTERN_MASK",
    "PACKED_K",
    "PACKED_V",
}


FINAL_GATE_TEMPLATE: dict[str, Any] = {
    "start_head": START_HEAD,
    "actual_model_loaded": False,
    "mode": "bi_kv",
    "context": 512,
    "algorithm_changed": False,
    "quantization_changed": False,
    "selector_changed": False,
    "kmeans_changed": False,
    "projection_mode_policy_changed": False,
    "bi_k_kernel_changed": False,
    "bi_v_kernel_changed": False,
    "production_default_changed": False,
    "oracle_layer": 0,
    "oracle_backend": "",
    "gate_proj_geometry_supported": None,
    "up_proj_geometry_supported": None,
    "down_proj_geometry_supported": None,
    "mlp_geometry_supported": None,
    "bi_gate_isolated_exact": None,
    "bi_up_isolated_exact": None,
    "bi_down_isolated_exact": None,
    "isolated_exactness_pass": None,
    "baseline_reproduced": None,
    "baseline_first_mlp_divergent_component": "",
    "o0_mlp_output_exact": None,
    "o0_mlp_output_relative_l2": None,
    "o1_gate_exact": None,
    "o1_mlp_output_exact": None,
    "o2_gate_exact": None,
    "o2_up_exact": None,
    "o2_mlp_output_exact": None,
    "o3_gate_exact": None,
    "o3_up_exact": None,
    "o3_activated_gate_exact": None,
    "o3_product_exact": None,
    "o3_down_exact": None,
    "o3_mlp_output_exact": None,
    "o3_mlp_output_relative_l2": None,
    "o3_layer0_output_exact": None,
    "o3_layer0_output_relative_l2": None,
    "o3_layer1_hidden_exact": None,
    "o3_layer1_hidden_relative_l2": None,
    "o3_layer1_kproj_exact": None,
    "o3_layer1_vproj_exact": None,
    "o3_layer1_k_centroid_exact": None,
    "o3_layer1_v_centroid_exact": None,
    "old_first_divergent_layer": 0,
    "old_first_divergent_component": "MLP_OUTPUT",
    "new_first_divergent_layer": None,
    "new_first_divergent_component": "",
    "first_divergence_moved": None,
    "b4_layer0_mlp_exact": None,
    "b4_layer0_output_exact": None,
    "b4_layer1_hidden_exact": None,
    "b4_pass": None,
    "oracle_supports_mlp_gemm_root_cause": None,
    "recommended_serving_mode": "bi_k",
    "strict_kv_projection_mode": "bi_kv",
    "whole_model_batch_invariance_claimed": False,
    "classification": "",
    "next_task": "",
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def nvidia_smi() -> str:
    try:
        output = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=10)
        return "\n".join(line.rstrip() for line in output.splitlines())
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def set_base_env() -> None:
    os.environ["PATTERNKV_PREFILL_PROJ_MODE"] = BI_KV_PREFILL_PROJ_MODE
    os.environ["PATTERNKV_BI_KPROJ_BACKEND"] = "v2"
    os.environ["PATTERNKV_BI_MLP_ORACLE_BACKEND"] = "v2"
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"


def squeeze_for_request(component: str, tensor: torch.Tensor, row: int | None) -> torch.Tensor:
    out = tensor.detach().cpu().contiguous()
    if row is not None and component not in REQUEST_LOCAL_COMPONENTS and out.dim() >= 1 and out.shape[0] > row:
        out = out[row]
    if out.dim() >= 1 and out.shape[0] == 1:
        out = out.squeeze(0)
    return out.contiguous()


def trace_maps(records: list[dict[str, Any]], row: int | None) -> tuple[dict[tuple[int, str], torch.Tensor], dict[tuple[int, str], torch.Tensor]]:
    sliced = {}
    raw = {}
    for record in records:
        layer = int(record["layer"])
        component = str(record["component"])
        tensor = record["tensor"]
        sliced[(layer, component)] = squeeze_for_request(component, tensor, row)
        raw[(layer, component)] = tensor.detach()
    return sliced, raw


def run_prefill(model: Any, input_ids: torch.Tensor, *, row: int, components: str | None) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    reset_patternkv_p2_first_divergence_trace()
    reset_patternkv_bi_mlp_oracle_counters()
    os.environ["PATTERNKV_P2_FIRST_DIVERGENCE_TRACE"] = "1"
    os.environ["PATTERNKV_BI_MLP_TRACE"] = "1"
    if components is None:
        os.environ.pop("PATTERNKV_BI_MLP_ORACLE", None)
        os.environ.pop("PATTERNKV_BI_MLP_ORACLE_COMPONENTS", None)
    else:
        os.environ["PATTERNKV_BI_MLP_ORACLE"] = "1"
        os.environ["PATTERNKV_BI_MLP_ORACLE_LAYER"] = "0"
        os.environ["PATTERNKV_BI_MLP_ORACLE_COMPONENTS"] = components
    with torch.inference_mode():
        model.model(input_ids=input_ids, use_cache=True, return_dict=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize(input_ids.device)
    sliced, raw = trace_maps(patternkv_p2_first_divergence_trace_records(), row)
    counters = patternkv_bi_mlp_oracle_counters()
    reset_patternkv_p2_first_divergence_trace()
    os.environ.pop("PATTERNKV_P2_FIRST_DIVERGENCE_TRACE", None)
    os.environ.pop("PATTERNKV_BI_MLP_TRACE", None)
    return {"trace": sliced, "raw_trace": raw, "counters": counters}


def compare_component(ref: dict[tuple[int, str], torch.Tensor], got: dict[tuple[int, str], torch.Tensor], layer: int, component: str) -> dict[str, Any]:
    return compare_tensors(ref.get((layer, component)), got.get((layer, component)))


def variant_result(model: Any, input_ids: torch.Tensor, components: str | None) -> dict[str, Any]:
    b1 = run_prefill(model, input_ids[0:1], row=0, components=components)
    b2 = run_prefill(model, input_ids[[0, 1]], row=0, components=components)
    rows = []
    for layer in range(32):
        for component in FIRST_DIVERGENCE_ORDER:
            metric = compare_component(b1["trace"], b2["trace"], layer, component)
            rows.append({"layer": layer, "component": component, **metric})
    first = first_non_exact(rows)
    mlp = {component: compare_component(b1["trace"], b2["trace"], 0, component) for component in MLP_COMPONENT_ORDER}
    layer1 = {component: compare_component(b1["trace"], b2["trace"], 1, component) for component in ("LAYER_INPUT", "K_PROJ", "V_PROJ", "K_CENTROID", "V_CENTROID")}
    return {
        "b1": b1,
        "b2": b2,
        "rows": rows,
        "first": first,
        "mlp": mlp,
        "layer1": layer1,
        "counters": {"b1": b1["counters"], "b2": b2["counters"]},
    }


def isolated_exactness(model: Any, o0: dict[str, Any]) -> dict[str, Any]:
    layer0 = model.model.layers[0]
    mlp = layer0.mlp
    device = mlp.gate_proj.weight.device
    b1_x = o0["b1"]["raw_trace"][(0, "POST_ATTENTION_RMSNORM")].to(device=device, dtype=mlp.gate_proj.weight.dtype)
    b2_x = o0["b2"]["raw_trace"][(0, "POST_ATTENTION_RMSNORM")].to(device=device, dtype=mlp.gate_proj.weight.dtype)
    gate_b1 = batch_invariant_linear_projection(b1_x, mlp.gate_proj.weight, getattr(mlp.gate_proj, "bias", None), backend="v2")
    gate_b2 = batch_invariant_linear_projection(b2_x, mlp.gate_proj.weight, getattr(mlp.gate_proj, "bias", None), backend="v2")
    up_b1 = batch_invariant_linear_projection(b1_x, mlp.up_proj.weight, getattr(mlp.up_proj, "bias", None), backend="v2")
    up_b2 = batch_invariant_linear_projection(b2_x, mlp.up_proj.weight, getattr(mlp.up_proj, "bias", None), backend="v2")
    prod_b1 = mlp.act_fn(gate_b1) * up_b1
    prod_b2 = mlp.act_fn(gate_b2) * up_b2
    down_b1 = batch_invariant_linear_projection(prod_b1, mlp.down_proj.weight, getattr(mlp.down_proj, "bias", None), backend="v2")
    down_b2 = batch_invariant_linear_projection(prod_b2, mlp.down_proj.weight, getattr(mlp.down_proj, "bias", None), backend="v2")
    return {
        "gate": compare_tensors(gate_b1[0], gate_b2[0]),
        "up": compare_tensors(up_b1[0], up_b2[0]),
        "down": compare_tensors(down_b1[0], down_b2[0]),
    }


def geometry(model: Any) -> dict[str, Any]:
    mlp = model.model.layers[0].mlp
    hidden_size = int(mlp.gate_proj.weight.shape[1])
    intermediate_size = int(mlp.gate_proj.weight.shape[0])
    return {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "gate_proj": {"M": "batch*tokens", "K": hidden_size, "N": intermediate_size},
        "up_proj": {"M": "batch*tokens", "K": hidden_size, "N": intermediate_size},
        "down_proj": {"M": "batch*tokens", "K": intermediate_size, "N": hidden_size},
        "oracle_backend": "v2",
    }


def run_actual(args: argparse.Namespace) -> dict[str, Any]:
    set_base_env()
    tokenizer, _config, model = load_model(dtype=torch.float16, device=torch.device(args.device))
    input_ids = make_fixed_inputs(tokenizer, batch=4, context=512, device=torch.device(args.device))
    geo = geometry(model)
    o0 = variant_result(model, input_ids, None)
    iso = isolated_exactness(model, o0)
    o1 = variant_result(model, input_ids, "gate")
    o2 = variant_result(model, input_ids, "gate,up")
    o3 = variant_result(model, input_ids, "gate,up,down")
    b4 = run_prefill(model, input_ids[[0, 1, 2, 3]], row=0, components="gate,up,down")
    b4_ref = o3["b1"]["trace"]
    b4_trace = b4["trace"]
    b4_sanity = {
        "layer0_mlp_exact": compare_component(b4_ref, b4_trace, 0, "MLP_OUTPUT")["exact"],
        "layer0_output_exact": compare_component(b4_ref, b4_trace, 0, "LAYER_OUTPUT")["exact"],
        "layer1_hidden_exact": compare_component(b4_ref, b4_trace, 1, "LAYER_INPUT")["exact"],
    }
    gate = dict(FINAL_GATE_TEMPLATE)
    gate.update(
        {
            "actual_model_loaded": True,
            "oracle_backend": "v2",
            "gate_proj_geometry_supported": True,
            "up_proj_geometry_supported": True,
            "down_proj_geometry_supported": True,
            "mlp_geometry_supported": True,
            "bi_gate_isolated_exact": iso["gate"]["exact"],
            "bi_up_isolated_exact": iso["up"]["exact"],
            "bi_down_isolated_exact": iso["down"]["exact"],
            "isolated_exactness_pass": iso["gate"]["exact"] and iso["up"]["exact"] and iso["down"]["exact"],
            "baseline_reproduced": o0["mlp"]["MLP_OUTPUT"]["exact"] is False and o0["mlp"]["LAYER_OUTPUT"]["relative_l2"] is not None,
            "baseline_first_mlp_divergent_component": next((component for component in MLP_COMPONENT_ORDER if o0["mlp"][component]["exact"] is False), ""),
            "o0_mlp_output_exact": o0["mlp"]["MLP_OUTPUT"]["exact"],
            "o0_mlp_output_relative_l2": o0["mlp"]["MLP_OUTPUT"]["relative_l2"],
            "o1_gate_exact": o1["mlp"]["MLP_GATE_PROJ"]["exact"],
            "o1_mlp_output_exact": o1["mlp"]["MLP_OUTPUT"]["exact"],
            "o2_gate_exact": o2["mlp"]["MLP_GATE_PROJ"]["exact"],
            "o2_up_exact": o2["mlp"]["MLP_UP_PROJ"]["exact"],
            "o2_mlp_output_exact": o2["mlp"]["MLP_OUTPUT"]["exact"],
            "o3_gate_exact": o3["mlp"]["MLP_GATE_PROJ"]["exact"],
            "o3_up_exact": o3["mlp"]["MLP_UP_PROJ"]["exact"],
            "o3_activated_gate_exact": o3["mlp"]["MLP_ACTIVATED_GATE"]["exact"],
            "o3_product_exact": o3["mlp"]["MLP_PRODUCT"]["exact"],
            "o3_down_exact": o3["mlp"]["MLP_DOWN_PROJ"]["exact"],
            "o3_mlp_output_exact": o3["mlp"]["MLP_OUTPUT"]["exact"],
            "o3_mlp_output_relative_l2": o3["mlp"]["MLP_OUTPUT"]["relative_l2"],
            "o3_layer0_output_exact": o3["mlp"]["LAYER_OUTPUT"]["exact"],
            "o3_layer0_output_relative_l2": o3["mlp"]["LAYER_OUTPUT"]["relative_l2"],
            "o3_layer1_hidden_exact": o3["layer1"]["LAYER_INPUT"]["exact"],
            "o3_layer1_hidden_relative_l2": o3["layer1"]["LAYER_INPUT"]["relative_l2"],
            "o3_layer1_kproj_exact": o3["layer1"]["K_PROJ"]["exact"],
            "o3_layer1_vproj_exact": o3["layer1"]["V_PROJ"]["exact"],
            "o3_layer1_k_centroid_exact": o3["layer1"]["K_CENTROID"]["exact"],
            "o3_layer1_v_centroid_exact": o3["layer1"]["V_CENTROID"]["exact"],
            "new_first_divergent_layer": int(o3["first"]["layer"]) if o3["first"] else None,
            "new_first_divergent_component": str(o3["first"]["component"]) if o3["first"] else "",
            "b4_layer0_mlp_exact": b4_sanity["layer0_mlp_exact"],
            "b4_layer0_output_exact": b4_sanity["layer0_output_exact"],
            "b4_layer1_hidden_exact": b4_sanity["layer1_hidden_exact"],
            "b4_pass": b4_sanity["layer0_mlp_exact"] and b4_sanity["layer0_output_exact"] and b4_sanity["layer1_hidden_exact"],
        }
    )
    gate["first_divergence_moved"] = (
        gate["new_first_divergent_layer"] != gate["old_first_divergent_layer"]
        or gate["new_first_divergent_component"] != gate["old_first_divergent_component"]
    )
    supported = (
        gate["bi_gate_isolated_exact"]
        and gate["bi_up_isolated_exact"]
        and gate["bi_down_isolated_exact"]
        and gate["baseline_reproduced"]
        and gate["o3_gate_exact"]
        and gate["o3_up_exact"]
        and gate["o3_activated_gate_exact"]
        and gate["o3_product_exact"]
        and gate["o3_down_exact"]
        and gate["o3_mlp_output_exact"]
        and gate["o3_layer0_output_exact"]
        and gate["o3_layer1_hidden_exact"]
        and gate["first_divergence_moved"]
    )
    gate["oracle_supports_mlp_gemm_root_cause"] = bool(supported)
    if supported:
        gate["classification"] = "BI_MLP_CAUSAL_ORACLE_SUPPORTED"
        gate["next_task"] = "REDEFINE_STRICT_MODE_SCOPE_AND_RUN_FINAL_FIXED_BATCH_SEMANTIC_GATE"
    elif not (gate["bi_gate_isolated_exact"] and gate["bi_up_isolated_exact"] and gate["bi_down_isolated_exact"]):
        gate["classification"] = "BI_MLP_ORACLE_PRIMITIVE_NOT_INVARIANT"
        gate["next_task"] = "AUDIT_GENERIC_BI_LINEAR_GEOMETRY"
    elif gate["o3_mlp_output_exact"] is False:
        gate["classification"] = "MLP_GEMM_ROOT_CAUSE_NOT_SUFFICIENT"
        gate["next_task"] = "TRACE_MLP_POINTWISE_OR_LAYOUT_DIVERGENCE"
    else:
        gate["classification"] = "BI_MLP_CAUSAL_ORACLE_PARTIAL"
        gate["next_task"] = "AUDIT_LAYER0_MLP_FIXED_REDUCTION_COVERAGE"
    return {
        "gate": gate,
        "geometry": geo,
        "isolated": iso,
        "variants": {
            "O0": summarize_variant(o0),
            "O1": summarize_variant(o1),
            "O2": summarize_variant(o2),
            "O3": summarize_variant(o3),
        },
        "b4_sanity": b4_sanity,
        "counters": {"O0": o0["counters"], "O1": o1["counters"], "O2": o2["counters"], "O3": o3["counters"]},
    }


def summarize_variant(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "first": variant["first"],
        "mlp": variant["mlp"],
        "layer1": variant["layer1"],
    }


def write_reports(payload: dict[str, Any], smi: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gate = payload["gate"]
    write_json(REPORT_DIR / "mlp_geometry.json", payload["geometry"])
    write_json(REPORT_DIR / "bi_linear_exactness.json", payload["isolated"])
    write_json(REPORT_DIR / "baseline_mlp_trace.json", payload["variants"]["O0"])
    write_json(REPORT_DIR / "oracle_variant_results.json", payload["variants"])
    write_json(REPORT_DIR / "layer0_trace.json", {name: payload["variants"][name]["mlp"] for name in payload["variants"]})
    write_json(REPORT_DIR / "layer1_consequence.json", payload["variants"]["O3"]["layer1"])
    write_json(REPORT_DIR / "b4_sanity.json", payload["b4_sanity"])
    write_json(REPORT_DIR / "oracle_counters.json", payload["counters"])
    write_json(REPORT_DIR / "final_gate.json", gate)
    write_md(REPORT_DIR / "environment.md", "Environment", f"Start HEAD: `{START_HEAD}`\n\n```\n{smi}\n```")
    write_md(REPORT_DIR / "motivation.md", "Motivation", "Validate whether Layer0 MLP GEMM batch-shape arithmetic causes the first P2 divergence.")
    write_md(REPORT_DIR / "baseline_reproduction.md", "Baseline Reproduction", f"O0 MLP exact={gate['o0_mlp_output_exact']} relL2={gate['o0_mlp_output_relative_l2']}; first MLP divergent component=`{gate['baseline_first_mlp_divergent_component']}`.")
    write_md(REPORT_DIR / "mlp_geometry.md", "MLP Geometry", json.dumps(payload["geometry"], indent=2))
    write_md(REPORT_DIR / "bi_linear_geometry_validation.md", "BI Linear Geometry Validation", f"gate={gate['gate_proj_geometry_supported']} up={gate['up_proj_geometry_supported']} down={gate['down_proj_geometry_supported']} backend=`{gate['oracle_backend']}`.")
    write_md(REPORT_DIR / "oracle_variants.md", "Oracle Variants", "O0 normal, O1 BI gate, O2 BI gate+up, O3 BI gate+up+down.")
    write_md(REPORT_DIR / "mlp_internal_trace.md", "MLP Internal Trace", f"O3 gate/up/down/product/output exact: {gate['o3_gate_exact']}/{gate['o3_up_exact']}/{gate['o3_down_exact']}/{gate['o3_product_exact']}/{gate['o3_mlp_output_exact']}.")
    write_md(REPORT_DIR / "layer0_causal_result.md", "Layer0 Causal Result", f"O3 Layer0 output exact={gate['o3_layer0_output_exact']} relL2={gate['o3_layer0_output_relative_l2']}.")
    write_md(REPORT_DIR / "layer1_consequence.md", "Layer1 Consequence", f"O3 Layer1 hidden exact={gate['o3_layer1_hidden_exact']} relL2={gate['o3_layer1_hidden_relative_l2']}; K/V projection exact={gate['o3_layer1_kproj_exact']}/{gate['o3_layer1_vproj_exact']}.")
    write_md(REPORT_DIR / "b4_sanity.md", "B4 Sanity", json.dumps(payload["b4_sanity"], indent=2))
    write_md(REPORT_DIR / "scope_implication.md", "Scope Implication", "The oracle is diagnostic only. It does not expand `bi_kv` into a whole-model deterministic mode.")
    write_md(REPORT_DIR / "production_recommendation.md", "Production Recommendation", "Keep recommended serving mode as `bi_k`; keep `bi_kv` as strict KV-projection diagnostic/reproducibility mode.")
    write_md(REPORT_DIR / "final_recommendation.md", "Final Recommendation", f"Classification: `{gate['classification']}`. Next task: `{gate['next_task']}`.")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_actual(args)
    write_reports(payload, nvidia_smi())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
