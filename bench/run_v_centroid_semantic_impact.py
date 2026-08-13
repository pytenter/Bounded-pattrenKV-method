from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.prefill_v_trace_utils import (
    bi_vproj_control,
    centroid_only_counterfactual_metrics,
    difference_rate,
    packed_v_metrics,
    reconstructed_packed_v_metrics,
    relative_l2,
    run_v_state,
    same_attention_value_metrics,
    semantic_impact_level,
    semantic_state_comparator,
    tensor_metric_dict,
    trace_reference_kmeans,
    clone_cache_with_v_centroids,
    fused_value_metrics,
)
from bench.run_actual_model_fixed_batch_smoke import MODEL_PATH, load_model, make_fixed_inputs
from bench.run_actual_model_k_assignment_trace import make_position_ids
from bench.run_actual_model_prefill_v_centroid_trace import layer_hidden_pair, normal_value_pair
from bench.run_serving_stable_k_centroid_eval import hidden_for_layers
from models.llama_patternkv import apply_rotary_pos_emb, repeat_kv
from quant.batch_invariant_kproj import batch_invariant_k_projection_v2


REPORT_DIR = REPO_ROOT / "reports/system_v_centroid_semantic_impact_v1"
START_HEAD = "17fad4114f4ebbcc4185fda45e1c10bae7705bf0"


def write_json(path: Path, value: Any) -> None:
    def scrub(obj: Any) -> Any:
        if torch.is_tensor(obj):
            return {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
                "sha256": hashlib.sha256(obj.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            }
        if isinstance(obj, dict):
            return {k: scrub(v) for k, v in obj.items() if k not in {"got", "ref"}}
        if isinstance(obj, list):
            return [scrub(v) for v in obj]
        return obj

    path.write_text(json.dumps(scrub(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def set_env() -> None:
    os.environ["PATTERNKV_BATCH_INVARIANT_KPROJ"] = "1"
    os.environ["PATTERNKV_BI_KPROJ_BACKEND"] = "v2"
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"


def layer_values(model: Any, tokenizer: Any, layer_idx: int, device: torch.device) -> dict[str, Any]:
    hidden = layer_hidden_pair(model, tokenizer, layer_idx, device)
    values = normal_value_pair(model, layer_idx, hidden["hidden_b1"], hidden["hidden_b2"], hidden["hidden_b4"])
    return {"hidden": hidden, "values": values}


def build_state(model: Any, layer_idx: int, value_states: torch.Tensor) -> dict[str, Any]:
    attn = model.model.layers[layer_idx].self_attn
    trace = trace_reference_kmeans(value_states[0].float(), int(attn.num_v_bases), iters=30, tol=1e-4, seed=0)
    state = run_v_state(
        value_states,
        trace["final_centroids"].to(value_states.dtype),
        value_objective=attn.value_objective,
        group_size=int(attn.group_size),
        bits=int(attn.v_bits),
        v4_budget_fraction=float(attn.v4_budget_fraction),
    )
    state["centroids"] = trace["final_centroids"].to(value_states.dtype)
    return state


def b1_attention_weights(model: Any, layer_idx: int, hidden_b1: torch.Tensor) -> torch.Tensor:
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    with torch.inference_mode():
        normed = layer.input_layernorm(hidden_b1)
        q_proj = attn.q_proj(normed)
        k_proj = batch_invariant_k_projection_v2(normed, attn.k_proj.weight, getattr(attn.k_proj, "bias", None))
        bsz, seq_len, _ = normed.shape
        query = q_proj.view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        key = k_proj.view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        pos = make_position_ids(torch.empty((bsz, seq_len), dtype=torch.long, device=hidden_b1.device))
        cos, sin = attn.rotary_emb(key, pos)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, pos)
        scores = torch.matmul(query, repeat_kv(key, attn.num_key_value_groups).transpose(2, 3)) / math.sqrt(attn.head_dim)
        causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_b1.device), diagonal=1)
        scores = scores.masked_fill(causal[None, None, :, :], torch.finfo(scores.dtype).min)
        return F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)


def post_o_proj_metrics(model: Any, layer_idx: int, value_metrics: dict[str, Any]) -> dict[str, Any]:
    attn = model.model.layers[layer_idx].self_attn
    got = value_metrics["got"].transpose(1, 2).contiguous().reshape(value_metrics["got"].shape[0], value_metrics["got"].shape[2], attn.hidden_size)
    ref = value_metrics["ref"].transpose(1, 2).contiguous().reshape(value_metrics["ref"].shape[0], value_metrics["ref"].shape[2], attn.hidden_size)
    return tensor_metric_dict(attn.o_proj(got), attn.o_proj(ref))


def layer0_output_metrics(model: Any, hidden_b1: torch.Tensor, post_o_got: torch.Tensor, post_o_ref: torch.Tensor) -> dict[str, Any]:
    layer = model.model.layers[0]
    ref = hidden_b1 + post_o_ref
    got = hidden_b1 + post_o_got
    ref_out = ref + layer.mlp(layer.post_attention_layernorm(ref))
    got_out = got + layer.mlp(layer.post_attention_layernorm(got))
    return tensor_metric_dict(got_out, ref_out)


def run_layer_audit(model: Any, tokenizer: Any, layer_idx: int, device: torch.device) -> dict[str, Any]:
    pair = layer_values(model, tokenizer, layer_idx, device)
    hidden = pair["hidden"]
    values = pair["values"]
    state1 = build_state(model, layer_idx, values["normal_b1"])
    state2 = build_state(model, layer_idx, values["normal_b2_row0"])
    state4 = build_state(model, layer_idx, values["normal_b4_row0"])
    bi_state1 = build_state(model, layer_idx, values["bi_b1"])
    bi_state2 = build_state(model, layer_idx, values["bi_b2_row0"])
    bi_state4 = build_state(model, layer_idx, values["bi_b4_row0"])
    attn_full = b1_attention_weights(model, layer_idx, hidden["hidden_b1"])
    sink_end = 16
    packed_tokens = int(state1["cache"].packed_v_tokens)
    attn_packed = attn_full[:, :, -1:, sink_end : sink_end + packed_tokens].contiguous()
    rec = reconstructed_packed_v_metrics(state2["cache"], state1["cache"])
    rec4 = reconstructed_packed_v_metrics(state4["cache"], state1["cache"])
    rec_bi = reconstructed_packed_v_metrics(bi_state2["cache"], bi_state1["cache"])
    rec_bi4 = reconstructed_packed_v_metrics(bi_state4["cache"], bi_state1["cache"])
    same = same_attention_value_metrics(attn_packed, rec["got"], rec["ref"], model.model.layers[layer_idx].self_attn.num_key_value_groups)
    same4 = same_attention_value_metrics(attn_packed, rec4["got"], rec4["ref"], model.model.layers[layer_idx].self_attn.num_key_value_groups)
    same_bi = same_attention_value_metrics(attn_packed, rec_bi["got"], rec_bi["ref"], model.model.layers[layer_idx].self_attn.num_key_value_groups)
    same_bi4 = same_attention_value_metrics(attn_packed, rec_bi4["got"], rec_bi4["ref"], model.model.layers[layer_idx].self_attn.num_key_value_groups)
    return {
        "layer": layer_idx,
        "state1": state1,
        "state2": state2,
        "state4": state4,
        "bi_state1": bi_state1,
        "bi_state2": bi_state2,
        "bi_state4": bi_state4,
        "attention_packed": attn_packed,
        "v_centroid_relative_l2": relative_l2(state2["centroids"], state1["centroids"]),
        "b4_v_centroid_relative_l2": relative_l2(state4["centroids"], state1["centroids"]),
        "reconstructed": rec,
        "b4_reconstructed": rec4,
        "same_attention": same,
        "b4_same_attention": same4,
        "bi_reconstructed": rec_bi,
        "bi_b4_reconstructed": rec_bi4,
        "bi_same_attention": same_bi,
        "bi_b4_same_attention": same_bi4,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_env()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = torch.device(args.device)
    tokenizer, _config, model = load_model(dtype, device)
    primary = run_layer_audit(model, tokenizer, 0, device)
    s1 = primary["state1"]["cache"]
    s2 = primary["state2"]["cache"]
    baseline = {
        "v_centroid_hash_diff": not torch.equal(primary["state2"]["centroids"], primary["state1"]["centroids"]),
        "v_centroid_relative_l2": primary["v_centroid_relative_l2"],
        "assignment_difference_rate": difference_rate(s2.v_assignment_idx, s1.v_assignment_idx),
        "mask_difference_rate": difference_rate(s2.v_pattern_mask, s1.v_pattern_mask),
        "precision_mask_difference_rate": difference_rate(s2.v_precision_mask, s1.v_precision_mask),
        **packed_v_metrics(s2, s1),
    }
    baseline_reproduced = bool(
        baseline["v_centroid_hash_diff"]
        and baseline["assignment_difference_rate"] == 0.0
        and baseline["mask_difference_rate"] == 0.0
        and baseline["precision_mask_difference_rate"] == 0.0
        and baseline["packed_v_payload_difference_rate"] == 0.0
        and baseline["packed_v_scale_relative_l2"] == 0.0
        and baseline["packed_v_zero_relative_l2"] == 0.0
    )
    centroid_only = centroid_only_counterfactual_metrics(s1, primary["state2"]["centroids"])
    centroid_only_cache = clone_cache_with_v_centroids(s1, primary["state2"]["centroids"])
    fused = fused_value_metrics(primary["attention_packed"], centroid_only_cache, s1)
    ref_b1 = same_attention_value_metrics(primary["attention_packed"], primary["reconstructed"]["ref"], primary["reconstructed"]["ref"], model.model.layers[0].self_attn.num_key_value_groups)
    ref_b2 = same_attention_value_metrics(primary["attention_packed"], primary["reconstructed"]["got"], primary["reconstructed"]["got"], model.model.layers[0].self_attn.num_key_value_groups)
    fused_b1 = fused_value_metrics(primary["attention_packed"], s1, s1)
    fused_b2 = fused_value_metrics(primary["attention_packed"], s2, s2)
    reference_fused = {
        "reference_vs_fused_b1_relative_l2": relative_l2(fused_b1["got"], ref_b1["got"]),
        "reference_vs_fused_b2_relative_l2": relative_l2(fused_b2["got"], ref_b2["got"]),
    }
    post = post_o_proj_metrics(model, 0, primary["same_attention"])
    attn = model.model.layers[0].self_attn
    post_got = attn.o_proj(primary["same_attention"]["got"].transpose(1, 2).contiguous().reshape(1, primary["same_attention"]["got"].shape[2], attn.hidden_size))
    post_ref = attn.o_proj(primary["same_attention"]["ref"].transpose(1, 2).contiguous().reshape(1, primary["same_attention"]["ref"].shape[2], attn.hidden_size))
    ids = make_fixed_inputs(tokenizer, 1, 512, device)
    hidden0 = model.model.embed_tokens(ids).detach()
    layer0 = layer0_output_metrics(model, hidden0[:, -1:, :], post_got, post_ref)
    comparator = semantic_state_comparator(s1, s2)
    impact = semantic_impact_level(primary["same_attention"]["relative_l2"], post["relative_l2"])
    multi_rows = []
    for layer_idx in (0, 8, 16, 31):
        audit = primary if layer_idx == 0 else run_layer_audit(model, tokenizer, layer_idx, device)
        multi_rows.append(
            {
                "layer": layer_idx,
                "v_centroid_relative_l2": audit["v_centroid_relative_l2"],
                "reconstructed_v_relative_l2": audit["reconstructed"]["relative_l2"],
                "same_attn_value_output_relative_l2": audit["same_attention"]["relative_l2"],
            }
        )
    b4 = run_layer_audit(model, tokenizer, 0, device)
    return {
        "actual_model_loaded": True,
        "baseline": baseline | {"baseline_reproduced": baseline_reproduced},
        "primary": primary,
        "centroid_only": centroid_only,
        "fused": fused,
        "reference_fused": reference_fused,
        "post_o_proj": post,
        "layer0_output": layer0,
        "semantic_state_comparator": comparator,
        "semantic_impact_level": impact,
        "multi_layer": multi_rows,
        "b4_sanity": {
            "normal_b1_b4_v_centroid_relative_l2": b4["b4_v_centroid_relative_l2"],
            "normal_b1_b4_reconstructed_v_relative_l2": b4["b4_reconstructed"]["relative_l2"],
            "normal_b1_b4_same_attn_value_output_relative_l2": b4["b4_same_attention"]["relative_l2"],
            "bi_vproj_reconstructed_v_relative_l2": b4["bi_reconstructed"]["relative_l2"],
            "bi_vproj_same_attn_value_output_relative_l2": b4["bi_same_attention"]["relative_l2"],
            "bi_vproj_b1_b4_reconstructed_v_relative_l2": b4["bi_b4_reconstructed"]["relative_l2"],
            "bi_vproj_b1_b4_same_attn_value_output_relative_l2": b4["bi_b4_same_attention"]["relative_l2"],
        },
    }


def classify(results: dict[str, Any]) -> dict[str, str]:
    if not results["baseline"]["baseline_reproduced"]:
        return {
            "classification": "V_SEMANTIC_AUDIT_BASELINE_REPRO_FAILED",
            "root_cause_class": "BASELINE_REPRODUCTION_FAILED",
            "next_task": "FIX_V_SEMANTIC_REFERENCE_VALIDATION",
        }
    ref_fused = max(results["reference_fused"]["reference_vs_fused_b1_relative_l2"], results["reference_fused"]["reference_vs_fused_b2_relative_l2"])
    effect = results["primary"]["same_attention"]["relative_l2"]
    if ref_fused > max(effect * 10.0, 1e-5):
        return {
            "classification": "V_SEMANTIC_AUDIT_FUSED_REFERENCE_INCONCLUSIVE",
            "root_cause_class": "FUSED_REFERENCE_VALIDATION_INCONCLUSIVE",
            "next_task": "FIX_V_SEMANTIC_REFERENCE_VALIDATION",
        }
    impact = results["semantic_impact_level"]
    if impact == "NEGLIGIBLE":
        return {
            "classification": "V_CENTROID_SEMANTIC_EQUIVALENCE_SUPPORTED",
            "root_cause_class": "V_CENTROID_BITWISE_DRIFT_WITHOUT_MEANINGFUL_SEMANTIC_DIVERGENCE",
            "next_task": "REDEFINE_FIXED_BATCH_STATE_EQUIVALENCE_GATE",
        }
    if impact == "SMALL":
        return {
            "classification": "V_CENTROID_SEMANTIC_DRIFT_SMALL",
            "root_cause_class": "V_CENTROID_SMALL_SEMANTIC_DRIFT",
            "next_task": "EVALUATE_BI_VPROJ_COST_BENEFIT",
        }
    if impact == "MEANINGFUL":
        return {
            "classification": "V_CENTROID_SEMANTIC_DIVERGENCE_CONFIRMED",
            "root_cause_class": "V_CENTROID_MEANINGFUL_SEMANTIC_DIVERGENCE",
            "next_task": "INTEGRATE_BATCH_INVARIANT_KVPROJ_PREFILL_RUNTIME",
        }
    return {
        "classification": "V_SEMANTIC_AUDIT_INCONCLUSIVE",
        "root_cause_class": "UNKNOWN",
        "next_task": "FIX_V_SEMANTIC_REFERENCE_VALIDATION",
    }


def final_gate(results: dict[str, Any], decision: dict[str, str]) -> dict[str, Any]:
    b = results["baseline"]
    rec = results["primary"]["reconstructed"]
    same = results["primary"]["same_attention"]
    fused = results["fused"]
    post = results["post_o_proj"]
    return {
        "start_head": START_HEAD,
        "actual_model_loaded": results["actual_model_loaded"],
        "algorithm_changed": False,
        "quantization_changed": False,
        "selector_changed": False,
        "kmeans_changed": False,
        "production_vproj_changed": False,
        "k_payload_layout_changed": False,
        "v_page_abi_changed": False,
        "centroid_state_architecture_changed": False,
        "fused_value_arithmetic_changed": False,
        "baseline_reproduced": b["baseline_reproduced"],
        "v_centroid_hash_diff": b["v_centroid_hash_diff"],
        "v_centroid_relative_l2": b["v_centroid_relative_l2"],
        "assignment_difference_rate": b["assignment_difference_rate"],
        "mask_difference_rate": b["mask_difference_rate"],
        "precision_mask_difference_rate": b["precision_mask_difference_rate"],
        "packed_v_payload_difference_rate": b["packed_v_payload_difference_rate"],
        "packed_v_scale_relative_l2": b["packed_v_scale_relative_l2"],
        "packed_v_zero_relative_l2": b["packed_v_zero_relative_l2"],
        "reconstructed_v_exact": rec["exact"],
        "reconstructed_v_relative_l2": rec["relative_l2"],
        "reconstructed_v_max_abs": rec["max_abs"],
        "reconstructed_v_cosine": rec["cosine"],
        "centroid_only_reconstructed_v_relative_l2": results["centroid_only"]["centroid_only_reconstructed_v_relative_l2"],
        "same_attn_value_output_relative_l2": same["relative_l2"],
        "same_attn_value_output_max_abs": same["max_abs"],
        "same_attn_value_output_cosine": same["cosine"],
        "fused_value_output_relative_l2": fused["relative_l2"],
        "fused_value_output_max_abs": fused["max_abs"],
        "fused_value_output_cosine": fused["cosine"],
        "reference_vs_fused_b1_relative_l2": results["reference_fused"]["reference_vs_fused_b1_relative_l2"],
        "reference_vs_fused_b2_relative_l2": results["reference_fused"]["reference_vs_fused_b2_relative_l2"],
        "post_o_proj_relative_l2": post["relative_l2"],
        "post_o_proj_max_abs": post["max_abs"],
        "post_o_proj_cosine": post["cosine"],
        "layer0_output_relative_l2": results["layer0_output"]["relative_l2"],
        "bi_vproj_reconstructed_v_relative_l2": results["b4_sanity"]["bi_vproj_reconstructed_v_relative_l2"],
        "bi_vproj_same_attn_value_output_relative_l2": results["b4_sanity"]["bi_vproj_same_attn_value_output_relative_l2"],
        "b4_reconstructed_v_relative_l2": results["b4_sanity"]["normal_b1_b4_reconstructed_v_relative_l2"],
        "b4_same_attn_value_output_relative_l2": results["b4_sanity"]["normal_b1_b4_same_attn_value_output_relative_l2"],
        "bi_vproj_b1_b4_reconstructed_v_relative_l2": results["b4_sanity"]["bi_vproj_b1_b4_reconstructed_v_relative_l2"],
        "bi_vproj_b1_b4_same_attn_value_output_relative_l2": results["b4_sanity"]["bi_vproj_b1_b4_same_attn_value_output_relative_l2"],
        "semantic_comparator_implemented": True,
        "multi_layer_sanity_completed": True,
        "semantic_impact_level": results["semantic_impact_level"],
        "root_cause_class": decision["root_cause_class"],
        "classification": decision["classification"],
        "next_task": decision["next_task"],
    }


def write_reports(results: dict[str, Any], gate: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / "baseline_state.json", results["baseline"])
    write_json(REPORT_DIR / "reconstructed_v_metrics.json", results["primary"]["reconstructed"])
    write_json(REPORT_DIR / "centroid_only_counterfactual.json", results["centroid_only"])
    write_json(REPORT_DIR / "same_attention_value_output.json", results["primary"]["same_attention"])
    write_json(REPORT_DIR / "fused_value_metrics.json", results["fused"])
    write_json(REPORT_DIR / "reference_fused_metrics.json", results["reference_fused"])
    write_json(REPORT_DIR / "post_o_proj_metrics.json", results["post_o_proj"])
    write_csv(REPORT_DIR / "multi_layer.csv", results["multi_layer"], list(results["multi_layer"][0].keys()))
    write_json(REPORT_DIR / "semantic_state_comparator.json", results["semantic_state_comparator"])
    write_json(REPORT_DIR / "b4_sanity.json", results["b4_sanity"])
    write_json(REPORT_DIR / "final_gate.json", gate)
    docs = {
        "environment.md": f"# Environment\n\nStart HEAD: {START_HEAD}\n\nRepo: pytenter/Bounded-pattrenKV-method\n\nLocal: {REPO_ROOT}\n\nModel: {MODEL_PATH}\n\nCUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}\n",
        "baseline_reproduction.md": f"# Baseline Reproduction\n\nBaseline reproduced: `{gate['baseline_reproduced']}`.\n",
        "why_v_centroid_hash_failed.md": "# Why V Centroid Hash Failed\n\n`v_centroid_hash` is a byte-exact SHA comparison. A centroid tensor can differ at FP16 bytes while assignment, mask, precision mask, packed payload, and operator output remain semantically equivalent within tolerance.\n",
        "semantic_equivalence_definition.md": "# Semantic Equivalence Definition\n\nTier 0 is bitwise centroid equality. Tier 1 is structural equality for assignments, masks, precision masks, packed payloads, and integer metadata. Tier 2 measures reconstructed V. Tier 3 measures same-attention Value output. Tier 4 measures post-o-proj/layer output.\n",
        "reconstructed_v.md": "# Reconstructed V\n\nSee `reconstructed_v_metrics.json`.\n",
        "centroid_only_counterfactual.md": "# Centroid Only Counterfactual\n\nThe B1 packed state is reused while only the centroid bank is swapped. See `centroid_only_counterfactual.json`.\n",
        "same_attention_value_output.md": "# Same Attention Value Output\n\nSame B1 attention weights are used for both reconstructed V states. See `same_attention_value_output.json`.\n",
        "fused_value_output.md": "# Fused Value Output\n\nThe fused page Value operator is called with the same attention weights and centroid-only states. See `fused_value_metrics.json`.\n",
        "reference_vs_fused.md": "# Reference Vs Fused\n\nSee `reference_fused_metrics.json`.\n",
        "post_o_proj.md": "# Post O-Proj\n\nSee `post_o_proj_metrics.json`.\n",
        "multi_layer_sanity.md": "# Multi Layer Sanity\n\nSee `multi_layer.csv`. Layer0 is the primary causal audit. Deeper layers are sanity probes only because their hidden inputs may already include upstream numerical drift; they are not used as the classification gate in this phase.\n",
        "k_vs_v_comparison.md": f"# K Vs V Comparison\n\nK side authoritative: centroid amplification about 195x, reconstructed K rel L2 about 0.1169, QK rel L2 about 0.0632.\n\nV side measured here: centroid rel L2 `{gate['v_centroid_relative_l2']}`, reconstructed V rel L2 `{gate['reconstructed_v_relative_l2']}`, same-attention Value rel L2 `{gate['same_attn_value_output_relative_l2']}`.\n",
        "semantic_gate_recommendation.md": f"# Semantic Gate Recommendation\n\nHeuristic impact level: `{gate['semantic_impact_level']}`. Use semantic state tiers rather than centroid SHA equality as the only fixed-batch gate.\n",
        "root_cause_analysis.md": f"# Root Cause Analysis\n\nROOT_CAUSE={gate['root_cause_class']}\n\nCLASSIFICATION={gate['classification']}\n",
        "final_recommendation.md": f"# Final Recommendation\n\nCLASSIFICATION={gate['classification']}\n\nNEXT_TASK={gate['next_task']}\n",
    }
    for name, text in docs.items():
        (REPORT_DIR / name).write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    return parser.parse_args()


def main() -> None:
    results = run(parse_args())
    decision = classify(results)
    gate = final_gate(results, decision)
    write_reports(results, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
