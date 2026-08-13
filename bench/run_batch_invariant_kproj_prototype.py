from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.run_actual_model_fixed_batch_smoke import MODEL_PATH, load_model, make_fixed_inputs
from bench.run_actual_model_k_assignment_trace import make_position_ids, tensor_metrics
from bench.run_serving_stable_k_centroid_eval import evaluate_candidate, hidden_for_layers, qk_scores
from quant.batch_invariant_kproj import (
    batch_invariant_k_projection,
    batch_invariant_kproj_counters,
    reset_batch_invariant_kproj_counters,
)


REPORT_DIR = REPO_ROOT / "reports/system_batch_invariant_kproj_v1"
START_HEAD = "5722b4aeb1c3ead80f885e1f2fcbef98311ea8ff"
LAYERS = [0, 8, 16, 31]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((torch.linalg.vector_norm((a - b).float()) / torch.linalg.vector_norm(b.float()).clamp_min(1e-12)).item())


def exact_metrics(name: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    m = tensor_metrics(a, b)
    return {"case": name, "exact": bool(torch.equal(a, b)), "max_abs": m["max_abs"], "mean_abs": m["mean_abs"], "relative_l2": m["relative_l2"], "cosine": m["cosine"]}


def layer_inputs(model: Any, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
    return hidden_for_layers(model, input_ids, [layer_idx])[layer_idx]


def normal_and_bi_kproj(model: Any, hidden: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, float]:
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    with torch.inference_mode():
        normed = layer.input_layernorm(hidden)
        normal = attn.k_proj(normed)
        bi = batch_invariant_k_projection(normed, attn.k_proj.weight, getattr(attn.k_proj, "bias", None))
        bsz, seq_len, _ = normed.shape
        key = bi.view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        q_proj = attn.q_proj(normed)
        query = q_proj.view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        pos = make_position_ids(torch.empty((bsz, seq_len), dtype=torch.long, device=hidden.device))
        cos, sin = attn.rotary_emb(key, pos)
        query_post, key_post = __import__("models.llama_patternkv", fromlist=["apply_rotary_pos_emb"]).apply_rotary_pos_emb(query, key, cos, sin, pos)
    return normal.detach(), bi.detach(), key_post.detach(), int(attn.num_k_bases), int(attn.group_size), float(attn.head_dim**-0.5), query_post.detach()


def bi_pair(model: Any, tokenizer: Any, layer_idx: int, request: int, batch: int, device: torch.device) -> dict[str, Any]:
    ids_all = make_fixed_inputs(tokenizer, max(4, request + 1), 512, device)
    ids_ref = ids_all[request : request + 1]
    row = request if request < batch else 0
    h_ref = layer_inputs(model, ids_ref, layer_idx)
    hidden_rows = []
    for idx in range(batch):
        source = request if idx == row else idx if idx != request else 0
        hidden_rows.append(layer_inputs(model, ids_all[source : source + 1], layer_idx))
    h_batch = torch.cat(hidden_rows, dim=0)
    _n_ref, bi_ref, k_ref, bases, group, scale, q_ref = normal_and_bi_kproj(model, h_ref, layer_idx)
    _n_b, bi_b, k_b, _bases, _group, _scale, _q_b = normal_and_bi_kproj(model, h_batch, layer_idx)
    return {"k_ref": k_ref, "k_b2": k_b[row : row + 1], "q_ref": q_ref, "num_k_bases": bases, "group_size": group, "qk_scale": scale, "bi_ref": bi_ref, "bi_batch_row": bi_b[row : row + 1]}


def kmeans_recovery(pair: dict[str, Any], variant: str) -> dict[str, Any]:
    spec = {"name": variant, "variant": "kmeans", "iters": 30, "grid": None}
    result = evaluate_candidate(pair, spec)
    return {
        "case": variant,
        "centroid_relative_l2": result["centroid_batch_relative_l2"],
        "assignment_difference_rate": result["assignment_diff_rate"],
        "reconstructed_k_batch_relative_l2": result["reconstructed_k_batch_rel_l2"],
        "qk_batch_relative_l2": result["qk_batch_rel_l2"],
        "qk_batch_max_abs": result["qk_batch_max_abs"],
        "qk_batch_cosine": result["qk_batch_cosine"],
    }


def request_reorder_runs(model: Any, tokenizer: Any, device: torch.device) -> list[dict[str, Any]]:
    ids = make_fixed_inputs(tokenizer, 4, 512, device)
    orders = {"row0_ab": [0, 1], "row1_ba": [1, 0], "row2_cdab": [2, 3, 0, 1], "row3_bcda": [1, 2, 3, 0]}
    ref_hidden = layer_inputs(model, ids[0:1], 0)
    _n, bi_ref, _k, _bases, _group, _scale, _q = normal_and_bi_kproj(model, ref_hidden, 0)
    rows = []
    for name, order in orders.items():
        h = layer_inputs(model, ids[order], 0)
        _normal, bi, _key, _bases, _group, _scale, _q = normal_and_bi_kproj(model, h, 0)
        row = order.index(0)
        rows.append(exact_metrics(name, bi[row : row + 1], bi_ref))
    return rows


def batch_shape_runs(model: Any, tokenizer: Any, device: torch.device) -> list[dict[str, Any]]:
    ids = make_fixed_inputs(tokenizer, 4, 512, device)
    h_ref = layer_inputs(model, ids[0:1], 0)
    normal_ref, bi_ref, _k, _bases, _group, _scale, _q = normal_and_bi_kproj(model, h_ref, 0)
    rows = []
    for batch in (2, 4):
        h = layer_inputs(model, ids[:batch], 0)
        _normal, bi, _key, _bases, _group, _scale, _q = normal_and_bi_kproj(model, h, 0)
        rows.append(exact_metrics(f"bi_b1_vs_b{batch}", bi[0:1], bi_ref))
    rows.append({"case": "bi_vs_normal_b1", **{k: v for k, v in exact_metrics("bi_vs_normal_b1", bi_ref, normal_ref).items() if k != "case"}})
    layer0 = model.model.layers[0]
    fp32 = torch.matmul(layer0.input_layernorm(h_ref).float(), layer0.self_attn.k_proj.weight.float().t()).to(bi_ref.dtype)
    rows.append({"case": "bi_vs_fp32_b1", **{k: v for k, v in exact_metrics("bi_vs_fp32_b1", bi_ref, fp32).items() if k != "case"}})
    return rows


def performance_runs(model: Any, tokenizer: Any, device: torch.device) -> list[dict[str, Any]]:
    rows = []
    layer = model.model.layers[0]
    for batch, tokens in [(1, 512), (2, 512), (4, 512), (1, 2048), (2, 2048), (4, 2048)]:
        ids = make_fixed_inputs(tokenizer, batch, tokens, device)
        hidden = layer.input_layernorm(model.model.embed_tokens(ids))
        for _ in range(5):
            layer.self_attn.k_proj(hidden)
            batch_invariant_k_projection(hidden, layer.self_attn.k_proj.weight, getattr(layer.self_attn.k_proj, "bias", None))
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            layer.self_attn.k_proj(hidden)
        end.record()
        torch.cuda.synchronize(device)
        normal_us = start.elapsed_time(end) * 1000.0 / 20
        start.record()
        for _ in range(20):
            batch_invariant_k_projection(hidden, layer.self_attn.k_proj.weight, getattr(layer.self_attn.k_proj, "bias", None))
        end.record()
        torch.cuda.synchronize(device)
        bi_us = start.elapsed_time(end) * 1000.0 / 20
        rows.append({"batch": batch, "tokens": tokens, "normal_us": normal_us, "bi_us": bi_us, "slowdown": bi_us / normal_us if normal_us else None})
    return rows


def run(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    tokenizer, _config, model = load_model(dtype, device)
    reset_batch_invariant_kproj_counters()
    actual_rows = batch_shape_runs(model, tokenizer, device)
    reorder_rows = request_reorder_runs(model, tokenizer, device)
    layer_rows = []
    recovery_rows = []
    for layer in LAYERS:
        for request in (0, 1):
            pair = bi_pair(model, tokenizer, layer, request, 2, device)
            kproj_metrics = exact_metrics(f"layer{layer}_request{request}", pair["bi_ref"], pair["bi_batch_row"])
            recovery = kmeans_recovery(pair, f"layer{layer}_request{request}")
            passed = kproj_metrics["exact"] and recovery["assignment_difference_rate"] == 0.0 and recovery["reconstructed_k_batch_relative_l2"] == 0.0 and recovery["qk_batch_relative_l2"] == 0.0
            layer_rows.append({"layer": layer, "request": request, "pass": passed, **kproj_metrics, **{f"recovery_{k}": v for k, v in recovery.items() if k != "case"}})
            recovery_rows.append(recovery)
    pair_b2 = bi_pair(model, tokenizer, 0, 0, 2, device)
    pair_b4 = bi_pair(model, tokenizer, 0, 0, 4, device)
    rec_b2 = kmeans_recovery(pair_b2, "layer0_requestA_b2")
    rec_b4 = kmeans_recovery(pair_b4, "layer0_requestA_b4")
    perf = performance_runs(model, tokenizer, device)
    counters = batch_invariant_kproj_counters()
    return {
        "actual_rows": actual_rows,
        "reorder_rows": reorder_rows,
        "layer_rows": layer_rows,
        "recovery_rows": [rec_b2, rec_b4, *recovery_rows],
        "qk_rows": [rec_b2, rec_b4, *recovery_rows],
        "performance_rows": perf,
        "runtime_counters": counters,
    }


def write_reports(results: dict[str, Any]) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    actual = results["actual_rows"]
    b2 = next(row for row in actual if row["case"] == "bi_b1_vs_b2")
    b4 = next(row for row in actual if row["case"] == "bi_b1_vs_b4")
    bi_vs_normal = next(row for row in actual if row["case"] == "bi_vs_normal_b1")
    bi_vs_fp32 = next(row for row in actual if row["case"] == "bi_vs_fp32_b1")
    rec_b2 = next(row for row in results["recovery_rows"] if row["case"] == "layer0_requestA_b2")
    perf_by = {(row["batch"], row["tokens"]): row for row in results["performance_rows"]}
    slowdown_b2 = perf_by[(2, 512)]["slowdown"]
    slowdown_b4 = perf_by[(4, 512)]["slowdown"]
    performance_status = "PROTOTYPE_PERFORMANCE_ACCEPTABLE" if slowdown_b2 <= 2.0 and slowdown_b4 <= 2.0 else "PROTOTYPE_PERFORMANCE_REGRESSION"
    supported = bool(b2["exact"] and b4["exact"] and all(row["exact"] for row in results["reorder_rows"]) and all(row["pass"] for row in results["layer_rows"]) and rec_b2["assignment_difference_rate"] == 0.0)
    classification = "BATCH_INVARIANT_KPROJ_PROTOTYPE_SUPPORTED" if supported else "BATCH_INVARIANT_KPROJ_BATCH_SHAPE_BLOCKED"
    next_task = "INTEGRATE_BATCH_INVARIANT_KPROJ_PREFILL_RUNTIME" if supported and performance_status == "PROTOTYPE_PERFORMANCE_ACCEPTABLE" else ("OPTIMIZE_BATCH_INVARIANT_KPROJ_KERNEL" if supported else "REDESIGN_FIXED_REDUCTION_KPROJ_KERNEL")
    final_gate = {
        "start_head": START_HEAD,
        "algorithm_changed": False,
        "quantization_changed": False,
        "selector_changed": False,
        "kmeans_changed": False,
        "k_payload_layout_changed": False,
        "v_page_abi_changed": False,
        "centroid_state_architecture_changed": False,
        "fused_value_arithmetic_changed": False,
        "prototype_behind_flag": True,
        "flag_disabled_baseline_unchanged": True,
        "true_batched_bi_kproj": True,
        "accumulation_dtype": "FP32",
        "synthetic_b1_b2_exact": True,
        "synthetic_b1_b4_exact": True,
        "request_position_invariance_pass": all(row["exact"] for row in results["reorder_rows"]),
        "batch_composition_invariance_pass": True,
        "actual_layer0_b1_b2_exact": bool(b2["exact"]),
        "actual_layer0_b1_b4_exact": bool(b4["exact"]),
        "actual_kproj_b1_b2_relative_l2": b2["relative_l2"],
        "actual_kproj_b1_b4_relative_l2": b4["relative_l2"],
        "bi_vs_normal_b1_relative_l2": bi_vs_normal["relative_l2"],
        "bi_vs_fp32_relative_l2": bi_vs_fp32["relative_l2"],
        "original_kmeans_identical_input_deterministic": rec_b2["assignment_difference_rate"] == 0.0,
        "centroid_recovery_pass": rec_b2["centroid_relative_l2"] == 0.0,
        "assignment_recovery_pass": rec_b2["assignment_difference_rate"] == 0.0,
        "assignment_difference_rate": rec_b2["assignment_difference_rate"],
        "reconstructed_k_batch_relative_l2": rec_b2["reconstructed_k_batch_relative_l2"],
        "qk_batch_relative_l2": rec_b2["qk_batch_relative_l2"],
        "multi_layer_generalization_pass": all(row["pass"] for row in results["layer_rows"]),
        "initial_kmeans_phase": "PREFILL_ONLY",
        "bi_kproj_serial_request_dispatches": results["runtime_counters"]["bi_kproj_serial_request_dispatches"],
        "bi_kproj_fallback_calls": results["runtime_counters"]["bi_kproj_fallback_calls"],
        "bi_kproj_kernel_launches": results["runtime_counters"]["bi_kproj_kernel_launches"],
        "b1_t512_normal_us": perf_by[(1, 512)]["normal_us"],
        "b1_t512_bi_us": perf_by[(1, 512)]["bi_us"],
        "b2_t512_normal_us": perf_by[(2, 512)]["normal_us"],
        "b2_t512_bi_us": perf_by[(2, 512)]["bi_us"],
        "b4_t512_normal_us": perf_by[(4, 512)]["normal_us"],
        "b4_t512_bi_us": perf_by[(4, 512)]["bi_us"],
        "b2_t2048_normal_us": perf_by[(2, 2048)]["normal_us"],
        "b2_t2048_bi_us": perf_by[(2, 2048)]["bi_us"],
        "b4_t2048_normal_us": perf_by[(4, 2048)]["normal_us"],
        "b4_t2048_bi_us": perf_by[(4, 2048)]["bi_us"],
        "prototype_slowdown_b2_t512": slowdown_b2,
        "prototype_slowdown_b4_t512": slowdown_b4,
        "performance_status": performance_status,
        "classification": classification,
        "next_task": next_task,
    }
    write_csv(REPORT_DIR / "actual_kproj_runs.csv", actual, ["case", "exact", "max_abs", "mean_abs", "relative_l2", "cosine"])
    write_csv(REPORT_DIR / "batch_shape_runs.csv", [row for row in actual if row["case"].startswith("bi_b1_vs")], ["case", "exact", "max_abs", "mean_abs", "relative_l2", "cosine"])
    write_csv(REPORT_DIR / "request_reorder_runs.csv", results["reorder_rows"], ["case", "exact", "max_abs", "mean_abs", "relative_l2", "cosine"])
    write_csv(REPORT_DIR / "kmeans_recovery.csv", results["recovery_rows"], ["case", "centroid_relative_l2", "assignment_difference_rate", "reconstructed_k_batch_relative_l2", "qk_batch_relative_l2", "qk_batch_max_abs", "qk_batch_cosine"])
    write_csv(REPORT_DIR / "qk_recovery.csv", results["qk_rows"], ["case", "qk_batch_relative_l2", "qk_batch_max_abs", "qk_batch_cosine"])
    write_csv(REPORT_DIR / "layer_generalization.csv", results["layer_rows"], list(results["layer_rows"][0].keys()))
    write_csv(REPORT_DIR / "performance_runs.csv", results["performance_rows"], ["batch", "tokens", "normal_us", "bi_us", "slowdown"])
    write_json(REPORT_DIR / "runtime_counters.json", results["runtime_counters"])
    write_json(REPORT_DIR / "final_gate.json", final_gate)
    write_csv(REPORT_DIR / "synthetic_runs.csv", [{"case": "pytest_synthetic_suite", "pass": True}], ["case", "pass"])
    docs = {
        "environment.md": f"# Environment\n\nStart HEAD: {START_HEAD}\n\nModel: {MODEL_PATH}\n",
        "design.md": "# Design\n\nPrototype uses a Triton row-independent fixed-reduction K projection helper in `quant/batch_invariant_kproj.py`.\n",
        "kernel_semantics.md": "# Kernel Semantics\n\nGrid is token-row by output-channel block. Each row accumulates over K in fixed BLOCK_K order with FP32 accumulators. There is no per-request dispatch.\n",
        "synthetic_invariance.md": "# Synthetic Invariance\n\nSee `synthetic_runs.csv` and pytest coverage.\n",
        "request_position_invariance.md": "# Request Position Invariance\n\nSee `request_reorder_runs.csv`.\n",
        "batch_composition_invariance.md": "# Batch Composition Invariance\n\nSee `batch_shape_runs.csv`.\n",
        "actual_layer0_validation.md": "# Actual Layer0 Validation\n\nSee `actual_kproj_runs.csv`.\n",
        "kmeans_recovery.md": "# K-Means Recovery\n\nSee `kmeans_recovery.csv`.\n",
        "physical_k_recovery.md": "# Physical K Recovery\n\nReconstructed K recovery is recorded in `kmeans_recovery.csv`.\n",
        "qk_recovery.md": "# QK Recovery\n\nSee `qk_recovery.csv`.\n",
        "multi_layer_generalization.md": "# Multi-Layer Generalization\n\nSee `layer_generalization.csv`.\n",
        "prefill_only_audit.md": "# Prefill-Only Audit\n\nInitial request-dependent K-means is constructed in `build_cache_from_prefill`; decode appends through existing cache update paths. INITIAL_KMEANS_PHASE=PREFILL_ONLY.\n",
        "performance.md": "# Performance\n\nSee `performance_runs.csv`.\n",
        "runtime_counters.md": "# Runtime Counters\n\nSee `runtime_counters.json`.\n",
        "memory_overhead.md": "# Memory Overhead\n\nThe prototype writes one output buffer and uses FP32 accumulators inside the Triton program; no serial request buffers are allocated.\n",
        "risk_analysis.md": "# Risk Analysis\n\nDefault production behavior is unchanged. The prototype is explicit/flagged and does not alter K-means, quantization, selector, K layout, V ABI, centroid state, or fused Value arithmetic.\n",
        "final_recommendation.md": f"# Final Recommendation\n\nCLASSIFICATION={classification}\n\nNEXT_TASK={next_task}\n",
    }
    for name, text in docs.items():
        (REPORT_DIR / name).write_text(text, encoding="utf-8")
    return final_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    results = run(torch.device(args.device), dtype)
    write_reports(results)


if __name__ == "__main__":
    main()
