from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.run_actual_model_fixed_batch_smoke import MODEL_PATH, load_model, make_fixed_inputs
from bench.run_actual_model_k_assignment_trace import make_position_ids, tensor_metrics
from bench.run_batch_invariant_kproj_prototype import exact_metrics, kmeans_recovery
from bench.run_serving_stable_k_centroid_eval import evaluate_candidate, hidden_for_layers
from models.llama_patternkv import apply_rotary_pos_emb
from quant.batch_invariant_kproj import (
    batch_invariant_k_projection,
    batch_invariant_k_projection_v1,
    batch_invariant_k_projection_v2,
    batch_invariant_kproj_counters,
    reset_batch_invariant_kproj_counters,
)


REPORT_DIR = REPO_ROOT / "reports/system_batch_invariant_kproj_v2"
START_HEAD = "96166290cba9e31b653c11ee18ee927e20e4187b"
LAYERS = [0, 8, 16, 31]
CONFIGS = [
    {"config": "A", "block_m": 128, "block_n": 256, "block_k": 64, "group_m": 8, "num_warps": 8, "num_stages": 3},
    {"config": "B", "block_m": 128, "block_n": 128, "block_k": 64, "group_m": 8, "num_warps": 8, "num_stages": 3},
    {"config": "C", "block_m": 64, "block_n": 128, "block_k": 64, "group_m": 8, "num_warps": 4, "num_stages": 3},
    {"config": "D", "block_m": 64, "block_n": 256, "block_k": 64, "group_m": 8, "num_warps": 8, "num_stages": 3},
    {"config": "E", "block_m": 32, "block_n": 128, "block_k": 64, "group_m": 8, "num_warps": 4, "num_stages": 3},
]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def v2(x: torch.Tensor, w: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    return batch_invariant_k_projection_v2(x, w, block_m=cfg["block_m"], block_n=cfg["block_n"], block_k=cfg["block_k"], group_m=cfg["group_m"], num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])


def synthetic_screen() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    summary = []
    shapes = [(1, 4096, 1024), (17, 4096, 1024), (129, 4096, 1024), (513, 4096, 1024), (65, 73, 257), (257, 511, 128)]
    for cfg in CONFIGS:
        exact_all = True
        rels = []
        for tokens, hidden, out in shapes:
            g = torch.Generator(device="cuda")
            g.manual_seed(1000 + tokens + hidden + out)
            x = torch.randn((4, tokens, hidden), device="cuda", dtype=torch.float16, generator=g)
            w = torch.randn((out, hidden), device="cuda", dtype=torch.float16, generator=g)
            ref = v2(x[0:1], w, cfg)
            b2 = v2(x[:2], w, cfg)[0:1]
            b4 = v2(x, w, cfg)[0:1]
            reorder = v2(torch.cat([x[1:2], x[0:1]], dim=0), w, cfg)[1:2]
            composition = v2(torch.cat([x[0:1], x[2:4]], dim=0), w, cfg)[0:1]
            fp32 = torch.matmul(x[0:1].float(), w.float().t()).to(torch.float16)
            rel = float((torch.linalg.vector_norm((ref - fp32).float()) / torch.linalg.vector_norm(fp32.float()).clamp_min(1e-12)).item())
            rels.append(rel)
            case_exact = torch.equal(ref, b2) and torch.equal(ref, b4) and torch.equal(ref, reorder) and torch.equal(ref, composition)
            exact_all = exact_all and case_exact
            rows.append({"config": cfg["config"], "tokens": tokens, "hidden": hidden, "out": out, "b1_b2_exact": torch.equal(ref, b2), "b1_b4_exact": torch.equal(ref, b4), "request_reorder_exact": torch.equal(ref, reorder), "batch_composition_exact": torch.equal(ref, composition), "v2_vs_fp32_relative_l2": rel})
        summary.append({**cfg, "exact_all": exact_all, "max_v2_vs_fp32_relative_l2": max(rels)})
    return rows, summary


def layer_hidden(model: Any, ids: torch.Tensor, layer: int) -> torch.Tensor:
    return hidden_for_layers(model, ids, [layer])[layer]


def normal_and_bi(model: Any, hidden: torch.Tensor, layer_idx: int, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, float]:
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    with torch.inference_mode():
        normed = layer.input_layernorm(hidden)
        normal = attn.k_proj(normed)
        bi = v2(normed, attn.k_proj.weight, cfg)
        bsz, seq_len, _ = normed.shape
        key = bi.view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        q_proj = attn.q_proj(normed)
        query = q_proj.view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        pos = make_position_ids(torch.empty((bsz, seq_len), dtype=torch.long, device=hidden.device))
        cos, sin = attn.rotary_emb(key, pos)
        query_post, key_post = apply_rotary_pos_emb(query, key, cos, sin, pos)
    return normal.detach(), bi.detach(), key_post.detach(), query_post.detach(), int(attn.num_k_bases), int(attn.group_size), float(attn.head_dim**-0.5)


def bi_pair(model: Any, tokenizer: Any, layer_idx: int, request: int, batch: int, cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    ids = make_fixed_inputs(tokenizer, 4, 512, device)
    row = request if request < batch else 0
    h_ref = layer_hidden(model, ids[request : request + 1], layer_idx)
    h_batch = torch.cat([layer_hidden(model, ids[(request if idx == row else idx if idx != request else 0) : (request if idx == row else idx if idx != request else 0) + 1], layer_idx) for idx in range(batch)], dim=0)
    _normal_ref, bi_ref, k_ref, q_ref, bases, group, scale = normal_and_bi(model, h_ref, layer_idx, cfg)
    _normal_b, bi_b, k_b, _q_b, _bases, _group, _scale = normal_and_bi(model, h_batch, layer_idx, cfg)
    return {"k_ref": k_ref, "k_b2": k_b[row : row + 1], "q_ref": q_ref, "num_k_bases": bases, "group_size": group, "qk_scale": scale, "bi_ref": bi_ref, "bi_batch_row": bi_b[row : row + 1]}


def performance(model: Any, tokenizer: Any, cfg: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    rows = []
    layer = model.model.layers[0]
    for batch, tokens in [(1, 512), (2, 512), (4, 512), (1, 2048), (2, 2048), (4, 2048)]:
        ids = make_fixed_inputs(tokenizer, batch, tokens, device)
        hidden = layer.input_layernorm(model.model.embed_tokens(ids))
        for _ in range(10):
            layer.self_attn.k_proj(hidden)
            batch_invariant_k_projection_v1(hidden, layer.self_attn.k_proj.weight)
            v2(hidden, layer.self_attn.k_proj.weight, cfg)
        torch.cuda.synchronize(device)
        def measure(fn):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(50):
                fn()
            end.record()
            torch.cuda.synchronize(device)
            return start.elapsed_time(end) * 1000.0 / 50
        torch_us = measure(lambda: layer.self_attn.k_proj(hidden))
        v1_us = measure(lambda: batch_invariant_k_projection_v1(hidden, layer.self_attn.k_proj.weight))
        v2_us = measure(lambda: v2(hidden, layer.self_attn.k_proj.weight, cfg))
        flops = 2.0 * batch * tokens * hidden.shape[-1] * layer.self_attn.k_proj.weight.shape[0]
        rows.append({"batch": batch, "tokens": tokens, "torch_us": torch_us, "v1_us": v1_us, "v2_us": v2_us, "v2_speedup_vs_v1": v1_us / v2_us if v2_us else None, "v2_slowdown_vs_torch": v2_us / torch_us if torch_us else None, "v2_approx_tflops": flops / (v2_us * 1e-6) / 1e12})
    return rows


def config_performance(model: Any, tokenizer: Any, summaries: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    rows = []
    ids = make_fixed_inputs(tokenizer, 2, 512, device)
    hidden = model.model.layers[0].input_layernorm(model.model.embed_tokens(ids))
    for cfg in summaries:
        if not cfg["exact_all"]:
            continue
        for _ in range(5):
            v2(hidden, model.model.layers[0].self_attn.k_proj.weight, cfg)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            v2(hidden, model.model.layers[0].self_attn.k_proj.weight, cfg)
        end.record()
        torch.cuda.synchronize(device)
        rows.append({**cfg, "b2_t512_v2_us": start.elapsed_time(end) * 1000.0 / 20})
    return rows


def run(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    synthetic_rows, config_summary = synthetic_screen()
    tokenizer, _config, model = load_model(dtype, device)
    cfg_perf = config_performance(model, tokenizer, config_summary, device)
    best_cfg = min(cfg_perf, key=lambda row: row["b2_t512_v2_us"])
    reset_batch_invariant_kproj_counters()
    actual_rows = []
    for batch in (2, 4):
        pair = bi_pair(model, tokenizer, 0, 0, batch, best_cfg, device)
        actual_rows.append(exact_metrics(f"layer0_requestA_b{batch}", pair["bi_ref"], pair["bi_batch_row"]))
    reorder_pair = bi_pair(model, tokenizer, 0, 1, 2, best_cfg, device)
    actual_rows.append(exact_metrics("layer0_requestB_b2", reorder_pair["bi_ref"], reorder_pair["bi_batch_row"]))
    rec_b2 = kmeans_recovery(bi_pair(model, tokenizer, 0, 0, 2, best_cfg, device), "layer0_requestA_b2")
    layer_rows = []
    recovery_rows = [rec_b2]
    for layer in LAYERS:
        for request in (0, 1):
            pair = bi_pair(model, tokenizer, layer, request, 2, best_cfg, device)
            metrics = exact_metrics(f"layer{layer}_request{request}", pair["bi_ref"], pair["bi_batch_row"])
            rec = kmeans_recovery(pair, f"layer{layer}_request{request}")
            passed = metrics["exact"] and rec["assignment_difference_rate"] == 0.0 and rec["reconstructed_k_batch_relative_l2"] == 0.0 and rec["qk_batch_relative_l2"] == 0.0
            layer_rows.append({"layer": layer, "request": request, "pass": passed, **metrics, **{f"recovery_{k}": v for k, v in rec.items() if k != "case"}})
            recovery_rows.append(rec)
    perf_rows = performance(model, tokenizer, best_cfg, device)
    return {"synthetic_rows": synthetic_rows, "config_summary": config_summary, "config_performance": cfg_perf, "best_config": best_cfg, "actual_rows": actual_rows, "kmeans_rows": recovery_rows, "layer_rows": layer_rows, "performance_rows": perf_rows, "runtime_counters": batch_invariant_kproj_counters()}


def write_reports(results: dict[str, Any]) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    best = results["best_config"]
    perf_by = {(row["batch"], row["tokens"]): row for row in results["performance_rows"]}
    layer0_b2 = next(row for row in results["actual_rows"] if row["case"] == "layer0_requestA_b2")
    layer0_b4 = next(row for row in results["actual_rows"] if row["case"] == "layer0_requestA_b4")
    rec = next(row for row in results["kmeans_rows"] if row["case"] == "layer0_requestA_b2")
    multi_pass = all(row["pass"] for row in results["layer_rows"])
    b2_slow = perf_by[(2, 512)]["v2_slowdown_vs_torch"]
    b4_slow = perf_by[(4, 512)]["v2_slowdown_vs_torch"]
    b2_speed = perf_by[(2, 512)]["v2_speedup_vs_v1"]
    if not all(row["exact_all"] for row in results["config_summary"]):
        classification = "PERSISTENT_V2_INVARIANCE_BLOCKED"
        performance_status = "CONFIG_INVARIANCE_FAIL"
        next_task = "REDESIGN_TILED_FIXED_REDUCTION_SEMANTICS"
    elif not (layer0_b2["exact"] and layer0_b4["exact"] and multi_pass):
        classification = "PERSISTENT_V2_INVARIANCE_BLOCKED"
        performance_status = "ACTUAL_INVARIANCE_FAIL"
        next_task = "REDESIGN_TILED_FIXED_REDUCTION_SEMANTICS"
    elif rec["assignment_difference_rate"] != 0.0 or rec["reconstructed_k_batch_relative_l2"] != 0.0 or rec["qk_batch_relative_l2"] != 0.0:
        classification = "PERSISTENT_V2_KMEANS_RECOVERY_REGRESSION"
        performance_status = "KMEANS_RECOVERY_FAIL"
        next_task = "REDESIGN_TILED_FIXED_REDUCTION_SEMANTICS"
    elif b2_speed < 2.0:
        classification = "PERSISTENT_V2_STRUCTURE_NOT_EFFECTIVE"
        performance_status = "V2_STRUCTURE_NOT_EFFECTIVE"
        next_task = "PROFILE_BI_KPROJ_MEMORY_AND_COMPUTE_EFFICIENCY"
    else:
        classification = "BATCH_INVARIANT_KPROJ_V2_SUPPORTED"
        if b2_slow <= 2.0 and b4_slow <= 2.0:
            performance_status = "PRODUCTION_FEASIBILITY_SUPPORTED"
            next_task = "INTEGRATE_BATCH_INVARIANT_KPROJ_PREFILL_RUNTIME"
        elif b2_slow <= 3.0 and b4_slow <= 3.0:
            performance_status = "PERSISTENT_OPTIMIZATION_SUPPORTED"
            next_task = "INTEGRATE_BATCH_INVARIANT_KPROJ_PREFILL_RUNTIME"
        else:
            performance_status = "FURTHER_KERNEL_OPTIMIZATION_REQUIRED"
            next_task = "PROFILE_AND_OPTIMIZE_BI_KPROJ_V2"
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
        "v1_preserved": True,
        "v2_persistent_implemented": True,
        "v2_uses_tl_dot": True,
        "v2_block_m": best["block_m"],
        "v2_block_n": best["block_n"],
        "v2_block_k": best["block_k"],
        "v2_group_m": best["group_m"],
        "v2_num_warps": best["num_warps"],
        "v2_num_stages": best["num_stages"],
        "tl_dot_precision_mode": "ieee",
        "persistent_grid_used": True,
        "num_sms": torch.cuda.get_device_properties(0).multi_processor_count,
        "synthetic_b1_b2_exact": all(row["b1_b2_exact"] for row in results["synthetic_rows"]),
        "synthetic_b1_b4_exact": all(row["b1_b4_exact"] for row in results["synthetic_rows"]),
        "request_reorder_pass": all(row["request_reorder_exact"] for row in results["synthetic_rows"]),
        "batch_composition_pass": all(row["batch_composition_exact"] for row in results["synthetic_rows"]),
        "actual_layer0_b1_b2_exact": layer0_b2["exact"],
        "actual_layer0_b1_b4_exact": layer0_b4["exact"],
        "assignment_difference_rate": rec["assignment_difference_rate"],
        "reconstructed_k_batch_relative_l2": rec["reconstructed_k_batch_relative_l2"],
        "qk_batch_relative_l2": rec["qk_batch_relative_l2"],
        "multi_layer_generalization_pass": multi_pass,
        "bi_vs_fp32_relative_l2": best["max_v2_vs_fp32_relative_l2"],
        "serial_request_dispatches": results["runtime_counters"]["bi_kproj_serial_request_dispatches"],
        "fallback_calls": results["runtime_counters"]["bi_kproj_fallback_calls"],
        "weight_transpose_copy_bytes": results["runtime_counters"]["bi_kproj_weight_copy_bytes"],
        "input_copy_bytes": results["runtime_counters"]["bi_kproj_input_copy_bytes"],
        "b2_t512_torch_us": perf_by[(2, 512)]["torch_us"],
        "b2_t512_v1_us": perf_by[(2, 512)]["v1_us"],
        "b2_t512_v2_us": perf_by[(2, 512)]["v2_us"],
        "b4_t512_torch_us": perf_by[(4, 512)]["torch_us"],
        "b4_t512_v1_us": perf_by[(4, 512)]["v1_us"],
        "b4_t512_v2_us": perf_by[(4, 512)]["v2_us"],
        "b2_t2048_torch_us": perf_by[(2, 2048)]["torch_us"],
        "b2_t2048_v1_us": perf_by[(2, 2048)]["v1_us"],
        "b2_t2048_v2_us": perf_by[(2, 2048)]["v2_us"],
        "b4_t2048_torch_us": perf_by[(4, 2048)]["torch_us"],
        "b4_t2048_v1_us": perf_by[(4, 2048)]["v1_us"],
        "b4_t2048_v2_us": perf_by[(4, 2048)]["v2_us"],
        "v2_speedup_vs_v1_b2_t512": perf_by[(2, 512)]["v2_speedup_vs_v1"],
        "v2_speedup_vs_v1_b4_t512": perf_by[(4, 512)]["v2_speedup_vs_v1"],
        "v2_slowdown_vs_torch_b2_t512": b2_slow,
        "v2_slowdown_vs_torch_b4_t512": b4_slow,
        "performance_status": performance_status,
        "classification": classification,
        "next_task": next_task,
    }
    write_csv(REPORT_DIR / "synthetic_invariance.csv", results["synthetic_rows"], ["config", "tokens", "hidden", "out", "b1_b2_exact", "b1_b4_exact", "request_reorder_exact", "batch_composition_exact", "v2_vs_fp32_relative_l2"])
    write_csv(REPORT_DIR / "config_invariance.csv", results["config_summary"], ["config", "block_m", "block_n", "block_k", "group_m", "num_warps", "num_stages", "exact_all", "max_v2_vs_fp32_relative_l2"])
    write_csv(REPORT_DIR / "config_performance.csv", results["config_performance"], ["config", "block_m", "block_n", "block_k", "group_m", "num_warps", "num_stages", "exact_all", "max_v2_vs_fp32_relative_l2", "b2_t512_v2_us"])
    write_csv(REPORT_DIR / "actual_kproj_runs.csv", results["actual_rows"], ["case", "exact", "max_abs", "mean_abs", "relative_l2", "cosine"])
    write_csv(REPORT_DIR / "kmeans_recovery.csv", results["kmeans_rows"], ["case", "centroid_relative_l2", "assignment_difference_rate", "reconstructed_k_batch_relative_l2", "qk_batch_relative_l2", "qk_batch_max_abs", "qk_batch_cosine"])
    write_csv(REPORT_DIR / "layer_generalization.csv", results["layer_rows"], list(results["layer_rows"][0].keys()))
    write_csv(REPORT_DIR / "performance_comparison.csv", results["performance_rows"], ["batch", "tokens", "torch_us", "v1_us", "v2_us", "v2_speedup_vs_v1", "v2_slowdown_vs_torch", "v2_approx_tflops"])
    profiling = {
        "kernel_launches_per_projection": 1,
        "num_sms": final_gate["num_sms"],
        "persistent_grid_used": True,
        "best_config": best,
        "weight_transpose_copy_bytes": final_gate["weight_transpose_copy_bytes"],
        "input_copy_bytes": final_gate["input_copy_bytes"],
        "output_bytes_b2_t512": 2 * 512 * 1024 * 2,
    }
    write_json(REPORT_DIR / "profiling.json", profiling)
    write_json(REPORT_DIR / "runtime_counters.json", results["runtime_counters"])
    write_json(REPORT_DIR / "final_gate.json", final_gate)
    docs = {
        "environment.md": f"# Environment\n\nStart HEAD: {START_HEAD}\n\nModel: {MODEL_PATH}\n\nGPU: RTX 3090 / SM86 target.\n",
        "v1_baseline.md": "# V1 Baseline\n\nV1 row-wise fixed-reduction kernel is preserved as `batch_invariant_k_projection_v1`.\n",
        "v2_design.md": "# V2 Design\n\nV2 uses persistent tiled Triton GEMM with BLOCK_M x BLOCK_N output tiles and fixed BLOCK_K reduction.\n",
        "persistent_schedule.md": "# Persistent Schedule\n\nGrid is capped at NUM_SMS and each program processes tile_id += NUM_SMS in deterministic grouped-M order.\n",
        "tl_dot_semantics.md": "# TL Dot Semantics\n\nV2 uses `tl.dot(..., input_precision=\"ieee\")` with FP32 accumulation. K tiling is fixed and independent of batch size.\n",
        "synthetic_invariance.md": "# Synthetic Invariance\n\nSee `synthetic_invariance.csv`.\n",
        "config_screening.md": "# Config Screening\n\nSee `config_invariance.csv` and `config_performance.csv`.\n",
        "actual_layer0_validation.md": "# Actual Layer0 Validation\n\nSee `actual_kproj_runs.csv`.\n",
        "kmeans_recovery.md": "# K-Means Recovery\n\nOriginal K-means is unchanged. See `kmeans_recovery.csv`.\n",
        "multi_layer_validation.md": "# Multi-Layer Validation\n\nSee `layer_generalization.csv`.\n",
        "performance.md": "# Performance\n\nSee `performance_comparison.csv`.\n",
        "profiling.md": "# Profiling\n\nSee `profiling.json`.\n",
        "memory_traffic.md": "# Memory Traffic\n\nWeight transpose copy bytes and input copy bytes are recorded in `final_gate.json` and `profiling.json`.\n",
        "runtime_counters.md": "# Runtime Counters\n\nSee `runtime_counters.json`.\n",
        "risk_analysis.md": "# Risk Analysis\n\nProduction behavior is not changed. K-means, quantization, selector, K layout, V ABI, centroid architecture, and fused Value arithmetic remain frozen.\n",
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
