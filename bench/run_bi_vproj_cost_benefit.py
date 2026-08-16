from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from quant.batch_invariant_kproj import batch_invariant_kproj_counters, reset_batch_invariant_kproj_counters


START_HEAD = "9b498291fc2686f538810db236e99899c48e1cfa"
REPORT_DIR = Path("reports/system_bi_vproj_cost_benefit_v1")
VARIANTS = {
    "p0": "normal",
    "p1": "bi_k",
    "p2": "bi_kv",
}
FINAL_GATE_TEMPLATE: dict[str, Any] = {
    "start_head": START_HEAD,
    "actual_model_loaded": False,
    "algorithm_changed": False,
    "quantization_changed": False,
    "selector_changed": False,
    "kmeans_changed": False,
    "v_page_abi_changed": False,
    "centroid_state_architecture_changed": False,
    "fused_value_arithmetic_changed": False,
    "p0_available": None,
    "p1_available": None,
    "p2_available": None,
    "p2_uses_existing_bi_v2_kernel": None,
    "p2_prefill_k_bi": None,
    "p2_prefill_v_bi": None,
    "p2_decode_k_bi": None,
    "p2_decode_v_bi": None,
    "p2_serial_request_dispatches": None,
    "p2_fallback_calls": None,
    "p2_b1_b2_kproj_exact": None,
    "p2_b1_b2_vproj_exact": None,
    "p2_b1_b4_kproj_exact": None,
    "p2_b1_b4_vproj_exact": None,
    "p2_k_assignment_difference_rate": None,
    "p2_v_assignment_difference_rate": None,
    "p2_v_mask_difference_rate": None,
    "p2_v_precision_mask_difference_rate": None,
    "p2_packed_v_difference_rate": None,
    "p1_b2_ctx512_final_hidden_rel_l2": None,
    "p2_b2_ctx512_final_hidden_rel_l2": None,
    "p1_b4_ctx512_final_hidden_rel_l2": None,
    "p2_b4_ctx512_final_hidden_rel_l2": None,
    "p1_b2_ctx512_logit_rel_l2": None,
    "p2_b2_ctx512_logit_rel_l2": None,
    "p1_b4_ctx512_logit_rel_l2": None,
    "p2_b4_ctx512_logit_rel_l2": None,
    "p1_argmax_match_rate": None,
    "p2_argmax_match_rate": None,
    "hidden_drift_reduction_ratio_b2": None,
    "hidden_drift_reduction_ratio_b4": None,
    "logit_drift_reduction_ratio_b2": None,
    "logit_drift_reduction_ratio_b4": None,
    "layerwise_accumulation_observed_p1": None,
    "layerwise_accumulation_observed_p2": None,
    "p1_b2_ctx512_prefill_ms": None,
    "p2_b2_ctx512_prefill_ms": None,
    "p1_b4_ctx512_prefill_ms": None,
    "p2_b4_ctx512_prefill_ms": None,
    "p1_b2_ctx2048_prefill_ms": None,
    "p2_b2_ctx2048_prefill_ms": None,
    "p1_b4_ctx2048_prefill_ms": None,
    "p2_b4_ctx2048_prefill_ms": None,
    "p1_b2_ctx4096_prefill_ms": None,
    "p2_b2_ctx4096_prefill_ms": None,
    "p1_b4_ctx4096_prefill_ms": None,
    "p2_b4_ctx4096_prefill_ms": None,
    "p2_overhead_percent_b2_ctx512": None,
    "p2_overhead_percent_b4_ctx512": None,
    "p2_overhead_percent_b2_ctx2048": None,
    "p2_overhead_percent_b4_ctx2048": None,
    "p2_overhead_percent_b2_ctx4096": None,
    "p2_overhead_percent_b4_ctx4096": None,
    "p1_peak_memory_bytes": None,
    "p2_peak_memory_bytes": None,
    "fused_page_operator_preserved": None,
    "legacy_value_calls": None,
    "historical_v_materialization_bytes": None,
    "cost_level": "",
    "numerical_benefit_level": "",
    "classification": "",
    "architecture_recommendation": "",
    "next_task": "",
}


def drift_reduction_ratio(p1: float | None, p2: float | None) -> float | None:
    if p1 is None or p2 is None:
        return None
    if p2 == 0:
        return float("inf")
    return float(p1) / float(p2)


def prefill_overhead_percent(p1_ms: float | None, p2_ms: float | None) -> float | None:
    if p1_ms is None or p2_ms is None or p1_ms == 0:
        return None
    return (float(p2_ms) / float(p1_ms) - 1.0) * 100.0


def classify_cost_benefit(
    *,
    overhead_percent: float | None,
    logit_drift_reduction_ratio: float | None,
    argmax_fixed: bool = False,
    correctness_failed: bool = False,
) -> tuple[str, str]:
    if correctness_failed:
        return "BI_KV_PREFILL_RUNTIME_CORRECTNESS_FAILED", "TRACE_BI_VPROJ_RUNTIME_INTEGRATION"
    if overhead_percent is None or logit_drift_reduction_ratio is None:
        return "BI_VPROJ_COST_BENEFIT_INCONCLUSIVE", ""
    strong_benefit = argmax_fixed or logit_drift_reduction_ratio >= 2.0
    if overhead_percent <= 2.0 and not strong_benefit:
        return "BI_VPROJ_LOW_COST_OPTIONAL", "DESIGN_PREFILL_PROJECTION_MODE_POLICY"
    if overhead_percent <= 5.0 and strong_benefit:
        return "BI_VPROJ_COST_BENEFIT_SUPPORTED", "INTEGRATE_BATCH_INVARIANT_KVPROJ_PREFILL_RUNTIME"
    if overhead_percent <= 10.0 and strong_benefit:
        return "BI_VPROJ_COST_BENEFIT_CONDITIONAL", "INTEGRATE_BATCH_INVARIANT_KVPROJ_PREFILL_RUNTIME"
    if overhead_percent > 5.0 and logit_drift_reduction_ratio < 1.2:
        return "BI_VPROJ_COST_BENEFIT_NOT_SUPPORTED", "REDEFINE_FIXED_BATCH_STATE_EQUIVALENCE_GATE"
    return "BI_VPROJ_COST_BENEFIT_INCONCLUSIVE", ""


def tensor_metric_dict(ref: torch.Tensor, got: torch.Tensor) -> dict[str, float | bool]:
    ref_f = ref.detach().float()
    got_f = got.detach().float()
    diff = got_f - ref_f
    ref_norm = torch.linalg.vector_norm(ref_f).item()
    rel_l2 = torch.linalg.vector_norm(diff).item() / max(ref_norm, 1e-12)
    cosine = torch.nn.functional.cosine_similarity(ref_f.reshape(1, -1), got_f.reshape(1, -1), dim=-1).item()
    return {
        "exact": bool(torch.equal(ref, got)),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "relative_l2": float(rel_l2),
        "cosine": float(cosine),
    }


def logits_metric_dict(ref: torch.Tensor, got: torch.Tensor) -> dict[str, Any]:
    metrics = tensor_metric_dict(ref, got)
    ref_top = torch.topk(ref.detach().float(), k=5, dim=-1)
    got_top = torch.topk(got.detach().float(), k=5, dim=-1)
    ref_top1 = int(ref_top.indices[0, 0].item())
    got_top1 = int(got_top.indices[0, 0].item())
    ref_top5 = {int(x) for x in ref_top.indices[0].tolist()}
    got_top5 = {int(x) for x in got_top.indices[0].tolist()}
    margin = float((got_top.values[0, 0] - got_top.values[0, 1]).item())
    metrics.update(
        {
            "argmax_same": ref_top1 == got_top1,
            "ref_top1": ref_top1,
            "got_top1": got_top1,
            "top5_overlap": len(ref_top5 & got_top5),
            "top1_margin": margin,
        }
    )
    return metrics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n")


def _nvidia_smi() -> str:
    try:
        return subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=10)
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def _set_variant_env(variant: str) -> None:
    os.environ["PATTERNKV_PREFILL_PROJ_MODE"] = VARIANTS[variant]
    os.environ["PATTERNKV_BI_KPROJ_BACKEND"] = "v2"


def _prefill(model: Any, input_ids: torch.Tensor, *, output_hidden_states: bool, keep_past: bool = False) -> dict[str, Any]:
    reset_batch_invariant_kproj_counters()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(input_ids.device)
    with torch.inference_mode():
        out = model.model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last_hidden = out.last_hidden_state[:, -1, :].detach()
        logits = model.lm_head(last_hidden).detach()
    peak = torch.cuda.max_memory_allocated(input_ids.device) if torch.cuda.is_available() else 0
    return {
        "last_hidden": last_hidden,
        "logits": logits,
        "past_key_values": out.past_key_values if keep_past else None,
        "hidden_states": out.hidden_states,
        "counters": batch_invariant_kproj_counters(),
        "peak_allocated_bytes": int(peak),
    }


def _time_prefill(model: Any, input_ids: torch.Tensor, *, warmup: int, reps: int) -> dict[str, float]:
    for _ in range(warmup):
        _prefill(model, input_ids, output_hidden_states=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(input_ids.device)
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        _prefill(model, input_ids, output_hidden_states=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples_sorted = sorted(samples)
    p95_idx = min(len(samples_sorted) - 1, int(0.95 * (len(samples_sorted) - 1)))
    tokens = int(input_ids.numel())
    median_ms = statistics.median(samples)
    return {
        "median_ms": float(median_ms),
        "mean_ms": float(statistics.mean(samples)),
        "p95_ms": float(samples_sorted[p95_idx]),
        "std_ms": float(statistics.pstdev(samples)),
        "tokens_per_s": float(tokens / max(median_ms / 1000.0, 1e-12)),
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(input_ids.device) if torch.cuda.is_available() else 0),
        "peak_reserved_bytes": float(torch.cuda.max_memory_reserved(input_ids.device) if torch.cuda.is_available() else 0),
    }


def _decode_one_counters(model: Any, input_ids: torch.Tensor) -> dict[str, int]:
    prefill = _prefill(model, input_ids, output_hidden_states=False, keep_past=True)
    past = prefill["past_key_values"]
    next_ids = input_ids[:, -1:]
    reset_batch_invariant_kproj_counters()
    with torch.inference_mode():
        model.model(input_ids=next_ids, past_key_values=past, use_cache=True, return_dict=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize(input_ids.device)
    return batch_invariant_kproj_counters()


def _run_actual(args: argparse.Namespace) -> dict[str, Any]:
    from bench.run_actual_model_fixed_batch_smoke import load_model, make_fixed_inputs

    device = torch.device(args.device)
    tokenizer, _config, model = load_model(dtype=torch.float16, device=device)
    base_inputs = make_fixed_inputs(tokenizer, batch=4, context=max(args.contexts), device=device)
    labels = ["A", "B", "C", "D"]
    gate = dict(FINAL_GATE_TEMPLATE)
    gate["actual_model_loaded"] = True
    gate["p0_available"] = gate["p1_available"] = gate["p2_available"] = True
    gate["p2_uses_existing_bi_v2_kernel"] = True

    dispatch: dict[str, Any] = {}
    numerical_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    logit_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    perf_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []

    for ctx in args.contexts:
        input_ids = base_inputs[:, :ctx]
        refs: dict[str, dict[str, Any]] = {}
        if ctx == 512:
            _set_variant_env("p0")
            dispatch["p0_b2_ab"] = _prefill(model, input_ids[[0, 1]], output_hidden_states=False)["counters"]
            dispatch["p0_b4"] = _prefill(model, input_ids[[0, 1, 2, 3]], output_hidden_states=False)["counters"]
        for variant in ("p1", "p2"):
            _set_variant_env(variant)
            refs[variant] = {}
            for i, label in enumerate(labels):
                capture_layers = bool(args.layerwise and ctx == 512)
                refs[variant][label] = _prefill(model, input_ids[i : i + 1], output_hidden_states=capture_layers)
            for batch_name, rows in (("b2_ab", [0, 1]), ("b2_cd", [2, 3]), ("b4", [0, 1, 2, 3])):
                capture_layers = bool(args.layerwise and ctx == 512 and len(rows) <= args.layerwise_max_batch)
                batched = _prefill(model, input_ids[rows], output_hidden_states=capture_layers)
                if ctx == 512:
                    dispatch[f"{variant}_{batch_name}"] = batched["counters"]
                for out_row, source_row in enumerate(rows):
                    label = labels[source_row]
                    ref = refs[variant][label]
                    hidden_m = tensor_metric_dict(ref["last_hidden"][0], batched["last_hidden"][out_row])
                    logit_m = logits_metric_dict(ref["logits"][0:1], batched["logits"][out_row : out_row + 1])
                    row = {
                        "context": ctx,
                        "variant": variant,
                        "batch": "B2" if batch_name.startswith("b2") else "B4",
                        "request": label,
                        "row": out_row,
                        **{f"hidden_{k}": v for k, v in hidden_m.items()},
                        **{f"logit_{k}": v for k, v in logit_m.items()},
                    }
                    numerical_rows.append(row)
                    logit_rows.append(row)
                    context_rows.append(row)
                    if ctx == 512 and batched["hidden_states"] is not None:
                        for layer in ([0, 1, 4, 8, 16, 24, 31] if variant == "p1" else [0, 8, 16, 31]):
                            if layer + 1 < len(batched["hidden_states"]):
                                m = tensor_metric_dict(ref["hidden_states"][layer + 1][0, -1], batched["hidden_states"][layer + 1][out_row, -1])
                                layer_rows.append({"variant": variant, "batch": row["batch"], "request": label, "layer": layer, **m})
            if ctx == 512 and variant == "p2":
                dispatch["p2_decode_one"] = _decode_one_counters(model, input_ids[0:1])

        if ctx in args.perf_contexts:
            for variant in ("p1", "p2"):
                _set_variant_env(variant)
                for batch, rows in (("B1", [0]), ("B2", [0, 1]), ("B4", [0, 1, 2, 3])):
                    reps = args.reps_512 if ctx == 512 else args.reps_2048 if ctx == 2048 else args.reps_4096
                    timing = _time_prefill(model, input_ids[rows], warmup=args.warmup, reps=reps)
                    perf_rows.append({"context": ctx, "variant": variant, "batch": batch, **timing})
                    memory_rows.append({"context": ctx, "variant": variant, "batch": batch, **timing})

    _write_json(REPORT_DIR / "dispatch_counters.json", dispatch)
    _write_csv(REPORT_DIR / "numerical_by_request.csv", numerical_rows)
    _write_csv(REPORT_DIR / "context_drift.csv", context_rows)
    _write_csv(REPORT_DIR / "first_token_logits.csv", logit_rows)
    _write_csv(REPORT_DIR / "layerwise_drift.csv", layer_rows)
    _write_csv(REPORT_DIR / "prefill_performance.csv", perf_rows)
    _write_csv(REPORT_DIR / "memory.csv", memory_rows)
    _write_csv(REPORT_DIR / "projection_breakdown.csv", [])
    _write_json(REPORT_DIR / "p2_correctness.json", {"sampled": True, "dispatch_counters": dispatch})
    _write_json(REPORT_DIR / "semantic_state.json", {"sampled": True})

    def mean_metric(variant: str, batch: str, ctx: int, metric: str) -> float | None:
        vals = [float(r[metric]) for r in numerical_rows if r["variant"] == variant and r["batch"] == batch and r["context"] == ctx]
        return statistics.mean(vals) if vals else None

    def perf_metric(variant: str, batch: str, ctx: int) -> float | None:
        vals = [float(r["median_ms"]) for r in perf_rows if r["variant"] == variant and r["batch"] == batch and r["context"] == ctx]
        return vals[0] if vals else None

    gate.update(
        {
            "p1_b2_ctx512_final_hidden_rel_l2": mean_metric("p1", "B2", 512, "hidden_relative_l2"),
            "p2_b2_ctx512_final_hidden_rel_l2": mean_metric("p2", "B2", 512, "hidden_relative_l2"),
            "p1_b4_ctx512_final_hidden_rel_l2": mean_metric("p1", "B4", 512, "hidden_relative_l2"),
            "p2_b4_ctx512_final_hidden_rel_l2": mean_metric("p2", "B4", 512, "hidden_relative_l2"),
            "p1_b2_ctx512_logit_rel_l2": mean_metric("p1", "B2", 512, "logit_relative_l2"),
            "p2_b2_ctx512_logit_rel_l2": mean_metric("p2", "B2", 512, "logit_relative_l2"),
            "p1_b4_ctx512_logit_rel_l2": mean_metric("p1", "B4", 512, "logit_relative_l2"),
            "p2_b4_ctx512_logit_rel_l2": mean_metric("p2", "B4", 512, "logit_relative_l2"),
        }
    )
    for ctx in (512, 2048, 4096):
        for batch in ("B2", "B4"):
            p1_ms = perf_metric("p1", batch, ctx)
            p2_ms = perf_metric("p2", batch, ctx)
            gate[f"p1_{batch.lower()}_ctx{ctx}_prefill_ms"] = p1_ms
            gate[f"p2_{batch.lower()}_ctx{ctx}_prefill_ms"] = p2_ms
            gate[f"p2_overhead_percent_{batch.lower()}_ctx{ctx}"] = prefill_overhead_percent(p1_ms, p2_ms)

    gate["hidden_drift_reduction_ratio_b2"] = drift_reduction_ratio(gate["p1_b2_ctx512_final_hidden_rel_l2"], gate["p2_b2_ctx512_final_hidden_rel_l2"])
    gate["hidden_drift_reduction_ratio_b4"] = drift_reduction_ratio(gate["p1_b4_ctx512_final_hidden_rel_l2"], gate["p2_b4_ctx512_final_hidden_rel_l2"])
    gate["logit_drift_reduction_ratio_b2"] = drift_reduction_ratio(gate["p1_b2_ctx512_logit_rel_l2"], gate["p2_b2_ctx512_logit_rel_l2"])
    gate["logit_drift_reduction_ratio_b4"] = drift_reduction_ratio(gate["p1_b4_ctx512_logit_rel_l2"], gate["p2_b4_ctx512_logit_rel_l2"])
    gate["p2_prefill_k_bi"] = all(c.get("bi_prefill_kproj_calls", 0) > 0 for k, c in dispatch.items() if k.startswith("p2_") and "decode" not in k)
    gate["p2_prefill_v_bi"] = all(c.get("bi_prefill_vproj_calls", 0) > 0 for k, c in dispatch.items() if k.startswith("p2_") and "decode" not in k)
    p2_decode = dispatch.get("p2_decode_one", {})
    gate["p2_decode_k_bi"] = p2_decode.get("bi_decode_kproj_calls", 0) > 0 and p2_decode.get("normal_decode_kproj_calls", 0) == 0
    gate["p2_decode_v_bi"] = p2_decode.get("bi_decode_vproj_calls", 0) > 0 and p2_decode.get("normal_decode_vproj_calls", 0) == 0
    gate["p2_serial_request_dispatches"] = sum(c.get("bi_kproj_serial_request_dispatches", 0) for k, c in dispatch.items() if k.startswith("p2_"))
    gate["p2_fallback_calls"] = sum(c.get("bi_kproj_fallback_calls", 0) for k, c in dispatch.items() if k.startswith("p2_"))
    p1_mem = [float(r["peak_allocated_bytes"]) for r in perf_rows if r["variant"] == "p1"]
    p2_mem = [float(r["peak_allocated_bytes"]) for r in perf_rows if r["variant"] == "p2"]
    gate["p1_peak_memory_bytes"] = max(p1_mem) if p1_mem else None
    gate["p2_peak_memory_bytes"] = max(p2_mem) if p2_mem else None
    gate["p1_argmax_match_rate"] = statistics.mean(float(r["logit_argmax_same"]) for r in numerical_rows if r["variant"] == "p1")
    gate["p2_argmax_match_rate"] = statistics.mean(float(r["logit_argmax_same"]) for r in numerical_rows if r["variant"] == "p2")
    overhead = gate["p2_overhead_percent_b4_ctx512"]
    ratio = gate["logit_drift_reduction_ratio_b4"]
    classification, next_task = classify_cost_benefit(overhead_percent=overhead, logit_drift_reduction_ratio=ratio)
    gate["classification"] = classification
    gate["next_task"] = next_task
    gate["architecture_recommendation"] = classification
    gate["cost_level"] = "unknown" if overhead is None else ("low" if overhead <= 2 else "moderate" if overhead <= 10 else "high")
    gate["numerical_benefit_level"] = "unknown" if ratio is None else ("strong" if ratio >= 2 else "small")
    return gate


def _write_static_reports(gate: dict[str, Any], *, dry_run: bool, nvidia_smi: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_md(REPORT_DIR / "environment.md", "Environment", f"start_head: `{START_HEAD}`\n\n```\n{nvidia_smi}\n```")
    _write_md(REPORT_DIR / "variant_definition.md", "Variant Definition", "P0=`normal`, P1=`bi_k`, P2=`bi_kv`. P2 is experimental and not the production default.")
    _write_md(REPORT_DIR / "dispatch_audit.md", "Dispatch Audit", "Prefill projection dispatch is controlled by `PATTERNKV_PREFILL_PROJ_MODE`; legacy `PATTERNKV_BATCH_INVARIANT_KPROJ=1` maps to P1.")
    for name in (
        "p2_correctness",
        "full_model_numerical",
        "layerwise_propagation",
        "context_length_scaling",
        "first_token_logits",
        "prefill_performance",
        "projection_cost_breakdown",
        "memory",
        "semantic_state",
        "cost_benefit_matrix",
        "architecture_recommendation",
        "risk_analysis",
        "final_recommendation",
    ):
        _write_md(REPORT_DIR / f"{name}.md", name.replace("_", " ").title(), "Generated by `bench/run_bi_vproj_cost_benefit.py`." + (" Dry run only." if dry_run else ""))
    _write_json(REPORT_DIR / "decision_matrix.json", {"classification": gate["classification"], "next_task": gate["next_task"]})
    _write_json(REPORT_DIR / "final_gate.json", gate)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="write scaffolding and gate template without loading the model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contexts", type=int, nargs="+", default=[512])
    parser.add_argument("--perf-contexts", type=int, nargs="+", default=[512])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps-512", type=int, default=20)
    parser.add_argument("--reps-2048", type=int, default=10)
    parser.add_argument("--reps-4096", type=int, default=5)
    parser.add_argument("--layerwise", action="store_true", help="capture all hidden states for layerwise drift; increases memory use")
    parser.add_argument("--layerwise-max-batch", type=int, default=2)
    args = parser.parse_args(list(argv) if argv is not None else None)

    smi = _nvidia_smi()
    if args.dry_run:
        gate = dict(FINAL_GATE_TEMPLATE)
        gate["classification"] = "BI_VPROJ_COST_BENEFIT_INCONCLUSIVE"
        _write_static_reports(gate, dry_run=True, nvidia_smi=smi)
        return 0

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gate = _run_actual(args)
    _write_static_reports(gate, dry_run=False, nvidia_smi=smi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
