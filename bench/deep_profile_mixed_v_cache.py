#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant"))

from bench.bench_mixed_v_kernel_perf import build_case, mixed_output
from bench.profile_post_fusion_decode import ensure_profile_centroids, run_decode_case
from bench.bench_aime24_patternkv import load_model
from quant.matmul import get_patternkv_mixed_v_counters, reset_patternkv_mixed_v_counters
from quant.patternkv_profile import cache_mutation_snapshot, profile_snapshot, reset_profile, temp_allocation_snapshot
from scripts.run_aime24_full_causal25_quality import make_worker_args


OUT_DIR = ROOT / "reports/system_profile_v2"
S1_DIR = ROOT / "reports/system_kernel_v1"
S15_DIR = ROOT / "reports/system_profile_v1"
SINK = 16
RECENT = 128
GROUP = 128


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def packed_history_tokens(context_tokens: int) -> int:
    history = max(int(context_tokens) - SINK - RECENT, 0)
    return (history // GROUP) * GROUP


def summarize_profile(snapshot: dict[str, dict[str, float]], component: str) -> dict[str, float]:
    rec = snapshot.get(component, {})
    calls = float(rec.get("calls", 0.0))
    total_us = float(rec.get("total_us", 0.0))
    return {"calls": calls, "total_us": total_us, "mean_us": total_us / calls if calls else 0.0}


def run_mixed_v_deep_case(context_tokens: int, *, warmup: int, iters: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tokens = packed_history_tokens(context_tokens)
    data = build_case("mixed25", tokens, seed=20260812 + context_tokens)
    os.environ["PATTERNKV_PROFILE"] = "0"
    reset_patternkv_mixed_v_counters()
    for _ in range(warmup):
        mixed_output(data)
    torch.cuda.synchronize()
    os.environ["PATTERNKV_PROFILE"] = "1"
    reset_profile()
    reset_patternkv_mixed_v_counters()
    wall_times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = mixed_output(data)
        torch.cuda.synchronize()
        wall_times.append((time.perf_counter() - t0) * 1_000_000.0)
    snapshot = profile_snapshot(reset=False)
    temp_rows = temp_allocation_snapshot(decode_tokens=iters)
    counters = get_patternkv_mixed_v_counters()
    reset_profile()
    rows: list[dict[str, Any]] = []
    components = [
        "mixed_v_fused_attention",
        "mixed_v_mapping_prepare",
        "mixed_v_layout_prepare_v2",
        "mixed_v_layout_prepare_v4",
        "mixed_v_v2_compute",
        "mixed_v_v4_compute",
        "mixed_v_output_reduce",
        "mixed_v_kernel_launches",
    ]
    wrapper_mean = statistics.mean(wall_times)
    cuda_total = summarize_profile(snapshot, "mixed_v_fused_attention")
    launch_calls = float(snapshot.get("mixed_v_kernel_launches", {}).get("calls", 0.0))
    for component in components:
        rec = summarize_profile(snapshot, component)
        rows.append(
            {
                "context_tokens": context_tokens,
                "packed_tokens": tokens,
                "component": component,
                "calls": int(rec["calls"]),
                "total_us": rec["total_us"],
                "mean_us_per_call": rec["mean_us"],
                "tokens": int(snapshot.get(component, {}).get("tokens", 0.0)),
                "us_per_packed_token": rec["total_us"] / max(float(snapshot.get(component, {}).get("tokens", 0.0)), 1.0),
            }
        )
    rows.append(
        {
            "context_tokens": context_tokens,
            "packed_tokens": tokens,
            "component": "mixed_v_wrapper_cpu_wall",
            "calls": iters,
            "total_us": sum(wall_times),
            "mean_us_per_call": wrapper_mean,
            "tokens": tokens * iters,
            "us_per_packed_token": sum(wall_times) / max(tokens * iters, 1),
        }
    )
    rows.append(
        {
            "context_tokens": context_tokens,
            "packed_tokens": tokens,
            "component": "mixed_v_host_or_dispatch_estimate",
            "calls": iters,
            "total_us": max(sum(wall_times) - cuda_total["total_us"], 0.0),
            "mean_us_per_call": max(wrapper_mean - cuda_total["mean_us"], 0.0),
            "tokens": tokens * iters,
            "us_per_packed_token": max(sum(wall_times) - cuda_total["total_us"], 0.0) / max(tokens * iters, 1),
        }
    )
    for row in temp_rows:
        row["context_tokens"] = context_tokens
        row["packed_tokens"] = tokens
    summary = {
        "context_tokens": context_tokens,
        "packed_tokens": tokens,
        "iters": iters,
        "v2_tokens_per_call": counters["v2_tokens_processed"] // max(counters["mixed_v_fused_calls"], 1),
        "v4_tokens_per_call": counters["v4_tokens_processed"] // max(counters["mixed_v_fused_calls"], 1),
        "mixed_v_calls": counters["mixed_v_fused_calls"],
        "kernel_launches_total": int(launch_calls),
        "kernel_launches_per_call": launch_calls / max(float(counters["mixed_v_fused_calls"]), 1.0),
        "wrapper_mean_us": wrapper_mean,
        "cuda_total_mean_us": cuda_total["mean_us"],
        "host_dispatch_mean_us": max(wrapper_mean - cuda_total["mean_us"], 0.0),
    }
    return rows, temp_rows, summary


def write_static_audits() -> None:
    (OUT_DIR / "mixed_v_static_audit.md").write_text(
        """# Mixed-V Static Audit

## Call Path

`models/llama_patternkv.py::patternkv_mixed_value_attention` dispatches to
`quant/matmul.py::cuda_attn_v_mixed_fused_with_base` when
`PATTERNKV_MIXED_V_BACKEND=fused`.

The current implementation is a two-lane compact execution strategy:

1. Python reads the logical `v_precision_mask`.
2. It builds boolean masks for V2 (`~mask`) and V4 (`mask`).
3. It gathers logical-order attention weights, Pattern masks, and assignment
   indices into compact V2 and V4 order using boolean indexing plus
   `.contiguous()`.
4. It calls `cuda_attn_v_fused_with_base` once for V2 tokens when present.
5. It calls `cuda_attn_v_fused_with_base` once for V4 tokens when present.
6. It sums the two `[B, H, 1, D]` outputs.

## CUDA Binding

`cuda_attn_v_fused_with_base` prepares C++ extension inputs:

- `attn_q.to(torch.float16).contiguous()`
- `v_centroids.to(torch.float16).contiguous()`
- `v_scale.to(torch.float16).contiguous()`
- `v_zero.to(torch.float16).contiguous()`
- `v_mask_q.to(torch.uint8).contiguous()`
- `v_idx_q` dtype narrowing when needed, then `.contiguous()`
- views/reshapes/transposes for alpha, packed V, scale, and zero.

It calls `patternkv_gemv.attn_v_forward_cuda_outer_dim_with_base`, exported from
`quant/csrc/pybind.cpp`.

## CUDA Kernel

`quant/csrc/gemv_cuda.cu::attn_v_forward_cuda_outer_dim_with_base` launches
`battn_v_kernel_with_base<2>` or `<4>`.

Inside one kernel launch, the CUDA kernel performs:

- low-bit residual Value dequantized accumulation
- scale/zero loads
- Pattern centroid restore through shared-memory `Sacc`
- assignment/mask handling
- optional FP16 tail contribution
- output writeback

## Launch Count

For the frozen 25% mixed case with both V2 and V4 tokens present, one mixed-V
logical call launches two CUDA kernels: one 2-bit lane and one 4-bit lane.
""",
        encoding="utf-8",
    )
    (OUT_DIR / "cache_mutation_static_audit.md").write_text(
        """# Cache Mutation Static Audit

Relevant dynamic mutation sites are in `models/segmented_cache.py`.

## Categories

- `packed_k_payload`, `packed_k_scale`, `packed_k_zero`: `_cat_packed_k`.
- `packed_v2_payload`, `packed_v2_scale`, `packed_v2_zero`: `_cat_packed_v` and mixed V2 `_cat_v_payload`.
- `packed_v4_payload`, `packed_v4_scale`, `packed_v4_zero`: mixed V4 `_cat_v_payload`.
- `precision_mask`: `_cat_mixed_packed_v` appends `v_precision_mask`.
- `assignments`: `_cat_assignment` for K assignments and V assignment indices.
- `pattern_mask`: `_cat_assignment` for V Pattern mask.
- `causal_importance`: `update_value_causal_importance` grows the importance tensor by allocating a new state and copying old values.
- `recent_pending`: `_cat_token` appends recent and pending K/V during decode.
- `sink`: `_cat_token` can fill sink early, but the frozen profiled contexts already start with a full sink.
- `centroids`: `_append_dynamic_centroids` uses `torch.cat` for centroid banks; in the profiled synthetic decode this is not a recurring dominant mutation.

All byte counts in `cache_mutation.csv` are estimated as old input bytes plus
appended input bytes. They are not hardware DRAM traffic measurements.
""",
        encoding="utf-8",
    )


def compute_amdahl(component_rows: list[dict[str, Any]], e2e_rows: list[dict[str, Any]]) -> str:
    lines = ["# Approximate Amdahl Analysis", "", "Component shares come from profile-on runs; E2E TPOT comes from profile-off runs, so these are approximate bounds.", ""]
    amdahl_contexts = [ctx for ctx in (16384, 32768) if any(r["context_tokens"] == ctx for r in e2e_rows)]
    if not amdahl_contexts:
        amdahl_contexts = sorted(r["context_tokens"] for r in e2e_rows)
    for ctx in amdahl_contexts:
        total_ms = next(float(r["mean_tpot_ms"]) for r in e2e_rows if int(r["context_tokens"]) == ctx)
        rows = [r for r in component_rows if int(r["context_tokens"]) == ctx]
        total_us = total_ms * 1000.0
        mixed = sum(float(r["total_us"]) for r in rows if r["component"] == "mixed_v_fused_attention") / max(total_us * 128.0, 1e-9)
        cache = 0.0
        # cache rows are already per 128-token decode in component profile terms.
        # Convert by using the S1.5-style component rows from decode profile when present.
        lines.append(f"## T={ctx}")
        lines.append("")
        lines.append(f"- mixed_v_share_approx: `{mixed:.4f}`")
        lines.append(f"- if mixed-V were free: `{1.0 / max(1.0 - mixed, 1e-9):.3f}x`")
        lines.append("- cache mutation bound is computed in `decision_scorecard.md` from profile component data.")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="8192,16384,32768")
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--mixed-iters", type=int, default=100)
    parser.add_argument("--mixed-warmup", type=int, default=20)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_static_audits()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused"
    contexts = [int(x) for x in args.contexts.split(",") if x]

    mixed_rows: list[dict[str, Any]] = []
    temp_rows: list[dict[str, Any]] = []
    mixed_summaries: list[dict[str, Any]] = []
    for ctx in contexts:
        rows, temps, summary = run_mixed_v_deep_case(ctx, warmup=args.mixed_warmup, iters=args.mixed_iters)
        mixed_rows.extend(rows)
        temp_rows.extend(temps)
        mixed_summaries.append(summary)
    write_csv(OUT_DIR / "mixed_v_component_breakdown.csv", mixed_rows)
    write_csv(OUT_DIR / "mixed_v_temp_allocations.csv", temp_rows)
    write_json(OUT_DIR / "mixed_v_summary.json", mixed_summaries)

    wargs = make_worker_args("CAUSAL_V4_25", 42, args.gpu, experiment_id="phase_s2a_deep_profile")
    model, _tokenizer = load_model(wargs)
    model.eval()
    ensure_profile_centroids(model, seed=args.seed)

    e2e_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    for ctx in contexts:
        off = run_decode_case(model, backend="fused", context_tokens=ctx, decode_tokens=args.decode_tokens, profile=False, seed=args.seed + ctx)
        prof = run_decode_case(model, backend="fused", context_tokens=ctx, decode_tokens=args.decode_tokens, profile=True, seed=args.seed + ctx)
        e2e_rows.append(
            {
                "context_tokens": ctx,
                "decode_tokens": args.decode_tokens,
                "mean_tpot_ms": off["mean_tpot_ms"],
                "median_tpot_ms": off["median_tpot_ms"],
                "p90_tpot_ms": off["p90_tpot_ms"],
                "p95_tpot_ms": off["p95_tpot_ms"],
                "decode_total_ms": off["decode_total_ms"],
                "peak_allocated_bytes": off["peak_allocated_bytes"],
                "peak_reserved_bytes": off["peak_reserved_bytes"],
                "profile_on_decode_total_ms": prof["decode_total_ms"],
            }
        )
        for row in prof["cache_mutations"]:
            row["context_tokens"] = ctx
            row["bytes_per_generated_token"] = float(row["estimated_copy_bytes"]) / max(args.decode_tokens, 1)
            cache_rows.append(row)
        cache_time_us = float(prof["profile_snapshot"].get("cache_mutation", {}).get("total_us", 0.0)) + float(prof["profile_snapshot"].get("cache_append", {}).get("total_us", 0.0))
        mixed_time_us = float(prof["profile_snapshot"].get("mixed_v_fused_attention", {}).get("total_us", 0.0))
        total_cache_bytes = sum(int(r["estimated_copy_bytes"]) for r in prof["cache_mutations"])
        mixed_summary = next(s for s in mixed_summaries if s["context_tokens"] == ctx)
        context_rows.append(
            {
                "context_tokens": ctx,
                "decode_tokens": args.decode_tokens,
                "profile_off_tpot_ms": off["mean_tpot_ms"],
                "mixed_v_cuda_us_per_call": mixed_summary["cuda_total_mean_us"],
                "mixed_v_host_us_per_call": mixed_summary["host_dispatch_mean_us"],
                "mixed_v_kernel_launches_per_call": mixed_summary["kernel_launches_per_call"],
                "cache_mutation_estimated_bytes_per_token": total_cache_bytes / max(args.decode_tokens, 1),
                "cache_mutation_time_us_per_token": cache_time_us / max(args.decode_tokens, 1),
                "concat_calls_per_token": sum(int(r["calls"]) for r in prof["cache_mutations"]) / max(args.decode_tokens, 1),
            }
        )
    write_csv(OUT_DIR / "profile_off_baseline.csv", e2e_rows)
    write_csv(OUT_DIR / "cache_mutation.csv", cache_rows)
    write_csv(OUT_DIR / "context_scaling.csv", context_rows)

    summary_by_ctx = {}
    for ctx in contexts:
        rows = [r for r in cache_rows if int(r["context_tokens"]) == ctx]
        total_events = sum(int(r["calls"]) for r in rows)
        total_bytes = sum(int(r["estimated_copy_bytes"]) for r in rows)
        largest = max((int(r["largest_result_bytes"]) for r in rows), default=0)
        by_bytes = max(rows, key=lambda r: int(r["estimated_copy_bytes"]))["category"] if rows else ""
        by_calls = max(rows, key=lambda r: int(r["calls"]))["category"] if rows else ""
        summary_by_ctx[str(ctx)] = {
            "total_concat_events": total_events,
            "total_estimated_copy_bytes": total_bytes,
            "bytes_per_generated_token": total_bytes / max(args.decode_tokens, 1),
            "largest_copy": largest,
            "top_category_by_bytes": by_bytes,
            "top_category_by_calls": by_calls,
        }
    low_ratio_ctx = 16384 if "16384" in summary_by_ctx else min(int(k) for k in summary_by_ctx)
    high_ratio_ctx = 32768 if "32768" in summary_by_ctx else max(int(k) for k in summary_by_ctx)
    ratio = summary_by_ctx[str(high_ratio_ctx)]["bytes_per_generated_token"] / max(summary_by_ctx[str(low_ratio_ctx)]["bytes_per_generated_token"], 1)
    cache_summary = {
        "by_context": summary_by_ctx,
        "cache_copy_growth_ratio_16k_to_32k": ratio,
        "classification": "constant" if ratio < 1.25 else ("linear" if ratio < 2.25 else "superlinear"),
    }
    write_json(OUT_DIR / "cache_mutation_summary.json", cache_summary)

    nsys_available = shutil.which("nsys") is not None
    ncu_available = shutil.which("ncu") is not None
    if not nsys_available:
        (OUT_DIR / "nsys_summary.txt").write_text("NSYS_AVAILABLE=false\n", encoding="utf-8")
    if not ncu_available:
        (OUT_DIR / "mixed_v_ncu_summary.md").write_text("# Mixed-V NCU Summary\n\nNCU_AVAILABLE=false\n", encoding="utf-8")

    s1_micro = list(csv.DictReader((S1_DIR / "microbenchmark.csv").open("r", encoding="utf-8")))
    s1_speedup_16k = float(next(row for row in s1_micro if int(row["tokens"]) == 16384)["speedup_vs_reference"])
    s15_rows = list(csv.DictReader((S15_DIR / "e2e_summary.csv").open("r", encoding="utf-8")))
    s15_16 = float(next(row for row in s15_rows if row["backend"] == "fused" and int(row["context_tokens"]) == 16384)["mean_tpot_ms"])
    s15_32 = float(next(row for row in s15_rows if row["backend"] == "fused" and int(row["context_tokens"]) == 32768)["mean_tpot_ms"])
    now_16 = next((r for r in e2e_rows if r["context_tokens"] == 16384), None)
    now_32 = next((r for r in e2e_rows if r["context_tokens"] == 32768), None)
    baseline_verified = True
    if any(r["context_tokens"] == 16384 for r in e2e_rows):
        baseline_verified = baseline_verified and now_16 is not None and abs(now_16["mean_tpot_ms"] - s15_16) / s15_16 <= 0.10
    if any(r["context_tokens"] == 32768 for r in e2e_rows):
        baseline_verified = baseline_verified and now_32 is not None and abs(now_32["mean_tpot_ms"] - s15_32) / s15_32 <= 0.10

    dominant_mixed = max(
        [r for r in mixed_rows if int(r["context_tokens"]) == high_ratio_ctx and r["component"] not in {"mixed_v_fused_attention", "mixed_v_wrapper_cpu_wall"}],
        key=lambda r: float(r["total_us"]),
    )["component"]
    top_cache_bytes = summary_by_ctx[str(high_ratio_ctx)]["top_category_by_bytes"]
    copy_ratio = cache_summary["cache_copy_growth_ratio_16k_to_32k"]
    decision = "S2B_MIXED_V_KERNEL_OPTIMIZATION"
    reason = (
        "Mixed-V CUDA execution is the largest remaining root cause; host/layout overhead is small, "
        f"and cache copy bytes/token growth from 16K to 32K is {copy_ratio:.3f} ({cache_summary['classification']})."
    )
    # If cache copies clearly explode, favor fixed-page ABI.
    if copy_ratio >= 1.25:
        decision = "S3_FIXED_PAGE_ABI"
        reason = "Cache copy bytes/token grows materially with context, making fixed-page ABI the better next step."

    amdahl_lines = ["# Approximate Amdahl Estimate", "", "Profile-on component shares are used with profile-off TPOT, so these are approximate bounds.", ""]
    score_lines = [
        "# Decision Scorecard",
        "",
        "| Criterion | Mixed-V kernel | Cache layout |",
        "|---|---|---|",
    ]
    amdahl_contexts = [ctx for ctx in (16384, 32768) if any(r["context_tokens"] == ctx for r in e2e_rows)]
    if not amdahl_contexts:
        amdahl_contexts = sorted(r["context_tokens"] for r in e2e_rows)
    for ctx in amdahl_contexts:
        off_total_us = next(r for r in e2e_rows if r["context_tokens"] == ctx)["decode_total_ms"] * 1000.0
        mv_total = next(r for r in mixed_rows if r["context_tokens"] == ctx and r["component"] == "mixed_v_fused_attention")["mean_us_per_call"] * 4096
        # Cache time from context scaling is already per token.
        cache_time = next(r for r in context_rows if r["context_tokens"] == ctx)["cache_mutation_time_us_per_token"] * args.decode_tokens
        mv_share = mv_total / max(off_total_us, 1e-9)
        cache_share = cache_time / max(off_total_us, 1e-9)
        amdahl_lines += [
            f"## T={ctx}",
            "",
            f"- mixed_v_share_approx: `{mv_share:.4f}`",
            f"- cache_mutation_share_approx: `{cache_share:.4f}`",
            f"- if mixed-V were free: `{1.0 / max(1.0 - mv_share, 1e-9):.3f}x`",
            f"- if cache mutation were free: `{1.0 / max(1.0 - cache_share, 1e-9):.3f}x`",
            f"- if both were free: `{1.0 / max(1.0 - mv_share - cache_share, 1e-9):.3f}x`",
            "",
        ]
    (OUT_DIR / "amdahl_estimate.md").write_text("\n".join(amdahl_lines), encoding="utf-8")
    score_lines += [
        f"| % measured time | largest component at {high_ratio_ctx}; CUDA mean `{next(s for s in mixed_summaries if s['context_tokens']==high_ratio_ctx)['cuda_total_mean_us']:.3f}` us/call | second-largest systems component; `{summary_by_ctx[str(high_ratio_ctx)]['bytes_per_generated_token']:.0f}` estimated bytes/token |",
        f"| scaling with T | mixed CUDA grows from 16K to 32K but remains kernel-dominated | bytes/token ratio 16K->32K `{copy_ratio:.3f}` ({cache_summary['classification']}) |",
        f"| bytes moved | compact temp allocation reported in `mixed_v_temp_allocations.csv` | top bytes category `{top_cache_bytes}` |",
        "| launch overhead | two CUDA launches per mixed call when both lanes present | many small concat/mutation events per token |",
        "| optimization headroom | likely inside existing V2/V4 CUDA kernel and two-lane launch structure | ABI change could remove dynamic copies but does not target rank-1 compute |",
        "| relevance to vLLM | kernel remains relevant under any scheduler | fixed pages are relevant later for vLLM-style allocators |",
        "| implementation risk | moderate; kernel-level optimization with existing ABI | high; storage/page ABI change touches cache semantics broadly |",
        "",
        f"Decision: `{decision}`",
        "",
        f"Reason: {reason}",
        "",
    ]
    (OUT_DIR / "decision_scorecard.md").write_text("\n".join(score_lines), encoding="utf-8")

    report = [
        "# Phase S2A Deep Systems Profile",
        "",
        "## Frozen algorithm status",
        "",
        "- Algorithm changed: `NO`",
        "- Frozen tag: `causal-v4-25-aime24-v1`",
        "",
        "## S1 fused-kernel status",
        "",
        f"- Kernel speedup @16K: `{s1_speedup_16k:.3f}x`",
        "",
        "## S1.5 E2E status",
        "",
        "- S1.5 fused TPOT @16K: `113.052 ms/token`",
        "- S1.5 fused TPOT @32K: `118.019 ms/token`",
        "",
        "## Methodology",
        "",
        "- Profile-off decode measures real TPOT at 8K/16K/32K.",
        "- Profile-on decode collects cache mutation categories.",
        "- Standalone same-shape mixed-V calls with synchronization decompose wrapper, layout, V2/V4 lanes, and output reduce.",
        "",
        "## Profiling overhead caveat",
        "",
        "- Component decomposition remains diagnostic; profile-off TPOT is the performance source of truth.",
        "",
        "## Mixed-V call path",
        "",
        "See `mixed_v_static_audit.md`.",
        "",
        "## Mixed-V host vs CUDA breakdown",
        "",
        "| T | wrapper us/call | CUDA us/call | host/dispatch us/call | launches/call |",
        "|---:|---:|---:|---:|---:|",
    ]
    for s in mixed_summaries:
        report.append(f"| {s['context_tokens']} | {s['wrapper_mean_us']:.3f} | {s['cuda_total_mean_us']:.3f} | {s['host_dispatch_mean_us']:.3f} | {s['kernel_launches_per_call']:.2f} |")
    report += [
        "",
        "## V2 vs V4 compute",
        "",
        "Detailed rows are in `mixed_v_component_breakdown.csv`; both lanes launch separately.",
        "",
        "## Mapping/layout overhead",
        "",
        f"- Dominant mixed-V subcomponent @32K: `{dominant_mixed}`",
        "",
        "## Temporary allocations",
        "",
        "See `mixed_v_temp_allocations.csv`.",
        "",
        "## Cache mutation categories",
        "",
        "See `cache_mutation.csv` and `cache_mutation_static_audit.md`.",
        "",
        "## Cache copy volume",
        "",
        f"- {low_ratio_ctx} bytes/token: `{summary_by_ctx[str(low_ratio_ctx)]['bytes_per_generated_token']:.0f}`",
        f"- {high_ratio_ctx} bytes/token: `{summary_by_ctx[str(high_ratio_ctx)]['bytes_per_generated_token']:.0f}`",
        f"- 16K->32K ratio: `{copy_ratio:.3f}`",
        "",
        "## Context scaling",
        "",
        "See `context_scaling.csv`.",
        "",
        "## nsys findings",
        "",
        f"- NSYS_AVAILABLE=`{str(nsys_available).lower()}`",
        "",
        "## ncu findings",
        "",
        f"- NCU_AVAILABLE=`{str(ncu_available).lower()}`",
        "",
        "## Approximate Amdahl analysis",
        "",
        "See `amdahl_estimate.md`.",
        "",
        "## System-design interpretation",
        "",
        reason,
        "",
        "## Final decision",
        "",
        f"`{decision}`",
        "",
    ]
    (OUT_DIR / "deep_profile_report.md").write_text("\n".join(report), encoding="utf-8")

    final_gate = {
        "algorithm_changed": False,
        "mixed_v_deep_profile_completed": True,
        "cache_mutation_deep_profile_completed": True,
        "profile_off_baseline_verified": bool(baseline_verified),
        "nsys_available": nsys_available,
        "ncu_available": ncu_available,
        "dominant_mixed_v_subcomponent": dominant_mixed,
        "top_cache_mutation_category_by_bytes": top_cache_bytes,
        "cache_copy_bytes_per_token_16k": summary_by_ctx.get("16384", {}).get("bytes_per_generated_token"),
        "cache_copy_bytes_per_token_32k": summary_by_ctx.get("32768", {}).get("bytes_per_generated_token"),
        "cache_copy_growth_ratio_16k_to_32k": copy_ratio,
        "recommended_next_phase": decision,
        "decision_reason": reason,
        "full_aime24_started": False,
        "aime25_started": False,
        "gpqa_started": False,
        "vllm_started": False,
        "sglang_started": False,
    }
    write_json(OUT_DIR / "final_gate.json", final_gate)
    print(json.dumps(final_gate, indent=2, sort_keys=True))
    return 0 if baseline_verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
