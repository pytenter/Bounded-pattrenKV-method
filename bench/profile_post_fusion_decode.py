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

from bench.bench_aime24_patternkv import load_model
from models.segmented_cache import PatternQuantizedKVCache, install_value_capacity_buffers, normalize_capacity_growth_backend, serialize_cache, validate_cache
from quant.matmul import get_patternkv_mixed_v_counters, reset_patternkv_mixed_v_counters
from quant.patternkv_profile import cache_mutation_snapshot, merge_profile_rows, profile_snapshot, reset_profile, temp_allocation_snapshot
from scripts.run_aime24_full_causal25_quality import make_worker_args


OUT_DIR = ROOT / "reports/system_profile_v1"
S1_DIR = ROOT / "reports/system_kernel_v1"


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return float(ordered[idx])


def precision_mask(tokens: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(1, tokens, dtype=torch.uint8, device=device)
    k = int(round(tokens * 0.25))
    mask[0, 1::4] = 1
    if int(mask.sum().item()) > k:
        chosen = torch.nonzero(mask[0].bool(), as_tuple=False).flatten()
        mask[0, chosen[k:]] = 0
    cursor = 0
    while int(mask.sum().item()) < k:
        mask[0, cursor] = 1
        cursor += 1
    return mask


def random_packed(shape: tuple[int, ...], *, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    return torch.randint(-(2**30), 2**30 - 1, shape, device=device, dtype=torch.int32, generator=generator)


def build_layer_cache(layer, *, context_tokens: int, seed: int) -> tuple[Any, ...]:
    attn = layer.self_attn
    device = next(attn.parameters()).device
    dtype = next(attn.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    bsz = 1
    kv_heads = int(attn.num_key_value_heads)
    head_dim = int(attn.head_dim)
    group = int(attn.group_size)
    k_bits = int(attn.k_bits)
    v_bits = int(attn.v_bits)
    sink_tokens = min(int(getattr(attn.config, "sink_length", 16)), context_tokens)
    recent_tokens = min(int(getattr(attn.config, "recent_length", 128)), max(context_tokens - sink_tokens, 0))
    history_tokens = max(context_tokens - sink_tokens - recent_tokens, 0)
    pending_tokens = history_tokens % group
    packed_tokens = history_tokens - pending_tokens
    sink_k = torch.randn(bsz, kv_heads, sink_tokens, head_dim, device=device, dtype=dtype, generator=generator) * 0.02 if sink_tokens else None
    sink_v = torch.randn(bsz, kv_heads, sink_tokens, head_dim, device=device, dtype=dtype, generator=generator) * 0.02 if sink_tokens else None
    pending_k = torch.randn(bsz, kv_heads, pending_tokens, head_dim, device=device, dtype=dtype, generator=generator) * 0.02 if pending_tokens else None
    pending_v = torch.randn(bsz, kv_heads, pending_tokens, head_dim, device=device, dtype=dtype, generator=generator) * 0.02 if pending_tokens else None
    recent_k = torch.randn(bsz, kv_heads, recent_tokens, head_dim, device=device, dtype=dtype, generator=generator) * 0.02 if recent_tokens else None
    recent_v = torch.randn(bsz, kv_heads, recent_tokens, head_dim, device=device, dtype=dtype, generator=generator) * 0.02 if recent_tokens else None
    k_pack = 32 // k_bits
    packed_k = random_packed((bsz, kv_heads, head_dim, packed_tokens // k_pack), device=device, generator=generator) if packed_tokens else None
    packed_k_scale = torch.rand(bsz, kv_heads, head_dim, packed_tokens // group, device=device, dtype=dtype, generator=generator) * 0.02 + 1e-4 if packed_tokens else None
    packed_k_zero = torch.randn(bsz, kv_heads, head_dim, packed_tokens // group, device=device, dtype=dtype, generator=generator) * 0.02 if packed_tokens else None
    mask = precision_mask(packed_tokens, device) if packed_tokens else None
    v4_tokens = int(mask.sum().item()) if mask is not None else 0
    v2_tokens = packed_tokens - v4_tokens
    packed_v = random_packed((bsz, kv_heads, v2_tokens, head_dim // (32 // 2)), device=device, generator=generator) if v2_tokens else None
    packed_v_scale = torch.rand(bsz, kv_heads, v2_tokens, head_dim // group, device=device, dtype=dtype, generator=generator) * 0.02 + 1e-4 if v2_tokens else None
    packed_v_zero = torch.randn(bsz, kv_heads, v2_tokens, head_dim // group, device=device, dtype=dtype, generator=generator) * 0.02 if v2_tokens else None
    packed_v4 = random_packed((bsz, kv_heads, v4_tokens, head_dim // (32 // 4)), device=device, generator=generator) if v4_tokens else None
    packed_v4_scale = torch.rand(bsz, kv_heads, v4_tokens, head_dim // group, device=device, dtype=dtype, generator=generator) * 0.02 + 1e-4 if v4_tokens else None
    packed_v4_zero = torch.randn(bsz, kv_heads, v4_tokens, head_dim // group, device=device, dtype=dtype, generator=generator) * 0.02 if v4_tokens else None
    centroid_count = int(attn.k_base.shape[1])
    k_assignments = torch.randint(0, centroid_count, (bsz, kv_heads, packed_tokens), device=device, dtype=torch.long, generator=generator) if packed_tokens else None
    v_assignment_idx = torch.randint(0, centroid_count, (bsz, kv_heads, packed_tokens), device=device, dtype=torch.long, generator=generator) if packed_tokens else None
    v_pattern_mask = torch.randint(0, 2, (bsz, kv_heads, packed_tokens), device=device, dtype=torch.uint8, generator=generator) if packed_tokens else None
    cache = PatternQuantizedKVCache(
        sink_k=sink_k,
        sink_v=sink_v,
        packed_k=packed_k,
        packed_k_scale=packed_k_scale,
        packed_k_zero=packed_k_zero,
        packed_v=packed_v,
        packed_v_scale=packed_v_scale,
        packed_v_zero=packed_v_zero,
        pending_k=pending_k,
        pending_v=pending_v,
        recent_k=recent_k,
        recent_v=recent_v,
        total_tokens=context_tokens,
        packed_k_tokens=packed_tokens,
        packed_v_tokens=packed_tokens,
        sink_length=sink_tokens,
        recent_length=recent_tokens,
        group_size=group,
        k_bits=k_bits,
        v_bits=v_bits,
        pack_count_k=packed_tokens // group,
        pack_count_v=packed_tokens // group,
        k_assignments=k_assignments,
        v_assignments=v_pattern_mask,
        v_assignment_idx=v_assignment_idx,
        v_pattern_mask=v_pattern_mask,
        k_centroids=attn.k_base.detach(),
        v_centroids=attn.v_centroids.detach(),
        centroid_updates_k=0,
        centroid_updates_v=0,
        value_objective=str(attn.value_objective),
        v_precision_selector=str(attn.v_precision_selector),
        v4_budget_fraction=float(attn.v4_budget_fraction),
        random_selector_seed=int(attn.random_selector_seed),
        v_precision_mask=mask,
        packed_v4=packed_v4,
        packed_v4_scale=packed_v4_scale,
        packed_v4_zero=packed_v4_zero,
        packed_v4_tokens=v4_tokens,
        v_causal_importance=torch.rand(bsz, context_tokens, device=device, dtype=torch.float32, generator=generator) if context_tokens else None,
        v_oracle_importance=None,
    )
    cache.selector_task_key = "phase_s1_5_synthetic"
    cache.selector_layer_idx = int(attn.layer_idx)
    cache.capacity_backend = normalize_capacity_growth_backend(None)
    install_value_capacity_buffers(cache)
    validate_cache(cache)
    return serialize_cache(cache)


def build_synthetic_past(model, *, context_tokens: int, seed: int) -> tuple[Any, ...]:
    start = time.perf_counter()
    past = tuple(build_layer_cache(layer, context_tokens=context_tokens, seed=seed + i * 17) for i, layer in enumerate(model.model.layers))
    torch.cuda.synchronize()
    return past, (time.perf_counter() - start) * 1000.0


def ensure_profile_centroids(model, *, seed: int) -> None:
    for idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        device = next(attn.parameters()).device
        dtype = next(attn.parameters()).dtype
        generator = torch.Generator(device=device).manual_seed(seed + idx)
        kv_heads = int(attn.num_key_value_heads)
        head_dim = int(attn.head_dim)
        if getattr(attn, "k_base", None) is None:
            attn.k_base = torch.randn(kv_heads, int(attn.num_k_bases), head_dim, device=device, dtype=dtype, generator=generator) * 0.02
        if getattr(attn, "v_centroids", None) is None:
            attn.v_centroids = torch.randn(kv_heads, int(attn.num_v_bases), head_dim, device=device, dtype=dtype, generator=generator) * 0.02


@torch.no_grad()
def run_decode_case(model, *, backend: str, context_tokens: int, decode_tokens: int, profile: bool, seed: int) -> dict[str, Any]:
    device = next(model.parameters()).device
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = backend
    os.environ["PATTERNKV_PROFILE"] = "1" if profile else "0"
    reset_profile()
    reset_patternkv_mixed_v_counters()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    past, setup_ms = build_synthetic_past(model, context_tokens=context_tokens, seed=seed)
    current = torch.randint(1, int(model.config.vocab_size), (1, 1), device=device, dtype=torch.long)
    full_mask = torch.ones(1, context_tokens + decode_tokens + 1, device=device, dtype=torch.long)
    events = []
    for step in range(decode_tokens):
        attention_mask = full_mask[:, : context_tokens + step + 1]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = model(input_ids=current, attention_mask=attention_mask, past_key_values=past, use_cache=True, return_dict=True)
        end.record()
        events.append((start, end))
        past = out.past_key_values
        current = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    torch.cuda.synchronize()
    token_ms = [float(start.elapsed_time(end)) for start, end in events]
    decode_total_ms = float(sum(token_ms))
    snapshot = profile_snapshot(reset=False) if profile else {}
    temp_allocations = temp_allocation_snapshot(decode_tokens=decode_tokens) if profile else []
    cache_mutations = cache_mutation_snapshot() if profile else []
    if profile:
        reset_profile()
    counters = get_patternkv_mixed_v_counters()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    return {
        "backend": backend,
        "context_tokens": context_tokens,
        "decode_tokens": decode_tokens,
        "prefill_ms": setup_ms,
        "cache_setup_ms": setup_ms,
        "decode_total_ms": decode_total_ms,
        "mean_tpot_ms": statistics.mean(token_ms),
        "median_tpot_ms": statistics.median(token_ms),
        "p90_tpot_ms": percentile(token_ms, 0.90),
        "p95_tpot_ms": percentile(token_ms, 0.95),
        "tokens_per_sec": 1000.0 / max(statistics.mean(token_ms), 1e-9),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "mixed_v_fused_calls": int(counters["mixed_v_fused_calls"]),
        "mixed_v_reference_calls": int(counters["mixed_v_reference_calls"]),
        "selector_calls": int(snapshot.get("selector_total", {}).get("calls", 0.0)),
        "importance_update_calls": int(snapshot.get("importance_update", {}).get("calls", 0.0)),
        "pack_calls": int(snapshot.get("pack_window", {}).get("calls", 0.0)),
        "cache_concat_events": int(snapshot.get("cache_cat_events", {}).get("calls", 0.0)),
        "cache_concat_bytes": int(snapshot.get("cache_cat_events", {}).get("bytes", 0.0)),
        "largest_cache_concat_bytes": int(snapshot.get("cache_cat_largest_bytes", {}).get("bytes", 0.0)),
        "profile_enabled": bool(profile),
        "profile_snapshot": snapshot,
        "temp_allocations": temp_allocations,
        "cache_mutations": cache_mutations,
    }


def summary_row(result: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "backend",
        "context_tokens",
        "decode_tokens",
        "prefill_ms",
        "cache_setup_ms",
        "decode_total_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p90_tpot_ms",
        "p95_tpot_ms",
        "tokens_per_sec",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "mixed_v_fused_calls",
        "mixed_v_reference_calls",
        "selector_calls",
        "importance_update_calls",
        "pack_calls",
        "cache_concat_events",
        "cache_concat_bytes",
        "largest_cache_concat_bytes",
        "component_profile_decode_total_ms",
    ]
    return {key: result[key] for key in keys}


def group_components(rows: list[dict[str, Any]]) -> dict[str, float]:
    groups = {
        "QK": ("qk_quantized_history", "qk_fp16_regions", "attention_score_concat"),
        "softmax": ("attention_softmax",),
        "importance_update": ("importance_update",),
        "mixed_v": ("mixed_v_fused_attention", "mixed_v_reference_attention"),
        "selector": ("selector_total",),
        "packing": ("pack_window",),
        "cache_mutation": ("cache_mutation", "cache_append", "cache_flush"),
        "output_projection": ("output_projection",),
        "lm_head": ("decode_lm_head",),
    }
    by_component = {row["component"]: float(row["total_us"]) for row in rows}
    return {name: sum(by_component.get(c, 0.0) for c in comps) for name, comps in groups.items()}


def scaling_rows(component_rows: list[dict[str, Any]], backend: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], float] = {}
    for row in component_rows:
        if row["backend"] != backend:
            continue
        grouped[(int(row["context_tokens"]), str(row["component"]))] = float(row["total_us"])
    contexts = sorted({ctx for ctx, _ in grouped})
    rows = []
    for component in sorted({comp for _, comp in grouped}):
        for prev, cur in zip(contexts, contexts[1:]):
            prev_us = grouped.get((prev, component), 0.0)
            cur_us = grouped.get((cur, component), 0.0)
            ratio = cur_us / prev_us if prev_us > 0 else None
            rows.append({"backend": backend, "component": component, "from_context": prev, "to_context": cur, "ratio": ratio})
    return rows


def classify_next_phase(group_share: dict[str, float]) -> tuple[str, str, list[tuple[str, float]]]:
    ranked = sorted(group_share.items(), key=lambda item: item[1], reverse=True)
    top = ranked[0][0] if ranked else "unknown"
    if group_share.get("selector", 0.0) >= 10.0:
        return "S2_GPU_SELECTOR", "selector/top-k share is a clear decode-time hotspot", ranked
    if top == "importance_update":
        return "S2_IMPORTANCE_UPDATE", "causal importance update is the largest measured component", ranked
    if top == "packing":
        return "S2_PACKING", "packing dominates measured decode components", ranked
    if top == "cache_mutation":
        return "S3_FIXED_PAGE_ABI", "dynamic cache mutation is the largest measured component", ranked
    if top in {"QK", "softmax", "mixed_v"}:
        return "PROFILE_INCONCLUSIVE", f"{top} dominates, but this phase did not authorize a matching optimization path", ranked
    return "PROFILE_INCONCLUSIVE", "dominant cost is outside the explicit next-phase rules", ranked


def write_report(
    *,
    summaries: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    scaling: list[dict[str, Any]],
    overhead: dict[str, Any],
    s1: dict[str, Any],
    final_gate: dict[str, Any],
) -> None:
    fused_rows = [row for row in component_rows if row["backend"] == "fused"]
    max_context = max(int(row["context_tokens"]) for row in fused_rows)
    fused_max = [row for row in fused_rows if int(row["context_tokens"]) == max_context]
    grouped_us = group_components(fused_max)
    decode_total_us = max(float(next(row for row in summaries if row["backend"] == "fused" and int(row["context_tokens"]) == max_context)["decode_total_ms"]) * 1000.0, 1e-9)
    group_share = {name: value * 100.0 / decode_total_us for name, value in grouped_us.items()}
    next_phase, reason, ranked = classify_next_phase(group_share)
    lines = [
        "# Phase S1.5 End-to-End Decode Profiling",
        "",
        "## Frozen algorithm",
        "",
        "- Frozen commit: `c73aeed3247c136859f695d5b238eeb357434b17`",
        "- Frozen tag: `causal-v4-25-aime24-v1`",
        "- Algorithm changed in this phase: `NO`",
        "",
        "## S1 kernel status",
        "",
        f"- S1 correctness: `{s1['correctness_passed']}/{s1['correctness_total']}`",
        f"- Kernel speedup @16K: `{s1['kernel_speedup_16k']:.3f}x`",
        "- E2E speedups below are separate from kernel microbenchmark speedups.",
        "",
        "## Profiling methodology",
        "",
        "- Decode-focused synthetic cache workload using the real DeepSeek-R1-Distill-Llama-8B PatternKV model.",
        "- Each case runs `q_len=1` decode with seeded legal PatternKV cache tensors matching the frozen cache ABI.",
        "- E2E TPOT is measured with `PATTERNKV_PROFILE=0`; component breakdown is measured in a separate `PATTERNKV_PROFILE=1` pass.",
        "- CUDA events aggregate component timing; ranges are nested, so component percentages are diagnostic shares, not an exclusive flamegraph.",
        "- Short real-model smoke is kept separate and is not used as a throughput benchmark.",
        "",
        "## Reference vs fused E2E results",
        "",
        "| T | reference TPOT ms | fused TPOT ms | E2E speedup |",
        "|---:|---:|---:|---:|",
    ]
    by_key = {(row["backend"], int(row["context_tokens"])): row for row in summaries}
    for ctx in sorted({int(row["context_tokens"]) for row in summaries if row["backend"] == "fused"}):
        ref = by_key.get(("reference", ctx))
        fused = by_key.get(("fused", ctx))
        if ref and fused:
            speedup = float(ref["mean_tpot_ms"]) / max(float(fused["mean_tpot_ms"]), 1e-9)
            lines.append(f"| {ctx} | {float(ref['mean_tpot_ms']):.3f} | {float(fused['mean_tpot_ms']):.3f} | {speedup:.3f} |")
    lines += [
        "",
        "## Component breakdown",
        "",
        f"Fused backend @T={max_context}:",
        "",
        "| Component group | Share of decode wall time |",
        "|---|---:|",
    ]
    for name, share in sorted(group_share.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {name} | {share:.2f}% |")
    lines += [
        "",
        "## Context-length scaling",
        "",
        "Scaling ratios are available in `scaling_analysis.csv` for all measured components.",
        "",
        "## Selector cost",
        "",
        f"- Selector share @T={max_context}: `{group_share.get('selector', 0.0):.2f}%`",
        "",
        "## Causal-importance-update cost",
        "",
        f"- Importance update share @T={max_context}: `{group_share.get('importance_update', 0.0):.2f}%`",
        "",
        "## Packing cost",
        "",
        f"- Packing share @T={max_context}: `{group_share.get('packing', 0.0):.2f}%`",
        "",
        "## Cache mutation / torch.cat cost",
        "",
        f"- Cache mutation share @T={max_context}: `{group_share.get('cache_mutation', 0.0):.2f}%`",
        "",
        "## Mixed fused kernel share",
        "",
        f"- Mixed fused/reference Value share @T={max_context}: `{group_share.get('mixed_v', 0.0):.2f}%`",
        "",
        "## GPU kernel profile",
        "",
        f"- NSYS_AVAILABLE=`{str(shutil.which('nsys') is not None).lower()}`",
        f"- NCU_AVAILABLE=`{str(shutil.which('ncu') is not None).lower()}`",
        "",
        "## Memory behavior",
        "",
        "Runtime memory peaks are recorded in `e2e_summary.csv`.",
        "",
        "## Profiling overhead",
        "",
        f"- Profile off TPOT: `{overhead['profile_off_tpot_ms']:.3f} ms`",
        f"- Profile on TPOT: `{overhead['profile_on_tpot_ms']:.3f} ms`",
        f"- Overhead: `{overhead['overhead_percent']:.2f}%`",
        "",
        "## Dominant bottleneck",
        "",
    ]
    for idx, (name, share) in enumerate(ranked[:5], start=1):
        lines.append(f"{idx}. {name}: {share:.2f}%")
    lines += [
        "",
        "## Recommended next systems phase",
        "",
        f"- `{next_phase}`",
        f"- Reason: {reason}",
        "",
    ]
    (OUT_DIR / "end_to_end_decode_profile.md").write_text("\n".join(lines), encoding="utf-8")
    final_gate["dominant_bottleneck"] = ranked[0][0] if ranked else "unknown"
    final_gate["recommended_next_phase"] = next_phase
    final_gate["recommendation_reason"] = reason


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="2048,4096,8192,16384,32768")
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--overhead-decode-tokens", type=int, default=64)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s1_summary = read_json(S1_DIR / "correctness_summary.json")
    s1_micro = list(csv.DictReader((S1_DIR / "microbenchmark.csv").open("r", encoding="utf-8")))
    s1_speedup_16k = float(next(row for row in s1_micro if int(row["tokens"]) == 16384)["speedup_vs_reference"])
    s1 = {
        "correctness_passed": int(s1_summary["passed_count"]),
        "correctness_total": int(s1_summary["case_count"]),
        "kernel_speedup_16k": s1_speedup_16k,
    }
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    wargs = make_worker_args("CAUSAL_V4_25", 42, args.gpu, experiment_id="phase_s1_5_decode_profile")
    model, _tokenizer = load_model(wargs)
    model.eval()
    ensure_profile_centroids(model, seed=args.seed)
    config_info = {
        "model": str(wargs.model_path),
        "dtype": str(next(model.parameters()).dtype),
        "num_layers": int(model.config.num_hidden_layers),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
        "head_dim": int(model.config.hidden_size // model.config.num_attention_heads),
        "batch_size": 1,
        "decode_tokens": int(args.decode_tokens),
        "contexts": [int(x) for x in args.contexts.split(",") if x],
    }
    write_json(OUT_DIR / "model_config.json", config_info)
    summaries = []
    component_rows = []
    failures = []
    contexts = config_info["contexts"]
    for context in contexts:
        for backend in ("reference", "fused"):
            try:
                e2e_result = run_decode_case(model, backend=backend, context_tokens=context, decode_tokens=args.decode_tokens, profile=False, seed=args.seed + context)
                profile_result = run_decode_case(model, backend=backend, context_tokens=context, decode_tokens=args.decode_tokens, profile=True, seed=args.seed + context)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                failures.append({"backend": backend, "context_tokens": context, "error": f"CUDA OOM: {exc}"})
                continue
            for key in ("selector_calls", "importance_update_calls", "pack_calls", "cache_concat_events", "cache_concat_bytes", "largest_cache_concat_bytes"):
                e2e_result[key] = profile_result[key]
            e2e_result["component_profile_decode_total_ms"] = profile_result["decode_total_ms"]
            summaries.append(summary_row(e2e_result))
            rows = merge_profile_rows(profile_result["profile_snapshot"], decode_tokens=args.decode_tokens, decode_total_us=e2e_result["decode_total_ms"] * 1000.0)
            for row in rows:
                row.update({"backend": backend, "context_tokens": context, "decode_tokens": args.decode_tokens})
            component_rows.extend(rows)
    if not summaries:
        raise RuntimeError("no profiling cases completed")
    write_csv(OUT_DIR / "e2e_summary.csv", summaries)
    write_csv(OUT_DIR / "component_breakdown.csv", component_rows)
    scaling = []
    for backend in ("reference", "fused"):
        scaling.extend(scaling_rows(component_rows, backend))
    write_csv(OUT_DIR / "scaling_analysis.csv", scaling)
    overhead_off = run_decode_case(model, backend="fused", context_tokens=min(contexts), decode_tokens=args.overhead_decode_tokens, profile=False, seed=args.seed + 77)
    overhead_on = run_decode_case(model, backend="fused", context_tokens=min(contexts), decode_tokens=args.overhead_decode_tokens, profile=True, seed=args.seed + 77)
    overhead = {
        "context_tokens": min(contexts),
        "decode_tokens": int(args.overhead_decode_tokens),
        "profile_off_tpot_ms": float(overhead_off["mean_tpot_ms"]),
        "profile_on_tpot_ms": float(overhead_on["mean_tpot_ms"]),
        "overhead_percent": (float(overhead_on["mean_tpot_ms"]) / max(float(overhead_off["mean_tpot_ms"]), 1e-9) - 1.0) * 100.0,
    }
    write_json(OUT_DIR / "profiling_overhead.json", overhead)
    tool_availability = {"nsys_available": shutil.which("nsys") is not None, "ncu_available": shutil.which("ncu") is not None}
    write_json(OUT_DIR / "gpu_tool_availability.json", tool_availability)
    final_gate = {
        "algorithm_changed": False,
        "full_aime24_rerun_started": False,
        "aime25_started": False,
        "vllm_started": False,
        "sglang_started": False,
        "reference_profile_completed": any(row["backend"] == "reference" for row in summaries),
        "fused_profile_completed": any(row["backend"] == "fused" for row in summaries),
        "profiling_overhead_measured": True,
        "component_breakdown_completed": bool(component_rows),
        "context_scaling_completed": bool(scaling),
        "contexts_completed": sorted({int(row["context_tokens"]) for row in summaries}),
        "contexts_failed": failures,
    }
    write_report(summaries=summaries, component_rows=component_rows, scaling=scaling, overhead=overhead, s1=s1, final_gate=final_gate)
    write_json(OUT_DIR / "final_gate.json", final_gate)
    print(json.dumps(final_gate, indent=2, sort_keys=True))
    return 0 if final_gate["reference_profile_completed"] and final_gate["fused_profile_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
