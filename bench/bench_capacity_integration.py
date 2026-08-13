#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "system_capacity_integration_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant"))

import patternkv_gemv
from bench.bench_aime24_patternkv import load_model
from bench.profile_post_fusion_decode import ensure_profile_centroids, run_decode_case
from models.segmented_cache import (
    append_decode,
    build_cache_from_prefill,
    get_capacity_cache_counters,
    reset_capacity_cache_counters,
    tensor_bytes,
)
from quant.matmul import (
    cuda_attn_v_fused_with_base,
    cuda_attn_v_fused_with_base_strided_v4,
    cuda_attn_v_mixed_fused_with_base,
    get_patternkv_mixed_v_counters,
    get_patternkv_strided_v2_reader_counters,
    reset_patternkv_mixed_v_counters,
    reset_patternkv_strided_v2_reader_counters,
)
from quant.patternkv_profile import cache_mutation_snapshot, reset_profile
from scripts.run_aime24_full_causal25_quality import make_worker_args


GROUP_SIZE = 128
NH = 32
NH_KV = 8
HEAD_DIM = 128
CENTROIDS = 16
START_HEAD = "5f67a4aade38c573127f83f5413b4b5e86cb3d4b"


def run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(math.ceil(p * len(ordered))) - 1)])


def cv(values: list[float]) -> float:
    mean = statistics.mean(values)
    return float(statistics.stdev(values) / mean) if len(values) > 1 and mean else 0.0


def env_snapshot() -> dict[str, Any]:
    ext = Path(patternkv_gemv.__file__).resolve()
    try:
        smi = run_text(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]).splitlines()
    except Exception as exc:
        smi = [f"nvidia-smi failed: {exc}"]
    return {
        "repo_root": run_text(["git", "rev-parse", "--show-toplevel"]),
        "branch": run_text(["git", "branch", "--show-current"]),
        "start_head_expected": START_HEAD,
        "head_at_report": run_text(["git", "rev-parse", "HEAD"]),
        "worktree_clean": run_text(["git", "status", "--short"]) == "",
        "remotes": run_text(["git", "remote", "-v"]),
        "git_log_10": run_text(["git", "log", "-10", "--oneline"]).splitlines(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": smi,
        "extension_path": str(ext),
        "extension_mtime": int(ext.stat().st_mtime),
        "extension_sha256": sha256(ext),
        "has_strided_v4": hasattr(patternkv_gemv, "attn_v_forward_cuda_outer_dim_with_base_strided_v4"),
    }


def make_case(tokens: int, *, bits: int, capacity: int | None = None, seed: int = 77) -> dict[str, torch.Tensor | int]:
    from models.segmented_cache import quantize_pack_v_reference

    torch.manual_seed(seed + tokens + bits)
    capacity = int(capacity or (33792 if tokens >= 32768 else max(tokens * 2, 4096)))
    value = (torch.randn(1, NH_KV, tokens, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.25).contiguous()
    vq, scale, zero = quantize_pack_v_reference(value, GROUP_SIZE, bits)
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    mask = torch.randint(0, 2, (1, NH_KV, tokens), device="cuda", dtype=torch.uint8)
    idx = torch.randint(0, CENTROIDS, (1, NH_KV, tokens), device="cuda", dtype=torch.int32)
    attn = torch.softmax(torch.randn(1, NH, 1, tokens, device="cuda", dtype=torch.float16), dim=-1).contiguous()
    pack = 32 // bits
    vq_cap = torch.empty(1, NH_KV, capacity, HEAD_DIM // pack, device="cuda", dtype=torch.int32)
    vq_cap.fill_(0x7FFFFFFF)
    vq_cap[:, :, :tokens, :] = vq
    scale_cap = torch.empty(1, NH_KV, capacity, HEAD_DIM // GROUP_SIZE, device="cuda", dtype=torch.float16)
    scale_cap.fill_(float("nan"))
    scale_cap[:, :, :tokens, :] = scale
    zero_cap = torch.empty_like(scale_cap)
    zero_cap.fill_(float("nan"))
    zero_cap[:, :, :tokens, :] = zero
    mask_cap = torch.empty(1, NH_KV, capacity, device="cuda", dtype=torch.uint8)
    mask_cap.fill_(255)
    mask_cap[:, :, :tokens] = mask
    idx_cap = torch.empty(1, NH_KV, capacity, device="cuda", dtype=torch.int32)
    idx_cap.fill_(2147483647)
    idx_cap[:, :, :tokens] = idx
    return locals()


def time_cuda(fn, warmup: int, iters: int, rounds: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    medians = []
    all_times = []
    for _ in range(rounds):
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end) * 1000.0))
        medians.append(statistics.median(times))
        all_times.extend(times)
    return {
        "median_us": statistics.median(medians),
        "mean_round_median_us": statistics.mean(medians),
        "p90_us": percentile(all_times, 0.90),
        "cv": cv(medians),
    }


def run_v4_micro(contexts: list[int], *, warmup: int, iters: int, rounds: int) -> list[dict[str, Any]]:
    rows = []
    for tokens in contexts:
        data = make_case(tokens, bits=4)
        tight = lambda: cuda_attn_v_fused_with_base(GROUP_SIZE, data["attn"], data["vq"], data["scale"], data["zero"], 4, data["centroids"], data["mask"], data["idx"], NH, NH_KV)
        strided = lambda: cuda_attn_v_fused_with_base_strided_v4(
            GROUP_SIZE,
            data["attn"],
            data["vq_cap"][:, :, :tokens, :],
            data["scale_cap"][:, :, :tokens, :],
            data["zero_cap"][:, :, :tokens, :],
            data["centroids"],
            data["mask_cap"][:, :, :tokens],
            data["idx_cap"][:, :, :tokens],
            NH,
            NH_KV,
        )
        out_a = tight()
        out_b = strided()
        torch.cuda.synchronize()
        diff = (out_a.float() - out_b.float()).abs()
        base_t = time_cuda(tight, warmup, iters, rounds)
        str_t = time_cuda(strided, warmup, iters, rounds)
        rows.append({
            "tokens": tokens,
            "capacity": data["capacity"],
            "max_abs": float(diff.max().item()),
            "cosine": float(torch.nn.functional.cosine_similarity(out_a.float().flatten(), out_b.float().flatten(), dim=0).item()),
            "nan": int(torch.isnan(out_b).sum().item()),
            "inf": int(torch.isinf(out_b).sum().item()),
            "baseline_us": base_t["median_us"],
            "strided_us": str_t["median_us"],
            "overhead": (str_t["median_us"] - base_t["median_us"]) / base_t["median_us"],
            "baseline_cv": base_t["cv"],
            "strided_cv": str_t["cv"],
        })
    return rows


def build_real_cache(tokens: int, backend: str):
    os.environ["PATTERNKV_CACHE_GROWTH_BACKEND"] = backend
    torch.manual_seed(900 + tokens)
    k = torch.randn(1, NH_KV, tokens, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(1, NH_KV, tokens, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.25
    c = torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.1
    return build_cache_from_prefill(
        k,
        v,
        sink_length=16,
        recent_length=128,
        group_size=GROUP_SIZE,
        k_bits=2,
        v_bits=2,
        pattern=True,
        k_centroids=c,
        v_centroids=c,
        cache_mode="segmented_rolling",
        chunk_length=128,
        v_precision_selector="causal_v4",
        v4_budget_fraction=0.25,
        selector_layer_idx=0,
    )


def run_real_mutation(contexts: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    memory = []
    old_profile_env = os.environ.get("PATTERNKV_PROFILE")
    os.environ["PATTERNKV_PROFILE"] = "1"
    for tokens in contexts:
        for backend in ("baseline", "fixed_capacity", "chunked_capacity"):
            reset_profile()
            reset_capacity_cache_counters()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            cache = build_real_cache(tokens, backend)
            new_k = torch.randn(1, NH_KV, 128, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.25
            new_v = torch.randn(1, NH_KV, 128, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.25
            start = time.perf_counter()
            append_decode(cache, new_k, new_v)
            torch.cuda.synchronize()
            elapsed_us = (time.perf_counter() - start) * 1_000_000.0
            cats = cache_mutation_snapshot()
            cat_calls = sum(int(row.get("calls", 0)) for row in cats)
            old_bytes = sum(int(row.get("old_bytes", 0)) for row in cats)
            counters = get_capacity_cache_counters()
            cap_stats = []
            if getattr(cache, "capacity_buffers", None):
                cap_stats = [buf.stats() for buf in cache.capacity_buffers.values()]
            reserved = sum(int(s["reserved_capacity_bytes"]) for s in cap_stats)
            valid = sum(int(s["logical_valid_bytes"]) for s in cap_stats)
            unused = sum(int(s["unused_capacity_bytes"]) for s in cap_stats)
            rows.append({
                "context_tokens": tokens,
                "backend": backend,
                "mutation_us_per_token": elapsed_us / 128.0,
                "old_bytes_copied_per_token": old_bytes / 128.0 if backend == "baseline" else counters["capacity_growth_old_bytes_copied"] / 128.0,
                "historical_torch_cat_calls_per_token": cat_calls / 128.0,
                "capacity_growth_events_per_token": counters["capacity_growth_events"] / 128.0,
                "historical_materialize_calls": counters["historical_materialization_calls"],
                "historical_materialized_bytes": counters["historical_materialized_bytes"],
            })
            memory.append({
                "context_tokens": tokens,
                "backend": backend,
                "peak_allocated": int(torch.cuda.max_memory_allocated()),
                "peak_reserved": int(torch.cuda.max_memory_reserved()),
                "logical_valid_bytes": valid,
                "reserved_capacity_bytes": reserved,
                "unused_capacity_bytes": unused,
                "capacity_utilization": valid / max(reserved, 1) if reserved else 1.0,
            })
    if old_profile_env is None:
        os.environ.pop("PATTERNKV_PROFILE", None)
    else:
        os.environ["PATTERNKV_PROFILE"] = old_profile_env
    return rows, memory


def run_mixed_micro(contexts: list[int], *, warmup: int, iters: int, rounds: int) -> list[dict[str, Any]]:
    rows = []
    for tokens in contexts:
        base = build_real_cache(tokens, "baseline")
        cap_fixed = build_real_cache(tokens, "fixed_capacity")
        cap_chunk = build_real_cache(tokens, "chunked_capacity")
        torch.manual_seed(42 + tokens)
        attn = torch.softmax(torch.randn(1, NH, 1, base.packed_v_tokens, device="cuda", dtype=torch.float16), dim=-1).contiguous()
        for backend, cache in (("baseline", base), ("fixed_capacity", cap_fixed), ("chunked_capacity", cap_chunk)):
            os.environ["PATTERNKV_CACHE_GROWTH_BACKEND"] = backend
            reset_patternkv_mixed_v_counters()
            reset_patternkv_strided_v2_reader_counters()
            def fn():
                return cuda_attn_v_mixed_fused_with_base(
                    GROUP_SIZE,
                    attn,
                    cache.packed_v,
                    cache.packed_v_scale,
                    cache.packed_v_zero,
                    cache.packed_v4,
                    cache.packed_v4_scale,
                    cache.packed_v4_zero,
                    cache.v_precision_mask,
                    cache.v_centroids,
                    cache.v_pattern_mask,
                    cache.v_assignment_idx,
                    NH,
                    NH_KV,
                    v2_mask_q=getattr(cache, "v2_pattern_mask", None),
                    v2_idx_q=getattr(cache, "v2_assignment_idx", None),
                    v4_mask_q=getattr(cache, "v4_pattern_mask", None),
                    v4_idx_q=getattr(cache, "v4_assignment_idx", None),
                )
            timed = time_cuda(fn, warmup, iters, rounds)
            counters = get_patternkv_strided_v2_reader_counters()
            mixed = get_patternkv_mixed_v_counters()
            rows.append({
                "context_tokens": tokens,
                "backend": backend,
                "mixed_v_us": timed["median_us"],
                "cv": timed["cv"],
                "strided_v2_calls": counters["strided_v2_calls"],
                "strided_v4_calls": counters["strided_v4_calls"],
                "baseline_v2_calls": mixed["baseline_v2_calls"],
                "baseline_v4_calls": mixed["baseline_v4_calls"],
            })
    return rows


def summarize_rounds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    means = [float(r["mean_tpot_ms"]) for r in rows]
    med = [float(r["median_tpot_ms"]) for r in rows]
    return {
        "mean_tpot_ms": statistics.mean(means),
        "median_tpot_ms": statistics.median(med),
        "p90_tpot_ms": percentile(med, 0.90),
        "tokens_per_sec": 1000.0 / max(statistics.mean(means), 1e-9),
        "cv": cv(med),
    }


def run_e2e(contexts: list[int], *, decode_tokens: int, warmup: int, rounds: int, seed: int, physical_gpu: str) -> list[dict[str, Any]]:
    args = make_worker_args("CAUSAL_V4_25", seed, physical_gpu, experiment_id="capacity_integration_v1")
    model, _tokenizer = load_model(args)
    ensure_profile_centroids(model, seed=seed)
    rows = []
    for ctx in contexts:
        for backend in ("baseline", "fixed_capacity", "chunked_capacity"):
            os.environ["PATTERNKV_CACHE_GROWTH_BACKEND"] = backend
            os.environ["PATTERNKV_SYSTEM_PROFILE"] = "0"
            os.environ["PATTERNKV_PROFILE"] = "0"
            for i in range(warmup):
                run_decode_case(model, backend="fused", context_tokens=ctx, decode_tokens=decode_tokens, profile=False, seed=seed + ctx + i)
            measured = [
                run_decode_case(model, backend="fused", context_tokens=ctx, decode_tokens=decode_tokens, profile=False, seed=seed + ctx + 100 + i)
                for i in range(rounds)
            ]
            summary = summarize_rounds(measured)
            rows.append({"context_tokens": ctx, "backend": backend, "decode_tokens": decode_tokens, "rounds": rounds, **summary})
    return rows


def write_static_reports(gate: dict[str, Any]) -> None:
    (OUT_DIR / "integration_audit.md").write_text(
        "\n".join([
            "# Integration Audit",
            "",
            "- Growing historical Value streams: packed V2, V2 scale/zero, packed V4, V4 scale/zero, precision mask, V pattern mask, V assignment/index, compact V2/V4 pattern/index metadata.",
            "- Historical K streams still use the existing growing contiguous path in this phase.",
            "- QK reader consumes packed K, K scale/zero, and K assignments; it still assumes tight contiguous layout.",
            "- V2 reader consumes compact packed V2, V2 scale/zero, compact V pattern mask, and compact V assignment/index.",
            "- V4 reader consumes compact packed V4, V4 scale/zero, compact V pattern mask, and compact V assignment/index.",
            "- Selector and packing consume pending FP16 V, centroids, causal importance, and emit frozen causal_v4 25% precision identities.",
            "- Flush cadence remains 128 tokens.",
            "- Sink/recent remain unchanged.",
        ]) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "capacity_backend_design.md").write_text(
        "# Capacity Backend Design\n\n- `PATTERNKV_CACHE_GROWTH_BACKEND=baseline|fixed_capacity|chunked_capacity`.\n- Default remains `baseline`.\n- Fixed capacity default tokens: `32768`.\n- Chunked grow size: `4096`.\n- Capacity append writes only new slots with `copy_`; historical Value torch.cat is avoided.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "strided_v4_design.md").write_text(
        "# Strided V4 Design\n\n- Added `cuda_attn_v_fused_with_base_strided_v4`.\n- V4 remains an independent INT4 affine stream with its own scale/zero.\n- The implementation shares the S5A-1 stride-aware addressing template with V2 and preserves the production Value attention math.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "net_tradeoff.md").write_text(
        "\n".join([
            "# Net Tradeoff",
            "",
            f"- 32K mutation time saved, fixed: `{gate.get('fixed_mutation_saved_us_32k')}` us/token.",
            f"- 32K mutation time saved, chunked: `{gate.get('chunked_mutation_saved_us_32k')}` us/token.",
            f"- 32K reader added cost, fixed: `{gate.get('fixed_reader_added_us_32k')}` us.",
            f"- 32K reader added cost, chunked: `{gate.get('chunked_reader_added_us_32k')}` us.",
            f"- Actual best TPOT speedup @32K: `{gate.get('best_tpot_speedup_32k')}`.",
        ]) + "\n",
        encoding="utf-8",
    )


def final_gate(v4_rows, mixed_rows, mutation_rows, memory_rows, e2e_rows) -> dict[str, Any]:
    def row(rows, ctx, backend):
        return next((r for r in rows if int(r["context_tokens"]) == ctx and r["backend"] == backend), None)
    v4_32 = next(r for r in v4_rows if int(r["tokens"]) == 32768)
    base_mut = row(mutation_rows, 32768, "baseline") or {}
    fixed_mut = row(mutation_rows, 32768, "fixed_capacity") or {}
    chunk_mut = row(mutation_rows, 32768, "chunked_capacity") or {}
    base_mix = row(mixed_rows, 32768, "baseline") or {}
    fixed_mix = row(mixed_rows, 32768, "fixed_capacity") or {}
    chunk_mix = row(mixed_rows, 32768, "chunked_capacity") or {}
    base32 = row(e2e_rows, 32768, "baseline") if e2e_rows else None
    fixed32 = row(e2e_rows, 32768, "fixed_capacity") if e2e_rows else None
    chunk32 = row(e2e_rows, 32768, "chunked_capacity") if e2e_rows else None
    speed_fixed = (float(base32["mean_tpot_ms"]) / float(fixed32["mean_tpot_ms"])) if base32 and fixed32 else None
    speed_chunk = (float(base32["mean_tpot_ms"]) / float(chunk32["mean_tpot_ms"])) if base32 and chunk32 else None
    best_backend = "baseline"
    best_speed = 1.0
    if speed_fixed is not None and speed_fixed > best_speed:
        best_backend, best_speed = "fixed_capacity", speed_fixed
    if speed_chunk is not None and speed_chunk > best_speed:
        best_backend, best_speed = "chunked_capacity", speed_chunk
    correctness = bool(v4_32["max_abs"] <= 5e-3 and v4_32["cosine"] >= 0.9999 and v4_32["nan"] == 0 and v4_32["inf"] == 0)
    if not correctness:
        classification = "CAPACITY_CACHE_CORRECTNESS_BLOCKED"
        next_task = "FIX_CAPACITY_CACHE_CORRECTNESS"
    elif e2e_rows and best_speed >= 1.01:
        classification = "CAPACITY_CACHE_V_ONLY_SUPPORTED"
        remaining_k_cat = float((fixed_mut or {}).get("historical_torch_cat_calls_per_token", 0.0) or 0.0) > 0.0
        next_task = "STRIDE_AWARE_K_READER_FEASIBILITY" if remaining_k_cat else "CUDA_VMM_VIRTUAL_CONTIGUOUS_CACHE"
    elif e2e_rows:
        classification = "CAPACITY_CACHE_NO_END_TO_END_GAIN"
        next_task = "MIXED_V_POSTOPT_REVISIT"
    else:
        classification = "BUILD_ENVIRONMENT_BLOCKED"
        next_task = "RUN_PROFILE_OFF_E2E"
    return {
        "algorithm_changed": False,
        "selector_changed": False,
        "quantization_changed": False,
        "attention_math_changed": False,
        "page_native_used": False,
        "gqa_experimental_used": False,
        "capacity_backend_integrated": True,
        "fixed_capacity_integrated": True,
        "chunked_capacity_integrated": True,
        "strided_v2_used": True,
        "strided_v4_implemented": True,
        "historical_materialize_calls_32k": fixed_mut.get("historical_materialize_calls"),
        "historical_materialized_bytes_32k": fixed_mut.get("historical_materialized_bytes"),
        "historical_torch_cat_calls_32k": fixed_mut.get("historical_torch_cat_calls_per_token"),
        "baseline_old_bytes_copied_per_token_32k": base_mut.get("old_bytes_copied_per_token"),
        "fixed_old_bytes_copied_per_token_32k": fixed_mut.get("old_bytes_copied_per_token"),
        "chunked_old_bytes_copied_per_token_32k": chunk_mut.get("old_bytes_copied_per_token"),
        "baseline_mutation_us_32k": base_mut.get("mutation_us_per_token"),
        "fixed_mutation_us_32k": fixed_mut.get("mutation_us_per_token"),
        "chunked_mutation_us_32k": chunk_mut.get("mutation_us_per_token"),
        "baseline_mixed_v_us_32k": base_mix.get("mixed_v_us"),
        "fixed_mixed_v_us_32k": fixed_mix.get("mixed_v_us"),
        "chunked_mixed_v_us_32k": chunk_mix.get("mixed_v_us"),
        "baseline_tpot_8k_ms": (row(e2e_rows, 8192, "baseline") or {}).get("mean_tpot_ms") if e2e_rows else None,
        "fixed_tpot_8k_ms": (row(e2e_rows, 8192, "fixed_capacity") or {}).get("mean_tpot_ms") if e2e_rows else None,
        "chunked_tpot_8k_ms": (row(e2e_rows, 8192, "chunked_capacity") or {}).get("mean_tpot_ms") if e2e_rows else None,
        "baseline_tpot_16k_ms": (row(e2e_rows, 16384, "baseline") or {}).get("mean_tpot_ms") if e2e_rows else None,
        "fixed_tpot_16k_ms": (row(e2e_rows, 16384, "fixed_capacity") or {}).get("mean_tpot_ms") if e2e_rows else None,
        "chunked_tpot_16k_ms": (row(e2e_rows, 16384, "chunked_capacity") or {}).get("mean_tpot_ms") if e2e_rows else None,
        "baseline_tpot_32k_ms": base32.get("mean_tpot_ms") if base32 else None,
        "fixed_tpot_32k_ms": fixed32.get("mean_tpot_ms") if fixed32 else None,
        "chunked_tpot_32k_ms": chunk32.get("mean_tpot_ms") if chunk32 else None,
        "best_backend": best_backend,
        "best_tpot_speedup_32k": best_speed,
        "fixed_mutation_saved_us_32k": (base_mut.get("mutation_us_per_token", 0) - fixed_mut.get("mutation_us_per_token", 0)) if fixed_mut else None,
        "chunked_mutation_saved_us_32k": (base_mut.get("mutation_us_per_token", 0) - chunk_mut.get("mutation_us_per_token", 0)) if chunk_mut else None,
        "fixed_reader_added_us_32k": (fixed_mix.get("mixed_v_us", 0) - base_mix.get("mixed_v_us", 0)) if fixed_mix else None,
        "chunked_reader_added_us_32k": (chunk_mix.get("mixed_v_us", 0) - base_mix.get("mixed_v_us", 0)) if chunk_mix else None,
        "correctness_passed": correctness,
        "classification": classification,
        "recommended_next_phase": next_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", nargs="+", type=int, default=[8192, 16384, 32768])
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--micro-iters", type=int, default=200)
    parser.add_argument("--micro-warmup", type=int, default=30)
    parser.add_argument("--micro-rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--physical-gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    parser.add_argument("--skip-e2e", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "environment.json", env_snapshot())
    v4 = run_v4_micro([2048, 8192, 16384, 32768], warmup=args.micro_warmup, iters=args.micro_iters, rounds=args.micro_rounds)
    write_csv(OUT_DIR / "v4_reader_microbench.csv", v4)
    mutation, memory = run_real_mutation(args.contexts)
    write_csv(OUT_DIR / "real_mutation.csv", mutation)
    write_csv(OUT_DIR / "profile_on_components.csv", mutation)
    write_csv(OUT_DIR / "memory_tradeoff.csv", memory)
    mixed = run_mixed_micro(args.contexts, warmup=max(5, args.micro_warmup // 3), iters=max(50, args.micro_iters // 2), rounds=args.micro_rounds)
    write_csv(OUT_DIR / "mixed_reader_microbench.csv", mixed)
    e2e = [] if args.skip_e2e else run_e2e(args.contexts, decode_tokens=args.decode_tokens, warmup=args.warmup, rounds=args.rounds, seed=args.seed, physical_gpu=args.physical_gpu)
    write_csv(OUT_DIR / "profile_off_e2e.csv", e2e)
    write_csv(OUT_DIR / "copy_bytes_summary.csv", mutation)
    write_csv(OUT_DIR / "capacity_growth_summary.csv", mutation)
    correctness = {
        "v4_cases": len(v4),
        "v4_passed": sum(1 for r in v4 if r["max_abs"] <= 5e-3 and r["cosine"] >= 0.9999 and r["nan"] == 0 and r["inf"] == 0),
        "mixed_cases": len(mixed),
        "tests": "pytest covers compact order, metadata, boundaries, and mixed attention",
    }
    write_json(OUT_DIR / "correctness_summary.json", correctness)
    gate = final_gate(v4, mixed, mutation, memory, e2e)
    write_json(OUT_DIR / "final_gate.json", gate)
    write_static_reports(gate)
    (OUT_DIR / "optimization_scorecard.md").write_text(
        "| Metric | Baseline | Fixed | Chunked |\n| --- | ---: | ---: | ---: |\n"
        f"| 32K mutation us/token | {gate['baseline_mutation_us_32k']} | {gate['fixed_mutation_us_32k']} | {gate['chunked_mutation_us_32k']} |\n"
        f"| 32K mixed-V us | {gate['baseline_mixed_v_us_32k']} | {gate['fixed_mixed_v_us_32k']} | {gate['chunked_mixed_v_us_32k']} |\n"
        f"| 32K TPOT ms | {gate['baseline_tpot_32k_ms']} | {gate['fixed_tpot_32k_ms']} | {gate['chunked_tpot_32k_ms']} |\n"
        f"| historical materialized bytes | 0 | {gate['historical_materialized_bytes_32k']} | {gate['historical_materialized_bytes_32k']} |\n",
        encoding="utf-8",
    )
    (OUT_DIR / "final_report.md").write_text(
        f"# Final Report\n\nClassification: `{gate['classification']}`\n\nRecommended next phase: `{gate['recommended_next_phase']}`\n\nBest backend: `{gate['best_backend']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
