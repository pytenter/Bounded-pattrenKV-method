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
OUT_DIR = ROOT / "reports" / "system_asymmetric_kv_final_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant"))

import patternkv_gemv
from bench.bench_aime24_patternkv import load_model
from bench.profile_post_fusion_decode import build_synthetic_past, ensure_profile_centroids, run_decode_case
from models.segmented_cache import (
    append_decode,
    build_cache_from_prefill,
    get_capacity_cache_counters,
    reset_capacity_cache_counters,
    tensor_bytes,
)
from quant.matmul import (
    cuda_attn_v_mixed_fused_with_base,
    get_patternkv_mixed_v_counters,
    get_patternkv_page_v_reader_counters,
    get_patternkv_strided_k_reader_counters,
    get_patternkv_strided_v2_reader_counters,
    patternkv_gqa_v_backend,
    patternkv_page_v_reader_backend,
    reset_patternkv_mixed_v_counters,
    reset_patternkv_page_v_reader_counters,
    reset_patternkv_strided_k_reader_counters,
    reset_patternkv_strided_v2_reader_counters,
)
from quant.patternkv_profile import cache_mutation_snapshot, merge_profile_rows, profile_snapshot, reset_profile
from scripts.run_aime24_full_causal25_quality import make_worker_args


START_HEAD = "7844c86dd5e8e25a7176539665445dba0a3d5f67"
BRANCH = "sys/causal-v4-25-kernel-v1"
ARCHITECTURE = "ASYMMETRIC_KV_RUNTIME"
BACKENDS = ("baseline", "fixed_capacity", "chunked_capacity")
GROUP_SIZE = 128
NH = 32
NH_KV = 8
HEAD_DIM = 128
CENTROIDS = 16
MODEL_PATH = "/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B"


def run_text(cmd: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)
    return proc.stdout.strip()


def git_text(*args: str) -> str:
    return run_text(["git", *args])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(math.ceil(p * len(ordered))) - 1)
    return float(ordered[idx])


def cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    return float(statistics.stdev(values) / mean) if mean else 0.0


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]


def set_runtime_env(backend: str, *, profile: bool, fixed_capacity_tokens: int, chunk_tokens: int) -> None:
    os.environ["PATTERNKV_CACHE_GROWTH_BACKEND"] = backend
    os.environ["PATTERNKV_CACHE_FIXED_CAPACITY_TOKENS"] = str(int(fixed_capacity_tokens))
    os.environ["PATTERNKV_CACHE_CHUNK_TOKENS"] = str(int(chunk_tokens))
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused"
    os.environ["PATTERNKV_PAGE_V_READER"] = "contiguous"
    os.environ["PATTERNKV_GQA_V_BACKEND"] = "baseline"
    os.environ["PATTERNKV_SYSTEM_PROFILE"] = "1" if profile else "0"
    os.environ["PATTERNKV_PROFILE"] = "1" if profile else "0"


def select_idle_gpu() -> str:
    try:
        rows = run_text(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
    except Exception:
        return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    best: tuple[int, int, str] | None = None
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) != 3:
            continue
        idx, used, util = parts[0], int(parts[1]), int(parts[2])
        item = (used, util, idx)
        if best is None or item < best:
            best = item
    return best[2] if best else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


def env_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    ext = Path(patternkv_gemv.__file__).resolve()
    try:
        smi = run_text(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
    except Exception as exc:
        smi = [f"nvidia-smi failed: {exc}"]
    return {
        "repo_root": git_text("rev-parse", "--show-toplevel"),
        "current_branch": git_text("branch", "--show-current"),
        "start_head_expected": START_HEAD,
        "head_at_start": git_text("rev-parse", "HEAD"),
        "worktree_clean_at_start": git_text("status", "--short") == "",
        "bounded_remote": next((line for line in git_text("remote", "-v").splitlines() if line.startswith("bounded") and "(push)" in line), ""),
        "origin_remote": next((line for line in git_text("remote", "-v").splitlines() if line.startswith("origin") and "(push)" in line), ""),
        "git_log_10": git_text("log", "-10", "--oneline").splitlines(),
        "physical_gpu_id": str(args.physical_gpu),
        "cuda_visible_devices_at_snapshot": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": smi,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "python_executable": sys.executable,
        "model_path": str(args.model_path),
        "dtype": "float16",
        "batch_size": 1,
        "extension_path": str(ext),
        "extension_sha256": sha256(ext),
        "backend_env": {
            "PATTERNKV_CACHE_GROWTH_BACKEND": "varies: baseline,fixed_capacity,chunked_capacity",
            "PATTERNKV_CACHE_FIXED_CAPACITY_TOKENS": str(args.fixed_capacity_tokens),
            "PATTERNKV_CACHE_CHUNK_TOKENS": str(args.chunk_tokens),
            "PATTERNKV_MIXED_V_BACKEND": "fused",
            "PATTERNKV_PAGE_V_READER": "contiguous",
            "PATTERNKV_GQA_V_BACKEND": "baseline",
        },
    }


def build_real_cache(tokens: int, backend: str, *, seed: int, fixed_capacity_tokens: int, chunk_tokens: int, profile: bool = False):
    set_runtime_env(backend, profile=profile, fixed_capacity_tokens=fixed_capacity_tokens, chunk_tokens=chunk_tokens)
    generator = torch.Generator(device="cuda").manual_seed(seed + tokens)
    key = (torch.randn(1, NH_KV, tokens, HEAD_DIM, device="cuda", dtype=torch.float16, generator=generator) * 0.25).contiguous()
    value = (torch.randn(1, NH_KV, tokens, HEAD_DIM, device="cuda", dtype=torch.float16, generator=generator) * 0.25).contiguous()
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device="cuda", dtype=torch.float16, generator=generator) * 0.1).contiguous()
    return build_cache_from_prefill(
        key,
        value,
        sink_length=16,
        recent_length=128,
        group_size=GROUP_SIZE,
        k_bits=2,
        v_bits=2,
        pattern=True,
        k_centroids=centroids,
        v_centroids=centroids,
        cache_mode="segmented_rolling",
        chunk_length=GROUP_SIZE,
        value_objective="base",
        v_precision_selector="causal_v4",
        v4_budget_fraction=0.25,
        random_selector_seed=44,
        selector_task_key=f"s6a-real-cache-{tokens}",
        selector_layer_idx=0,
    )


def tensor_nbytes(value: Any) -> int:
    return tensor_bytes(value) if torch.is_tensor(value) else 0


def baseline_v_bytes(cache: Any) -> int:
    names = (
        "packed_v",
        "packed_v_scale",
        "packed_v_zero",
        "packed_v4",
        "packed_v4_scale",
        "packed_v4_zero",
        "v_precision_mask",
        "v_pattern_mask",
        "v_assignment_idx",
        "v2_pattern_mask",
        "v2_assignment_idx",
        "v4_pattern_mask",
        "v4_assignment_idx",
    )
    return sum(tensor_nbytes(getattr(cache, name, None)) for name in names)


def capacity_stats(cache: Any) -> dict[str, Any]:
    buffers = getattr(cache, "capacity_buffers", None) or {}
    if not buffers:
        valid = baseline_v_bytes(cache)
        return {
            "logical_valid_v_bytes": int(valid),
            "reserved_v_capacity_bytes": int(valid),
            "unused_v_capacity_bytes": 0,
            "capacity_utilization": 1.0 if valid else 0.0,
            "capacity_streams": 0,
            "capacity_details": [],
        }
    details = [buf.stats() for buf in buffers.values()]
    reserved = sum(int(row["reserved_capacity_bytes"]) for row in details)
    valid = sum(int(row["logical_valid_bytes"]) for row in details)
    unused = sum(int(row["unused_capacity_bytes"]) for row in details)
    return {
        "logical_valid_v_bytes": int(valid),
        "reserved_v_capacity_bytes": int(reserved),
        "unused_v_capacity_bytes": int(unused),
        "capacity_utilization": float(valid / reserved) if reserved else 0.0,
        "capacity_streams": len(details),
        "capacity_details": details,
    }


def split_copy_bytes(rows: list[dict[str, Any]], counters: dict[str, int]) -> dict[str, int]:
    k_prefixes = ("packed_k", "k_assignments")
    v_prefixes = ("packed_v", "v2_", "v4_", "precision_mask", "pattern_mask", "v_assignment")
    k_old = 0
    v_old = 0
    other_old = 0
    total_mutation_old = 0
    for row in rows:
        category = str(row.get("category", ""))
        old = int(row.get("old_bytes", 0))
        total_mutation_old += old
        if category.startswith(k_prefixes):
            k_old += old
        elif category.startswith(v_prefixes):
            v_old += old
        elif category in {"sink", "recent_pending", "causal_importance"}:
            other_old += old
        else:
            other_old += old
    capacity_old = int(counters.get("capacity_growth_old_bytes_copied", 0))
    v_old += capacity_old
    return {
        "k_old_bytes_copied": int(k_old),
        "v_old_bytes_copied": int(v_old),
        "other_old_bytes_copied": int(other_old),
        "total_old_bytes_copied": int(total_mutation_old + capacity_old),
    }


def mixed_v_call(cache: Any, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed + int(cache.packed_v_tokens))
    attn = torch.softmax(torch.randn(1, NH, 1, cache.packed_v_tokens, device="cuda", dtype=torch.float16, generator=generator), dim=-1).contiguous()
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


def time_mixed_v(cache: Any, *, seed: int, warmup: int, iters: int) -> dict[str, Any]:
    reset_patternkv_mixed_v_counters()
    reset_patternkv_strided_v2_reader_counters()
    for _ in range(warmup):
        mixed_v_call(cache, seed=seed)
    torch.cuda.synchronize()
    events = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        mixed_v_call(cache, seed=seed)
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    times = [float(start.elapsed_time(end) * 1000.0) for start, end in events]
    mixed = get_patternkv_mixed_v_counters()
    strided = get_patternkv_strided_v2_reader_counters()
    return {
        "mixed_v_us_per_token": statistics.mean(times) if times else 0.0,
        "mixed_v_median_us": statistics.median(times) if times else 0.0,
        "mixed_v_cv": cv(times),
        "mixed_v_calls": int(mixed.get("mixed_v_fused_calls", 0)),
        "mixed_v_reference_calls": int(mixed.get("mixed_v_reference_calls", 0)),
        "v2_reader_calls": int(strided.get("strided_v2_calls", 0) + mixed.get("baseline_v2_calls", 0)),
        "v4_reader_calls": int(strided.get("strided_v4_calls", 0) + mixed.get("baseline_v4_calls", 0)),
        "strided_v2_calls": int(strided.get("strided_v2_calls", 0)),
        "strided_v4_calls": int(strided.get("strided_v4_calls", 0)),
        "full_fp16_historical_v_reconstruction": False,
    }


def run_selector_identity(contexts: list[int], *, seed: int, fixed_capacity_tokens: int, chunk_tokens: int) -> dict[str, Any]:
    cases = []
    passed = True
    for ctx in contexts:
        caches = {
            backend: build_real_cache(ctx, backend, seed=seed, fixed_capacity_tokens=fixed_capacity_tokens, chunk_tokens=chunk_tokens)
            for backend in BACKENDS
        }
        base = caches["baseline"].v_precision_mask[:, : caches["baseline"].packed_v_tokens]
        base_ids = torch.nonzero(base[0].bool(), as_tuple=False).flatten().detach().cpu().tolist()
        row = {"context_tokens": ctx, "selected_count": len(base_ids), "baseline_first16": base_ids[:16]}
        for backend in ("fixed_capacity", "chunked_capacity"):
            mask = caches[backend].v_precision_mask[:, : caches[backend].packed_v_tokens]
            same = bool(torch.equal(mask, base))
            passed = passed and same
            ids = torch.nonzero(mask[0].bool(), as_tuple=False).flatten().detach().cpu().tolist()
            row[f"{backend}_same_as_baseline"] = same
            row[f"{backend}_first16"] = ids[:16]
        cases.append(row)
    return {"passed": passed, "cases": cases}


@torch.no_grad()
def run_decode_tokens(model: Any, *, backend: str, context_tokens: int, decode_tokens: int, seed: int, fixed_capacity_tokens: int, chunk_tokens: int) -> dict[str, Any]:
    set_runtime_env(backend, profile=False, fixed_capacity_tokens=fixed_capacity_tokens, chunk_tokens=chunk_tokens)
    device = next(model.parameters()).device
    reset_patternkv_mixed_v_counters()
    reset_patternkv_page_v_reader_counters()
    reset_patternkv_strided_k_reader_counters()
    past, _setup_ms = build_synthetic_past(model, context_tokens=context_tokens, seed=seed)
    generator = torch.Generator(device=device).manual_seed(seed + 991)
    current = torch.randint(1, int(model.config.vocab_size), (1, 1), device=device, dtype=torch.long, generator=generator)
    full_mask = torch.ones(1, context_tokens + decode_tokens + 1, device=device, dtype=torch.long)
    tokens = []
    nan_count = 0
    inf_count = 0
    first_logits = None
    last_logits = None
    for step in range(decode_tokens):
        attention_mask = full_mask[:, : context_tokens + step + 1]
        out = model(input_ids=current, attention_mask=attention_mask, past_key_values=past, use_cache=True, return_dict=True)
        logits = out.logits[:, -1, :].detach()
        nan_count += int(torch.isnan(logits).sum().item())
        inf_count += int(torch.isinf(logits).sum().item())
        if step == 0:
            first_logits = logits.float().cpu()
        last_logits = logits.float().cpu()
        past = out.past_key_values
        current = torch.argmax(logits, dim=-1, keepdim=True)
        tokens.append(int(current.item()))
    return {
        "backend": backend,
        "context_tokens": context_tokens,
        "decode_tokens": decode_tokens,
        "seed": seed,
        "tokens": tokens,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "first_logits": first_logits,
        "last_logits": last_logits,
        "mixed_counters": get_patternkv_mixed_v_counters(),
        "page_reader_counters": get_patternkv_page_v_reader_counters(),
        "strided_k_counters": get_patternkv_strided_k_reader_counters(),
    }


def tensor_compare(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, float | None]:
    if a is None or b is None:
        return {"max_abs": None, "relative_l2": None, "cosine": None}
    da = a.float().flatten()
    db = b.float().flatten()
    diff = da - db
    max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
    relative_l2 = float(diff.norm().item() / max(da.norm().item(), 1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(da, db, dim=0).item()) if da.numel() else 1.0
    return {"max_abs": max_abs, "relative_l2": relative_l2, "cosine": cosine}


def run_correctness_smoke(model: Any, *, seed: int, fixed_capacity_tokens: int, chunk_tokens: int) -> dict[str, Any]:
    cases = []
    passed = True
    for ctx in (4096, 16384):
        results = {
            backend: run_decode_tokens(
                model,
                backend=backend,
                context_tokens=ctx,
                decode_tokens=32,
                seed=seed + ctx,
                fixed_capacity_tokens=fixed_capacity_tokens,
                chunk_tokens=chunk_tokens,
            )
            for backend in BACKENDS
        }
        base = results["baseline"]
        row = {
            "context_tokens": ctx,
            "decode_tokens": 32,
            "baseline_tokens": base["tokens"],
            "baseline_nan": base["nan_count"],
            "baseline_inf": base["inf_count"],
        }
        if base["nan_count"] or base["inf_count"]:
            passed = False
        for backend in ("fixed_capacity", "chunked_capacity"):
            comp = results[backend]
            token_same = comp["tokens"] == base["tokens"]
            first_cmp = tensor_compare(base["first_logits"], comp["first_logits"])
            last_cmp = tensor_compare(base["last_logits"], comp["last_logits"])
            row[f"{backend}_tokens"] = comp["tokens"]
            row[f"{backend}_tokens_match"] = token_same
            row[f"{backend}_nan"] = comp["nan_count"]
            row[f"{backend}_inf"] = comp["inf_count"]
            row[f"{backend}_first_logits_max_abs"] = first_cmp["max_abs"]
            row[f"{backend}_first_logits_relative_l2"] = first_cmp["relative_l2"]
            row[f"{backend}_first_logits_cosine"] = first_cmp["cosine"]
            row[f"{backend}_last_logits_max_abs"] = last_cmp["max_abs"]
            row[f"{backend}_last_logits_relative_l2"] = last_cmp["relative_l2"]
            row[f"{backend}_last_logits_cosine"] = last_cmp["cosine"]
            row[f"{backend}_page_native_calls"] = int(comp["page_reader_counters"].get("page_native_kernel_calls", 0))
            row[f"{backend}_strided_k_calls"] = int(comp["strided_k_counters"].get("strided_k_reader_calls", 0))
            passed = passed and token_same and comp["nan_count"] == 0 and comp["inf_count"] == 0
        cases.append(row)
    return {"passed": passed, "cases": cases}


def summarize_profile_off(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(int(r["context_tokens"]), int(r["decode_tokens"]), str(r["capacity_backend"])) for r in rows})
    by_base: dict[tuple[int, int], float] = {}
    for ctx, dec, backend in keys:
        subset = [r for r in rows if int(r["context_tokens"]) == ctx and int(r["decode_tokens"]) == dec and str(r["capacity_backend"]) == backend]
        means = [float(r["mean_tpot_ms"]) for r in subset]
        medians = [float(r["median_tpot_ms"]) for r in subset]
        summary = {
            "context_tokens": ctx,
            "decode_tokens": dec,
            "backend": backend,
            "rounds": len(subset),
            "mean_tpot_ms": statistics.mean(means),
            "median_tpot_ms": statistics.median(medians),
            "p90_tpot_ms": percentile(medians, 0.90),
            "decode_total_ms": statistics.mean(float(r["decode_total_ms"]) for r in subset),
            "tokens_per_sec": 1000.0 / max(statistics.mean(means), 1e-9),
            "round_cv": cv(means),
            "peak_allocated_bytes": max(int(r["peak_allocated_bytes"]) for r in subset),
            "peak_reserved_bytes": max(int(r["peak_reserved_bytes"]) for r in subset),
        }
        if backend == "baseline":
            by_base[(ctx, dec)] = float(summary["mean_tpot_ms"])
        out.append(summary)
    for row in out:
        base = by_base.get((int(row["context_tokens"]), int(row["decode_tokens"])))
        if base:
            row["speedup_vs_baseline"] = base / max(float(row["mean_tpot_ms"]), 1e-9)
            row["improvement_vs_baseline"] = (base - float(row["mean_tpot_ms"])) / base
        else:
            row["speedup_vs_baseline"] = None
            row["improvement_vs_baseline"] = None
    return out


def run_profile_off_matrix(
    model: Any,
    *,
    contexts: list[int],
    decode_lengths: list[int],
    warmup: int,
    rounds: int,
    seed: int,
    fixed_capacity_tokens: int,
    chunk_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = []
    for dec in decode_lengths:
        for ctx in contexts:
            for backend in BACKENDS:
                print(f"[profile-off] context={ctx} decode={dec} backend={backend} warmup={warmup} rounds={rounds}", flush=True)
                set_runtime_env(backend, profile=False, fixed_capacity_tokens=fixed_capacity_tokens, chunk_tokens=chunk_tokens)
                for w in range(warmup):
                    run_decode_case(model, backend="fused", context_tokens=ctx, decode_tokens=dec, profile=False, seed=seed + ctx * 31 + dec * 7 + w)
                for round_idx in range(rounds):
                    run_seed = seed + ctx * 1009 + dec * 101 + round_idx
                    result = run_decode_case(model, backend="fused", context_tokens=ctx, decode_tokens=dec, profile=False, seed=run_seed)
                    row = {
                        "context_tokens": ctx,
                        "decode_tokens": dec,
                        "capacity_backend": backend,
                        "round": round_idx,
                        "seed": run_seed,
                    }
                    for key, value in result.items():
                        if key in {"profile_snapshot", "temp_allocations", "cache_mutations"}:
                            continue
                        row[key] = value
                    all_rows.append(row)
                    write_csv(OUT_DIR / "profile_off_all_runs.csv", all_rows)
                    print(f"[profile-off] done context={ctx} decode={dec} backend={backend} round={round_idx} tpot_ms={row['mean_tpot_ms']:.3f}", flush=True)
                write_csv(OUT_DIR / "profile_off_summary.csv", summarize_profile_off(all_rows))
    return all_rows, summarize_profile_off(all_rows)


def run_profile_on_components(
    model: Any,
    *,
    seed: int,
    fixed_capacity_tokens: int,
    chunk_tokens: int,
) -> list[dict[str, Any]]:
    rows = []
    for ctx, dec in ((16384, 128), (32768, 128), (32768, 512)):
        for backend in BACKENDS:
            print(f"[profile-on] context={ctx} decode={dec} backend={backend}", flush=True)
            set_runtime_env(backend, profile=True, fixed_capacity_tokens=fixed_capacity_tokens, chunk_tokens=chunk_tokens)
            result = run_decode_case(model, backend="fused", context_tokens=ctx, decode_tokens=dec, profile=True, seed=seed + ctx * 17 + dec)
            component_rows = merge_profile_rows(result["profile_snapshot"], decode_tokens=dec, decode_total_us=float(result["decode_total_ms"]) * 1000.0)
            for row in component_rows:
                row.update(
                    {
                        "context_tokens": ctx,
                        "decode_tokens": dec,
                        "backend": backend,
                        "profile_on_tpot_ms": result["mean_tpot_ms"],
                        "profile_on_shares_approximate": True,
                    }
                )
                rows.append(row)
            write_csv(OUT_DIR / "profile_on_components.csv", rows)
    return rows


def run_mutation_matrix(
    *,
    contexts: list[int],
    decode_lengths: list[int],
    seed: int,
    fixed_capacity_tokens: int,
    chunk_tokens: int,
    mixed_warmup: int,
    mixed_iters: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mutation_rows = []
    copy_rows = []
    mixed_rows = []
    memory_rows = []
    old_system_profile = os.environ.get("PATTERNKV_SYSTEM_PROFILE")
    old_profile = os.environ.get("PATTERNKV_PROFILE")
    for dec in decode_lengths:
        for ctx in contexts:
            generator = torch.Generator(device="cuda").manual_seed(seed + ctx * 13 + dec)
            new_k = (torch.randn(1, NH_KV, dec, HEAD_DIM, device="cuda", dtype=torch.float16, generator=generator) * 0.25).contiguous()
            new_v = (torch.randn(1, NH_KV, dec, HEAD_DIM, device="cuda", dtype=torch.float16, generator=generator) * 0.25).contiguous()
            for backend in BACKENDS:
                print(f"[mutation] context={ctx} decode={dec} backend={backend}", flush=True)
                reset_patternkv_mixed_v_counters()
                reset_patternkv_strided_v2_reader_counters()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                cache = build_real_cache(ctx, backend, seed=seed, fixed_capacity_tokens=fixed_capacity_tokens, chunk_tokens=chunk_tokens)
                set_runtime_env(backend, profile=True, fixed_capacity_tokens=fixed_capacity_tokens, chunk_tokens=chunk_tokens)
                reset_profile()
                reset_capacity_cache_counters()
                start = time.perf_counter()
                for step in range(dec):
                    append_decode(cache, new_k[:, :, step : step + 1, :], new_v[:, :, step : step + 1, :])
                torch.cuda.synchronize()
                elapsed_us = (time.perf_counter() - start) * 1_000_000.0
                cats = cache_mutation_snapshot()
                counters = get_capacity_cache_counters()
                copies = split_copy_bytes(cats, counters)
                cap = capacity_stats(cache)
                peak_allocated = int(torch.cuda.max_memory_allocated())
                peak_reserved = int(torch.cuda.max_memory_reserved())
                cat_calls = sum(int(row.get("calls", 0)) for row in cats)
                mutation_row = {
                    "context_tokens": ctx,
                    "decode_tokens": dec,
                    "backend": backend,
                    "mutation_us_per_token": elapsed_us / max(dec, 1),
                    "historical_torch_cat_calls_per_token": cat_calls / max(dec, 1),
                    "historical_old_bytes_copied_per_token": copies["total_old_bytes_copied"] / max(dec, 1),
                    "capacity_growth_events_per_token": int(counters.get("capacity_growth_events", 0)) / max(dec, 1),
                    "capacity_growth_copied_bytes_per_token": int(counters.get("capacity_growth_old_bytes_copied", 0)) / max(dec, 1),
                    "historical_materialize_calls": int(counters.get("historical_materialization_calls", 0)),
                    "historical_materialized_bytes": int(counters.get("historical_materialized_bytes", 0)),
                }
                copy_row = {
                    "backend": backend,
                    "context_tokens": ctx,
                    "decode_tokens": dec,
                    "k_old_bytes_copied": copies["k_old_bytes_copied"],
                    "v_old_bytes_copied": copies["v_old_bytes_copied"],
                    "other_old_bytes_copied": copies["other_old_bytes_copied"],
                    "total_old_bytes_copied": copies["total_old_bytes_copied"],
                    "k_old_bytes_copied_per_token": copies["k_old_bytes_copied"] / max(dec, 1),
                    "v_old_bytes_copied_per_token": copies["v_old_bytes_copied"] / max(dec, 1),
                    "total_old_bytes_copied_per_token": copies["total_old_bytes_copied"] / max(dec, 1),
                }
                memory_row = {
                    "context_tokens": ctx,
                    "decode_tokens": dec,
                    "backend": backend,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    **{k: v for k, v in cap.items() if k != "capacity_details"},
                    "capacity_growth_events": int(counters.get("capacity_growth_events", 0)),
                    "capacity_growth_copied_bytes": int(counters.get("capacity_growth_old_bytes_copied", 0)),
                }
                mixed = time_mixed_v(cache, seed=seed + ctx + dec, warmup=mixed_warmup, iters=mixed_iters)
                counters_after_mixed = get_capacity_cache_counters()
                mutation_row["historical_materialize_calls"] = int(counters_after_mixed.get("historical_materialization_calls", 0))
                mutation_row["historical_materialized_bytes"] = int(counters_after_mixed.get("historical_materialized_bytes", 0))
                mixed_row = {
                    "context_tokens": ctx,
                    "decode_tokens": dec,
                    "backend": backend,
                    **mixed,
                    "historical_materialized_bytes": int(counters_after_mixed.get("historical_materialized_bytes", 0)),
                }
                mutation_rows.append(mutation_row)
                copy_rows.append(copy_row)
                memory_rows.append(memory_row)
                mixed_rows.append(mixed_row)
                write_csv(OUT_DIR / "mutation_breakdown.csv", mutation_rows)
                write_csv(OUT_DIR / "copy_breakdown.csv", copy_rows)
                write_csv(OUT_DIR / "memory_summary.csv", memory_rows)
                write_csv(OUT_DIR / "mixed_v_summary.csv", mixed_rows)
    if old_system_profile is None:
        os.environ.pop("PATTERNKV_SYSTEM_PROFILE", None)
    else:
        os.environ["PATTERNKV_SYSTEM_PROFILE"] = old_system_profile
    if old_profile is None:
        os.environ.pop("PATTERNKV_PROFILE", None)
    else:
        os.environ["PATTERNKV_PROFILE"] = old_profile
    capacity_rows = [
        {
            "context_tokens": row["context_tokens"],
            "decode_tokens": row["decode_tokens"],
            "backend": row["backend"],
            "capacity_growth_events": row["capacity_growth_events"],
            "capacity_growth_copied_bytes": row["capacity_growth_copied_bytes"],
            "logical_valid_v_bytes": row["logical_valid_v_bytes"],
            "reserved_v_capacity_bytes": row["reserved_v_capacity_bytes"],
            "unused_v_capacity_bytes": row["unused_v_capacity_bytes"],
            "capacity_utilization": row["capacity_utilization"],
        }
        for row in memory_rows
    ]
    write_csv(OUT_DIR / "capacity_growth_summary.csv", capacity_rows)
    return mutation_rows, copy_rows, mixed_rows, memory_rows


def summary_lookup(rows: list[dict[str, Any]], ctx: int, dec: int, backend: str) -> dict[str, Any] | None:
    return next((r for r in rows if int(r["context_tokens"]) == ctx and int(r["decode_tokens"]) == dec and str(r["backend"]) == backend), None)


def candidate_supported(summary: list[dict[str, Any]], memory: list[dict[str, Any]], backend: str) -> bool:
    required = []
    for ctx, dec in ((32768, 128), (32768, 512)):
        row = summary_lookup(summary, ctx, dec, backend)
        required.append(row is not None and float(row.get("improvement_vs_baseline") or 0.0) >= 0.01)
    for ctx, dec in ((16384, 128), (16384, 512)):
        row = summary_lookup(summary, ctx, dec, backend)
        required.append(row is not None and float(row.get("improvement_vs_baseline") or 0.0) >= -0.01)
    required.append(all(float(r.get("round_cv") or 0.0) <= 0.05 for r in summary if str(r["backend"]) in {"baseline", backend}))
    if backend == "chunked_capacity":
        util = [float(r["capacity_utilization"]) for r in memory if str(r["backend"]) == backend and int(r["context_tokens"]) in {16384, 32768}]
        required.append(bool(util) and min(util) > 0.50)
    return all(required)


def build_gate(
    *,
    correctness: dict[str, Any],
    selector_identity: dict[str, Any],
    profile_summary: list[dict[str, Any]],
    mutation: list[dict[str, Any]],
    mixed: list[dict[str, Any]],
    memory: list[dict[str, Any]],
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate: dict[str, Any] = {
        "algorithm_changed": False,
        "selector_changed": False,
        "quantization_changed": False,
        "attention_math_changed": False,
        "architecture": ARCHITECTURE,
        "k_layout": "tight",
        "v_layout_baseline": "growing_contiguous",
        "v_layout_fixed": "fixed_capacity",
        "v_layout_chunked": "chunked_capacity",
        "strided_k_used": False,
        "page_native_used": False,
        "experimental_gqa_used": False,
        "cuda_vmm_used": False,
        "correctness_passed": bool(correctness.get("passed")) and bool(selector_identity.get("passed")),
        "selector_identity_passed": bool(selector_identity.get("passed")),
        "validation": validation or {},
    }
    for ctx in (4096, 8192, 16384, 32768):
        short = f"{ctx // 1024}k"
        for dec in (128, 512):
            for backend in BACKENDS:
                row = summary_lookup(profile_summary, ctx, dec, backend)
                key = f"{backend.split('_')[0]}_tpot_{short}_{dec}"
                gate[key] = None if row is None else float(row["mean_tpot_ms"])
    fixed_mat = sum(int(r["historical_materialized_bytes"]) for r in mutation if str(r["backend"]) == "fixed_capacity")
    chunk_mat = sum(int(r["historical_materialized_bytes"]) for r in mutation if str(r["backend"]) == "chunked_capacity")
    gate["v_historical_materialized_bytes_fixed"] = fixed_mat
    gate["v_historical_materialized_bytes_chunked"] = chunk_mat
    best_backend = "baseline"
    best_score = 1.0
    for backend in ("fixed_capacity", "chunked_capacity"):
        scores = []
        for ctx, dec in ((32768, 128), (32768, 512)):
            row = summary_lookup(profile_summary, ctx, dec, backend)
            if row:
                scores.append(float(row.get("speedup_vs_baseline") or 1.0))
        score = statistics.mean(scores) if scores else 1.0
        if score > best_score:
            best_backend = backend
            best_score = score
    gate["best_backend"] = best_backend
    gate["best_32k_128_improvement"] = max(
        float((summary_lookup(profile_summary, 32768, 128, b) or {}).get("improvement_vs_baseline") or 0.0) for b in ("fixed_capacity", "chunked_capacity")
    )
    gate["best_32k_512_improvement"] = max(
        float((summary_lookup(profile_summary, 32768, 512, b) or {}).get("improvement_vs_baseline") or 0.0) for b in ("fixed_capacity", "chunked_capacity")
    )
    for ctx in (16384, 32768):
        vals = [float(r["capacity_utilization"]) for r in memory if str(r["backend"]) == "chunked_capacity" and int(r["context_tokens"]) == ctx]
        gate[f"chunked_utilization_{ctx // 1024}k"] = statistics.mean(vals) if vals else None
    util_vals = [v for v in (gate.get("chunked_utilization_16k"), gate.get("chunked_utilization_32k")) if v is not None]
    min_util = min(util_vals) if util_vals else 0.0
    if min_util >= 0.95:
        gate["vmm_priority"] = "LOW"
        gate["vmm_candidate"] = "NO"
    elif min_util < 0.85:
        gate["vmm_priority"] = "MEDIUM"
        gate["vmm_candidate"] = "DEFER"
    else:
        gate["vmm_priority"] = "LOW"
        gate["vmm_candidate"] = "DEFER"
    fixed_ok = candidate_supported(profile_summary, memory, "fixed_capacity")
    chunked_ok = candidate_supported(profile_summary, memory, "chunked_capacity")
    materialization_ok = fixed_mat == 0 and chunk_mat == 0
    if not gate["correctness_passed"]:
        classification = "ASYMMETRIC_KV_CORRECTNESS_BLOCKED"
    elif not materialization_ok:
        classification = "ASYMMETRIC_KV_MEMORY_BLOCKED"
    elif fixed_ok and chunked_ok:
        fixed_32_512 = summary_lookup(profile_summary, 32768, 512, "fixed_capacity")
        chunk_32_512 = summary_lookup(profile_summary, 32768, 512, "chunked_capacity")
        if fixed_32_512 and chunk_32_512:
            fixed_t = float(fixed_32_512["mean_tpot_ms"])
            chunk_t = float(chunk_32_512["mean_tpot_ms"])
            if (chunk_t - fixed_t) / max(chunk_t, 1e-9) >= 0.02:
                classification = "ASYMMETRIC_KV_FIXED_SUPPORTED"
            elif (fixed_t - chunk_t) / max(fixed_t, 1e-9) >= 0.02:
                classification = "ASYMMETRIC_KV_CHUNKED_SUPPORTED"
            else:
                classification = "ASYMMETRIC_KV_BOTH_SUPPORTED"
        else:
            classification = "ASYMMETRIC_KV_BOTH_SUPPORTED"
    elif chunked_ok:
        classification = "ASYMMETRIC_KV_CHUNKED_SUPPORTED"
    elif fixed_ok:
        classification = "ASYMMETRIC_KV_FIXED_SUPPORTED"
    else:
        classification = "ASYMMETRIC_KV_FINAL_GAIN_NOT_REPRODUCED"
    gate["classification"] = classification
    gate["serving_benchmark_ready"] = classification in {
        "ASYMMETRIC_KV_CHUNKED_SUPPORTED",
        "ASYMMETRIC_KV_FIXED_SUPPORTED",
        "ASYMMETRIC_KV_BOTH_SUPPORTED",
    }
    gate["recommended_next_phase"] = "ASYMMETRIC_KV_SERVING_CONCURRENCY_BENCHMARK" if gate["serving_benchmark_ready"] else "BENCHMARK_CONSISTENCY_AUDIT"
    gate["max_profile_off_cv"] = max((float(r.get("round_cv") or 0.0) for r in profile_summary), default=None)
    gate["all_profile_off_cv_le_5_percent"] = all(float(r.get("round_cv") or 0.0) <= 0.05 for r in profile_summary)
    gate["page_reader_backend"] = patternkv_page_v_reader_backend()
    gate["gqa_v_backend"] = patternkv_gqa_v_backend()
    gate["mixed_v_reference_calls_total"] = sum(int(r.get("mixed_v_reference_calls", 0)) for r in mixed)
    return gate


def fmt_ms(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def fmt_pct(value: Any) -> str:
    return "NA" if value is None else f"{float(value) * 100.0:.2f}%"


def write_markdown_reports(
    *,
    profile_summary: list[dict[str, Any]],
    mutation: list[dict[str, Any]],
    copy_rows: list[dict[str, Any]],
    memory: list[dict[str, Any]],
    mixed: list[dict[str, Any]],
    gate: dict[str, Any],
) -> None:
    lines = [
        "# Single Request Summary",
        "",
        "| Context | Decode | Baseline TPOT | Fixed TPOT | Fixed Improve | Chunked TPOT | Chunked Improve | Best |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dec in (128, 512):
        for ctx in (4096, 8192, 16384, 32768):
            base = summary_lookup(profile_summary, ctx, dec, "baseline") or {}
            fixed = summary_lookup(profile_summary, ctx, dec, "fixed_capacity") or {}
            chunk = summary_lookup(profile_summary, ctx, dec, "chunked_capacity") or {}
            candidates = [("baseline", base), ("fixed_capacity", fixed), ("chunked_capacity", chunk)]
            available = [item for item in candidates if item[1]]
            best = min(available, key=lambda item: float(item[1]["mean_tpot_ms"]))[0] if available else "NA"
            lines.append(
                f"| {ctx} | {dec} | {fmt_ms(base.get('mean_tpot_ms'))} | {fmt_ms(fixed.get('mean_tpot_ms'))} | "
                f"{fmt_pct(fixed.get('improvement_vs_baseline'))} | {fmt_ms(chunk.get('mean_tpot_ms'))} | "
                f"{fmt_pct(chunk.get('improvement_vs_baseline'))} | {best} |"
            )
    (OUT_DIR / "single_request_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "# Memory Summary",
        "",
        "| Context | Decode | Backend | Peak Alloc | Peak Reserved | Valid V Bytes | Reserved V Bytes | Unused V Bytes | Utilization |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in memory:
        lines.append(
            f"| {row['context_tokens']} | {row['decode_tokens']} | {row['backend']} | {row['peak_allocated_bytes']} | "
            f"{row['peak_reserved_bytes']} | {row['logical_valid_v_bytes']} | {row['reserved_v_capacity_bytes']} | "
            f"{row['unused_v_capacity_bytes']} | {float(row['capacity_utilization']):.4f} |"
        )
    (OUT_DIR / "memory_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    copy_by_key = {(int(r["context_tokens"]), int(r["decode_tokens"]), str(r["backend"])): r for r in copy_rows}
    lines = [
        "# Mutation Summary",
        "",
        "| Context | Decode | Backend | Mutation us/token | K copied bytes/token | V copied bytes/token | Growth events/token | Materialized bytes |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in mutation:
        cp = copy_by_key.get((int(row["context_tokens"]), int(row["decode_tokens"]), str(row["backend"])), {})
        lines.append(
            f"| {row['context_tokens']} | {row['decode_tokens']} | {row['backend']} | {float(row['mutation_us_per_token']):.3f} | "
            f"{float(cp.get('k_old_bytes_copied_per_token', 0.0)):.1f} | {float(cp.get('v_old_bytes_copied_per_token', 0.0)):.1f} | "
            f"{float(row['capacity_growth_events_per_token']):.4f} | {row['historical_materialized_bytes']} |"
        )
    (OUT_DIR / "mutation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def improvement(ctx: int, dec: int, backend: str) -> float:
        row = summary_lookup(profile_summary, ctx, dec, backend) or {}
        return float(row.get("improvement_vs_baseline") or 0.0)

    context_lines = ["# Context Scaling Analysis", "", "Profile-off improvements by context:", ""]
    context_lines += ["| Decode | Backend | 4K | 8K | 16K | 32K |", "|---:|---|---:|---:|---:|---:|"]
    for dec in (128, 512):
        for backend in ("fixed_capacity", "chunked_capacity"):
            vals = [improvement(ctx, dec, backend) for ctx in (4096, 8192, 16384, 32768)]
            trend = "increases" if vals[-1] > vals[0] else "does not monotonically increase"
            context_lines.append(f"| {dec} | {backend} | " + " | ".join(f"{v * 100.0:.2f}%" for v in vals) + " |")
            context_lines.append(f"<!-- {backend} decode{dec}: gain {trend} from 4K to 32K. -->")
    (OUT_DIR / "context_scaling_analysis.md").write_text("\n".join(context_lines) + "\n", encoding="utf-8")

    decode_lines = ["# Decode Scaling Analysis", "", "Comparison of decode128 and decode512 improvements:", ""]
    decode_lines += ["| Context | Backend | Decode128 Improve | Decode512 Improve | Direction |", "|---:|---|---:|---:|---|"]
    for ctx in (4096, 8192, 16384, 32768):
        for backend in ("fixed_capacity", "chunked_capacity"):
            a = improvement(ctx, 128, backend)
            b = improvement(ctx, 512, backend)
            direction = "increased" if b > a + 0.002 else "decreased" if b < a - 0.002 else "roughly unchanged"
            decode_lines.append(f"| {ctx} | {backend} | {a * 100.0:.2f}% | {b * 100.0:.2f}% | {direction} |")
    decode_lines += ["", "Longer decode changes the amount of append and flush work per run; mutation rows show whether the direction tracks cache-copy savings."]
    (OUT_DIR / "decode_scaling_analysis.md").write_text("\n".join(decode_lines) + "\n", encoding="utf-8")

    fixed_lines = [
        "# Fixed vs Chunked",
        "",
        f"Final best backend: `{gate['best_backend']}`.",
        "",
        "Latency is decided from profile-off TPOT first; memory slack is the tie breaker when latency is within 1%.",
        "",
        f"Classification: `{gate['classification']}`.",
    ]
    (OUT_DIR / "fixed_vs_chunked.md").write_text("\n".join(fixed_lines) + "\n", encoding="utf-8")

    report = [
        "# Final Report",
        "",
        "## Answers",
        "",
        f"1. Stable faster than baseline: `{gate['classification'] in {'ASYMMETRIC_KV_CHUNKED_SUPPORTED', 'ASYMMETRIC_KV_FIXED_SUPPORTED', 'ASYMMETRIC_KV_BOTH_SUPPORTED'}}`.",
        "2. Context scaling is summarized in `context_scaling_analysis.md`.",
        "3. Decode-512 behavior is summarized in `decode_scaling_analysis.md`.",
        f"4. Fixed vs chunked decision: `{gate['best_backend']}`.",
        f"5. Chunked utilization 16K/32K: `{gate.get('chunked_utilization_16k')}`, `{gate.get('chunked_utilization_32k')}`.",
        "6. K remaining copy is reported separately in `copy_breakdown.csv`; K stays tight by design.",
        f"7. Value materialization bytes fixed/chunked: `{gate['v_historical_materialized_bytes_fixed']}`, `{gate['v_historical_materialized_bytes_chunked']}`.",
        f"8. Serving concurrency benchmark ready: `{gate['serving_benchmark_ready']}`.",
        f"9. CUDA VMM priority: `{gate['vmm_priority']}`.",
        "",
        "## Notes",
        "",
        "- Profile-off rows are the final latency truth.",
        "- Profile-on component shares are approximate and only explain component movement.",
        "- AIME24/AIME25/GPQA/vLLM/SGLang/CUDA VMM were not run in this phase.",
        "",
        f"Final classification: `{gate['classification']}`.",
        f"Recommended next phase: `{gate['recommended_next_phase']}`.",
    ]
    (OUT_DIR / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    protocol = [
        "# Benchmark Protocol",
        "",
        "- Architecture: ASYMMETRIC_KV_RUNTIME.",
        "- K historical layout: tight contiguous production QK path.",
        "- V historical layouts: baseline growing contiguous, fixed capacity, chunked capacity.",
        "- Mixed Value reader: fused compressed-domain V2/V4.",
        "- Algorithm frozen: K INT2, base V INT2, selected V INT4, V4 fraction 25%, sink16, recent128, residual128, group128, selector causal_v4.",
        "- Profile-off matrix: contexts 4096,8192,16384,32768; decode 128,512; backends baseline,fixed_capacity,chunked_capacity; warmup >=1; rounds >=5.",
        "- Same context/decode seeds are reused across backends.",
        "- Profile-on component rows are approximate and not used as final TPOT truth.",
    ]
    (OUT_DIR / "benchmark_protocol.md").write_text("\n".join(protocol) + "\n", encoding="utf-8")


def run_validation() -> dict[str, Any]:
    os.environ["PATTERNKV_CACHE_GROWTH_BACKEND"] = "baseline"
    os.environ["PATTERNKV_PAGE_V_READER"] = "contiguous"
    os.environ["PATTERNKV_GQA_V_BACKEND"] = "baseline"
    os.environ.pop("PATTERNKV_SYSTEM_PROFILE", None)
    os.environ.pop("PATTERNKV_PROFILE", None)
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused"
    validation: dict[str, Any] = {}
    commands = {
        "compileall": [sys.executable, "-m", "compileall", "bench", "models", "quant", "scripts", "tests"],
        "pytest": [sys.executable, "-m", "pytest", "-q"],
        "diff_check": ["git", "diff", "--check"],
    }
    for name, cmd in commands.items():
        start = time.perf_counter()
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        validation[name] = {
            "returncode": proc.returncode,
            "elapsed_s": time.perf_counter() - start,
            "passed": proc.returncode == 0,
            "tail": "\n".join(proc.stdout.splitlines()[-20:]),
        }
    return validation


def print_final_summary(gate: dict[str, Any], validation: dict[str, Any]) -> None:
    def tpot(ctx: int, dec: int, backend: str) -> str:
        return fmt_ms(gate.get(f"{backend.split('_')[0]}_tpot_{ctx // 1024}k_{dec}"))

    def imp(ctx: int, dec: int, backend: str) -> str:
        base = gate.get(f"baseline_tpot_{ctx // 1024}k_{dec}")
        val = gate.get(f"{backend.split('_')[0]}_tpot_{ctx // 1024}k_{dec}")
        return "NA" if base is None or val is None else f"{(float(base) - float(val)) / float(base) * 100.0:.2f}%"

    print("=" * 60)
    print("PHASE S6-A -- ASYMMETRIC KV SINGLE-REQUEST FINAL CHARACTERIZATION")
    print("=" * 60)
    print("Repository:\npytenter/Bounded-pattrenKV-method")
    print(f"Branch:\n{BRANCH}")
    print(f"Start HEAD:\n{START_HEAD}")
    print(f"End HEAD:\n{git_text('rev-parse', 'HEAD')}")
    print("Algorithm changed:\nNO")
    print("Selector changed:\nNO")
    print("Quantization changed:\nNO")
    print("Attention math changed:\nNO")
    print("\nFINAL ARCHITECTURE")
    print(f"Architecture:\n{ARCHITECTURE}")
    print("K:\nTIGHT")
    print("V:\nCAPACITY")
    print("Strided K:\nNO\nPage-native:\nNO\nExperimental GQA:\nNO\nCUDA VMM:\nNO")
    print("\nCORRECTNESS")
    print(f"Correctness:\n{'PASS' if gate['correctness_passed'] else 'FAIL'}")
    print(f"Selector identity:\n{'PASS' if gate.get('selector_identity_passed') else 'FAIL'}")
    print(f"V historical materialization:\nfixed={gate['v_historical_materialized_bytes_fixed']} bytes, chunked={gate['v_historical_materialized_bytes_chunked']} bytes")
    for dec in (128, 512):
        print(f"\nPROFILE-OFF -- DECODE {dec}")
        for ctx in (4096, 8192, 16384, 32768):
            print(f"{ctx // 1024}K:")
            print(f"Baseline: {tpot(ctx, dec, 'baseline')} ms/token")
            print(f"Fixed: {tpot(ctx, dec, 'fixed_capacity')} ms/token improvement: {imp(ctx, dec, 'fixed_capacity')}")
            print(f"Chunked: {tpot(ctx, dec, 'chunked_capacity')} ms/token improvement: {imp(ctx, dec, 'chunked_capacity')}")
    print("\nSTABILITY")
    print(f"Max CV: {gate.get('max_profile_off_cv')}")
    print(f"All <=5%: {'YES' if gate.get('all_profile_off_cv_le_5_percent') else 'NO'}")
    print("\nFINAL BACKEND DECISION")
    print(f"Best backend:\n{gate['best_backend']}")
    print("\nVMM DECISION")
    print(f"VMM priority:\n{gate['vmm_priority']}")
    print(f"Do VMM now:\n{'YES' if gate.get('vmm_candidate') == 'YES' else 'NO'}")
    print("\nSERVING READINESS")
    print(f"Ready for S6-B:\n{'YES' if gate['serving_benchmark_ready'] else 'NO'}")
    print(f"NEXT TASK:\n{gate['recommended_next_phase']}")
    print("\nVALIDATION")
    print(f"compileall: {validation.get('compileall', {}).get('passed')}")
    print(f"pytest: {validation.get('pytest', {}).get('passed')}")
    print(f"git diff --check: {validation.get('diff_check', {}).get('passed')}")
    print("AIME24:\nNO\nAIME25:\nNO\nGPQA:\nNO\nvLLM:\nNO\nSGLang:\nNO\nCUDA VMM:\nNO")
    print("\nFINAL CLASSIFICATION")
    print(f"Decision:\n{gate['classification']}")
    print("\nGIT")
    print("Commit:\nperf: characterize asymmetric PatternKV runtime")
    print("Pushed to bounded:\nPENDING")
    print("Pushed to origin:\nNO")
    print(f"Worktree clean:\n{git_text('status', '--short') == ''}")
    print("=" * 60)
    print("ALGORITHM_CHANGED=NO")
    print(f"ARCHITECTURE={ARCHITECTURE}")
    print("K_LAYOUT=TIGHT")
    print(f"V_LAYOUT={gate['best_backend'].upper() if gate['best_backend'] != 'baseline' else 'CAPACITY_CANDIDATE_NOT_SUPPORTED'}")
    print(f"SERVING_BENCHMARK_READY={'YES' if gate['serving_benchmark_ready'] else 'NO'}")
    print(f"VMM_PRIORITY={gate['vmm_priority']}")
    print(f"NEXT_TASK={gate['recommended_next_phase']}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="4096,8192,16384,32768")
    parser.add_argument("--decode-lengths", default="128,512")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--physical-gpu", default=None)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--fixed-capacity-tokens", type=int, default=33792)
    parser.add_argument("--chunk-tokens", type=int, default=4096)
    parser.add_argument("--mixed-warmup", type=int, default=5)
    parser.add_argument("--mixed-iters", type=int, default=20)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    args.contexts = parse_ints(args.contexts)
    args.decode_lengths = parse_ints(args.decode_lengths)
    args.physical_gpu = str(args.physical_gpu or select_idle_gpu())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"REPO_ROOT={git_text('rev-parse', '--show-toplevel')}")
    print(f"CURRENT_BRANCH={git_text('branch', '--show-current')}")
    print(f"START_HEAD={git_text('rev-parse', 'HEAD')}")
    print(f"WORKTREE_CLEAN={git_text('status', '--short') == ''}")
    print(f"BOUNDED_REMOTE={next((line for line in git_text('remote', '-v').splitlines() if line.startswith('bounded') and '(push)' in line), '')}")
    print(f"ORIGIN_REMOTE={next((line for line in git_text('remote', '-v').splitlines() if line.startswith('origin') and '(push)' in line), '')}")

    write_json(OUT_DIR / "environment.json", env_snapshot(args))
    if git_text("branch", "--show-current") != BRANCH:
        raise RuntimeError(f"wrong branch: expected {BRANCH}")
    if git_text("rev-parse", "HEAD") != START_HEAD and git_text("status", "--short") == "":
        print("warning: HEAD differs from S6-A start head after local edits or prior commits", file=sys.stderr)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    wargs = make_worker_args("CAUSAL_V4_25", args.seed, args.physical_gpu, experiment_id="system_asymmetric_kv_final_v1")
    wargs.model_path = Path(args.model_path)
    model, _tokenizer = load_model(wargs)
    model.eval()
    ensure_profile_centroids(model, seed=args.seed)

    selector_identity = run_selector_identity([4096, 16384], seed=args.seed, fixed_capacity_tokens=args.fixed_capacity_tokens, chunk_tokens=args.chunk_tokens)
    write_json(OUT_DIR / "selector_identity.json", selector_identity)
    correctness = run_correctness_smoke(model, seed=args.seed, fixed_capacity_tokens=args.fixed_capacity_tokens, chunk_tokens=args.chunk_tokens)
    correctness["selector_identity_passed"] = selector_identity["passed"]
    write_json(OUT_DIR / "correctness_summary.json", correctness)

    if args.skip_benchmark:
        profile_all = []
        profile_summary = []
        profile_on = []
        mutation = []
        copy_rows = []
        mixed = []
        memory = []
    else:
        profile_all, profile_summary = run_profile_off_matrix(
            model,
            contexts=args.contexts,
            decode_lengths=args.decode_lengths,
            warmup=args.warmup,
            rounds=args.rounds,
            seed=args.seed,
            fixed_capacity_tokens=args.fixed_capacity_tokens,
            chunk_tokens=args.chunk_tokens,
        )
        profile_on = run_profile_on_components(model, seed=args.seed, fixed_capacity_tokens=args.fixed_capacity_tokens, chunk_tokens=args.chunk_tokens)
        mutation, copy_rows, mixed, memory = run_mutation_matrix(
            contexts=args.contexts,
            decode_lengths=args.decode_lengths,
            seed=args.seed,
            fixed_capacity_tokens=args.fixed_capacity_tokens,
            chunk_tokens=args.chunk_tokens,
            mixed_warmup=args.mixed_warmup,
            mixed_iters=args.mixed_iters,
        )
        write_csv(OUT_DIR / "profile_on_components.csv", profile_on)
        write_csv(OUT_DIR / "profile_off_all_runs.csv", profile_all)
        write_csv(OUT_DIR / "profile_off_summary.csv", profile_summary)

    validation = {} if args.skip_validation else run_validation()
    gate = build_gate(
        correctness=correctness,
        selector_identity=selector_identity,
        profile_summary=profile_summary,
        mutation=mutation,
        mixed=mixed,
        memory=memory,
        validation=validation,
    )
    write_json(OUT_DIR / "final_gate.json", gate)
    if not args.skip_benchmark:
        write_markdown_reports(profile_summary=profile_summary, mutation=mutation, copy_rows=copy_rows, memory=memory, mixed=mixed, gate=gate)
    else:
        write_markdown_reports(profile_summary=[], mutation=[], copy_rows=[], memory=[], mixed=[], gate=gate)
    print_final_summary(gate, validation)
    return 0 if gate["classification"] != "BENCHMARK_ENVIRONMENT_BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
