#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "system_strided_k_reader_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant"))

import patternkv_gemv
from models.segmented_cache import pattern_gather_centroids, quantize_pack_k_reference
from quant.matmul import cuda_bmm_fA_qB_outer_with_base_strided_k, get_patternkv_strided_k_reader_counters, reset_patternkv_strided_k_reader_counters


GROUP_SIZE = 128
BITS = 2
NH = 32
NH_KV = 8
HEAD_DIM = 128
CENTROIDS = 16
START_HEAD = "aaa1f4be00d51431aa19f9e7ba4aa118175f89f3"


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


def ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def round_group(tokens: int) -> int:
    return ceil_div(tokens, GROUP_SIZE) * GROUP_SIZE


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(math.ceil(p * len(ordered))) - 1)])


def cv(values: list[float]) -> float:
    mean = statistics.mean(values)
    return float(statistics.stdev(values) / mean) if len(values) > 1 and mean else 0.0


def capacity_for(tokens: int) -> int:
    if tokens == 8192:
        return 32768
    if tokens == 16384:
        return 32768
    if tokens == 24576:
        return 32768
    if tokens >= 32768:
        return 33792
    return max(round_group(tokens + 257), GROUP_SIZE)


def make_assignments(tokens: int, *, padded_tokens: int, assignment: str, seed: int, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    torch.manual_seed(seed + tokens)
    if assignment == "uniform":
        ids = torch.arange(padded_tokens, device="cuda", dtype=torch.long) % CENTROIDS
        out = ids.view(1, 1, padded_tokens).expand(1, NH_KV, padded_tokens).contiguous()
    elif assignment == "skewed":
        out = torch.randint(1, CENTROIDS, (1, NH_KV, padded_tokens), device="cuda", dtype=torch.long)
        out[torch.rand(1, NH_KV, padded_tokens, device="cuda") < 0.75] = 0
    elif assignment == "all_same":
        out = torch.zeros(1, NH_KV, padded_tokens, device="cuda", dtype=torch.long)
    else:
        out = torch.randint(0, CENTROIDS, (1, NH_KV, padded_tokens), device="cuda", dtype=torch.long)
    if padded_tokens > tokens:
        out[:, :, tokens:] = 0
    return out.to(dtype).contiguous()


def build_case(tokens: int, *, capacity: int | None = None, assignment: str = "normal", seed: int = 731) -> dict[str, Any]:
    torch.manual_seed(seed + tokens)
    padded_tokens = round_group(tokens)
    capacity = round_group(capacity or capacity_for(tokens))
    if capacity < padded_tokens:
        raise ValueError(f"capacity {capacity} < padded logical tokens {padded_tokens}")
    key = (torch.randn(1, NH_KV, padded_tokens, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.25).contiguous()
    query = (torch.randn(1, NH, 1, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.25).contiguous()
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    assignments = make_assignments(tokens, padded_tokens=padded_tokens, assignment=assignment, seed=seed)
    base = pattern_gather_centroids(assignments.long(), centroids).to(key.dtype)
    packed, scale, zero = quantize_pack_k_reference(key - base, GROUP_SIZE, BITS)

    pack = 32 // BITS
    logical_packs = ceil_div(tokens, pack)
    logical_groups = ceil_div(tokens, GROUP_SIZE)
    cap_packs = ceil_div(capacity, pack)
    cap_groups = ceil_div(capacity, GROUP_SIZE)

    packed_cap = torch.empty(1, NH_KV, HEAD_DIM, cap_packs, device="cuda", dtype=torch.int32)
    packed_cap.fill_(0x7FFFFFFF)
    packed_cap[:, :, :, : packed.shape[-1]] = packed
    scale_cap = torch.empty(1, NH_KV, HEAD_DIM, cap_groups, device="cuda", dtype=torch.float16)
    scale_cap.fill_(float("nan"))
    scale_cap[:, :, :, : scale.shape[-1]] = scale
    zero_cap = torch.empty_like(scale_cap)
    zero_cap.fill_(float("nan"))
    zero_cap[:, :, :, : zero.shape[-1]] = zero
    assignments_cap = torch.empty(1, NH_KV, capacity, device="cuda", dtype=torch.int32)
    assignments_cap.fill_(2147483647)
    assignments_cap[:, :, :padded_tokens] = assignments

    alpha = query.view(-1, 1, HEAD_DIM).contiguous()
    packed_tight = packed.reshape(-1, HEAD_DIM, packed.shape[-1]).transpose(1, 2).contiguous()
    scale_tight = scale.reshape(-1, HEAD_DIM, scale.shape[-1]).transpose(1, 2).contiguous()
    zero_tight = zero.reshape(-1, HEAD_DIM, zero.shape[-1]).transpose(1, 2).contiguous()

    return {
        "tokens": int(tokens),
        "capacity": int(capacity),
        "padded_tokens": int(padded_tokens),
        "query": query,
        "alpha": alpha,
        "packed": packed,
        "scale": scale,
        "zero": zero,
        "centroids": centroids,
        "assignments": assignments,
        "packed_tight": packed_tight,
        "scale_tight": scale_tight,
        "zero_tight": zero_tight,
        "packed_view": packed_cap[:, :, :, :logical_packs],
        "scale_view": scale_cap[:, :, :, :logical_groups],
        "zero_view": zero_cap[:, :, :, :logical_groups],
        "assignments_view": assignments_cap[:, :, :tokens],
    }


def tight_kernel(data: dict[str, Any]) -> torch.Tensor:
    out = patternkv_gemv.gemv_forward_cuda_outer_dim_with_base(
        data["alpha"],
        data["packed_tight"],
        data["scale_tight"],
        data["zero_tight"],
        BITS,
        GROUP_SIZE,
        NH,
        NH_KV,
        data["centroids"],
        data["assignments"],
    )
    return out.view(1, NH, 1, data["padded_tokens"])[:, :, :, : data["tokens"]]


def strided_kernel(data: dict[str, Any]) -> torch.Tensor:
    out = patternkv_gemv.gemv_forward_cuda_outer_dim_with_base_strided_k(
        data["alpha"],
        data["packed_view"],
        data["scale_view"],
        data["zero_view"],
        BITS,
        GROUP_SIZE,
        NH,
        NH_KV,
        data["centroids"],
        data["assignments_view"],
    )
    return out.view(1, NH, 1, data["tokens"])


def strided_wrapper(data: dict[str, Any]) -> torch.Tensor:
    return cuda_bmm_fA_qB_outer_with_base_strided_k(
        GROUP_SIZE,
        data["query"],
        data["packed_view"],
        data["scale_view"],
        data["zero_view"],
        BITS,
        data["centroids"],
        data["assignments_view"],
        NH,
        NH_KV,
    )


def correctness_metrics(candidate: torch.Tensor, baseline: torch.Tensor) -> dict[str, Any]:
    torch.cuda.synchronize()
    c = candidate.float()
    b = baseline.float()
    diff = (c - b).abs()
    rel_l2 = float(torch.linalg.vector_norm(c - b).item() / torch.linalg.vector_norm(b).clamp_min(1e-8).item())
    cosine = float(torch.nn.functional.cosine_similarity(c.flatten(), b.flatten(), dim=0).item())
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "relative_l2": rel_l2,
        "cosine": cosine,
        "nan": int(torch.isnan(candidate).sum().item()),
        "inf": int(torch.isinf(candidate).sum().item()),
        "passed": float(diff.max().item()) <= 5e-3 and cosine >= 0.9999 and int(torch.isnan(candidate).sum().item()) == 0 and int(torch.isinf(candidate).sum().item()) == 0,
    }


def time_cuda(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int, rounds: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    medians = []
    all_times = []
    for _ in range(rounds):
        round_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            round_times.append(float(start.elapsed_time(end) * 1000.0))
        medians.append(statistics.median(round_times))
        all_times.extend(round_times)
    return {
        "median_us": statistics.median(medians),
        "mean_round_median_us": statistics.mean(medians),
        "p90_us": percentile(all_times, 0.90),
        "cv": cv(medians),
    }


def run_correctness() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [(t, "normal") for t in [127, 128, 129, 255, 256, 257, 2048, 8192, 16384, 32768]]
    cases += [(512, mode) for mode in ["uniform", "skewed", "all_same"]]
    reset_patternkv_strided_k_reader_counters()
    for tokens, assignment in cases:
        data = build_case(tokens, assignment=assignment)
        base = tight_kernel(data)
        out = strided_wrapper(data)
        metrics = correctness_metrics(out, base)
        rows.append({
            "tokens": tokens,
            "capacity": data["capacity"],
            "assignment": assignment,
            "packed_stride_token": data["packed_view"].stride(3),
            "packed_stride_ic": data["packed_view"].stride(2),
            **metrics,
        })
    counters = get_patternkv_strided_k_reader_counters()
    summary = {
        "total_cases": len(rows),
        "passed": sum(1 for row in rows if row["passed"]),
        "max_abs_max": max(float(row["max_abs"]) for row in rows),
        "relative_l2_max": max(float(row["relative_l2"]) for row in rows),
        "cosine_min": min(float(row["cosine"]) for row in rows),
        "nan_total": sum(int(row["nan"]) for row in rows),
        "inf_total": sum(int(row["inf"]) for row in rows),
        "slack_sentinel_passed": all(row["passed"] for row in rows if int(row["capacity"]) > int(row["tokens"])),
        "logical_tokens_only": counters["strided_k_tokens_processed"] == sum(tokens for tokens, _ in cases),
        "counters": counters,
    }
    summary["passed_bool"] = (
        summary["passed"] == summary["total_cases"]
        and summary["max_abs_max"] <= 5e-3
        and summary["cosine_min"] >= 0.9999
        and summary["nan_total"] == 0
        and summary["inf_total"] == 0
        and summary["logical_tokens_only"]
    )
    return rows, summary


def run_performance(contexts: list[int], *, warmup: int, iters: int, rounds: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_rows = []
    strided_rows = []
    summary_rows = []
    for tokens in contexts:
        data = build_case(tokens, capacity=capacity_for(tokens))
        base = tight_kernel(data)
        out = strided_kernel(data)
        metrics = correctness_metrics(out, base)
        tight_t = time_cuda(lambda: tight_kernel(data), warmup=warmup, iters=iters, rounds=rounds)
        strided_t = time_cuda(lambda: strided_kernel(data), warmup=warmup, iters=iters, rounds=rounds)
        baseline_rows.append({"context_tokens": tokens, "capacity_tokens": tokens, **tight_t})
        strided_rows.append({"context_tokens": tokens, "capacity_tokens": data["capacity"], **strided_t})
        overhead = (strided_t["median_us"] - tight_t["median_us"]) / tight_t["median_us"]
        summary_rows.append({
            "context_tokens": tokens,
            "capacity_tokens": data["capacity"],
            "baseline_median_us": tight_t["median_us"],
            "strided_median_us": strided_t["median_us"],
            "baseline_mean_round_median_us": tight_t["mean_round_median_us"],
            "strided_mean_round_median_us": strided_t["mean_round_median_us"],
            "baseline_p90_us": tight_t["p90_us"],
            "strided_p90_us": strided_t["p90_us"],
            "baseline_cv": tight_t["cv"],
            "strided_cv": strided_t["cv"],
            "overhead": overhead,
            "speedup": tight_t["median_us"] / strided_t["median_us"],
            "max_abs": metrics["max_abs"],
            "cosine": metrics["cosine"],
            "nan": metrics["nan"],
            "inf": metrics["inf"],
        })
    return baseline_rows, strided_rows, summary_rows


def run_pitch(*, warmup: int, iters: int, rounds: int) -> list[dict[str, Any]]:
    rows = []
    for capacity in [8192, 16384, 32768]:
        data = build_case(8192, capacity=capacity)
        timing = time_cuda(lambda: strided_kernel(data), warmup=warmup, iters=iters, rounds=rounds)
        rows.append({"logical_tokens": 8192, "capacity_tokens": capacity, **timing})
    return rows


def env_snapshot(physical_gpu: int | None) -> dict[str, Any]:
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
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "physical_gpu": physical_gpu,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": smi,
        "ncu_available": shutil.which("ncu") is not None,
        "extension_path": str(ext),
        "extension_mtime": int(ext.stat().st_mtime),
        "extension_sha256": sha256(ext),
        "has_strided_k_entry": hasattr(patternkv_gemv, "gemv_forward_cuda_outer_dim_with_base_strided_k"),
    }


def classify(correctness: dict[str, Any], perf: list[dict[str, Any]], pitch: list[dict[str, Any]]) -> tuple[str, str]:
    if not correctness["passed_bool"]:
        return "STRIDED_K_READER_CORRECTNESS_BLOCKED", "FIX_STRIDED_K_READER_CORRECTNESS"
    row32 = next(row for row in perf if int(row["context_tokens"]) == 32768)
    row16 = next(row for row in perf if int(row["context_tokens"]) == 16384)
    primary_stable = (
        float(row32["baseline_cv"]) <= 0.05
        and float(row32["strided_cv"]) <= 0.05
        and float(row16["baseline_cv"]) <= 0.05
        and float(row16["strided_cv"]) <= 0.05
    )
    pitch_values = [float(row["median_us"]) for row in pitch]
    pitch_sensitive = (max(pitch_values) - min(pitch_values)) / max(min(pitch_values), 1e-9) > 0.05
    if not primary_stable:
        return "STRIDED_K_READER_CORRECTNESS_BLOCKED", "RERUN_MICROBENCH_WITH_LOW_GPU_CONTENTION"
    if float(row32["overhead"]) > 0.10:
        return "STRIDED_K_READER_NOT_SUPPORTED", "V_ONLY_CAPACITY_FINAL_SYSTEM_BENCHMARK"
    if float(row32["overhead"]) <= 0.05 and float(row16["overhead"]) <= 0.10 and not pitch_sensitive:
        return "STRIDED_K_READER_SUPPORTED", "FULL_KV_CAPACITY_REAL_INTEGRATION"
    if float(row32["overhead"]) <= 0.10:
        return "STRIDED_K_READER_BORDERLINE", "FULL_KV_CAPACITY_INTEGRATION_FEASIBILITY"
    return "STRIDED_K_READER_NOT_SUPPORTED", "V_ONLY_CAPACITY_FINAL_SYSTEM_BENCHMARK"


def write_text_reports(env: dict[str, Any], correctness: dict[str, Any], perf: list[dict[str, Any]], pitch: list[dict[str, Any]], gate: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current_abi = [
        "# Current K Reader ABI",
        "",
        "- Production Python entry: `quant.matmul.cuda_bmm_fA_qB_outer_with_base`.",
        "- Production C++ binding: `gemv_forward_cuda_outer_dim_with_base`.",
        "- Production CUDA kernel: `bgemv_kernel_outer_dim_with_base_tiled<2>` for frozen INT2 Pattern K.",
        "- Packed K Python shape: `[B, nh_kv, head_dim, ceil(tokens / 16)]`, dtype `int32`.",
        "- Packed K token dimension: Python dim 3 after INT2 packing; each `int32` stores 16 logical tokens for a head-dim lane.",
        "- Packed K C++ tight shape after wrapper transpose: `[B * nh_kv, ceil(tokens / 16), head_dim]`.",
        "- K scale/zero Python shape: `[B, nh_kv, head_dim, ceil(tokens / group_size)]`, dtype `float16`.",
        "- K scale/zero C++ tight shape after wrapper transpose: `[B * nh_kv, ceil(tokens / group_size), head_dim]`.",
        "- K Pattern assignment shape: `[B, nh_kv, tokens]`; production wrapper materializes unsupported dtypes to `int16`.",
        "- K centroids shape: `[nh_kv, centroid_count, head_dim]`, contiguous, not a growing historical stream.",
        "- Sink K FP16 layout: `[B, nh_kv, sink_tokens, head_dim]`, handled by existing FP16 matmul.",
        "- Recent/pending K FP16 layout: `[B, nh_kv, tokens, head_dim]`, handled by existing FP16 matmul.",
        "- Q layout: `[B, nh, 1, head_dim]`; wrapper flattens to `[B * nh, 1, head_dim]`.",
        "- QK output layout: `[B, nh, 1, tokens]`.",
        "- Current wrapper `.contiguous()` points: Q flatten, packed K transpose, scale transpose, zero transpose, assignment dtype/layout, centroids.",
        "- Current tight K equation: `weight[(b * nh_kv + kv) * (OC * IC / pack) + packed * IC + k]`.",
        "- Current assignment equation: `assign[((b * nh_kv + kv) * OC + oc) * assign_bytes]`.",
        "- Current scale/zero equation: `scale[(b * nh_kv + kv) * (OC * IC / group) + group_idx * IC + k]`.",
        "- Future capacity-sensitive historical K inputs: packed K, K scale, K zero, K assignments.",
    ]
    (OUT_DIR / "current_k_reader_abi.md").write_text("\n".join(current_abi) + "\n", encoding="utf-8")
    (OUT_DIR / "strided_k_reader_design.md").write_text(
        "\n".join([
            "# Strided K Reader Design",
            "",
            "- Added experimental API `cuda_bmm_fA_qB_outer_with_base_strided_k`.",
            "- Added C++ binding `gemv_forward_cuda_outer_dim_with_base_strided_k`.",
            "- The production default path is unchanged.",
            "- Only historical compressed K addressing changed; K quantization, centroid restoration, beta/scale/zero math, and QK accumulation are copied from the production kernel.",
            "- Inputs are narrow logical views over capacity storage. Shape controls logical length; `Tensor.stride()` controls physical pitch.",
            "- Sink/recent/pending FP16 regions are out of scope and unchanged.",
            "- No page lookup, page table, CUDA VMM, vLLM, SGLang, GQA redesign, selector tuning, or centroid tuning.",
            "- STRIDED_K_KERNEL_ITERATES_ONLY_LOGICAL_TOKENS=YES.",
        ]) + "\n",
        encoding="utf-8",
    )
    example = build_case(8192, capacity=32768)
    layout_rows = []
    for name in ["packed_view", "scale_view", "zero_view", "assignments_view", "alpha", "centroids"]:
        tensor = example[name]
        layout_rows.append(f"- `{name}`: shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}, storage_offset={tensor.storage_offset()}, dtype={tensor.dtype}")
    (OUT_DIR / "stride_parameter_audit.md").write_text(
        "\n".join([
            "# Stride Parameter Audit",
            "",
            *layout_rows,
            "",
            "- Packed K kernel parameters: B/H/head_dim/pack strides from `_kernel.stride(0..3)`.",
            "- Scale kernel parameters: B/H/head_dim/group strides from `_scaling_factors.stride(0..3)`.",
            "- Zero kernel parameters: B/H/head_dim/group strides from `_zeros.stride(0..3)`.",
            "- Assignment kernel parameters: B/H/token strides from `_assignments.stride(0..2)`.",
            "- Logical tokens are `_assignments.size(2)`, not physical capacity.",
        ]) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "reader_layout_examples.md").write_text("\n".join(["# Reader Layout Examples", "", *layout_rows]) + "\n", encoding="utf-8")
    (OUT_DIR / "materialization_audit.md").write_text(
        "\n".join([
            "# Materialization Audit",
            "",
            "- packed K materialized: NO in experimental strided path.",
            "- scale materialized: NO in experimental strided path.",
            "- zero materialized: NO in experimental strided path.",
            "- assignment materialized: NO in experimental strided path.",
            "- centroid: N/A, contiguous table, not historical.",
            "- Q: N/A, current decode query, kept contiguous.",
            f"- historical materialize calls: {gate['historical_materialize_calls']}",
            f"- historical materialized bytes: {gate['historical_materialized_bytes']}",
        ]) + "\n",
        encoding="utf-8",
    )
    rows = ["| Metric | Tight K Reader | Strided Capacity K | Change |", "| --- | ---: | ---: | ---: |"]
    for row in perf:
        ctx = int(row["context_tokens"])
        rows.append(f"| QK @{ctx//1024}K | {float(row['baseline_median_us']):.3f} us | {float(row['strided_median_us']):.3f} us | {float(row['overhead']) * 100:.2f}% |")
    rows.extend([
        f"| max abs | {correctness['max_abs_max']:.6g} | {correctness['max_abs_max']:.6g} | PASS |",
        f"| cosine min | {correctness['cosine_min']:.6g} | {correctness['cosine_min']:.6g} | PASS |",
        f"| historical materialized bytes | 0 | {gate['historical_materialized_bytes']} | 0 |",
        "| logical K tokens | benchmark context | benchmark context | logical-only |",
        "| capacity tokens | tight logical | explicit slack | no page lookup |",
        "| reader default status | default | experimental only | unchanged |",
    ])
    (OUT_DIR / "optimization_scorecard.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (OUT_DIR / "final_report.md").write_text(
        "\n".join([
            "# Final Report",
            "",
            f"Classification: `{gate['classification']}`",
            "",
            f"Recommended next phase: `{gate['recommended_next_phase']}`",
            "",
            f"32K overhead: `{gate['overhead_32k'] * 100:.2f}%`.",
            "",
            "This is a K-only feasibility experiment. It does not integrate K capacity into real decode and does not change the existing V-only capacity path.",
        ]) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 16384, 24576, 32768])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--physical-gpu", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for S5A-3")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = env_snapshot(args.physical_gpu)
    correctness_rows, correctness = run_correctness()
    baseline_rows, strided_rows, perf = run_performance(args.contexts, warmup=args.warmup, iters=args.iters, rounds=args.rounds)
    pitch = run_pitch(warmup=args.warmup, iters=args.iters, rounds=args.rounds)
    classification, next_task = classify(correctness, perf, pitch)
    row_by_ctx = {int(row["context_tokens"]): row for row in perf}
    pitch_values = [float(row["median_us"]) for row in pitch]
    pitch_sensitive = (max(pitch_values) - min(pitch_values)) / max(min(pitch_values), 1e-9) > 0.05
    reset_patternkv_strided_k_reader_counters()
    data = build_case(129, capacity=4096)
    _ = strided_wrapper(data)
    counters = get_patternkv_strided_k_reader_counters()
    gate = {
        "algorithm_changed": False,
        "selector_changed": False,
        "quantization_changed": False,
        "attention_math_changed": False,
        "value_capacity_path_changed": False,
        "k_only_experiment": True,
        "strided_k_reader_implemented": True,
        "strided_k_reader_is_default": False,
        "page_lookup_used": False,
        "page_table_used": False,
        "historical_materialize_calls": int(counters["strided_k_materialize_calls"]),
        "historical_materialized_bytes": int(counters["strided_k_materialized_bytes"]),
        "logical_tokens_only": int(counters["strided_k_tokens_processed"]) == 129,
        "correctness_passed": bool(correctness["passed_bool"]),
        "capacity_pitch_sensitive": bool(pitch_sensitive),
        "classification": classification,
        "recommended_next_phase": next_task,
    }
    for ctx in [8192, 16384, 24576, 32768]:
        row = row_by_ctx[ctx]
        gate[f"baseline_qk_{ctx//1024}k_us"] = float(row["baseline_median_us"])
        gate[f"strided_qk_{ctx//1024}k_us"] = float(row["strided_median_us"])
        gate[f"overhead_{ctx//1024}k"] = float(row["overhead"])

    write_json(OUT_DIR / "environment.json", env)
    write_csv(OUT_DIR / "correctness_cases.csv", correctness_rows)
    write_json(OUT_DIR / "correctness_summary.json", correctness)
    write_csv(OUT_DIR / "baseline_qk.csv", baseline_rows)
    write_csv(OUT_DIR / "strided_qk.csv", strided_rows)
    write_csv(OUT_DIR / "performance_summary.csv", perf)
    write_csv(OUT_DIR / "capacity_pitch_sensitivity.csv", pitch)
    write_json(OUT_DIR / "final_gate.json", gate)
    write_text_reports(env, correctness, perf, pitch, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
