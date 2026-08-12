#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant"))

from bench.bench_mixed_v_kernel_perf import build_case, metrics, mixed_output, reference_output
from bench.bench_aime24_patternkv import load_model
from bench.profile_post_fusion_decode import ensure_profile_centroids, run_decode_case
from quant.matmul import cuda_attn_v_fused_with_base
from scripts.run_aime24_full_causal25_quality import make_worker_args


OUT_DIR = ROOT / "reports/system_kernel_v2"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return float(ordered[idx])


def time_cuda(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> tuple[list[float], torch.Tensor]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for idx in range(iters):
        starts[idx].record()
        out = fn()
        ends[idx].record()
    torch.cuda.synchronize()
    times = [float(starts[idx].elapsed_time(ends[idx]) * 1000.0) for idx in range(iters)]
    assert out is not None
    return times, out


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_us": statistics.median(values),
        "mean_us": statistics.mean(values),
        "p90_us": percentile(values, 0.90),
    }


def lane_inputs(data: dict[str, Any], *, high: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    precision = data["precision"][0].bool()
    mask = precision if high else ~precision
    return (
        data["attn"][..., mask].contiguous(),
        data["v_pattern_mask"][:, :, mask].contiguous(),
        data["v_idx"][:, :, mask].contiguous(),
    )


def run_kernel_bench(*, contexts: list[int], warmup: int, iters: int, label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    correctness_rows = []
    for context in contexts:
        tokens = ((context - 16 - 128) // 128) * 128
        data = build_case("mixed25", tokens, seed=20260814 + context)
        p2 = data["p2"]
        p4 = data["p4"]
        attn2, mask2, idx2 = lane_inputs(data, high=False)
        attn4, mask4, idx4 = lane_inputs(data, high=True)
        v2_tokens = int(attn2.shape[-1])
        v4_tokens = int(attn4.shape[-1])
        v2_times, _ = time_cuda(
            lambda: cuda_attn_v_fused_with_base(
                128, attn2, p2[0], p2[1], p2[2], 2, data["centroids"], mask2, idx2, nh=32, nh_kv=8
            ),
            warmup=warmup,
            iters=iters,
        )
        v4_times, _ = time_cuda(
            lambda: cuda_attn_v_fused_with_base(
                128, attn4, p4[0], p4[1], p4[2], 4, data["centroids"], mask4, idx4, nh=32, nh_kv=8
            ),
            warmup=warmup,
            iters=iters,
        )
        mixed_times, fused = time_cuda(lambda: mixed_output(data), warmup=warmup, iters=iters)
        ref = reference_output(data)
        m = metrics(fused, ref)
        correctness_rows.append({"context_tokens": context, "v2_tokens": v2_tokens, "v4_tokens": v4_tokens, **m})
        v2 = summarize(v2_times)
        v4 = summarize(v4_times)
        mixed = summarize(mixed_times)
        rows.append(
            {
                "label": label,
                "context_tokens": context,
                "v2_tokens": v2_tokens,
                "v4_tokens": v4_tokens,
                "v2_kernel_median_us": v2["median_us"],
                "v2_kernel_mean_us": v2["mean_us"],
                "v2_kernel_p90_us": v2["p90_us"],
                "v4_kernel_median_us": v4["median_us"],
                "v4_kernel_mean_us": v4["mean_us"],
                "v4_kernel_p90_us": v4["p90_us"],
                "mixed_v_median_us": mixed["median_us"],
                "mixed_v_mean_us": mixed["mean_us"],
                "mixed_v_p90_us": mixed["p90_us"],
                "kernel_launches": 2,
                "max_abs_error": m["max_abs_error"],
                "mean_abs_error": m["mean_abs_error"],
                "relative_l2": m["relative_l2"],
                "cosine_similarity": m["cosine_similarity"],
                "nan_count": m["nan_count"],
                "inf_count": m["inf_count"],
                "correctness_passed": bool(m["passed"]),
            }
        )
    summary = {
        "label": label,
        "all_passed": all(bool(row["passed"]) for row in correctness_rows),
        "max_abs_error": max(float(row["max_abs_error"]) for row in correctness_rows),
        "mean_abs_error_max": max(float(row["mean_abs_error"]) for row in correctness_rows),
        "relative_l2_max": max(float(row["relative_l2"]) for row in correctness_rows),
        "cosine_similarity_min": min(float(row["cosine_similarity"]) for row in correctness_rows),
        "nan_failures": sum(int(row["nan_count"]) for row in correctness_rows),
        "inf_failures": sum(int(row["inf_count"]) for row in correctness_rows),
        "rows": correctness_rows,
    }
    return rows, summary


def run_e2e(*, contexts: list[int], decode_tokens: int, gpu: str, label: str) -> list[dict[str, Any]]:
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused"
    os.environ["PATTERNKV_PROFILE"] = "0"
    wargs = make_worker_args("CAUSAL_V4_25", 42, gpu, experiment_id=f"phase_s2b1_{label}_e2e")
    model, _tokenizer = load_model(wargs)
    model.eval()
    ensure_profile_centroids(model, seed=20260814)
    rows = []
    for context in contexts:
        rec = run_decode_case(model, backend="fused", context_tokens=context, decode_tokens=decode_tokens, profile=False, seed=20260814 + context)
        rows.append(
            {
                "label": label,
                "context_tokens": context,
                "decode_tokens": decode_tokens,
                "mean_tpot_ms": rec["mean_tpot_ms"],
                "median_tpot_ms": rec["median_tpot_ms"],
                "p90_tpot_ms": rec["p90_tpot_ms"],
                "tokens_per_sec": rec["tokens_per_sec"],
                "peak_allocated_bytes": rec["peak_allocated_bytes"],
                "peak_reserved_bytes": rec["peak_reserved_bytes"],
            }
        )
    return rows


def extension_info() -> dict[str, Any]:
    import patternkv_gemv

    path = Path(patternkv_gemv.__file__).resolve()
    return {
        "loaded_extension_path": str(path),
        "binary_mtime": path.stat().st_mtime,
        "binary_sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--contexts", default="8192,16384,32768")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--e2e", action="store_true")
    parser.add_argument("--e2e-contexts", default="16384,32768")
    parser.add_argument("--decode-tokens", type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    contexts = [int(x) for x in args.contexts.split(",") if x]
    rows, correctness = run_kernel_bench(contexts=contexts, warmup=args.warmup, iters=args.iters, label=args.label)
    write_csv(OUT_DIR / f"{args.label}.csv", rows)
    write_json(OUT_DIR / f"{args.label}_correctness.json", correctness)
    write_json(OUT_DIR / f"{args.label}_environment.json", {
        "label": args.label,
        "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup": args.warmup,
        "iterations": args.iters,
        **extension_info(),
    })
    if args.e2e:
        e2e_contexts = [int(x) for x in args.e2e_contexts.split(",") if x]
        e2e_rows = run_e2e(contexts=e2e_contexts, decode_tokens=args.decode_tokens, gpu=args.gpu, label=args.label)
        write_csv(OUT_DIR / f"{args.label}_e2e.csv", e2e_rows)
    print(json.dumps({"label": args.label, "all_passed": correctness["all_passed"], "rows": rows}, indent=2))
    return 0 if correctness["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
