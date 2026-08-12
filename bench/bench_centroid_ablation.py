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
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant"))

from bench.bench_mixed_v_kernel_perf import (  # noqa: E402
    CENTROIDS,
    GROUP_SIZE,
    HEAD_DIM,
    NH,
    NH_KV,
    build_case,
    metrics,
    mixed_output,
    reference_output,
)
from models.segmented_cache import quantize_pack_v_reference  # noqa: E402
from quant.matmul import cuda_attn_v_fused_with_base, cuda_attn_v_fused_with_base_debug  # noqa: E402


OUT_DIR = ROOT / "reports/system_centroid_v1"
MODES = ("FULL", "RESIDUAL_ONLY", "NO_CENTROID_HISTOGRAM", "CENTROID_ONLY")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(math.ceil(p * len(ordered))) - 1)
    return float(ordered[idx])


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_us": float(statistics.median(values)),
        "mean_us": float(statistics.mean(values)),
        "p90_us": percentile(values, 0.90),
    }


def cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    return float(statistics.stdev(values) / mean) if mean else 0.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extension_info() -> dict[str, Any]:
    import patternkv_gemv

    path = Path(patternkv_gemv.__file__).resolve()
    return {
        "loaded_extension_path": str(path),
        "binary_mtime": path.stat().st_mtime,
        "binary_sha256": sha256_file(path),
        "has_debug_entry": hasattr(patternkv_gemv, "attn_v_forward_cuda_outer_dim_with_base_debug"),
    }


def time_cuda(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for idx in range(iters):
        starts[idx].record()
        fn()
        ends[idx].record()
    torch.cuda.synchronize()
    return [float(starts[idx].elapsed_time(ends[idx]) * 1000.0) for idx in range(iters)]


def lane_inputs(data: dict[str, Any], *, bit: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    precision = data["precision"][0].bool()
    token_mask = ~precision if bit == 2 else precision
    payload = data["p2"] if bit == 2 else data["p4"]
    return (
        data["attn"][..., token_mask].contiguous(),
        data["v_pattern_mask"][:, :, token_mask].contiguous(),
        data["v_idx"][:, :, token_mask].contiguous(),
        payload,
    )


def run_mode(
    *,
    attn: torch.Tensor,
    mask: torch.Tensor,
    idx: torch.Tensor,
    payload: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    bit: int,
    centroids: torch.Tensor,
    mode: str,
    warmup: int,
    iters: int,
) -> list[float]:
    vq, scale, zero = payload
    if mode == "PRODUCTION":
        fn = lambda: cuda_attn_v_fused_with_base(GROUP_SIZE, attn, vq, scale, zero, bit, centroids, mask, idx, NH, NH_KV)
    else:
        fn = lambda: cuda_attn_v_fused_with_base_debug(
            GROUP_SIZE,
            attn,
            vq,
            scale,
            zero,
            bit,
            centroids,
            mask,
            idx,
            NH,
            NH_KV,
            debug_mode=mode,
        )
    return time_cuda(fn, warmup=warmup, iters=iters)


def context_to_quant_tokens(context: int) -> int:
    return ((context - 16 - 128) // 128) * 128


def run_ablation(*, contexts: list[int], rounds: int, warmup: int, iters: int) -> list[dict[str, Any]]:
    rows = []
    for context in contexts:
        tokens = context_to_quant_tokens(context)
        data = build_case("mixed25", tokens, seed=20260816 + context)
        for bit in (2, 4):
            attn, mask, idx, payload = lane_inputs(data, bit=bit)
            token_count = int(attn.shape[-1])
            for mode in MODES:
                for round_idx in range(rounds):
                    torch.cuda.empty_cache()
                    times = run_mode(
                        attn=attn,
                        mask=mask,
                        idx=idx,
                        payload=payload,
                        bit=bit,
                        centroids=data["centroids"],
                        mode=mode,
                        warmup=warmup,
                        iters=iters,
                    )
                    s = summarize(times)
                    rows.append(
                        {
                            "context_tokens": context,
                            "bit": bit,
                            "mode": mode,
                            "round": round_idx,
                            "tokens": token_count,
                            **s,
                        }
                    )
    return rows


def build_all_v2_case(tokens: int, *, seed: int, density: float, assignment: str = "RANDOM_UNIFORM") -> dict[str, Any]:
    device = torch.device("cuda")
    torch.manual_seed(seed + tokens + int(density * 1000))
    v_adjusted = (torch.randn(1, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16) * 0.25).contiguous()
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1).contiguous()
    attn = torch.softmax(torch.randn(1, NH, 1, tokens, device=device, dtype=torch.float16), dim=-1).contiguous()
    mask = (torch.rand(1, NH_KV, tokens, device=device) < density).to(torch.uint8)
    if assignment == "SKEWED":
        idx = torch.randint(1, CENTROIDS, (1, NH_KV, tokens), device=device, dtype=torch.int64)
        skew = torch.rand(1, NH_KV, tokens, device=device) < 0.5
        idx[skew] = 0
    else:
        idx = torch.randint(0, CENTROIDS, (1, NH_KV, tokens), device=device, dtype=torch.int64)
    payload = quantize_pack_v_reference(v_adjusted, GROUP_SIZE, 2)
    return {"attn": attn, "mask": mask, "idx": idx, "centroids": centroids, "payload": payload}


def run_mask_density(*, context: int, rounds: int, warmup: int, iters: int) -> list[dict[str, Any]]:
    rows = []
    tokens = context_to_quant_tokens(context)
    for density in (0.0, 0.25, 0.50, 0.75, 1.0):
        data = build_all_v2_case(tokens, seed=20260817, density=density)
        full_rounds = []
        residual_rounds = []
        for round_idx in range(rounds):
            full = summarize(
                run_mode(
                    attn=data["attn"],
                    mask=data["mask"],
                    idx=data["idx"],
                    payload=data["payload"],
                    bit=2,
                    centroids=data["centroids"],
                    mode="FULL",
                    warmup=warmup,
                    iters=iters,
                )
            )
            residual = summarize(
                run_mode(
                    attn=data["attn"],
                    mask=data["mask"],
                    idx=data["idx"],
                    payload=data["payload"],
                    bit=2,
                    centroids=data["centroids"],
                    mode="RESIDUAL_ONLY",
                    warmup=warmup,
                    iters=iters,
                )
            )
            full_rounds.append(full["median_us"])
            residual_rounds.append(residual["median_us"])
            est = full["median_us"] - residual["median_us"]
            rows.append(
                {
                    "density": density,
                    "context_tokens": context,
                    "bit": 2,
                    "round": round_idx,
                    "full_median_us": full["median_us"],
                    "residual_only_median_us": residual["median_us"],
                    "estimated_centroid_us": est,
                    "estimated_centroid_fraction": est / full["median_us"] if full["median_us"] else 0.0,
                }
            )
        rows.append(
            {
                "density": density,
                "context_tokens": context,
                "bit": 2,
                "round": "summary",
                "full_median_us": statistics.median(full_rounds),
                "residual_only_median_us": statistics.median(residual_rounds),
                "estimated_centroid_us": statistics.median(full_rounds) - statistics.median(residual_rounds),
                "estimated_centroid_fraction": (statistics.median(full_rounds) - statistics.median(residual_rounds))
                / statistics.median(full_rounds)
                if statistics.median(full_rounds)
                else 0.0,
            }
        )
    return rows


def assignment_stats(idx: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    active = idx[mask.bool()].detach().flatten().to(torch.long)
    if active.numel() == 0:
        return {
            "active_assignments": 0,
            "centroid_count": CENTROIDS,
            "top_centroid": None,
            "top_count": 0,
            "max_fraction": 0.0,
            "entropy": 0.0,
            "tokens_per_centroid_mean": 0.0,
        }
    counts = torch.bincount(active, minlength=CENTROIDS).float().cpu()
    probs = counts / counts.sum().clamp_min(1.0)
    entropy = float((-(probs[probs > 0] * torch.log2(probs[probs > 0]))).sum().item())
    top_count = int(counts.max().item())
    return {
        "active_assignments": int(active.numel()),
        "centroid_count": CENTROIDS,
        "top_centroid": int(counts.argmax().item()),
        "top_count": top_count,
        "max_fraction": float(top_count / max(int(active.numel()), 1)),
        "entropy": entropy,
        "tokens_per_centroid_mean": float(active.numel() / CENTROIDS),
    }


def run_assignment_contention(*, context: int, rounds: int, warmup: int, iters: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    stats = {}
    tokens = context_to_quant_tokens(context)
    for assignment in ("RANDOM_UNIFORM", "SKEWED"):
        data = build_all_v2_case(tokens, seed=20260818, density=1.0, assignment=assignment)
        stats[assignment] = assignment_stats(data["idx"], data["mask"])
        for round_idx in range(rounds):
            s = summarize(
                run_mode(
                    attn=data["attn"],
                    mask=data["mask"],
                    idx=data["idx"],
                    payload=data["payload"],
                    bit=2,
                    centroids=data["centroids"],
                    mode="FULL",
                    warmup=warmup,
                    iters=iters,
                )
            )
            rows.append(
                {
                    "assignment": assignment,
                    "context_tokens": context,
                    "bit": 2,
                    "round": round_idx,
                    "tokens": tokens,
                    **s,
                }
            )
    return rows, stats


def aggregate_ablation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((int(row["context_tokens"]), int(row["bit"]), str(row["mode"])), []).append(float(row["median_us"]))
    summary: dict[str, Any] = {"groups": {}, "estimates": {}}
    for key, medians in grouped.items():
        context, bit, mode = key
        label = f"{context}/bit{bit}/{mode}"
        summary["groups"][label] = {
            "round_median_us": float(statistics.median(medians)),
            "round_mean_us": float(statistics.mean(medians)),
            "round_cv": cv(medians),
            "stability": "STABLE" if cv(medians) <= 0.05 else "UNSTABLE",
            "rounds": len(medians),
        }
    for context in sorted({int(r["context_tokens"]) for r in rows}):
        for bit in (2, 4):
            full = summary["groups"].get(f"{context}/bit{bit}/FULL")
            residual = summary["groups"].get(f"{context}/bit{bit}/RESIDUAL_ONLY")
            if not full or not residual:
                continue
            overhead = full["round_median_us"] - residual["round_median_us"]
            summary["estimates"][f"{context}/bit{bit}"] = {
                "centroid_overhead_estimate_us": overhead,
                "centroid_overhead_fraction": overhead / full["round_median_us"] if full["round_median_us"] else 0.0,
                "note": "NOT_STRICTLY_ADDITIVE_APPROXIMATE_ABLATION_COST",
            }
    return summary


def run_correctness(contexts: list[int]) -> dict[str, Any]:
    rows = []
    unchanged = []
    for context in contexts:
        tokens = context_to_quant_tokens(context)
        data = build_case("mixed25", tokens, seed=20260819 + context)
        fused = mixed_output(data)
        ref = reference_output(data)
        rows.append({"context_tokens": context, **metrics(fused, ref)})
        for bit in (2, 4):
            attn, mask, idx, payload = lane_inputs(data, bit=bit)
            prod = cuda_attn_v_fused_with_base(GROUP_SIZE, attn, payload[0], payload[1], payload[2], bit, data["centroids"], mask, idx, NH, NH_KV)
            full = cuda_attn_v_fused_with_base_debug(
                GROUP_SIZE,
                attn,
                payload[0],
                payload[1],
                payload[2],
                bit,
                data["centroids"],
                mask,
                idx,
                NH,
                NH_KV,
                debug_mode="FULL",
            )
            torch.testing.assert_close(full, prod, rtol=1e-5, atol=1e-5)
            unchanged.append({"context_tokens": context, "bit": bit, "full_debug_matches_production": True})
    return {
        "rows": rows,
        "production_unchanged_rows": unchanged,
        "all_passed": all(bool(row["passed"]) for row in rows),
        "max_abs_error": max(float(row["max_abs_error"]) for row in rows),
        "relative_l2_max": max(float(row["relative_l2"]) for row in rows),
        "cosine_similarity_min": min(float(row["cosine_similarity"]) for row in rows),
        "nan_count": sum(int(row["nan_count"]) for row in rows),
        "inf_count": sum(int(row["inf_count"]) for row in rows),
        "production_full_unchanged": True,
    }


def environment_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup": args.warmup,
        "iters": args.iters,
        "rounds": args.rounds,
        "contexts": args.contexts,
        **extension_info(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="8192,16384,32768")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--mask-context", type=int, default=16384)
    parser.add_argument("--gpu", default="1")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    contexts = [int(x) for x in args.contexts.split(",") if x]
    torch.cuda.set_device(0)

    correctness = run_correctness(contexts)
    write_json(OUT_DIR / "correctness_summary.json", correctness)

    rows = run_ablation(contexts=contexts, rounds=args.rounds, warmup=args.warmup, iters=args.iters)
    write_csv(OUT_DIR / "centroid_ablation.csv", rows)
    summary = aggregate_ablation(rows)
    write_json(OUT_DIR / "centroid_ablation_summary.json", summary)

    mask_rows = run_mask_density(context=args.mask_context, rounds=args.rounds, warmup=args.warmup, iters=args.iters)
    write_csv(OUT_DIR / "mask_density.csv", mask_rows)

    contention_rows, contention_stats = run_assignment_contention(
        context=args.mask_context, rounds=args.rounds, warmup=args.warmup, iters=args.iters
    )
    write_csv(OUT_DIR / "assignment_contention.csv", contention_rows)
    write_json(OUT_DIR / "assignment_contention_stats.json", contention_stats)

    write_json(OUT_DIR / "environment.json", environment_payload(args))
    print(json.dumps({"correctness": correctness, "summary": summary, "contention_stats": contention_stats}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
