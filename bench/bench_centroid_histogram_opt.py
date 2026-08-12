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


OUT_DIR = ROOT / "reports/system_centroid_opt_v1"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
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


def context_to_quant_tokens(context: int) -> int:
    return ((context - 16 - 128) // 128) * 128


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
    assert out is not None
    return [float(starts[idx].elapsed_time(ends[idx]) * 1000.0) for idx in range(iters)], out


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
    }


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


def build_all_v2_case(tokens: int, *, seed: int, density: float, assignment: str) -> dict[str, Any]:
    device = torch.device("cuda")
    torch.manual_seed(seed + tokens + int(density * 1000))
    v_adjusted = (torch.randn(1, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16) * 0.25).contiguous()
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1).contiguous()
    attn = torch.softmax(torch.randn(1, NH, 1, tokens, device=device, dtype=torch.float16), dim=-1).contiguous()
    mask = (torch.rand(1, NH_KV, tokens, device=device) < density).to(torch.uint8)
    if assignment == "ALL_SAME":
        idx = torch.zeros(1, NH_KV, tokens, device=device, dtype=torch.int64)
    elif assignment == "SKEWED":
        idx = torch.randint(1, CENTROIDS, (1, NH_KV, tokens), device=device, dtype=torch.int64)
        idx[torch.rand(1, NH_KV, tokens, device=device) < 0.5] = 0
    else:
        idx = torch.randint(0, CENTROIDS, (1, NH_KV, tokens), device=device, dtype=torch.int64)
    payload = quantize_pack_v_reference(v_adjusted, GROUP_SIZE, 2)
    return {"attn": attn, "mask": mask, "idx": idx, "centroids": centroids, "payload": payload}


def call_kernel(
    *,
    attn: torch.Tensor,
    mask: torch.Tensor,
    idx: torch.Tensor,
    payload: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    bit: int,
    centroids: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    vq, scale, zero = payload
    if mode == "PRODUCTION":
        return cuda_attn_v_fused_with_base(GROUP_SIZE, attn, vq, scale, zero, bit, centroids, mask, idx, NH, NH_KV)
    return cuda_attn_v_fused_with_base_debug(
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


def run_kernel_rounds(
    *,
    attn: torch.Tensor,
    mask: torch.Tensor,
    idx: torch.Tensor,
    payload: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    bit: int,
    centroids: torch.Tensor,
    mode: str,
    rounds: int,
    warmup: int,
    iters: int,
    label: str,
    context: int,
    assignment: str,
    density: str,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    rows = []
    last = None
    for round_idx in range(rounds):
        torch.cuda.empty_cache()
        times, last = time_cuda(
            lambda: call_kernel(attn=attn, mask=mask, idx=idx, payload=payload, bit=bit, centroids=centroids, mode=mode),
            warmup=warmup,
            iters=iters,
        )
        rows.append(
            {
                "label": label,
                "context_tokens": context,
                "bit": bit,
                "mode": mode,
                "assignment": assignment,
                "density": density,
                "round": round_idx,
                "tokens": int(attn.shape[-1]),
                **summarize(times),
            }
        )
    assert last is not None
    return rows, last


def add_cv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        key = (row["label"], row["context_tokens"], row["bit"], row["mode"], row["assignment"], row["density"])
        grouped.setdefault(key, []).append(float(row["median_us"]))
    out = list(rows)
    for key, medians in grouped.items():
        label, context, bit, mode, assignment, density = key
        out.append(
            {
                "label": label,
                "context_tokens": context,
                "bit": bit,
                "mode": mode,
                "assignment": assignment,
                "density": density,
                "round": "summary",
                "tokens": next(r["tokens"] for r in rows if (r["label"], r["context_tokens"], r["bit"], r["mode"], r["assignment"], r["density"]) == key),
                "median_us": float(statistics.median(medians)),
                "mean_us": float(statistics.mean(medians)),
                "p90_us": max(medians),
                "round_cv": cv(medians),
                "stability": "STABLE" if cv(medians) <= 0.05 else "UNSTABLE",
            }
        )
    return out


def estimate_warp_agg_atomics(idx: torch.Tensor, mask: torch.Tensor) -> dict[str, int | float]:
    # Logical model of the current kernel: wy==0 processes lane*4+i groups.
    idx_cpu = idx.detach().to("cpu", torch.long)
    mask_cpu = mask.detach().to("cpu", torch.bool)
    original = int(mask_cpu.sum().item())
    candidate = 0
    _, nh_kv, tokens = idx_cpu.shape
    for b in range(idx_cpu.shape[0]):
        for hk in range(nh_kv):
            for tile in range(0, tokens, 128):
                for i in range(4):
                    vals = []
                    for lane in range(32):
                        t = tile + lane * 4 + i
                        if t < tokens and bool(mask_cpu[b, hk, t]):
                            v = int(idx_cpu[b, hk, t])
                            if 0 <= v < CENTROIDS:
                                vals.append(v)
                    candidate += len(set(vals))
    reduction = 1.0 - (candidate / original) if original else 0.0
    return {"original_atomic_ops": original, "candidate_atomic_ops": candidate, "atomic_reduction_fraction": reduction}


def compare_tensors(candidate: torch.Tensor, baseline: torch.Tensor) -> dict[str, Any]:
    diff = (candidate.float() - baseline.float()).abs()
    denom = torch.linalg.vector_norm(baseline.float()).clamp_min(1e-8)
    return {
        "baseline_vs_candidate_max_abs": float(diff.max().item()),
        "baseline_vs_candidate_mean_abs": float(diff.mean().item()),
        "baseline_vs_candidate_relative_l2": float((torch.linalg.vector_norm((candidate - baseline).float()) / denom).item()),
        "baseline_vs_candidate_cosine": float(torch.nn.functional.cosine_similarity(candidate.float().flatten(), baseline.float().flatten(), dim=0).item()),
    }


def run_baseline(args: argparse.Namespace) -> None:
    contexts = [int(x) for x in args.contexts.split(",") if x]
    rows = []
    contention = []
    for context in contexts:
        tokens = context_to_quant_tokens(context)
        data = build_case("mixed25", tokens, seed=20260820 + context)
        for bit in (2, 4):
            attn, mask, idx, payload = lane_inputs(data, bit=bit)
            r, _ = run_kernel_rounds(
                attn=attn,
                mask=mask,
                idx=idx,
                payload=payload,
                bit=bit,
                centroids=data["centroids"],
                mode="PRODUCTION",
                rounds=args.rounds,
                warmup=args.warmup,
                iters=args.iters,
                label="baseline",
                context=context,
                assignment="mixed25_normal",
                density="normal",
            )
            rows.extend(r)
        if context in (16384, 32768):
            for assignment in ("RANDOM_UNIFORM", "SKEWED"):
                case = build_all_v2_case(tokens, seed=20260821 + context, density=1.0, assignment=assignment)
                r, _ = run_kernel_rounds(
                    attn=case["attn"],
                    mask=case["mask"],
                    idx=case["idx"],
                    payload=case["payload"],
                    bit=2,
                    centroids=case["centroids"],
                    mode="PRODUCTION",
                    rounds=args.rounds,
                    warmup=args.warmup,
                    iters=args.iters,
                    label="baseline",
                    context=context,
                    assignment=assignment,
                    density="1.0",
                )
                contention.extend(r)
    mask_rows = []
    tokens = context_to_quant_tokens(16384)
    for density in (0.0, 0.25, 0.50, 0.75, 1.0):
        case = build_all_v2_case(tokens, seed=20260822, density=density, assignment="RANDOM_UNIFORM")
        r, _ = run_kernel_rounds(
            attn=case["attn"],
            mask=case["mask"],
            idx=case["idx"],
            payload=case["payload"],
            bit=2,
            centroids=case["centroids"],
            mode="PRODUCTION",
            rounds=args.rounds,
            warmup=args.warmup,
            iters=args.iters,
            label="baseline_mask_density",
            context=16384,
            assignment="RANDOM_UNIFORM",
            density=str(density),
        )
        mask_rows.extend(r)
    write_csv(OUT_DIR / "baseline.csv", add_cv(rows))
    write_csv(OUT_DIR / "contention_reproduction.csv", add_cv(contention + mask_rows))
    write_json(OUT_DIR / "baseline_environment.json", environment(args))


def run_candidate(args: argparse.Namespace, *, mode: str, label: str) -> None:
    contexts = [int(x) for x in args.contexts.split(",") if x]
    rows = []
    atomic_rows = []
    correctness_rows = []
    for context in contexts:
        tokens = context_to_quant_tokens(context)
        data = build_case("mixed25", tokens, seed=20260820 + context)
        for bit in (2, 4):
            attn, mask, idx, payload = lane_inputs(data, bit=bit)
            baseline = call_kernel(attn=attn, mask=mask, idx=idx, payload=payload, bit=bit, centroids=data["centroids"], mode="PRODUCTION")
            r, candidate = run_kernel_rounds(
                attn=attn,
                mask=mask,
                idx=idx,
                payload=payload,
                bit=bit,
                centroids=data["centroids"],
                mode=mode,
                rounds=args.rounds,
                warmup=args.warmup,
                iters=args.iters,
                label=label,
                context=context,
                assignment="mixed25_normal",
                density="normal",
            )
            rows.extend(r)
            correctness_rows.append({"context_tokens": context, "bit": bit, **compare_tensors(candidate, baseline)})
            atomic_rows.append({"label": label, "context_tokens": context, "bit": bit, "assignment": "mixed25_normal", "density": "normal", **estimate_warp_agg_atomics(idx, mask)})
        if context in (16384, 32768):
            for assignment in ("RANDOM_UNIFORM", "SKEWED", "ALL_SAME"):
                case = build_all_v2_case(tokens, seed=20260821 + context, density=1.0, assignment=assignment)
                baseline = call_kernel(attn=case["attn"], mask=case["mask"], idx=case["idx"], payload=case["payload"], bit=2, centroids=case["centroids"], mode="PRODUCTION")
                r, candidate = run_kernel_rounds(
                    attn=case["attn"],
                    mask=case["mask"],
                    idx=case["idx"],
                    payload=case["payload"],
                    bit=2,
                    centroids=case["centroids"],
                    mode=mode,
                    rounds=args.rounds,
                    warmup=args.warmup,
                    iters=args.iters,
                    label=label,
                    context=context,
                    assignment=assignment,
                    density="1.0",
                )
                rows.extend(r)
                correctness_rows.append({"context_tokens": context, "bit": 2, "assignment": assignment, **compare_tensors(candidate, baseline)})
                atomic_rows.append({"label": label, "context_tokens": context, "bit": 2, "assignment": assignment, "density": "1.0", **estimate_warp_agg_atomics(case["idx"], case["mask"])})
    write_csv(OUT_DIR / f"{label}_results.csv", add_cv(rows))
    write_csv(OUT_DIR / "atomic_count_estimate.csv", atomic_rows)
    write_json(OUT_DIR / "correctness_summary.json", {"label": label, "all_passed": all(r["baseline_vs_candidate_cosine"] >= 0.9999 and r["baseline_vs_candidate_max_abs"] <= 5e-3 for r in correctness_rows), "rows": correctness_rows})
    write_json(OUT_DIR / f"{label}_environment.json", environment(args))


def run_mixed(args: argparse.Namespace, *, mode: str | None, label: str) -> None:
    rows = []
    for context in (16384, 32768):
        tokens = context_to_quant_tokens(context)
        data = build_case("mixed25", tokens, seed=20260823 + context)
        def fn() -> torch.Tensor:
            if mode is None:
                return mixed_output(data)
            p2, p4 = data["p2"], data["p4"]
            precision = data["precision"][0].bool()
            attn2 = data["attn"][..., ~precision].contiguous()
            attn4 = data["attn"][..., precision].contiguous()
            mask2 = data["v_pattern_mask"][:, :, ~precision].contiguous()
            mask4 = data["v_pattern_mask"][:, :, precision].contiguous()
            idx2 = data["v_idx"][:, :, ~precision].contiguous()
            idx4 = data["v_idx"][:, :, precision].contiguous()
            out2 = call_kernel(attn=attn2, mask=mask2, idx=idx2, payload=p2, bit=2, centroids=data["centroids"], mode=mode)
            out4 = call_kernel(attn=attn4, mask=mask4, idx=idx4, payload=p4, bit=4, centroids=data["centroids"], mode=mode)
            return out2 + out4
        for round_idx in range(args.rounds):
            times, _ = time_cuda(fn, warmup=args.warmup, iters=args.iters)
            rows.append(
                {
                    "label": label,
                    "context_tokens": context,
                    "bit": "mixed",
                    "mode": mode or "PRODUCTION",
                    "assignment": "mixed25_normal",
                    "density": "normal",
                    "round": round_idx,
                    "tokens": tokens,
                    **summarize(times),
                }
            )
    write_csv(OUT_DIR / f"mixed_v_{label}.csv", add_cv(rows))


def environment(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup": args.warmup,
        "iters": args.iters,
        "rounds": args.rounds,
        **extension_info(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["baseline", "candidate", "mixed"], required=True)
    parser.add_argument("--mode", default="PRODUCTION")
    parser.add_argument("--label", default="candidate_a")
    parser.add_argument("--contexts", default="8192,16384,32768")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.phase == "baseline":
        run_baseline(args)
    elif args.phase == "candidate":
        run_candidate(args, mode=args.mode, label=args.label)
    else:
        run_mixed(args, mode=None if args.mode == "PRODUCTION" else args.mode, label=args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
