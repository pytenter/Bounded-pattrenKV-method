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

from bench.bench_mixed_v_kernel_perf import CENTROIDS, GROUP_SIZE, NH, NH_KV, build_case, metrics, mixed_output, reference_output  # noqa: E402
from quant.matmul import cuda_attn_v_fused_with_base, cuda_attn_v_fused_with_base_debug  # noqa: E402


OUT_DIR = ROOT / "reports/system_centroid_table_v1"


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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(math.ceil(len(ordered) * p)) - 1)])


def summarize(values: list[float]) -> dict[str, float]:
    return {"median_us": float(statistics.median(values)), "mean_us": float(statistics.mean(values)), "p90_us": percentile(values, 0.90)}


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


def add_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple((k, row.get(k)) for k in ("label", "context_tokens", "bit", "mode"))
        grouped.setdefault(key, []).append(row)
    out = list(rows)
    for key, group in grouped.items():
        medians = [float(r["median_us"]) for r in group]
        base = dict(key)
        base.update(
            {
                "round": "summary",
                "median_us": float(statistics.median(medians)),
                "mean_us": float(statistics.mean(medians)),
                "p90_us": max(medians),
                "round_cv": cv(medians),
                "stability": "STABLE" if cv(medians) <= 0.05 else "UNSTABLE",
            }
        )
        out.append(base)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extension_info() -> dict[str, Any]:
    import patternkv_gemv

    path = Path(patternkv_gemv.__file__).resolve()
    return {"loaded_extension_path": str(path), "binary_mtime": path.stat().st_mtime, "binary_sha256": sha256_file(path)}


def environment(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "iters": args.iters,
        **extension_info(),
    }


def compare_tensors(candidate: torch.Tensor, baseline: torch.Tensor) -> dict[str, Any]:
    diff = (candidate.float() - baseline.float()).abs()
    denom = torch.linalg.vector_norm(baseline.float()).clamp_min(1e-8)
    return {
        "baseline_vs_candidate_max_abs": float(diff.max().item()),
        "baseline_vs_candidate_mean_abs": float(diff.mean().item()),
        "baseline_vs_candidate_relative_l2": float((torch.linalg.vector_norm((candidate - baseline).float()) / denom).item()),
        "baseline_vs_candidate_cosine": float(torch.nn.functional.cosine_similarity(candidate.float().flatten(), baseline.float().flatten(), dim=0).item()),
    }


def run_decomposition(args: argparse.Namespace) -> None:
    rows = []
    active_rows = []
    correctness_rows = []
    modes = ("FULL", "RESIDUAL_ONLY", "NO_TABLE_CONTRIBUTION", "LANE0_TABLE_FULL")
    for context in [int(x) for x in args.contexts.split(",") if x]:
        data = build_case("mixed25", context_to_quant_tokens(context), seed=20260824 + context)
        for bit in (2, 4):
            if bit == 4 and context == 8192:
                continue
            attn, mask, idx, payload = lane_inputs(data, bit=bit)
            active = ((mask.bool()) & (idx >= 0) & (idx < CENTROIDS)).to(torch.bool)
            counts = []
            for hk in range(active.shape[1]):
                ids = idx[0, hk][active[0, hk]].detach().cpu().tolist()
                counts.append(len(set(int(x) for x in ids)))
            active_rows.append(
                {
                    "context_tokens": context,
                    "bit": bit,
                    "tokens": int(attn.shape[-1]),
                    "mcent": CENTROIDS,
                    "active_centroid_min": min(counts) if counts else 0,
                    "active_centroid_median": statistics.median(counts) if counts else 0,
                    "active_centroid_max": max(counts) if counts else 0,
                    "active_centroid_fraction_median": (statistics.median(counts) / CENTROIDS) if counts else 0.0,
                }
            )
            baseline = call_kernel(attn=attn, mask=mask, idx=idx, payload=payload, bit=bit, centroids=data["centroids"], mode="FULL")
            for mode in modes:
                last = None
                for round_idx in range(args.rounds):
                    times, last = time_cuda(
                        lambda mode=mode: call_kernel(attn=attn, mask=mask, idx=idx, payload=payload, bit=bit, centroids=data["centroids"], mode=mode),
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    rows.append({"label": "post_histogram", "context_tokens": context, "bit": bit, "mode": mode, "round": round_idx, **summarize(times)})
                if mode == "LANE0_TABLE_FULL":
                    assert last is not None
                    correctness_rows.append({"context_tokens": context, "bit": bit, **compare_tensors(last, baseline)})
    write_csv(OUT_DIR / "post_histogram_decomposition.csv", add_summary(rows))
    write_csv(OUT_DIR / "active_centroid_stats.csv", active_rows)
    write_json(
        OUT_DIR / "correctness_summary.json",
        {
            "all_passed": all(r["baseline_vs_candidate_max_abs"] <= 5e-3 and r["baseline_vs_candidate_cosine"] >= 0.9999 for r in correctness_rows),
            "rows": correctness_rows,
        },
    )
    write_json(OUT_DIR / "environment.json", environment(args))


def run_mixed(args: argparse.Namespace, *, mode: str, label: str) -> None:
    rows = []
    for context in (16384, 32768):
        data = build_case("mixed25", context_to_quant_tokens(context), seed=20260825 + context)
        def fn() -> torch.Tensor:
            if mode == "PRODUCTION":
                return mixed_output(data)
            p2, p4 = data["p2"], data["p4"]
            precision = data["precision"][0].bool()
            out2 = call_kernel(
                attn=data["attn"][..., ~precision].contiguous(),
                mask=data["v_pattern_mask"][:, :, ~precision].contiguous(),
                idx=data["v_idx"][:, :, ~precision].contiguous(),
                payload=p2,
                bit=2,
                centroids=data["centroids"],
                mode=mode,
            )
            out4 = call_kernel(
                attn=data["attn"][..., precision].contiguous(),
                mask=data["v_pattern_mask"][:, :, precision].contiguous(),
                idx=data["v_idx"][:, :, precision].contiguous(),
                payload=p4,
                bit=4,
                centroids=data["centroids"],
                mode=mode,
            )
            return out2 + out4
        for round_idx in range(args.rounds):
            times, _ = time_cuda(fn, warmup=args.warmup, iters=args.iters)
            rows.append({"label": label, "context_tokens": context, "bit": "mixed", "mode": mode, "round": round_idx, **summarize(times)})
    write_csv(OUT_DIR / f"mixed_v_{label}.csv", add_summary(rows))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["decomposition", "mixed"], required=True)
    parser.add_argument("--mode", default="PRODUCTION")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--contexts", default="8192,16384,32768")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.phase == "decomposition":
        run_decomposition(args)
    else:
        run_mixed(args, mode=args.mode, label=args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
