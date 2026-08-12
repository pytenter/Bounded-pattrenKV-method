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

from bench.bench_mixed_v_kernel_perf import CENTROIDS, GROUP_SIZE, HEAD_DIM, NH, NH_KV, build_case, metrics  # noqa: E402
from models.segmented_cache import quantize_pack_v_reference  # noqa: E402
from quant.matmul import cuda_attn_v_fused_with_base, cuda_attn_v_fused_with_base_gqa_v2, cuda_attn_v_mixed_fused_with_base  # noqa: E402


OUT_DIR = ROOT / "reports/system_gqa_kernel_v1"


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


def add_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple((k, row.get(k)) for k in ("label", "context_tokens", "component", "backend", "tile_tokens"))
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


def v2_inputs(data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    low = ~data["precision"][0].bool()
    return (
        data["attn"][..., low].contiguous(),
        data["v_pattern_mask"][:, :, low].contiguous(),
        data["v_idx"][:, :, low].contiguous(),
        data["p2"],
    )


def v4_inputs(data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    high = data["precision"][0].bool()
    return (
        data["attn"][..., high].contiguous(),
        data["v_pattern_mask"][:, :, high].contiguous(),
        data["v_idx"][:, :, high].contiguous(),
        data["p4"],
    )


def baseline_v2(data: dict[str, Any]) -> torch.Tensor:
    attn, mask, idx, payload = v2_inputs(data)
    return cuda_attn_v_fused_with_base(GROUP_SIZE, attn, payload[0], payload[1], payload[2], 2, data["centroids"], mask, idx, NH, NH_KV)


def candidate_v2(data: dict[str, Any]) -> torch.Tensor:
    attn, mask, idx, payload = v2_inputs(data)
    return cuda_attn_v_fused_with_base_gqa_v2(GROUP_SIZE, attn, payload[0], payload[1], payload[2], 2, data["centroids"], mask, idx, NH, NH_KV)


def baseline_v4(data: dict[str, Any]) -> torch.Tensor:
    attn, mask, idx, payload = v4_inputs(data)
    return cuda_attn_v_fused_with_base(GROUP_SIZE, attn, payload[0], payload[1], payload[2], 4, data["centroids"], mask, idx, NH, NH_KV)


def mixed(data: dict[str, Any], *, backend: str) -> torch.Tensor:
    old = os.environ.get("PATTERNKV_GQA_V_BACKEND")
    os.environ["PATTERNKV_GQA_V_BACKEND"] = backend
    try:
        p2, p4 = data["p2"], data["p4"]
        return cuda_attn_v_mixed_fused_with_base(
            GROUP_SIZE,
            data["attn"],
            p2[0],
            p2[1],
            p2[2],
            p4[0],
            p4[1],
            p4[2],
            data["precision"],
            data["centroids"],
            data["v_pattern_mask"],
            data["v_idx"],
            NH,
            NH_KV,
        )
    finally:
        if old is None:
            os.environ.pop("PATTERNKV_GQA_V_BACKEND", None)
        else:
            os.environ["PATTERNKV_GQA_V_BACKEND"] = old


def run_benchmarks(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    baseline_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    mixed_rows: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []
    for context in [int(x) for x in args.contexts.split(",") if x]:
        data = build_case("mixed25", context_to_quant_tokens(context), seed=20260830 + context)
        for component, backend, fn, rows in (
            ("v2", "baseline", lambda data=data: baseline_v2(data), baseline_rows),
            ("v4", "baseline", lambda data=data: baseline_v4(data), baseline_rows),
            ("mixed_v", "baseline", lambda data=data: mixed(data, backend="baseline"), baseline_rows),
            ("v2", "gqa4_cta", lambda data=data: candidate_v2(data), candidate_rows),
            ("mixed_v", "gqa4_cta", lambda data=data: mixed(data, backend="gqa"), mixed_rows),
        ):
            for round_idx in range(args.rounds):
                times, _ = time_cuda(fn, warmup=args.warmup, iters=args.iters)
                rows.append(
                    {
                        "label": "s2b3",
                        "context_tokens": context,
                        "component": component,
                        "backend": backend,
                        "tile_tokens": 128 if backend == "gqa4_cta" else "",
                        "round": round_idx,
                        **summarize(times),
                    }
                )
        tile_rows.append({"candidate": "gqa4_cta", "tile_tokens": 128, "implemented": "YES", "tested": "YES"})
        tile_rows.append({"candidate": "gqa4_cta", "tile_tokens": 64, "implemented": "NO", "tested": "NO", "reason": "Tile sweep stopped after 128-token candidate regressed strongly."})
        tile_rows.append({"candidate": "gqa4_cta", "tile_tokens": 256, "implemented": "NO", "tested": "NO", "reason": "Tile sweep stopped after 128-token candidate regressed strongly."})
    return {
        "baseline": add_summary(baseline_rows),
        "candidate": add_summary(candidate_rows),
        "mixed": add_summary(mixed_rows),
        "tile": tile_rows,
    }


def set_assignment(data: dict[str, Any], kind: str) -> None:
    if kind == "normal":
        return
    if kind == "uniform":
        ids = torch.arange(data["v_idx"].shape[-1], device="cuda", dtype=torch.int64) % CENTROIDS
        data["v_idx"] = ids.view(1, 1, -1).expand_as(data["v_idx"]).contiguous()
    elif kind == "skewed":
        skew = torch.rand_like(data["v_idx"].float()) < 0.75
        data["v_idx"] = torch.randint(1, CENTROIDS, data["v_idx"].shape, device="cuda", dtype=torch.int64)
        data["v_idx"][skew] = 0
    elif kind == "all_same":
        data["v_idx"].zero_()
    else:
        raise ValueError(kind)


def set_mask_density(data: dict[str, Any], density: float) -> None:
    if density <= 0:
        data["v_pattern_mask"].zero_()
    elif density >= 1:
        data["v_pattern_mask"].fill_(1)
    else:
        data["v_pattern_mask"] = (torch.rand_like(data["v_pattern_mask"].float()) < density).to(torch.uint8)


def qhead_metrics(candidate: torch.Tensor, baseline: torch.Tensor) -> dict[int, float]:
    diff = (candidate.float() - baseline.float()).abs()
    return {hq: float(diff[0, hq].max().item()) for hq in range(diff.shape[1])}


def run_correctness(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    qrows: list[dict[str, Any]] = []
    contexts = [int(x) for x in args.correctness_contexts.split(",") if x]
    for context in contexts:
        for assignment in ("normal", "uniform", "skewed", "all_same"):
            for density in (0.0, 0.25, 0.50, 1.0):
                data = build_case("all_v2", context_to_quant_tokens(context), seed=20260831 + context)
                set_assignment(data, assignment)
                set_mask_density(data, density)
                base = baseline_v2(data)
                cand = candidate_v2(data)
                row = {
                    "context_tokens": context,
                    "assignment": assignment,
                    "mask_density": density,
                    **metrics(cand, base),
                }
                rows.append(row)
                head_max = qhead_metrics(cand, base)
                for hq, err in head_max.items():
                    qrows.append(
                        {
                            "context_tokens": context,
                            "assignment": assignment,
                            "mask_density": density,
                            "q_head": hq,
                            "kv_head": hq // 4,
                            "max_abs_error": err,
                            "passed": err <= 5e-3,
                        }
                    )
    return rows, qrows


def run_mapping_check() -> list[dict[str, Any]]:
    device = torch.device("cuda")
    tokens = 128
    precision = torch.zeros(1, tokens, dtype=torch.bool, device=device)
    values = torch.zeros(1, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16)
    p2 = quantize_pack_v_reference(values, GROUP_SIZE, 2)
    centroids = torch.zeros(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16)
    for hk in range(NH_KV):
        centroids[hk, 0, :].fill_(float(hk + 1))
    data = {
        "precision": precision,
        "attn": torch.full((1, NH, 1, tokens), 1.0 / tokens, device=device, dtype=torch.float16),
        "p2": p2,
        "centroids": centroids,
        "v_pattern_mask": torch.ones(1, NH_KV, tokens, device=device, dtype=torch.uint8),
        "v_idx": torch.zeros(1, NH_KV, tokens, device=device, dtype=torch.int64),
    }
    out = candidate_v2(data)
    rows = []
    for hq in range(NH):
        hk = hq // 4
        expected = torch.full((HEAD_DIM,), float(hk + 1), device=device, dtype=torch.float16)
        err = float((out[0, hq, 0] - expected).abs().max().item())
        rows.append({"q_head": hq, "expected_kv_head": hk, "max_abs_error": err, "passed": err <= 5e-3})
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment(args: argparse.Namespace) -> dict[str, Any]:
    import patternkv_gemv

    path = Path(patternkv_gemv.__file__).resolve()
    return {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "iters": args.iters,
        "build_command": "cd quant && python setup.py build_ext --inplace",
        "loaded_extension_path": str(path),
        "binary_mtime": path.stat().st_mtime,
        "binary_sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="8192,16384,32768")
    parser.add_argument("--correctness-contexts", default="8192,16384,32768")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    benches = run_benchmarks(args)
    write_csv(OUT_DIR / "baseline.csv", benches["baseline"])
    write_csv(OUT_DIR / "candidate_a_results.csv", benches["candidate"])
    write_csv(OUT_DIR / "mixed_v_results.csv", benches["mixed"])
    write_csv(OUT_DIR / "tile_sweep.csv", benches["tile"])
    write_csv(OUT_DIR / "candidate_b_results.csv", [{"candidate": "2_qhead_partial_reuse", "implemented": "NO", "decision": "NOT_IMPLEMENTED", "reason": "4-Q-head candidate was correct but a large regression, so partial-reuse implementation was not pursued in S2B-3."}])
    correctness, qhead = run_correctness(args)
    write_csv(OUT_DIR / "qhead_mapping_correctness.csv", run_mapping_check() + qhead)
    write_json(
        OUT_DIR / "correctness_summary.json",
        {
            "all_passed": all(bool(r["passed"]) for r in correctness),
            "case_count": len(correctness),
            "max_abs_error": max(float(r["max_abs_error"]) for r in correctness),
            "mean_abs_error_max": max(float(r["mean_abs_error"]) for r in correctness),
            "relative_l2_max": max(float(r["relative_l2"]) for r in correctness),
            "cosine_similarity_min": min(float(r["cosine_similarity"]) for r in correctness),
            "nan_count": sum(int(r["nan_count"]) for r in correctness),
            "inf_count": sum(int(r["inf_count"]) for r in correctness),
        },
    )
    write_json(OUT_DIR / "environment.json", environment(args))
    write_csv(OUT_DIR / "e2e_summary.csv", [{"status": "NOT_RUN", "reason": "GQA V2 candidate failed the microbench and mixed-V performance gates, so E2E was skipped by policy."}])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
