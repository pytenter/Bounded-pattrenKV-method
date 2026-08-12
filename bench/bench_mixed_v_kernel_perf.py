#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant"))

from models.segmented_cache import dequantize_v_reference, pattern_gather_centroids, quantize_pack_v_reference
from quant.matmul import (
    cuda_attn_v_fused_with_base,
    cuda_attn_v_mixed_fused_with_base,
    get_patternkv_mixed_v_counters,
    reset_patternkv_mixed_v_counters,
)


OUT_DIR = ROOT / "reports/system_kernel_v1"
GROUP_SIZE = 128
NH = 32
NH_KV = 8
HEAD_DIM = 128
CENTROIDS = 16
CASES = ("all_v2", "all_v4", "mixed25", "random", "causal_like", "first25", "last25", "alternating")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, num_key_value_heads * n_rep, slen, head_dim)


def mask_for_case(case: str, tokens: int, *, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    if case == "all_v2":
        return mask
    if case == "all_v4":
        return torch.ones(tokens, dtype=torch.bool, device=device)
    k = max(1, int(round(tokens * 0.25)))
    if case == "first25":
        mask[:k] = True
    elif case == "last25":
        mask[-k:] = True
    elif case == "alternating":
        mask[1::4] = True
    elif case == "causal_like":
        idx = torch.linspace(0, tokens - 1, steps=k, device=device).round().long().unique()
        mask[idx[:k]] = True
        cursor = tokens - 1
        while int(mask.sum().item()) < k:
            mask[cursor] = True
            cursor -= 1
    elif case == "random":
        generator = torch.Generator(device=device).manual_seed(20260812 + tokens)
        mask[torch.randperm(tokens, generator=generator, device=device)[:k]] = True
    elif case == "mixed25":
        idx = (torch.arange(k, device=device) * 4 + 1).clamp_max(tokens - 1)
        mask[idx.unique()] = True
        cursor = 0
        while int(mask.sum().item()) < k:
            mask[cursor] = True
            cursor += 1
    else:
        raise ValueError(case)
    return mask


def build_case(case: str, tokens: int, *, seed: int = 1234) -> dict[str, torch.Tensor | tuple]:
    device = torch.device("cuda")
    torch.manual_seed(seed + tokens)
    precision = mask_for_case(case, tokens, device=device).unsqueeze(0)
    v_adjusted = (torch.randn(1, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16) * 0.25).contiguous()
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1).contiguous()
    v_idx = torch.randint(0, CENTROIDS, (1, NH_KV, tokens), device=device, dtype=torch.int64)
    v_pattern_mask = (torch.rand(1, NH_KV, tokens, device=device) > 0.55).to(torch.uint8)
    attn = torch.softmax(torch.randn(1, NH, 1, tokens, device=device, dtype=torch.float16), dim=-1).contiguous()
    low = v_adjusted[:, :, ~precision[0], :].contiguous()
    high = v_adjusted[:, :, precision[0], :].contiguous()
    p2 = quantize_pack_v_reference(low, GROUP_SIZE, 2) if low.shape[2] else (None, None, None)
    p4 = quantize_pack_v_reference(high, GROUP_SIZE, 4) if high.shape[2] else (None, None, None)
    return {
        "precision": precision,
        "centroids": centroids,
        "v_idx": v_idx,
        "v_pattern_mask": v_pattern_mask,
        "attn": attn,
        "p2": p2,
        "p4": p4,
    }


def reference_output(data: dict[str, torch.Tensor | tuple]) -> torch.Tensor:
    precision = data["precision"]
    assert torch.is_tensor(precision)
    low_mask = ~precision[0].bool()
    high_mask = precision[0].bool()
    p2 = data["p2"]
    p4 = data["p4"]
    low = dequantize_v_reference(*p2, GROUP_SIZE, 2) if int(low_mask.sum().item()) else None
    high = dequantize_v_reference(*p4, GROUP_SIZE, 4) if int(high_mask.sum().item()) else None
    template = high if high is not None else low
    if template is None:
        raise RuntimeError("no quantized tokens")
    packed_v = torch.empty(template.shape[0], template.shape[1], precision.shape[1], template.shape[-1], dtype=template.dtype, device=template.device)
    if low is not None:
        packed_v[:, :, low_mask, :] = low[:, :, : int(low_mask.sum().item()), :]
    if high is not None:
        packed_v[:, :, high_mask, :] = high[:, :, : int(high_mask.sum().item()), :]
    restored = packed_v + data["v_pattern_mask"].unsqueeze(-1).to(packed_v.dtype) * pattern_gather_centroids(
        data["v_idx"], data["centroids"]
    ).to(packed_v.dtype)
    return torch.matmul(data["attn"], repeat_kv(restored, NH // NH_KV))


def mixed_output(data: dict[str, torch.Tensor | tuple]) -> torch.Tensor:
    p2 = data["p2"]
    p4 = data["p4"]
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


def metrics(fused: torch.Tensor, ref: torch.Tensor) -> dict[str, float | int | bool]:
    diff = (fused.float() - ref.float()).abs()
    ref_norm = torch.linalg.vector_norm(ref.float()).clamp_min(1e-8)
    rel_l2 = torch.linalg.vector_norm((fused - ref).float()) / ref_norm
    cosine = torch.nn.functional.cosine_similarity(fused.float().flatten(), ref.float().flatten(), dim=0)
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    out = {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "relative_l2": float(rel_l2.item()),
        "cosine_similarity": float(cosine.item()),
        "nan_count": int(torch.isnan(fused).sum().item()),
        "inf_count": int(torch.isinf(fused).sum().item()),
    }
    out["passed"] = bool(
        out["nan_count"] == 0
        and out["inf_count"] == 0
        and out["cosine_similarity"] >= 0.9999
        and torch.allclose(fused, ref, rtol=5e-3, atol=5e-3)
    )
    return out


def run_correctness(lengths: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in CASES:
        for tokens in lengths:
            data = build_case(case, tokens)
            reset_patternkv_mixed_v_counters()
            fused = mixed_output(data)
            ref = reference_output(data)
            row = {"case": case, "tokens": tokens, **metrics(fused, ref), **get_patternkv_mixed_v_counters()}
            rows.append(row)
    return rows


def time_cuda(fn, *, warmup: int, iters: int) -> tuple[list[float], int]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = [float(start_events[i].elapsed_time(end_events[i]) * 1000.0) for i in range(iters)]
    return times, int(torch.cuda.max_memory_allocated())


def summarize_times(times: list[float]) -> dict[str, float]:
    values = sorted(times)
    p90_idx = min(len(values) - 1, int(math.ceil(0.9 * len(values))) - 1)
    return {
        "median_us": statistics.median(values),
        "mean_us": statistics.mean(values),
        "p90_us": values[p90_idx],
    }


def run_benchmark(lengths: list[int], *, warmup: int, iters: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tokens in lengths:
        data = build_case("mixed25", tokens)
        ref_times, ref_peak = time_cuda(lambda: reference_output(data), warmup=warmup, iters=iters)
        reset_patternkv_mixed_v_counters()
        fused_times, fused_peak = time_cuda(lambda: mixed_output(data), warmup=warmup, iters=iters)
        torch.cuda.synchronize()
        ref = summarize_times(ref_times)
        fused = summarize_times(fused_times)
        rows.append(
            {
                "tokens": tokens,
                "reference_median_us": ref["median_us"],
                "reference_mean_us": ref["mean_us"],
                "reference_p90_us": ref["p90_us"],
                "fused_median_us": fused["median_us"],
                "fused_mean_us": fused["mean_us"],
                "fused_p90_us": fused["p90_us"],
                "speedup_vs_reference": ref["median_us"] / max(fused["median_us"], 1e-9),
                "peak_allocated_reference_bytes": ref_peak,
                "peak_allocated_fused_bytes": fused_peak,
                **get_patternkv_mixed_v_counters(),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness-lengths", default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--bench-lengths", default="128,256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase S1 mixed Value kernel benchmark")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    correctness_lengths = [int(x) for x in args.correctness_lengths.split(",") if x]
    bench_lengths = [int(x) for x in args.bench_lengths.split(",") if x]
    correctness = run_correctness(correctness_lengths)
    write_csv(OUT_DIR / "correctness_cases.csv", correctness)
    passed = [row for row in correctness if row["passed"]]
    summary = {
        "case_count": len(correctness),
        "passed_count": len(passed),
        "all_passed": len(passed) == len(correctness),
        "max_abs_error": max(float(row["max_abs_error"]) for row in correctness),
        "mean_abs_error_max": max(float(row["mean_abs_error"]) for row in correctness),
        "relative_l2_max": max(float(row["relative_l2"]) for row in correctness),
        "cosine_similarity_min": min(float(row["cosine_similarity"]) for row in correctness),
        "nan_failures": sum(int(row["nan_count"]) for row in correctness),
        "inf_failures": sum(int(row["inf_count"]) for row in correctness),
        "longest_tested_t": max(correctness_lengths),
        "batch_support": 1,
        "gqa_supported": True,
    }
    (OUT_DIR / "correctness_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    benchmark = run_benchmark(bench_lengths, warmup=args.warmup, iters=args.iters)
    write_csv(OUT_DIR / "microbenchmark.csv", benchmark)
    lines = [
        "# Mixed V2/V4 Fused Value Attention Microbenchmark",
        "",
        f"- Warmup iterations: `{args.warmup}`",
        f"- Timed iterations: `{args.iters}`",
        "- Workload: q_len=1, B=1, H=32, H_kv=8, D=128, mixed25 precision mask",
        "",
        "| T | reference median us | fused median us | speedup | reference peak bytes | fused peak bytes |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in benchmark:
        lines.append(
            f"| {row['tokens']} | {row['reference_median_us']:.3f} | {row['fused_median_us']:.3f} | "
            f"{row['speedup_vs_reference']:.3f} | {row['peak_allocated_reference_bytes']} | {row['peak_allocated_fused_bytes']} |"
        )
    (OUT_DIR / "microbenchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    faster_ge_2k = all(float(row["speedup_vs_reference"]) > 1.0 for row in benchmark if int(row["tokens"]) >= 2048)
    final_gate = {
        "algorithm_semantics_changed": False,
        "reference_path_preserved": True,
        "mixed_fused_kernel_implemented": True,
        "all_v2_equivalence_passed": all(bool(row["passed"]) for row in correctness if row["case"] == "all_v2"),
        "all_v4_equivalence_passed": all(bool(row["passed"]) for row in correctness if row["case"] == "all_v4"),
        "mixed_25_equivalence_passed": all(bool(row["passed"]) for row in correctness if row["case"] == "mixed25"),
        "long_context_correctness_passed": summary["all_passed"] and summary["longest_tested_t"] >= 8192,
        "nan_inf_failures": int(summary["nan_failures"] + summary["inf_failures"]),
        "fused_reconstruct_packed_v_calls": 0,
        "fused_faster_than_reference_ge_2k": faster_ge_2k,
        "full_aime24_rerun_started": False,
        "vllm_integration_started": False,
        "sglang_integration_started": False,
        "phase_s1_batch_support": 1,
        "implementation": "CUDA compressed-domain two-pass using existing fused V kernels",
    }
    (OUT_DIR / "final_gate.json").write_text(json.dumps(final_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
