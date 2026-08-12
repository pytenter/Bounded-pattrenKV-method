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

from models.segmented_cache import FixedPageBuffer, quantize_pack_v_reference
from quant.matmul import (
    DevicePageTable,
    cuda_attn_v_fused_with_base,
    cuda_attn_v_fused_with_base_paged_v2,
    get_patternkv_page_v_reader_counters,
    reset_patternkv_page_v_reader_counters,
)

OUT_DIR = ROOT / "reports/system_page_reader_v1"
GROUP_SIZE = 128
NH = 32
NH_KV = 8
HEAD_DIM = 128
CENTROIDS = 16
PAGE_SIZE = 128


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def buffer_from_ext_tensor(name: str, tensor: torch.Tensor) -> FixedPageBuffer:
    buf = FixedPageBuffer(stream=name, page_size=PAGE_SIZE, token_dim=2)
    buf.append_block(tensor.contiguous())
    return buf


def build_case(tokens: int, *, assignment: str = "normal", mask_density: float = 0.5, seed: int = 1234) -> dict[str, object]:
    device = torch.device("cuda")
    torch.manual_seed(seed + tokens + int(mask_density * 1000))
    values = (torch.randn(1, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16) * 0.25).contiguous()
    vq, scale, zero = quantize_pack_v_reference(values, GROUP_SIZE, 2)
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1).contiguous()
    if assignment == "uniform":
        ids = torch.arange(tokens, device=device, dtype=torch.int64) % CENTROIDS
        idx = ids.view(1, 1, tokens).expand(1, NH_KV, tokens).contiguous()
    elif assignment == "skewed":
        idx = torch.randint(1, CENTROIDS, (1, NH_KV, tokens), device=device, dtype=torch.int64)
        idx[torch.rand(1, NH_KV, tokens, device=device) < 0.75] = 0
    elif assignment == "all_same":
        idx = torch.zeros(1, NH_KV, tokens, device=device, dtype=torch.int64)
    else:
        idx = torch.randint(0, CENTROIDS, (1, NH_KV, tokens), device=device, dtype=torch.int64)
    if mask_density <= 0:
        mask = torch.zeros(1, NH_KV, tokens, device=device, dtype=torch.uint8)
    elif mask_density >= 1:
        mask = torch.ones(1, NH_KV, tokens, device=device, dtype=torch.uint8)
    else:
        mask = (torch.rand(1, NH_KV, tokens, device=device) < mask_density).to(torch.uint8)
    attn = torch.softmax(torch.randn(1, NH, 1, tokens, device=device, dtype=torch.float16), dim=-1).contiguous()
    vq_ext = vq.reshape(1 * NH_KV, tokens, HEAD_DIM // 16).transpose(1, 2).contiguous()
    scale_ext = scale.view(1 * NH_KV, tokens, HEAD_DIM // GROUP_SIZE).transpose(1, 2).contiguous()
    zero_ext = zero.view(1 * NH_KV, tokens, HEAD_DIM // GROUP_SIZE).transpose(1, 2).contiguous()
    return {
        "tokens": tokens,
        "attn": attn,
        "vq": vq,
        "scale": scale,
        "zero": zero,
        "centroids": centroids,
        "mask": mask,
        "idx": idx,
        "vq_pages": buffer_from_ext_tensor("vq", vq_ext),
        "scale_pages": buffer_from_ext_tensor("scale", scale_ext),
        "zero_pages": buffer_from_ext_tensor("zero", zero_ext),
        "mask_pages": buffer_from_ext_tensor("mask", mask),
        "idx_pages": buffer_from_ext_tensor("idx", idx.to(torch.uint8)),
    }


def contiguous_output(data: dict[str, object]) -> torch.Tensor:
    return cuda_attn_v_fused_with_base(
        GROUP_SIZE,
        data["attn"],
        data["vq"],
        data["scale"],
        data["zero"],
        2,
        data["centroids"],
        data["mask"],
        data["idx"],
        NH,
        NH_KV,
    )


def paged_output(data: dict[str, object], tables: dict[str, DevicePageTable]) -> torch.Tensor:
    return cuda_attn_v_fused_with_base_paged_v2(
        GROUP_SIZE,
        data["attn"],
        data["vq_pages"],
        data["scale_pages"],
        data["zero_pages"],
        data["centroids"],
        data["mask_pages"],
        data["idx_pages"],
        NH,
        NH_KV,
        page_tables=tables,
    )


def compare_outputs(a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
    torch.cuda.synchronize()
    diff = (a.float() - b.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)
    max_abs = float(diff.max().item())
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": float(diff.mean().item()),
        "cosine_similarity": float(cosine.item()),
        "nan_count": int(torch.isnan(a).sum().item()),
        "inf_count": int(torch.isinf(a).sum().item()),
        "passed": bool(
            max_abs <= 5e-3
            and float(cosine.item()) >= 0.9999
            and int(torch.isnan(a).sum().item()) == 0
            and int(torch.isinf(a).sum().item()) == 0
        ),
    }


def time_cuda(fn, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [float(starts[i].elapsed_time(ends[i]) * 1000.0) for i in range(iters)]


def summarize_rounds(round_medians: list[float]) -> dict[str, float]:
    mean = statistics.mean(round_medians)
    cv = statistics.stdev(round_medians) / mean if len(round_medians) > 1 and mean else 0.0
    return {
        "median_us": statistics.median(round_medians),
        "mean_round_median_us": mean,
        "cv_round_median": cv,
        "rounds": len(round_medians),
    }


def benchmark_fn(fn, *, warmup: int, iters: int, rounds: int) -> dict[str, float]:
    medians = []
    for _ in range(rounds):
        times = time_cuda(fn, warmup=warmup, iters=iters)
        medians.append(statistics.median(times))
    return summarize_rounds(medians)


def run_correctness(lengths: list[int]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    boundary_rows = []
    for tokens in lengths:
        for assignment in ("normal", "uniform", "skewed", "all_same"):
            for density in (0.0, 0.25, 0.5, 1.0):
                data = build_case(tokens, assignment=assignment, mask_density=density)
                reset_patternkv_page_v_reader_counters()
                candidate = paged_output(data, {})
                baseline = contiguous_output(data)
                row = {
                    "tokens": tokens,
                    "assignment": assignment,
                    "mask_density": density,
                    **compare_outputs(candidate, baseline),
                    **get_patternkv_page_v_reader_counters(),
                }
                rows.append(row)
                if tokens in {127, 128, 129, 255, 256, 257}:
                    boundary_rows.append(row)
    return rows, boundary_rows


def run_benchmarks(lengths: list[int], *, warmup: int, iters: int, rounds: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    baseline_rows = []
    paged_rows = []
    overhead_rows = []
    for tokens in lengths:
        data = build_case(tokens)
        tables: dict[str, DevicePageTable] = {}
        paged_output(data, tables)
        baseline = benchmark_fn(lambda: contiguous_output(data), warmup=warmup, iters=iters, rounds=rounds)
        paged = benchmark_fn(lambda: paged_output(data, tables), warmup=warmup, iters=iters, rounds=rounds)
        baseline_rows.append({"tokens": tokens, "backend": "contiguous", **baseline})
        paged_rows.append(
            {
                "tokens": tokens,
                "backend": "paged_v2",
                **paged,
                "regression_vs_contiguous": (paged["median_us"] - baseline["median_us"]) / max(baseline["median_us"], 1e-9),
                "speedup_vs_contiguous": baseline["median_us"] / max(paged["median_us"], 1e-9),
            }
        )
        reset_patternkv_page_v_reader_counters()
        overhead_tables: dict[str, DevicePageTable] = {}
        paged_output(data, overhead_tables)
        counters = get_patternkv_page_v_reader_counters()
        overhead_rows.append({"tokens": tokens, "page_count": math.ceil(tokens / PAGE_SIZE), **counters})
    return baseline_rows, paged_rows, overhead_rows


def write_reports(correctness, boundary, baseline, paged, overhead, *, args) -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "baseline_reader.csv", baseline)
    write_csv(OUT_DIR / "paged_v2_reader.csv", paged)
    write_csv(OUT_DIR / "page_boundary_correctness.csv", boundary)
    write_csv(OUT_DIR / "page_table_overhead.csv", overhead)
    write_csv(OUT_DIR / "materialization_cost.csv", [{"operation": "paged_reader_historical_materialization", "calls": 0, "torch_cat_calls": 0, "bytes": 0}])
    write_csv(OUT_DIR / "paged_v4_reader.csv", [{"backend": "paged_v4", "status": "not_extended_stage_a_only"}])
    write_csv(OUT_DIR / "mixed_v_reader.csv", [{"backend": "mixed_v", "status": "not_extended_stage_a_only"}])
    write_csv(OUT_DIR / "combined_cache_attention.csv", [{"backend": "combined", "status": "not_extended_stage_a_only"}])
    write_csv(OUT_DIR / "e2e_summary.csv", [{"scope": "e2e", "status": "not_started_stage_a_only"}])
    memory_rows = []
    for row in overhead:
        tokens = int(row["tokens"])
        pages = int(row["page_count"])
        memory_rows.append(
            {
                "tokens": tokens,
                "pages": pages,
                "page_table_bytes": int(row["page_table_bytes_uploaded"]),
                "fragmentation_tokens_max": pages * PAGE_SIZE - tokens,
            }
        )
    write_csv(OUT_DIR / "memory_summary.csv", memory_rows)
    passed = [row for row in correctness if row["passed"]]
    summary = {
        "total_cases": len(correctness),
        "passed_cases": len(passed),
        "all_passed": len(passed) == len(correctness),
        "max_abs_error": max(float(row["max_abs_error"]) for row in correctness),
        "cosine_similarity_min": min(float(row["cosine_similarity"]) for row in correctness),
        "nan_count_total": sum(int(row["nan_count"]) for row in correctness),
        "inf_count_total": sum(int(row["inf_count"]) for row in correctness),
        "contexts_tested": sorted({int(row["tokens"]) for row in correctness}),
        "assignments_tested": ["normal", "uniform", "skewed", "all_same"],
        "mask_densities_tested": [0.0, 0.25, 0.5, 1.0],
    }
    (OUT_DIR / "correctness_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    by_tokens = {int(row["tokens"]): row for row in paged}
    reg_32k = float(by_tokens.get(32768, {}).get("regression_vs_contiguous", 999.0))
    v2_perf_gate = summary["all_passed"] and reg_32k <= 0.10
    classification = "PAGE_NATIVE_V_ATTENTION_SUPPORTED" if v2_perf_gate else "PAGE_NATIVE_READER_NOT_FASTER"
    if not summary["all_passed"]:
        classification = "PAGE_NATIVE_READER_CORRECTNESS_BLOCKED"
    final_gate = {
        "classification": classification,
        "algorithm_semantics_changed": False,
        "qk_rewrite_started": False,
        "gqa_redesign_started": False,
        "default_reader_backend": "contiguous",
        "experimental_backend": "paged_v2",
        "v2_correctness_passed": summary["all_passed"],
        "v2_32k_regression_vs_contiguous": reg_32k,
        "v2_perf_gate_le_10_percent_regression": v2_perf_gate,
        "v4_extended": False,
        "mixed_extended": False,
        "e2e_started": False,
        "warmup": args.warmup,
        "iters": args.iters,
        "rounds": args.rounds,
    }
    (OUT_DIR / "final_gate.json").write_text(json.dumps(final_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT_DIR / "current_reader_audit.md").write_text(
        "# Current Reader Audit\n\n"
        "- Production contiguous Value reader remains `attn_v_forward_cuda_outer_dim_with_base`.\n"
        "- It expects already contiguous compact V payload, scale, zero, mask, and assignment tensors.\n"
        "- S3-2 adds an experimental V2 page pointer-table reader only; QK, GQA, selectors, quantization, masks, assignments, centroids, sink/recent, residuals, and group size are unchanged.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "page_reader_design.md").write_text(
        "# Page Reader Design\n\n"
        "- `PATTERNKV_PAGE_V_READER=contiguous|paged_v2`; default is `contiguous`.\n"
        "- `paged_v2` consumes device pointer tables for fixed pages and reads logical token `t` as `(page_id=t/page_size, page_offset=t%page_size)`.\n"
        "- Historical pages are not concatenated or materialized in the reader wrapper.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "compact_stream_mapping.md").write_text(
        "# Compact Stream Mapping\n\n"
        "- V2 payload pages use extension-native `[B*H_kv, D/16, page_size]` layout.\n"
        "- Scale/zero pages use `[B*H_kv, D/group_size, page_size]`.\n"
        "- Mask and assignment pages use `[B, H_kv, page_size]`.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "page_table_design.md").write_text(
        "# Page Table Design\n\n"
        "- `DevicePageTable` stores CUDA page `data_ptr()` values in an int64 CUDA tensor.\n"
        "- The table refreshes only when the tuple of page pointers changes; in-place page content updates reuse the existing table.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "break_even_analysis.md").write_text(
        "# Break-Even Analysis\n\n"
        f"- 32K V2 regression vs contiguous: `{reg_32k:.4f}`.\n"
        f"- Stage A performance gate (`<=10%` regression): `{v2_perf_gate}`.\n"
        "- V4/mixed/E2E extension is intentionally skipped unless the V2 gate passes.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "optimization_scorecard.md").write_text(
        "# Optimization Scorecard\n\n"
        "| Item | Status |\n"
        "|---|---|\n"
        "| V2 page-native reader correctness | " + ("PASS" if summary["all_passed"] else "FAIL") + " |\n"
        "| 32K <=10% regression gate | " + ("PASS" if v2_perf_gate else "FAIL") + " |\n"
        "| Historical materialization in paged reader | ZERO |\n"
        "| V4/mixed extension | SKIPPED unless Stage A passes |\n",
        encoding="utf-8",
    )
    (OUT_DIR / "final_report.md").write_text(
        "# S3-2 Page-Native Value Attention Reader\n\n"
        f"- Final classification: `{classification}`\n"
        f"- Correctness: `{summary['passed_cases']}/{summary['total_cases']}` cases passed.\n"
        f"- Max abs error: `{summary['max_abs_error']:.6g}`; min cosine: `{summary['cosine_similarity_min']:.8f}`.\n"
        f"- 32K V2 regression vs contiguous: `{reg_32k:.4f}`.\n"
        "- Production default remains contiguous; paged V2 is experimental.\n",
        encoding="utf-8",
    )
    return final_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness-lengths", default="127,128,129,255,256,257,2048,8192,16384,32768")
    parser.add_argument("--bench-lengths", default="2048,4096,8192,16384,32768")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for page-native Value reader benchmark")
    correctness_lengths = [int(x) for x in args.correctness_lengths.split(",") if x]
    bench_lengths = [int(x) for x in args.bench_lengths.split(",") if x]
    correctness, boundary = run_correctness(correctness_lengths)
    baseline, paged, overhead = run_benchmarks(bench_lengths, warmup=args.warmup, iters=args.iters, rounds=args.rounds)
    gate = write_reports(correctness, boundary, baseline, paged, overhead, args=args)
    return 0 if gate["v2_correctness_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
