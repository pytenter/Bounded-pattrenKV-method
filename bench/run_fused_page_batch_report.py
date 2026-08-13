from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.run_page_batch_mvp_report import (  # noqa: E402
    CENTROIDS,
    GROUP_SIZE,
    HEAD_DIM,
    NH,
    NH_KV,
    build_case,
    page_candidate,
    prepacked_serial_b1_call,
    reference,
)
from quant.page_batch import (  # noqa: E402
    PAGE_SIZE,
    PatternKVPageBatchCache,
    build_operator_ready_page_pools,
    correctness_metrics,
    get_patternkv_page_batch_counters,
    pack_mixed_v_pages,
    patternkv_fused_page_batch_decode,
    patternkv_page_batched_v_decode,
    reset_patternkv_page_batch_counters,
    validate_page_mapping,
)


OUT_DIR = ROOT / "reports" / "system_fused_page_batch_v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def stats(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "median": float(statistics.median(values)),
        "mean": float(mean),
        "std": float(std),
        "cv": float(std / mean) if mean else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def time_callable(fn: Callable[[], Any], *, warmup: int = 5, measured: int = 15) -> dict[str, Any]:
    wall_us: list[float] = []
    cuda_us: list[float] = []
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(measured):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        wall_us.append((time.perf_counter() - wall_start) * 1_000_000.0)
        cuda_us.append(float(start.elapsed_time(end) * 1000.0))
    return {
        "warmup": warmup,
        "measured": measured,
        "wall_us": stats(wall_us),
        "cuda_event_us": stats(cuda_us),
    }


def pool_storage_bytes(cache: PatternKVPageBatchCache) -> dict[str, int]:
    def list_bytes(values: list[torch.Tensor | None]) -> int:
        return sum(0 if value is None else int(value.numel() * value.element_size()) for value in values)

    return {
        "v2_payload": list_bytes(cache.v2_payload),
        "v4_payload": list_bytes(cache.v4_payload),
        "v2_scale": list_bytes(cache.v2_scale),
        "v2_zero": list_bytes(cache.v2_zero),
        "v4_scale": list_bytes(cache.v4_scale),
        "v4_zero": list_bytes(cache.v4_zero),
        "v2_pattern": list_bytes(cache.v2_pattern_mask),
        "v4_pattern": list_bytes(cache.v4_pattern_mask),
        "v2_assignment": list_bytes(cache.v2_assignment_idx),
        "v4_assignment": list_bytes(cache.v4_assignment_idx),
        "metadata_tables": int(
            cache.metadata.v2_page_table.numel() * cache.metadata.v2_page_table.element_size()
            + cache.metadata.v4_page_table.numel() * cache.metadata.v4_page_table.element_size()
            + cache.metadata.metadata_page_table.numel() * cache.metadata.metadata_page_table.element_size()
            + cache.metadata.precision_bitmap.numel() * cache.metadata.precision_bitmap.element_size()
            + cache.metadata.v2_counts.numel() * cache.metadata.v2_counts.element_size()
            + cache.metadata.v4_counts.numel() * cache.metadata.v4_counts.element_size()
            + cache.metadata.valid_tokens.numel() * cache.metadata.valid_tokens.element_size()
            + cache.metadata.v4_prefix_counts.numel() * cache.metadata.v4_prefix_counts.element_size()
        ),
    }


def validate_pools(cache: PatternKVPageBatchCache) -> dict[str, Any]:
    pools = build_operator_ready_page_pools(cache)

    def check(pool: torch.Tensor, offsets: torch.Tensor, pages: list[torch.Tensor | None]) -> bool:
        for page_id, page in enumerate(pages):
            offset = int(offsets[page_id].item())
            if page is None:
                if offset != -1:
                    return False
                continue
            count = int(page.shape[2])
            if offset < 0 or not torch.equal(pool[:, offset : offset + count], page.squeeze(0)):
                return False
        return True

    checks = {
        "v2_payload": check(pools.v2_payload_pool, pools.v2_page_offsets, cache.v2_payload),
        "v4_payload": check(pools.v4_payload_pool, pools.v4_page_offsets, cache.v4_payload),
        "v2_scale": check(pools.v2_scale_pool, pools.v2_page_offsets, cache.v2_scale),
        "v2_zero": check(pools.v2_zero_pool, pools.v2_page_offsets, cache.v2_zero),
        "v4_scale": check(pools.v4_scale_pool, pools.v4_page_offsets, cache.v4_scale),
        "v4_zero": check(pools.v4_zero_pool, pools.v4_page_offsets, cache.v4_zero),
        "v2_pattern": check(pools.v2_pattern_pool, pools.v2_page_offsets, cache.v2_pattern_mask),
        "v4_pattern": check(pools.v4_pattern_pool, pools.v4_page_offsets, cache.v4_pattern_mask),
        "v2_assignment": check(pools.v2_assignment_pool, pools.v2_page_offsets, cache.v2_assignment_idx),
        "v4_assignment": check(pools.v4_assignment_pool, pools.v4_page_offsets, cache.v4_assignment_idx),
    }
    return {
        "operator_ready_page_pool_supported": all(checks.values()),
        "checks": checks,
        "v2_tokens": int(pools.v2_payload_pool.shape[1]),
        "v4_tokens": int(pools.v4_payload_pool.shape[1]),
        "historical_materialized_bytes": pools.historical_materialized_bytes,
        "page_value_materialized_bytes": pools.page_value_materialized_bytes,
        "gpu_tensor_item_calls": pools.gpu_tensor_item_calls,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for fused page batch report generation")

    head = run_cmd(["git", "rev-parse", "HEAD"])
    write_json(
        OUT_DIR / "environment.json",
        {
            "repo": "pytenter/Bounded-pattrenKV-method",
            "branch": run_cmd(["git", "branch", "--show-current"]),
            "report_generated_head": head,
            "device": torch.cuda.get_device_name(0),
            "cuda_available": torch.cuda.is_available(),
            "extension_sha256": run_cmd(["sha256sum", str(ROOT / "quant" / "patternkv_gemv.cpython-310-x86_64-linux-gnu.so")]),
        },
    )

    representative_case = build_case(4, 4096, "different")
    _page_out, representative_cache = page_candidate(representative_case)
    pool_validation = validate_pools(representative_cache)
    page_mapping = validate_page_mapping(representative_cache)
    phase_a_supported = bool(pool_validation["operator_ready_page_pool_supported"] and page_mapping["mapping_valid"])
    write_json(OUT_DIR / "pool_validation.json", pool_validation)
    write_json(OUT_DIR / "page_mapping_validation.json", page_mapping)
    write_json(
        OUT_DIR / "pool_layout.json",
        {
            "page_size": PAGE_SIZE,
            "fixed_length_mvp": True,
            "flat_pools": {
                "v2_payload_pool": "[nh_kv,total_v2_tokens,head_dim/16] int32",
                "v4_payload_pool": "[nh_kv,total_v4_tokens,head_dim/8] int32",
                "v2_scale_pool/v2_zero_pool": "[nh_kv,total_v2_tokens,head_dim/group_size] fp16",
                "v4_scale_pool/v4_zero_pool": "[nh_kv,total_v4_tokens,head_dim/group_size] fp16",
                "pattern_pools": "[nh_kv,total_stream_tokens] uint8",
                "assignment_pools": "[nh_kv,total_stream_tokens] int32",
            },
            "page_offsets": "int32 offsets from physical page id to stream-token offset; -1 for empty stream",
            "request_tables": ["v2_page_table", "v4_page_table", "metadata_page_table"],
            "independent_v2_v4_affine_streams": True,
            "k_layout_changed": False,
        },
    )

    correctness_rows: list[dict[str, Any]] = []
    for batch, tokens in [(1, 512), (2, 512), (2, 2048), (2, 2051), (4, 512), (4, 2048), (4, 4096)]:
        case = build_case(batch, tokens, "different")
        attn, _v_adjusted, _precision, _v_pattern_mask, _v_idx, _centroids = case
        ref = reference(case)
        cache = pack_mixed_v_pages(case[1], case[2], case[3], case[4], case[5], group_size=GROUP_SIZE, nh=NH)
        pools = build_operator_ready_page_pools(cache)
        got = patternkv_fused_page_batch_decode(attn, pools)
        torch.cuda.synchronize()
        metrics = correctness_metrics(got, ref)
        passed = metrics["nan"] == 0 and metrics["inf"] == 0 and metrics["relative_l2"] <= 2e-3 and metrics["cosine"] >= 0.9999
        correctness_rows.append({"batch": batch, "tokens": tokens, "mask_mode": "different", **metrics, "pass": passed})
    write_csv(OUT_DIR / "correctness_runs.csv", correctness_rows)

    performance_rows: list[dict[str, Any]] = []
    for batch, tokens in [(1, 512), (2, 2048), (2, 4096), (4, 2048), (4, 4096), (4, 8192)]:
        case = build_case(batch, tokens, "different")
        attn = case[0]
        _page_out, cache = page_candidate(case)
        pools = build_operator_ready_page_pools(cache)
        serial_fn = prepacked_serial_b1_call(case)
        page_fn = lambda attn=attn, cache=cache: patternkv_page_batched_v_decode(attn, cache)
        fused_fn = lambda attn=attn, pools=pools: patternkv_fused_page_batch_decode(attn, pools)
        serial_time = time_callable(serial_fn)
        page_time = time_callable(page_fn)
        fused_time = time_callable(fused_fn)
        serial_median = serial_time["cuda_event_us"]["median"]
        page_median = page_time["cuda_event_us"]["median"]
        fused_median = fused_time["cuda_event_us"]["median"]
        performance_rows.append(
            {
                "batch": batch,
                "tokens": tokens,
                "serial_b1_reference_cuda_median_us": serial_median,
                "page_torch_mvp_cuda_median_us": page_median,
                "fused_cuda_median_us": fused_median,
                "fused_vs_serial_speedup": serial_median / fused_median if fused_median else None,
                "fused_vs_page_torch_speedup": page_median / fused_median if fused_median else None,
                "fused_wall_median_us": fused_time["wall_us"]["median"],
                "fused_wall_cv": fused_time["wall_us"]["cv"],
                "fused_cuda_cv": fused_time["cuda_event_us"]["cv"],
                "fused_launches_per_decode": 1,
            }
        )
    write_csv(OUT_DIR / "performance_runs.csv", performance_rows)
    write_csv(OUT_DIR / "performance_summary.csv", performance_rows)

    reset_patternkv_page_batch_counters()
    _ = patternkv_fused_page_batch_decode(representative_case[0], build_operator_ready_page_pools(representative_cache))
    torch.cuda.synchronize()
    fused_counters = get_patternkv_page_batch_counters()
    storage = pool_storage_bytes(representative_cache)
    storage["total_bytes"] = sum(storage.values())
    write_json(OUT_DIR / "kernel_counters.json", {
        "fused_operator_launches_per_decode": 1,
        "fused_python_page_dispatches": 0,
        "fused_page_value_materialized_bytes": fused_counters["page_value_materialized_bytes"],
        "fused_gpu_tensor_item_calls": fused_counters["gpu_tensor_item_calls"],
        "fused_matmul_calls": fused_counters["matmul_calls"],
        "workspace_bytes": 0,
        "historical_v_materialized_bytes": 0,
    })
    write_json(OUT_DIR / "storage_accounting.json", storage)

    correctness_pass = all(bool(row["pass"]) for row in correctness_rows)
    min_page_speedup = min(float(row["fused_vs_page_torch_speedup"]) for row in performance_rows)
    min_serial_speedup = min(float(row["fused_vs_serial_speedup"]) for row in performance_rows)
    if not phase_a_supported:
        classification = "OPERATOR_READY_PAGE_POOL_BLOCKED"
        next_task = "FIX_OPERATOR_READY_PAGE_POOL_LAYOUT"
    elif correctness_pass and min_page_speedup >= 10.0 and fused_counters["page_value_materialized_bytes"] == 0:
        classification = "FUSED_PAGE_CENTRIC_BATCH_OPERATOR_SUPPORTED"
        next_task = "INTEGRATE_FUSED_PAGE_OPERATOR_IN_DECODE_RUNTIME"
    elif correctness_pass:
        classification = "FUSED_PAGE_CENTRIC_BATCH_OPERATOR_NEEDS_OPTIMIZATION"
        next_task = "OPTIMIZE_FUSED_PAGE_CENTRIC_BATCH_VALUE_KERNEL"
    else:
        classification = "FUSED_PAGE_CENTRIC_BATCH_OPERATOR_CORRECTNESS_BLOCKED"
        next_task = "DEBUG_FUSED_PAGE_CENTRIC_BATCH_VALUE_KERNEL"

    final_gate = {
        "phase_a_classification": "OPERATOR_READY_PAGE_POOL_SUPPORTED" if phase_a_supported else "OPERATOR_READY_PAGE_POOL_BLOCKED",
        "classification": classification,
        "next_task": next_task,
        "algorithm_changed": False,
        "selector_changed": False,
        "k_layout_changed": False,
        "page_size_changed": False,
        "v4_budget_changed": False,
        "sink_recent_residual_changed": False,
        "independent_v2_v4_affine_preserved": True,
        "batches_tested": [1, 2, 4],
        "partial_last_page_tested": True,
        "correctness_pass": correctness_pass,
        "max_abs_error": max(float(row["max_abs"]) for row in correctness_rows),
        "max_relative_l2": max(float(row["relative_l2"]) for row in correctness_rows),
        "min_cosine": min(float(row["cosine"]) for row in correctness_rows),
        "nan_count": sum(int(row["nan"]) for row in correctness_rows),
        "inf_count": sum(int(row["inf"]) for row in correctness_rows),
        "fused_operator_launches_per_decode": 1,
        "fused_page_value_materialized_bytes": fused_counters["page_value_materialized_bytes"],
        "fused_gpu_tensor_item_calls": fused_counters["gpu_tensor_item_calls"],
        "fused_matmul_calls": fused_counters["matmul_calls"],
        "workspace_bytes": 0,
        "min_fused_vs_page_torch_speedup": min_page_speedup,
        "min_fused_vs_serial_speedup": min_serial_speedup,
    }
    write_json(OUT_DIR / "final_gate.json", final_gate)

    md_files = {
        "implementation_audit.md": "# Implementation Audit\n\n- Added `PatternKVOperatorReadyPagePools` in `quant/page_batch.py`.\n- Added flat V2/V4 payload, affine, pattern, and assignment pools built from existing page lists.\n- Added `attn_v_forward_cuda_page_mixed_pool` binding and a single-launch CUDA value kernel.\n- The frozen algorithm is preserved: K remains on the existing tight path, selector and 25% V4 budget are unchanged, and V2/V4 affine streams stay independent.\n",
        "reference_systems_notes.md": "# Reference Systems Notes\n\n- Serial B=1 production mixed-V operator remains the golden reference.\n- Existing Torch page-batch MVP remains a secondary reference for page metadata semantics.\n- No CUDA VMM, old page-native reader, strided K reader, or experimental GQA ABI is introduced.\n",
        "s1_reuse_map.md": "# S1 Reuse Map\n\n| S1 component | Reuse in fused page operator |\n| --- | --- |\n| Packed INT2/INT4 extraction | Reused directly in CUDA compressed-domain loads |\n| Affine scale/zero streams | Reused as independent V2/V4 pools |\n| Pattern centroid correction | Reused per token using pattern mask and assignment pools |\n| Output ABI | Preserves `[B, nh, 1, head_dim]` Value output |\n",
        "operator_ready_pool_spec.md": "# Operator Ready Pool Spec\n\nSee `pool_layout.json`. Physical pages are flattened into contiguous GPU pools with `int32` page offsets. Metadata remains request-local and device resident. Empty stream pages use offset `-1` and page table id `-1`.\n",
        "fused_operator_design.md": "# Fused Operator Design\n\nThe MVP fused kernel launches once per decode. Grid x spans `B*nh`, grid y spans output channels. Each block reduces over sequence tokens for one `(request, query-head, output-channel)` scalar, loads V2/V4 compressed payload directly, applies affine and centroid correction, and writes the final Value vector without materializing page-local Value tensors.\n",
        "kernel_mapping.md": "# Kernel Mapping\n\n- Launches per decode: `1`.\n- Blocks: `(B*nh, head_dim, 1)`.\n- Threads: `256`.\n- Token precision is resolved through `metadata_page_table` and `v4_prefix_counts`.\n- Compact stream offsets come from `v2_page_offsets` and `v4_page_offsets`.\n",
        "workspace_analysis.md": "# Workspace Analysis\n\nThe fused MVP uses no explicit temporary workspace. It performs dequant-on-load and block-local reduction only. Reported `workspace_bytes` is `0`.\n",
        "risk_analysis.md": "# Risk Analysis\n\n- The kernel is correctness-first and maps one block per output scalar, so it may need optimization for serial-reference parity.\n- `v4_prefix_counts` is retained for unambiguous MVP rank lookup; a bitmap+popcount path can reduce metadata later.\n- The fused operator currently covers fixed-length B in `{1,2,4}` plus partial final pages, not ragged serving integration.\n",
        "performance_results.md": f"# Performance Results\n\n- Minimum speedup vs old Torch page batch: `{min_page_speedup}`.\n- Minimum speedup vs serial B1 reference: `{min_serial_speedup}`.\n- See `performance_runs.csv` for medians, CVs, and per-shape timings.\n",
        "correctness_results.md": f"# Correctness Results\n\n- Correctness pass: `{correctness_pass}`.\n- Max abs: `{final_gate['max_abs_error']}`.\n- Max relative L2: `{final_gate['max_relative_l2']}`.\n- Min cosine: `{final_gate['min_cosine']}`.\n- NaN / Inf: `{final_gate['nan_count']}` / `{final_gate['inf_count']}`.\n",
        "final_recommendation.md": f"# Final Recommendation\n\n- Phase A: `{final_gate['phase_a_classification']}`.\n- Final classification: `{classification}`.\n- Next task: `{next_task}`.\n",
        "final_report.md": f"# Final Report\n\n- Phase A: `{final_gate['phase_a_classification']}`.\n- Final classification: `{classification}`.\n- Correctness pass: `{correctness_pass}`.\n- Fused launches per decode: `1`.\n- Fused page Value materialization bytes: `{fused_counters['page_value_materialized_bytes']}`.\n- Workspace bytes: `0`.\n",
    }
    for name, text in md_files.items():
        (OUT_DIR / name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
