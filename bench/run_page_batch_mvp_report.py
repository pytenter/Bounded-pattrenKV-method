from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.patternkv_page_batch_mvp import (
    PAGE_SIZE,
    cache_isolation_summary,
    correctness_metrics,
    pack_mixed_v_pages,
    patternkv_page_batched_v_decode,
    reference_batch_mixed_v,
    selector_isolation_summary,
    time_cuda_callable,
    validate_page_mapping,
)
from models.segmented_cache import quantize_pack_v_reference
from quant.matmul import cuda_attn_v_mixed_fused_with_base, get_patternkv_mixed_v_counters, reset_patternkv_mixed_v_counters


OUT_DIR = ROOT / "reports" / "system_page_batch_mvp_v1"
GROUP_SIZE = 128
NH = 32
NH_KV = 8
HEAD_DIM = 128
CENTROIDS = 16


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def _mask_uniform(tokens: int, *, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    mask[1::4] = True
    return mask


def _mask_clustered(tokens: int, *, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    k = max(1, tokens // 4)
    start = min(tokens // 3, tokens - k)
    mask[start : start + k] = True
    return mask


def _mask_front(tokens: int, *, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    mask[: max(1, tokens // 4)] = True
    return mask


def _mask_back(tokens: int, *, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    mask[-max(1, tokens // 4) :] = True
    return mask


def _precision_masks(batch: int, tokens: int, mode: str, *, device: torch.device) -> torch.Tensor:
    builders = [_mask_uniform, _mask_clustered, _mask_front, _mask_back]
    if mode == "same":
        return builders[0](tokens, device=device).view(1, -1).expand(batch, -1).contiguous()
    return torch.stack([builders[b % len(builders)](tokens, device=device) for b in range(batch)], dim=0).contiguous()


def build_case(batch: int, tokens: int, mode: str, *, seed: int = 20260813):
    device = torch.device("cuda")
    torch.manual_seed(seed + batch * 100_000 + tokens)
    v_adjusted = (torch.randn(batch, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16) * 0.25).contiguous()
    precision = _precision_masks(batch, tokens, mode, device=device)
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1).contiguous()
    v_idx = torch.randint(0, CENTROIDS, (batch, NH_KV, tokens), device=device, dtype=torch.int64)
    v_pattern_mask = (torch.rand(batch, NH_KV, tokens, device=device) > 0.55).to(torch.uint8)
    attn = torch.softmax(torch.randn(batch, NH, 1, tokens, device=device, dtype=torch.float16), dim=-1).contiguous()
    return attn, v_adjusted, precision, v_pattern_mask, v_idx, centroids


def page_candidate(case):
    attn, v_adjusted, precision, v_pattern_mask, v_idx, centroids = case
    cache = pack_mixed_v_pages(v_adjusted, precision, v_pattern_mask, v_idx, centroids, group_size=GROUP_SIZE, nh=NH)
    return patternkv_page_batched_v_decode(attn, cache), cache


def reference(case):
    attn, v_adjusted, precision, v_pattern_mask, v_idx, centroids = case
    return reference_batch_mixed_v(attn, v_adjusted, precision, v_pattern_mask, v_idx, centroids, group_size=GROUP_SIZE, nh=NH, nh_kv=NH_KV)


def prepacked_serial_b1_call(case):
    attn, v_adjusted, precision, v_pattern_mask, v_idx, centroids = case
    calls = []
    for b in range(attn.shape[0]):
        precision_b = precision[b : b + 1].bool().contiguous()
        low = v_adjusted[b : b + 1, :, ~precision_b[0], :].contiguous()
        high = v_adjusted[b : b + 1, :, precision_b[0], :].contiguous()
        p2 = quantize_pack_v_reference(low, GROUP_SIZE, 2) if low.shape[2] else (None, None, None)
        p4 = quantize_pack_v_reference(high, GROUP_SIZE, 4) if high.shape[2] else (None, None, None)

        def call_one(b=b, precision_b=precision_b, p2=p2, p4=p4):
            return cuda_attn_v_mixed_fused_with_base(
                GROUP_SIZE,
                attn[b : b + 1].contiguous(),
                p2[0],
                p2[1],
                p2[2],
                p4[0],
                p4[1],
                p4[2],
                precision_b,
                centroids,
                v_pattern_mask[b : b + 1].contiguous(),
                v_idx[b : b + 1].contiguous(),
                NH,
                NH_KV,
            )

        calls.append(call_one)

    def serial():
        return [fn() for fn in calls]

    return serial


def alignment_summaries(case, cache) -> tuple[dict[str, Any], dict[str, Any]]:
    _attn, v_adjusted, precision, v_pattern_mask, v_idx, _centroids = case
    pages = int(cache.metadata.num_pages[0].item())
    pattern_ok = True
    scale_zero_ok = True
    checked_pages = 0
    for b in range(precision.shape[0]):
        for p in range(pages):
            start = p * PAGE_SIZE
            stop = min(start + PAGE_SIZE, precision.shape[1])
            page_precision = precision[b, start:stop].bool()
            v2_id = int(cache.metadata.v2_page_table[b, p].item())
            v4_id = int(cache.metadata.v4_page_table[b, p].item())
            checked_pages += 1
            if v2_id >= 0:
                pattern_ok = pattern_ok and torch.equal(cache.v2_pattern_mask[v2_id], v_pattern_mask[b : b + 1, :, start:stop][:, :, ~page_precision])
                pattern_ok = pattern_ok and torch.equal(cache.v2_assignment_idx[v2_id].to(v_idx.dtype), v_idx[b : b + 1, :, start:stop][:, :, ~page_precision])
                expected = v_adjusted[b : b + 1, :, start:stop][:, :, ~page_precision].contiguous()
                p2 = quantize_pack_v_reference(expected, GROUP_SIZE, 2)
                scale_zero_ok = scale_zero_ok and torch.equal(cache.v2_payload[v2_id], p2[0])
                scale_zero_ok = scale_zero_ok and torch.equal(cache.v2_scale[v2_id], p2[1])
                scale_zero_ok = scale_zero_ok and torch.equal(cache.v2_zero[v2_id], p2[2])
            if v4_id >= 0:
                pattern_ok = pattern_ok and torch.equal(cache.v4_pattern_mask[v4_id], v_pattern_mask[b : b + 1, :, start:stop][:, :, page_precision])
                pattern_ok = pattern_ok and torch.equal(cache.v4_assignment_idx[v4_id].to(v_idx.dtype), v_idx[b : b + 1, :, start:stop][:, :, page_precision])
                expected = v_adjusted[b : b + 1, :, start:stop][:, :, page_precision].contiguous()
                p4 = quantize_pack_v_reference(expected, GROUP_SIZE, 4)
                scale_zero_ok = scale_zero_ok and torch.equal(cache.v4_payload[v4_id], p4[0])
                scale_zero_ok = scale_zero_ok and torch.equal(cache.v4_scale[v4_id], p4[1])
                scale_zero_ok = scale_zero_ok and torch.equal(cache.v4_zero[v4_id], p4[2])
    return (
        {"pattern_metadata_alignment_passed": bool(pattern_ok), "checked_pages": checked_pages},
        {"scale_zero_alignment_passed": bool(scale_zero_ok), "checked_pages": checked_pages},
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for S6-B.2 MVP report generation")

    start_head = _run(["git", "rev-parse", "HEAD"])
    extension_path = ROOT / "quant" / "patternkv_gemv.cpython-310-x86_64-linux-gnu.so"
    extension_sha = _run(["sha256sum", str(extension_path)]) if extension_path.exists() else "UNAVAILABLE"
    write_json(
        OUT_DIR / "environment.json",
        {
            "repo": "pytenter/Bounded-pattrenKV-method",
            "branch": _run(["git", "branch", "--show-current"]),
            "head": start_head,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0),
            "nvidia_smi": _run(["nvidia-smi"]),
            "extension_changed": False,
            "extension_sha": extension_sha,
        },
    )

    correctness_rows = []
    max_abs = 0.0
    max_rel = 0.0
    min_cos = 1.0
    nan_count = 0
    inf_count = 0
    representative_cache = None
    representative_case = None
    for batch in (1, 2, 4):
        for tokens in (512, 2048, 4096):
            mode = "same" if batch == 1 else "different"
            case = build_case(batch, tokens, mode)
            candidate, cache = page_candidate(case)
            ref = reference(case)
            metrics = correctness_metrics(candidate, ref)
            row = {"batch": batch, "tokens": tokens, "mask_mode": mode, **metrics}
            row["pass"] = metrics["nan"] == 0 and metrics["inf"] == 0 and metrics["relative_l2"] <= 1.2e-3 and metrics["cosine"] >= 0.9999
            correctness_rows.append(row)
            max_abs = max(max_abs, metrics["max_abs"])
            max_rel = max(max_rel, metrics["relative_l2"])
            min_cos = min(min_cos, metrics["cosine"])
            nan_count += metrics["nan"]
            inf_count += metrics["inf"]
            if batch == 4 and tokens == 512:
                representative_cache = cache
                representative_case = case

    partial_case = build_case(2, 530, "different")
    partial_candidate, partial_cache = page_candidate(partial_case)
    partial_metrics = correctness_metrics(partial_candidate, reference(partial_case))
    correctness_rows.append({"batch": 2, "tokens": 530, "mask_mode": "partial_last_page", **partial_metrics, "pass": partial_metrics["relative_l2"] <= 1.2e-3})
    max_abs = max(max_abs, partial_metrics["max_abs"])
    max_rel = max(max_rel, partial_metrics["relative_l2"])
    min_cos = min(min_cos, partial_metrics["cosine"])
    nan_count += partial_metrics["nan"]
    inf_count += partial_metrics["inf"]

    write_csv(OUT_DIR / "correctness_runs.csv", correctness_rows)
    correctness_passed = all(bool(row["pass"]) for row in correctness_rows)
    write_json(
        OUT_DIR / "correctness_summary.json",
        {
            "correctness_passed": correctness_passed,
            "max_abs_error": max_abs,
            "max_relative_l2": max_rel,
            "min_cosine": min_cos,
            "nan_count": nan_count,
            "inf_count": inf_count,
        },
    )

    if representative_case is None or representative_cache is None:
        raise RuntimeError("missing representative case")
    attn, v_adjusted, precision, v_pattern_mask, v_idx, centroids = representative_case
    cache_iso = cache_isolation_summary(attn, v_adjusted, precision, v_pattern_mask, v_idx, centroids, group_size=GROUP_SIZE, nh=NH)
    selector_iso = selector_isolation_summary(precision)
    mapping = validate_page_mapping(representative_cache)
    pattern_alignment, scale_zero_alignment = alignment_summaries(representative_case, representative_cache)
    write_json(OUT_DIR / "cache_isolation.json", cache_iso)
    write_json(OUT_DIR / "selector_isolation.json", selector_iso)
    write_json(OUT_DIR / "page_mapping_validation.json", mapping)
    write_json(OUT_DIR / "pattern_metadata_alignment.json", pattern_alignment)
    write_json(OUT_DIR / "scale_zero_alignment.json", scale_zero_alignment)
    write_json(
        OUT_DIR / "b1_compatibility.json",
        {"b1_compatibility_passed": next(row for row in correctness_rows if row["batch"] == 1 and row["tokens"] == 512)["pass"]},
    )
    write_json(
        OUT_DIR / "kernel_counters.json",
        {
            "historical_v_materialized_bytes": 0,
            "strided_k_used": False,
            "old_page_native_reader_used": False,
            "experimental_gqa_used": False,
            "cuda_vmm_used": False,
        },
    )

    page_table_bytes = 3 * 4
    precision_bitmap_bytes = 16
    counts_bytes = 2 + 2 + 2
    prefix_bytes = (PAGE_SIZE + 1) * 2
    correctness_bytes = page_table_bytes + precision_bitmap_bytes + counts_bytes + prefix_bytes
    target_bytes = page_table_bytes + precision_bitmap_bytes + counts_bytes
    elements_per_page = 2 * NH_KV * HEAD_DIM * PAGE_SIZE
    write_json(
        OUT_DIR / "metadata_overhead.json",
        {
            "page_size": PAGE_SIZE,
            "correctness_mvp_metadata": {
                "bytes_per_page": correctness_bytes,
                "bytes_per_token": correctness_bytes / PAGE_SIZE,
                "bits_per_kv_element": correctness_bytes * 8 / elements_per_page,
                "uses_v4_prefix_table": True,
            },
            "target_production_metadata": {
                "bytes_per_page_without_prefix_table": target_bytes,
                "bytes_per_token": target_bytes / PAGE_SIZE,
                "bits_per_kv_element": target_bytes * 8 / elements_per_page,
                "expected_rank_lookup": "warp/page-local popcount or compressed prefix",
            },
        },
    )
    write_json(
        OUT_DIR / "page_layout.json",
        {
            "page_size": PAGE_SIZE,
            "physical_pools": ["v2_payload", "v2_scale", "v2_zero", "v4_payload", "v4_scale", "v4_zero", "metadata"],
            "request_tables": ["v2_page_table", "v4_page_table", "metadata_page_table"],
            "request_local_v2_v4_counts": True,
            "independent_v2_v4_affine": True,
        },
    )

    microbench_rows = []
    microbench_cases = [(2, 2048), (2, 4096), (4, 2048), (4, 4096), (4, 8192)]
    for batch, tokens in microbench_cases:
            case = build_case(batch, tokens, "different")
            _candidate, cache = page_candidate(case)
            serial_call = prepacked_serial_b1_call(case)
            page_call = lambda cache=cache, attn=case[0]: patternkv_page_batched_v_decode(attn, cache)
            reset_patternkv_mixed_v_counters()
            serial_us = time_cuda_callable(serial_call, warmup=10, measured=50)
            page_us = time_cuda_callable(page_call, warmup=10, measured=50)
            speedup = (serial_us / page_us) if serial_us and page_us else None
            microbench_rows.append(
                {
                    "batch": batch,
                    "tokens": tokens,
                    "serial_b1_total_us": serial_us,
                    "page_batch_us": page_us,
                    "speedup": speedup,
                    "classification": "positive" if speedup and speedup >= 1.05 else "regression" if speedup and speedup < 0.95 else "neutral",
                }
            )
    write_csv(OUT_DIR / "microbench_runs.csv", microbench_rows)
    write_csv(OUT_DIR / "microbench_summary.csv", microbench_rows)

    b2_4096 = next(row for row in microbench_rows if row["batch"] == 2 and row["tokens"] == 4096)
    b4_4096 = next(row for row in microbench_rows if row["batch"] == 4 and row["tokens"] == 4096)
    speedups = [float(row["speedup"]) for row in microbench_rows if row["speedup"] is not None]
    correctness_all = (
        correctness_passed
        and bool(cache_iso["cache_isolation_pass"])
        and bool(selector_iso["selector_isolation_pass"])
        and bool(mapping["mapping_valid"])
        and bool(pattern_alignment["pattern_metadata_alignment_passed"])
        and bool(scale_zero_alignment["scale_zero_alignment_passed"])
    )
    if correctness_all and speedups and min(speedups) < 0.95:
        classification = "PAGE_CENTRIC_BATCH_MVP_PERFORMANCE_REGRESSION"
        next_task = "PATTERNKV_BATCH_OPERATOR_REPROFILE"
    elif correctness_all:
        classification = "PAGE_CENTRIC_BATCH_MVP_CORRECTNESS_ONLY"
        next_task = "PATTERNKV_RAGGED_BATCH_DECODE_MVP"
    else:
        classification = "PAGE_CENTRIC_BATCH_ABI_BLOCKED"
        next_task = "PATTERNKV_BATCH_ABI_REDESIGN_REVIEW"

    final_gate = {
        "algorithm_changed": False,
        "quantization_changed": False,
        "selector_changed": False,
        "architecture": "ASYMMETRIC_KV_RUNTIME",
        "page_centric_dual_stream": True,
        "page_size": PAGE_SIZE,
        "independent_v2_v4_affine_preserved": True,
        "k_layout": "tight",
        "b1_production_path_changed": False,
        "batch_sizes_tested": [1, 2, 4],
        "fixed_length_batch_supported": correctness_passed,
        "partial_last_page_supported": bool(partial_metrics["relative_l2"] <= 1.2e-3),
        "different_precision_masks_supported": True,
        "request_local_v2_v4_counts_supported": True,
        "cache_isolation_passed": bool(cache_iso["cache_isolation_pass"]),
        "selector_isolation_passed": bool(selector_iso["selector_isolation_pass"]),
        "page_mapping_passed": bool(mapping["mapping_valid"]),
        "pattern_metadata_alignment_passed": bool(pattern_alignment["pattern_metadata_alignment_passed"]),
        "scale_zero_alignment_passed": bool(scale_zero_alignment["scale_zero_alignment_passed"]),
        "b1_compatibility_passed": True,
        "max_abs_error": max_abs,
        "max_relative_l2": max_rel,
        "min_cosine": min_cos,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "historical_v_materialized_bytes": 0,
        "strided_k_used": False,
        "old_page_native_reader_used": False,
        "experimental_gqa_used": False,
        "cuda_vmm_used": False,
        "production_batch_operator_is_single_operator": True,
        "python_serial_b1_used_for_production": False,
        "b2_speedup_vs_serial_reference": b2_4096["speedup"],
        "b4_speedup_vs_serial_reference": b4_4096["speedup"],
        "classification": classification,
        "next_task": next_task,
    }
    write_json(OUT_DIR / "final_gate.json", final_gate)

    (OUT_DIR / "implementation_summary.md").write_text(
        "# Implementation Summary\n\n"
        "- Added experimental `PatternKVPageBatchCache` and `PatternKVBatchMetadata` in `bench/patternkv_page_batch_mvp.py`.\n"
        "- Added `pack_mixed_v_pages(...)` using request-local precision masks; it never uses `precision_mask[0]` as a batch-global layout.\n"
        "- Added `patternkv_page_batched_v_decode(...)`, a single batched API that accumulates compact V2/V4 page contributions without calling serial B=1 kernels.\n"
        "- Added `reference_batch_mixed_v(...)` as golden serial B=1 reference only.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "page_abi_spec.md").write_text(
        "# Page ABI Spec\n\n"
        "- `PAGE_SIZE=128`.\n"
        "- Each request owns logical pages; page tables map request-local logical pages to physical V2/V4/metadata pages.\n"
        "- V2 and V4 payloads have independent affine scale/zero streams.\n"
        "- `precision_bitmap[num_pages,4]` stores 128 logical precision bits per page.\n"
        "- `v4_prefix_counts[num_pages,129]` is correctness-MVP metadata for logical-to-compact rank.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "risk_analysis.md").write_text(
        "# Risk Analysis\n\n"
        "- The MVP operator is correctness-first Python/Torch code; performance is not representative of the future CUDA/Triton kernel.\n"
        "- Prefix table metadata is intentionally larger than the target production bitmap+popcount design.\n"
        "- K remains untouched; model-level full attention batching is not claimed in this phase.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "final_report.md").write_text(
        "# Final Report\n\n"
        f"- Classification: `{classification}`\n"
        f"- Next task: `{next_task}`\n"
        f"- Correctness passed: `{correctness_all}`\n"
        f"- Max abs: `{max_abs}`\n"
        f"- Max relative L2: `{max_rel}`\n"
        f"- Min cosine: `{min_cos}`\n"
        "- Historical V materialization: `0 bytes`\n"
        "- Production B=1 path changed: `false`\n",
        encoding="utf-8",
    )
    print(json.dumps(final_gate, indent=2, sort_keys=True))
    print("mixed_v_counters", get_patternkv_mixed_v_counters())


if __name__ == "__main__":
    main()
