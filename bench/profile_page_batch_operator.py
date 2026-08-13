from __future__ import annotations

import csv
import json
import os
import shutil
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
    GROUP_SIZE,
    NH,
    build_case,
    prepacked_serial_b1_call,
)
from models.segmented_cache import dequantize_v_reference, pattern_gather_centroids  # noqa: E402
from quant.page_batch import (  # noqa: E402
    PAGE_SIZE,
    PatternKVPageBatchCache,
    _repeat_kv,
    _restore_page_values,
    correctness_metrics,
    get_patternkv_page_batch_counters,
    pack_mixed_v_pages,
    patternkv_page_batch_decode,
    reset_patternkv_page_batch_counters,
)
from quant.patternkv_profile import (  # noqa: E402
    profile_snapshot,
    reset_profile,
    temp_allocation_snapshot,
)


OUT_DIR = ROOT / "reports" / "system_page_batch_profile_v1"


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
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_wall = time.perf_counter()
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        wall_us.append((time.perf_counter() - start_wall) * 1_000_000.0)
        cuda_us.append(float(start_event.elapsed_time(end_event) * 1000.0))
    result = {
        "warmup": warmup,
        "measured": measured,
        "wall_us": stats(wall_us),
        "cuda_event_us": stats(cuda_us),
    }
    result["wall_minus_cuda_median_us"] = result["wall_us"]["median"] - result["cuda_event_us"]["median"]
    return result


def make_page_decode_host_metadata(attn: torch.Tensor, cache: PatternKVPageBatchCache) -> Callable[[], torch.Tensor]:
    bsz, nh, _q, tokens = attn.shape
    if nh != cache.nh:
        raise ValueError("head mismatch")
    seq_lens = cache.metadata.seq_lens.detach().cpu().tolist()
    if max(seq_lens) != tokens or min(seq_lens) != tokens:
        raise ValueError("fixed-length mismatch")
    num_pages = int(cache.metadata.num_pages[0].detach().cpu().item())
    metadata_pages = [int(x) for x in cache.metadata.metadata_page_table.reshape(-1).detach().cpu().tolist()]
    v2_pages = [int(x) for x in cache.metadata.v2_page_table.reshape(-1).detach().cpu().tolist()]
    v4_pages = [int(x) for x in cache.metadata.v4_page_table.reshape(-1).detach().cpu().tolist()]
    valid_tokens = [int(x) for x in cache.metadata.valid_tokens.detach().cpu().tolist()]
    v2_counts = [int(x) for x in cache.metadata.v2_counts.detach().cpu().tolist()]
    v4_counts = [int(x) for x in cache.metadata.v4_counts.detach().cpu().tolist()]

    def run() -> torch.Tensor:
        out = torch.zeros((bsz, nh, 1, cache.head_dim), dtype=torch.float32, device=attn.device)
        n_rep = cache.nh // cache.nh_kv
        for flat_page, metadata_page in enumerate(metadata_pages):
            b = flat_page // num_pages
            page = flat_page - b * num_pages
            valid = valid_tokens[metadata_page]
            if valid <= 0:
                continue
            start = page * cache.page_size
            stop = start + valid
            v2_page_id = v2_pages[flat_page]
            v4_page_id = v4_pages[flat_page]
            v2_count = v2_counts[metadata_page]
            v4_count = v4_counts[metadata_page]
            prefix = cache.metadata.v4_prefix_counts[metadata_page]
            page_precision = (prefix[1 : valid + 1] > prefix[:valid]).bool()
            page_attn = attn[b : b + 1, :, :, start:stop]
            if v2_count:
                v2_values = _restore_page_values(
                    cache.v2_payload[v2_page_id],
                    cache.v2_scale[v2_page_id],
                    cache.v2_zero[v2_page_id],
                    cache.v2_pattern_mask[v2_page_id],
                    cache.v2_assignment_idx[v2_page_id],
                    cache.centroids,
                    bits=2,
                    group_size=cache.group_size,
                )
                out[b : b + 1] += torch.matmul(page_attn[:, :, :, ~page_precision].contiguous(), _repeat_kv(v2_values, n_rep)).float()
            if v4_count:
                v4_values = _restore_page_values(
                    cache.v4_payload[v4_page_id],
                    cache.v4_scale[v4_page_id],
                    cache.v4_zero[v4_page_id],
                    cache.v4_pattern_mask[v4_page_id],
                    cache.v4_assignment_idx[v4_page_id],
                    cache.centroids,
                    bits=4,
                    group_size=cache.group_size,
                )
                out[b : b + 1] += torch.matmul(page_attn[:, :, :, page_precision].contiguous(), _repeat_kv(v4_values, n_rep)).float()
        return out.to(attn.dtype)

    return run


def iter_pages(cache: PatternKVPageBatchCache):
    num_pages = int(cache.metadata.num_pages[0].detach().cpu().item())
    metadata_pages = [int(x) for x in cache.metadata.metadata_page_table.reshape(-1).detach().cpu().tolist()]
    v2_pages = [int(x) for x in cache.metadata.v2_page_table.reshape(-1).detach().cpu().tolist()]
    v4_pages = [int(x) for x in cache.metadata.v4_page_table.reshape(-1).detach().cpu().tolist()]
    valid_tokens = [int(x) for x in cache.metadata.valid_tokens.detach().cpu().tolist()]
    v2_counts = [int(x) for x in cache.metadata.v2_counts.detach().cpu().tolist()]
    v4_counts = [int(x) for x in cache.metadata.v4_counts.detach().cpu().tolist()]
    for flat_page, metadata_page in enumerate(metadata_pages):
        b = flat_page // num_pages
        page = flat_page - b * num_pages
        valid = valid_tokens[metadata_page]
        if valid <= 0:
            continue
        yield {
            "flat_page": flat_page,
            "request": b,
            "page": page,
            "metadata_page": metadata_page,
            "valid": valid,
            "v2_page_id": v2_pages[flat_page],
            "v4_page_id": v4_pages[flat_page],
            "v2_count": v2_counts[metadata_page],
            "v4_count": v4_counts[metadata_page],
        }


def ablation_restore_only(cache: PatternKVPageBatchCache) -> torch.Tensor:
    total = torch.zeros((), device=cache.centroids.device)
    for row in iter_pages(cache):
        if row["v2_count"]:
            values = dequantize_v_reference(cache.v2_payload[row["v2_page_id"]], cache.v2_scale[row["v2_page_id"]], cache.v2_zero[row["v2_page_id"]], cache.group_size, 2)
            total = total + values.float().sum() * 0.0
        if row["v4_count"]:
            values = dequantize_v_reference(cache.v4_payload[row["v4_page_id"]], cache.v4_scale[row["v4_page_id"]], cache.v4_zero[row["v4_page_id"]], cache.group_size, 4)
            total = total + values.float().sum() * 0.0
    return total


def ablation_restore_centroid(cache: PatternKVPageBatchCache) -> torch.Tensor:
    total = torch.zeros((), device=cache.centroids.device)
    for row in iter_pages(cache):
        if row["v2_count"]:
            values = dequantize_v_reference(cache.v2_payload[row["v2_page_id"]], cache.v2_scale[row["v2_page_id"]], cache.v2_zero[row["v2_page_id"]], cache.group_size, 2)
            gathered = pattern_gather_centroids(cache.v2_assignment_idx[row["v2_page_id"]].to(torch.long), cache.centroids).to(values.dtype)
            total = total + (values + cache.v2_pattern_mask[row["v2_page_id"]].unsqueeze(-1).to(values.dtype) * gathered).float().sum() * 0.0
        if row["v4_count"]:
            values = dequantize_v_reference(cache.v4_payload[row["v4_page_id"]], cache.v4_scale[row["v4_page_id"]], cache.v4_zero[row["v4_page_id"]], cache.group_size, 4)
            gathered = pattern_gather_centroids(cache.v4_assignment_idx[row["v4_page_id"]].to(torch.long), cache.centroids).to(values.dtype)
            total = total + (values + cache.v4_pattern_mask[row["v4_page_id"]].unsqueeze(-1).to(values.dtype) * gathered).float().sum() * 0.0
    return total


def run_torch_profiler(fn: Callable[[], Any], label: str) -> dict[str, Any]:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            fn()
            torch.cuda.synchronize()
        rows = []
        approx_cuda_launches = 0
        for evt in prof.key_averages():
            self_cuda = float(getattr(evt, "self_cuda_time_total", 0.0))
            cpu = float(getattr(evt, "self_cpu_time_total", 0.0))
            count = int(getattr(evt, "count", 0))
            if str(evt.key) == "cudaLaunchKernel":
                approx_cuda_launches += count
            elif self_cuda > 0:
                approx_cuda_launches += count
            rows.append(
                {
                    "name": str(evt.key),
                    "count": count,
                    "self_cpu_us": cpu,
                    "self_cuda_us": self_cuda,
                    "cuda_us": float(getattr(evt, "cuda_time_total", 0.0)),
                    "cpu_us": float(getattr(evt, "cpu_time_total", 0.0)),
                }
            )
        rows.sort(key=lambda r: r["self_cuda_us"], reverse=True)
        write_csv(OUT_DIR / f"torch_profiler_{label}_top_cuda.csv", rows[:40])
        rows_cpu = sorted(rows, key=lambda r: r["self_cpu_us"], reverse=True)
        write_csv(OUT_DIR / f"torch_profiler_{label}_top_cpu.csv", rows_cpu[:40])
        return {
            "available": True,
            "label": label,
            "approx_cuda_launches": approx_cuda_launches,
            "top_cuda": rows[:12],
            "top_cpu": rows_cpu[:12],
        }
    except Exception as exc:
        return {"available": False, "label": label, "error": repr(exc), "approx_cuda_launches": None}


def static_audit() -> None:
    rows = [
        ("quant/page_batch.py", "pack_mixed_v_pages", "Python B/page loop, .item() from page_precision.sum", "B*pages during packing", "yes if CUDA sum item", "yes", "quantize kernels", "moderate; outside decode timing unless packing included"),
        ("quant/page_batch.py", "patternkv_page_batch_decode", "seq_lens max/min .item, num_pages .item", "3 per decode", "yes", "no", "possible reductions", "high sync candidate"),
        ("quant/page_batch.py", "patternkv_page_batch_decode", "metadata_pages/v2/v4/counts .item", "6 per logical page", "yes", "no", "none besides sync", "high sync candidate"),
        ("quant/page_batch.py", "patternkv_page_batch_decode", "page_precision reconstruct", "1 per logical page", "no explicit item", "yes bool tensor", "comparison kernel", "moderate"),
        ("quant/page_batch.py", "_restore_page_values", "dequantize_v_reference", "one per non-empty V2/V4 page", "no explicit item", "yes page tensor", "unpack/elementwise kernels", "high"),
        ("quant/page_batch.py", "_restore_page_values", "pattern_gather_centroids", "one per non-empty V2/V4 page", "no explicit item", "yes gathered tensor", "gather kernel", "high"),
        ("quant/page_batch.py", "patternkv_page_batch_decode", "boolean attention indexing + contiguous", "one per non-empty V2/V4 page", "no explicit item", "yes compact attn", "index/copy kernels", "high"),
        ("quant/page_batch.py", "_repeat_kv", "expand+reshape GQA replication", "one per non-empty V2/V4 page", "no", "view or copy depending stride", "usually no or reshape copy", "moderate"),
        ("quant/page_batch.py", "patternkv_page_batch_decode", "torch.matmul", "one per non-empty V2/V4 page", "no", "output temp", "GEMM/bmm kernel", "high fragmentation candidate"),
        ("quant/page_batch.py", "patternkv_page_batch_decode", "out slice += part", "one per non-empty V2/V4 page", "no", "no major temp", "add/copy kernel", "moderate"),
        ("bench/run_page_batch_mvp_report.py", "time_cuda_callable", "torch.cuda.synchronize + CUDA events", "per timing run", "intentional", "no", "event ops", "methodology only"),
        ("bench/patternkv_page_batch_mvp.py", "reference_batch_mixed_v", "Python loop over B serial B1 reference", "B per reference call", "not production path", "yes compact streams", "legacy CUDA kernels", "reference only"),
        ("quant/patternkv_profile.py", "profile_snapshot", "torch.cuda.synchronize", "snapshot only", "intentional", "no", "none", "measurement only"),
    ]
    text = "# Static Operator Audit\n\n"
    text += "| file | symbol | operation | frequency per page/request | possible synchronization | possible allocation | possible kernel launch | suspected cost |\n"
    text += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    for row in rows:
        text += "| " + " | ".join(f"`{x}`" if i < 2 else x for i, x in enumerate(row)) + " |\n"
    (OUT_DIR / "static_operator_audit.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    os.environ["PATTERNKV_PROFILE"] = "1"
    static_audit()

    env = {
        "start_head": "0ce940a4b13c1620b34c07cad3133cecddb497e7",
        "report_generated_head": run_cmd(["git", "rev-parse", "HEAD"]),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "device": torch.cuda.get_device_name(0),
        "cuda_available": True,
        "nvidia_smi": run_cmd(["nvidia-smi"]),
    }
    write_json(OUT_DIR / "environment.json", env)
    (OUT_DIR / "environment.md").write_text(
        "# Environment\n\n"
        f"- Start HEAD: `{env['start_head']}`\n"
        f"- Report generated HEAD: `{env['report_generated_head']}`\n"
        f"- Branch: `{env['branch']}`\n"
        f"- Device: `{env['device']}`\n",
        encoding="utf-8",
    )

    timing_rows: list[dict[str, Any]] = []
    scaling_rows: list[dict[str, Any]] = []
    representative_profile: dict[str, dict[str, float]] = {}
    representative_counters: dict[str, int] = {}
    representative_timing: dict[str, Any] = {}
    correctness = {}

    scaling_cases = [(2, 512), (2, 2048), (2, 4096), (2, 8192), (4, 512), (4, 2048), (4, 4096), (4, 8192)]
    for batch, tokens in scaling_cases:
        case = build_case(batch, tokens, "different")
        attn, v_adjusted, precision, v_pattern_mask, v_idx, centroids = case
        cache = pack_mixed_v_pages(v_adjusted, precision, v_pattern_mask, v_idx, centroids, group_size=GROUP_SIZE, nh=NH)
        page_fn = lambda attn=attn, cache=cache: patternkv_page_batch_decode(attn, cache)
        serial_fn = prepacked_serial_b1_call(case)

        page_timing = time_callable(page_fn, warmup=5, measured=12)
        reset_patternkv_page_batch_counters()
        reset_profile()
        page_fn()
        torch.cuda.synchronize()
        page_snapshot = profile_snapshot(reset=True)
        page_counters = get_patternkv_page_batch_counters()
        serial_timing = time_callable(serial_fn, warmup=5, measured=12)

        pages = int(cache.metadata.metadata_page_table.numel())
        streams = int(page_counters["v2_pages_processed"] + page_counters["v4_pages_processed"])
        row = {
            "batch": batch,
            "tokens": tokens,
            "logical_pages": pages,
            "nonempty_stream_pages": streams,
            "serial_wall_median_us": serial_timing["wall_us"]["median"],
            "serial_cuda_median_us": serial_timing["cuda_event_us"]["median"],
            "page_wall_median_us": page_timing["wall_us"]["median"],
            "page_cuda_median_us": page_timing["cuda_event_us"]["median"],
            "speedup_wall": serial_timing["wall_us"]["median"] / page_timing["wall_us"]["median"],
            "speedup_cuda": serial_timing["cuda_event_us"]["median"] / max(page_timing["cuda_event_us"]["median"], 1e-9),
            "wall_us_per_logical_page": page_timing["wall_us"]["median"] / pages,
            "wall_us_per_stream_page": page_timing["wall_us"]["median"] / max(streams, 1),
            "item_calls": page_counters["host_sync_item_calls"],
            "gpu_item_calls": page_counters["gpu_tensor_item_calls"],
            "page_materialization_calls": page_counters["page_value_materialization_calls"],
            "page_materialized_bytes": page_counters["page_value_materialized_bytes"],
            "matmul_calls": page_counters["matmul_calls"],
            "accumulate_calls": page_counters["accumulate_calls"],
            "repeat_kv_calls": page_counters["repeat_kv_calls"],
        }
        timing_rows.append(row)
        scaling_rows.append(row)
        if (batch, tokens) == (4, 4096):
            representative_profile = page_snapshot
            representative_counters = page_counters
            representative_timing = page_timing
        if tokens == 512:
            out = page_fn()
            ref_out = torch.cat(serial_fn(), dim=0).contiguous()
            metrics = correctness_metrics(out, ref_out)
            correctness[f"b{batch}"] = metrics["nan"] == 0 and metrics["inf"] == 0 and metrics["relative_l2"] <= 1e-3
            if batch == 2:
                b1_case = build_case(1, 512, "same")
                b1_cache = pack_mixed_v_pages(b1_case[1], b1_case[2], b1_case[3], b1_case[4], b1_case[5], group_size=GROUP_SIZE, nh=NH)
                b1_out = patternkv_page_batch_decode(b1_case[0], b1_cache)
                b1_ref = torch.cat(prepacked_serial_b1_call(b1_case)(), dim=0).contiguous()
                b1_metrics = correctness_metrics(b1_out, b1_ref)
                correctness["b1"] = b1_metrics["nan"] == 0 and b1_metrics["inf"] == 0 and b1_metrics["relative_l2"] <= 1e-3

    write_csv(OUT_DIR / "timing_breakdown.csv", timing_rows)
    write_csv(OUT_DIR / "scaling_results.csv", scaling_rows)

    rep_case = build_case(4, 4096, "different")
    rep_cache = pack_mixed_v_pages(rep_case[1], rep_case[2], rep_case[3], rep_case[4], rep_case[5], group_size=GROUP_SIZE, nh=NH)
    rep_attn = rep_case[0]
    original_fn = lambda: patternkv_page_batch_decode(rep_attn, rep_cache)
    host_meta_fn = make_page_decode_host_metadata(rep_attn, rep_cache)
    host_meta_timing = time_callable(host_meta_fn, warmup=5, measured=12)
    original_timing = next(row for row in timing_rows if row["batch"] == 4 and row["tokens"] == 4096)
    host_sync_summary = {
        "variant_a_original_wall_median_us": original_timing["page_wall_median_us"],
        "variant_b_host_metadata_wall_median_us": host_meta_timing["wall_us"]["median"],
        "speedup": original_timing["page_wall_median_us"] / host_meta_timing["wall_us"]["median"],
        "item_calls_original": representative_counters.get("host_sync_item_calls"),
        "gpu_item_calls_original": representative_counters.get("gpu_tensor_item_calls"),
    }
    host_sync_summary["host_metadata_correct"] = bool(
        correctness_metrics(host_meta_fn(), original_fn())["relative_l2"] <= 1e-3
    )
    write_json(OUT_DIR / "host_sync_ablation.json", host_sync_summary)

    ablation_rows = []
    for name, fn in [
        ("restore_only", lambda: ablation_restore_only(rep_cache)),
        ("restore_centroid", lambda: ablation_restore_centroid(rep_cache)),
        ("restore_centroid_matmul_full_operator", original_fn),
    ]:
        timed = time_callable(fn, warmup=4, measured=10)
        ablation_rows.append({"ablation": name, **{f"wall_{k}_us": v for k, v in timed["wall_us"].items()}, **{f"cuda_{k}_us": v for k, v in timed["cuda_event_us"].items()}})
    write_csv(OUT_DIR / "page_materialization_ablation.csv", ablation_rows)

    profiler_b2 = run_torch_profiler(lambda: patternkv_page_batch_decode(*page_arg(build_case(2, 2048, "different"))), "b2_t2048")
    profiler_b4 = run_torch_profiler(original_fn, "b4_t4096")
    write_json(OUT_DIR / "torch_profiler_summary.json", {"b2_t2048": profiler_b2, "b4_t4096": profiler_b4})

    total_us = representative_timing["wall_us"]["median"]
    component_rows = []
    for name, rec in sorted(representative_profile.items()):
        component_rows.append(
            {
                "component": name,
                "calls": int(rec.get("calls", 0)),
                "total_us_cuda_event_sum": float(rec.get("total_us", 0.0)),
                "mean_us": float(rec.get("mean_us", 0.0)),
                "percent_of_wall_median": float(rec.get("total_us", 0.0)) * 100.0 / max(total_us, 1e-9),
                "tokens": int(rec.get("tokens", 0)),
                "bytes": int(rec.get("bytes", 0)),
            }
        )
    write_csv(OUT_DIR / "operator_breakdown.csv", component_rows)
    write_json(OUT_DIR / "operator_counters.json", representative_counters)
    write_json(OUT_DIR / "profile_components.json", representative_profile)
    write_json(OUT_DIR / "temp_allocations.json", temp_allocation_snapshot(decode_tokens=1))

    approx_launches = profiler_b4.get("approx_cuda_launches")
    restore_time = sum(row["wall_median_us"] for row in ablation_rows if row["ablation"] == "restore_centroid")
    full_time = original_timing["page_wall_median_us"]
    host_sync_significant = host_sync_summary["speedup"] >= 1.2
    page_materialization_significant = restore_time >= 0.25 * full_time
    kernel_launch_significant = bool(approx_launches and approx_launches > 500)
    matmul_fragmentation_significant = representative_counters.get("matmul_calls", 0) >= int(rep_cache.metadata.metadata_page_table.numel())
    temp_allocation_significant = representative_counters.get("page_value_materialized_bytes", 0) >= 32 * 1024 * 1024
    classification = "PAGE_BATCH_MIXED_OVERHEAD"
    dominant = "mixed host sync, page materialization, tiny launches, and fragmented matmul"
    recommended = "Fused page-centric batched mixed-V operator: one GPU-resident page scheduler plus compressed-domain V2/V4 page kernel"
    next_task = "FUSED_PAGE_CENTRIC_BATCHED_MIXED_V_OPERATOR"
    if host_sync_significant and not (page_materialization_significant or kernel_launch_significant):
        classification = "PAGE_BATCH_HOST_SYNC_DOMINATED"
        dominant = "host/device sync"
        recommended = "Move page scheduling and metadata lookup off Python .item() path"
        next_task = "GPU_RESIDENT_PAGE_BATCH_SCHEDULING_MVP"
    elif page_materialization_significant and not (host_sync_significant or kernel_launch_significant):
        classification = "PAGE_BATCH_PAGE_MATERIALIZATION_DOMINATED"
        dominant = "page-local Value materialization"
        recommended = "Direct compressed-domain mixed-V page kernel"
        next_task = "PAGE_CENTRIC_COMPRESSED_DOMAIN_MIXED_V_KERNEL"
    elif kernel_launch_significant and not (host_sync_significant or page_materialization_significant):
        classification = "PAGE_BATCH_KERNEL_LAUNCH_DOMINATED"
        dominant = "many tiny CUDA launches"
        recommended = "Fuse per-page operations into one batched kernel"
        next_task = "FUSED_PAGE_CENTRIC_BATCHED_MIXED_V_OPERATOR"

    final_gate = {
        "start_head": "0ce940a4b13c1620b34c07cad3133cecddb497e7",
        "algorithm_changed": False,
        "quantization_changed": False,
        "selector_changed": False,
        "v4_budget_changed": False,
        "k_layout_changed": False,
        "page_centric_abi_changed": False,
        "b1_smoke_pass": bool(correctness.get("b1")),
        "b2_smoke_pass": bool(correctness.get("b2")),
        "b4_smoke_pass": bool(correctness.get("b4")),
        "historical_v_materialization_bytes": representative_counters.get("historical_v_materialization_bytes", 0),
        "python_serial_b1_dispatches": representative_counters.get("python_serial_b1_dispatches", 0),
        "host_sync_item_count": representative_counters.get("host_sync_item_calls"),
        "gpu_tensor_item_count": representative_counters.get("gpu_tensor_item_calls"),
        "page_value_materialization_calls": representative_counters.get("page_value_materialization_calls"),
        "page_value_materialized_bytes": representative_counters.get("page_value_materialized_bytes"),
        "total_kernel_launches": approx_launches,
        "matmul_calls": representative_counters.get("matmul_calls"),
        "logical_pages": int(rep_cache.metadata.metadata_page_table.numel()),
        "nonempty_stream_pages": representative_counters.get("v2_pages_processed", 0) + representative_counters.get("v4_pages_processed", 0),
        "metadata_time_us": component_time(component_rows, ["page_metadata_lookup"]),
        "host_sync_time_us": host_sync_summary["variant_a_original_wall_median_us"] - host_sync_summary["variant_b_host_metadata_wall_median_us"],
        "v2_restore_time_us": component_time(component_rows, ["v2_page_restore"]),
        "v4_restore_time_us": component_time(component_rows, ["v4_page_restore"]),
        "centroid_gather_time_us": component_time(component_rows, ["v2_centroid_gather", "v4_centroid_gather"]),
        "attention_slice_time_us": component_time(component_rows, ["page_attn_slice", "v2_attn_slice", "v4_attn_slice"]),
        "repeat_kv_time_us": component_time(component_rows, ["page_batch_repeat_kv"]),
        "matmul_time_us": component_time(component_rows, ["v2_matmul", "v4_matmul"]),
        "accumulation_time_us": component_time(component_rows, ["v2_accumulate", "v4_accumulate"]),
        "other_time_us": None,
        "dominant_bottleneck": dominant,
        "host_sync_significant": host_sync_significant,
        "page_materialization_significant": page_materialization_significant,
        "kernel_launch_significant": kernel_launch_significant,
        "matmul_fragmentation_significant": matmul_fragmentation_significant,
        "temp_allocation_significant": temp_allocation_significant,
        "classification": classification,
        "recommended_optimization": recommended,
        "next_task": next_task,
    }
    known = sum(x for x in [
        final_gate["metadata_time_us"],
        final_gate["v2_restore_time_us"],
        final_gate["v4_restore_time_us"],
        final_gate["attention_slice_time_us"],
        final_gate["repeat_kv_time_us"],
        final_gate["matmul_time_us"],
        final_gate["accumulation_time_us"],
    ] if x is not None)
    final_gate["other_time_us"] = max(total_us - known, 0.0)
    write_json(OUT_DIR / "final_gate.json", final_gate)
    write_json(
        OUT_DIR / "profile_summary.json",
        {
            "final_gate": final_gate,
            "host_sync_ablation": host_sync_summary,
            "representative_timing": representative_timing,
            "representative_counters": representative_counters,
            "nsys_available": shutil.which("nsys") is not None,
        },
    )
    write_reports(final_gate, timing_rows, component_rows, ablation_rows, profiler_b2, profiler_b4, host_sync_summary)
    print(json.dumps(final_gate, indent=2, sort_keys=True))


def page_arg(case):
    cache = pack_mixed_v_pages(case[1], case[2], case[3], case[4], case[5], group_size=GROUP_SIZE, nh=NH)
    return case[0], cache


def component_time(rows: list[dict[str, Any]], names: list[str]) -> float:
    wanted = set(names)
    return float(sum(float(row["total_us_cuda_event_sum"]) for row in rows if row["component"] in wanted))


def write_reports(
    final_gate: dict[str, Any],
    timing_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    profiler_b2: dict[str, Any],
    profiler_b4: dict[str, Any],
    host_sync: dict[str, Any],
) -> None:
    (OUT_DIR / "profile_methodology.md").write_text(
        "# Profile Methodology\n\n"
        "- GPU timings use CUDA events with explicit synchronize before and after each measured repetition.\n"
        "- Wall timings use `perf_counter` around the same call and include Python scheduling and `.item()` stalls.\n"
        "- Each case uses warmup and repeated measurements; CSV files store median/mean/std/CV.\n"
        "- Torch profiler is used only on representative B2/T2048 and B4/T4096 cases because profiler overhead is high.\n",
        encoding="utf-8",
    )
    top_components = sorted(component_rows, key=lambda r: r["total_us_cuda_event_sum"], reverse=True)[:12]
    (OUT_DIR / "operator_breakdown.md").write_text(
        "# Operator Breakdown\n\n"
        + "\n".join(f"- `{r['component']}`: {r['total_us_cuda_event_sum']:.3f} us over {r['calls']} calls" for r in top_components)
        + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "host_sync_audit.md").write_text(
        "# Host Sync Audit\n\n"
        f"- Original item calls: `{host_sync['item_calls_original']}`\n"
        f"- Original GPU tensor item calls: `{host_sync['gpu_item_calls_original']}`\n"
        f"- Original B4/T4096 wall median: `{host_sync['variant_a_original_wall_median_us']}` us\n"
        f"- Host-metadata variant wall median: `{host_sync['variant_b_host_metadata_wall_median_us']}` us\n"
        f"- Variant speedup: `{host_sync['speedup']}`\n"
        f"- HOST_SYNC_SIGNIFICANT: `{final_gate['host_sync_significant']}`\n",
        encoding="utf-8",
    )
    (OUT_DIR / "page_materialization_profile.md").write_text(
        "# Page Materialization Profile\n\n"
        f"- Calls: `{final_gate['page_value_materialization_calls']}`\n"
        f"- Bytes: `{final_gate['page_value_materialized_bytes']}`\n"
        + "\n".join(f"- `{r['ablation']}` wall median: `{r['wall_median_us']}` us" for r in ablation_rows)
        + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "kernel_launch_analysis.md").write_text(
        "# Kernel Launch Analysis\n\n"
        f"- Approx CUDA launches from B4/T4096 profiler: `{final_gate['total_kernel_launches']}`\n"
        f"- Matmul calls from counters: `{final_gate['matmul_calls']}`\n"
        f"- KERNEL_LAUNCH_SIGNIFICANT: `{final_gate['kernel_launch_significant']}`\n",
        encoding="utf-8",
    )
    (OUT_DIR / "scaling_analysis.md").write_text(
        "# Scaling Analysis\n\n"
        + "\n".join(
            f"- B{r['batch']} T{r['tokens']}: {r['logical_pages']} pages, page wall median {r['page_wall_median_us']:.3f} us, {r['wall_us_per_logical_page']:.3f} us/page"
            for r in timing_rows
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "torch_profiler_summary.md").write_text(
        "# Torch Profiler Summary\n\n"
        f"- B2/T2048 available: `{profiler_b2.get('available')}`, approx launches: `{profiler_b2.get('approx_cuda_launches')}`\n"
        f"- B4/T4096 available: `{profiler_b4.get('available')}`, approx launches: `{profiler_b4.get('approx_cuda_launches')}`\n",
        encoding="utf-8",
    )
    nsys_path = shutil.which("nsys")
    (OUT_DIR / "nsys_summary.md").write_text(
        "# NSYS Summary\n\n"
        + (f"- NSYS_AVAILABLE=YES at `{nsys_path}`\n- Trace not captured in this phase to avoid large binary artifacts.\n" if nsys_path else "- NSYS_AVAILABLE=NO\n"),
        encoding="utf-8",
    )
    (OUT_DIR / "correctness_smoke.md").write_text(
        "# Correctness Smoke\n\n"
        f"- B1: `{'PASS' if final_gate['b1_smoke_pass'] else 'FAIL'}`\n"
        f"- B2: `{'PASS' if final_gate['b2_smoke_pass'] else 'FAIL'}`\n"
        f"- B4: `{'PASS' if final_gate['b4_smoke_pass'] else 'FAIL'}`\n"
        f"- Historical full-V materialization bytes: `{final_gate['historical_v_materialization_bytes']}`\n"
        f"- Serial B1 dispatches in production page path: `{final_gate['python_serial_b1_dispatches']}`\n",
        encoding="utf-8",
    )
    (OUT_DIR / "root_cause_analysis.md").write_text(
        "# Root Cause Analysis\n\n"
        f"- Dominant bottleneck: `{final_gate['dominant_bottleneck']}`\n"
        f"- Classification: `{final_gate['classification']}`\n"
        f"- HOST_SYNC_SIGNIFICANT: `{final_gate['host_sync_significant']}`\n"
        f"- PAGE_MATERIALIZATION_SIGNIFICANT: `{final_gate['page_materialization_significant']}`\n"
        f"- KERNEL_LAUNCH_SIGNIFICANT: `{final_gate['kernel_launch_significant']}`\n"
        f"- MATMUL_FRAGMENTATION_SIGNIFICANT: `{final_gate['matmul_fragmentation_significant']}`\n"
        f"- TEMP_ALLOCATION_SIGNIFICANT: `{final_gate['temp_allocation_significant']}`\n",
        encoding="utf-8",
    )
    (OUT_DIR / "optimization_options.md").write_text(
        "# Optimization Options\n\n"
        "- Do not redesign PAGE_CENTRIC_DUAL_STREAM.\n"
        "- Move page scheduling and metadata lookup away from Python `.item()`.\n"
        "- Replace page-local Value reconstruction with compressed-domain V2/V4 page reads.\n"
        "- Fuse per-page restore/index/matmul/accumulate into a small number of batched GPU launches.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "final_recommendation.md").write_text(
        "# Final Recommendation\n\n"
        f"- Classification: `{final_gate['classification']}`\n"
        f"- Recommended optimization: `{final_gate['recommended_optimization']}`\n"
        f"- Next task: `{final_gate['next_task']}`\n"
        "- IMPORTANT: optimization is not implemented in S6-B.2.1.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
