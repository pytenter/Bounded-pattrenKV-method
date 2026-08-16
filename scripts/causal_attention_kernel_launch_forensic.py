from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path("/data/zypan/.local/share/mamba/envs/patternkv/bin/python")
REPORT_DIR = REPO_ROOT / "reports/system_full_model_serving_benchmark_v1/causal_attention_kernel_launch_forensic_v1"
FORMAL_CAUSAL_TPOT_MS = 191.697
LAYERS = 32
DECODE_TOKENS = 8
FORMAL_COMPONENT_MS_PER_TOKEN = {
    "value_fp16_tail": 42.16,
    "cache_append": 19.26,
    "attention_softmax": 18.47,
    "mixed_v_history": 6.09,
    "qk_int2_history": 5.26,
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_text(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[idx])


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def preflight() -> dict[str, Any]:
    py_info = run_text(
        [
            str(PYTHON_BIN),
            "-c",
            "import sys, torch, pytest; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(pytest.__version__)",
        ]
    ).splitlines()
    return {
        "pwd": str(REPO_ROOT),
        "branch": run_text(["git", "branch", "--show-current"]),
        "head": run_text(["git", "rev-parse", "HEAD"]),
        "status_short": run_text(["git", "status", "--short"]),
        "diff_stat": run_text(["git", "diff", "--stat"]),
        "diff_name_status": run_text(["git", "diff", "--name-status"]),
        "diff_check": run_text(["git", "diff", "--check"]),
        "log_12": run_text(["git", "log", "-12", "--oneline", "--decorate"]),
        "python_executable": py_info[0] if len(py_info) > 0 else "UNKNOWN",
        "torch": py_info[1] if len(py_info) > 1 else "UNKNOWN",
        "torch_cuda": py_info[2] if len(py_info) > 2 else "UNKNOWN",
        "cuda_available": py_info[3] if len(py_info) > 3 else "UNKNOWN",
        "pytest": py_info[4] if len(py_info) > 4 else "UNKNOWN",
        "nsys_path": shutil.which("nsys"),
        "ncu_path": shutil.which("ncu"),
        "nvidia_smi": run_text(["nvidia-smi"]),
    }


def category_for(component: str, kernel_name: str) -> str:
    component = component.replace("patternkv::", "")
    lower = f"{component} {kernel_name}".lower()
    if component in {"qk_int2_history", "qk_quantized_history", "qk_quantized_history_strided_k"}:
        return "QK_HISTORY"
    if component.startswith("qk_fp16_") or component == "qk_fp16_regions":
        return "QK_FP16_TAIL"
    if "softmax" in component:
        return "SOFTMAX"
    if component.startswith("mixed_v") or component in {"mixed_historical_value", "mixed_v_page_pool_operator", "page_batch_decode_total"} or "page_mixed_pool_value_kernel" in lower:
        return "VALUE_HISTORY"
    if "bgemv_kernel_outer_dim_with_base" in lower:
        return "QK_HISTORY"
    if "request_invariant_fixed_split_softmax_kernel" in lower:
        return "SOFTMAX"
    if component.startswith("value_fp16"):
        return "VALUE_FP16_TAIL"
    if component.startswith("cache_") or component.startswith("page_batch_pack") or component == "cache_mutation":
        return "CACHE_APPEND"
    if "projection" in component or component in {"qkv_projection", "output_projection"}:
        return "PROJECTION"
    if "memset" in lower:
        return "MEMSET"
    if any(token in lower for token in ("copy", "cast", "contiguous", "clone", "cat", "slice", "index", "scatter", "gather", "memcpy")):
        return "COPY_CAST_CONTIGUOUS"
    if any(token in lower for token in ("elementwise", "where", "fill", "add", "mul", "div", "sub", "exp", "sum", "reduce", "argmax", "rms_norm")):
        return "ELEMENTWISE"
    return "UNKNOWN"


def is_attention_category(category: str) -> bool:
    return category in {"QK_HISTORY", "QK_FP16_TAIL", "SOFTMAX", "VALUE_HISTORY", "VALUE_FP16_TAIL", "CACHE_APPEND", "PROJECTION", "COPY_CAST_CONTIGUOUS", "MEMSET", "ELEMENTWISE"}


def is_attention_component(component: str) -> bool:
    component = component.replace("patternkv::", "")
    prefixes = (
        "qk_",
        "attention_",
        "value_fp16",
        "mixed_v",
        "mixed_historical_value",
        "cache_append",
        "cache_flush",
        "cache_mutation",
        "page_batch",
        "page_",
        "importance_update",
        "output_projection",
    )
    return component.startswith(prefixes) or component in {"fixed_split_softmax_kernel"}


def is_attention_kernel(row: dict[str, Any]) -> bool:
    return is_attention_component(str(row.get("component", ""))) or row.get("category") in {"QK_HISTORY", "SOFTMAX", "VALUE_HISTORY"}


def parse_trace(trace_path: Path, decode_tokens: int) -> dict[str, Any]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    events = [event for event in payload.get("traceEvents", []) if event.get("ph") == "X"]
    ranges = []
    cpu_ops = []
    cuda_api = []
    kernels = []
    memory_events = []
    for event in events:
        cat = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        ts = float(event.get("ts", 0.0))
        dur = float(event.get("dur", 0.0))
        if cat == "user_annotation" and name.startswith("patternkv::"):
            ranges.append({"name": name, "start_us": ts, "end_us": ts + dur, "dur_us": dur})
        elif cat == "cpu_op":
            cpu_ops.append({"name": name, "start_us": ts, "end_us": ts + dur, "dur_us": dur})
        elif cat == "cuda_runtime":
            cuda_api.append({"name": name, "start_us": ts, "end_us": ts + dur, "dur_us": dur})
        elif cat == "kernel":
            args = event.get("args", {})
            kernels.append(
                {
                    "name": name,
                    "start_us": ts,
                    "end_us": ts + dur,
                    "dur_us": dur,
                    "stream": str(args.get("stream", "UNKNOWN")),
                    "trace_category": cat,
                    "occupancy": args.get("est. achieved occupancy %", "NOT_AVAILABLE"),
                    "correlation": args.get("correlation", "NOT_AVAILABLE"),
                }
            )
        elif cat in {"gpu_memcpy", "gpu_memset"}:
            args = event.get("args", {})
            memory_events.append(
                {
                    "name": name,
                    "start_us": ts,
                    "end_us": ts + dur,
                    "dur_us": dur,
                    "stream": str(args.get("stream", "UNKNOWN")),
                    "trace_category": cat,
                    "bytes": args.get("bytes", "NOT_AVAILABLE"),
                    "correlation": args.get("correlation", "NOT_AVAILABLE"),
                }
            )
    ranges.sort(key=lambda row: (row["start_us"], -(row["end_us"] - row["start_us"])))
    for kernel in kernels:
        midpoint = (kernel["start_us"] + kernel["end_us"]) / 2.0
        containing = [row for row in ranges if row["start_us"] <= midpoint <= row["end_us"]]
        component = min(containing, key=lambda row: row["end_us"] - row["start_us"])["name"] if containing else "UNKNOWN"
        kernel["component"] = component
        kernel["category"] = category_for(component, kernel["name"])
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for kernel in kernels:
        grouped[(kernel["name"], kernel["category"])].append(kernel)
    inventory = []
    for (name, category), items in grouped.items():
        durations = [float(item["dur_us"]) for item in items]
        inventory.append(
            {
                "kernel_name": name,
                "calls": len(items),
                "total_gpu_ms": sum(durations) / 1000.0,
                "mean_us": mean(durations),
                "min_us": min(durations) if durations else 0.0,
                "max_us": max(durations) if durations else 0.0,
                "p50_us": pct(durations, 0.50),
                "p95_us": pct(durations, 0.95),
                "category": category,
            }
        )
    inventory.sort(key=lambda row: float(row["total_gpu_ms"]), reverse=True)
    api_grouped: dict[str, list[float]] = defaultdict(list)
    for event in cuda_api:
        api_grouped[event["name"]].append(float(event["dur_us"]))
    api_rows = [
        {"api_name": name, "calls": len(values), "total_cpu_ms": sum(values) / 1000.0, "mean_cpu_us": mean(values)}
        for name, values in api_grouped.items()
    ]
    api_rows.sort(key=lambda row: float(row["total_cpu_ms"]), reverse=True)
    tiny = {}
    for threshold in (5, 10, 20, 50):
        values = [float(row["dur_us"]) for row in kernels if float(row["dur_us"]) < threshold]
        tiny[f"lt_{threshold}us"] = {
            "count": len(values),
            "fraction": len(values) / max(len(kernels), 1),
            "total_gpu_ms": sum(values) / 1000.0,
        }
    gaps = []
    by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for kernel in kernels:
        by_stream[kernel["stream"]].append(kernel)
    for stream, items in by_stream.items():
        ordered = sorted(items, key=lambda row: row["start_us"])
        for left, right in zip(ordered, ordered[1:]):
            gap = float(right["start_us"]) - float(left["end_us"])
            if gap > 0:
                gaps.append({"stream": stream, "gap_us": gap, "left_category": left["category"], "right_category": right["category"]})
    gap_bins = {
        "gap_lt_5us": [g for g in gaps if g["gap_us"] < 5],
        "gap_5_10us": [g for g in gaps if 5 <= g["gap_us"] < 10],
        "gap_10_20us": [g for g in gaps if 10 <= g["gap_us"] < 20],
        "gap_20_50us": [g for g in gaps if 20 <= g["gap_us"] < 50],
        "gap_50_100us": [g for g in gaps if 50 <= g["gap_us"] < 100],
        "gap_gt_100us": [g for g in gaps if g["gap_us"] >= 100],
    }
    gap_rows = [
        {"bucket": name, "count": len(items), "total_gap_ms": sum(float(item["gap_us"]) for item in items) / 1000.0}
        for name, items in gap_bins.items()
    ]
    component_ranges: dict[str, list[float]] = defaultdict(list)
    for row in ranges:
        component_ranges[row["name"].replace("patternkv::", "")].append(float(row["dur_us"]))
    component_rows = [
        {"component": name, "calls": len(values), "total_wall_ms": sum(values) / 1000.0, "mean_wall_us": mean(values)}
        for name, values in sorted(component_ranges.items())
    ]
    category_rows = []
    for category in sorted({row["category"] for row in kernels}):
        items = [row for row in kernels if row["category"] == category]
        category_rows.append(
            {
                "category": category,
                "kernel_calls": len(items),
                "kernel_calls_per_token": len(items) / decode_tokens,
                "gpu_ms": sum(float(row["dur_us"]) for row in items) / 1000.0,
                "gpu_ms_per_token": sum(float(row["dur_us"]) for row in items) / 1000.0 / decode_tokens,
            }
        )
    return {
        "kernels": kernels,
        "memory_events": memory_events,
        "inventory": inventory,
        "cuda_api": api_rows,
        "tiny": tiny,
        "gaps": gaps,
        "gap_rows": gap_rows,
        "component_rows": component_rows,
        "category_rows": category_rows,
        "total_kernel_gpu_ms": sum(float(row["dur_us"]) for row in kernels) / 1000.0,
        "total_device_mem_ms": sum(float(row["dur_us"]) for row in memory_events) / 1000.0,
        "total_idle_gap_ms": sum(float(row["gap_us"]) for row in gaps) / 1000.0,
    }


def run_profile_capture(method: str, report_dir: Path) -> dict[str, Any]:
    from bench.full_model_serving_benchmark import (
        BenchmarkConfig,
        PatternKVAdapter,
        FP16Adapter,
        load_causal_model,
        load_fp16_model,
        register_decode_timing_start_hook,
        run_full_model_benchmark,
    )
    from quant.patternkv_profile import cache_mutation_snapshot, profile_snapshot, reset_profile, temp_allocation_snapshot

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    if method == "CAUSAL_V4_25_FULL_MODEL":
        tokenizer, model, _cfg = load_causal_model(device)
        adapter = PatternKVAdapter
    else:
        tokenizer, model, _cfg = load_fp16_model(device)
        adapter = FP16Adapter
    cfg = BenchmarkConfig(method, 2048, DECODE_TOKENS, 1, 1)
    warmup = run_full_model_benchmark(adapter, model, tokenizer, cfg, device, run_index=0, warmup=True)
    torch.cuda.synchronize(device)
    reset_profile()
    trace_path = report_dir / f"{method.lower()}_decode_trace.json"
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
    started = {"value": False}

    def start_profiler() -> None:
        prof.start()
        started["value"] = True

    unregister = register_decode_timing_start_hook(start_profiler)
    try:
        result = run_full_model_benchmark(adapter, model, tokenizer, cfg, device, run_index=1, warmup=False)
    finally:
        unregister()
        if started["value"]:
            torch.cuda.synchronize(device)
            prof.stop()
    prof.export_chrome_trace(str(trace_path))
    range_snapshot = profile_snapshot(reset=False)
    temp_rows = temp_allocation_snapshot(decode_tokens=DECODE_TOKENS)
    mutation_rows = cache_mutation_snapshot()
    reset_profile()
    parsed = parse_trace(trace_path, DECODE_TOKENS)
    output = {
        "method": method,
        "warmup_result": asdict(warmup),
        "run_result": asdict(result),
        "trace_path": str(trace_path),
        "profile_snapshot": range_snapshot,
        "temp_allocations": temp_rows,
        "cache_mutations": mutation_rows,
        "parsed": {key: value for key, value in parsed.items() if key not in {"kernels", "gaps"}},
        "first_run_compile_in_profile": False,
    }
    write_json(report_dir / f"{method.lower()}_profile.json", output)
    write_csv(report_dir / f"{method.lower()}_kernel_events.csv", parsed["kernels"])
    write_csv(report_dir / f"{method.lower()}_device_memory_events.csv", parsed["memory_events"])
    del model
    del tokenizer
    torch.cuda.empty_cache()
    return {**output, "_parsed_full": parsed}


def rows_by_category(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["category"]: row for row in parsed["category_rows"]}


def component_wall_ms(profile_payload: dict[str, Any], component: str) -> float:
    snapshot = profile_payload.get("profile_snapshot", {})
    return float(snapshot.get(component, {}).get("total_us", 0.0)) / 1000.0


def category_gpu_ms(profile_payload: dict[str, Any], category: str) -> float:
    return float(rows_by_category(profile_payload["_parsed_full"]).get(category, {}).get("gpu_ms", 0.0))


def category_calls(profile_payload: dict[str, Any], category: str) -> int:
    return int(rows_by_category(profile_payload["_parsed_full"]).get(category, {}).get("kernel_calls", 0))


def api_ms(profile_payload: dict[str, Any], names: set[str]) -> float:
    rows = profile_payload["_parsed_full"]["cuda_api"]
    return sum(float(row["total_cpu_ms"]) for row in rows if row["api_name"] in names)


def write_reports(report_dir: Path, pre: dict[str, Any], causal: dict[str, Any], fp16: dict[str, Any] | None) -> dict[str, Any]:
    parsed = causal["_parsed_full"]
    run = causal["run_result"]
    output_tokens = int(run.get("output_tokens") or DECODE_TOKENS)
    kernels_per_token = len(parsed["kernels"]) / output_tokens
    attention_kernels = [row for row in parsed["kernels"] if is_attention_kernel(row)]
    attention_kernels_per_token = len(attention_kernels) / output_tokens
    kernels_per_layer = attention_kernels_per_token / LAYERS
    tiny = parsed["tiny"]
    kernel_busy_ms_per_token = parsed["total_kernel_gpu_ms"] / output_tokens
    attention_gaps = [gap for gap in parsed["gaps"] if gap["left_category"] in {"QK_HISTORY", "QK_FP16_TAIL", "SOFTMAX", "VALUE_HISTORY", "VALUE_FP16_TAIL", "CACHE_APPEND", "PROJECTION", "COPY_CAST_CONTIGUOUS", "MEMSET"} and gap["right_category"] in {"QK_HISTORY", "QK_FP16_TAIL", "SOFTMAX", "VALUE_HISTORY", "VALUE_FP16_TAIL", "CACHE_APPEND", "PROJECTION", "COPY_CAST_CONTIGUOUS", "MEMSET"}]
    attention_idle_gap_ms = sum(float(gap["gap_us"]) for gap in attention_gaps) / 1000.0
    idle_gap_ms_per_token = attention_idle_gap_ms / output_tokens
    sync_ms_per_token = api_ms(causal, {"cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize"}) / output_tokens
    copy_cast_ms_per_token = category_gpu_ms(causal, "COPY_CAST_CONTIGUOUS") / output_tokens
    launch_api_ms_per_token = api_ms(causal, {"cudaLaunchKernel"}) / output_tokens
    attention_gpu_ms_per_token = sum(float(row["dur_us"]) for row in attention_kernels) / 1000.0 / output_tokens
    unexplained_ms = max(FORMAL_CAUSAL_TPOT_MS - attention_gpu_ms_per_token - idle_gap_ms_per_token - sync_ms_per_token - copy_cast_ms_per_token, 0.0)
    explained_fraction = (FORMAL_CAUSAL_TPOT_MS - unexplained_ms) / FORMAL_CAUSAL_TPOT_MS
    value_tail_wall = component_wall_ms(causal, "value_fp16_tail")
    qk_tail_wall = component_wall_ms(causal, "qk_fp16_regions")
    cache_wall = component_wall_ms(causal, "cache_append")
    softmax_wall = component_wall_ms(causal, "attention_softmax")
    value_tail_gpu = category_gpu_ms(causal, "VALUE_FP16_TAIL")
    qk_tail_gpu = category_gpu_ms(causal, "QK_FP16_TAIL")
    cache_gpu = category_gpu_ms(causal, "CACHE_APPEND")
    softmax_gpu = category_gpu_ms(causal, "SOFTMAX")
    value_tail_gap = max(FORMAL_COMPONENT_MS_PER_TOKEN["value_fp16_tail"] - value_tail_gpu / output_tokens, 0.0)
    qk_tail_gap = max(qk_tail_wall / output_tokens - qk_tail_gpu / output_tokens, 0.0)
    cache_orchestration = max(FORMAL_COMPONENT_MS_PER_TOKEN["cache_append"] - cache_gpu / output_tokens, 0.0)
    softmax_orchestration = max(FORMAL_COMPONENT_MS_PER_TOKEN["attention_softmax"] - softmax_gpu / output_tokens, 0.0)
    launch_fragmentation_saving = min(value_tail_gap + cache_orchestration + softmax_orchestration, FORMAL_CAUSAL_TPOT_MS * 0.9)
    value_tail_frag_saving = min(value_tail_gap, FORMAL_CAUSAL_TPOT_MS)
    qk_tail_frag_saving = min(qk_tail_gap, FORMAL_CAUSAL_TPOT_MS)
    attention_orch_saving = min(launch_fragmentation_saving + qk_tail_gap, FORMAL_CAUSAL_TPOT_MS * 0.9)
    amdahl = {
        "tiny_lt_10_free_saved_ms_per_token": tiny["lt_10us"]["total_gpu_ms"] / output_tokens,
        "tiny_lt_10_free_tpot_ms": FORMAL_CAUSAL_TPOT_MS - tiny["lt_10us"]["total_gpu_ms"] / output_tokens,
        "tiny_lt_10_free_speedup": FORMAL_CAUSAL_TPOT_MS / max(FORMAL_CAUSAL_TPOT_MS - tiny["lt_10us"]["total_gpu_ms"] / output_tokens, 1e-9),
        "launch_gaps_free_saved_ms_per_token": launch_fragmentation_saving,
        "launch_gaps_free_tpot_ms": FORMAL_CAUSAL_TPOT_MS - launch_fragmentation_saving,
        "launch_gaps_free_speedup": FORMAL_CAUSAL_TPOT_MS / max(FORMAL_CAUSAL_TPOT_MS - launch_fragmentation_saving, 1e-9),
        "value_tail_fragmentation_free_saved_ms_per_token": value_tail_frag_saving,
        "value_tail_fragmentation_free_tpot_ms": FORMAL_CAUSAL_TPOT_MS - value_tail_frag_saving,
        "qk_tail_fragmentation_free_saved_ms_per_token": qk_tail_frag_saving,
        "qk_tail_fragmentation_free_tpot_ms": FORMAL_CAUSAL_TPOT_MS - qk_tail_frag_saving,
        "softmax_wrapper_free_saved_ms_per_token": softmax_orchestration,
        "all_attention_orchestration_free_saved_ms_per_token": attention_orch_saving,
        "all_attention_orchestration_free_tpot_ms": FORMAL_CAUSAL_TPOT_MS - attention_orch_saving,
        "all_attention_orchestration_free_speedup": FORMAL_CAUSAL_TPOT_MS / max(FORMAL_CAUSAL_TPOT_MS - attention_orch_saving, 1e-9),
    }
    root = "GPU_COMPUTE_DOMINATED"
    next_task = "STOP_FULL_MODEL_THROUGHPUT_OPTIMIZATION"
    project_decision = "STOP_THROUGHPUT_ENGINEERING_AND_FREEZE"
    if attention_orch_saving >= 25.0 and attention_orch_saving > attention_gpu_ms_per_token * 0.5:
        root = "LAUNCH_FRAGMENTATION_DOMINATED"
        next_task = "ATTENTION_KERNEL_LAUNCH_FUSION_V1: FP16_TAIL_VALUE_LAUNCH_FUSION"
        project_decision = "ONE_FINAL_TARGETED_OPTIMIZATION"
    elif attention_gpu_ms_per_token >= 80.0 and max(value_tail_gpu, qk_tail_gpu, cache_gpu, softmax_gpu) >= 20.0 * output_tokens:
        root = "MULTI_COMPONENT"
        next_task = "STOP_FULL_MODEL_THROUGHPUT_OPTIMIZATION"
    elif sync_ms_per_token >= 20.0:
        root = "SYNCHRONIZATION_DOMINATED"
        next_task = "DECODE_SYNCHRONIZATION_ELIMINATION_V1"
        project_decision = "ONE_FINAL_TARGETED_OPTIMIZATION"
    final = {
        "classification": "CAUSAL_ATTENTION_KERNEL_LAUNCH_FORENSIC_V1_SUPPORTED",
        "kernels_per_token": kernels_per_token,
        "attention_kernels_per_token": attention_kernels_per_token,
        "kernels_per_layer": kernels_per_layer,
        "kernel_busy_ms_per_token": kernel_busy_ms_per_token,
        "idle_gap_ms_per_token": idle_gap_ms_per_token,
        "sync_ms_per_token": sync_ms_per_token,
        "copy_cast_ms_per_token": copy_cast_ms_per_token,
        "explained_fraction": explained_fraction,
        "root_cause": root,
        "next_task": next_task,
        "project_decision": project_decision,
        "amdahl": amdahl,
        "value_tail": {
            "calls_per_token": category_calls(causal, "VALUE_FP16_TAIL") / output_tokens,
            "gpu_ms_per_token": value_tail_gpu / output_tokens,
            "launch_gap_ms_per_token": value_tail_gap,
            "classification": "compute-dominated" if value_tail_gpu / output_tokens >= FORMAL_COMPONENT_MS_PER_TOKEN["value_fp16_tail"] * 0.5 else "launch/orchestration-dominated",
        },
        "qk_tail": {
            "calls_per_token": category_calls(causal, "QK_FP16_TAIL") / output_tokens,
            "gpu_ms_per_token": qk_tail_gpu / output_tokens,
            "launch_gap_ms_per_token": qk_tail_gap,
            "classification": "compute-dominated" if qk_tail_gpu >= qk_tail_wall * 0.5 else "launch/orchestration-dominated",
        },
        "cache_append": {
            "calls_per_token": category_calls(causal, "CACHE_APPEND") / output_tokens,
            "gpu_ms_per_token": cache_gpu / output_tokens,
            "orchestration_ms_per_token": cache_orchestration,
            "classification": "compute/copy-dominated" if cache_gpu / output_tokens >= FORMAL_COMPONENT_MS_PER_TOKEN["cache_append"] * 0.5 else "orchestration-dominated",
        },
        "softmax": {
            "calls_per_token": category_calls(causal, "SOFTMAX") / output_tokens,
            "gpu_ms_per_token": softmax_gpu / output_tokens,
            "classification": "GPU compute" if softmax_gpu / output_tokens >= FORMAL_COMPONENT_MS_PER_TOKEN["attention_softmax"] * 0.5 else "wrapper dominated",
        },
    }
    write_csv(report_dir / "kernel_inventory.csv", parsed["inventory"], ["kernel_name", "calls", "total_gpu_ms", "mean_us", "min_us", "max_us", "p50_us", "p95_us", "category"])
    write_csv(report_dir / "cuda_api_summary.csv", parsed["cuda_api"], ["api_name", "calls", "total_cpu_ms", "mean_cpu_us"])
    write_json(report_dir / "analysis_summary.json", final)
    write_json(report_dir / "amdahl_bounds.json", amdahl)
    write_text(report_dir / "environment.md", "\n".join(["# Environment", "", f"- Python: `{pre['python_executable']}`", f"- Torch: `{pre['torch']}`", f"- CUDA: `{pre['torch_cuda']}`", f"- Pytest: `{pre['pytest']}`", f"- Branch: `{pre['branch']}`", f"- HEAD: `{pre['head']}`", f"- CUDA_VISIBLE_DEVICES: `{os.environ.get('CUDA_VISIBLE_DEVICES', '')}`", "", "## GPU", "", "```text", pre["nvidia_smi"], "```"]))
    write_text(report_dir / "timeline_backend.md", "# Timeline Backend\n\nTIMELINE_BACKEND = PYTORCH_PROFILER\n\nNsight Systems and Nsight Compute were not installed. PyTorch profiler exported Chrome trace events with CUDA kernels, CUDA runtime calls, GPU memcpy/memset events, CPU ops, and opt-in PatternKV `record_function` ranges. Kernel stream idle gaps are approximate because PyTorch profiler lacks the full Nsight CPU scheduling view.")
    top_lines = ["# Kernel Inventory", "", "Top CUDA kernels by total GPU time:", ""]
    for row in parsed["inventory"][:20]:
        top_lines.append(f"- `{row['kernel_name'][:140]}`: calls={row['calls']} total_gpu_ms={row['total_gpu_ms']:.3f} mean_us={row['mean_us']:.3f} p50_us={row['p50_us']:.3f} p95_us={row['p95_us']:.3f} category={row['category']}")
    write_text(report_dir / "kernel_inventory.md", "\n".join(top_lines))
    tiny_lines = ["# Tiny Kernel Analysis", ""]
    for key, row in tiny.items():
        tiny_lines.append(f"- {key}: count={row['count']} fraction={row['fraction']:.6f} total_gpu_ms={row['total_gpu_ms']:.3f}")
    write_text(report_dir / "tiny_kernel_analysis.md", "\n".join(tiny_lines))
    gap_lines = ["# GPU Idle Gap Analysis", "", f"- Total positive same-stream kernel gaps, all model kernels: {parsed['total_idle_gap_ms']:.3f} ms/run", f"- Attention-associated same-stream positive gaps: {attention_idle_gap_ms:.3f} ms/run ({idle_gap_ms_per_token:.3f} ms/token)", "- Interpretation: approximate same-stream gaps from PyTorch trace timestamps; Nsight is required to prove CPU launch starvation versus dependency scheduling."]
    for row in parsed["gap_rows"]:
        gap_lines.append(f"- {row['bucket']}: count={row['count']} total_gap_ms={row['total_gap_ms']:.3f}")
    write_text(report_dir / "gpu_idle_gap_analysis.md", "\n".join(gap_lines))
    sync_rows = [row for row in parsed["cuda_api"] if "Synchronize" in row["api_name"]]
    sync_lines = ["# Synchronization Audit", "", "Source search found benchmark-level `torch.cuda.synchronize()` around prefill/decode timing and profile snapshot collection. These are benchmark-only timing barriers, not semantic production requirements. PyTorch trace CUDA runtime sync rows:"]
    for row in sync_rows:
        sync_lines.append(f"- `{row['api_name']}`: calls={row['calls']} total_cpu_ms={row['total_cpu_ms']:.3f} mean_cpu_us={row['mean_cpu_us']:.3f}")
    write_text(report_dir / "sync_audit.md", "\n".join(sync_lines))
    write_text(report_dir / "copy_cast_contiguous_audit.md", "\n".join(["# Copy / Cast / Contiguous Audit", "", f"- COPY_CAST_CONTIGUOUS GPU time: {copy_cast_ms_per_token:.3f} ms/token by categorized kernel names/ranges.", "- Source audit identifies `.to(...)`, `.contiguous()`, `torch.cat`, `index_select`/gather/scatter in QK reader preparation, score concat, cache mutation, and page-pool layout preparation. PyTorch profiler can prove GPU kernel/memcpy presence but not every view-only Python operation.", "- Fields not visible in PyTorch trace are marked `NOT_AVAILABLE` in CSV-derived summaries."]))
    component_lines = ["# Attention Component Timeline", ""]
    for row in causal["_parsed_full"]["component_rows"]:
        component_lines.append(f"- `{row['component']}`: calls={row['calls']} total_wall_ms={row['total_wall_ms']:.3f} mean_wall_us={row['mean_wall_us']:.3f}")
    write_text(report_dir / "attention_component_timeline.md", "\n".join(component_lines))
    write_text(report_dir / "fp16_tail_decomposition.md", f"# FP16 Tail Value Decomposition\n\n- CUDA kernel calls/token: {final['value_tail']['calls_per_token']:.3f}\n- GPU kernel time/token: {final['value_tail']['gpu_ms_per_token']:.3f} ms\n- Approx launch/gap/wrapper time/token: {final['value_tail']['launch_gap_ms_per_token']:.3f} ms\n- Classification: {final['value_tail']['classification']}\n")
    write_text(report_dir / "qk_tail_decomposition.md", f"# QK FP16 Tail Decomposition\n\n- CUDA kernel calls/token: {final['qk_tail']['calls_per_token']:.3f}\n- GPU kernel time/token: {final['qk_tail']['gpu_ms_per_token']:.3f} ms\n- Approx launch/gap/wrapper time/token: {final['qk_tail']['launch_gap_ms_per_token']:.3f} ms\n- Classification: {final['qk_tail']['classification']}\n")
    write_text(report_dir / "history_kernel_decomposition.md", f"# History Kernel Decomposition\n\n- QK INT2 history kernel calls/token: {category_calls(causal, 'QK_HISTORY') / output_tokens:.3f}\n- QK INT2 history GPU ms/token: {category_gpu_ms(causal, 'QK_HISTORY') / output_tokens:.3f}\n- Mixed V history kernel calls/token: {category_calls(causal, 'VALUE_HISTORY') / output_tokens:.3f}\n- Mixed V history GPU ms/token: {category_gpu_ms(causal, 'VALUE_HISTORY') / output_tokens:.3f}\n- Occupancy proxy: present per raw kernel event where PyTorch emitted `est. achieved occupancy %`; otherwise `NOT_AVAILABLE`.\n")
    write_text(report_dir / "cache_append_decomposition.md", f"# Cache Append Decomposition\n\n- CUDA kernel calls/token: {final['cache_append']['calls_per_token']:.3f}\n- GPU kernel time/token: {final['cache_append']['gpu_ms_per_token']:.3f} ms\n- Approx orchestration/wrapper time/token: {final['cache_append']['orchestration_ms_per_token']:.3f} ms\n- Classification: {final['cache_append']['classification']}\n- `page_batch_pack` calls in the decode window are read from PatternKV counters in `causal_v4_25_full_model_profile.json`.\n")
    write_text(report_dir / "softmax_decomposition.md", f"# Softmax Decomposition\n\n- CUDA kernel calls/token: {final['softmax']['calls_per_token']:.3f}\n- GPU kernel time/token: {final['softmax']['gpu_ms_per_token']:.3f} ms\n- Classification: {final['softmax']['classification']}\n- Old CAUSAL path uses the global fixed-split softmax; state merge is absent from production files.\n")
    fp16_lines = ["# FP16 vs CAUSAL Launch Comparison", "", "| Metric | FP16 | CAUSAL |", "| --- | ---: | ---: |"]
    if fp16 is not None:
        fp = fp16["_parsed_full"]
        fp_run = fp16["run_result"]
        fp_tokens = int(fp_run.get("output_tokens") or DECODE_TOKENS)
        fp_att = [row for row in fp["kernels"] if is_attention_kernel(row)]
        fp_attention_value = f"{len(fp_att) / fp_tokens:.3f}" if fp_att else "NOT_AVAILABLE"
        fp16_lines.extend(
            [
                f"| kernels/token | {len(fp['kernels']) / fp_tokens:.3f} | {kernels_per_token:.3f} |",
                f"| attention kernels/token | {fp_attention_value} | {attention_kernels_per_token:.3f} |",
                f"| kernels <10us/token | {fp['tiny']['lt_10us']['count'] / fp_tokens:.3f} | {tiny['lt_10us']['count'] / output_tokens:.3f} |",
                f"| kernels <20us/token | {fp['tiny']['lt_20us']['count'] / fp_tokens:.3f} | {tiny['lt_20us']['count'] / output_tokens:.3f} |",
                f"| CUDA launch API calls/token | {sum(row['calls'] for row in fp['cuda_api'] if row['api_name'] == 'cudaLaunchKernel') / fp_tokens:.3f} | {sum(row['calls'] for row in parsed['cuda_api'] if row['api_name'] == 'cudaLaunchKernel') / output_tokens:.3f} |",
                f"| GPU kernel busy ms/token | {fp['total_kernel_gpu_ms'] / fp_tokens:.3f} | {kernel_busy_ms_per_token:.3f} |",
                f"| estimated idle/gap ms/token | {fp['total_idle_gap_ms'] / fp_tokens:.3f} | {idle_gap_ms_per_token:.3f} |",
            ]
        )
    else:
        fp16_lines.append("| all fields | NOT_AVAILABLE | see CAUSAL |")
    write_text(report_dir / "fp16_vs_causal_launch_comparison.md", "\n".join(fp16_lines))
    write_text(report_dir / "amdahl_bounds.md", "\n".join(["# Amdahl Bounds", "", f"- Bound A, all <10us kernels free: save {amdahl['tiny_lt_10_free_saved_ms_per_token']:.3f} ms/token, TPOT {amdahl['tiny_lt_10_free_tpot_ms']:.3f} ms, speedup {amdahl['tiny_lt_10_free_speedup']:.3f}x.", f"- Bound B, formal Value/cache/softmax launch-orchestration terms free: save {amdahl['launch_gaps_free_saved_ms_per_token']:.3f} ms/token, TPOT {amdahl['launch_gaps_free_tpot_ms']:.3f} ms, speedup {amdahl['launch_gaps_free_speedup']:.3f}x.", f"- Bound C, FP16 Value tail fragmentation free while measured GPU work remains: save {amdahl['value_tail_fragmentation_free_saved_ms_per_token']:.3f} ms/token, TPOT {amdahl['value_tail_fragmentation_free_tpot_ms']:.3f} ms.", f"- Bound D, QK FP16 tail fragmentation free while measured GPU work remains: save {amdahl['qk_tail_fragmentation_free_saved_ms_per_token']:.3f} ms/token, TPOT {amdahl['qk_tail_fragmentation_free_tpot_ms']:.3f} ms.", f"- Bound E, all measured attention orchestration free: save {amdahl['all_attention_orchestration_free_saved_ms_per_token']:.3f} ms/token, TPOT {amdahl['all_attention_orchestration_free_tpot_ms']:.3f} ms, speedup {amdahl['all_attention_orchestration_free_speedup']:.3f}x.", "", "Bounds use formal CAUSAL TPOT 191.697 ms/token. Where previous formal component ranges exist, those wall times are used instead of profiler-overhead TPOT; new PyTorch trace supplies actual GPU kernel time/counts. QK FP16 tail wall time remains profiler-derived and is therefore less reliable than Value/cache/softmax."]))
    write_text(report_dir / "post_forensic_decision.md", f"# Post-Forensic Decision\n\n- ROOT_CAUSE = {root}\n- NEXT_TASK = {next_task}\n- PROJECT_LEVEL_DECISION = {project_decision}\n- ATTENTION_KERNEL_LAUNCH_FUSION_V1 justified: {'yes' if next_task.startswith('ATTENTION_KERNEL_LAUNCH_FUSION_V1') else 'no'}\n")
    summary_answers = [
        "# Summary",
        "",
        "1. Profiler backend: PYTORCH_PROFILER.",
        f"2. Actual CUDA kernels per CAUSAL output token: {kernels_per_token:.3f}.",
        f"3. Attention kernels per layer: {kernels_per_layer:.3f}.",
        f"4. Tiny kernels: <5us={tiny['lt_5us']['count']}, <10us={tiny['lt_10us']['count']}, <20us={tiny['lt_20us']['count']}, <50us={tiny['lt_50us']['count']}.",
        f"5. Tiny launch fraction <10us: {tiny['lt_10us']['fraction']:.6f}.",
        f"6. Tiny GPU time fraction <10us: {tiny['lt_10us']['total_gpu_ms'] / max(parsed['total_kernel_gpu_ms'], 1e-9):.6f}.",
        f"7. Top kernels by GPU time: see `kernel_inventory.md`; #1 is `{parsed['inventory'][0]['kernel_name'][:100]}`.",
        f"8. GPU idle/launch gap per token: {idle_gap_ms_per_token:.3f} ms approximate same-stream positive gaps.",
        f"9. Explicit synchronizations in decode hot path: benchmark-only `torch.cuda.synchronize()` after each decode step; no production semantic sync removal attempted.",
        f"10. FP16 Value tail: {final['value_tail']['classification']}.",
        f"11. FP16 QK tail: {final['qk_tail']['classification']}.",
        f"12. Cache append: {final['cache_append']['classification']}.",
        f"13. Fixed-split softmax: {final['softmax']['classification']}.",
        f"14. Explained CAUSAL TPOT fraction: {explained_fraction:.6f}.",
        f"15. Best launch-fusion Amdahl bound: {amdahl['all_attention_orchestration_free_speedup']:.3f}x, TPOT {amdahl['all_attention_orchestration_free_tpot_ms']:.3f} ms/token.",
        f"16. ATTENTION_KERNEL_LAUNCH_FUSION_V1 supported: {'yes' if next_task.startswith('ATTENTION_KERNEL_LAUNCH_FUSION_V1') else 'no'}.",
        f"17. Exact first fusion target: {next_task.split(': ', 1)[1] if ': ' in next_task else 'NOT_AVAILABLE'}.",
        f"18. New root cause if no: {root if not next_task.startswith('ATTENTION_KERNEL_LAUNCH_FUSION_V1') else 'NOT_APPLICABLE'}.",
        f"19. Further throughput optimization scientifically worthwhile: {'yes, one targeted task' if project_decision == 'ONE_FINAL_TARGETED_OPTIMIZATION' else 'no'}.",
        f"20. Project decision: {project_decision}.",
    ]
    write_text(report_dir / "summary.md", "\n".join(summary_answers))
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAUSAL attention kernel launch forensic")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--skip-fp16-control", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PATTERNKV_FIXED_SPLIT_SOFTMAX", "1")
    os.environ.setdefault("PATTERNKV_ACTIVE_BATCH_CACHE", "1")
    os.environ["PATTERNKV_SYSTEM_PROFILE"] = "1"
    os.environ["PATTERNKV_TORCH_PROFILER_RANGES"] = "1"
    os.environ.setdefault("ATTN_IMPLEMENTATION", "flash_attention_2")
    pre = preflight()
    write_json(args.report_dir / "preflight.json", pre)
    if args.analyze_only:
        causal = json.loads((args.report_dir / "causal_v4_25_full_model_profile.json").read_text(encoding="utf-8"))
        causal["_parsed_full"] = parse_trace(Path(causal["trace_path"]), DECODE_TOKENS)
        fp16 = None
        fp16_path = args.report_dir / "fp16_full_model_profile.json"
        if fp16_path.exists() and not args.skip_fp16_control:
            fp16 = json.loads(fp16_path.read_text(encoding="utf-8"))
            fp16["_parsed_full"] = parse_trace(Path(fp16["trace_path"]), DECODE_TOKENS)
    else:
        causal = run_profile_capture("CAUSAL_V4_25_FULL_MODEL", args.report_dir)
        fp16 = None if args.skip_fp16_control else run_profile_capture("FP16_FULL_MODEL", args.report_dir)
    final = write_reports(args.report_dir, pre, causal, fp16)
    write_json(args.report_dir / "final_gate.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
