from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from bench.full_model_serving_benchmark import (  # noqa: E402
    ActiveBatchState,
    BenchmarkConfig,
    PatternKVAdapter,
    RequestState,
    build_request_inputs,
    load_causal_model,
    stack_inputs,
)
from quant.patternkv_profile import profile_snapshot, reset_profile  # noqa: E402


REPORT_DIR = REPO_ROOT / "reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def command_output(args: list[str]) -> str:
    return subprocess.run(args, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()


def preflight() -> dict[str, Any]:
    return {
        "branch": command_output(["git", "branch", "--show-current"]),
        "head": command_output(["git", "rev-parse", "HEAD"]),
        "status_short": command_output(["git", "status", "--short"]),
        "diff_stat": command_output(["git", "diff", "--stat"]),
        "diff_check": command_output(["git", "diff", "--check"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "python": os.sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
    }


def physical_gpu() -> int | str:
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if visible and visible[0].isdigit():
        return int(visible[0])
    return torch.cuda.current_device()


def run_causal_diagnostic(
    model: Any,
    tokenizer: Any,
    config: BenchmarkConfig,
    *,
    run_index: int,
    profile: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    old_profile_flag = os.environ.get("PATTERNKV_SYSTEM_PROFILE")
    if profile:
        os.environ["PATTERNKV_SYSTEM_PROFILE"] = "1"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    if profile:
        reset_profile()

    requests = [
        RequestState(f"R{i:04d}", input_ids)
        for i, input_ids in enumerate(build_request_inputs(tokenizer, config.total_requests, config.context_length, device))
    ]
    waiting = deque(requests)
    running: list[RequestState] = []
    finished: list[RequestState] = []
    active_batch = ActiveBatchState()
    per_iter_rows: list[dict[str, Any]] = []
    refill_rows: list[dict[str, Any]] = []
    output_tokens = 0
    decode_cuda_ms_total = 0.0
    measured_refill_ms_total = 0.0
    initial_prefill_ms = 0.0
    refill_prefill_ms_total = 0.0

    def prefill_group(group: list[RequestState], now: float, *, phase: str) -> float:
        if not group:
            return 0.0
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch_input = stack_inputs(group)
        batch_cache, next_tokens = PatternKVAdapter.prefill_active_batch(model, batch_input)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        for request, token in zip(group, next_tokens):
            request.cache = None
            request.next_token = token.view(1)
            request.tokens_generated = 0
            request.admitted_at = now
            running.append(request)
        active_batch.set(running, batch_cache)
        refill_rows.append(
            {
                "run_index": run_index,
                "phase": phase,
                "group_size": len(group),
                "active_after": len(running),
                "prefill_ms": elapsed_ms,
            }
        )
        return elapsed_ms

    def refill(now: float, *, phase: str) -> float:
        free_slots = config.active_capacity - len(running)
        if free_slots <= 0 or not waiting:
            return 0.0
        group = [waiting.popleft() for _ in range(min(free_slots, len(waiting)))]
        return prefill_group(group, now, phase=phase)

    try:
        with torch.inference_mode():
            initial_prefill_ms = refill(time.perf_counter(), phase="initial_prefill")
            if profile:
                reset_profile()
            torch.cuda.synchronize()
            start_wall = time.perf_counter()
            iteration = 0
            while running:
                membership_changed = not active_batch.matches(running)
                batch_tokens = torch.cat([request.next_token for request in running], dim=0)
                if active_batch.matches(running):
                    batch_cache = active_batch.cache
                else:
                    batch_cache = PatternKVAdapter.assemble_batch([request.cache for request in running])
                    active_batch.set(running, batch_cache)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                cpu_start = time.perf_counter()
                start_event.record()
                next_cache, next_tokens = PatternKVAdapter.decode_batch(model, batch_tokens, batch_cache)
                end_event.record()
                torch.cuda.synchronize()
                cpu_end = time.perf_counter()
                decode_cuda_ms = float(start_event.elapsed_time(end_event))
                decode_cpu_ms = (cpu_end - cpu_start) * 1000.0
                decode_cuda_ms_total += decode_cuda_ms
                snapshot = profile_snapshot(reset=False) if profile else {}
                output_tokens += len(running)
                still_running: list[RequestState] = []
                for request, token in zip(running, next_tokens):
                    request.tokens_generated += 1
                    request.next_token = token.view(1)
                    if request.tokens_generated >= config.decode_length:
                        finished.append(request)
                    else:
                        still_running.append(request)
                active_batch.set(still_running, next_cache) if still_running else active_batch.clear()
                running = still_running
                refill_ms = refill(time.perf_counter(), phase="measured_refill")
                measured_refill_ms_total += refill_ms
                refill_prefill_ms_total += refill_ms
                per_iter_rows.append(
                    {
                        "run_index": run_index,
                        "protocol": config.arrival_protocol,
                        "decode": config.decode_length,
                        "iteration": iteration,
                        "active_requests_before_decode": len(batch_tokens),
                        "tokens_generated": len(batch_tokens),
                        "membership_changed_before_decode": membership_changed,
                        "request_finished_this_iteration": len(still_running) < len(batch_tokens),
                        "refill_prefill_ms_after_decode": refill_ms,
                        "decode_cuda_ms": decode_cuda_ms,
                        "decode_cpu_ms": decode_cpu_ms,
                        "iteration_plan_builds": snapshot.get("iteration_plan_builds", {}).get("calls", 0.0),
                        "metadata_rebuilds": snapshot.get("layer_metadata_rebuilds", {}).get("calls", 0.0),
                        "row_slice_calls": snapshot.get("row_slice_real_copy", {}).get("calls", 0.0),
                        "row_slice_bytes": snapshot.get("row_slice_real_copy", {}).get("bytes", 0.0),
                        "qk_quantized_history_calls": snapshot.get("qk_quantized_history", {}).get("calls", 0.0),
                        "mixed_v_fused_attention_calls": snapshot.get("mixed_v_fused_attention", {}).get("calls", 0.0),
                        "mixed_v_v2_compute_calls": snapshot.get("mixed_v_v2_compute", {}).get("calls", 0.0),
                        "mixed_v_v4_compute_calls": snapshot.get("mixed_v_v4_compute", {}).get("calls", 0.0),
                        "fixed_split_wrapper_calls": snapshot.get("attention_softmax", {}).get("calls", 0.0),
                        "fixed_split_cuda_kernel_calls": snapshot.get("fixed_split_softmax_kernel", {}).get("calls", 0.0),
                        "cache_append_calls": snapshot.get("cache_append", {}).get("calls", 0.0),
                        "importance_update_calls": snapshot.get("importance_update", {}).get("calls", 0.0),
                    }
                )
                iteration += 1
            torch.cuda.synchronize()
            end_wall = time.perf_counter()
            final_snapshot = profile_snapshot(reset=True) if profile else {}
    finally:
        if profile:
            if old_profile_flag is None:
                os.environ.pop("PATTERNKV_SYSTEM_PROFILE", None)
            else:
                os.environ["PATTERNKV_SYSTEM_PROFILE"] = old_profile_flag

    wall_ms = (end_wall - start_wall) * 1000.0
    result = {
        "method": config.method,
        "protocol": config.arrival_protocol,
        "context": config.context_length,
        "B": config.active_capacity,
        "decode": config.decode_length,
        "total_requests": config.total_requests,
        "run_index": run_index,
        "profile_enabled": profile,
        "completed_requests": len(finished),
        "output_tokens": output_tokens,
        "wall_ms": wall_ms,
        "ms_per_output_token": wall_ms / max(output_tokens, 1),
        "tok_s": output_tokens / max(wall_ms / 1000.0, 1e-9),
        "decode_cuda_ms_total": decode_cuda_ms_total,
        "decode_cuda_ms_per_iteration": decode_cuda_ms_total / max(iteration, 1),
        "decode_only_ms_per_output_token": decode_cuda_ms_total / max(output_tokens, 1),
        "initial_prefill_ms_excluded_from_wall": initial_prefill_ms,
        "measured_refill_prefill_ms_total": measured_refill_ms_total,
        "measured_refill_ms_per_output_token": measured_refill_ms_total / max(output_tokens, 1),
        "iterations": iteration,
        "peak_allocated_GB": torch.cuda.max_memory_allocated(device) / 1e9,
        "peak_reserved_GB": torch.cuda.max_memory_reserved(device) / 1e9,
        "profile_snapshot": final_snapshot,
    }
    return result, per_iter_rows, refill_rows


def profile_rows(snapshot: dict[str, dict[str, float]], *, full_model_ms: float, attention_ms: float, iterations: int) -> list[dict[str, Any]]:
    rows = []
    attention_keys = {
        "decode_q_projection",
        "decode_k_projection",
        "decode_v_projection",
        "rope_position",
        "cache_append",
        "qk_quantized_history",
        "qk_fp16_regions",
        "attention_score_concat",
        "attention_softmax",
        "fixed_split_softmax_kernel",
        "mixed_v_fused_attention",
        "mixed_v_mapping_prepare",
        "mixed_v_layout_prepare_v2",
        "mixed_v_v2_compute",
        "mixed_v_layout_prepare_v4",
        "mixed_v_v4_compute",
        "mixed_v_output_reduce",
        "mixed_v_page_pool_operator",
        "value_fp16_tail",
        "output_projection",
        "importance_update",
    }
    for key in sorted(attention_keys):
        rec = snapshot.get(key, {})
        total_ms = float(rec.get("total_us", 0.0)) / 1000.0
        rows.append(
            {
                "component": key,
                "ms_total": total_ms,
                "ms_per_iteration": total_ms / max(iterations, 1),
                "percent_attention": total_ms * 100.0 / max(attention_ms, 1e-9),
                "percent_full_model": total_ms * 100.0 / max(full_model_ms, 1e-9),
                "kernel_count_or_calls": int(rec.get("calls", 0.0)),
                "bound_classification": "",
                "effective_bandwidth": "",
                "occupancy": "",
                "main_stall": "",
                "optimization_opportunity": "",
                "confidence": "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.report_dir / "environment.json", preflight())

    os.environ["PATTERNKV_FIXED_SPLIT_SOFTMAX"] = "1"
    os.environ["PATTERNKV_ACTIVE_BATCH_CACHE"] = "1"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"

    device = torch.device("cuda:0")
    tokenizer, model, _model_cfg = load_causal_model(device)
    per_iter_rows: list[dict[str, Any]] = []
    refill_rows: list[dict[str, Any]] = []
    if not args.profile_only:
        matrix_rows: list[dict[str, Any]] = []
        protocols = [
            ("profile_v2_protocol", 1),
            ("scaling_protocol", 2),
        ]
        for protocol_name, total_requests in protocols:
            for decode in (4, 8):
                cfg = BenchmarkConfig(
                    method="CAUSAL_V4_25_FULL_MODEL",
                    context_length=args.context,
                    decode_length=decode,
                    active_capacity=1,
                    total_requests=total_requests,
                    arrival_protocol=protocol_name,
                )
                for warmup_idx in range(args.warmup):
                    run_causal_diagnostic(model, tokenizer, cfg, run_index=warmup_idx, profile=False)
                for run_idx in range(args.runs):
                    result, iter_rows, refill = run_causal_diagnostic(model, tokenizer, cfg, run_index=run_idx, profile=False)
                    matrix_rows.append(result | {"warmup": False})
                    per_iter_rows.extend(iter_rows)
                    refill_rows.extend(refill)
        write_csv(args.report_dir / "minimal_repro_matrix.csv", matrix_rows)
        write_json(args.report_dir / "minimal_repro_matrix.json", matrix_rows)

    canonical_cfg = BenchmarkConfig(
        method="CAUSAL_V4_25_FULL_MODEL",
        context_length=args.context,
        decode_length=8,
        active_capacity=1,
        total_requests=2,
        arrival_protocol="scaling_protocol",
    )
    profiled, prof_iter_rows, prof_refill_rows = run_causal_diagnostic(model, tokenizer, canonical_cfg, run_index=900, profile=True)
    per_iter_rows.extend(prof_iter_rows)
    refill_rows.extend(prof_refill_rows)
    snapshot = profiled.pop("profile_snapshot")
    attention_ms = sum(
        float(snapshot.get(key, {}).get("total_us", 0.0)) / 1000.0
        for key in (
            "decode_q_projection",
            "decode_k_projection",
            "decode_v_projection",
            "rope_position",
            "cache_append",
            "qk_quantized_history",
            "qk_fp16_regions",
            "attention_score_concat",
            "attention_softmax",
            "mixed_v_fused_attention",
            "mixed_v_page_pool_operator",
            "value_fp16_tail",
            "output_projection",
            "importance_update",
        )
    )
    attention_rows = profile_rows(
        snapshot,
        full_model_ms=float(profiled["decode_cuda_ms_total"]),
        attention_ms=attention_ms,
        iterations=int(profiled["iterations"]),
    )

    per_iter_path = "per_iteration_counters_profile.csv" if args.profile_only else "per_iteration_counters.csv"
    refill_path = "refill_events_profile.csv" if args.profile_only else "refill_events.csv"
    write_csv(args.report_dir / per_iter_path, per_iter_rows)
    write_json(args.report_dir / per_iter_path.replace(".csv", ".json"), per_iter_rows)
    write_csv(args.report_dir / refill_path, refill_rows)
    write_json(args.report_dir / refill_path.replace(".csv", ".json"), refill_rows)
    write_json(args.report_dir / "canonical_profile_raw.json", {"result": profiled, "profile": snapshot})
    write_csv(args.report_dir / "attention_component_profile.csv", attention_rows)
    write_json(args.report_dir / "attention_component_profile.json", attention_rows)
    write_json(
        args.report_dir / "diagnostic_summary.json",
        {
            "physical_gpu": physical_gpu(),
            "canonical_profile": profiled,
            "attention_ms_total_profiled": attention_ms,
            "note": "Headline matrix runs have PATTERNKV_SYSTEM_PROFILE disabled; canonical profile run enables CUDA-event profile ranges and is diagnostic only.",
        },
    )


if __name__ == "__main__":
    main()
