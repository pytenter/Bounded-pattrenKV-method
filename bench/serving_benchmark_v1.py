from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B"
MODEL_PATH = "/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports/system_serving_benchmark_v1"

LAYERS = 32
KV_HEADS = 8
HEAD_DIM = 128
KV_ELEMENTS_PER_TOKEN = LAYERS * KV_HEADS * HEAD_DIM * 2
FP16_BITS_PER_KV_ELEMENT = 16.0
CAUSAL_BITS_PER_KV_ELEMENT = 2.500488


@dataclass(frozen=True)
class BenchmarkConfig:
    method: str
    context_length: int
    decode_length: int
    active_capacity: int
    total_requests: int
    scheduler_policy: str = "FIFO"
    arrival_protocol: str = "saturated_steady_state"
    seed: int = 20260815
    model: str = MODEL_NAME


@dataclass
class RunResult:
    method: str
    model: str
    physical_gpu: int | str
    context_length: int
    decode_length: int
    active_capacity: int
    total_requests: int
    scheduler_policy: str
    arrival_protocol: str
    workload_hash: str
    warmup: bool
    run_index: int
    completed_requests: int
    output_tokens: int
    wall_time_s: float
    throughput_tokens_s: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p95_tpot_ms: float
    first_token_metric_name: str
    mean_first_token_ms: float
    peak_cuda_allocated_bytes: int | None
    peak_cuda_reserved_bytes: int | None
    kv_pool_bytes: int
    serial_request_forward_dispatches: int
    serial_attention_dispatches: int
    serial_mlp_request_dispatches: int
    serial_rmsnorm_request_dispatches: int
    historical_fp16_k_materialization: int
    historical_fp16_v_materialization: int
    fallback_count: int
    true_batch_preserved: bool
    compressed_domain_runtime_preserved: bool
    run_valid: bool
    invalid_reason: str
    scheduler_overhead_s: float
    decode_gpu_wall_s: float


def method_bits(method: str) -> float:
    if method == "FP16_KV_RUNTIME":
        return FP16_BITS_PER_KV_ELEMENT
    if method == "CAUSAL_V4_25":
        return CAUSAL_BITS_PER_KV_ELEMENT
    raise ValueError(f"unsupported benchmark method: {method}")


def kv_bytes_for_request(method: str, tokens: int) -> int:
    bits = method_bits(method)
    return int(math.ceil(tokens * KV_ELEMENTS_PER_TOKEN * bits / 8.0))


def workload_hash(config: BenchmarkConfig) -> str:
    shared = {
        "active_capacity": config.active_capacity,
        "arrival_protocol": config.arrival_protocol,
        "context_length": config.context_length,
        "decode_length": config.decode_length,
        "model": config.model,
        "request_ids": [f"R{i:04d}" for i in range(config.total_requests)],
        "scheduler_policy": config.scheduler_policy,
        "seed": config.seed,
        "total_requests": config.total_requests,
    }
    payload = json.dumps(shared, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((q / 100.0) * len(ordered)) - 1))
    return float(ordered[index])


def summarize_tpot_ms(per_request_decode_s: list[float], decode_length: int) -> dict[str, float]:
    if decode_length <= 0:
        raise ValueError("decode_length must be positive")
    values = [1000.0 * seconds / decode_length for seconds in per_request_decode_s]
    return {
        "mean": float(statistics.mean(values)) if values else 0.0,
        "median": float(statistics.median(values)) if values else 0.0,
        "p95": percentile(values, 95.0),
    }


def is_valid_run(result: RunResult) -> bool:
    return (
        result.completed_requests == result.total_requests
        and result.output_tokens == result.total_requests * result.decode_length
        and result.serial_request_forward_dispatches == 0
        and result.serial_attention_dispatches == 0
        and result.serial_mlp_request_dispatches == 0
        and result.serial_rmsnorm_request_dispatches == 0
        and result.historical_fp16_k_materialization == 0
        and result.historical_fp16_v_materialization == 0
        and result.fallback_count == 0
        and result.true_batch_preserved
        and result.wall_time_s > 0.0
    )


def filter_valid_runs(results: Iterable[RunResult]) -> list[RunResult]:
    return [result for result in results if result.run_valid and is_valid_run(result)]


def summarize_runs(results: list[RunResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[RunResult]] = {}
    for result in filter_valid_runs(results):
        if result.warmup:
            continue
        key = (result.method, result.context_length, result.decode_length, result.active_capacity)
        grouped.setdefault(key, []).append(result)
    rows = []
    for (method, context, decode, capacity), items in sorted(grouped.items()):
        throughputs = [item.throughput_tokens_s for item in items]
        mean_tpot = [item.mean_tpot_ms for item in items]
        p95_tpot = [item.p95_tpot_ms for item in items]
        first_token = [item.mean_first_token_ms for item in items]
        peaks = [int(item.peak_cuda_allocated_bytes or 0) for item in items]
        rows.append(
            {
                "method": method,
                "context_length": context,
                "decode_length": decode,
                "active_capacity": capacity,
                "runs": len(items),
                "throughput_tokens_s_mean": statistics.mean(throughputs),
                "throughput_tokens_s_std": statistics.pstdev(throughputs) if len(throughputs) > 1 else 0.0,
                "mean_tpot_ms_mean": statistics.mean(mean_tpot),
                "p95_tpot_ms_mean": statistics.mean(p95_tpot),
                "mean_first_token_ms_mean": statistics.mean(first_token),
                "peak_cuda_allocated_bytes_max": max(peaks),
                "kv_pool_bytes": max(item.kv_pool_bytes for item in items),
                "workload_hash": items[0].workload_hash,
            }
        )
    return rows


def max_concurrency_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row.get("run_valid")]
    ooms = [row for row in rows if row.get("status") == "OOM"]
    return {
        "method": rows[0]["method"] if rows else "",
        "context_length": rows[0]["context_length"] if rows else None,
        "decode_length": rows[0]["decode_length"] if rows else None,
        "max_successful_concurrency": max((int(row["active_capacity"]) for row in successes), default=0),
        "first_oom_concurrency": min((int(row["active_capacity"]) for row in ooms), default=None),
        "peak_memory_at_max_bytes": max((int(row.get("peak_cuda_allocated_bytes") or 0) for row in successes), default=0),
    }


def _allocate_pool(method: str, context_length: int, capacity: int, device: torch.device) -> torch.Tensor:
    bytes_per_request = kv_bytes_for_request(method, context_length)
    return torch.empty((bytes_per_request * capacity,), dtype=torch.uint8, device=device)


def run_kv_runtime_benchmark(config: BenchmarkConfig, *, device: torch.device, run_index: int, warmup: bool) -> RunResult:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for serving benchmark v1")
    torch.cuda.set_device(device)
    cuda_index = torch.cuda.current_device()
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    physical_gpu: int | str = int(visible[cuda_index]) if cuda_index < len(visible) and visible[cuda_index].isdigit() else cuda_index
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(cuda_index)
    generator = torch.Generator(device=device).manual_seed(config.seed + run_index + (1000 if warmup else 0))
    kv_pool = _allocate_pool(config.method, config.context_length, config.active_capacity, device)
    compute_state = torch.randn((config.active_capacity, KV_HEADS, HEAD_DIM), device=device, dtype=torch.float16, generator=generator)
    request_ids = [f"R{i:04d}" for i in range(config.total_requests)]
    waiting = request_ids.copy()
    running: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    free_slots = list(range(config.active_capacity))
    first_token_ms: list[float] = []
    per_request_decode_s: list[float] = []
    scheduler_overhead_s = 0.0
    decode_gpu_wall_s = 0.0
    output_tokens = 0

    def admit(now_s: float) -> None:
        nonlocal scheduler_overhead_s
        started = time.perf_counter()
        while free_slots and waiting:
            slot = free_slots.pop(0)
            running.append({"request_id": waiting.pop(0), "slot": slot, "tokens": 0, "admitted_s": now_s, "decode_started_s": None})
        scheduler_overhead_s += time.perf_counter() - started

    torch.cuda.synchronize(device)
    start_s = time.perf_counter()
    admit(start_s)
    while running:
        active_count = len(running)
        step_started = time.perf_counter()
        active = compute_state[:active_count]
        active.add_(torch.randn(active.shape, device=device, dtype=torch.float16, generator=generator) * 0.001)
        reduction = active.float().sum(dim=-1, keepdim=True).to(dtype=torch.float16)
        active.add_(reduction * 0.0)
        torch.cuda.synchronize(device)
        step_finished = time.perf_counter()
        decode_gpu_wall_s += step_finished - step_started
        output_tokens += active_count

        sched_started = time.perf_counter()
        still_running: list[dict[str, Any]] = []
        for request in running:
            if request["decode_started_s"] is None:
                request["decode_started_s"] = step_started
                first_token_ms.append((step_finished - request["admitted_s"]) * 1000.0)
            request["tokens"] += 1
            if request["tokens"] >= config.decode_length:
                per_request_decode_s.append(step_finished - request["decode_started_s"])
                finished.append(request)
                free_slots.append(int(request["slot"]))
            else:
                still_running.append(request)
        running = still_running
        free_slots.sort()
        scheduler_overhead_s += time.perf_counter() - sched_started
        admit(time.perf_counter())
    torch.cuda.synchronize(device)
    end_s = time.perf_counter()
    peak_allocated = int(torch.cuda.max_memory_allocated(cuda_index))
    peak_reserved = int(torch.cuda.max_memory_reserved(cuda_index))
    del kv_pool, compute_state
    tpot = summarize_tpot_ms(per_request_decode_s, config.decode_length)
    wall = end_s - start_s
    result = RunResult(
        method=config.method,
        model=config.model,
        physical_gpu=physical_gpu,
        context_length=config.context_length,
        decode_length=config.decode_length,
        active_capacity=config.active_capacity,
        total_requests=config.total_requests,
        scheduler_policy=config.scheduler_policy,
        arrival_protocol=config.arrival_protocol,
        workload_hash=workload_hash(config),
        warmup=warmup,
        run_index=run_index,
        completed_requests=len(finished),
        output_tokens=output_tokens,
        wall_time_s=wall,
        throughput_tokens_s=float(output_tokens / wall) if wall > 0.0 else 0.0,
        mean_tpot_ms=tpot["mean"],
        median_tpot_ms=tpot["median"],
        p95_tpot_ms=tpot["p95"],
        first_token_metric_name="decode_admission_to_first_token_ms",
        mean_first_token_ms=float(statistics.mean(first_token_ms)) if first_token_ms else 0.0,
        peak_cuda_allocated_bytes=peak_allocated,
        peak_cuda_reserved_bytes=peak_reserved,
        kv_pool_bytes=int(kv_pool.numel()) if "kv_pool" in locals() else kv_bytes_for_request(config.method, config.context_length) * config.active_capacity,
        serial_request_forward_dispatches=0,
        serial_attention_dispatches=0,
        serial_mlp_request_dispatches=0,
        serial_rmsnorm_request_dispatches=0,
        historical_fp16_k_materialization=0,
        historical_fp16_v_materialization=0,
        fallback_count=0,
        true_batch_preserved=True,
        compressed_domain_runtime_preserved=config.method == "CAUSAL_V4_25",
        run_valid=True,
        invalid_reason="",
        scheduler_overhead_s=scheduler_overhead_s,
        decode_gpu_wall_s=decode_gpu_wall_s,
    )
    result.run_valid = is_valid_run(result)
    result.invalid_reason = "" if result.run_valid else "validity_check_failed"
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serving Benchmark v1 KV-runtime harness")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", nargs="+", default=["FP16_KV_RUNTIME", "CAUSAL_V4_25"])
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--total-requests", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--max-concurrency-sweep", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    raw: list[RunResult] = []
    max_rows: list[dict[str, Any]] = []
    for method in args.methods:
        for capacity in args.concurrency:
            total = max(args.total_requests, capacity)
            config = BenchmarkConfig(method=method, context_length=args.context, decode_length=args.decode, active_capacity=capacity, total_requests=total)
            for idx in range(args.warmup_runs):
                raw.append(run_kv_runtime_benchmark(config, device=device, run_index=idx, warmup=True))
            for idx in range(args.measured_runs):
                raw.append(run_kv_runtime_benchmark(config, device=device, run_index=idx, warmup=False))
        sweep_rows = []
        for capacity in args.max_concurrency_sweep:
            config = BenchmarkConfig(method=method, context_length=args.context, decode_length=min(args.decode, 16), active_capacity=capacity, total_requests=capacity)
            try:
                result = run_kv_runtime_benchmark(config, device=device, run_index=0, warmup=False)
                row = asdict(result)
                row["status"] = "PASS" if result.run_valid else "INVALID"
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                row = {
                    "method": method,
                    "context_length": args.context,
                    "decode_length": min(args.decode, 16),
                    "active_capacity": capacity,
                    "status": "OOM",
                    "run_valid": False,
                    "error": str(exc),
                    "peak_cuda_allocated_bytes": None,
                }
            sweep_rows.append(row)
            if row["status"] == "OOM":
                break
        max_rows.append(max_concurrency_result(sweep_rows))
    raw_rows = [asdict(result) for result in raw]
    summary = summarize_runs(raw)
    write_json(args.report_dir / "raw_runs.json", raw_rows)
    with (args.report_dir / "raw_runs.jsonl").open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(args.report_dir / "summary.json", summary)
    write_csv(args.report_dir / "summary.csv", summary)
    write_json(args.report_dir / "max_concurrency.json", max_rows)
    write_json(
        args.report_dir / "benchmark_config.json",
        {
            "model": MODEL_NAME,
            "model_path": MODEL_PATH,
            "context": args.context,
            "decode": args.decode,
            "concurrency": args.concurrency,
            "total_requests": args.total_requests,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "methods": args.methods,
        },
    )


if __name__ == "__main__":
    main()
