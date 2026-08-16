from __future__ import annotations

from dataclasses import replace

from bench.serving_benchmark_v1 import (
    BenchmarkConfig,
    RunResult,
    filter_valid_runs,
    is_valid_run,
    kv_bytes_for_request,
    max_concurrency_result,
    summarize_runs,
    summarize_tpot_ms,
    workload_hash,
)


def _result(**overrides) -> RunResult:
    base = RunResult(
        method="CAUSAL_V4_25",
        model="DeepSeek-R1-Distill-Llama-8B",
        physical_gpu=2,
        context_length=16384,
        decode_length=128,
        active_capacity=4,
        total_requests=8,
        scheduler_policy="FIFO",
        arrival_protocol="saturated_steady_state",
        workload_hash="abc",
        warmup=False,
        run_index=0,
        completed_requests=8,
        output_tokens=1024,
        wall_time_s=2.0,
        throughput_tokens_s=512.0,
        mean_tpot_ms=8.0,
        median_tpot_ms=8.0,
        p95_tpot_ms=9.0,
        first_token_metric_name="decode_admission_to_first_token_ms",
        mean_first_token_ms=10.0,
        peak_cuda_allocated_bytes=100,
        peak_cuda_reserved_bytes=120,
        kv_pool_bytes=80,
        serial_request_forward_dispatches=0,
        serial_attention_dispatches=0,
        serial_mlp_request_dispatches=0,
        serial_rmsnorm_request_dispatches=0,
        historical_fp16_k_materialization=0,
        historical_fp16_v_materialization=0,
        fallback_count=0,
        true_batch_preserved=True,
        compressed_domain_runtime_preserved=True,
        run_valid=True,
        invalid_reason="",
        scheduler_overhead_s=0.1,
        decode_gpu_wall_s=1.9,
    )
    return replace(base, **overrides)


def test_workload_hash_ignores_method_for_fair_comparison() -> None:
    fp16 = BenchmarkConfig("FP16_KV_RUNTIME", 16384, 128, 4, 32)
    causal = BenchmarkConfig("CAUSAL_V4_25", 16384, 128, 4, 32)
    changed = BenchmarkConfig("CAUSAL_V4_25", 8192, 128, 4, 32)
    assert workload_hash(fp16) == workload_hash(causal)
    assert workload_hash(causal) != workload_hash(changed)


def test_tpot_calculation_reports_ms_per_token() -> None:
    stats = summarize_tpot_ms([1.28, 2.56, 1.92], 128)
    assert stats["mean"] == 15.0
    assert stats["median"] == 15.0
    assert stats["p95"] == 20.0


def test_validity_checks_exclude_serial_or_fallback_runs() -> None:
    valid = _result()
    serial = _result(serial_request_forward_dispatches=1)
    fallback = _result(fallback_count=1)
    incomplete = _result(completed_requests=7)
    assert is_valid_run(valid)
    assert not is_valid_run(serial)
    assert not is_valid_run(fallback)
    assert not is_valid_run(incomplete)
    assert filter_valid_runs([valid, serial, fallback, incomplete]) == [valid]


def test_summary_uses_only_valid_measured_runs() -> None:
    rows = summarize_runs([_result(throughput_tokens_s=512.0), _result(run_valid=False, throughput_tokens_s=9999.0)])
    assert len(rows) == 1
    assert rows[0]["throughput_tokens_s_mean"] == 512.0
    assert rows[0]["runs"] == 1


def test_max_concurrency_records_first_oom() -> None:
    rows = [
        {"method": "CAUSAL_V4_25", "context_length": 16384, "decode_length": 16, "active_capacity": 1, "run_valid": True, "status": "PASS", "peak_cuda_allocated_bytes": 10},
        {"method": "CAUSAL_V4_25", "context_length": 16384, "decode_length": 16, "active_capacity": 2, "run_valid": True, "status": "PASS", "peak_cuda_allocated_bytes": 20},
        {"method": "CAUSAL_V4_25", "context_length": 16384, "decode_length": 16, "active_capacity": 4, "run_valid": False, "status": "OOM", "peak_cuda_allocated_bytes": None},
    ]
    result = max_concurrency_result(rows)
    assert result["max_successful_concurrency"] == 2
    assert result["first_oom_concurrency"] == 4
    assert result["peak_memory_at_max_bytes"] == 20


def test_causal_kv_pool_smaller_than_fp16() -> None:
    causal = kv_bytes_for_request("CAUSAL_V4_25", 16384)
    fp16 = kv_bytes_for_request("FP16_KV_RUNTIME", 16384)
    assert causal < fp16
    assert fp16 / causal > 6.0
