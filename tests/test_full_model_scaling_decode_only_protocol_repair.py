from __future__ import annotations

import subprocess
from pathlib import Path

from scripts import full_model_scaling_decode_only_protocol_repair as sweep


def _output_json_from_cmd(cmd: list[str]) -> Path:
    return Path(cmd[cmd.index("--output-json") + 1])


def _write_worker_payload(point: sweep.Point, output_json: Path, *, status: str = sweep.PASS_STATUS, row_updates: dict[str, object] | None = None) -> None:
    row = {
        "warmup": False,
        "run_valid": status == sweep.PASS_STATUS,
        "initial_prefill_ms": 10.0,
        "decode_only_wall_ms": 20.0,
        "mean_tpot_ms": 2.5,
        "throughput_tokens_s": 400.0,
        "prefill_calls_in_timed_window": 0,
        "prefill_tokens_in_timed_window": 0,
        "refill_calls_in_timed_window": 0,
        "membership_changes_in_timed_window": 0,
        "decode_window_peak_cuda_allocated_bytes": 100,
        "decode_window_peak_cuda_reserved_bytes": 200,
        "full_lifecycle_peak_cuda_allocated_bytes": 300,
        "full_lifecycle_peak_cuda_reserved_bytes": 400,
        "peak_cuda_allocated_bytes": 300,
        "peak_cuda_reserved_bytes": 400,
        "output_tokens": point.batch_size * point.decode_tokens,
        "min_active_batch_size": point.batch_size,
        "max_active_batch_size": point.batch_size,
        "mean_active_batch_size": float(point.batch_size),
        "serial_request_forward_dispatches": 0,
        "serial_attention_dispatches": 0,
        "serial_mlp_request_dispatches": 0,
        "serial_rmsnorm_request_dispatches": 0,
        "historical_fp16_k_materialization": 0 if point.method == "CAUSAL_V4_25_FULL_MODEL" else None,
        "historical_fp16_v_materialization": 0 if point.method == "CAUSAL_V4_25_FULL_MODEL" else None,
        "fallback_count": 0,
        "true_batch_preserved": True,
        "compressed_domain_runtime_preserved": point.method == "CAUSAL_V4_25_FULL_MODEL",
        "invalid_reason": "" if status == sweep.PASS_STATUS else "simulated",
    }
    if row_updates:
        row.update(row_updates)
    summary = sweep.summarize_point(point, [row], status=status, error_message=row.get("invalid_reason", ""))
    sweep.write_json(output_json, {"config": sweep.point_config_payload(point), "status": status, "summary": summary, "runs": [row]})


def test_master_invokes_one_subprocess_per_point(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    points = [
        sweep.Point("smoke", "FP16_FULL_MODEL", 2048, 1, 8, 1, 1, warmup_runs=0, measured_runs=1),
        sweep.Point("smoke", "CAUSAL_V4_25_FULL_MODEL", 2048, 1, 8, 1, 1, warmup_runs=0, measured_runs=1),
        sweep.Point("smoke_repeat", "FP16_FULL_MODEL", 2048, 1, 8, 1, 1, warmup_runs=0, measured_runs=1),
    ]

    def fake_runner(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        point = sweep.Point(
            phase=cmd[cmd.index("--phase") + 1],
            method=cmd[cmd.index("--method") + 1],
            context_length=int(cmd[cmd.index("--context") + 1]),
            batch_size=int(cmd[cmd.index("--batch-size") + 1]),
            decode_tokens=int(cmd[cmd.index("--decode-tokens") + 1]),
            active_capacity=int(cmd[cmd.index("--active-capacity") + 1]),
            total_requests=int(cmd[cmd.index("--total-requests") + 1]),
            warmup_runs=int(cmd[cmd.index("--warmup-runs") + 1]),
            measured_runs=int(cmd[cmd.index("--measured-runs") + 1]),
        )
        _write_worker_payload(point, _output_json_from_cmd(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    for point in points:
        summary = sweep.run_point_subprocess(point, report_dir=tmp_path, retry=True, runner=fake_runner)
        assert summary["status"] == sweep.PASS_STATUS

    assert len(calls) == 3
    assert all("--worker" in cmd for cmd in calls)
    assert len({_output_json_from_cmd(cmd) for cmd in calls}) == 3


def test_nonzero_child_error_is_recorded_and_next_point_can_continue(tmp_path: Path) -> None:
    first = sweep.Point("smoke", "FP16_FULL_MODEL", 2048, 1, 8, 1, 1, warmup_runs=0, measured_runs=1)
    second = sweep.Point("smoke", "CAUSAL_V4_25_FULL_MODEL", 2048, 1, 8, 1, 1, warmup_runs=0, measured_runs=1)

    def fake_runner(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if cmd[cmd.index("--method") + 1] == "FP16_FULL_MODEL":
            return subprocess.CompletedProcess(cmd, 1, stdout="boom")
        point = second
        _write_worker_payload(point, _output_json_from_cmd(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    first_summary = sweep.run_point_subprocess(first, report_dir=tmp_path, retry=True, runner=fake_runner)
    second_summary = sweep.run_point_subprocess(second, report_dir=tmp_path, retry=True, runner=fake_runner)

    assert first_summary["status"] == sweep.ERROR_STATUS
    assert first_summary["error_type"] == "MissingWorkerOutput"
    assert second_summary["status"] == sweep.PASS_STATUS


def test_worker_oom_payload_is_structured_not_fatal(tmp_path: Path) -> None:
    point = sweep.Point("capacity", "FP16_FULL_MODEL", 4096, 4, 8, 4, 4, warmup_runs=0, measured_runs=1)

    def fake_runner(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        output_json = _output_json_from_cmd(cmd)
        summary = sweep.summarize_point(point, [], status=sweep.OOM_STATUS, error_type="OutOfMemoryError", error_message="CUDA out of memory")
        sweep.write_json(output_json, {"config": sweep.point_config_payload(point), "status": sweep.OOM_STATUS, "summary": summary, "runs": []})
        return subprocess.CompletedProcess(cmd, 0, stdout="oom")

    summary = sweep.run_point_subprocess(point, report_dir=tmp_path, retry=True, runner=fake_runner)

    assert summary["status"] == sweep.OOM_STATUS
    assert summary["error_type"] == "OutOfMemoryError"


def test_resume_skips_existing_compatible_pass(tmp_path: Path) -> None:
    point = sweep.Point("smoke", "FP16_FULL_MODEL", 2048, 1, 8, 1, 1, warmup_runs=0, measured_runs=1)
    output_json = tmp_path / "points" / f"{point.key}.json"
    _write_worker_payload(point, output_json)
    calls = 0

    def fake_runner(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    summary = sweep.run_point_subprocess(point, report_dir=tmp_path, retry=False, runner=fake_runner)

    assert calls == 0
    assert summary["status"] == sweep.SKIPPED_STATUS


def test_protocol_invariant_failure_is_not_pass() -> None:
    point = sweep.Point("matched_b", "CAUSAL_V4_25_FULL_MODEL", 2048, 4, 8, 4, 4, warmup_runs=0, measured_runs=1)
    row = {
        "warmup": False,
        "run_valid": True,
        "prefill_calls_in_timed_window": 1,
        "prefill_tokens_in_timed_window": 0,
        "refill_calls_in_timed_window": 0,
        "membership_changes_in_timed_window": 0,
    }

    status, reason = sweep.status_from_rows(point, [row])

    assert status == sweep.PROTOCOL_FAIL_STATUS
    assert "invariant failed" in reason
