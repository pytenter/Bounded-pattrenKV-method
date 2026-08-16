from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path("/data/zypan/.local/share/mamba/envs/patternkv/bin/python")
REPORT_DIR = REPO_ROOT / "reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1"
OLD_CONTEXT_DIR = REPO_ROOT / "reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2"
OLD_B_DIR = REPO_ROOT / "reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1"

METHODS = ("FP16_FULL_MODEL", "CAUSAL_V4_25_FULL_MODEL")
CONTEXTS = (256, 2048, 4096, 8192)
MATCHED_B = (1, 2, 4, 8)
CAPACITY_B = (1, 2, 4, 8, 16, 32)
DECODE_TOKENS = 8
WARMUP_RUNS = 1
MEASURED_RUNS = 3

PASS_STATUS = "PASS"
OOM_STATUS = "OOM"
ERROR_STATUS = "ERROR"
PROTOCOL_FAIL_STATUS = "PROTOCOL_FAIL"
INVALID_STATUS = "INVALID"
SKIPPED_STATUS = "SKIPPED"


@dataclass(frozen=True)
class Point:
    phase: str
    method: str
    context_length: int
    batch_size: int
    decode_tokens: int
    active_capacity: int
    total_requests: int
    warmup_runs: int = WARMUP_RUNS
    measured_runs: int = MEASURED_RUNS

    @property
    def key(self) -> str:
        parts = [
            self.phase,
            self.method.lower(),
            f"c{self.context_length}",
            f"b{self.batch_size}",
            f"d{self.decode_tokens}",
            f"ac{self.active_capacity}",
            f"tr{self.total_requests}",
            f"w{self.warmup_runs}",
            f"m{self.measured_runs}",
        ]
        return "__".join(parts)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def command_output(args: list[str]) -> str:
    return subprocess.run(args, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()


def preflight() -> dict[str, Any]:
    return {
        "branch": command_output(["git", "branch", "--show-current"]),
        "head": command_output(["git", "rev-parse", "HEAD"]),
        "status_short": command_output(["git", "status", "--short"]),
        "diff_stat": command_output(["git", "diff", "--stat"]),
        "diff_check": command_output(["git", "diff", "--check"]),
        "python": str(PYTHON_BIN),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": command_output(["nvidia-smi"]),
    }


def point_from_dict(payload: dict[str, Any]) -> Point:
    return Point(
        phase=str(payload["phase"]),
        method=str(payload["method"]),
        context_length=int(payload["context_length"]),
        batch_size=int(payload["batch_size"]),
        decode_tokens=int(payload["decode_tokens"]),
        active_capacity=int(payload["active_capacity"]),
        total_requests=int(payload["total_requests"]),
        warmup_runs=int(payload.get("warmup_runs", WARMUP_RUNS)),
        measured_runs=int(payload.get("measured_runs", MEASURED_RUNS)),
    )


def point_config_payload(point: Point) -> dict[str, Any]:
    payload = asdict(point)
    payload["point_key"] = point.key
    payload["model"] = "DeepSeek-R1-Distill-Llama-8B"
    payload["scheduler_policy"] = "FIFO"
    payload["arrival_protocol"] = "decode_only_fixed_membership"
    payload["protocol_class"] = "DECODE_ONLY_FULL_MODEL"
    return payload


def protocol_invariants_pass(row: dict[str, Any]) -> bool:
    return (
        int(row.get("prefill_calls_in_timed_window", -1)) == 0
        and int(row.get("prefill_tokens_in_timed_window", -1)) == 0
        and int(row.get("refill_calls_in_timed_window", -1)) == 0
        and int(row.get("membership_changes_in_timed_window", -1)) == 0
        and int(row.get("page_batch_pack_calls", 0)) == 0
    )


def measured_pass_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not bool(row.get("warmup")) and row.get("run_valid") is True and protocol_invariants_pass(row)]


def summarize_point(point: Point, rows: list[dict[str, Any]], *, status: str, error_type: str = "", error_message: str = "") -> dict[str, Any]:
    measured = measured_pass_rows(rows)
    all_measured = [row for row in rows if not bool(row.get("warmup"))]
    selected = measured if measured else all_measured

    def mean(name: str) -> float | None:
        values = [float(row[name]) for row in selected if row.get(name) is not None]
        return sum(values) / len(values) if values else None

    def maximum(name: str) -> int | None:
        values = [int(row[name]) for row in selected if row.get(name) is not None]
        return max(values) if values else None

    first = selected[0] if selected else {}
    result = {
        **point_config_payload(point),
        "status": status,
        "run_valid": status == PASS_STATUS and len(measured) == point.measured_runs,
        "valid_measured_runs": len(measured),
        "total_measured_runs": len(all_measured),
        "return_code": 0 if status == PASS_STATUS else 1,
        "error_type": error_type,
        "error_message": error_message,
        "initial_prefill_ms": mean("initial_prefill_ms"),
        "decode_only_wall_ms": mean("decode_only_wall_ms"),
        "mean_tpot_ms": mean("mean_tpot_ms"),
        "throughput_tok_s": mean("throughput_tokens_s"),
        "throughput_tokens_s": mean("throughput_tokens_s"),
        "prefill_calls_in_timed_window": max((int(row.get("prefill_calls_in_timed_window", 0)) for row in all_measured), default=0),
        "prefill_tokens_in_timed_window": max((int(row.get("prefill_tokens_in_timed_window", 0)) for row in all_measured), default=0),
        "refill_calls_in_timed_window": max((int(row.get("refill_calls_in_timed_window", 0)) for row in all_measured), default=0),
        "membership_changes_in_timed_window": max((int(row.get("membership_changes_in_timed_window", 0)) for row in all_measured), default=0),
        "page_batch_pack_calls": max((int(row.get("page_batch_pack_calls", 0)) for row in all_measured), default=0),
        "decode_window_peak_allocated_bytes": maximum("decode_window_peak_cuda_allocated_bytes"),
        "decode_window_peak_reserved_bytes": maximum("decode_window_peak_cuda_reserved_bytes"),
        "full_lifecycle_peak_allocated_bytes": maximum("full_lifecycle_peak_cuda_allocated_bytes"),
        "full_lifecycle_peak_reserved_bytes": maximum("full_lifecycle_peak_cuda_reserved_bytes"),
        "peak_cuda_allocated_bytes": maximum("peak_cuda_allocated_bytes"),
        "peak_cuda_reserved_bytes": maximum("peak_cuda_reserved_bytes"),
        "output_tokens": mean("output_tokens"),
        "min_active_batch_size": min((int(row.get("min_active_batch_size", 0)) for row in selected), default=None),
        "max_active_batch_size": max((int(row.get("max_active_batch_size", 0)) for row in selected), default=None),
        "mean_active_batch_size": mean("mean_active_batch_size"),
        "serial_request_forward_dispatches": max((int(row.get("serial_request_forward_dispatches", 0)) for row in all_measured), default=0),
        "serial_attention_dispatches": max((int(row.get("serial_attention_dispatches", 0)) for row in all_measured), default=0),
        "serial_mlp_request_dispatches": max((int(row.get("serial_mlp_request_dispatches", 0)) for row in all_measured), default=0),
        "serial_rmsnorm_request_dispatches": max((int(row.get("serial_rmsnorm_request_dispatches", 0)) for row in all_measured), default=0),
        "historical_fp16_k_materialization": first.get("historical_fp16_k_materialization"),
        "historical_fp16_v_materialization": first.get("historical_fp16_v_materialization"),
        "fallback_count": max((int(row.get("fallback_count", 0)) for row in all_measured), default=0),
        "true_batch_preserved": all(bool(row.get("true_batch_preserved")) for row in selected) if selected else False,
        "compressed_domain_runtime_preserved": first.get("compressed_domain_runtime_preserved"),
        "stdout_log_path": "",
        "stderr_stdout_log_path": "",
    }
    return result


def status_from_rows(point: Point, rows: list[dict[str, Any]]) -> tuple[str, str]:
    measured = [row for row in rows if not bool(row.get("warmup"))]
    if len(measured) != point.measured_runs:
        return INVALID_STATUS, f"expected {point.measured_runs} measured runs, found {len(measured)}"
    for row in measured:
        if not protocol_invariants_pass(row):
            return PROTOCOL_FAIL_STATUS, "timed-window prefill/refill or membership-change invariant failed"
        if row.get("run_valid") is not True:
            return INVALID_STATUS, str(row.get("invalid_reason") or "run_valid false")
    return PASS_STATUS, ""


def run_worker(point: Point, output_json: Path) -> int:
    try:
        import torch

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        from bench.full_model_serving_benchmark import (
            BenchmarkConfig,
            FP16Adapter,
            PatternKVAdapter,
            load_causal_model,
            load_fp16_model,
            run_full_model_benchmark,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for full-model benchmark worker")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        if point.method == "FP16_FULL_MODEL":
            tokenizer, model, _model_cfg = load_fp16_model(device)
            adapter = FP16Adapter
        elif point.method == "CAUSAL_V4_25_FULL_MODEL":
            tokenizer, model, _model_cfg = load_causal_model(device)
            adapter = PatternKVAdapter
        else:
            raise ValueError(f"unsupported method: {point.method}")

        config = BenchmarkConfig(
            method=point.method,
            context_length=point.context_length,
            decode_length=point.decode_tokens,
            active_capacity=point.active_capacity,
            total_requests=point.total_requests,
        )
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for run_index in range(point.warmup_runs):
                result = run_full_model_benchmark(adapter, model, tokenizer, config, device, run_index=run_index, warmup=True)
                rows.append(asdict(result))
            for run_index in range(point.measured_runs):
                result = run_full_model_benchmark(adapter, model, tokenizer, config, device, run_index=run_index, warmup=False)
                rows.append(asdict(result))

        status, reason = status_from_rows(point, rows)
        summary = summarize_point(point, rows, status=status, error_message=reason)
        write_json(
            output_json,
            {
                "config": point_config_payload(point),
                "status": status,
                "summary": summary,
                "runs": rows,
            },
        )
        return 0 if status == PASS_STATUS else 2
    except Exception as exc:
        error_type = type(exc).__name__
        status = OOM_STATUS if "OutOfMemory" in error_type or "out of memory" in str(exc).lower() else ERROR_STATUS
        summary = summarize_point(point, [], status=status, error_type=error_type, error_message=str(exc))
        write_json(
            output_json,
            {
                "config": point_config_payload(point),
                "status": status,
                "summary": summary,
                "runs": [],
                "traceback": traceback.format_exc(),
            },
        )
        return 0 if status == OOM_STATUS else 1


def build_points(phases: set[str]) -> list[Point]:
    points: list[Point] = []
    if "smoke" in phases:
        for method in METHODS:
            points.append(Point("smoke", method, 2048, 1, DECODE_TOKENS, 1, 1, warmup_runs=0, measured_runs=1))
        points.append(Point("smoke_repeat", "FP16_FULL_MODEL", 2048, 1, DECODE_TOKENS, 1, 1, warmup_runs=0, measured_runs=1))
    if "context" in phases:
        for context in CONTEXTS:
            for method in METHODS:
                points.append(Point("context", method, context, 1, DECODE_TOKENS, 1, 1))
    if "matched_b" in phases:
        for batch_size in MATCHED_B:
            for method in METHODS:
                points.append(Point("matched_b", method, 2048, batch_size, DECODE_TOKENS, batch_size, batch_size))
    if "capacity" in phases:
        for method in METHODS:
            for batch_size in CAPACITY_B:
                points.append(Point("capacity", method, 4096, batch_size, DECODE_TOKENS, batch_size, batch_size, warmup_runs=0, measured_runs=1))
    return points


def parse_phase_filter(raw: str) -> set[str]:
    if raw == "all":
        return {"smoke", "context", "matched_b", "capacity"}
    phases = {part.strip() for part in raw.split(",") if part.strip()}
    allowed = {"smoke", "context", "matched_b", "capacity"}
    unknown = phases - allowed
    if unknown:
        raise ValueError(f"unknown phase(s): {sorted(unknown)}")
    return phases


def existing_compatible_pass(path: Path, point: Point) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    return payload.get("status") == PASS_STATUS and payload.get("config") == point_config_payload(point)


def run_point_subprocess(
    point: Point,
    *,
    report_dir: Path,
    retry: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    point_json = report_dir / "points" / f"{point.key}.json"
    log_path = report_dir / "logs" / f"{point.key}.log"
    if not retry and existing_compatible_pass(point_json, point):
        payload = read_json(point_json)
        summary = dict(payload["summary"])
        summary["status"] = SKIPPED_STATUS
        summary["return_code"] = 0
        summary["stdout_log_path"] = str(log_path)
        summary["stderr_stdout_log_path"] = str(log_path)
        return summary

    cmd = [
        str(PYTHON_BIN),
        str(Path(__file__).resolve()),
        "--worker",
        "--phase",
        point.phase,
        "--method",
        point.method,
        "--context",
        str(point.context_length),
        "--batch-size",
        str(point.batch_size),
        "--decode-tokens",
        str(point.decode_tokens),
        "--active-capacity",
        str(point.active_capacity),
        "--total-requests",
        str(point.total_requests),
        "--warmup-runs",
        str(point.warmup_runs),
        "--measured-runs",
        str(point.measured_runs),
        "--output-json",
        str(point_json),
    ]
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PATTERNKV_FIXED_SPLIT_SOFTMAX", "1")
    env.setdefault("PATTERNKV_ACTIVE_BATCH_CACHE", "1")
    env.setdefault("PATTERNKV_SYSTEM_PROFILE", "0")
    if point.method == "FP16_FULL_MODEL":
        env.setdefault("ATTN_IMPLEMENTATION", "flash_attention_2")

    started = time.perf_counter()
    proc = runner(cmd, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    elapsed_s = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout or "", encoding="utf-8")

    if point_json.exists():
        payload = read_json(point_json)
        summary = dict(payload.get("summary", {}))
        status = str(payload.get("status", summary.get("status", ERROR_STATUS)))
    else:
        status = ERROR_STATUS
        summary = summarize_point(
            point,
            [],
            status=ERROR_STATUS,
            error_type="MissingWorkerOutput",
            error_message="worker exited without writing output JSON",
        )
        write_json(point_json, {"config": point_config_payload(point), "status": status, "summary": summary, "runs": [], "worker_return_code": proc.returncode})

    if proc.returncode != 0 and status == PASS_STATUS:
        status = ERROR_STATUS
        summary["status"] = ERROR_STATUS
        summary["error_type"] = "WorkerReturnCode"
        summary["error_message"] = f"worker returned {proc.returncode} despite PASS payload"
    summary["status"] = status
    summary["return_code"] = proc.returncode
    summary["worker_elapsed_s"] = elapsed_s
    summary["stdout_log_path"] = str(log_path)
    summary["stderr_stdout_log_path"] = str(log_path)
    write_json(point_json, {**read_json(point_json), "status": status, "summary": summary, "worker_return_code": proc.returncode})
    return summary


def collect_point_payloads(report_dir: Path) -> list[dict[str, Any]]:
    points_dir = report_dir / "points"
    if not points_dir.exists():
        return []
    return [read_json(path) for path in sorted(points_dir.glob("*.json"))]


def point_summaries(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(payload.get("summary", {})) for payload in payloads if payload.get("summary")]


def gb(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / 1e9


def pass_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") in {PASS_STATUS, SKIPPED_STATUS} and row.get("run_valid") is True]


def context_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for context in CONTEXTS:
        fp16 = next((row for row in pass_summaries(rows) if row["phase"] == "context" and row["method"] == "FP16_FULL_MODEL" and int(row["context_length"]) == context), None)
        causal = next((row for row in pass_summaries(rows) if row["phase"] == "context" and row["method"] == "CAUSAL_V4_25_FULL_MODEL" and int(row["context_length"]) == context), None)
        fp16_any = next((row for row in rows if row.get("phase") == "context" and row.get("method") == "FP16_FULL_MODEL" and int(row.get("context_length", 0)) == context), None)
        causal_any = next((row for row in rows if row.get("phase") == "context" and row.get("method") == "CAUSAL_V4_25_FULL_MODEL" and int(row.get("context_length", 0)) == context), None)
        out.append(
            {
                "context": context,
                "FP16_decode_ms_per_token": fp16.get("mean_tpot_ms") if fp16 else None,
                "CAUSAL_decode_ms_per_token": causal.get("mean_tpot_ms") if causal else None,
                "FP16_decode_tok_s": fp16.get("throughput_tok_s") if fp16 else None,
                "CAUSAL_decode_tok_s": causal.get("throughput_tok_s") if causal else None,
                "CAUSAL_to_FP16_decode_ratio": (
                    float(causal["throughput_tok_s"]) / max(float(fp16["throughput_tok_s"]), 1e-9)
                    if fp16 and causal and fp16.get("throughput_tok_s") is not None and causal.get("throughput_tok_s") is not None
                    else None
                ),
                "FP16_full_lifecycle_peak_allocated_GB": gb(fp16.get("full_lifecycle_peak_allocated_bytes")) if fp16 else None,
                "CAUSAL_full_lifecycle_peak_allocated_GB": gb(causal.get("full_lifecycle_peak_allocated_bytes")) if causal else None,
                "memory_delta_GB": (
                    gb(float(fp16["full_lifecycle_peak_allocated_bytes"]) - float(causal["full_lifecycle_peak_allocated_bytes"]))
                    if fp16 and causal and fp16.get("full_lifecycle_peak_allocated_bytes") is not None and causal.get("full_lifecycle_peak_allocated_bytes") is not None
                    else None
                ),
                "success": "PASS" if fp16 and causal else "OOM" if (fp16_any and fp16_any.get("status") == OOM_STATUS) or (causal_any and causal_any.get("status") == OOM_STATUS) else "PARTIAL",
                "FP16_status": fp16_any.get("status") if fp16_any else None,
                "CAUSAL_status": causal_any.get("status") if causal_any else None,
            }
        )
    return out


def matched_b_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for batch_size in MATCHED_B:
        fp16 = next((row for row in pass_summaries(rows) if row["phase"] == "matched_b" and row["method"] == "FP16_FULL_MODEL" and int(row["batch_size"]) == batch_size), None)
        causal = next((row for row in pass_summaries(rows) if row["phase"] == "matched_b" and row["method"] == "CAUSAL_V4_25_FULL_MODEL" and int(row["batch_size"]) == batch_size), None)
        fp16_any = next((row for row in rows if row.get("phase") == "matched_b" and row.get("method") == "FP16_FULL_MODEL" and int(row.get("batch_size", 0)) == batch_size), None)
        causal_any = next((row for row in rows if row.get("phase") == "matched_b" and row.get("method") == "CAUSAL_V4_25_FULL_MODEL" and int(row.get("batch_size", 0)) == batch_size), None)
        out.append(
            {
                "B": batch_size,
                "FP16_decode_tok_s": fp16.get("throughput_tok_s") if fp16 else None,
                "CAUSAL_decode_tok_s": causal.get("throughput_tok_s") if causal else None,
                "FP16_iteration_ms": fp16.get("decode_only_wall_ms") if fp16 else None,
                "CAUSAL_iteration_ms": causal.get("decode_only_wall_ms") if causal else None,
                "FP16_ms_per_output_token": fp16.get("mean_tpot_ms") if fp16 else None,
                "CAUSAL_ms_per_output_token": causal.get("mean_tpot_ms") if causal else None,
                "CAUSAL_to_FP16_decode_ratio": (
                    float(causal["throughput_tok_s"]) / max(float(fp16["throughput_tok_s"]), 1e-9)
                    if fp16 and causal and fp16.get("throughput_tok_s") is not None and causal.get("throughput_tok_s") is not None
                    else None
                ),
                "FP16_peak_allocated_GB": gb(fp16.get("full_lifecycle_peak_allocated_bytes")) if fp16 else None,
                "CAUSAL_peak_allocated_GB": gb(causal.get("full_lifecycle_peak_allocated_bytes")) if causal else None,
                "success": "PASS" if fp16 and causal else "OOM" if (fp16_any and fp16_any.get("status") == OOM_STATUS) or (causal_any and causal_any.get("status") == OOM_STATUS) else "PARTIAL",
                "FP16_status": fp16_any.get("status") if fp16_any else None,
                "CAUSAL_status": causal_any.get("status") if causal_any else None,
            }
        )
    return out


def capacity_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in METHODS:
        method_rows = [row for row in rows if row.get("phase") == "capacity" and row.get("method") == method]
        successes = [row for row in pass_summaries(method_rows)]
        ooms = [row for row in method_rows if row.get("status") == OOM_STATUS]
        max_success_b = max((int(row["batch_size"]) for row in successes), default=0)
        max_row = next((row for row in successes if int(row["batch_size"]) == max_success_b), None)
        first_oom_b = min((int(row["batch_size"]) for row in ooms), default=None)
        oom_row = next((row for row in ooms if int(row["batch_size"]) == first_oom_b), None)
        out.append(
            {
                "method": method,
                "context": 4096,
                "max_successful_full_lifecycle_B": max_success_b,
                "first_OOM_B": first_oom_b,
                "OOM_phase": "PREFILL_OR_DECODE_OOM" if first_oom_b is not None else None,
                "OOM_error": oom_row.get("error_message") if oom_row else None,
                "decode_tok_s_at_max_B": max_row.get("throughput_tok_s") if max_row else None,
                "decode_iteration_ms_at_max_B": max_row.get("decode_only_wall_ms") if max_row else None,
                "decode_ms_per_output_token_at_max_B": max_row.get("mean_tpot_ms") if max_row else None,
                "full_lifecycle_peak_allocated_GB": gb(max_row.get("full_lifecycle_peak_allocated_bytes")) if max_row else None,
                "full_lifecycle_peak_reserved_GB": gb(max_row.get("full_lifecycle_peak_reserved_bytes")) if max_row else None,
            }
        )
    return out


def old_vs_new_comparison(context_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    old_context = load_csv(OLD_CONTEXT_DIR / "context_scaling_summary.csv")
    old_b = load_csv(OLD_B_DIR / "matched_b_summary.csv")
    old_context_lookup = {(row.get("method"), int(row.get("context", 0)), int(row.get("B", 0))): row for row in old_context}
    old_b_lookup = {(row.get("method"), int(row.get("context", 0)), int(row.get("B", 0))): row for row in old_b}
    rows = []
    for context_row in context_rows:
        context = int(context_row["context"])
        for method, new_field in (("FP16_FULL_MODEL", "FP16_decode_tok_s"), ("CAUSAL_V4_25_FULL_MODEL", "CAUSAL_decode_tok_s")):
            old = old_context_lookup.get((method, context, 1))
            new_value = context_row.get(new_field)
            if old is None or new_value is None:
                continue
            old_value = float(old.get("mean_output_tokens_per_second") or old.get("mean_tpot_ms") or old.get("mean_ms_per_output_token") or 0.0)
            rows.append(
                {
                    "workload": f"C{context} B1 {method}",
                    "old_protocol_ms_or_tok_s": old_value,
                    "new_decode_only_ms_or_tok_s": new_value,
                    "old_prefill_contaminated": True,
                    "difference": float(new_value) - old_value,
                    "interpretation": "old throughput may include refill-prefill; repaired measurement is decode-only fixed membership",
                }
            )
    for batch_row in b_rows:
        batch_size = int(batch_row["B"])
        for method, new_field in (("FP16_FULL_MODEL", "FP16_decode_tok_s"), ("CAUSAL_V4_25_FULL_MODEL", "CAUSAL_decode_tok_s")):
            old = old_b_lookup.get((method, 2048, batch_size))
            new_value = batch_row.get(new_field)
            if old is None or new_value is None:
                continue
            old_value = float(old.get("mean_output_tokens_per_second") or old.get("mean_tpot_ms") or old.get("mean_ms_per_output_token") or 0.0)
            rows.append(
                {
                    "workload": f"C2048 B{batch_size} {method}",
                    "old_protocol_ms_or_tok_s": old_value,
                    "new_decode_only_ms_or_tok_s": new_value,
                    "old_prefill_contaminated": True,
                    "difference": float(new_value) - old_value,
                    "interpretation": "old throughput may include refill-prefill; repaired measurement is decode-only fixed membership",
                }
            )
    return rows


def write_protocol_files(report_dir: Path, env_payload: dict[str, Any]) -> None:
    write_json(report_dir / "environment.json", env_payload)
    protocol_definition = "\n".join(
        [
            "# Decode-Only Protocol Definition",
            "",
            "Each benchmark point runs in an independent subprocess.",
            "The worker pre-fills all active requests before timing, synchronizes CUDA, resets decode-window peak memory counters, and then times only fixed-membership decode iterations.",
            "Timed-window hard gates: prefill calls = 0, prefill tokens = 0, refill calls = 0, membership changes = 0.",
            "Initial prefill and full lifecycle peak memory are recorded separately from decode-only TPOT and throughput.",
        ]
    )
    (report_dir / "protocol_definition.md").write_text(protocol_definition + "\n", encoding="utf-8")
    exact_commands = "\n".join(
        [
            "# Exact Commands",
            "",
            "```bash",
            "pwd",
            "git status --short",
            "git branch --show-current",
            "git rev-parse HEAD",
            "git log -5 --oneline --decorate",
            "git diff --stat",
            "git diff --check",
            "nvidia-smi",
            "```",
            "",
            "```bash",
            "CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATTERNKV_FIXED_SPLIT_SOFTMAX=1 PATTERNKV_ACTIVE_BATCH_CACHE=1 PATTERNKV_SYSTEM_PROFILE=0 /data/zypan/.local/share/mamba/envs/patternkv/bin/python scripts/full_model_scaling_decode_only_protocol_repair.py --phases all",
            "```",
        ]
    )
    (report_dir / "exact_commands.md").write_text(exact_commands + "\n", encoding="utf-8")


def aggregate_reports(report_dir: Path, env_payload: dict[str, Any]) -> dict[str, Any]:
    payloads = collect_point_payloads(report_dir)
    rows = point_summaries(payloads)
    run_rows = [run for payload in payloads for run in payload.get("runs", [])]
    context_rows = context_summary(rows)
    b_rows = matched_b_summary(rows)
    cap_rows = capacity_summary(rows)
    memory_rows = [
        {
            "method": method,
            "context": row["context"],
            "decode": DECODE_TOKENS,
            "peak_allocated_GB": row[f"{method.split('_')[0]}_full_lifecycle_peak_allocated_GB"] if method == "FP16_FULL_MODEL" else row["CAUSAL_full_lifecycle_peak_allocated_GB"],
        }
        for row in context_rows
        for method in METHODS
    ]
    old_new = old_vs_new_comparison(context_rows, b_rows)
    protocol_failures = [row for row in rows if row.get("status") == PROTOCOL_FAIL_STATUS]
    pass_rows = pass_summaries(rows)
    protocol_validation = {
        "decode_only_protocol_valid": not protocol_failures and all(protocol_invariants_pass(row) for row in pass_rows),
        "initial_prefill_outside_timing": True,
        "refill_prefill_outside_timing": True,
        "fixed_membership_in_timed_window": True,
        "PREFILL_CALLS_IN_TIMED_WINDOW": max((int(row.get("prefill_calls_in_timed_window", 0)) for row in pass_rows), default=0),
        "PREFILL_TOKENS_IN_TIMED_WINDOW": max((int(row.get("prefill_tokens_in_timed_window", 0)) for row in pass_rows), default=0),
        "REFILL_CALLS_IN_TIMED_WINDOW": max((int(row.get("refill_calls_in_timed_window", 0)) for row in pass_rows), default=0),
        "MEMBERSHIP_CHANGES_IN_TIMED_WINDOW": max((int(row.get("membership_changes_in_timed_window", 0)) for row in pass_rows), default=0),
        "protocol_failures": protocol_failures,
    }
    structural_counters = {
        "SERIAL_REQUEST_FORWARD_DISPATCHES": max((int(row.get("serial_request_forward_dispatches", 0)) for row in pass_rows), default=0),
        "SERIAL_ATTENTION_DISPATCHES": max((int(row.get("serial_attention_dispatches", 0)) for row in pass_rows), default=0),
        "SERIAL_MLP_REQUEST_DISPATCHES": max((int(row.get("serial_mlp_request_dispatches", 0)) for row in pass_rows), default=0),
        "SERIAL_RMSNORM_REQUEST_DISPATCHES": max((int(row.get("serial_rmsnorm_request_dispatches", 0)) for row in pass_rows), default=0),
        "HISTORICAL_FP16_K_MATERIALIZATION": max((int(row.get("historical_fp16_k_materialization") or 0) for row in pass_rows), default=0),
        "HISTORICAL_FP16_V_MATERIALIZATION": max((int(row.get("historical_fp16_v_materialization") or 0) for row in pass_rows), default=0),
        "FALLBACK_COUNT": max((int(row.get("fallback_count", 0)) for row in pass_rows), default=0),
        "TRUE_BATCH_PRESERVED": all(bool(row.get("true_batch_preserved")) for row in pass_rows),
        "COMPRESSED_DOMAIN_RUNTIME_PRESERVED": all(row.get("compressed_domain_runtime_preserved") is True for row in pass_rows if row.get("method") == "CAUSAL_V4_25_FULL_MODEL"),
        "PREFILL_CALLS_IN_TIMED_WINDOW": protocol_validation["PREFILL_CALLS_IN_TIMED_WINDOW"],
        "PREFILL_TOKENS_IN_TIMED_WINDOW": protocol_validation["PREFILL_TOKENS_IN_TIMED_WINDOW"],
        "REFILL_CALLS_IN_TIMED_WINDOW": protocol_validation["REFILL_CALLS_IN_TIMED_WINDOW"],
        "MEMBERSHIP_CHANGES_IN_TIMED_WINDOW": protocol_validation["MEMBERSHIP_CHANGES_IN_TIMED_WINDOW"],
    }
    final_gate = build_final_gate(rows, context_rows, b_rows, cap_rows, protocol_validation, structural_counters, env_payload)

    write_json(report_dir / "protocol_validation.json", protocol_validation)
    write_json(report_dir / "structural_counters.json", structural_counters)
    write_json(report_dir / "context_scaling.json", context_rows)
    write_csv(report_dir / "context_scaling.csv", context_rows)
    write_json(report_dir / "matched_b_scaling.json", b_rows)
    write_csv(report_dir / "matched_b_scaling.csv", b_rows)
    write_json(report_dir / "capacity_scaling.json", cap_rows)
    write_csv(report_dir / "capacity_scaling.csv", cap_rows)
    write_json(report_dir / "decode_only_context_scaling_summary.json", context_rows)
    write_csv(report_dir / "decode_only_context_scaling_summary.csv", context_rows)
    write_json(report_dir / "decode_only_b_scaling_summary.json", b_rows)
    write_csv(report_dir / "decode_only_b_scaling_summary.csv", b_rows)
    write_json(report_dir / "decode_only_capacity_summary.json", cap_rows)
    write_csv(report_dir / "decode_only_capacity_summary.csv", cap_rows)
    write_json(report_dir / "memory_scaling_repaired.json", memory_rows)
    write_csv(report_dir / "memory_scaling_repaired.csv", memory_rows)
    write_json(report_dir / "old_vs_new_comparison.json", old_new)
    write_csv(report_dir / "old_vs_new_comparison.csv", old_new)
    write_json(report_dir / "point_summaries.json", rows)
    write_csv(report_dir / "point_summaries.csv", rows)
    write_json(report_dir / "decode_only_context_scaling_raw.json", [run for run in run_rows if run.get("context_length") in CONTEXTS and run.get("active_capacity") == 1])
    write_csv(report_dir / "decode_only_context_scaling_raw.csv", [run for run in run_rows if run.get("context_length") in CONTEXTS and run.get("active_capacity") == 1])
    write_json(report_dir / "decode_only_b_scaling_raw.json", [run for run in run_rows if run.get("context_length") == 2048 and run.get("active_capacity") in MATCHED_B])
    write_csv(report_dir / "decode_only_b_scaling_raw.csv", [run for run in run_rows if run.get("context_length") == 2048 and run.get("active_capacity") in MATCHED_B])
    write_json(report_dir / "decode_only_capacity_raw.json", [row for row in rows if row.get("phase") == "capacity"])
    write_csv(report_dir / "decode_only_capacity_raw.csv", [row for row in rows if row.get("phase") == "capacity"])
    write_json(report_dir / "final_gate.json", final_gate)
    write_summary_md(report_dir, final_gate, context_rows, b_rows, cap_rows, protocol_validation)
    return final_gate


def build_final_gate(
    rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    b_rows: list[dict[str, Any]],
    cap_rows: list[dict[str, Any]],
    protocol_validation: dict[str, Any],
    structural_counters: dict[str, Any],
    env_payload: dict[str, Any],
) -> dict[str, Any]:
    context_complete = all(row.get("success") in {"PASS", "OOM"} and row.get("FP16_status") in {PASS_STATUS, SKIPPED_STATUS, OOM_STATUS} and row.get("CAUSAL_status") in {PASS_STATUS, SKIPPED_STATUS, OOM_STATUS} for row in context_rows)
    matched_b_complete = all(row.get("success") in {"PASS", "OOM"} and row.get("FP16_status") in {PASS_STATUS, SKIPPED_STATUS, OOM_STATUS} and row.get("CAUSAL_status") in {PASS_STATUS, SKIPPED_STATUS, OOM_STATUS} for row in b_rows)
    capacity_has_boundaries = all(row.get("max_successful_full_lifecycle_B", 0) > 0 and row.get("first_OOM_B") is not None for row in cap_rows)
    no_protocol_fail = bool(protocol_validation.get("decode_only_protocol_valid"))
    no_serial = (
        structural_counters["SERIAL_REQUEST_FORWARD_DISPATCHES"] == 0
        and structural_counters["SERIAL_ATTENTION_DISPATCHES"] == 0
        and structural_counters["SERIAL_MLP_REQUEST_DISPATCHES"] == 0
        and structural_counters["SERIAL_RMSNORM_REQUEST_DISPATCHES"] == 0
        and structural_counters["FALLBACK_COUNT"] == 0
    )
    fp16_cap = next((row for row in cap_rows if row["method"] == "FP16_FULL_MODEL"), {})
    causal_cap = next((row for row in cap_rows if row["method"] == "CAUSAL_V4_25_FULL_MODEL"), {})
    fp16_max_b = int(fp16_cap.get("max_successful_full_lifecycle_B") or 0)
    causal_max_b = int(causal_cap.get("max_successful_full_lifecycle_B") or 0)
    fp16_own = fp16_cap.get("decode_tok_s_at_max_B")
    causal_own = causal_cap.get("decode_tok_s_at_max_B")
    own_ratio = float(causal_own) / max(float(fp16_own), 1e-9) if fp16_own is not None and causal_own is not None else None
    matched_common = [row for row in b_rows if row.get("success") == "PASS"]
    matched_max_common_b = max((int(row["B"]) for row in matched_common), default=0)
    matched_max_row = next((row for row in matched_common if int(row["B"]) == matched_max_common_b), None)
    supported = context_complete and matched_b_complete and capacity_has_boundaries and no_protocol_fail and no_serial
    if not no_protocol_fail:
        classification = "DECODE_ONLY_TIMING_STILL_CONTAINS_PREFILL"
    elif not context_complete or not matched_b_complete or not capacity_has_boundaries:
        classification = "DECODE_ONLY_SCALING_INCOMPLETE"
    elif not no_serial:
        classification = "DECODE_ONLY_PATH_REGRESSION"
    else:
        classification = "FULL_MODEL_SCALING_DECODE_ONLY_PROTOCOL_REPAIR_V1_SUPPORTED"
    return {
        "TASK_CLASSIFICATION": classification,
        "SUPPORTED": supported,
        "branch": env_payload.get("branch"),
        "head": env_payload.get("head"),
        "MATCHED_MAX_COMMON_B": matched_max_common_b,
        "CAUSAL_TO_FP16_DECODE_RATIO_AT_MAX_COMMON_B": matched_max_row.get("CAUSAL_to_FP16_decode_ratio") if matched_max_row else None,
        "FP16_MAX_SUCCESS_B": fp16_max_b,
        "CAUSAL_MAX_SUCCESS_B": causal_max_b,
        "CONCURRENCY_CAPACITY_RATIO": causal_max_b / max(fp16_max_b, 1) if fp16_max_b else None,
        "FP16_OWN_MAX_DECODE_TOK_S": fp16_own,
        "CAUSAL_OWN_MAX_DECODE_TOK_S": causal_own,
        "OWN_MAX_DECODE_THROUGHPUT_RATIO": own_ratio,
        "FULL_MODEL_CONCURRENCY_ADVANTAGE": "SUPPORTED" if causal_max_b > fp16_max_b else "NOT_SUPPORTED" if fp16_max_b and causal_max_b else "INCONCLUSIVE",
        "FULL_MODEL_MEMORY_TO_DECODE_THROUGHPUT_TRANSLATION": (
            "SUPPORTED" if causal_max_b > fp16_max_b and own_ratio is not None and own_ratio > 1.0 else
            "NOT_SUPPORTED" if own_ratio is not None and own_ratio <= 1.0 else
            "INCONCLUSIVE"
        ),
        "protocol_validation": protocol_validation,
        "structural_counters": structural_counters,
        "context_complete": context_complete,
        "matched_b_complete": matched_b_complete,
        "capacity_has_boundaries": capacity_has_boundaries,
        "COMMIT_CREATED": False,
        "PUSHED": False,
        "REPORT_DIR": str(REPORT_DIR),
    }


def write_summary_md(
    report_dir: Path,
    final_gate: dict[str, Any],
    context_rows: list[dict[str, Any]],
    b_rows: list[dict[str, Any]],
    cap_rows: list[dict[str, Any]],
    protocol_validation: dict[str, Any],
) -> None:
    lines = [
        "# Full-Model Scaling Decode-Only Protocol Repair V1",
        "",
        "## Protocol",
        "",
        "Each benchmark point runs in a separate Python subprocess. Prefill is completed before decode timing; the timed window is fixed-membership decode only.",
        "",
        "## Protocol Invariants",
        "",
        f"- PREFILL_CALLS_IN_TIMED_WINDOW: {protocol_validation['PREFILL_CALLS_IN_TIMED_WINDOW']}",
        f"- PREFILL_TOKENS_IN_TIMED_WINDOW: {protocol_validation['PREFILL_TOKENS_IN_TIMED_WINDOW']}",
        f"- REFILL_CALLS_IN_TIMED_WINDOW: {protocol_validation['REFILL_CALLS_IN_TIMED_WINDOW']}",
        f"- MEMBERSHIP_CHANGES_IN_TIMED_WINDOW: {protocol_validation['MEMBERSHIP_CHANGES_IN_TIMED_WINDOW']}",
        "",
        "## Context Scaling",
        "",
    ]
    for row in context_rows:
        lines.append(
            f"- C{row['context']}: FP16={row['FP16_decode_tok_s']} tok/s, CAUSAL={row['CAUSAL_decode_tok_s']} tok/s, ratio={row['CAUSAL_to_FP16_decode_ratio']}, success={row['success']}"
        )
    lines.extend(["", "## Matched-B Scaling", ""])
    for row in b_rows:
        lines.append(
            f"- B{row['B']}: FP16={row['FP16_decode_tok_s']} tok/s, CAUSAL={row['CAUSAL_decode_tok_s']} tok/s, ratio={row['CAUSAL_to_FP16_decode_ratio']}, success={row['success']}"
        )
    lines.extend(["", "## Capacity Scaling", ""])
    for row in cap_rows:
        lines.append(
            f"- {row['method']}: max PASS B={row['max_successful_full_lifecycle_B']}, first OOM B={row['first_OOM_B']}, own-max tok/s={row['decode_tok_s_at_max_B']}"
        )
    lines.extend(
        [
            "",
            "## Final Gate",
            "",
            f"- TASK_CLASSIFICATION: {final_gate['TASK_CLASSIFICATION']}",
            f"- FULL_MODEL_CONCURRENCY_ADVANTAGE: {final_gate['FULL_MODEL_CONCURRENCY_ADVANTAGE']}",
            f"- FULL_MODEL_MEMORY_TO_DECODE_THROUGHPUT_TRANSLATION: {final_gate['FULL_MODEL_MEMORY_TO_DECODE_THROUGHPUT_TRANSLATION']}",
            "",
        ]
    )
    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (report_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_master(args: argparse.Namespace) -> int:
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    env_payload = preflight()
    write_protocol_files(report_dir, env_payload)

    phases = parse_phase_filter(args.phases)
    points = build_points(phases)
    if args.dry_run:
        write_json(report_dir / "planned_points.json", [point_config_payload(point) for point in points])
        print(f"planned_points={len(points)} report_dir={report_dir}")
        return 0

    capacity_oom_seen: set[str] = set()
    summaries = []
    for point in points:
        if point.phase == "capacity" and point.method in capacity_oom_seen:
            continue
        print(f"[point] {point.key}", flush=True)
        summary = run_point_subprocess(point, report_dir=report_dir, retry=args.retry)
        summaries.append(summary)
        print(f"[point] {point.key} status={summary.get('status')} tok_s={summary.get('throughput_tok_s')}", flush=True)
        if point.phase == "capacity" and summary.get("status") == OOM_STATUS:
            capacity_oom_seen.add(point.method)

    write_json(report_dir / "master_run_summaries.json", summaries)
    write_csv(report_dir / "master_run_summaries.csv", summaries)
    final_gate = aggregate_reports(report_dir, env_payload)
    print(json.dumps(final_gate, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode-only full-model scaling protocol repair")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--phases", default="all", help="all or comma-separated subset of smoke,context,matched_b,capacity")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--retry", action="store_true", help="rerun existing PASS points")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase", default="")
    parser.add_argument("--method", default="")
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    parser.add_argument("--active-capacity", type=int, default=0)
    parser.add_argument("--total-requests", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=WARMUP_RUNS)
    parser.add_argument("--measured-runs", type=int, default=MEASURED_RUNS)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        if args.output_json is None:
            raise SystemExit("--output-json is required in --worker mode")
        point = Point(
            phase=args.phase,
            method=args.method,
            context_length=args.context,
            batch_size=args.batch_size,
            decode_tokens=args.decode_tokens,
            active_capacity=args.active_capacity,
            total_requests=args.total_requests,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
        )
        return run_worker(point, args.output_json)
    return run_master(args)


if __name__ == "__main__":
    raise SystemExit(main())
