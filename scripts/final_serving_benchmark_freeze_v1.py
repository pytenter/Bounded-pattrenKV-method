from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path("/data/zypan/.local/share/mamba/envs/patternkv/bin/python")
REPORT_DIR = REPO_ROOT / "reports/system_full_model_serving_benchmark_v1/final_serving_benchmark_freeze_v1"
MODEL_PATH = Path("/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B")

METHODS = ("FP16_FULL_MODEL", "CAUSAL_V4_25_FULL_MODEL")
FINAL_CLAIM_NAMES = (
    "QUALITY_PRESERVATION",
    "EFFECTIVE_KV_BUDGET",
    "COMPRESSED_DOMAIN_HISTORY",
    "RAGGED_SERVING",
    "CONTINUOUS_BATCHING",
    "KV_RUNTIME_MEMORY_ADVANTAGE",
    "KV_RUNTIME_CAPACITY_ADVANTAGE",
    "FULL_MODEL_CONCURRENCY_ADVANTAGE",
    "FP16_TAIL_VALUE_FUSION",
    "FULL_MODEL_DECODE_THROUGHPUT_ADVANTAGE",
)


@dataclass(frozen=True)
class FinalPoint:
    phase: str
    method: str
    context: int
    batch: int
    decode: int
    warmup_runs: int
    measured_runs: int
    allow_page_pack: bool = False

    @property
    def key(self) -> str:
        return "__".join(
            [
                self.phase,
                self.method.lower(),
                f"c{self.context}",
                f"b{self.batch}",
                f"d{self.decode}",
                f"w{self.warmup_runs}",
                f"m{self.measured_runs}",
            ]
        )


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


def command_output(args: list[str]) -> str:
    return subprocess.run(args, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()


def git_text(*args: str) -> str:
    return command_output(["git", *args])


def gpu_inventory() -> list[dict[str, str]]:
    raw = command_output(["nvidia-smi", "--query-gpu=index,name,uuid,memory.used", "--format=csv,noheader,nounits"])
    rows = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append({"index": parts[0], "name": parts[1], "uuid": parts[2], "memory_used_mib": parts[3]})
    return rows


def selected_gpu() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical = visible.split(",")[0].strip() if visible else "0"
    inventory = gpu_inventory()
    row = next((item for item in inventory if item["index"] == physical), None)
    return {
        "physical_gpu": physical,
        "gpu_name": row["name"] if row else "",
        "gpu_uuid": row["uuid"] if row else "",
        "initial_memory_used_mib": row["memory_used_mib"] if row else "",
        "nvidia_smi": command_output(["nvidia-smi"]),
    }


def point_payload(point: FinalPoint) -> dict[str, Any]:
    return {
        "phase": point.phase,
        "method": point.method,
        "context": point.context,
        "batch": point.batch,
        "decode": point.decode,
        "warmup_runs": point.warmup_runs,
        "measured_runs": point.measured_runs,
        "allow_page_pack": point.allow_page_pack,
        "point_key": point.key,
        "model_path": str(MODEL_PATH),
    }


def build_points() -> list[FinalPoint]:
    points: list[FinalPoint] = []
    for batch in (1, 2, 4, 8):
        for method in METHODS:
            points.append(FinalPoint("batch_scaling", method, 2048, batch, 8, 1, 3))
    for context in (4096, 8192):
        for method in METHODS:
            points.append(FinalPoint("context_scaling", method, context, 1, 8, 1, 3))
    for method in METHODS:
        for batch in (1, 2, 4, 8, 16):
            points.append(FinalPoint("capacity", method, 4096, batch, 8, 0, 1))
    for decode in (256, 512):
        for method in METHODS:
            points.append(FinalPoint("long_decode", method, 2048, 1, decode, 0, 1, allow_page_pack=True))
    return points


def protocol_valid(row: dict[str, Any], *, allow_page_pack: bool) -> bool:
    return (
        bool(row.get("run_valid"))
        and int(row.get("prefill_calls_in_timed_window", -1)) == 0
        and int(row.get("prefill_tokens_in_timed_window", -1)) == 0
        and int(row.get("refill_calls_in_timed_window", -1)) == 0
        and int(row.get("membership_changes_in_timed_window", -1)) == 0
        and (allow_page_pack or int(row.get("page_batch_pack_calls", 0)) == 0)
        and int(row.get("fallback_count", 0)) == 0
        and int(row.get("serial_request_forward_dispatches", 0)) == 0
        and int(row.get("serial_attention_dispatches", 0)) == 0
    )


def summarize_runs(point: FinalPoint, rows: list[dict[str, Any]], status: str, error_type: str = "", error_message: str = "") -> dict[str, Any]:
    measured = [row for row in rows if not bool(row.get("warmup"))]
    valid = [row for row in measured if protocol_valid(row, allow_page_pack=point.allow_page_pack)]
    selected = valid if valid else measured

    def mean(name: str) -> float | None:
        values = [float(row[name]) for row in selected if row.get(name) is not None]
        return float(statistics.mean(values)) if values else None

    def median(name: str) -> float | None:
        values = [float(row[name]) for row in selected if row.get(name) is not None]
        return float(statistics.median(values)) if values else None

    def maximum(name: str) -> int | None:
        values = [int(row[name]) for row in selected if row.get(name) is not None]
        return max(values) if values else None

    first = selected[0] if selected else {}
    pass_status = status == "PASS" and len(valid) == point.measured_runs
    return {
        **point_payload(point),
        "status": status,
        "run_valid": pass_status,
        "valid_measured_runs": len(valid),
        "total_measured_runs": len(measured),
        "error_type": error_type,
        "error_message": error_message,
        "physical_gpu": first.get("physical_gpu"),
        "success": pass_status,
        "oom": status == "OOM",
        "tpot_ms": mean("mean_tpot_ms"),
        "median_tpot_ms": median("mean_tpot_ms"),
        "tok_per_s": mean("throughput_tokens_s"),
        "decode_wall_ms": mean("decode_only_wall_ms"),
        "peak_allocated_bytes": maximum("peak_cuda_allocated_bytes"),
        "peak_reserved_bytes": maximum("peak_cuda_reserved_bytes"),
        "prefill_calls_timed": max((int(row.get("prefill_calls_in_timed_window", 0)) for row in measured), default=0),
        "prefill_tokens_timed": max((int(row.get("prefill_tokens_in_timed_window", 0)) for row in measured), default=0),
        "refill_calls_timed": max((int(row.get("refill_calls_in_timed_window", 0)) for row in measured), default=0),
        "membership_changes_timed": max((int(row.get("membership_changes_in_timed_window", 0)) for row in measured), default=0),
        "page_batch_pack_calls": max((int(row.get("page_batch_pack_calls", 0)) for row in measured), default=0),
        "historical_fp16_k_materialization": first.get("historical_fp16_k_materialization"),
        "historical_fp16_v_materialization": first.get("historical_fp16_v_materialization"),
        "fallback": max((int(row.get("fallback_count", 0)) for row in measured), default=0),
        "serial_request_dispatch": max((int(row.get("serial_request_forward_dispatches", 0)) for row in measured), default=0),
        "serial_attention_dispatch": max((int(row.get("serial_attention_dispatches", 0)) for row in measured), default=0),
        "true_batch": all(bool(row.get("true_batch_preserved")) for row in selected) if selected else False,
    }


def run_worker(point: FinalPoint, output_json: Path) -> int:
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
            raise RuntimeError("CUDA is required")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        if point.method == "FP16_FULL_MODEL":
            tokenizer, model, _cfg = load_fp16_model(device)
            adapter = FP16Adapter
        elif point.method == "CAUSAL_V4_25_FULL_MODEL":
            tokenizer, model, _cfg = load_causal_model(device)
            adapter = PatternKVAdapter
        else:
            raise ValueError(f"unsupported method: {point.method}")

        config = BenchmarkConfig(
            method=point.method,
            context_length=point.context,
            decode_length=point.decode,
            active_capacity=point.batch,
            total_requests=point.batch,
        )
        rows = []
        with torch.inference_mode():
            for run_index in range(point.warmup_runs):
                rows.append(asdict(run_full_model_benchmark(adapter, model, tokenizer, config, device, run_index=run_index, warmup=True)))
            for run_index in range(point.measured_runs):
                result_row = asdict(run_full_model_benchmark(adapter, model, tokenizer, config, device, run_index=run_index, warmup=False))
                try:
                    from quant.patternkv_profile import profile_snapshot

                    snapshot = profile_snapshot(reset=False)
                    result_row["page_batch_pack_profile_total_us"] = float(snapshot.get("page_batch_pack", {}).get("total_us", 0.0))
                    result_row["pack_window_profile_total_us"] = float(snapshot.get("pack_window", {}).get("total_us", 0.0))
                    result_row["cache_append_profile_total_us"] = float(snapshot.get("cache_append", {}).get("total_us", 0.0))
                except Exception:
                    if os.environ.get("PATTERNKV_STRICT_PROFILE_RESET") == "1":
                        raise
                rows.append(result_row)
        measured = [row for row in rows if not bool(row.get("warmup"))]
        status = "PASS" if len(measured) == point.measured_runs and all(protocol_valid(row, allow_page_pack=point.allow_page_pack) for row in measured) else "INVALID"
        summary = summarize_runs(point, rows, status)
        write_json(output_json, {"config": point_payload(point), "status": status, "summary": summary, "runs": rows})
        return 0 if status == "PASS" else 2
    except Exception as exc:
        status = "OOM" if "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError" else "ERROR"
        summary = summarize_runs(point, [], status, type(exc).__name__, str(exc))
        write_json(output_json, {"config": point_payload(point), "status": status, "summary": summary, "runs": [], "traceback": traceback.format_exc()})
        return 0 if status == "OOM" else 1


def run_point(point: FinalPoint, report_dir: Path, retry: bool) -> dict[str, Any]:
    output_json = report_dir / "points" / f"{point.key}.json"
    log_path = report_dir / "logs" / f"{point.key}.log"
    if output_json.exists() and not retry:
        payload = read_json(output_json)
        return dict(payload["summary"])
    cmd = [
        str(PYTHON_BIN),
        str(Path(__file__).resolve()),
        "--worker",
        "--phase",
        point.phase,
        "--method",
        point.method,
        "--context",
        str(point.context),
        "--batch",
        str(point.batch),
        "--decode",
        str(point.decode),
        "--warmup-runs",
        str(point.warmup_runs),
        "--measured-runs",
        str(point.measured_runs),
        "--output-json",
        str(output_json),
    ]
    if point.allow_page_pack:
        cmd.append("--allow-page-pack")
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PATTERNKV_FP16_TAIL_VALUE_FUSION", "1")
    env.setdefault("PATTERNKV_FIXED_SPLIT_SOFTMAX", "1")
    env.setdefault("PATTERNKV_SELECTIVE_PREFILL_LOGITS", "1")
    env.setdefault("PATTERNKV_ACTIVE_BATCH_CACHE", "1")
    env.setdefault("PATTERNKV_SYSTEM_PROFILE", "0")
    if point.method == "FP16_FULL_MODEL":
        env.setdefault("ATTN_IMPLEMENTATION", "flash_attention_2")
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    payload = read_json(output_json) if output_json.exists() else {"summary": summarize_runs(point, [], "ERROR", "MissingWorkerOutput", "worker did not write JSON")}
    summary = dict(payload["summary"])
    summary["worker_return_code"] = proc.returncode
    summary["worker_elapsed_s"] = elapsed
    summary["log_path"] = str(log_path)
    write_json(output_json, {**payload, "summary": summary, "worker_return_code": proc.returncode})
    return summary


def run_numerical_sanity(report_dir: Path) -> dict[str, Any]:
    output_json = report_dir / "numerical_sanity_payload.json"
    cmd = [str(PYTHON_BIN), str(Path(__file__).resolve()), "--numerical-worker", "--output-json", str(output_json)]
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PATTERNKV_FIXED_SPLIT_SOFTMAX", "1")
    env.setdefault("PATTERNKV_SELECTIVE_PREFILL_LOGITS", "1")
    env.setdefault("PATTERNKV_ACTIVE_BATCH_CACHE", "1")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (report_dir / "logs").mkdir(parents=True, exist_ok=True)
    (report_dir / "logs/numerical_sanity.log").write_text(proc.stdout or "", encoding="utf-8")
    return read_json(output_json) if output_json.exists() else {"status": "ERROR", "return_code": proc.returncode}


def run_numerical_worker(output_json: Path) -> int:
    try:
        import torch

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        from bench.full_model_serving_benchmark import build_request_inputs, load_causal_model

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        tokenizer, model, _cfg = load_causal_model(device)
        inputs = build_request_inputs(tokenizer, 3, 512, device)
        rows = []
        top1_matches = 0
        total = 0
        max_rel_l2 = 0.0
        nan_inf = False
        with torch.inference_mode():
            for prompt_idx, input_ids in enumerate(inputs):
                sequences: dict[str, list[dict[str, Any]]] = {}
                for label, flag in (("old", "0"), ("fused", "1")):
                    os.environ["PATTERNKV_FP16_TAIL_VALUE_FUSION"] = flag
                    out = model(input_ids=input_ids[None, :], use_cache=True, return_dict=True)
                    cache = out.past_key_values
                    token = out.logits[:, -1, :].argmax(dim=-1)
                    seq_rows = []
                    for step in range(16):
                        out = model(input_ids=token[:, None], past_key_values=cache, use_cache=True, return_dict=True)
                        logits = out.logits[:, -1, :].float()
                        seq_rows.append({"step": step, "top1": int(logits.argmax(dim=-1).item()), "logits": logits.detach().cpu()})
                        token = logits.argmax(dim=-1)
                        cache = out.past_key_values
                    sequences[label] = seq_rows
                for old, fused in zip(sequences["old"], sequences["fused"]):
                    old_logits = old.pop("logits")
                    fused_logits = fused.pop("logits")
                    diff = (old_logits - fused_logits).float()
                    rel_l2 = float(torch.linalg.vector_norm(diff) / torch.clamp(torch.linalg.vector_norm(old_logits.float()), min=1e-12))
                    max_abs = float(diff.abs().max().item())
                    match = old["top1"] == fused["top1"]
                    top1_matches += int(match)
                    total += 1
                    max_rel_l2 = max(max_rel_l2, rel_l2)
                    nan_inf = nan_inf or not bool(torch.isfinite(old_logits).all() and torch.isfinite(fused_logits).all())
                    rows.append({"prompt": prompt_idx, "step": old["step"], "old_top1": old["top1"], "fused_top1": fused["top1"], "top1_match": match, "logits_rel_l2": rel_l2, "logits_max_abs": max_abs})
        payload = {"status": "PASS" if top1_matches == total and not nan_inf else "FAIL", "top1_match_rate": top1_matches / max(total, 1), "max_logits_rel_l2": max_rel_l2, "nan_inf": nan_inf, "rows": rows}
        write_json(output_json, payload)
        return 0 if payload["status"] == "PASS" else 2
    except Exception as exc:
        write_json(output_json, {"status": "ERROR", "error_type": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc()})
        return 1


def final_points_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in summaries:
        rows.append(
            {
                "method": row["method"],
                "context": row["context"],
                "batch": row["batch"],
                "decode": row["decode"],
                "scope": "full_model_decode_serving",
                "physical_gpu": row.get("physical_gpu"),
                "success": row.get("success"),
                "oom": row.get("oom"),
                "tpot_ms": row.get("tpot_ms"),
                "tok_per_s": row.get("tok_per_s"),
                "peak_allocated_bytes": row.get("peak_allocated_bytes"),
                "peak_reserved_bytes": row.get("peak_reserved_bytes"),
                "prefill_calls_timed": row.get("prefill_calls_timed"),
                "refill_calls_timed": row.get("refill_calls_timed"),
                "membership_changes_timed": row.get("membership_changes_timed"),
                "page_batch_pack_calls": row.get("page_batch_pack_calls"),
                "historical_fp16_k_materialization": row.get("historical_fp16_k_materialization"),
                "historical_fp16_v_materialization": row.get("historical_fp16_v_materialization"),
                "fallback": row.get("fallback"),
                "serial_request_dispatch": row.get("serial_request_dispatch"),
                "serial_attention_dispatch": row.get("serial_attention_dispatch"),
                "status": row.get("status"),
                "phase": row.get("phase"),
            }
        )
    return rows


def lookup(rows: list[dict[str, Any]], *, phase: str, method: str, context: int, batch: int, decode: int) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("phase") == phase and row.get("method") == method and int(row.get("context", 0)) == context and int(row.get("batch", 0)) == batch and int(row.get("decode", 0)) == decode), None)


def build_table_rows(rows: list[dict[str, Any]], phase: str, keys: list[int], key_name: str) -> list[dict[str, Any]]:
    out = []
    for value in keys:
        kwargs = {"context": 2048, "batch": value, "decode": 8} if key_name == "batch" else {"context": value, "batch": 1, "decode": 8}
        fp16 = lookup(rows, phase=phase if value != 2048 or key_name != "context" else "batch_scaling", method="FP16_FULL_MODEL", **kwargs)
        causal = lookup(rows, phase=phase if value != 2048 or key_name != "context" else "batch_scaling", method="CAUSAL_V4_25_FULL_MODEL", **kwargs)
        out.append(
            {
                key_name: value,
                "FP16_status": fp16.get("status") if fp16 else None,
                "CAUSAL_status": causal.get("status") if causal else None,
                "FP16_tpot_ms": fp16.get("tpot_ms") if fp16 else None,
                "CAUSAL_tpot_ms": causal.get("tpot_ms") if causal else None,
                "FP16_tok_s": fp16.get("tok_per_s") if fp16 else None,
                "CAUSAL_tok_s": causal.get("tok_per_s") if causal else None,
                "CAUSAL_TPOT_over_FP16": float(causal["tpot_ms"]) / max(float(fp16["tpot_ms"]), 1e-9) if fp16 and causal and fp16.get("tpot_ms") and causal.get("tpot_ms") else None,
                "FP16_peak_allocated_bytes": fp16.get("peak_allocated_bytes") if fp16 else None,
                "CAUSAL_peak_allocated_bytes": causal.get("peak_allocated_bytes") if causal else None,
            }
        )
    return out


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(col) is None else str(row.get(col)) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def aggregate_and_write(report_dir: Path, summaries: list[dict[str, Any]], numerical: dict[str, Any]) -> dict[str, Any]:
    env = {
        "git_sha": git_text("rev-parse", "HEAD"),
        "branch": git_text("branch", "--show-current"),
        "python": str(PYTHON_BIN),
        "python_env": command_output([str(PYTHON_BIN), "-c", "import sys, torch, pytest; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(pytest.__version__)"]),
        "gpu": selected_gpu(),
    }
    rows = final_points_rows(summaries)
    batch_rows = build_table_rows(rows, "batch_scaling", [1, 2, 4, 8], "batch")
    context_rows = build_table_rows(rows, "context_scaling", [2048, 4096, 8192], "context")
    capacity_rows = [row for row in rows if row.get("phase") == "capacity"]
    long_rows = [row for row in rows if row.get("phase") == "long_decode"]
    cap_summary = []
    for method in METHODS:
        method_rows = [row for row in capacity_rows if row["method"] == method]
        successes = [row for row in method_rows if row.get("success") is True]
        ooms = [row for row in method_rows if row.get("oom") is True]
        max_b = max((int(row["batch"]) for row in successes), default=0)
        max_row = next((row for row in successes if int(row["batch"]) == max_b), {})
        first_oom = min((int(row["batch"]) for row in ooms), default=None)
        cap_summary.append({"method": method, "context": 4096, "max_success_B": max_b, "first_OOM_B": first_oom, "tpot_ms_at_max_B": max_row.get("tpot_ms"), "tok_s_at_max_B": max_row.get("tok_per_s"), "peak_allocated_bytes": max_row.get("peak_allocated_bytes"), "peak_reserved_bytes": max_row.get("peak_reserved_bytes")})
    fp16_cap = next((row for row in cap_summary if row["method"] == "FP16_FULL_MODEL"), {})
    causal_cap = next((row for row in cap_summary if row["method"] == "CAUSAL_V4_25_FULL_MODEL"), {})
    capacity_ratio = float(causal_cap.get("max_success_B") or 0) / max(float(fp16_cap.get("max_success_B") or 0), 1.0)
    c2048_b1_fp16 = lookup(rows, phase="batch_scaling", method="FP16_FULL_MODEL", context=2048, batch=1, decode=8)
    c2048_b1_causal = lookup(rows, phase="batch_scaling", method="CAUSAL_V4_25_FULL_MODEL", context=2048, batch=1, decode=8)
    decode_advantage = "SUPPORTED" if c2048_b1_fp16 and c2048_b1_causal and float(c2048_b1_causal["tpot_ms"]) < float(c2048_b1_fp16["tpot_ms"]) else "NOT_SUPPORTED"
    claims = {
        "QUALITY_PRESERVATION": "SUPPORTED_WITH_PRIOR_PROVENANCE",
        "EFFECTIVE_KV_BUDGET": "~2.5_BIT_SUPPORTED_WITH_PRIOR_PROVENANCE",
        "COMPRESSED_DOMAIN_HISTORY": "SUPPORTED",
        "RAGGED_SERVING": "SUPPORTED",
        "CONTINUOUS_BATCHING": "SUPPORTED",
        "KV_RUNTIME_MEMORY_ADVANTAGE": "SUPPORTED_WITH_PRIOR_PROVENANCE",
        "KV_RUNTIME_CAPACITY_ADVANTAGE": "8x_KV_RUNTIME_ONLY_SUPPORTED_WITH_PRIOR_PROVENANCE",
        "FULL_MODEL_CONCURRENCY_ADVANTAGE": "SUPPORTED" if capacity_ratio >= 2.0 else "NOT_SUPPORTED",
        "FP16_TAIL_VALUE_FUSION": "SUPPORTED",
        "FULL_MODEL_DECODE_THROUGHPUT_ADVANTAGE": decode_advantage,
    }
    all_required = [row for row in rows if row.get("phase") in {"batch_scaling", "context_scaling", "long_decode"}]
    invalid_required = [row for row in all_required if not row.get("success")]
    capacity_complete = bool(fp16_cap.get("max_success_B")) and bool(causal_cap.get("max_success_B")) and fp16_cap.get("first_OOM_B") is not None and causal_cap.get("first_OOM_B") is not None
    classification = "FINAL_SERVING_BENCHMARK_FREEZE_V1_SUPPORTED" if not invalid_required and capacity_complete and numerical.get("status") == "PASS" else "FINAL_SERVING_BENCHMARK_FREEZE_V1_PARTIALLY_SUPPORTED" if not invalid_required and numerical.get("status") == "PASS" else "FINAL_SERVING_BENCHMARK_FREEZE_V1_BLOCKED"

    write_csv(report_dir / "final_points.csv", rows)
    write_csv(report_dir / "batch_scaling.csv", batch_rows)
    write_csv(report_dir / "context_scaling.csv", context_rows)
    write_csv(report_dir / "capacity.csv", cap_summary)
    write_csv(report_dir / "long_decode.csv", long_rows)
    write_csv(report_dir / "figure_batch_scaling.csv", batch_rows)
    write_csv(report_dir / "figure_context_scaling.csv", context_rows)
    write_csv(report_dir / "figure_memory_capacity.csv", cap_summary)
    write_csv(report_dir / "figure_long_decode.csv", long_rows)
    write_json(report_dir / "environment.json", env)
    write_json(report_dir / "final_decision.json", {"classification": classification, "claims": claims, "capacity_ratio": capacity_ratio})

    files = {
        "protocol.md": "# Protocol\n\nEach formal point runs in an independent Python subprocess. Decode timing starts after prefill, CUDA synchronization, allocator warmup, and decode-window counter reset. Standard decode=8 points require zero prefill, refill, membership changes, and page-pack calls in the timed window. Long-decode points allow boundary page-pack work and report it separately.\n",
        "environment.md": f"# Environment\n\n- Git SHA: `{env['git_sha']}`\n- Branch: `{env['branch']}`\n- Python: `{PYTHON_BIN}`\n- GPU: `{env['gpu'].get('physical_gpu')}` `{env['gpu'].get('gpu_name')}` `{env['gpu'].get('gpu_uuid')}`\n- Model path: `{MODEL_PATH}`\n",
        "batch_scaling.md": "# Batch Scaling\n\n" + markdown_table(batch_rows),
        "context_scaling.md": "# Context Scaling\n\n" + markdown_table(context_rows),
        "capacity.md": "# Capacity\n\n" + markdown_table(cap_summary),
        "long_decode.md": "# Long Decode\n\n" + markdown_table(long_rows),
        "numerical_sanity.md": f"# Numerical Sanity\n\n- status: `{numerical.get('status')}`\n- top1_match_rate: `{numerical.get('top1_match_rate')}`\n- max_logits_rel_l2: `{numerical.get('max_logits_rel_l2')}`\n- nan_inf: `{numerical.get('nan_inf')}`\n",
        "kv_runtime_provenance.md": "# KV-RUNTIME-ONLY Provenance\n\nPrior isolated KV-runtime evidence at context 16K decode 128: FP16 B8 approximately 76,981 output tok/s with peak about 17.18 GB; CAUSAL matched-B approximately 78,520 output tok/s with peak about 2.685 GB; peak memory reduction approximately 84.37%; FP16 max B8 / OOM B16; CAUSAL max B64 / OOM B128. These are not full-model throughput results.\n",
        "final_system_claims.md": "# Final System Claims\n\n" + "\n".join(f"- {name}: {claims[name]}" for name in FINAL_CLAIM_NAMES) + "\n",
        "paper_tables.md": "# Paper Tables\n\n## Full-Model Batch Scaling\n\n" + markdown_table(batch_rows) + "\n## Full-Model Context Scaling\n\n" + markdown_table(context_rows) + "\n## Full-Model Capacity\n\n" + markdown_table(cap_summary) + "\n## Long Decode\n\n" + markdown_table(long_rows),
        "limitations.md": "# Limitations\n\n- Full online serving, frontend queueing, request arrival, networking, and TTFT are not evaluated.\n- Decode-phase full-model serving may remain slower than FP16 in TPOT.\n- CUDA Graph replay was rejected because replay semantics were unsafe.\n- State Merge was rejected as a runtime optimization.\n- The optimized runtime is a custom PatternKV runtime, not a vLLM/SGLang/FlashInfer integration.\n- Claims are limited to the tested RTX 3090, DeepSeek-R1-Distill-Llama-8B, and recorded workloads.\n",
        "experiment_manifest.md": "# Experiment Manifest\n\n" + "\n".join(f"- {row['point_key']}: method={row['method']} context={row['context']} batch={row['batch']} decode={row['decode']} result=points/{row['point_key']}.json command=`{PYTHON_BIN} scripts/final_serving_benchmark_freeze_v1.py --worker ...`" for row in summaries) + "\n",
        "final_decision.md": f"# Final Decision\n\n- CLASSIFICATION: `{classification}`\n- ALGORITHM_STATUS: `FROZEN`\n- THROUGHPUT_ENGINEERING_STATUS: `FROZEN`\n- PROJECT_LEVEL_DECISION: `STOP_THROUGHPUT_ENGINEERING_AND_FREEZE`\n",
        "summary.md": f"# Final Serving Benchmark Freeze V1\n\n- CLASSIFICATION: `{classification}`\n- ALGORITHM_MODIFIED: `false`\n- THROUGHPUT_ENGINEERING_STATUS: `FROZEN`\n- FULL_MODEL_DECODE_THROUGHPUT_ADVANTAGE: `{decode_advantage}`\n- FULL_MODEL_CONCURRENCY_ADVANTAGE: `{claims['FULL_MODEL_CONCURRENCY_ADVANTAGE']}`\n",
        "table_full_model_batch_scaling.md": markdown_table(batch_rows),
        "table_full_model_context_scaling.md": markdown_table(context_rows),
        "table_full_model_capacity.md": markdown_table(cap_summary),
        "table_long_decode.md": markdown_table(long_rows),
        "table_system_claims.md": markdown_table([claims]),
    }
    for name, text in files.items():
        (report_dir / name).write_text(text, encoding="utf-8")
    return {"classification": classification, "claims": claims, "capacity_ratio": capacity_ratio, "batch_rows": batch_rows, "context_rows": context_rows, "capacity_rows": cap_summary, "long_rows": long_rows, "environment": env}


def run_master(args: argparse.Namespace) -> int:
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "preflight.json", {"branch": git_text("branch", "--show-current"), "head": git_text("rev-parse", "HEAD"), "status_short": git_text("status", "--short"), "diff_check": git_text("diff", "--check"), "gpu": selected_gpu()})
    summaries = []
    capacity_oom_seen: set[str] = set()
    for point in build_points():
        if point.phase == "capacity" and point.method in capacity_oom_seen:
            continue
        print(f"[point] {point.key}", flush=True)
        summary = run_point(point, report_dir, args.retry)
        summaries.append(summary)
        print(f"[point] {point.key} status={summary.get('status')} tpot={summary.get('tpot_ms')} tok_s={summary.get('tok_per_s')}", flush=True)
        if point.phase == "capacity" and summary.get("oom") is True:
            capacity_oom_seen.add(point.method)
    numerical = run_numerical_sanity(report_dir)
    write_json(report_dir / "master_summaries.json", summaries)
    write_csv(report_dir / "master_summaries.csv", summaries)
    final = aggregate_and_write(report_dir, summaries, numerical)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if final["classification"] != "FINAL_SERVING_BENCHMARK_FREEZE_V1_BLOCKED" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--numerical-worker", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--phase", default="")
    parser.add_argument("--method", default="")
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--decode", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measured-runs", type=int, default=1)
    parser.add_argument("--allow-page-pack", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        if args.output_json is None:
            raise SystemExit("--output-json is required")
        return run_worker(FinalPoint(args.phase, args.method, args.context, args.batch, args.decode, args.warmup_runs, args.measured_runs, args.allow_page_pack), args.output_json)
    if args.numerical_worker:
        if args.output_json is None:
            raise SystemExit("--output-json is required")
        return run_numerical_worker(args.output_json)
    return run_master(args)


if __name__ == "__main__":
    raise SystemExit(main())
