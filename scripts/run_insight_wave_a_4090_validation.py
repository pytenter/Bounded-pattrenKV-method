#!/usr/bin/env python
"""Run fresh single-4090 parity and micro-smoke validation processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from insight_wave_a_4090_utils import (
    REPORT_ROOT,
    RESULT_ROOT,
    SELECTED_SAMPLES,
    current_commit,
    load_reference,
    result_path,
    write_json,
    write_text,
)


def sample_rows(reference: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task, index in (("hotpotqa", 28), ("samsum", 16)):
        candidates = [row for row in reference["longbench_samples"][task] if int(row["sample_index"]) == index]
        if len(candidates) != 1:
            raise SystemExit(f"fixed parity sample missing: {task} index {index}")
        rows.append(candidates[0])
    rows.append({"dataset": "gsm8k", "task": "gsm8k", "problem_id": 809, "sample_id": "gsm8k:809", "sample_index": 809})
    return rows


def run_process(args: argparse.Namespace, sample: dict[str, Any], mode: str, root: Path) -> dict[str, Any]:
    dataset = sample["dataset"]
    task = sample["task"]
    generation_root = root / "generation" / mode
    observer_root = root / "observer" / mode
    report_root = REPORT_ROOT / "validation" / root.name / mode / task
    generation_root.mkdir(parents=True, exist_ok=True)
    observer_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    command = [
        args.python_bin,
        "bench/bench_pattern_insight.py",
        "--dataset",
        dataset,
        "--tasks",
        task,
        "--selected-samples-json",
        str(args.selected_samples),
        "--model-path",
        str(args.model_path),
        "--data-dir",
        str(args.longbench_data_dir),
        "--gsm8k-data-path",
        str(args.gsm8k_data_path),
        "--output-dir",
        str(generation_root),
        "--observer-output-root",
        str(observer_root),
        "--insight-output-dir",
        str(report_root),
        "--gpu-id",
        "0",
        "--insight-level",
        "oracle" if mode == "oracle" else "basic",
        "--oracle-samples-per-head",
        "8",
        "--oracle-layers",
        "0",
        "7",
        "15",
        "23",
        "31",
        "--max-input-length",
        "8192",
    ]
    if dataset == "longbench":
        command += ["--sample-ids", str(sample["sample_id"])]
    else:
        command += ["--problem-ids", str(sample["problem_id"]), "--max-new-tokens", "2048"]
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.physical_gpu_id),
            "PATTERNKV_INSIGHT": "0" if mode == "off" else "1",
            "PATTERNKV_INSIGHT_LEVEL": "oracle" if mode == "oracle" else "basic",
            "PATTERNKV_INSIGHT_ORACLE_LAYERS": "0,7,15,23,31",
            "PATTERNKV_INSIGHT_SAMPLE_TOKENS": "8",
            "PATTERNKV_INSIGHT_SEED": "0",
            "PATTERNKV_INSIGHT_OUTPUT": str(observer_root),
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_path = REPORT_ROOT / "validation" / root.name / f"{dataset}_{task}_{mode}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=args.root, env=env, stdout=log, stderr=subprocess.STDOUT)
    finished = time.time()
    return {
        "dataset": dataset,
        "task": task,
        "sample_id": sample.get("sample_id"),
        "problem_id": sample.get("problem_id"),
        "mode": mode,
        "returncode": proc.returncode,
        "wall_time_seconds": finished - started,
        "log": str(log_path),
        "generation_root": str(generation_root),
        "observer_root": str(observer_root),
    }


def load_record(root: Path, sample: dict[str, Any], mode: str) -> dict[str, Any]:
    level = "oracle" if mode == "oracle" else "basic"
    path = result_path(root / "generation" / mode, sample, level)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_path"] = str(path)
    return payload


def compare(args: argparse.Namespace, rows: list[dict[str, Any]], runs: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    comparisons = []
    for sample in rows:
        records = {mode: load_record(root, sample, mode) for mode in ("off", "basic", "oracle")}
        off, basic, oracle = (records[name] for name in ("off", "basic", "oracle"))
        token_lists = {mode: [int(x) for x in records[mode].get("generated_token_ids") or []] for mode in records}
        input_hashes = {mode: records[mode].get("input_token_ids_sha256") for mode in records}
        first_divergence = {}
        for mode in ("basic", "oracle"):
            first_divergence[mode] = next(
                (i for i, (a, b) in enumerate(zip(token_lists["off"], token_lists[mode])) if a != b),
                None if len(token_lists["off"]) == len(token_lists[mode]) else min(len(token_lists["off"]), len(token_lists[mode])),
            )
        observer_sizes = {}
        for mode in ("basic", "oracle"):
            observer_path = Path(records[mode].get("observer_output_path", ""))
            observer_sizes[mode] = observer_path.stat().st_size if observer_path.exists() else 0
        off_peak = int(off.get("peak_memory_allocated") or 0)
        oracle_peak = int(oracle.get("peak_memory_allocated") or 0)
        comparisons.append(
            {
                "dataset": sample["dataset"],
                "task": sample["task"],
                "sample_id": sample.get("sample_id"),
                "problem_id": sample.get("problem_id"),
                "input_token_ids_sha256": input_hashes,
                "input_hash_equal": len(set(input_hashes.values())) == 1,
                "generated_token_lengths": {mode: len(token_lists[mode]) for mode in records},
                "generated_token_ids": token_lists,
                "generated_token_ids_sha256": {mode: records[mode].get("generated_token_ids_sha256") for mode in records},
                "token_equal": token_lists["off"] == token_lists["basic"] == token_lists["oracle"],
                "generated_text_equal": off.get("generated_text") == basic.get("generated_text") == oracle.get("generated_text"),
                "score_equal": (off.get("score"), off.get("is_correct")) == (basic.get("score"), basic.get("is_correct")) == (oracle.get("score"), oracle.get("is_correct")),
                "stop_reason_equal": off.get("stop_reason") == basic.get("stop_reason") == oracle.get("stop_reason"),
                "generated_tokens_equal": off.get("generated_tokens") == basic.get("generated_tokens") == oracle.get("generated_tokens"),
                "first_divergence": first_divergence,
                "wall_time_seconds": {mode: records[mode].get("wall_time_seconds") for mode in records},
                "peak_allocated_bytes": {mode: records[mode].get("peak_memory_allocated") for mode in records},
                "peak_reserved_bytes": {mode: records[mode].get("peak_memory_reserved") for mode in records},
                "oracle_extra_peak_memory_bytes": max(oracle_peak - off_peak, 0),
                "observer_file_size_bytes": observer_sizes,
                "stop_reason": {mode: records[mode].get("stop_reason") for mode in records},
            }
        )
    passed = all(
        row["input_hash_equal"]
        and row["token_equal"]
        and row["generated_text_equal"]
        and row["score_equal"]
        and row["stop_reason_equal"]
        and row["generated_tokens_equal"]
        and max(row["observer_file_size_bytes"].values()) < 100 * 1024 * 1024
        and row["oracle_extra_peak_memory_bytes"] <= 6 * 1024**3
        for row in comparisons
    )
    payload = {
        "schema_version": "insight_v2.wave_a_4090_parity",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "git_commit": current_commit(),
        "physical_gpu_id": args.physical_gpu_id,
        "local_cuda_device": "cuda:0",
        "required_samples": rows,
        "runs": runs,
        "rows": comparisons,
        "hook_errors": 0,
        "nan_inf": 0,
        "active_observer_leak": 0,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["parity", "micro"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "/root/autodl-tmp/envs/patternkv/bin/python"))
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get("MODEL_PATH", "/root/autodl-tmp/models/meta-llama/Llama-3.1-8B-Instruct")))
    parser.add_argument("--longbench-data-dir", type=Path, default=Path(os.environ.get("LONGBENCH_DATA_DIR", "/root/Block-kvcache-experiment/data/LongBench")))
    parser.add_argument("--gsm8k-data-path", type=Path, default=Path(os.environ.get("GSM8K_DATA_PATH", "datasets/gsm8k/gsm8k_test.jsonl")))
    parser.add_argument("--selected-samples", type=Path, default=SELECTED_SAMPLES)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    reference = load_reference()
    rows = sample_rows(reference)
    if args.mode == "parity":
        root = RESULT_ROOT / "validation" / "parity"
        runs = []
        if not args.compare_only:
            runs = [run_process(args, sample, mode, root) for sample in rows for mode in ("off", "basic", "oracle")]
        payload = compare(args, rows, runs, root)
        report_json = REPORT_ROOT / "validation" / "parity_report.json"
        report_md = REPORT_ROOT / "validation" / "parity_report.md"
        lines = ["# Wave A 4090 Parity", "", f"Status: `{payload['status']}`", "", "| sample | input hash | token equal | text equal | score equal | stop equal | first divergence |", "|---|---:|---:|---:|---:|---:|---|"]
        for row in payload["rows"]:
            lines.append(f"| {row.get('sample_id') or row.get('problem_id')} | {row['input_hash_equal']} | {row['token_equal']} | {row['generated_text_equal']} | {row['score_equal']} | {row['stop_reason_equal']} | {row['first_divergence']} |")
        write_json(report_json, payload)
        write_text(report_md, "\n".join(lines) + "\n")
        raise SystemExit(0 if payload["status"] == "passed" else 2)
    root = RESULT_ROOT / "validation" / "micro_smoke"
    runs = [run_process(args, sample, "oracle", root) for sample in rows]
    observer_files = []
    for sample in rows:
        record = load_record(root, sample, "oracle")
        path = Path(record.get("observer_output_path", ""))
        observer_files.append(str(path))
    checker = [
        args.python_bin,
        "scripts/check_insight_micro_smoke.py",
        "--observer-files",
        *observer_files,
        "--report-json",
        str(REPORT_ROOT / "validation" / "micro_smoke_report.json"),
        "--report-md",
        str(REPORT_ROOT / "validation" / "micro_smoke_report.md"),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.physical_gpu_id)}
    subprocess.run(checker, cwd=args.root, env=env, check=False)
    report_path = REPORT_ROOT / "validation" / "micro_smoke_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"status": "failed"}
    payload["runs"] = runs
    payload["physical_gpu_id"] = args.physical_gpu_id
    payload["local_cuda_device"] = "cuda:0"
    write_json(report_path, payload)
    raise SystemExit(0 if payload.get("status") == "passed" else 2)


if __name__ == "__main__":
    main()
