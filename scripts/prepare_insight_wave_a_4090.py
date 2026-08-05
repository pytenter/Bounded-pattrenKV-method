#!/usr/bin/env python
"""Freeze V100 identity and record the single RTX 4090 execution environment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from insight_wave_a_4090_utils import (
    LONG_BENCH_TASKS,
    REPORT_ROOT,
    RUNTIME_SENSITIVE_PREFIXES,
    SELECTED_SAMPLES,
    V100_MANIFEST,
    current_commit,
    ordered_sha256,
    sha256_file,
    write_json,
    write_text,
)


V100_RUNTIME_COMMIT = "6c88fb81f92934dbce22e2171501965723df9038"
V100_SUMMARY_COMMIT = "c67ba9a1feb39ac50f404db5fa6ff0a70fb50881"


def run(command: list[str], *, check: bool = True) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def nvidia_rows() -> list[dict[str, Any]]:
    raw = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in raw.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != 7:
            continue
        rows.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "uuid": values[2],
                "driver_version": values[3],
                "memory_total_mib": int(values[4]),
                "memory_used_mib": int(values[5]),
                "utilization_gpu": int(values[6]),
            }
        )
    return rows


def compute_apps() -> list[dict[str, str]]:
    raw = run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=False,
    )
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        values = [part.strip() for part in line.split(",", 3)]
        if len(values) == 4:
            rows.append({"gpu_uuid": values[0], "pid": values[1], "process_name": values[2], "used_memory": values[3]})
    return rows


def detect_hardware() -> dict[str, Any]:
    rows = nvidia_rows()
    targets = [row for row in rows if "RTX 4090" in row["name"]]
    if len(targets) != 1:
        raise SystemExit(f"expected exactly one RTX 4090, found {len(targets)}: {targets}")
    target = targets[0]
    apps = [app for app in compute_apps() if app["gpu_uuid"] == target["uuid"]]
    target["free_memory_mib"] = target["memory_total_mib"] - target["memory_used_mib"]
    target["compute_apps"] = apps
    target["idle_guard_passed"] = target["memory_used_mib"] <= 1024 and not apps
    if not target["idle_guard_passed"]:
        raise SystemExit(f"target RTX 4090 is not idle: {json.dumps(target, sort_keys=True)}")
    visible = subprocess.check_output(
        [
            os.environ.get("PYTHON_BIN", sys.executable),
            "-c",
            "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')",
        ],
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(target["index"])},
    ).splitlines()
    target["visible_cuda_devices"] = int(visible[0])
    target["local_cuda_device"] = "cuda:0"
    if target["visible_cuda_devices"] != 1:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES did not isolate one device: {visible}")
    return target


def make_reference(args: argparse.Namespace) -> dict[str, Any]:
    v100 = json.loads(args.v100_manifest.read_text(encoding="utf-8"))
    if v100.get("commit") != V100_RUNTIME_COMMIT:
        raise SystemExit(f"unexpected V100 runtime commit: {v100.get('commit')}")
    selected = json.loads(args.selected_samples.read_text(encoding="utf-8")).get("selected", [])
    longbench: dict[str, list[dict[str, Any]]] = {}
    for task in LONG_BENCH_TASKS:
        rows = [row for row in selected if row.get("dataset") == "longbench" and row.get("task") == task]
        if len(rows) != 12:
            raise SystemExit(f"expected 12 selected LongBench rows for {task}, found {len(rows)}")
        longbench[task] = rows
    gsm8k_ids: list[int] = []
    for job in v100.get("jobs", []):
        if job.get("dataset") == "gsm8k":
            gsm8k_ids.extend(int(value) for value in job.get("problem_ids", []))
    if len(gsm8k_ids) != 80 or len(set(gsm8k_ids)) != 80:
        raise SystemExit(f"V100 GSM8K identity check failed: total={len(gsm8k_ids)} unique={len(set(gsm8k_ids))}")
    if sum(len(rows) for rows in longbench.values()) != 60:
        raise SystemExit("V100 LongBench identity check failed")
    payload = {
        "schema_version": "insight_v2.wave_a_4090_reference_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "v100_manifest_path": str(args.v100_manifest),
        "v100_manifest_sha256": sha256_file(args.v100_manifest),
        "selected_samples_path": str(args.selected_samples),
        "selected_samples_sha256": sha256_file(args.selected_samples),
        "v100_summary_commit": V100_SUMMARY_COMMIT,
        "v100_runtime_commit": V100_RUNTIME_COMMIT,
        "longbench_tasks": {task: 12 for task in LONG_BENCH_TASKS},
        "longbench_samples": longbench,
        "gsm8k_problem_ids": gsm8k_ids,
        "gsm8k_problem_ids_ordered_sha256": ordered_sha256(gsm8k_ids),
        "longbench_total": 60,
        "gsm8k_total": 80,
        "total": 140,
        "patternkv_config": v100.get("patternkv_config"),
        "insight_config": v100.get("insight_config"),
        "generation_config": {
            "batch_size": 1,
            "do_sample": False,
            "longbench_max_input": 8192,
            "longbench_max_gen": "task-specific MAX_NEW_TOKENS",
            "gsm8k_max_new_tokens": 2048,
        },
        "collector_config": {
            "source": "V100 runtime",
            "record_limit": "runtime InsightCollector max_sample_records",
            "serialization_policy": "unchanged from V100 runtime",
        },
        "v100_csv_paths": {
            name: str(args.v100_manifest.parent / name)
            for name in (
                "pattern_gain_map.csv",
                "matching_oracle_gap.csv",
                "v_gate_confusion.csv",
                "dynamic_pattern_utility.csv",
            )
        },
    }
    return payload


def make_runtime_report(args: argparse.Namespace) -> dict[str, Any]:
    try:
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", args.runtime_commit, "HEAD", "--", *RUNTIME_SENSITIVE_PREFIXES],
            cwd=args.root,
            text=True,
        ).splitlines()
    except subprocess.CalledProcessError:
        changed = ["runtime_diff_unavailable"]
    runtime_equivalent = not changed
    payload = {
        "schema_version": "insight_v2.wave_a_4090_runtime_equivalence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_equivalent": runtime_equivalent,
        "v100_runtime_commit": args.runtime_commit,
        "current_runtime_commit": current_commit(),
        "changed_runtime_sensitive_files": changed,
        "allowed_difference_categories": [
            "single-4090 scheduling",
            "status/stop scripts",
            "validation scripts",
            "reports and tests",
        ],
        "decision": "run_allowed" if runtime_equivalent else "use independent exact-runtime worktree",
        "note": "The actual run is executed from an exact-runtime worktree when this report is false on the c67 audit branch.",
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--v100-manifest", type=Path, default=Path(os.environ.get("V100_MANIFEST_SOURCE", str(V100_MANIFEST))))
    parser.add_argument("--selected-samples", type=Path, default=SELECTED_SAMPLES)
    parser.add_argument("--runtime-commit", default=V100_RUNTIME_COMMIT)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/root/autodl-tmp/models/meta-llama/Llama-3.1-8B-Instruct"))
    parser.add_argument("--longbench-data-dir", default=os.environ.get("LONGBENCH_DATA_DIR", "/root/Block-kvcache-experiment/data/LongBench"))
    parser.add_argument("--gsm8k-data-path", default=os.environ.get("GSM8K_DATA_PATH", "datasets/gsm8k/gsm8k_test.jsonl"))
    args = parser.parse_args()
    args.report_root.mkdir(parents=True, exist_ok=True)
    reference = make_reference(args)
    runtime = make_runtime_report(args)
    hardware = detect_hardware()
    py = os.environ.get("PYTHON_BIN", sys.executable)
    try:
        torch_info = subprocess.check_output(
            [py, "-c", "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"],
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(hardware["index"])},
        ).splitlines()
    except Exception as exc:
        torch_info = [f"error: {exc!r}"]
    try:
        triton_version = subprocess.check_output([py, "-c", "import triton; print(triton.__version__)"], text=True).strip()
    except Exception as exc:
        triton_version = f"error: {exc!r}"
    hardware.update(
        {
            "hostname": socket.gethostname(),
            "python": sys.version,
            "python_path": py,
            "torch": torch_info[0] if torch_info else "unknown",
            "cuda": torch_info[1] if len(torch_info) > 1 else "unknown",
            "triton": triton_version,
            "model_path": args.model_path,
            "longbench_data_dir": args.longbench_data_dir,
            "gsm8k_data_path": args.gsm8k_data_path,
            "runtime_commit": current_commit(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    ref_text = [
        "# Wave A 4090 Reference Manifest",
        "",
        f"- V100 manifest: `{reference['v100_manifest_path']}`",
        f"- V100 manifest SHA256: `{reference['v100_manifest_sha256']}`",
        f"- selected_samples.json SHA256: `{reference['selected_samples_sha256']}`",
        f"- V100 runtime commit: `{reference['v100_runtime_commit']}`",
        f"- V100 summary commit: `{reference['v100_summary_commit']}`",
        f"- LongBench total: `{reference['longbench_total']}`",
        f"- GSM8K total: `{reference['gsm8k_total']}`",
        f"- Total: `{reference['total']}`",
        "",
        "## LongBench Samples",
        "",
    ]
    for task, rows in reference["longbench_samples"].items():
        ref_text.append(f"### {task} (12)")
        ref_text.extend(f"- index={row.get('sample_index')} sample_id={row.get('sample_id')}" for row in rows)
        ref_text.append("")
    ref_text += [
        "## GSM8K Problem IDs",
        "",
        "```text",
        " ".join(str(x) for x in reference["gsm8k_problem_ids"]),
        "```",
        "",
        f"Ordered SHA256: `{reference['gsm8k_problem_ids_ordered_sha256']}`",
    ]
    changed_lines = [f"- `{path}`" for path in runtime["changed_runtime_sensitive_files"]]
    if not changed_lines:
        changed_lines = ["- none"]
    runtime_text = [
        "# Wave A 4090 Runtime Equivalence",
        "",
        f"- Runtime equivalent: `{runtime['runtime_equivalent']}`",
        f"- V100 runtime commit: `{runtime['v100_runtime_commit']}`",
        f"- Current runtime commit: `{runtime['current_runtime_commit']}`",
        "",
        "Changed runtime-sensitive files:",
        *changed_lines,
        "",
        runtime["note"],
    ]
    hardware_text = (
        "# Wave A 4090 Hardware Manifest\n\n"
        f"- GPU: `{hardware['name']}`\n"
        f"- Physical index: `{hardware['index']}`\n"
        f"- UUID: `{hardware['uuid']}`\n"
        f"- Driver: `{hardware['driver_version']}`\n"
        f"- Total memory MiB: `{hardware['memory_total_mib']}`\n"
        f"- Free memory MiB: `{hardware['free_memory_mib']}`\n"
        f"- Visible CUDA devices: `{hardware['visible_cuda_devices']}`\n"
        f"- Process device: `{hardware['local_cuda_device']}`\n"
        f"- Idle guard: `{hardware['idle_guard_passed']}`\n"
        f"- Runtime commit: `{hardware['runtime_commit']}`\n"
    )
    write_json(args.report_root / "reference_manifest.json", reference)
    write_text(args.report_root / "reference_manifest.md", "\n".join(ref_text) + "\n")
    write_json(args.report_root / "runtime_equivalence.json", runtime)
    write_text(args.report_root / "runtime_equivalence.md", "\n".join(runtime_text) + "\n")
    write_json(args.report_root / "hardware_manifest.json", hardware)
    write_text(args.report_root / "hardware_manifest.md", hardware_text)
    validation = args.report_root / "validation"
    write_json(validation / "runtime_equivalence.json", runtime)
    write_json(validation / "hardware_manifest.json", hardware)
    print(json.dumps({"reference_total": reference["total"], "runtime_equivalent": runtime["runtime_equivalent"], "physical_gpu_id": hardware["index"]}, sort_keys=True))


if __name__ == "__main__":
    main()
