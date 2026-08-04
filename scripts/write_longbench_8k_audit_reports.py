#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.longbench_config import MAX_NEW_TOKENS, METRIC_NAMES, PROMPT_TEMPLATES, SUBTASKS, expected_samples

METHODS = ("fp16", "kivi_paper_g128", "patternkv_paper")
MODEL_PATH = Path("/root/autodl-tmp/models/meta-llama/Llama-3.1-8B-Instruct")
PROMPT_PATH = ROOT / "bench/longbench_config/dataset2prompt.json"
MAXLEN_PATH = ROOT / "bench/longbench_config/dataset2maxlen.json"
SCORER_PATH = ROOT / "bench/_longbench_scorer.py"
CONFIG_PATH = ROOT / "configs/longbench_paper_v2_8k_single4090.yaml"
EXT_PATH = ROOT / "quant/patternkv_gemv.cpython-310-x86_64-linux-gnu.so"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def latest_by_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest = {}
    for row in rows:
        key = (row.get("method"), row.get("task"), row.get("sample_id"))
        latest[key] = row
    return list(latest.values())


def result_rows(base: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(base.glob("*/*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    latest = latest_by_sample(rows)
    return {
        "records": len(rows),
        "latest_unique": len(latest),
        "success": sum(1 for row in latest if row.get("stop_reason") not in ("oom", "error")),
        "oom": sum(1 for row in latest if row.get("stop_reason") == "oom"),
        "error": sum(1 for row in latest if row.get("stop_reason") == "error"),
    }


def peak_gib(row: dict[str, Any]) -> str:
    peak = row.get("peak_memory_reserved_bytes") or row.get("peak_memory_allocated_bytes")
    if peak is None:
        return "n/a"
    return f"{peak / (1024 ** 3):.2f}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def dataset_audit(data_dir: Path, report_dir: Path, sample_limit: int | None) -> None:
    lines = [
        "# Dataset Audit",
        "",
        "Experiment: PatternKV paper-v2 configuration-aligned LongBench reproduction with an 8K input cap.",
        f"Run scope: 21 tasks x {sample_limit} samples per task x 3 methods." if sample_limit else "Run scope: full local LongBench panel x 3 methods.",
        "",
        "| task | local path | local samples | selected samples | max_gen | metric | prompt_template_hash |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for task in SUBTASKS:
        path = data_dir / "data" / f"{task}.jsonl"
        if not path.exists():
            path = data_dir / f"{task}.jsonl"
        count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.exists() else 0
        selected = min(count, sample_limit) if sample_limit else count
        prompt_hash = hashlib.sha256(PROMPT_TEMPLATES[task].encode("utf-8")).hexdigest()
        lines.append(f"| {task} | `{path}` | {count} | {selected} | {MAX_NEW_TOKENS[task]} | {METRIC_NAMES[task]} | `{prompt_hash}` |")
    write(report_dir / "dataset_audit.md", "\n".join(lines))


def config_audit(report_dir: Path, output_tag: str, sample_limit: int | None) -> None:
    model_hashes = {
        "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "tokenizer_config_sha256": sha256_file(MODEL_PATH / "tokenizer_config.json"),
        "generation_config_sha256": sha256_file(MODEL_PATH / "generation_config.json"),
        "prompt_sha256": sha256_file(PROMPT_PATH),
        "maxlen_sha256": sha256_file(MAXLEN_PATH),
        "scorer_sha256": sha256_file(SCORER_PATH),
        "8k_config_sha256": sha256_file(CONFIG_PATH),
        "patternkv_extension_sha256": sha256_file(EXT_PATH),
    }
    lines = [
        "# Config Audit",
        "",
        f"Output tag: `{output_tag}`",
        f"Run scope: `21x{sample_limit}` subset" if sample_limit else "Run scope: full local LongBench panel",
        "Experiment name: `longbench_paper_v2_8k_single4090`",
        "Description: PatternKV paper-v2 configuration-aligned LongBench reproduction with an 8K input cap. This is not the paper's strict 31.5K reproduction.",
        "",
        f"Git branch: `{command(['git', 'branch', '--show-current'])}`",
        f"Git commit: `{command(['git', 'rev-parse', 'HEAD'])}`",
        f"Model path: `{MODEL_PATH}`",
        "Hardware target: NVIDIA GeForce RTX 4090 D 24GB, GPU 0 only.",
        "MAX_INPUT_LENGTH: `8192`",
        "Batch size: `1`; decoding: greedy; `use_cache=true`.",
        "",
        "Methods:",
        "",
        "- `fp16`: KV quantization disabled; backend `fp16`.",
        "- `kivi_paper_g128`: k/v 2-bit, group_size 128, residual_length 128, asymmetric, official KIVI backend, persistent KV heads 8 for Llama GQA.",
        "- `patternkv_paper`: k/v 2-bit, group_size 128, residual_length 128, 32 K patterns, 32 V patterns, G_pattern 128, post-RoPE selection.",
        "",
        "Hashes:",
        "",
        "```json",
        json.dumps(model_hashes, indent=2, sort_keys=True),
        "```",
    ]
    write(report_dir / "config_audit.md", "\n".join(lines))


def protocol(report_dir: Path, sample_limit: int | None) -> None:
    lines = [
        "# Experiment Protocol",
        "",
        "This run uses Llama-3.1-8B-Instruct and the 21 LongBench tasks with the repository's official task-specific prompts, max generation lengths, and scorer.",
        "KIVI and PatternKV use the paper-v2-aligned 2-bit G128/R128 quantization configuration.",
        "The input cap is changed from the paper's approximately 31.5K setting to 8192 tokens because this worker is a single RTX 4090 D 24GB server.",
        "Therefore this experiment must be described as an 8K-capped reproduction, not a strict 31.5K paper reproduction.",
        "Samples shorter than 8192 tokens keep their natural length; longer samples use middle truncation.",
        f"The current formal run scope is 21 tasks x {sample_limit} samples per task x 3 methods, per the latest user instruction." if sample_limit else "The formal run scope is the full local LongBench panel x 3 methods.",
    ]
    write(report_dir / "experiment_protocol.md", "\n".join(lines))


def smoke_report(report_dir: Path, smoke_dir: Path, edge_dir: Path) -> None:
    smoke_rows = latest_by_sample(result_rows(smoke_dir))
    edge_rows = latest_by_sample(result_rows(edge_dir))
    lines = ["# Smoke Report", "", "Functional smoke uses natural short samples from `trec`, `samsum`, and `passage_count`.", ""]
    lines += ["| method | latest unique | success | OOM | error |", "| --- | ---: | ---: | ---: | ---: |"]
    for method in METHODS:
        rows = [r for r in smoke_rows if r.get("method") == method]
        c = status_counts(rows)
        lines.append(f"| {method} | {c['latest_unique']} | {c['success']} | {c['oom']} | {c['error']} |")
    write(report_dir / "smoke_report.md", "\n".join(lines))

    edge_lines = ["# Edge Smoke Report", "", "Edge smoke reuses a real LongBench sample near the 8K cap. No padding or synthetic text is used.", ""]
    edge_lines += ["| method | task | input_tokens | max_gen | stop_reason | peak_reserved_GiB |", "| --- | --- | ---: | ---: | --- | ---: |"]
    for row in edge_rows:
        edge_lines.append(f"| {row.get('method')} | {row.get('task')} | {row.get('input_tokens_after_special_tokens')} | {row.get('max_gen')} | {row.get('stop_reason')} | {peak_gib(row)} |")
    write(report_dir / "edge_smoke_report.md", "\n".join(edge_lines))


def eta_report(report_dir: Path, smoke_dir: Path, edge_dir: Path, planned_total: int) -> None:
    rows = latest_by_sample(result_rows(smoke_dir) + result_rows(edge_dir))
    ok = [r for r in rows if r.get("stop_reason") not in ("oom", "error") and r.get("wall_time_seconds")]
    by_method = {}
    for method in METHODS:
        vals = [float(r["wall_time_seconds"]) for r in ok if r.get("method") == method]
        by_method[method] = {
            "calibration_records": len(vals),
            "mean_generation_seconds": statistics.mean(vals) if vals else None,
            "max_peak_reserved_bytes": max((int(r.get("peak_memory_reserved_bytes") or 0) for r in ok if r.get("method") == method), default=None),
        }
    means = [v["mean_generation_seconds"] for v in by_method.values() if v["mean_generation_seconds"] is not None]
    mean_seconds = statistics.mean(means) if means else None
    eta_seconds = mean_seconds * planned_total if mean_seconds else None
    data = {
        "basis": "completed functional smoke plus edge smoke records",
        "planned_total": planned_total,
        "by_method": by_method,
        "rough_eta_seconds": eta_seconds,
        "rough_eta_hours": eta_seconds / 3600 if eta_seconds else None,
        "note": "This is a lightweight preflight estimate; long summarization tasks may run slower than the smoke mix.",
    }
    (report_dir / "eta_report.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ETA Report",
        "",
        f"Planned records: `{planned_total}`",
        f"Rough ETA hours: `{data['rough_eta_hours']}`",
        "",
        "| method | calibration_records | mean_generation_seconds | max_peak_reserved_GiB |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, item in by_method.items():
        peak = item["max_peak_reserved_bytes"]
        lines.append(f"| {method} | {item['calibration_records']} | {item['mean_generation_seconds']} | {(peak / (1024 ** 3)) if peak else None} |")
    lines.append("")
    lines.append(data["note"])
    write(report_dir / "eta_report.md", "\n".join(lines))


def test_report(report_dir: Path, log_path: Path) -> None:
    log = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    summary = "unknown"
    for line in reversed(log.splitlines()):
        if " passed" in line:
            summary = line.strip()
            break
    write(report_dir / "test_report.md", f"# Test Report\n\nPytest log: `{log_path}`\n\nResult: `{summary}`")


def run_status(report_dir: Path, output_tag: str, planned_total: int) -> None:
    rows = result_rows(ROOT / "results/paper_repro_v2" / output_tag)
    c = status_counts(rows)
    pid_path = ROOT / "logs/paper_repro_v2" / output_tag / "launcher.pid"
    status_path = ROOT / "run/paper_repro_v2" / output_tag / "runner.status.json"
    pid = pid_path.read_text(encoding="utf-8").strip() if pid_path.exists() else None
    alive = pid is not None and command(["bash", "-lc", f"kill -0 {pid} 2>/dev/null && echo alive || echo stopped"]) == "alive"
    runner = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    state = "RUNNING" if alive else ("READY TO RUN" if c["records"] == 0 else "PAUSED / RESUMABLE")
    lines = [
        "# Run Status",
        "",
        f"Updated at: `{utc_now()}`",
        f"Output tag: `{output_tag}`",
        f"State: `{state}`",
        f"Launcher PID: `{pid}`",
        f"Launcher alive: `{alive}`",
        f"Current method: `{runner.get('current_method')}`",
        f"Current task: `{runner.get('current_task')}`",
        f"Current sample: `{runner.get('current_sample')}`",
        f"Planned records: `{planned_total}`",
        f"Completed records: `{c['records']}`",
        f"Latest unique records: `{c['latest_unique']}`",
        f"Success: `{c['success']}`",
        f"OOM: `{c['oom']}`",
        f"Error: `{c['error']}`",
    ]
    write(report_dir / "run_status.md", "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-tag", default="longbench_21x50_8k_4090")
    ap.add_argument("--sample-limit-per-task", type=int, default=50)
    ap.add_argument("--data-dir", type=Path, default=Path("/root/Block-kvcache-experiment/data/LongBench"))
    args = ap.parse_args()
    report_dir = ROOT / "reports/paper_repro_v2" / args.output_tag
    planned_total = len(SUBTASKS) * args.sample_limit_per_task * len(METHODS)
    report_dir.mkdir(parents=True, exist_ok=True)
    config_audit(report_dir, args.output_tag, args.sample_limit_per_task)
    protocol(report_dir, args.sample_limit_per_task)
    dataset_audit(args.data_dir, report_dir, args.sample_limit_per_task)
    smoke_report(report_dir, ROOT / "results/paper_repro_v2/longbench_8k_4090_smoke", ROOT / "results/paper_repro_v2/longbench_8k_4090_edge_smoke")
    eta_report(report_dir, ROOT / "results/paper_repro_v2/longbench_8k_4090_smoke", ROOT / "results/paper_repro_v2/longbench_8k_4090_edge_smoke", planned_total)
    test_report(report_dir, ROOT / "logs/paper_repro_v2" / args.output_tag / "pytest.log")
    run_status(report_dir, args.output_tag, planned_total)


if __name__ == "__main__":
    main()
