#!/usr/bin/env python
"""Summarize PatternKV Insight outputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.io import atomic_write_json, atomic_write_text, write_csv


FINAL_FILES = [
    "pattern_gain_map.csv",
    "matching_oracle_gap.csv",
    "v_gate_confusion.csv",
    "attention_error.csv",
    "dynamic_pattern_utility.csv",
]


def git_commit() -> str:
    """Return current git commit."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read CSV rows if the file exists."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ensure_final_csvs(report_dir: Path) -> None:
    """Ensure required final CSVs exist, even before GPU observer waves run."""
    report_dir.mkdir(parents=True, exist_ok=True)
    for name in FINAL_FILES:
        path = report_dir / name
        if not path.exists():
            write_csv(path, [], ["status", "reason"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/insight_v1"))
    parser.add_argument("--v0-dir", type=Path, default=Path("reports/insight_v1/v0"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/insight_v1/final"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    ensure_final_csvs(args.report_dir)
    selected_path = args.v0_dir / "selected_samples.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else {"selected": []}
    lb_rows = read_csv_rows(args.v0_dir / "longbench_task_summary.csv")
    gsm_rows = read_csv_rows(args.v0_dir / "gsm8k_outcome_groups.csv")

    summary_lines = [
        "# PatternKV Insight Summary",
        "",
        f"git_commit: `{git_commit()}`",
        f"results_dir: `{args.results_dir}`",
        f"selected_samples: `{len(selected.get('selected', []))}`",
        "",
        "## Current Evidence",
        "",
        "V0 offline pairing and sample selection are available. Observer wave data is required before answering layer/head/K/V gain questions.",
        "",
        "## Required Questions",
        "",
        "1. Pattern收益主要来自K还是V？ Data insufficient until observer gain maps exist.",
        "2. Positive和Negative tasks的差异是什么？ V0 task deltas are available; layer/head evidence is pending.",
        "3. 哪些layer/head收益最高？ Data insufficient until Wave A/B observer output exists.",
        "4. 哪些layer/head Pattern有害？ Data insufficient until Pattern Gain Map exists.",
        "5. L2与min-max assignment mismatch是多少？ Data insufficient until oracle diagnostics run.",
        "6. Min-max距离MSE oracle还有多大gap？ Data insufficient until oracle diagnostics run.",
        "7. MSE收益是否与attention error收益一致？ Data insufficient until attention level is run.",
        "8. V gate的FP/FN是多少？ Data insufficient until V gate confusion output exists.",
        "9. Negative tasks的V gate FP是否更高？ Data insufficient until V gate confusion output exists.",
        "10. Dynamic Pattern是否真正被使用？ Data insufficient until dynamic utility output exists.",
        "11. 哪个创新方向得到最强证据？ Not decidable from V0 alone.",
        "12. 哪些结论数据不足，不能下结论？ All observer-dependent conclusions remain insufficient.",
        "",
        "## V0 LongBench Task Deltas",
        "",
    ]
    for row in lb_rows:
        summary_lines.append(f"- `{row.get('task')}`: PatternKV-KIVI `{row.get('pattern_minus_kivi')}`")
    summary_lines.extend(["", "## V0 GSM8K Outcome Groups", ""])
    for row in gsm_rows:
        summary_lines.append(f"- `{row.get('group')}`: `{row.get('count')}`")
    atomic_write_text(args.report_dir / "summary.md", "\n".join(summary_lines) + "\n")

    decision_lines = [
        "# Insight Decision Matrix",
        "",
        "No direction is marked strong candidate until observer/oracle Wave A/B data exists.",
        "",
        "| Direction | Status | Reason |",
        "| --- | --- | --- |",
        "| Range-aware Pattern Mining | insufficient_data | Needs L2/minmax mismatch and Range Regret. |",
        "| Quantization-aware Pattern Matching | insufficient_data | Needs minmax/MSE oracle mismatch and MSE Oracle Gap. |",
        "| Attention-aware Pattern Matching | insufficient_data | Needs attention-level run. |",
        "| Layer/Head Adaptive Allocation | insufficient_data | Needs layer/head gain map. |",
        "| Benefit-aware V Gating | insufficient_data | Needs V gate confusion. |",
        "| Selective Decode Pattern Update | insufficient_data | Needs dynamic utility statistics. |",
    ]
    atomic_write_text(args.report_dir / "decision_matrix.md", "\n".join(decision_lines) + "\n")

    impl = {
        "schema_version": "insight_v1.implementation_report",
        "pre_commit_head": git_commit(),
        "initial_branch": "repro/patternkv-paper-longbench-gsm8k-rerun",
        "working_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "initial_head": "c7324746d7447be532bd6bdbe0c8d58dd5e30c67",
        "current_main_head_at_start": "c7324746d7447be532bd6bdbe0c8d58dd5e30c67",
        "longbench_results_dir": "results/paper_repro_v2/longbench_21x50_8k_4090",
        "gsm8k_results_dir": "results/paper_repro_v2/gsm8k_full_2048",
        "results_dir": str(args.results_dir),
        "v0_dir": str(args.v0_dir),
        "selected_samples": len(selected.get("selected", [])),
        "standard_baseline_methods": ["fp16", "kivi_paper_g128", "patternkv_paper"],
        "tests": "pytest -q",
        "parity_status": "not_run",
        "wave_a_status": "not_run",
        "wave_b_status": "not_run",
        "pattern_gain_map_status": "not_run",
        "matching_oracle_gap_status": "not_run",
        "v_gate_status": "not_run",
        "dynamic_pattern_status": "not_run",
        "recommended_next_step": "Run Wave A basic+oracle after implementing model observer hook parity.",
        "observer_dependent_conclusions": "insufficient_data",
    }
    atomic_write_json(args.report_dir / "implementation_report.json", impl)
    atomic_write_text(
        args.report_dir / "implementation_report.md",
        "# PatternKV Insight Implementation Report\n\n"
        f"- initial_branch: `{impl['initial_branch']}`\n"
        f"- working_branch: `{impl['working_branch']}`\n"
        f"- initial_HEAD: `{impl['initial_head']}`\n"
        f"- pre_commit_HEAD_at_report_generation: `{impl['pre_commit_head']}`\n"
        f"- current_main_HEAD_at_start: `{impl['current_main_head_at_start']}`\n"
        f"- LongBench results: `{impl['longbench_results_dir']}`\n"
        f"- GSM8K results: `{impl['gsm8k_results_dir']}`\n"
        f"- selected_samples: `{impl['selected_samples']}`\n"
        "\n"
        "## Implemented\n\n"
        "- V0 offline pairing, baseline integrity audit, task summaries, length analysis, and fixed sample selection.\n"
        "- Config guardrails from `configs/standard_baselines.paper_v2.yaml`.\n"
        "- Passive insight scaffolding modules for collector, reference quantization, gain, oracle, gate, dynamic, and attention metrics.\n"
        "- Conservative `bench/bench_pattern_insight.py` entrypoint that validates `patternkv_paper` and fixed sample selection.\n"
        "\n"
        "## Not Completed\n\n"
        "- Model observer hook parity was not run.\n"
        "- Wave A/B GPU diagnostics were not run.\n"
        "- Pattern Gain Map, Matching Oracle Gap, V Gate Confusion, Attention-aware error, and Dynamic Pattern Utility remain data-insufficient.\n"
        "\n"
        "## Unsupported Conclusions\n\n"
        "No observer-dependent conclusion is supported by V0 alone. The current evidence only validates existing-result pairing and sample selection.\n"
        "\n"
        "## Recommended Next Step\n\n"
        "Implement the minimal model observer hook and run parity before Wave A.\n",
    )
    atomic_write_text(
        args.report_dir.parent / "observer_overhead.md",
        "# Observer Overhead\n\n"
        "Observer overhead has not been measured yet because Wave A/B GPU diagnostics were not run.\n\n"
        "These future timings must not be reported as formal PatternKV inference speed because diagnostics perform extra reference quantization and oracle traversal.\n",
    )
    print(json.dumps({"selected_samples": len(selected.get("selected", [])), "wave_a_status": "not_run", "wave_b_status": "not_run"}, sort_keys=True))


if __name__ == "__main__":
    main()
