#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench._longbench_scorer import score_subtask
from bench.longbench_config import METRIC_NAMES, SUBTASKS, expected_samples

METHODS = ("fp16", "kivi_paper_g128", "patternkv_paper", "causal_v4_25")
METHOD_LABELS = {
    "fp16": "FP16",
    "kivi_paper_g128": "KIVI",
    "patternkv_paper": "PatternKV",
    "causal_v4_25": "CAUSAL_V4_25",
}
TASK_CATEGORIES = {
    "narrativeqa": "SQA",
    "qasper": "SQA",
    "multifieldqa_en": "SQA",
    "multifieldqa_zh": "MQA",
    "hotpotqa": "MQA",
    "2wikimqa": "MQA",
    "musique": "MQA",
    "dureader": "MQA",
    "gov_report": "Summ.",
    "qmsum": "Summ.",
    "multi_news": "Summ.",
    "vcsum": "Summ.",
    "trec": "Few-shot",
    "triviaqa": "Few-shot",
    "samsum": "Few-shot",
    "lsht": "Few-shot",
    "passage_count": "Synth.",
    "passage_retrieval_en": "Synth.",
    "passage_retrieval_zh": "Synth.",
    "lcc": "Code",
    "repobench-p": "Code",
}
PAPER_TABLE1_LLAMA31_8B = {
    # PatternKV paper Table 1, LLaMA-3.1-8B-Instruct, LongBench.
    # Source: https://arxiv.org/html/2510.05176v1
    "fp16": {"MQA": 36.63, "SQA": 46.56, "Summ.": 25.54, "Few-shot": 61.16, "Synth.": 59.99, "Code": 59.42, "Avg": 46.59},
    "kivi_paper_g128": {"MQA": 34.86, "SQA": 43.96, "Summ.": 24.98, "Few-shot": 60.35, "Synth.": 54.43, "Code": 55.53, "Avg": 44.33},
    "patternkv_paper": {"MQA": 35.49, "SQA": 45.08, "Summ.": 25.12, "Few-shot": 60.58, "Synth.": 57.89, "Code": 56.55, "Avg": 45.33},
    "causal_v4_25": {"MQA": None, "SQA": None, "Summ.": None, "Few-shot": None, "Synth.": None, "Code": None, "Avg": None},
}


def read_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def pct(vals: list[float], q: float):
    if not vals:
        return None
    vals = sorted(vals)
    idx = min(len(vals) - 1, round((len(vals) - 1) * q))
    return vals[idx]


def row_score(task: str, successes: list[dict]) -> float | None:
    stored_scores = [float(r["score"]) for r in successes if r.get("score") is not None]
    if stored_scores:
        return round(statistics.mean(stored_scores), 2)
    preds = [str(r.get("prediction") or "") for r in successes]
    refs = [list(r.get("reference") or r.get("answers") or []) for r in successes]
    all_classes = None
    for r in successes:
        if r.get("all_classes"):
            all_classes = list(r["all_classes"])
            break
    return score_subtask(task, preds, refs, all_classes=all_classes)["score"] if successes else None


def summarize(base: Path, sample_limit_per_task: int | None = None) -> dict:
    task_rows = []
    all_scores = {m: [] for m in METHODS}
    category_scores = {m: {} for m in METHODS}
    for method in METHODS:
        for task in SUBTASKS:
            rows = read_rows(base / method / f"{task}.jsonl")
            expected = expected_samples(task)
            planned = min(expected, sample_limit_per_task) if sample_limit_per_task else expected
            successes = [r for r in rows if r.get("stop_reason") not in ("oom", "error") and r.get("prediction") is not None]
            score = row_score(task, successes)
            if score is not None and len(rows) >= planned:
                all_scores[method].append(score)
                category_scores[method].setdefault(TASK_CATEGORIES[task], []).append(score)
            eff = [int(r.get("input_tokens_after_special_tokens") or 0) for r in rows if r.get("input_tokens_after_special_tokens")]
            gen = [int(r.get("generated_tokens") or 0) for r in rows]
            wall = [float(r.get("wall_time_seconds") or 0) for r in rows if r.get("wall_time_seconds") is not None]
            peaks = [int(r.get("peak_memory_reserved_bytes") or 0) for r in rows if r.get("peak_memory_reserved_bytes")]
            task_rows.append(
                {
                    "method": method,
                    "task": task,
                    "category": TASK_CATEGORIES[task],
                    "planned": planned,
                    "completed": len(rows),
                    "success": len(successes),
                    "OOM": sum(1 for r in rows if r.get("stop_reason") == "oom"),
                    "error": sum(1 for r in rows if r.get("stop_reason") == "error"),
                    "was_truncated_count": sum(1 for r in rows if r.get("was_truncated")),
                    "mean_raw_input_tokens": statistics.mean([r["raw_input_tokens"] for r in rows if r.get("raw_input_tokens") is not None]) if rows else None,
                    "mean_effective_input_tokens": statistics.mean(eff) if eff else None,
                    "p95_effective_input_tokens": pct(eff, 0.95),
                    "mean_generated_tokens": statistics.mean(gen) if gen else None,
                    "p95_generated_tokens": pct(gen, 0.95),
                    "mean_wall_time": statistics.mean(wall) if wall else None,
                    "peak_memory_max": max(peaks) if peaks else None,
                    "official_metric": METRIC_NAMES[task],
                    "official_score_complete_case": score,
                }
            )
    macro = {m: (statistics.mean(v) if v else None) for m, v in all_scores.items()}
    categories = {
        m: {cat: (statistics.mean(vals) if vals else None) for cat, vals in scores.items()}
        for m, scores in category_scores.items()
    }
    paper_comparison = {}
    for method in METHODS:
        local = {**categories.get(method, {}), "Avg": macro[method]}
        paper = PAPER_TABLE1_LLAMA31_8B[method]
        paper_comparison[method] = {
            key: {
                "local_21x50_8k": local.get(key),
                "paper_table1": paper.get(key),
                "delta_local_minus_paper": (local.get(key) - paper.get(key)) if local.get(key) is not None and paper.get(key) is not None else None,
            }
            for key in ("MQA", "SQA", "Summ.", "Few-shot", "Synth.", "Code", "Avg")
        }
    return {
        "tasks": task_rows,
        "macro_average_complete_case": macro,
        "category_average_complete_case": categories,
        "deltas_complete_case": {
            "patternkv_minus_kivi": (macro["patternkv_paper"] - macro["kivi_paper_g128"]) if macro["patternkv_paper"] is not None and macro["kivi_paper_g128"] is not None else None,
            "patternkv_minus_fp16": (macro["patternkv_paper"] - macro["fp16"]) if macro["patternkv_paper"] is not None and macro["fp16"] is not None else None,
            "kivi_minus_fp16": (macro["kivi_paper_g128"] - macro["fp16"]) if macro["kivi_paper_g128"] is not None and macro["fp16"] is not None else None,
            "causal_minus_patternkv": (macro["causal_v4_25"] - macro["patternkv_paper"]) if macro["causal_v4_25"] is not None and macro["patternkv_paper"] is not None else None,
            "causal_minus_fp16": (macro["causal_v4_25"] - macro["fp16"]) if macro["causal_v4_25"] is not None and macro["fp16"] is not None else None,
        },
        "planned_total": sum(min(expected_samples(t), sample_limit_per_task) if sample_limit_per_task else expected_samples(t) for t in SUBTASKS) * len(METHODS),
        "completed_total": sum(r["completed"] for r in task_rows),
        "success_total": sum(r["success"] for r in task_rows),
        "oom_total": sum(r["OOM"] for r in task_rows),
        "error_total": sum(r["error"] for r in task_rows),
        "failed_tasks": [r for r in task_rows if r["OOM"] or r["error"] or r["completed"] < r["planned"]],
        "paper_table1_llama31_8b": PAPER_TABLE1_LLAMA31_8B,
        "paper_comparison": paper_comparison,
        "scoring_note": "Task scores are aggregated from per-row `score` when present. This preserves LongBench classification scoring for trec/lsht even if `all_classes` was not persisted in output rows.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results/paper_repro_v2/longbench_full_8k_4090"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports/paper_repro_v2/longbench_full_8k_4090"))
    ap.add_argument("--sample-limit-per-task", type=int)
    args = ap.parse_args()
    out = summarize(args.results_dir, sample_limit_per_task=args.sample_limit_per_task)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LongBench 8K 4090 Summary",
        "",
        f"Planned total: `{out['planned_total']}`",
        f"Completed total: `{out['completed_total']}`",
        f"Success total: `{out['success_total']}`",
        f"OOM total: `{out['oom_total']}`",
        f"Error total: `{out['error_total']}`",
        "",
        "## Macro Average Complete Case",
        "",
        "```json",
        json.dumps(out["macro_average_complete_case"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Category Average Complete Case",
        "",
        "```json",
        json.dumps(out["category_average_complete_case"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Paper Table 1 Comparison",
        "",
        "Source: `https://arxiv.org/html/2510.05176v1`",
        "",
        "| method | MQA | SQA | Summ. | Few-shot | Synth. | Code | Avg | Paper Avg | Delta Avg |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        cats = out["category_average_complete_case"][method]
        macro = out["macro_average_complete_case"][method]
        paper_avg = out["paper_table1_llama31_8b"][method]["Avg"]
        delta = out["paper_comparison"][method]["Avg"]["delta_local_minus_paper"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {cats.get('MQA')} | {cats.get('SQA')} | {cats.get('Summ.')} | {cats.get('Few-shot')} | {cats.get('Synth.')} | {cats.get('Code')} | {macro} | {paper_avg} | {delta} |"
        )
    lines.extend(
        [
            "",
            "## Task Scores",
            "",
            "| method | task | category | planned | completed | success | OOM | error | score |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in out["tasks"]:
        lines.append(f"| {METHOD_LABELS[r['method']]} | {r['task']} | {r['category']} | {r['planned']} | {r['completed']} | {r['success']} | {r['OOM']} | {r['error']} | {r['official_score_complete_case']} |")
    (args.report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
