#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "paper_quality_evidence_assembly_v1"
TABLES = OUT / "tables"


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write(path: str | Path, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pct(correct: int, total: int) -> float:
    return 100.0 * correct / total


def fmt_pct(value: float) -> str:
    return f"{value:.4f}%"


def rg_files(pattern: str) -> list[str]:
    cmd = [
        "rg",
        "-l",
        "-i",
        "--glob",
        "!reports/paper_quality_evidence_assembly_v1/**",
        pattern,
        "reports",
        "results",
        "docs",
        "scripts",
        "bench",
        "tests",
        "models",
        "releases",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return sorted([line for line in proc.stdout.splitlines() if line.strip()])


def load_causal_gsm8k() -> dict[str, Any]:
    rows = []
    for path in sorted((ROOT / "results/causal_v4_25_generalization_v1/gsm8k_full/causal_v4_25").glob("*.json")):
        rows.append(read_json(path))
    correct = sum(1 for row in rows if row.get("is_correct"))
    parse_success = sum(1 for row in rows if row.get("parsed_answer") is not None)
    length = sum(1 for row in rows if row.get("stop_reason") == "length")
    errors = sum(1 for row in rows if row.get("error"))
    return {
        "rows": rows,
        "n": len(rows),
        "correct": correct,
        "accuracy": pct(correct, len(rows)),
        "parse_success": parse_success,
        "length": length,
        "errors": errors,
        "source": "results/causal_v4_25_generalization_v1/gsm8k_full/causal_v4_25/",
    }


def load_baseline_gsm8k_raw(method: str) -> list[dict[str, Any]]:
    root = ROOT / "results/paper_repro_v2/gsm8k_full_2048" / method
    return [read_json(path) for path in sorted(root.glob("*.json"))]


def gsm8k_pair(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> int:
        return int(row.get("sample_index", row.get("problem_id", -1)))

    left = {key(row): bool(row.get("is_correct")) for row in left_rows}
    right = {key(row): bool(row.get("is_correct")) for row in right_rows}
    keys = sorted(set(left) & set(right))
    b = sum(1 for k in keys if left[k] and not right[k])
    c = sum(1 for k in keys if (not left[k]) and right[k])
    both_correct = sum(1 for k in keys if left[k] and right[k])
    both_wrong = sum(1 for k in keys if (not left[k]) and (not right[k]))
    n_disc = b + c
    if n_disc == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n_disc, i) for i in range(0, min(b, c) + 1)) / (2**n_disc)
        p_value = min(1.0, 2.0 * tail)
    return {
        "paired_n": len(keys),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "left_correct_right_wrong": b,
        "left_wrong_right_correct": c,
        "paired_accuracy_difference": (b - c) / len(keys),
        "mcnemar_exact_p": p_value,
    }


def load_causal_longbench() -> dict[str, Any]:
    root = ROOT / "results/causal_v4_25_generalization_v1/longbench_full/causal_v4_25"
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    errors: dict[str, int] = {}
    ooms: dict[str, int] = {}
    for path in sorted(root.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        task = path.stem
        counts[task] = len(rows)
        scores[task] = sum(float(row.get("score", 0.0)) for row in rows) / len(rows)
        errors[task] = sum(1 for row in rows if row.get("exception_type"))
        ooms[task] = sum(1 for row in rows if row.get("oom_stage"))
    return {
        "tasks": len(scores),
        "samples_per_task": sorted(set(counts.values())),
        "total": sum(counts.values()),
        "errors": sum(errors.values()),
        "ooms": sum(ooms.values()),
        "macro": sum(scores.values()) / len(scores),
        "task_scores": scores,
        "source": "results/causal_v4_25_generalization_v1/longbench_full/causal_v4_25/",
    }


def latex_escape(text: str) -> str:
    return text.replace("%", "\\%").replace("_", "\\_")


def make_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def make_latex_table(headers: list[str], rows: list[list[Any]], caption: str, label: str) -> str:
    cols = "l" * len(headers)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(latex_escape(h) for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(str(item)) for item in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    release = subprocess.check_output(["git", "rev-parse", "release/causal-v4-25-system-final"], cwd=ROOT, text=True).strip()

    aime = read_json(ROOT / "reports/aime24_full_causal25_quality_4gpu/full_aime24_quality_summary.json")
    bootstrap = read_json(ROOT / "reports/aime24_full_causal25_quality_4gpu/paired_bootstrap.json")
    gen_cfg = read_json(ROOT / "reports/aime24_full_causal25_quality_4gpu/generation_config.json")
    method_cfg = read_json(ROOT / "reports/aime24_full_causal25_quality_4gpu/method_configs.json")
    gsm_base = read_json(ROOT / "reports/paper_repro_v2/gsm8k_full_2048/summary_4gpu_sequence.json")
    lb_base = read_json(ROOT / "reports/paper_repro_v2/longbench_21x50_8k_4090/summary.json")
    budget = read_json(ROOT / "reports/aime24_value_capacity_budget_3090/budget_response_summary.json")
    value = read_json(ROOT / "reports/aime24_routing_vdirection_3090/routing_vdirection_summary.json")
    bits = read_json(ROOT / "releases/causal_v4_25_aime24_v1/bit_accounting.json")

    aime_acc = {row["method"]: row for row in aime["accuracy"]}
    gsm_causal = load_causal_gsm8k()
    gsm_rows = {
        "FP16": {"correct": gsm_base["methods"]["fp16"]["correct"], "total": 1319, "accuracy": gsm_base["methods"]["fp16"]["strict_accuracy"]},
        "KIVI": {"correct": gsm_base["methods"]["kivi_paper_g128"]["correct"], "total": 1319, "accuracy": gsm_base["methods"]["kivi_paper_g128"]["strict_accuracy"]},
        "PatternKV": {"correct": gsm_base["methods"]["patternkv_paper"]["correct"], "total": 1319, "accuracy": gsm_base["methods"]["patternkv_paper"]["strict_accuracy"]},
        "CAUSAL-V4@25%": {"correct": gsm_causal["correct"], "total": gsm_causal["n"], "accuracy": gsm_causal["accuracy"]},
    }
    causal_gsm_raw = gsm_causal["rows"]
    gsm_pairs = {
        "CAUSAL_vs_FP16": gsm8k_pair(causal_gsm_raw, load_baseline_gsm8k_raw("fp16")),
        "CAUSAL_vs_KIVI": gsm8k_pair(causal_gsm_raw, load_baseline_gsm8k_raw("kivi_paper_g128")),
        "CAUSAL_vs_PatternKV": gsm8k_pair(causal_gsm_raw, load_baseline_gsm8k_raw("patternkv_paper")),
    }

    lb_causal = load_causal_longbench()
    lb_macro = {
        "FP16": lb_base["macro_average_complete_case"]["fp16"],
        "KIVI": lb_base["macro_average_complete_case"]["kivi_paper_g128"],
        "PatternKV": lb_base["macro_average_complete_case"]["patternkv_paper"],
        "CAUSAL-V4@25%": lb_causal["macro"],
    }

    quality_rows = [
        ["FP16", "16-bit KV", "45/90 (50.00%)", f"1029/1319 ({fmt_pct(gsm_rows['FP16']['accuracy'])})", f"{lb_macro['FP16']:.4f}", "Full precision reference."],
        ["KIVI", "2.25-bit quantized-region", "not run in canonical AIME24 four-method task-quality table", f"909/1319 ({fmt_pct(gsm_rows['KIVI']['accuracy'])})", f"{lb_macro['KIVI']:.4f}", "Canonical baseline for GSM8K/LongBench."],
        ["PatternKV", "2.25-bit quantized-region", "32/90 (35.56%)", f"973/1319 ({fmt_pct(gsm_rows['PatternKV']['accuracy'])})", f"{lb_macro['PatternKV']:.4f}", "AIME24 row is Pattern Base."],
        ["Random-25%", "~2.500488 bit/KV element", "36/90 (40.00%)", "not run", "not run", "Same-budget AIME24 control."],
        ["CAUSAL-V4@25%", "~2.500488 bit/KV element", "45/90 (50.00%)", f"1041/1319 ({fmt_pct(gsm_rows['CAUSAL-V4@25%']['accuracy'])})", f"{lb_macro['CAUSAL-V4@25%']:.4f}", "Selective heterogeneous V2/V4 method."],
    ]
    write(TABLES / "quality_main_table.md", "# Provisional Quality Main Table\n\n" + make_markdown_table(["Method", "Effective KV Bits", "AIME24", "GSM8K", "LongBench", "Notes"], quality_rows))
    write(TABLES / "quality_main_table.tex", make_latex_table(["Method", "Effective KV Bits", "AIME24", "GSM8K", "LongBench", "Notes"], quality_rows, "Existing canonical quality evidence.", "tab:quality-main"))

    selector_rows = [
        ["FP16", "reference", "45/90", "50.00%", "CANONICAL"],
        ["Pattern Base", "base_v2", "32/90", "35.56%", "CANONICAL"],
        ["Random-25%", "random_v4", "36/90", "40.00%", "CANONICAL"],
        ["Importance-Only-25%", "not implemented as separate selector", "NOT_RUN", "NOT_RUN", "MISSING"],
        ["Error-Reduction-Only-25%", "not implemented as separate selector", "NOT_RUN", "NOT_RUN", "MISSING"],
        ["CAUSAL-25%", "causal_v4", "45/90", "50.00%", "CANONICAL"],
    ]
    write(TABLES / "aime24_selector_ablation_status.md", "# AIME24 Selector Ablation Status\n\n" + make_markdown_table(["Method", "Selector", "Correct", "Accuracy", "Status"], selector_rows))

    matrix_rows = [
        ["AIME24", "CANONICAL", "NOT_RUN", "CANONICAL", "GSM/LongBench baselines only", "CANONICAL Pattern Base", "CANONICAL", "main Long-CoT evidence"],
        ["AIME25", "NOT_RUN", "NOT_RUN", "NOT_RUN", "NOT_RUN", "NOT_RUN", "NOT_RUN", "P0 gap"],
        ["AMC24", "NOT_RUN", "NOT_RUN", "NOT_RUN", "NOT_RUN", "NOT_RUN", "NOT_RUN", "P1 gap"],
        ["AMC23", "NOT_RUN", "NOT_RUN", "NOT_RUN", "NOT_RUN", "NOT_RUN", "NOT_RUN", "P2/optional"],
        ["GPQA", "NOT_RUN", "NOT_RUN", "NOT_RUN", "NOT_RUN", "NOT_RUN", "NOT_RUN", "P2/optional"],
        ["MATH-like historical", "NOT_COMPARABLE", "NOT_COMPARABLE", "NOT_COMPARABLE", "NOT_COMPARABLE", "NOT_COMPARABLE", "NOT_COMPARABLE", "exclude unless CAUSAL protocol exists"],
    ]
    write(TABLES / "long_cot_benchmark_matrix.md", "# Long-CoT Benchmark Matrix\n\n" + make_markdown_table(["Benchmark", "Llama CAUSAL", "Qwen CAUSAL", "FP16", "KIVI", "PatternKV", "Random", "Paper Role"], matrix_rows))

    canonical = {
        "classification": "PAPER_QUALITY_EVIDENCE_ASSEMBLY_V1_SUPPORTED",
        "head_at_assembly": head,
        "frozen_system_release": release,
        "new_gpu_experiments_run": False,
        "aime24": {
            "methods": {m: {"correct": r["total_correct"], "total": r["total"], "accuracy": r["mean_accuracy"]} for m, r in aime_acc.items()},
            "paired_bootstrap": bootstrap,
            "generation_config": gen_cfg,
            "method_configs": method_cfg,
        },
        "gsm8k": {"methods": gsm_rows, "paired": gsm_pairs},
        "longbench": {"macro_average": lb_macro, "causal": lb_causal},
        "effective_bits": bits,
        "budget_response": budget,
        "value_path": value,
    }
    write_json(OUT / "canonical_quality_numbers.json", canonical)
    with (OUT / "canonical_quality_numbers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["benchmark", "method", "metric", "value", "source"])
        for method, row in aime_acc.items():
            writer.writerow(["AIME24", method, "correct", row["total_correct"], "reports/aime24_full_causal25_quality_4gpu/full_aime24_quality_summary.json"])
            writer.writerow(["AIME24", method, "accuracy", row["mean_accuracy"], "reports/aime24_full_causal25_quality_4gpu/full_aime24_quality_summary.json"])
        for method, row in gsm_rows.items():
            writer.writerow(["GSM8K", method, "correct", row["correct"], "reports/paper_repro_v2/gsm8k_full_2048/summary_4gpu_sequence.json or results/causal_v4_25_generalization_v1/gsm8k_full/causal_v4_25/"])
            writer.writerow(["GSM8K", method, "accuracy_percent", row["accuracy"], "reports/paper_repro_v2/gsm8k_full_2048/summary_4gpu_sequence.json or results/causal_v4_25_generalization_v1/gsm8k_full/causal_v4_25/"])
        for method, score in lb_macro.items():
            writer.writerow(["LongBench", method, "macro_average", score, "reports/paper_repro_v2/longbench_21x50_8k_4090/summary.json or results/causal_v4_25_generalization_v1/longbench_full/causal_v4_25/"])

    source_rows = [
        ["AIME24 accuracy", "reports/aime24_full_causal25_quality_4gpu/full_aime24_quality_summary.json", "CANONICAL", "DeepSeek-R1-Distill-Llama-8B; 30 questions x 3 seeds x 4 methods"],
        ["AIME24 paired bootstrap", "reports/aime24_full_causal25_quality_4gpu/paired_bootstrap.json", "CANONICAL", "Question-level bootstrap, 10000 resamples"],
        ["AIME24 protocol", "reports/aime24_full_causal25_quality_4gpu/generation_config.json", "CANONICAL", "DeepSeek-R1 prompt, temperature 0.6, top_p 0.95, max_new_tokens 32768"],
        ["AIME24 raw provenance", "reports/aime24_full_causal25_quality_4gpu/raw_generation_manifest.json", "REFERENCE", "Raw generations are local/ignored; compact committed records exist"],
        ["GSM8K baselines", "reports/paper_repro_v2/gsm8k_full_2048/summary_4gpu_sequence.json", "CANONICAL", "Llama-3.1-8B-Instruct, full 1319-test split"],
        ["GSM8K CAUSAL", "results/causal_v4_25_generalization_v1/gsm8k_full/causal_v4_25/", "CANONICAL_RAW", "Committed per-sample JSON, full 1319-test split"],
        ["LongBench baselines", "reports/paper_repro_v2/longbench_21x50_8k_4090/summary.json", "CANONICAL_FINAL_BASELINE", "21 tasks x 50, 8K cap"],
        ["LongBench CAUSAL", "results/causal_v4_25_generalization_v1/longbench_full/causal_v4_25/", "CANONICAL_RAW", "21 tasks x 50, 8K cap"],
        ["Budget/effective bits", "reports/aime24_value_capacity_budget_3090/budget_response_summary.json", "CANONICAL_FORENSIC", "Budgets 0, 12.5%, 25%, 50%, 100%"],
        ["Bit accounting", "releases/causal_v4_25_aime24_v1/bit_accounting.json", "CANONICAL", "Formal payload-and-metadata project metric"],
        ["Error accumulation", "reports/aime24_pseudodecode_3090_8gpu/pseudodecode_accumulation_report.md", "CANONICAL_FORENSIC", "Matched checkpoints 128-4096"],
        ["Value-path forensic", "reports/aime24_routing_vdirection_3090/routing_vdirection_summary.json", "CANONICAL_FORENSIC", "6/6 tasks value-dominant"],
        ["System evidence reference", "reports/paper_system_table_and_figure_assembly_v1/", "FROZEN_REFERENCE_ONLY", "No system regeneration in this task"],
    ]
    write(OUT / "source_manifest.md", "# Source Manifest\n\n" + make_markdown_table(["Evidence", "Source Path", "Status", "Scope"], source_rows))

    claim_rows = [
        ["Q1", "CAUSAL-V4@25% preserves Long-CoT quality better than Pattern Base.", "PRIMARY_SUPPORTED", "AIME24 45/90 vs 32/90; CAUSAL-BASE bootstrap CI positive."],
        ["Q2", "CAUSAL-V4@25% preserves Long-CoT quality better than same-budget Random.", "SUPPORTED_WITH_SCOPE", "AIME24 aggregate 45/90 vs 36/90; bootstrap CI crosses zero."],
        ["Q3", "CAUSAL-V4@25% can match FP16 aggregate AIME24 accuracy in the tested protocol.", "SUPPORTED_WITH_SCOPE", "Both 45/90; no significance claim."],
        ["Q4", "CAUSAL improves GSM8K quality over PatternKV and KIVI.", "SUPPORTED_WITH_SCOPE", "Aggregate full split: +5.1554 pp vs Pattern, +10.0076 pp vs KIVI."],
        ["Q5", "CAUSAL improves LongBench average over PatternKV and KIVI while remaining close to FP16.", "SUPPORTED_WITH_SCOPE", "21x50 8K macro: +0.8538 vs Pattern, +1.2514 vs KIVI, -0.8205 vs FP16."],
        ["Q6", "The 25% V4 budget is a useful quality/bit-efficiency operating point.", "PARTIAL", "Forensic budget curve shows saturation/utility at 25%; not a full task-quality sweep."],
        ["Q7", "Long-CoT quantization error accumulates recursively through persistent KV state.", "SUPPLEMENTARY", "Matched pseudo-decode accumulation supports this mechanism over tested AIME24 cohort."],
        ["Q8", "Value-path propagation dominates routing-only propagation in the tested forensic regime.", "SUPPLEMENTARY", "6/6 tasks value-dominant; scoped to this regime."],
        ["Q9", "CAUSAL benefit is not merely same number of INT4 values randomly allocated.", "SUPPORTED_WITH_SCOPE", "AIME24 same-budget Random lags aggregate; CI for CAUSAL-Random crosses zero."],
        ["Q10", "CAUSAL operates at approximately ~2.500488 effective KV bits under project accounting.", "PRIMARY_SUPPORTED", "Bit accounting release and AIME24 bit_cost agree."],
    ]
    write(OUT / "claim_inventory.md", "# Claim Inventory\n\n" + make_markdown_table(["ID", "Claim", "Classification", "Evidence"], claim_rows))
    write(OUT / "claim_audit.md", "# Claim Audit\n\n" + make_markdown_table(["Claim", "Status", "Paper-Safe Wording"], [
        ["AIME24 CAUSAL = FP16 aggregate", "SUPPORTED_WITH_SCOPE", "Matches FP16 aggregate accuracy in tested three-seed AIME24."],
        ["CAUSAL > Pattern Base AIME24", "SUPPORTED", "Supported by canonical paired bootstrap CI."],
        ["CAUSAL > Random AIME24", "SUPPORTED_WITH_SCOPE", "Aggregate advantage; do not call significant at 95%."],
        ["CAUSAL > FP16 GSM8K", "PARTIAL", "Aggregate numerical result only unless further stats are used."],
        ["CAUSAL > Pattern/KIVI GSM8K", "SUPPORTED_WITH_SCOPE", "Aggregate full-test result and offline paired counts support direction."],
        ["CAUSAL > Pattern/KIVI LongBench", "SUPPORTED_WITH_SCOPE", "Aggregate over tested 21x50 8K setup."],
        ["Value-path universally dominates", "NOT_SUPPORTED", "Only tested Long-CoT forensic regime supports dominance."],
        ["CAUSAL universally matches FP16", "NOT_SUPPORTED", "AIME24 aggregate only; LongBench is below FP16."],
        ["25% universally optimal", "NOT_SUPPORTED", "25% is a supported operating point, not universal optimum."],
    ]))

    write(OUT / "aime24_evidence.md", f"""# AIME24 Evidence

Status: CANONICAL_PRIMARY.

Source: `reports/aime24_full_causal25_quality_4gpu/full_aime24_quality_summary.json`.

Protocol: `{gen_cfg['prompt_protocol']}`, `do_sample={gen_cfg['do_sample']}`, temperature `{gen_cfg['temperature']}`, top_p `{gen_cfg['top_p']}`, max_new_tokens `{gen_cfg['max_new_tokens']}`, seeds `42, 43, 44`, 30 questions, paired by problem and seed.

| Method | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| FP16 | 45 | 90 | 50.00% |
| Pattern Base | 32 | 90 | 35.56% |
| Random-25% | 36 | 90 | 40.00% |
| CAUSAL-V4@25% | 45 | 90 | 50.00% |

Paired bootstrap:

- CAUSAL - Random: mean `{bootstrap['causal_minus_random']['mean_delta']}`, CI95 `[{bootstrap['causal_minus_random']['ci95_low']}, {bootstrap['causal_minus_random']['ci95_high']}]`; aggregate advantage, CI crosses zero.
- CAUSAL - Base: mean `{bootstrap['causal_minus_base']['mean_delta']}`, CI95 `[{bootstrap['causal_minus_base']['ci95_low']}, {bootstrap['causal_minus_base']['ci95_high']}]`; CI is positive.

Paper-safe language: CAUSAL matches FP16 aggregate accuracy in the tested three-seed AIME24 evaluation. Do not infer statistical equivalence from equal aggregate accuracy.
""")

    aime25_files = rg_files("aime25|aime_2025|AIME25")
    write(OUT / "aime25_inventory.md", "# AIME25 Inventory\n\n" + make_markdown_table(["Artifact Type", "Status", "Evidence"], [
        ["Implementation/scripts", "IMPLEMENTATION_ONLY_OR_GUARDRAIL_REFERENCES", ", ".join(aime25_files[:8]) if aime25_files else "No files found"],
        ["Smoke results", "MISSING", "No AIME25 smoke result directory or summary found."],
        ["Partial results", "MISSING", "No committed AIME25 sample outputs found."],
        ["Full results", "MISSING", "No canonical full AIME25 report found."],
        ["Canonical status", "MISSING", "AIME25 remains a P0 gap after repository search."],
    ]))

    long_cot_files = rg_files("AMC23|AMC24|GPQA|MATH500|MATH-500|Olympiad|long[_-]cot|reasoning")
    long_cot_hit_text = "\n".join(f"- `{path}`" for path in long_cot_files[:40]) if long_cot_files else "- No matching artifacts found."
    write(OUT / "long_cot_benchmark_inventory.md", "# Long-CoT Benchmark Inventory\n\n" + make_markdown_table(["Benchmark Family", "Status", "Evidence"], [
        ["AIME24", "CANONICAL", "Full four-method CAUSAL/Random/Base/FP16 evidence exists."],
        ["AIME25", "MISSING", "Only not-run references/guardrails found."],
        ["AMC23", "MISSING", "No current CAUSAL quality evidence found."],
        ["AMC24", "MISSING", "No current CAUSAL quality evidence found."],
        ["GPQA", "MISSING", "Only system exclusion/not-run references found."],
        ["MATH/MATH500/Olympiad", "NOT_COMPARABLE_OR_MISSING", "Search found no current CAUSAL protocol result; historical unrelated artifacts are excluded."],
    ]) + "\n\nSearch hits sampled:\n\n" + long_cot_hit_text)

    write(OUT / "gsm8k_evidence.md", "# GSM8K Evidence\n\n" + make_markdown_table(["Method", "Correct", "Total", "Accuracy"], [[m, r["correct"], r["total"], fmt_pct(r["accuracy"])] for m, r in gsm_rows.items()]) + f"""

Status: CANONICAL_WITH_CAUSAL_RAW_AGGREGATION.

Canonical baseline source: `reports/paper_repro_v2/gsm8k_full_2048/summary_4gpu_sequence.json`.
Canonical CAUSAL source: `results/causal_v4_25_generalization_v1/gsm8k_full/causal_v4_25/`.

Deltas:

- CAUSAL - FP16: `{gsm_rows['CAUSAL-V4@25%']['accuracy'] - gsm_rows['FP16']['accuracy']:.4f}` percentage points.
- CAUSAL - PatternKV: `{gsm_rows['CAUSAL-V4@25%']['accuracy'] - gsm_rows['PatternKV']['accuracy']:.4f}` percentage points.
- CAUSAL - KIVI: `{gsm_rows['CAUSAL-V4@25%']['accuracy'] - gsm_rows['KIVI']['accuracy']:.4f}` percentage points.

Offline paired counts were computed from committed per-sample outputs only:

```json
{json.dumps(gsm_pairs, indent=2, sort_keys=True)}
```

Do not claim significance beyond these offline paired statistics without a predeclared statistical protocol.
""")

    write(OUT / "longbench_evidence.md", "# LongBench Evidence\n\n" + make_markdown_table(["Method", "Macro Average"], [[m, f"{v:.4f}"] for m, v in lb_macro.items()]) + f"""

Status: CANONICAL_FINAL_CAUSAL_21TASK for the 8K-capped 21 x 50 setup.

Baseline source: `reports/paper_repro_v2/longbench_21x50_8k_4090/summary.json`.
CAUSAL source: `results/causal_v4_25_generalization_v1/longbench_full/causal_v4_25/`.

Scope: `{lb_causal['tasks']}` tasks, `{lb_causal['samples_per_task'][0]}` samples per task, `{lb_causal['total']}` total samples per method for CAUSAL, errors `{lb_causal['errors']}`, OOM `{lb_causal['ooms']}`.

Deltas:

- CAUSAL - FP16: `{lb_macro['CAUSAL-V4@25%'] - lb_macro['FP16']:.4f}`.
- CAUSAL - PatternKV: `{lb_macro['CAUSAL-V4@25%'] - lb_macro['PatternKV']:.4f}`.
- CAUSAL - KIVI: `{lb_macro['CAUSAL-V4@25%'] - lb_macro['KIVI']:.4f}`.

FULL_OFFICIAL_LONGBENCH_EXISTS: `false`.

This is not a strict full official LongBench split run; it is an 8K-capped 21-task x 50-sample reproduction.
""")
    write(OUT / "longbench_experiment_lineage.md", "# LongBench Experiment Lineage\n\n" + make_markdown_table(["Artifact", "Classification", "Reason"], [
        ["reports/paper_repro_v2/longbench_21x50_8k_4090/", "CANONICAL_FINAL_BASELINES", "21 tasks x 50, final baseline source."],
        ["results/causal_v4_25_generalization_v1/longbench_full/causal_v4_25/", "CANONICAL_FINAL_CAUSAL_21TASK", "21 tasks x 50 committed raw CAUSAL outputs."],
        ["reports/longbench_fp16_patternkv_kivi_8x50.md", "EARLY_8TASK_BASELINE", "Superseded by 21-task final baseline."],
        ["reports/kivi_official_longbench_8x50.md", "SUPERSEDED_OR_LEGACY", "Legacy KIVI-only/8-task result; not final four-method evidence."],
        ["reports/kivi_invalid_axis0_1_20260803_171756/", "INVALID", "Known invalid axis KIVI lineage; excluded."],
    ]))

    write(OUT / "same_budget_random_evidence.md", f"""# Same-Budget Random Evidence

Status: SUPPORTED_WITH_SCOPE.

`RANDOM_V4_25` and `CAUSAL_V4_25` use the same formal bit budget:

```json
{json.dumps(aime['bit_cost'], indent=2, sort_keys=True)}
```

AIME24 aggregate: Random-25% `36/90` vs CAUSAL-V4@25% `45/90`.
The paired bootstrap CI for CAUSAL - Random crosses zero, so this is an aggregate advantage and same-budget control, not a 95% significance claim.
""")

    selector_files = rg_files("importance[_ -]?only|error[_ -]?only|score[_ -]?only|quant[_ -]?error[_ -]?only|delta[_ -]?error|local[_ -]?error|selector ablation")
    selector_hit_text = "\n".join(f"- `{path}`" for path in selector_files[:30]) if selector_files else "- No separate importance-only/error-only selector ablation artifacts found."
    write(OUT / "selector_ablation_inventory.md", "# Selector Ablation Inventory\n\n" + make_markdown_table(["Variant", "Implementation Exists", "Test Exists", "Quality Result Exists", "Canonical Result Exists", "Status"], [
        ["Random-25%", "yes", "yes", "yes", "yes", "DIRECT_CONTROL"],
        ["CAUSAL-25%", "yes", "yes", "yes", "yes", "PRIMARY_METHOD"],
        ["Importance-Only-25%", "no separate selector found", "no", "no", "no", "MISSING_P0"],
        ["Error-Reduction-Only-25%", "no separate selector found", "no", "no", "no", "MISSING_P0"],
        ["Oracle/Future", "yes, forensic only", "yes", "forensic only", "no task-quality canonical", "SUPPLEMENTARY_NOT_DEPLOYABLE"],
    ]) + "\n\nSelector implementation currently normalizes to `base_v2`, `all_v2`, `all_v4`, `random_v4`, `causal_v4`, and `oracle_v4`. Search hits sampled:\n\n" + selector_hit_text)

    write(OUT / "budget_sweep_audit.md", f"""# Budget Sweep Audit

Status: BUDGET_KNEE_SUPPORTED_WITH_LIMITED_POINTS.

Existing points: `0.0`, `0.125`, `0.25`, `0.5`, `1.0`.

Canonical source: `reports/aime24_value_capacity_budget_3090/budget_response_summary.json`.

Budget classification: `{budget['budget_response_classification']}`.
Capacity saturation budget: `{budget['capacity_saturation_budget']}`.

These points support 25% as a useful operating point in the forensic budget curve. They do not establish a universal optimum or a full task-quality sweep across all budgets.
""")

    write(OUT / "effective_bit_accounting.md", f"""# Effective Bit Accounting

Canonical source: `releases/causal_v4_25_aime24_v1/bit_accounting.json`.

Formal project metric:

- Pattern Base: `{bits['level_a_formal_budget_metric']['PATTERN_BASE']}` bit/KV element.
- Random-25%: `{bits['level_a_formal_budget_metric']['RANDOM_V4_25']}` bit/KV element.
- CAUSAL-V4@25%: `{bits['level_a_formal_budget_metric']['CAUSAL_V4_25']}` bit/KV element.
- Same-bit control valid: `{bits['level_a_formal_budget_metric']['SAME_BIT_CONTROL_VALID']}`.

Scope: payload-and-metadata effective quantization budget. This is not physical Python tensor storage, allocator memory, sink/recent full precision storage, or whole-GPU memory.
""")

    write(OUT / "long_cot_error_mechanism.md", "# Long-CoT Error Mechanism\n\n" + make_markdown_table(["Mechanism Claim", "Status", "Evidence"], [
        ["Persistent KV quantization error exists", "SUPPORTED_WITH_SCOPE", "Matched pseudo/static formal audit over checkpoints 128-4096."],
        ["Autoregressive recursion accumulates error", "SUPPORTED_WITH_SCOPE", "Pattern S16 pseudo degradation grows beyond static degradation after 512 tokens."],
        ["Early error acts as accumulation seed", "SUPPLEMENTARY", "Sink16 reduces AUC for Pattern/KIVI in tested cohort."],
        ["Universal context behavior", "NOT_SUPPORTED", "Extended 8192/16384 matched static rows were hardware-limited and excluded."],
    ]))
    write(OUT / "value_path_forensic.md", f"""# Value-Path Forensic Evidence

Canonical source: `reports/aime24_routing_vdirection_3090/routing_vdirection_summary.json`.

- Actual attention-output AUC median: `{value['actual_attention_output_auc_median']}`.
- Routing-only output AUC median: `{value['routing_only_output_auc_median']}`.
- Value-only output AUC median: `{value['value_only_output_auc_median']}`.
- Value-dominant tasks: `{value['value_dominant_tasks']}/{value['task_count']}`.
- Routing-dominant tasks: `{value['routing_dominant_tasks']}/{value['task_count']}`.
- Classification: `{value['recursive_propagation_classification']}`.

Paper-safe claim: Value-path propagation dominated the tested Long-CoT error regime. Do not claim V is always more important than K.
""")

    qwen_files = rg_files("Qwen|Qwen2|Qwen2.5|Qwen3|DeepSeek-R1-Distill-Qwen")
    qwen_hit_text = "\n".join(f"- `{path}`" for path in qwen_files[:40]) if qwen_files else "- No Qwen CAUSAL quality artifacts found."
    write(OUT / "model_generalization_inventory.md", "# Model Generalization Inventory\n\n" + make_markdown_table(["Model", "CAUSAL Implementation", "Semantic Gate", "AIME24", "AIME25", "GSM8K", "LongBench", "Canonical Quality Evidence"], [
        ["DeepSeek-R1-Distill-Llama-8B", "yes", "yes", "canonical", "not run", "not applicable", "not applicable", "AIME24 canonical Long-CoT"],
        ["Llama-3.1-8B-Instruct", "yes", "yes", "not run", "not run", "canonical CAUSAL raw", "canonical CAUSAL raw", "GSM8K and LongBench"],
        ["Qwen / DeepSeek-Qwen family", "not established", "not found", "not run", "not run", "not run", "not run", "MISSING second-backbone evidence"],
    ]) + "\n\nQwen search hits sampled:\n\n" + qwen_hit_text)

    baseline_files = rg_files("ZipCache|SKVQ|OTT|KVQuant|AQUA-KV|KVarN|KIVI|PatternKV")
    write(OUT / "baseline_inventory.md", "# Baseline Inventory\n\n" + make_markdown_table(["Baseline", "Classification", "Paper Role"], [
        ["FP16", "DIRECT_PRIMARY_BASELINE", "Reference for AIME24/GSM8K/LongBench."],
        ["KIVI", "DIRECT_PRIMARY_BASELINE", "Canonical GSM8K/LongBench baseline; system baseline."],
        ["PatternKV / Pattern Base", "DIRECT_PRIMARY_BASELINE", "Compressed baseline and AIME24 base row."],
        ["Random-25%", "DIRECT_PRIMARY_BASELINE", "Same-budget AIME24 control."],
        ["ZipCache", "MISSING", "No current CAUSAL protocol result found."],
        ["SKVQ", "MISSING", "No current CAUSAL protocol result found."],
        ["OTT", "MISSING", "No current CAUSAL protocol result found."],
        ["KVQuant", "MISSING", "No current CAUSAL protocol result found."],
        ["AQUA-KV", "MISSING", "No current CAUSAL protocol result found."],
        ["KVarN", "RELATED_HISTORICAL_ONLY", "Do not count as CAUSAL evidence without current protocol participation."],
    ]) + "\n\nBaseline search hits sampled:\n\n" + "\n".join(f"- `{path}`" for path in baseline_files[:40]))

    write(OUT / "statistics_inventory.md", "# Statistics Inventory\n\n" + make_markdown_table(["Benchmark", "Status", "Available Statistics", "Additional Offline Stats Needed"], [
        ["AIME24", "STATISTICS_COMPLETE_FOR_PRIMARY_CLAIMS", "Question-level paired bootstrap for CAUSAL-Random and CAUSAL-Base.", "None for current claims."],
        ["GSM8K", "PARTIAL_WITH_NEW_OFFLINE_COUNTS", "Baseline paired counts; CAUSAL-vs-baseline McNemar counts computed from committed raw outputs.", "Predeclare if using p-values in paper."],
        ["LongBench", "PARTIAL", "Task-level aggregates and raw per-sample scores exist.", "Optional bootstrap over tasks/samples if paper needs intervals."],
        ["Budget sweep", "PARTIAL", "Forensic bootstrap CIs for selector advantage in budget summary.", "No task-quality budget sweep stats."],
        ["Random", "STATISTICS_COMPLETE_FOR_AIME24_SCOPE", "Same-budget AIME24 paired bootstrap vs CAUSAL.", "None unless expanding to AIME25."],
    ]))

    write(OUT / "raw_evidence_inventory.md", "# Raw Evidence Inventory\n\n" + make_markdown_table(["Evidence", "Availability", "Path", "Notes"], [
        ["AIME24 full quality", "COMPACT_PAIRED_RECORDS_COMMITTED", "reports/aime24_full_causal25_quality_4gpu/sample_results_compact.csv and .jsonl.gz", "Raw generations manifest points to ignored local text files."],
        ["GSM8K baselines", "RAW_COMMITTED", "results/paper_repro_v2/gsm8k_full_2048/", "Full per-sample JSON for 3 baselines."],
        ["GSM8K CAUSAL", "RAW_COMMITTED", "results/causal_v4_25_generalization_v1/gsm8k_full/causal_v4_25/", "Full per-sample JSON."],
        ["LongBench baselines", "RAW_COMMITTED", "results/paper_repro_v2/longbench_21x50_8k_4090/", "Per-task JSONL for 3 baselines."],
        ["LongBench CAUSAL", "RAW_COMMITTED", "results/causal_v4_25_generalization_v1/longbench_full/causal_v4_25/", "Per-task JSONL."],
        ["Budget/mechanism", "RAW_COMMITTED", "results/aime24_value_capacity_budget_3090/shards/", "Shard CSV/JSON forensic records."],
    ]))

    write(OUT / "final_experiment_gap_audit.md", "# Final Experiment Gap Audit\n\n" + make_markdown_table(["Priority", "Gap", "Status", "Reason"], [
        ["P0", "AIME24 selector component ablation: Importance-only and Error-reduction-only", "MISSING", "Random and CAUSAL exist; separate components do not."],
        ["P0", "AIME25 Llama four-method quality", "MISSING", "No smoke, partial, or full canonical AIME25 results found."],
        ["P0", "Second-backbone CAUSAL quality evidence", "MISSING", "No Qwen/second-backbone CAUSAL quality run found."],
        ["P1", "Full official-split LongBench", "MISSING_BUT_OPTIONAL", "Current evidence is 21x50 with 8K cap, not full official split."],
        ["P1", "AMC24", "MISSING", "Useful long-reasoning external validation after P0."],
        ["P1", "AIME24 Avg@8/Maj@8", "MISSING", "Would improve sampling robustness but current AIME24 is already canonical aggregate."],
        ["P2", "AMC23/GPQA/MATH-like", "OPTIONAL", "No current CAUSAL evidence; add only if narrative needs broader reasoning."],
        ["STOP", "System optimization or system reruns", "DO_NOT_RUN", "System track is frozen."],
    ]))

    write(OUT / "gpu_budget_estimate.md", "# GPU Budget Estimate\n\n" + make_markdown_table(["P0 Item", "Generation Count", "Parallelization Candidate", "Relative Cost"], [
        ["Selector ablation: 2 missing methods x 30 x 3", "180", "AIME24 shard by method/seed/problem", "LOW"],
        ["AIME25: 4 methods x 30 x 8", "960", "Shard by method and seed across available GPUs", "HIGH"],
        ["Second-backbone Qwen AIME24: 4 methods x 30 x 3", "360", "Shard by method/seed/problem", "MEDIUM"],
        ["Total P0", "1500", "Staged execution after offline plan approval", "HIGH"],
    ]))

    write(OUT / "paper_quality_story.md", f"""# Paper Quality Story

INT2 quantization error enters persistent KV state and is recursively reused by later autoregressive steps. The matched pseudo-decode evidence supports error accumulation over the tested AIME24 trajectories, and the routing/value forensic shows value-path propagation dominated the tested regime.

The primary quality result is AIME24: CAUSAL-V4@25% reaches `45/90` and matches FP16 aggregate accuracy under the three-seed protocol, while Pattern Base reaches `32/90` and same-budget Random reaches `36/90`. The CAUSAL-vs-Base paired bootstrap CI is positive; the CAUSAL-vs-Random CI crosses zero and should be described as an aggregate advantage only.

General reasoning evidence exists on GSM8K: CAUSAL reaches `{gsm_causal['correct']}/1319` (`{fmt_pct(gsm_causal['accuracy'])}`), above FP16, PatternKV, and KIVI in aggregate. Long-context evidence exists on the 8K-capped LongBench 21x50 setup: CAUSAL averages `{lb_macro['CAUSAL-V4@25%']:.4f}`, above PatternKV and KIVI but below FP16.

Efficiency evidence supports CAUSAL-V4@25% at approximately `{bits['level_a_formal_budget_metric']['CAUSAL_V4_25']}` effective bit/KV element under project payload-and-metadata accounting. The 25% budget is a useful operating point in the current forensic budget curve, not a universal optimum.

Genuine remaining gaps are selector component quality ablations, AIME25 generalization, and second-backbone CAUSAL quality evidence.
""")

    write(OUT / "README.md", "# Paper Quality Evidence Assembly V1\n\nOffline-only quality evidence audit and paper assembly package. No GPU experiments, model loads, CUDA workloads, or new generations were run.\n\nPrimary outputs are `claim_inventory.md`, benchmark evidence files, `tables/quality_main_table.md`, `final_experiment_gap_audit.md`, `gpu_budget_estimate.md`, and `final_gate.json`.")

    write_json(OUT / "final_gate.json", {
        "classification": "PAPER_QUALITY_EVIDENCE_ASSEMBLY_V1_SUPPORTED",
        "mode": "OFFLINE_ONLY",
        "system_track": "FROZEN",
        "source_system_checkpoint": "52a3297b777810543e18608bfff0088e624324d1",
        "frozen_system_release": release,
        "release_unchanged": release == "8d60485b5d2c93b7c1d478efc449de56d28159c3",
        "new_gpu_experiments_run": False,
        "model_loaded": False,
        "cuda_workloads_run": False,
        "new_generations_run": False,
        "algorithm_changed": False,
        "runtime_changed": False,
        "quality_freeze_status": "QUALITY_EVIDENCE_GAPS_REMAIN",
        "canonical_quality_numbers": "reports/paper_quality_evidence_assembly_v1/canonical_quality_numbers.json",
        "p0_gaps": [
            "selector_importance_only_error_only_aime24",
            "aime25_llama_four_method",
            "second_backbone_causal_quality",
        ],
    })

    write(OUT / "summary.md", f"""# Summary

Final classification: PAPER_QUALITY_EVIDENCE_ASSEMBLY_V1_SUPPORTED.

System track: FROZEN.
New GPU experiments run: false.

Canonical evidence found:

- AIME24: FP16 `45/90`, Pattern Base `32/90`, Random-25% `36/90`, CAUSAL-V4@25% `45/90`.
- GSM8K: FP16 `1029/1319`, KIVI `909/1319`, PatternKV `973/1319`, CAUSAL-V4@25% `{gsm_causal['correct']}/1319`.
- LongBench 21x50 8K: FP16 `{lb_macro['FP16']:.4f}`, KIVI `{lb_macro['KIVI']:.4f}`, PatternKV `{lb_macro['PatternKV']:.4f}`, CAUSAL-V4@25% `{lb_macro['CAUSAL-V4@25%']:.4f}`.

Quality evidence gaps remain. The smallest P0 set is selector component quality ablations, AIME25 Llama four-method validation, and second-backbone CAUSAL quality evidence.
""")


if __name__ == "__main__":
    main()
