#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "datasets/amc24_text_45/amc24_text_45.jsonl"
RESULT_DIR = ROOT / "results/amc24_text_45_four_method_quality_v1/formal"
REPORT_DIR = ROOT / "reports/amc24_n3_fast_resume_v3"
EXPECTED_DATASET_SHA256 = "59a7450d9e480a41aa0d9db6dc2d89d16b1188cdf9a1ea8fd12e19dd2033c4b9"
BENCHMARK_ID = "amc24_text_45"
NORMALIZER_VERSION = "amc24_text_normalizer_v1"
RESPONSE_IDS = (0, 1, 2)
SEEDS = (42, 43, 44)
METHOD_ORDER = ("FP16", "KIVI", "PatternKV", "CAUSAL")
METHOD_SLUG = {
    "FP16": "fp16",
    "KIVI": "kivi",
    "PatternKV": "patternkv",
    "CAUSAL": "causal_v4_25",
}
DISPLAY_METHOD = {
    "FP16": "FP16",
    "KIVI": "KIVI",
    "PatternKV": "PatternKV",
    "CAUSAL": "CAUSAL-V4@25%",
}
AUDITED_EQUIVALENT_FALSE_NEGATIVES = (
    {
        "method": "FP16",
        "problem_id": "12B_23",
        "response_id": 0,
        "prediction": r"\frac{\sqrt{2}+1}{2}",
        "gold": r"\frac{1+\sqrt{2}}{2}",
        "reason": "commutative numerator terms are mathematically equivalent",
    },
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 45:
        raise RuntimeError(f"expected 45 AMC24-Text rows, found {len(rows)}")
    dataset_sha = sha256_file(DATASET_PATH)
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise RuntimeError(f"dataset sha mismatch: {dataset_sha}")
    return rows


def result_path(method: str, problem_id: str, response_id: int) -> Path:
    return RESULT_DIR / METHOD_SLUG[method] / problem_id / f"r{response_id:02d}.json"


def load_result(method: str, problem_id: str, response_id: int) -> dict[str, Any] | None:
    path = result_path(method, problem_id, response_id)
    if not path.exists():
        return None
    rec = read_json(path)
    rec["_relative_path"] = str(path.relative_to(ROOT))
    return rec


def majority_vote(keys: list[str | None], gold_key: str | None) -> dict[str, Any]:
    valid = [key for key in keys if key is not None]
    if not valid:
        return {
            "prediction": None,
            "correct": False,
            "tie": False,
            "valid_votes": 0,
            "votes": {},
        }
    counts = Counter(valid)
    top = max(counts.values())
    winners = sorted(key for key, count in counts.items() if count == top)
    prediction = winners[0] if len(winners) == 1 else None
    return {
        "prediction": prediction,
        "correct": prediction is not None and gold_key is not None and prediction == gold_key,
        "tie": len(winners) > 1,
        "valid_votes": len(valid),
        "votes": dict(sorted(counts.items())),
    }


def is_audited_equivalent_false_negative(rec: dict[str, Any]) -> bool:
    for item in AUDITED_EQUIVALENT_FALSE_NEGATIVES:
        if (
            rec.get("method") == item["method"]
            and rec.get("problem_id") == item["problem_id"]
            and int(rec.get("response_id", -1)) == item["response_id"]
            and rec.get("canonical_prediction_key") == item["prediction"]
        ):
            return True
    return False


def numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def build_summary() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_rows = load_dataset()
    per_question: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    for row in dataset_rows:
        problem_id = str(row["problem_id"])
        question_record: dict[str, Any] = {
            "problem_id": problem_id,
            "competition": row.get("competition"),
            "problem_number": row.get("problem_number"),
            "ground_truth": row.get("answer"),
        }
        for method in METHOD_ORDER:
            method_rows: list[dict[str, Any]] = []
            for response_id in RESPONSE_IDS:
                rec = load_result(method, problem_id, response_id)
                if rec is None:
                    missing.append({"method": method, "problem_id": problem_id, "response_id": response_id})
                    continue
                if rec.get("benchmark_id") != BENCHMARK_ID or rec.get("normalizer_version") != NORMALIZER_VERSION:
                    invalid.append(
                        {
                            "method": method,
                            "problem_id": problem_id,
                            "response_id": response_id,
                            "path": rec["_relative_path"],
                            "benchmark_id": rec.get("benchmark_id"),
                            "normalizer_version": rec.get("normalizer_version"),
                        }
                    )
                method_rows.append(rec)
                flat_rows.append(rec)
            ordered = sorted(method_rows, key=lambda item: int(item["response_id"]))
            keys = [rec.get("canonical_prediction_key") for rec in ordered]
            gold_key = ordered[0].get("canonical_ground_truth_key") if ordered else None
            if gold_key is None and ordered:
                gold_key = ordered[0].get("gold_canonical_answer_key")
            vote = majority_vote(keys, gold_key)
            exact_correct = sum(bool(rec.get("correct")) for rec in ordered)
            audited_extra = sum(is_audited_equivalent_false_negative(rec) for rec in ordered)
            question_record[method] = {
                "completed": len(ordered),
                "correct": exact_correct,
                "audited_correct_lower_bound": exact_correct + audited_extra,
                "mean_correct": exact_correct / len(RESPONSE_IDS),
                "audited_mean_correct_lower_bound": (exact_correct + audited_extra) / len(RESPONSE_IDS),
                "majority_prediction": vote["prediction"],
                "majority_correct": bool(vote["correct"]),
                "majority_tie": bool(vote["tie"]),
                "majority_votes": vote["votes"],
                "parser_failures": sum(bool(rec.get("parser_failed")) for rec in ordered),
                "length_stops": sum(rec.get("stop_reason") == "length" for rec in ordered),
                "canonical_answers": keys,
            }
        per_question.append(question_record)

    methods: dict[str, dict[str, Any]] = {}
    for method in METHOD_ORDER:
        rows = [rec for rec in flat_rows if rec.get("method") == method]
        audited_extra = sum(is_audited_equivalent_false_negative(rec) for rec in rows)
        correct = sum(bool(rec.get("correct")) for rec in rows)
        majority_correct = sum(bool(row[method]["majority_correct"]) for row in per_question)
        tokens = [float(rec.get("generated_token_count") or 0) for rec in rows]
        wall = [float(rec.get("wall_time_seconds") or 0) for rec in rows if rec.get("wall_time_seconds") is not None]
        methods[method] = {
            "method": method,
            "display_method": DISPLAY_METHOD[method],
            "problems": len(dataset_rows),
            "responses_per_problem": len(RESPONSE_IDS),
            "expected_generations": len(dataset_rows) * len(RESPONSE_IDS),
            "completed": len(rows),
            "correct_responses_exact": correct,
            "accuracy_exact": correct / (len(dataset_rows) * len(RESPONSE_IDS)),
            "audited_equivalent_false_negatives": audited_extra,
            "correct_responses_audited_lower_bound": correct + audited_extra,
            "accuracy_audited_lower_bound": (correct + audited_extra) / (len(dataset_rows) * len(RESPONSE_IDS)),
            "majority_correct_exact": majority_correct,
            "majority_accuracy_exact": majority_correct / len(dataset_rows),
            "parser_failures": sum(bool(rec.get("parser_failed")) for rec in rows),
            "length_stops": sum(rec.get("stop_reason") == "length" for rec in rows),
            "eos_stops": sum(rec.get("stop_reason") == "eos" for rec in rows),
            "truncated": sum(bool(rec.get("truncated")) for rec in rows),
            "runtime_errors": sum(bool(rec.get("runtime_error")) or rec.get("status") == "failed" for rec in rows),
            "oom": sum(bool(rec.get("oom")) for rec in rows),
            "generated_tokens": numeric_summary(tokens),
            "wall_time_seconds": numeric_summary(wall),
        }

    summary = {
        "benchmark_id": BENCHMARK_ID,
        "classification": "AMC24_TEXT_45_N3_FAST_RESUME_COMPLETE",
        "dataset_sha256": sha256_file(DATASET_PATH),
        "normalizer_version": NORMALIZER_VERSION,
        "result_root": str(RESULT_DIR.relative_to(ROOT)),
        "report_root": str(REPORT_DIR.relative_to(ROOT)),
        "frozen_response_ids": list(RESPONSE_IDS),
        "frozen_seeds": list(SEEDS),
        "problem_count": len(dataset_rows),
        "methods": methods,
        "completion": {
            "expected": len(dataset_rows) * len(RESPONSE_IDS) * len(METHOD_ORDER),
            "completed": len(flat_rows),
            "missing": missing,
            "invalid": invalid,
            "complete": not missing and not invalid and len(flat_rows) == len(dataset_rows) * len(RESPONSE_IDS) * len(METHOD_ORDER),
        },
        "deltas_accuracy_exact": {
            f"CAUSAL_vs_{baseline}": methods["CAUSAL"]["accuracy_exact"] - methods[baseline]["accuracy_exact"]
            for baseline in ("FP16", "KIVI", "PatternKV")
        },
        "deltas_majority_exact": {
            f"CAUSAL_vs_{baseline}": methods["CAUSAL"]["majority_accuracy_exact"] - methods[baseline]["majority_accuracy_exact"]
            for baseline in ("FP16", "KIVI", "PatternKV")
        },
        "audited_equivalence_caveat": {
            "items": list(AUDITED_EQUIVALENT_FALSE_NEGATIVES),
            "interpretation": "Exact normalized-string scoring undercounts FP16 by at least one response; majority result is left as exact-scored unless separately audited.",
        },
    }
    return summary, per_question, flat_rows


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def method_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in METHOD_ORDER:
        item = summary["methods"][method]
        rows.append(
            {
                "method": item["display_method"],
                "completed": item["completed"],
                "expected": item["expected_generations"],
                "correct_responses_exact": item["correct_responses_exact"],
                "accuracy_exact": item["accuracy_exact"],
                "audited_equivalent_false_negatives": item["audited_equivalent_false_negatives"],
                "correct_responses_audited_lower_bound": item["correct_responses_audited_lower_bound"],
                "accuracy_audited_lower_bound": item["accuracy_audited_lower_bound"],
                "majority_correct_exact": item["majority_correct_exact"],
                "majority_accuracy_exact": item["majority_accuracy_exact"],
                "parser_failures": item["parser_failures"],
                "length_stops": item["length_stops"],
                "mean_generated_tokens": item["generated_tokens"]["mean"],
                "median_generated_tokens": item["generated_tokens"]["median"],
                "max_generated_tokens": item["generated_tokens"]["max"],
            }
        )
    return rows


def write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# AMC24-Text N3 Fast Resume Final Summary",
        "",
        "This freezes the N3 fast-resume result set: 45 AMC24-Text problems, three sampled responses per problem, four methods.",
        "",
        "## Completion",
        "",
        f"- Classification: `{summary['classification']}`",
        f"- Complete: `{summary['completion']['complete']}`",
        f"- Completed generations: `{summary['completion']['completed']}/{summary['completion']['expected']}`",
        f"- Frozen response IDs: `{summary['frozen_response_ids']}`",
        f"- Frozen seeds: `{summary['frozen_seeds']}`",
        f"- Dataset SHA256: `{summary['dataset_sha256']}`",
        f"- Normalizer: `{summary['normalizer_version']}`",
        "",
        "## Method Results",
        "",
        "| Method | Completed | Exact correct | Exact accuracy | Audited lower-bound correct | Audited lower-bound accuracy | Majority exact | Parser failures | Length stops |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_csv_rows(summary):
        lines.append(
            f"| {row['method']} | {row['completed']}/{row['expected']} | "
            f"{row['correct_responses_exact']}/{row['expected']} | {100 * row['accuracy_exact']:.2f}% | "
            f"{row['correct_responses_audited_lower_bound']}/{row['expected']} | {100 * row['accuracy_audited_lower_bound']:.2f}% | "
            f"{row['majority_correct_exact']}/45 ({100 * row['majority_accuracy_exact']:.2f}%) | "
            f"{row['parser_failures']} | {row['length_stops']} |"
        )
    lines.extend(
        [
            "",
            "## Exact Deltas",
            "",
            "| Comparison | Response accuracy delta | Majority accuracy delta |",
            "|---|---:|---:|",
        ]
    )
    for baseline in ("FP16", "KIVI", "PatternKV"):
        key = f"CAUSAL_vs_{baseline}"
        lines.append(
            f"| CAUSAL-V4@25% - {baseline} | "
            f"{100 * summary['deltas_accuracy_exact'][key]:.2f} pp | "
            f"{100 * summary['deltas_majority_exact'][key]:.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Scoring Caveat",
            "",
            "- Exact normalized-string scoring counts FP16 `12B_23:r0` as incorrect even though `\\frac{\\sqrt{2}+1}{2}` and `\\frac{1+\\sqrt{2}}{2}` are mathematically equivalent.",
            "- Therefore FP16 exact response score `80/135` should be read as at least `81/135` under audited equivalence. Majority scores in this summary remain exact-scored.",
            "",
            "## Artifacts",
            "",
            "- `final_summary.json`: machine-readable frozen summary.",
            "- `method_summary.csv`: method-level table.",
            "- `per_question_summary.json`: question-level N3 votes and correctness.",
            "- `existing_result_provenance_audit.md`: provenance and scorer caveat audit.",
        ]
    )
    (REPORT_DIR / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary, per_question, _flat_rows = build_summary()
    write_json(REPORT_DIR / "final_summary.json", summary)
    write_json(REPORT_DIR / "per_question_summary.json", per_question)
    write_csv_rows(REPORT_DIR / "method_summary.csv", method_csv_rows(summary))
    write_markdown(summary)
    print(json.dumps(summary["completion"], indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
