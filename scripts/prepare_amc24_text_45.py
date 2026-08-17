from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "datasets" / "amc24_text_45"
DATASET_PATH = OUT_DIR / "amc24_text_45.jsonl"
MANIFEST_PATH = OUT_DIR / "manifest.json"
PROTOCOL_PATH = OUT_DIR / "protocol.json"
README_PATH = OUT_DIR / "README.md"

BENCHMARK_ID = "amc24_text_45"
DISPLAY_NAME = "AMC24-Text"
DESCRIPTION = "2024 AMC 12A/12B text-only subset (45 problems)"
UPSTREAM_NAME = "rawsh/2024_AMC12"
UPSTREAM_REVISION = "47b35303156a75cdfc6fcca694db66905d5b2033"
UPSTREAM_FILE = "amc12-2024.jsonl"
UPSTREAM_URL = (
    "https://huggingface.co/datasets/rawsh/2024_AMC12/resolve/"
    f"{UPSTREAM_REVISION}/{UPSTREAM_FILE}"
)
UPSTREAM_README_URL = (
    "https://huggingface.co/datasets/rawsh/2024_AMC12/raw/"
    f"{UPSTREAM_REVISION}/README.md"
)
UPSTREAM_JSONL_SHA256 = "3e020a1c03a42d9b846892ca92b4c7dc55490492f08cdc00df3c0379d2556a58"
RETRIEVAL_DATE = "2026-08-17"
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
EXCLUSIONS = [
    {"competition": "AMC12A", "year": 2024, "problem_number": 14, "reason": "figure-dependent; excluded by upstream dataset README"},
    {"competition": "AMC12A", "year": 2024, "problem_number": 18, "reason": "figure-dependent; excluded by upstream dataset README"},
    {"competition": "AMC12A", "year": 2024, "problem_number": 22, "reason": "figure-dependent; excluded by upstream dataset README"},
    {"competition": "AMC12B", "year": 2024, "problem_number": 7, "reason": "figure-dependent; excluded by upstream dataset README"},
    {"competition": "AMC12B", "year": 2024, "problem_number": 19, "reason": "figure-dependent; excluded by upstream dataset README"},
]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_upstream() -> bytes:
    with urllib.request.urlopen(UPSTREAM_URL, timeout=30) as response:
        return response.read()


def parse_upstream(raw: bytes) -> list[dict[str, Any]]:
    rows = []
    for source_row_index, line in enumerate(raw.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        exam = item["exam"]
        competition = exam.replace("2024 ", "").replace(" ", "")
        problem_number = int(item["problem_number"])
        if competition not in {"AMC12A", "AMC12B"}:
            raise ValueError(f"unexpected competition: {exam}")
        problem_id = f"12{competition[-1]}_{problem_number:02d}"
        row = {
            "benchmark": BENCHMARK_ID,
            "problem_id": problem_id,
            "competition": competition,
            "year": 2024,
            "problem_number": problem_number,
            "problem": str(item["problem"]).strip(),
            "choices": [],
            "answer": str(item["answer"]).strip(),
            "answer_format": "source_answer_string",
            "source": UPSTREAM_NAME,
            "source_row_id": f"{UPSTREAM_NAME}:{UPSTREAM_FILE}:{source_row_index}",
            "source_revision": UPSTREAM_REVISION,
            "metadata": {"upstream_exam": exam},
        }
        rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row["problem_id"] for row in rows]
    problem_texts = [row["problem"] for row in rows]
    counts = Counter(row["competition"] for row in rows)
    missing_problem = [row["problem_id"] for row in rows if not row["problem"]]
    missing_answer = [row["problem_id"] for row in rows if not row["answer"]]
    invalid_answer = [row["problem_id"] for row in rows if not isinstance(row["answer"], str) or not row["answer"].strip()]
    expected_a = [n for n in range(1, 26) if n not in {14, 18, 22}]
    expected_b = [n for n in range(1, 26) if n not in {7, 19}]
    actual_a = [row["problem_number"] for row in rows if row["competition"] == "AMC12A"]
    actual_b = [row["problem_number"] for row in rows if row["competition"] == "AMC12B"]
    checks = {
        "row_count": len(rows),
        "row_count_pass": len(rows) == 45,
        "unique_problem_id_count": len(set(ids)),
        "unique_problem_id_pass": len(set(ids)) == 45,
        "duplicate_problem_ids": sorted([pid for pid, count in Counter(ids).items() if count > 1]),
        "duplicate_problem_text_count": sum(1 for count in Counter(problem_texts).values() if count > 1),
        "duplicate_problem_text_pass": len(set(problem_texts)) == len(problem_texts),
        "missing_problem": missing_problem,
        "missing_problem_pass": not missing_problem,
        "missing_answer": missing_answer,
        "missing_answer_pass": not missing_answer,
        "invalid_answer": invalid_answer,
        "invalid_answer_pass": not invalid_answer,
        "competition_counts": dict(sorted(counts.items())),
        "amc12a_problem_numbers": actual_a,
        "amc12b_problem_numbers": actual_b,
        "amc12a_membership_pass": actual_a == expected_a,
        "amc12b_membership_pass": actual_b == expected_b,
    }
    checks["all_pass"] = all(value for key, value in checks.items() if key.endswith("_pass"))
    return checks


def write_jsonl(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with DATASET_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_manifest(rows: list[dict[str, Any]], checks: dict[str, Any], dataset_sha256: str) -> dict[str, Any]:
    return {
        "benchmark_id": BENCHMARK_ID,
        "display_name": DISPLAY_NAME,
        "description": DESCRIPTION,
        "canonical_status": "SUPPORTED_PUBLIC_PREREGISTERED_TEXT_ONLY",
        "upstream_name": UPSTREAM_NAME,
        "upstream_repository": "https://huggingface.co/datasets/rawsh/2024_AMC12",
        "upstream_dataset": UPSTREAM_NAME,
        "upstream_config": "default",
        "upstream_split": "train",
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_file": UPSTREAM_FILE,
        "upstream_url": UPSTREAM_URL,
        "upstream_readme_url": UPSTREAM_README_URL,
        "upstream_jsonl_sha256": UPSTREAM_JSONL_SHA256,
        "license": "Not specified by dataset metadata; README notes MAA copyright for problems",
        "filter_rule": "include all rows in pinned upstream amc12-2024.jsonl; upstream has already removed figure-dependent problems",
        "exclusion_policy": "exclude problems whose required information depends on non-text figure/diagram/image absent from the canonical prompt",
        "exclusions": EXCLUSIONS,
        "problem_count": len(rows),
        "problem_ids": [row["problem_id"] for row in rows],
        "source_row_ids": [row["source_row_id"] for row in rows],
        "ground_truth_format": "source_answer_string",
        "answer_space": "open source answer strings from upstream answer field",
        "choices_available": False,
        "dataset_file": str(DATASET_PATH.relative_to(ROOT)),
        "dataset_sha256": dataset_sha256,
        "problem_text_hashes": {row["problem_id"]: sha256_text(row["problem"]) for row in rows},
        "answer_hashes": {row["problem_id"]: sha256_text(row["answer"]) for row in rows},
        "duplicate_policy": "fail on duplicate problem_id or duplicate problem text",
        "normalization_policy": "preserve upstream problem and answer strings; parser scoring normalizes only generated answer and gold string for exact string comparison",
        "retrieval_date": RETRIEVAL_DATE,
        "checks": checks,
        "provenance_evidence": [
            {
                "source_type": "Hugging Face dataset API",
                "source": "https://huggingface.co/api/datasets/rawsh/2024_AMC12",
                "revision": UPSTREAM_REVISION,
                "supports": "dataset identity, immutable revision, file list, row count metadata absence, README description",
            },
            {
                "source_type": "Hugging Face dataset README",
                "source": UPSTREAM_README_URL,
                "revision": UPSTREAM_REVISION,
                "supports": "AoPS source URLs and five figure-dependent exclusions",
            },
            {
                "source_type": "Hugging Face dataset JSONL",
                "source": UPSTREAM_URL,
                "revision": UPSTREAM_REVISION,
                "supports": "canonical 45 rows, source schema, problem text, answer strings",
            },
        ],
    }


def build_protocol(dataset_sha256: str) -> dict[str, Any]:
    return {
        "benchmark_id": BENCHMARK_ID,
        "display_name": DISPLAY_NAME,
        "description": DESCRIPTION,
        "problem_count": 45,
        "responses_per_problem": 8,
        "dataset_sha256": dataset_sha256,
        "prompt": {
            "source_status": "PROJECT_CANONICAL_REUSED_WITH_DATASET_SPECIFIC_GROUND_TRUTH_FORMAT",
            "aime24_source": "bench/bench_aime24_patternkv.py:render_prompt",
            "system_prompt": None,
            "user_prompt_template": "{problem}\\n\\nPlease reason step by step, and put your final answer within \\\\boxed{}.",
            "choices": "not included; upstream public source has no choices field",
            "chat_template": "tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)",
            "assistant_suffix": "<think>\\n",
            "manual_assistant_newline": False,
            "duplicate_newline_prevention": "assistant suffix is exactly '<think>\\n' appended once after chat template generation prompt; no extra blank line is appended",
        },
        "sampling": {
            "source_status": "PROJECT_CANONICAL_REUSED",
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": None,
            "repetition_penalty": 1.0,
            "num_return_sequences": 1,
            "max_new_tokens": 32768,
            "max_model_len": "model/tokenizer context limit validated in runner before generation",
            "stop_policy": "use normalized tokenizer/model EOS token ids; record eos or length stop reason",
        },
        "seeds": {
            "source_status": "PROJECT_PREREGISTERED",
            "seed_source": "new fixed 8-response list; no prior 8-response project seed list found",
            "response_id_to_seed": {str(i): seed for i, seed in enumerate(SEEDS)},
            "seeds": SEEDS,
            "pairing": "same problem_id, response_id, and seed for every method",
        },
        "parser": {
            "source_status": "PROJECT_PREREGISTERED",
            "path": "evaluation/amc_source_answer_parser.py",
            "normalizer": "normalize_answer",
            "normalizer_version": "amc24_text_normalizer_v1",
            "strategy": "prefer final boxed answer; otherwise explicit final answer line; no whole-CoT gold-informed search",
            "gold_visible_to_parser": False,
            "parser_failure_policy": "incorrect for Avg@8; contributes no canonical answer vote for Maj@8",
        },
        "avg8": {
            "definition": "correct independent responses / (45 x 8)",
            "equivalent_definition": "mean of per-problem mean correctness across eight responses",
        },
        "maj8": {
            "definition": "for each problem, count parsed normalized answers across eight responses; unique modal answer is the prediction; score exact normalized match to gold",
            "parser_failures": "do not vote; denominator remains 45 problems",
        },
        "tie_policy": {
            "rule": "if there is no unique modal parsed answer, Maj@8 prediction is unresolved and correctness is 0",
            "rationale": "conservative, no arbitrary favorable tie-break, no seed-order dependency, no gold-informed selection",
        },
        "truncation_policy": {
            "rule": "if max_new_tokens is reached, parse the emitted text with the same parser; parsed answers score normally, parse failures are incorrect; no selective reruns",
        },
        "method_matrix": [
            {"method": "FP16", "dtype": "float16", "kv_method": "full_precision", "k_bits": 16, "v_bits": 16},
            {"method": "KIVI", "dtype": "float16", "kv_method": "kivi_paper_g128", "k_bits": 2, "v_bits": 2, "group_size": 128, "residual_length": 128},
            {"method": "PatternKV", "dtype": "float16", "kv_method": "patternkv_paper", "k_bits": 2, "v_bits": 2, "group_size": 128, "residual_length": 128, "initial_pattern_count": 32},
            {
                "method": "CAUSAL-V4@25%",
                "dtype": "float16",
                "kv_method": "causal_v4_25",
                "k_bits": 2,
                "v_base_bits": 2,
                "selected_v_bits": 4,
                "v4_budget_fraction": 0.25,
                "sink": 16,
                "recent": 128,
                "residual_pending": 128,
                "group_size": 128,
                "selector": "historical causal importance x positive local quantization-error reduction V2 -> V4",
                "algorithm_checkpoint": "c73aeed3247c136859f695d5b238eeb357434b17",
            },
        ],
        "expected_full_run": {
            "problems": 45,
            "responses_per_problem": 8,
            "methods": 4,
            "generations_per_method": 360,
            "total_generations": 1440,
        },
        "patternkv_paper_amc24": {
            "status": "REFERENCE_ONLY",
            "directly_comparable": False,
            "reason": "PatternKV exact AMC24 rows/protocol are unpublished; AMC24-Text-45 is an independent public benchmark",
        },
        "frozen_before_generation": True,
    }


def write_readme(manifest: dict[str, Any]) -> None:
    README_PATH.write_text(
        "\n".join(
            [
                "# AMC24-Text",
                "",
                "Benchmark ID: `amc24_text_45`",
                "",
                "This is a public, independent, preregistered 45-problem text-only benchmark built from `rawsh/2024_AMC12`.",
                "It contains 2024 AMC 12A and AMC 12B problems whose required information is present in text.",
                "",
                "It is not an exact reproduction of PatternKV's unpublished AMC24 protocol.",
                "",
                f"- Upstream revision: `{UPSTREAM_REVISION}`",
                f"- Dataset SHA256: `{manifest['dataset_sha256']}`",
                "- Rows: `45`",
                "- AMC12A rows: `22`",
                "- AMC12B rows: `23`",
                "- Excluded: `12A_14`, `12A_18`, `12A_22`, `12B_07`, `12B_19`",
                "- Ground truth: upstream `answer` string",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    raw = fetch_upstream()
    upstream_sha = sha256_bytes(raw)
    if upstream_sha != UPSTREAM_JSONL_SHA256:
        raise SystemExit(f"upstream SHA mismatch: {upstream_sha} != {UPSTREAM_JSONL_SHA256}")
    rows = parse_upstream(raw)
    checks = validate_rows(rows)
    if not checks["all_pass"]:
        raise SystemExit(f"dataset validation failed: {checks}")
    write_jsonl(rows)
    dataset_sha256 = sha256_bytes(DATASET_PATH.read_bytes())
    manifest = build_manifest(rows, checks, dataset_sha256)
    protocol = build_protocol(dataset_sha256)
    write_json(MANIFEST_PATH, manifest)
    write_json(PROTOCOL_PATH, protocol)
    write_readme(manifest)
    print(json.dumps({"dataset": str(DATASET_PATH), "rows": len(rows), "sha256": dataset_sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
