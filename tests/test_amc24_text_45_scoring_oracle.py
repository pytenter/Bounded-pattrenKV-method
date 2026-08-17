import json
from collections import defaultdict
from pathlib import Path

from evaluation.amc_source_answer_parser import NORMALIZER_VERSION, normalize_answer


DATASET = Path("datasets/amc24_text_45/amc24_text_45.jsonl")


def rows():
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_all_45_gold_answers_normalize_deterministically():
    normalized = []
    for row in rows():
        first = normalize_answer(row["answer"])
        second = normalize_answer(row["answer"])
        assert first is not None, row["problem_id"]
        assert first == second, row["problem_id"]
        normalized.append(first)
    assert len(normalized) == 45


def test_canonical_collision_audit_benign_only():
    grouped = defaultdict(list)
    for row in rows():
        grouped[normalize_answer(row["answer"])].append((row["problem_id"], row["answer"]))
    collisions = {key: values for key, values in grouped.items() if len(values) > 1}
    assert collisions == {
        "0": [("12A_08", "0"), ("12B_02", "0")],
        "3": [("12A_06", "3"), ("12B_10", "3"), ("12B_24", "3")],
        "5": [("12B_14", "5"), ("12B_16", "5")],
        "15": [("12B_05", "15"), ("12B_22", "15")],
        "20": [("12A_11", "20"), ("12B_06", "20")],
        "21": [("12A_03", "21"), ("12A_12", "21"), ("12B_03", "21")],
    }


def test_normalizer_version_frozen():
    assert NORMALIZER_VERSION == "amc24_text_normalizer_v1"
