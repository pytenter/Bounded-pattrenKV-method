import hashlib
import json
from collections import Counter
from pathlib import Path


DATASET = Path("datasets/amc24_text_45/amc24_text_45.jsonl")
MANIFEST = Path("datasets/amc24_text_45/manifest.json")


def read_rows():
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_dataset_has_45_rows_and_expected_membership():
    rows = read_rows()
    assert len(rows) == 45
    assert Counter(row["competition"] for row in rows) == {"AMC12A": 22, "AMC12B": 23}
    assert [row["problem_number"] for row in rows if row["competition"] == "AMC12A"] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        15,
        16,
        17,
        19,
        20,
        21,
        23,
        24,
        25,
    ]
    assert [row["problem_number"] for row in rows if row["competition"] == "AMC12B"] == [
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        20,
        21,
        22,
        23,
        24,
        25,
    ]


def test_dataset_identity_and_ground_truth_fields():
    rows = read_rows()
    ids = [row["problem_id"] for row in rows]
    problems = [row["problem"] for row in rows]
    assert len(set(ids)) == 45
    assert len(set(problems)) == 45
    assert all(row["benchmark"] == "amc24_text_45" for row in rows)
    assert all(row["year"] == 2024 for row in rows)
    assert all(row["choices"] == [] for row in rows)
    assert all(row["answer"].strip() for row in rows)
    assert all(row["source_revision"] == "47b35303156a75cdfc6fcca694db66905d5b2033" for row in rows)


def test_manifest_hash_matches_dataset():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    assert manifest["dataset_sha256"] == digest
    assert manifest["problem_count"] == 45
    assert manifest["checks"]["all_pass"] is True
    assert manifest["exclusions"] == [
        {"competition": "AMC12A", "problem_number": 14, "reason": "figure-dependent; excluded by upstream dataset README", "year": 2024},
        {"competition": "AMC12A", "problem_number": 18, "reason": "figure-dependent; excluded by upstream dataset README", "year": 2024},
        {"competition": "AMC12A", "problem_number": 22, "reason": "figure-dependent; excluded by upstream dataset README", "year": 2024},
        {"competition": "AMC12B", "problem_number": 7, "reason": "figure-dependent; excluded by upstream dataset README", "year": 2024},
        {"competition": "AMC12B", "problem_number": 19, "reason": "figure-dependent; excluded by upstream dataset README", "year": 2024},
    ]
