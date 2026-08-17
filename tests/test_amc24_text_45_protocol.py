import hashlib
import json
import subprocess
from pathlib import Path


DATASET = Path("datasets/amc24_text_45/amc24_text_45.jsonl")
PROTOCOL = Path("datasets/amc24_text_45/protocol.json")


def test_protocol_freezes_required_fields():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["benchmark_id"] == "amc24_text_45"
    assert protocol["display_name"] == "AMC24-Text"
    assert protocol["problem_count"] == 45
    assert protocol["responses_per_problem"] == 8
    assert protocol["dataset_sha256"] == hashlib.sha256(DATASET.read_bytes()).hexdigest()
    assert protocol["seeds"]["seeds"] == [42, 43, 44, 45, 46, 47, 48, 49]
    assert protocol["avg8"]["definition"] == "correct independent responses / (45 x 8)"
    assert protocol["tie_policy"]["rule"].startswith("if there is no unique modal parsed answer")
    assert protocol["frozen_before_generation"] is True


def test_method_matrix_and_expected_full_run():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    methods = [row["method"] for row in protocol["method_matrix"]]
    assert methods == ["FP16", "KIVI", "PatternKV", "CAUSAL-V4@25%"]
    assert protocol["expected_full_run"] == {
        "generations_per_method": 360,
        "methods": 4,
        "problems": 45,
        "responses_per_problem": 8,
        "total_generations": 1440,
    }
    causal = protocol["method_matrix"][3]
    assert causal["algorithm_checkpoint"] == "c73aeed3247c136859f695d5b238eeb357434b17"
    assert causal["v4_budget_fraction"] == 0.25


def test_prepare_script_reproduces_hash():
    before = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    subprocess.run(
        ["/data/zypan/.local/share/mamba/envs/patternkv/bin/python", "scripts/prepare_amc24_text_45.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    after = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    assert after == before
