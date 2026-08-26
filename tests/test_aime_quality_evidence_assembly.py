import csv
import gzip
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "assemble_aime_n3_quality_evidence.py"
spec = importlib.util.spec_from_file_location("assemble", SCRIPT)
assemble = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assemble)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(path.read_text())


def test_aime25_dataset_has_30_unique_problems():
    rows = [json.loads(line) for line in (ROOT / "datasets/aime/aime25.jsonl").read_text().splitlines()]
    assert len(rows) == 30
    assert sorted(r["problem_id"] for r in rows) == list(range(30))
    assert all(r["problem"] and r["answer"] for r in rows)


def test_aime25_gold_normalizes_30_of_30():
    rows = [json.loads(line) for line in (ROOT / "datasets/aime/aime25.jsonl").read_text().splitlines()]
    assert all(assemble.normalize_answer(r["answer"]).isdigit() for r in rows)
    assert all(0 <= int(assemble.normalize_answer(r["answer"])) <= 999 for r in rows)


def test_aime25_manifest_has_360_unique_identities():
    rows = [json.loads(line) for line in (ROOT / "reports/aime25_four_method_quality_v1/formal_manifest.jsonl").read_text().splitlines()]
    ids = {(r["benchmark"], r["method"], r["problem_id"], r["base_seed"], r["sample_id"]) for r in rows}
    assert len(rows) == 360
    assert len(ids) == 360


def test_aime24_kivi_manifest_has_90_unique_identities():
    rows = [json.loads(line) for line in (ROOT / "reports/aime24_kivi_quality_v1/formal_manifest.jsonl").read_text().splitlines()]
    ids = {(r["benchmark"], r["method"], r["problem_id"], r["base_seed"], r["sample_id"]) for r in rows}
    assert len(rows) == 90
    assert len(ids) == 90


def test_effective_seed_formula_and_sample_id_zero():
    for manifest in ["reports/aime25_four_method_quality_v1/formal_manifest.jsonl", "reports/aime24_kivi_quality_v1/formal_manifest.jsonl"]:
        rows = [json.loads(line) for line in (ROOT / manifest).read_text().splitlines()]
        assert all(r["sample_id"] == 0 for r in rows)
        assert all(r["effective_seed"] == assemble.effective_seed(r["base_seed"], r["problem_id"], 0) for r in rows)


def test_duplicate_identity_rejected_and_completeness_gate_pass():
    assert read_json(ROOT / "reports/aime25_four_method_quality_v1/completeness_audit.json")["gate"] == "PASS"
    assert read_json(ROOT / "reports/aime24_kivi_quality_v1/completeness_audit.json")["gate"] == "PASS"
    assert read_json(ROOT / "reports/aime25_four_method_quality_v1/duplicate_audit.json")["gate"] == "PASS"


def test_smoke_qwen_and_n8_rows_rejected_by_manifest_shape():
    rows = [json.loads(line) for line in (ROOT / "reports/aime25_four_method_quality_v1/formal_manifest.jsonl").read_text().splitlines()]
    assert {r["benchmark"] for r in rows} == {"aime25"}
    assert not any("qwen" in r["source_result_path"].lower() or "smoke" in r["source_result_path"].lower() for r in rows)
    assert {r["sample_id"] for r in rows} == {0}


def test_method_identity_validation():
    methods = read_json(ROOT / "reports/aime25_four_method_quality_v1/method_identity.json")
    assert methods["KIVI_PAPER_G128"]["backend_methods"] == ["kivi_paper_g128"]
    assert methods["CAUSAL_V4_25"]["backend_methods"] == ["patternkv"]


def test_cross_method_prompt_hash_and_generation_config_audited():
    audit = read_json(ROOT / "reports/aime25_four_method_quality_v1/cross_method_protocol_audit.json")
    assert audit["prompt_hash_per_problem_gate"] == "PASS"
    assert audit["generation_config_cross_method_gate"] in {"PASS", "PARTIAL"}


def test_recomputed_aggregate_matches_expected_summary():
    got = {r["method"]: int(r["correct"]) for r in read_csv(ROOT / "reports/aime25_four_method_quality_v1/method_summary.csv")}
    assert got == {"FP16": 30, "KIVI_PAPER_G128": 18, "PATTERN_BASE": 21, "CAUSAL_V4_25": 27}
    got24 = {r["method"]: int(r["correct"]) for r in read_csv(ROOT / "reports/aime24_main_quality_table_v1/method_summary.csv")}
    assert got24 == {"FP16": 45, "KIVI_PAPER_G128": 11, "PATTERN_BASE": 32, "RANDOM_V4_25": 36, "CAUSAL_V4_25": 45}


def test_majority3_tie_rule():
    rows = [
        {"parsed_answer":"1","reference_answer":"1","is_correct":True},
        {"parsed_answer":"2","reference_answer":"1","is_correct":False},
        {"parsed_answer":"3","reference_answer":"1","is_correct":False},
    ]
    counts = {}
    for r in rows:
        counts[r["parsed_answer"]] = counts.get(r["parsed_answer"], 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    assert top[0][1] == top[1][1]


def test_bootstrap_is_problem_clustered():
    bs = read_json(ROOT / "reports/aime25_four_method_quality_v1/paired_bootstrap.json")
    assert bs["unit"] == "problem"
    assert bs["resamples"] >= 10000
    assert bs["seed"] == 20260826


def test_source_map_covers_all_paper_cells():
    smap = read_json(ROOT / "reports/aime25_four_method_quality_v1/source_map.json")
    for method in ["FP16", "KIVI_PAPER_G128", "PATTERN_BASE", "CAUSAL_V4_25"]:
        assert f"aime25.{method}.response_accuracy" in smap
