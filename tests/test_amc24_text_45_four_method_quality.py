import json

from scripts import run_amc24_text_45_four_method_quality as runner


def test_frozen_seed_mapping_is_literal_response_seed():
    assert runner.SEEDS == (42, 43, 44, 45, 46, 47, 48, 49)
    assert runner.phase_response_ids("formal", None) == list(range(8))
    assert [runner.SEEDS[i] for i in runner.phase_response_ids("formal", None)] == [42, 43, 44, 45, 46, 47, 48, 49]


def test_method_configs_match_frozen_protocol():
    assert runner.METHOD_ORDER == ("FP16", "KIVI", "PatternKV", "CAUSAL")
    assert runner.METHOD_CONFIGS["FP16"]["backend_method"] == "fp16"
    assert runner.METHOD_CONFIGS["KIVI"]["method_arg"] == "kivi_paper_g128"
    assert runner.METHOD_CONFIGS["KIVI"]["k_bits"] == 2
    assert runner.METHOD_CONFIGS["KIVI"]["v_bits"] == 2
    assert runner.METHOD_CONFIGS["KIVI"]["group_size"] == 128
    assert runner.METHOD_CONFIGS["KIVI"]["residual_length"] == 128
    assert runner.METHOD_CONFIGS["PatternKV"]["method_arg"] == "patternkv_paper"
    assert runner.METHOD_CONFIGS["PatternKV"]["selector"] == "base_v2"
    assert runner.METHOD_CONFIGS["PatternKV"]["num_k_base"] == 32
    assert runner.METHOD_CONFIGS["PatternKV"]["num_v_base"] == 32
    assert runner.METHOD_CONFIGS["CAUSAL"]["selector"] == "causal_v4"
    assert runner.METHOD_CONFIGS["CAUSAL"]["v4_budget_fraction"] == 0.25
    assert runner.METHOD_CONFIGS["CAUSAL"]["sink_length"] == 16


def test_expected_formal_matrix_count():
    expected = runner.expected_keys("formal")
    assert len(expected) == 1440
    assert len([key for key in expected if key[0] == "FP16"]) == 360


def test_majority_uses_canonical_keys_and_keeps_tie_unresolved():
    gold = r"\frac{39}{7}"
    rows = [
        {"response_id": 0, "canonical_prediction_key": r"\frac{39}{7}"},
        {"response_id": 1, "canonical_prediction_key": r"\frac{39}{7}"},
        {"response_id": 2, "canonical_prediction_key": r"\frac{39}{7}"},
        {"response_id": 3, "canonical_prediction_key": r"\frac{39}{7}"},
        {"response_id": 4, "canonical_prediction_key": "5"},
        {"response_id": 5, "canonical_prediction_key": "5"},
        {"response_id": 6, "canonical_prediction_key": "6"},
        {"response_id": 7, "canonical_prediction_key": None},
    ]
    vote = runner.majority_for_problem(rows, gold)
    assert vote["prediction"] == r"\frac{39}{7}"
    assert vote["correct"] is True

    tie = runner.majority_for_problem(
        [
            {"response_id": 0, "canonical_prediction_key": "4"},
            {"response_id": 1, "canonical_prediction_key": "4"},
            {"response_id": 2, "canonical_prediction_key": "5"},
            {"response_id": 3, "canonical_prediction_key": "5"},
        ],
        "4",
    )
    assert tie["prediction"] is None
    assert tie["correct"] is False


def test_paired_bootstrap_uses_question_level_rows():
    rows = []
    for idx in range(45):
        rows.append(
            {
                "problem_id": f"p{idx}",
                "FP16": {"mean_correct": 0.5, "majority_correct": False},
                "KIVI": {"mean_correct": 0.25, "majority_correct": False},
                "PatternKV": {"mean_correct": 0.25, "majority_correct": False},
                "CAUSAL": {"mean_correct": 0.75, "majority_correct": True},
            }
        )
    avg = runner.paired_bootstrap(rows, "mean_correct")
    maj = runner.paired_bootstrap(rows, "majority_correct")
    assert avg["bootstrap_unit"] == "question"
    assert avg["CAUSAL_vs_PatternKV"]["mean_difference"] == 0.5
    assert maj["CAUSAL_vs_PatternKV"]["mean_difference"] == 1.0


def test_deterministic_sharding_covers_all_identities_without_overlap():
    identities = [(row["problem_id"], response_id) for row in runner.load_dataset() for response_id in range(len(runner.SEEDS))]
    for method in runner.METHOD_ORDER:
        for num_shards in (1, 2, 3, 4):
            shards = []
            for shard_id in range(num_shards):
                shard = {
                    identity
                    for identity in identities
                    if runner.deterministic_shard_id(method, identity[0], identity[1], num_shards) == shard_id
                }
                shards.append(shard)
            union = set().union(*shards)
            assert union == set(identities)
            for left in range(num_shards):
                for right in range(left + 1, num_shards):
                    assert shards[left].isdisjoint(shards[right])


def test_update_work_manifest_classifies_records(tmp_path, monkeypatch):
    manifest_path = tmp_path / "work_manifest.json"
    monkeypatch.setattr(runner, "WORK_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(
        runner,
        "formal_identity_records",
        lambda: [
            {
                "benchmark": runner.BENCHMARK_ID,
                "method": "FP16",
                "problem_id": "12A_01",
                "response_id": 0,
                "seed": 42,
                "dataset_sha256": runner.EXPECTED_DATASET_SHA256,
                "protocol_hash": runner.formal_config_hash(),
                "method_config_hash": runner.config_hash(runner.METHOD_CONFIGS["FP16"]),
                "status": "PENDING",
                "attempt_count": 0,
                "output_path": "results/amc24_text_45_four_method_quality_v1/formal/fp16/12A_01/r00.json",
                "task_key": "k0",
            },
            {
                "benchmark": runner.BENCHMARK_ID,
                "method": "KIVI",
                "problem_id": "12A_01",
                "response_id": 0,
                "seed": 42,
                "dataset_sha256": runner.EXPECTED_DATASET_SHA256,
                "protocol_hash": runner.formal_config_hash(),
                "method_config_hash": runner.config_hash(runner.METHOD_CONFIGS["KIVI"]),
                "status": "PENDING",
                "attempt_count": 0,
                "output_path": "results/amc24_text_45_four_method_quality_v1/formal/kivi/12A_01/r00.json",
                "task_key": "k1",
            },
        ],
    )
    payload = runner.update_work_manifest(
        [
            {
                "method": "FP16",
                "problem_id": "12A_01",
                "response_id": 0,
                "status": "completed",
                "retry_count": 0,
                "raw_generation_path": "results/amc24_text_45_four_method_quality_v1/formal/fp16/12A_01/r00.txt",
            },
            {
                "method": "KIVI",
                "problem_id": "12A_01",
                "response_id": 0,
                "status": "failed",
                "retry_count": 1,
                "oom": True,
                "runtime_error": "oom",
                "raw_generation_path": None,
            },
        ]
    )
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["total_identities"] == 2
    assert payload["statuses"]["COMPLETE"] == 1
    assert payload["statuses"]["OOM"] == 1
    assert saved["statuses"] == payload["statuses"]
