from __future__ import annotations

from collections import Counter
from pathlib import Path

from scripts import aime24_full30_3seed_formal as formal


def test_full30_manifest_has_three_global_seeds() -> None:
    tasks = formal.formal_tasks()
    assert len(tasks) == 90
    assert sorted({task["seed"] for task in tasks}) == [42, 43, 44]
    assert Counter(task["seed"] for task in tasks) == {42: 30, 43: 30, 44: 30}
    assert {task["sample_id"] for task in tasks} == {0, 1, 2}


def test_seeded_result_paths_are_unique() -> None:
    from bench.aime_utils import result_path

    path42 = result_path(Path("results"), "fp16", 0, 0, "cfg", seed=42)
    path43 = result_path(Path("results"), "fp16", 0, 0, "cfg", seed=43)
    assert path42 != path43
    assert "seed42" in path42.name
    assert "seed43" in path43.name


def test_formal_config_set_is_exact() -> None:
    assert [cfg["name"] for cfg in formal.CONFIGS] == [
        "fp16",
        "patternkv_paper",
        "pattern_rolling_s0_r128",
        "pattern_rolling_s16_r128",
        "kivi_paper",
        "kivi_rolling_s0_r128",
        "kivi_rolling_s16_r128",
    ]
    assert formal.CONFIGS[1]["cache_mode"] == "legacy_tuple_chunked"
    assert formal.CONFIGS[2]["cache_mode"] == "segmented_rolling"
    assert formal.CONFIGS[3]["sink_length"] == 16
    assert formal.CONFIGS[6]["sink_length"] == 16


def test_paired_rows_counts_rescues_and_regressions() -> None:
    rows = [
        {"config": "pattern_rolling_s0_r128", "problem_id": 0, "seed": 42, "strict_correct": False},
        {"config": "pattern_rolling_s16_r128", "problem_id": 0, "seed": 42, "strict_correct": True},
        {"config": "pattern_rolling_s0_r128", "problem_id": 1, "seed": 42, "strict_correct": True},
        {"config": "pattern_rolling_s16_r128", "problem_id": 1, "seed": 42, "strict_correct": False},
    ]
    for pid in range(2, 30):
        for seed in formal.SEEDS:
            rows.extend(
                [
                    {"config": "pattern_rolling_s0_r128", "problem_id": pid, "seed": seed, "strict_correct": False},
                    {"config": "pattern_rolling_s16_r128", "problem_id": pid, "seed": seed, "strict_correct": False},
                ]
            )
    paired = [row for row in formal.paired_rows(rows) if row["comparison_name"] == "pattern_s0_to_s16" and row["seed"] == 42][0]
    assert paired["rescues"] == 1
    assert paired["regressions"] == 1
    assert paired["net_gain"] == 0


def test_consistency_stable_rescue() -> None:
    rows = []
    for seed in formal.SEEDS:
        rows.append({"config": "pattern_rolling_s0_r128", "problem_id": 0, "seed": seed, "strict_correct": False})
        rows.append({"config": "pattern_rolling_s16_r128", "problem_id": 0, "seed": seed, "strict_correct": True})
    out = [row for row in formal.consistency_rows(rows) if row["comparison_name"] == "pattern_s0_to_s16" and row["problem_id"] == 0][0]
    assert out["classification"] == "STABLE_RESCUE"
