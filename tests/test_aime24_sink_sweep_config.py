from __future__ import annotations

import re
from pathlib import Path

from scripts.prepare_wave1a3_sink_sweep_manifest import (
    EXPECTED_GENERATION_HASH,
    EXPECTED_TASK_HASH,
    LOGICAL_CONFIGS,
    NEW_GPU_MAPPING,
)


def test_sink_sweep_new_only_configs_are_isolated() -> None:
    script = Path("scripts/run_aime24_int2_wave1_8gpu.sh").read_text(encoding="utf-8")
    match = re.search(r"SINK_SWEEP_NEW_CONFIGS=\(\n(?P<body>.*?)\n\)", script, re.S)
    assert match is not None

    rows = [
        line.strip().strip('"')
        for line in match.group("body").splitlines()
        if line.strip().startswith('"')
    ]
    assert len(rows) == 6

    configs = []
    for row in rows:
        gpu, name, method, k_bits, v_bits, sink, recent, cache_path, cache_mode, mask = row.split()
        configs.append(
            {
                "gpu": int(gpu),
                "name": name,
                "method": method,
                "k_bits": int(k_bits),
                "v_bits": int(v_bits),
                "sink": int(sink),
                "recent": int(recent),
                "cache_path": cache_path,
                "cache_mode": cache_mode,
                "mask": mask,
            }
        )

    assert [cfg["gpu"] for cfg in configs] == [0, 1, 2, 3, 4, 5]
    assert {cfg["sink"] for cfg in configs} == {16, 32, 128}
    assert all(cfg["recent"] == 128 for cfg in configs)
    assert all((cfg["k_bits"], cfg["v_bits"]) == (2, 2) for cfg in configs)
    assert all(cfg["cache_path"] == "segmented" for cfg in configs)
    assert all(cfg["cache_mode"] == "segmented_rolling" for cfg in configs)
    assert all(cfg["mask"] == "none" for cfg in configs)
    assert "pattern_rolling_k2v2_s0_r128" not in {cfg["name"] for cfg in configs}
    assert "pattern_rolling_k2v2_s64_r128" not in {cfg["name"] for cfg in configs}
    assert "kivi_rolling_k2v2_s0_r128" not in {cfg["name"] for cfg in configs}
    assert "kivi_rolling_k2v2_s64_r128" not in {cfg["name"] for cfg in configs}


def test_sink_sweep_manifest_constants_cover_2_methods_and_5_sinks() -> None:
    assert EXPECTED_TASK_HASH == "ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e"
    assert EXPECTED_GENERATION_HASH == "a7d6b2f8bab37893b6331c66b3e5eb6a"
    assert len(LOGICAL_CONFIGS) == 10

    by_method = {}
    for cfg in LOGICAL_CONFIGS:
        by_method.setdefault(cfg["method_group"], []).append(cfg)
        assert cfg["cache_mode"] == "segmented_rolling"
        assert cfg["recent_length"] == 128
        assert cfg["group_size"] == 128
        assert (cfg["k_bits"], cfg["v_bits"]) == (2, 2)

    assert set(by_method) == {"PatternKV", "KIVI"}
    assert {cfg["sink_length"] for cfg in by_method["PatternKV"]} == {0, 16, 32, 64, 128}
    assert {cfg["sink_length"] for cfg in by_method["KIVI"]} == {0, 16, 32, 64, 128}
    assert sum(1 for cfg in LOGICAL_CONFIGS if cfg["result_source"] == "reused") == 4
    assert sum(1 for cfg in LOGICAL_CONFIGS if cfg["result_source"] == "newly_run") == 6


def test_sink_sweep_new_gpu_mapping_runs_only_missing_sinks() -> None:
    assert [row["gpu"] for row in NEW_GPU_MAPPING] == [0, 1, 2, 3, 4, 5]
    mapped = {row["config_name"] for row in NEW_GPU_MAPPING}
    assert mapped == {
        "pattern_rolling_k2v2_s16_r128",
        "pattern_rolling_k2v2_s32_r128",
        "pattern_rolling_k2v2_s128_r128",
        "kivi_rolling_k2v2_s16_r128",
        "kivi_rolling_k2v2_s32_r128",
        "kivi_rolling_k2v2_s128_r128",
    }
