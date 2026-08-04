from pathlib import Path


def test_result_dirs_are_isolated():
    assert Path("results/paper_repro_v2/longbench_full_strict") != Path("results/paper_repro_v2/longbench_full_8k_cap")
