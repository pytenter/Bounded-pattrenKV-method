from pathlib import Path


def test_8k_outputs_are_isolated_from_strict_outputs():
    full = Path("results/paper_repro_v2/longbench_full_8k_4090")
    strict = Path("results/paper_repro_v2/longbench_full_strict")
    assert full != strict
    assert "strict" not in str(full)


def test_smoke_outputs_are_not_formal_outputs():
    assert Path("results/paper_repro_v2/longbench_8k_4090_smoke") != Path("results/paper_repro_v2/longbench_full_8k_4090")
    assert Path("results/paper_repro_v2/longbench_8k_4090_edge_smoke") != Path("results/paper_repro_v2/longbench_full_8k_4090")
