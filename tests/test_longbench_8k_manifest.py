from bench.longbench_config import SUBTASKS, expected_samples
from scripts.run_longbench_paper_8k_single4090 import METHODS


def test_8k_manifest_total_is_full_longbench_panel():
    assert len(SUBTASKS) == 21
    assert sum(expected_samples(t) for t in SUBTASKS) == 4750
    assert len(METHODS) == 3
