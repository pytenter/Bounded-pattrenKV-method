from pathlib import Path

from scripts.summarize_longbench_8k_4090 import summarize


def test_summary_handles_empty_results_dir(tmp_path: Path):
    out = summarize(tmp_path)
    assert out["planned_total"] == 4750 * 3
    assert out["completed_total"] == 0
    assert out["oom_total"] == 0
    assert out["error_total"] == 0
    assert out["failed_tasks"]


def test_summary_supports_21x50_planned_accounting(tmp_path: Path):
    out = summarize(tmp_path, sample_limit_per_task=50)
    assert out["planned_total"] == 21 * 50 * 3
