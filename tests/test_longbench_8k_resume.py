from scripts.run_longbench_paper_8k_single4090 import row_is_complete


def test_resume_requires_matching_config_hash_and_success():
    row = {"config_hash": "a", "stop_reason": "eos", "experiment_id": "x", "method": "fp16", "task": "trec", "sample_id": "s", "prediction": "x", "metric_name": "classification"}
    assert row_is_complete(row, "a")
    assert not row_is_complete(row, "b")
    row["stop_reason"] = "oom"
    assert not row_is_complete(row, "a")
    row["stop_reason"] = "error"
    assert not row_is_complete(row, "a")
