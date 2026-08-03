from pathlib import Path

from bench.aime_utils import is_complete_result, write_json_atomic


def test_complete_result_skipped(tmp_path: Path):
    path = tmp_path / "r.json"
    write_json_atomic(path, {"task_key": "aime24:p0:s0", "method": "fp16", "problem_id": 0, "sample_id": 0, "config_hash": "abc", "generated_text": "x", "stop_reason": "eos", "parsed_answer": "1"})
    assert is_complete_result(path, "abc")


def test_config_hash_mismatch_not_skipped(tmp_path: Path):
    path = tmp_path / "r.json"
    write_json_atomic(path, {"task_key": "aime24:p0:s0", "method": "fp16", "problem_id": 0, "sample_id": 0, "config_hash": "abc", "generated_text": "x", "stop_reason": "eos", "parsed_answer": "1"})
    assert not is_complete_result(path, "def")


def test_oom_retry_flag(tmp_path: Path):
    path = tmp_path / "r.json"
    write_json_atomic(path, {"task_key": "aime24:p0:s0", "method": "fp16", "problem_id": 0, "sample_id": 0, "config_hash": "abc", "generated_text": "", "stop_reason": "oom", "parsed_answer": None})
    assert is_complete_result(path, "abc")
    assert not is_complete_result(path, "abc", retry_oom=True)
