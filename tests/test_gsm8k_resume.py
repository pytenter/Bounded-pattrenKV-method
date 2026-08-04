from pathlib import Path

from bench.gsm8k_paper_utils import is_complete, write_json_atomic


def test_resume_config_hash(tmp_path: Path):
    p = tmp_path / "r.json"
    write_json_atomic(p, {"problem_id": 0, "method": "fp16", "config_hash": "a", "generated_text": "x", "parsed_answer": "1", "stop_reason": "eos", "is_correct": True})
    assert is_complete(p, "a")
    assert not is_complete(p, "b")


def test_oom_retry(tmp_path: Path):
    p = tmp_path / "r.json"
    write_json_atomic(p, {"problem_id": 0, "method": "fp16", "config_hash": "a", "generated_text": "", "parsed_answer": None, "stop_reason": "oom", "is_correct": False})
    assert is_complete(p, "a")
    assert not is_complete(p, "a", retry_oom=True)
