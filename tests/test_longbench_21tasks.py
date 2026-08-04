from bench.longbench_config import SUBTASKS


def test_longbench_21_tasks():
    assert len(SUBTASKS) == 21
    assert "repobench-p" in SUBTASKS
    assert "passage_retrieval_zh" in SUBTASKS
