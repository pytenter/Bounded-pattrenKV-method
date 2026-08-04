from bench.longbench_config import MAX_NEW_TOKENS


def test_task_specific_max_gen():
    assert MAX_NEW_TOKENS["gov_report"] == 512
    assert MAX_NEW_TOKENS["lcc"] == 64
    assert MAX_NEW_TOKENS["passage_retrieval_en"] == 32
