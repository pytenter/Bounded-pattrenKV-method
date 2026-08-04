from bench.longbench_config import PROMPT_TEMPLATES, SUBTASKS


def test_prompts_cover_all_tasks():
    assert set(SUBTASKS).issubset(PROMPT_TEMPLATES)
    assert "{context}" in PROMPT_TEMPLATES["qasper"]
    assert "{input}" in PROMPT_TEMPLATES["repobench-p"]
