"""Vendored LongBench-EN configuration for the PatternKV reproduction."""
import json
from pathlib import Path


_HERE = Path(__file__).parent

with (_HERE / "dataset2prompt.json").open(encoding="utf-8") as _f:
    PROMPT_TEMPLATES: dict[str, str] = json.load(_f)
with (_HERE / "dataset2maxlen.json").open(encoding="utf-8") as _f:
    MAX_NEW_TOKENS: dict[str, int] = json.load(_f)

SUBTASKS: tuple[str, ...] = (
    "2wikimqa",
    "gov_report",
    "hotpotqa",
    "lcc",
    "multifieldqa_en",
    "passage_retrieval_en",
    "qasper",
    "trec",
)

METRIC_NAMES: dict[str, str] = {
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "gov_report": "rouge_l",
    "trec": "classification",
    "passage_retrieval_en": "retrieval",
    "lcc": "code_sim",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
    "triviaqa": "qa_f1",
    "samsum": "rouge_l",
    "repobench-p": "code_sim",
}

LONGBENCH_PIN = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
DEFAULT_INPUT_CAP = 31500


def assert_consistent() -> None:
    panel = set(SUBTASKS)
    assert panel.issubset(PROMPT_TEMPLATES), f"prompts missing {panel - set(PROMPT_TEMPLATES)}"
    assert panel.issubset(MAX_NEW_TOKENS), f"maxlen missing {panel - set(MAX_NEW_TOKENS)}"
    assert panel.issubset(METRIC_NAMES), f"metrics missing {panel - set(METRIC_NAMES)}"


assert_consistent()
