"""Vendored LongBench configuration for the PatternKV paper v2 reproduction."""
import json
from pathlib import Path


_HERE = Path(__file__).parent

with (_HERE / "dataset2prompt.json").open(encoding="utf-8") as _f:
    PROMPT_TEMPLATES: dict[str, str] = json.load(_f)
with (_HERE / "dataset2maxlen.json").open(encoding="utf-8") as _f:
    MAX_NEW_TOKENS: dict[str, int] = json.load(_f)

PAPER_V2_SUBTASKS: tuple[str, ...] = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
)

LONG_BENCH_SUBSET_8X50: tuple[str, ...] = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "trec",
    "passage_retrieval_en",
    "lcc",
)

SUBTASKS = PAPER_V2_SUBTASKS

METRIC_NAMES: dict[str, str] = {
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "multifieldqa_zh": "qa_f1_zh",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    "dureader": "rouge_l_zh",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
    "vcsum": "rouge_l_zh",
    "trec": "classification",
    "triviaqa": "qa_f1",
    "samsum": "rouge_l",
    "lsht": "classification",
    "passage_count": "count",
    "passage_retrieval_en": "retrieval",
    "passage_retrieval_zh": "retrieval_zh",
    "lcc": "code_sim",
    "repobench-p": "code_sim",
}

LONGBENCH_PIN = "THUDM/LongBench main LongBench/config as of 2026-08-04"
DEFAULT_INPUT_CAP = 31500

PAPER_V2_EXPECTED_SAMPLES: dict[str, int] = {
    "multifieldqa_en": 150,
    "lcc": 500,
    "repobench-p": 500,
}


def expected_samples(task: str) -> int:
    return PAPER_V2_EXPECTED_SAMPLES.get(task, 200)


def assert_consistent() -> None:
    panel = set(SUBTASKS)
    assert panel.issubset(PROMPT_TEMPLATES), f"prompts missing {panel - set(PROMPT_TEMPLATES)}"
    assert panel.issubset(MAX_NEW_TOKENS), f"maxlen missing {panel - set(MAX_NEW_TOKENS)}"
    assert panel.issubset(METRIC_NAMES), f"metrics missing {panel - set(METRIC_NAMES)}"


assert_consistent()
