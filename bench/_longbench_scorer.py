"""Vendored LongBench scorer.

Metric primitives mirror THUDM/LongBench metrics.py and eval.py routing at
commit 2e00731f8d0bff23dc4325161044d0ed8af94c1e. Scores are on the LongBench
[0, 100] scale.
"""
import re
import string
from collections import Counter
from typing import List, Optional, Sequence

from fuzzywuzzy import fuzz
from rouge import Rouge
try:
    import jieba
except Exception:  # pragma: no cover - only used on zh LongBench tasks.
    jieba = None

from bench.longbench_config import METRIC_NAMES, SUBTASKS


def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _normalize_zh_answer(s: str) -> str:
    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        return "".join(ch for ch in text if ch not in set(string.punctuation + cn_punctuation))

    return white_space_fix(remove_punc(s.lower()))


def _f1_score(prediction_tokens, ground_truth_tokens) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    p_tok = _normalize_answer(prediction).split()
    g_tok = _normalize_answer(ground_truth).split()
    if not p_tok or not g_tok:
        return 0.0
    return _f1_score(p_tok, g_tok)


def _qa_f1_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    if jieba is None:
        p_tok = list(_normalize_zh_answer(prediction))
        g_tok = list(_normalize_zh_answer(ground_truth))
    else:
        p_tok = [_normalize_zh_answer(tok) for tok in jieba.cut(prediction, cut_all=False)]
        g_tok = [_normalize_zh_answer(tok) for tok in jieba.cut(ground_truth, cut_all=False)]
        p_tok = [tok for tok in p_tok if tok]
        g_tok = [tok for tok in g_tok if tok]
    if not p_tok or not g_tok:
        return 0.0
    return _f1_score(p_tok, g_tok)


def _rouge_score(prediction: str, ground_truth: str, **kwargs) -> float:
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def _rouge_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    if jieba is None:
        prediction = " ".join(_normalize_zh_answer(prediction))
        ground_truth = " ".join(_normalize_zh_answer(ground_truth))
    else:
        prediction = " ".join(jieba.cut(prediction, cut_all=False))
        ground_truth = " ".join(jieba.cut(ground_truth, cut_all=False))
    return _rouge_score(prediction, ground_truth)


def _count_score(prediction: str, ground_truth: str, **kwargs) -> float:
    nums = re.findall(r"\d+", prediction)
    if not nums:
        return 0.0
    return sum(1 for n in nums if str(n) == str(ground_truth)) / len(nums)


def _retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    gt_id = matches[0]
    nums = re.findall(r"\d+", prediction)
    if not nums:
        return 0.0
    right = sum(1 for n in nums if str(n) == str(gt_id))
    return right / len(nums)


def _retrieval_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    matches = re.findall(r"段落(\d+)", ground_truth)
    if not matches:
        return 0.0
    gt_id = matches[0]
    nums = re.findall(r"\d+", prediction)
    if not nums:
        return 0.0
    return sum(1 for n in nums if str(n) == str(gt_id)) / len(nums)


def _code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    pred = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            pred = line
            break
    return fuzz.ratio(pred, ground_truth) / 100.0


def _classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    all_classes = kwargs.get("all_classes") or []
    em = [c for c in all_classes if c in prediction]
    em = [c for c in em if not (c in ground_truth and c != ground_truth)]
    if ground_truth in em:
        return 1.0 / len(em)
    return 0.0


_DATASET2METRIC = {
    "narrativeqa": _qa_f1_score,
    "qasper": _qa_f1_score,
    "multifieldqa_en": _qa_f1_score,
    "multifieldqa_zh": _qa_f1_zh_score,
    "hotpotqa": _qa_f1_score,
    "2wikimqa": _qa_f1_score,
    "musique": _qa_f1_score,
    "dureader": _rouge_zh_score,
    "gov_report": _rouge_score,
    "qmsum": _rouge_score,
    "multi_news": _rouge_score,
    "vcsum": _rouge_zh_score,
    "trec": _classification_score,
    "triviaqa": _qa_f1_score,
    "samsum": _rouge_score,
    "lsht": _classification_score,
    "passage_count": _count_score,
    "passage_retrieval_en": _retrieval_score,
    "passage_retrieval_zh": _retrieval_zh_score,
    "lcc": _code_sim_score,
    "repobench-p": _code_sim_score,
}

_FIRST_LINE_ONLY = {"trec", "triviaqa", "samsum", "lsht"}


def _per_example_score(
    subtask: str,
    prediction: str,
    ground_truths: Sequence[str],
    all_classes: Optional[Sequence[str]],
) -> float:
    pred = prediction
    if subtask in _FIRST_LINE_ONLY:
        pred = pred.lstrip("\n").split("\n")[0]
    fn = _DATASET2METRIC[subtask]
    best = 0.0
    for gt in ground_truths:
        best = max(best, fn(pred, gt, all_classes=all_classes))
    return best


def score_subtask(
    subtask: str,
    predictions: List[str],
    references: List[List[str]],
    all_classes: Optional[List[str]] = None,
) -> dict:
    if subtask not in _DATASET2METRIC:
        raise ValueError(f"Unsupported subtask {subtask!r}. Known: {sorted(SUBTASKS)}")
    if len(predictions) != len(references):
        raise ValueError(f"len(predictions)={len(predictions)} != len(references)={len(references)}")
    if not predictions:
        return {"score": float("nan"), "metric": METRIC_NAMES[subtask], "n": 0}
    total = 0.0
    for pred, refs in zip(predictions, references):
        if isinstance(refs, str):
            refs = [refs]
        total += _per_example_score(subtask, pred, refs, all_classes)
    return {"score": round(100.0 * total / len(predictions), 2), "metric": METRIC_NAMES[subtask], "n": len(predictions)}


def score_example(
    subtask: str,
    prediction: str,
    references: List[str],
    all_classes: Optional[List[str]] = None,
) -> float:
    return round(100.0 * _per_example_score(subtask, prediction, references, all_classes), 2)


__all__ = ["score_subtask", "score_example", "SUBTASKS", "METRIC_NAMES"]
