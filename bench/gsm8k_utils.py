import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)
HASH_RE = re.compile(r"####\s*([^\n\r]+)")
ANSWER_RE = re.compile(r"(?:the\s+answer\s+is|answer\s*[:=])\s*([^\n\r.]+(?:\.[0-9]+)?)", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


def normalize_numeric_answer(value: str | int | float | Decimal | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("\\", "")
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.strip().rstrip(".")
    text = re.sub(r"\s+", "", text)
    if not text:
        return None
    try:
        dec = Decimal(text)
    except InvalidOperation:
        return None
    if dec == dec.to_integral_value():
        return str(int(dec))
    return format(dec.normalize(), "f").rstrip("0").rstrip(".")


def extract_gold_answer(answer: str) -> str | None:
    matches = HASH_RE.findall(answer or "")
    if matches:
        return normalize_numeric_answer(matches[-1])
    numbers = NUMBER_RE.findall(answer or "")
    if numbers:
        return normalize_numeric_answer(numbers[-1])
    return None


def parse_prediction(text: str) -> dict[str, Any]:
    text = text or ""
    for source, pattern in (
        ("boxed", BOXED_RE),
        ("hash", HASH_RE),
        ("answer_phrase", ANSWER_RE),
    ):
        matches = pattern.findall(text)
        if matches:
            parsed = normalize_numeric_answer(matches[-1])
            if parsed is not None:
                return {"parsed_answer": parsed, "parser_source": source, "parser_failure": False}
    numbers = NUMBER_RE.findall(text)
    if numbers:
        parsed = normalize_numeric_answer(numbers[-1])
        if parsed is not None:
            return {"parsed_answer": parsed, "parser_source": "last_number", "parser_failure": False}
    return {"parsed_answer": None, "parser_source": "failure", "parser_failure": True}


def normalize_eos_token_ids(*values: Any) -> list[int]:
    ids: list[int] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, int):
            ids.append(value)
            continue
        if isinstance(value, (list, tuple, set)):
            for item in value:
                if item is not None:
                    ids.append(int(item))
    return sorted(set(ids))


def compute_stop_state(generated_token_ids: list[int], max_new_tokens: int, eos_token_ids: list[int]) -> dict[str, Any]:
    output_tokens = len(generated_token_ids)
    last_generated_token_id = generated_token_ids[-1] if generated_token_ids else None
    eos_set = set(eos_token_ids)
    ended_with_eos = bool(generated_token_ids) and last_generated_token_id in eos_set
    hit_max_new_tokens = output_tokens >= max_new_tokens
    length_truncated = hit_max_new_tokens and not ended_with_eos
    if ended_with_eos:
        stop_reason = "eos"
    elif length_truncated:
        stop_reason = "length"
    elif output_tokens == 0:
        stop_reason = "unknown"
    else:
        stop_reason = "unknown"
    return {
        "last_generated_token_id": last_generated_token_id,
        "eos_token_ids": sorted(eos_set),
        "ended_with_eos": ended_with_eos,
        "hit_max_new_tokens": hit_max_new_tokens,
        "length_truncated": length_truncated,
        "stop_reason": stop_reason,
    }


def load_gsm8k_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_gsm8k_rows(rows: list[dict[str, Any]], expected: int = 1319) -> list[str]:
    issues = []
    if len(rows) != expected:
        issues.append(f"wrong_sample_count:{len(rows)}!=expected:{expected}")
    indices = [row.get("sample_index") for row in rows]
    if len(set(indices)) != len(indices):
        issues.append("duplicate_sample_index")
    if set(indices) != set(range(len(rows))):
        missing = sorted(set(range(len(rows))) - set(indices))[:20]
        extra = sorted(set(indices) - set(range(len(rows))))[:20]
        issues.append(f"sample_index_not_contiguous missing={missing} extra={extra}")
    for row in rows[:expected]:
        idx = row.get("sample_index")
        if not str(row.get("question") or "").strip():
            issues.append(f"empty_question:{idx}")
        if not str(row.get("answer") or "").strip():
            issues.append(f"empty_answer:{idx}")
        if normalize_numeric_answer(row.get("gold_answer")) is None:
            issues.append(f"unparseable_gold_answer:{idx}")
    return issues


def build_prompt(question: str, style: str = "concat") -> str:
    instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    if style == "newline":
        return f"{question}\n{instruction}"
    return f"{question}{instruction}"
