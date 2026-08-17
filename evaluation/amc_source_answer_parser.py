from __future__ import annotations

import re
from collections import Counter
from typing import Any


NORMALIZER_VERSION = "amc24_text_normalizer_v1"

ANSWER_LINE_RE = re.compile(
    r"(?:final\s+answer|answer|therefore,\s+the\s+answer\s+is|the\s+answer\s+is)\s*(?:is|:)?\s*(.+)",
    re.IGNORECASE,
)
SIMPLE_FRAC_COMMAND_RE = re.compile(r"\\frac([A-Za-z0-9])([A-Za-z0-9])")
SIMPLE_SQRT_COMMAND_RE = re.compile(r"\\sqrt([A-Za-z0-9])")
SIMPLE_SLASH_FRAC_RE = re.compile(r"(?<![A-Za-z0-9}])(-?\d+|\\pi|\\alpha)/(-?\d+)(?![A-Za-z0-9{])")


def boxed_contents(text: str) -> list[str]:
    out: list[str] = []
    i = 0
    marker = r"\boxed"
    while True:
        start = text.find(marker, i)
        if start < 0:
            break
        brace = text.find("{", start + len(marker))
        if brace < 0:
            break
        depth = 0
        for j in range(brace, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[brace + 1 : j])
                    i = j + 1
                    break
        else:
            break
    return out


def _strip_outer_math_wrappers(text: str) -> str:
    changed = True
    while changed:
        changed = False
        stripped = text.strip()
        for left, right in (("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
            if stripped.startswith(left) and stripped.endswith(right):
                text = stripped[len(left) : len(stripped) - len(right)]
                changed = True
                break
        if changed:
            continue
        boxes = boxed_contents(stripped)
        if len(boxes) == 1 and stripped == rf"\boxed{{{boxes[0]}}}":
            text = boxes[0]
            changed = True
    return text.strip()


def normalize_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("−", "-").replace("π", r"\pi").replace("α", r"\alpha")
    text = _strip_outer_math_wrappers(text)
    text = text.replace("$", "")
    text = re.sub(r"\\(?:dfrac|tfrac)\b", r"\\frac", text)
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\left|\\right", "", text)
    text = text.strip()
    text = text.rstrip(".。")
    text = re.sub(r"\s+", "", text)
    text = SIMPLE_FRAC_COMMAND_RE.sub(r"\\frac{\1}{\2}", text)
    text = SIMPLE_SQRT_COMMAND_RE.sub(r"\\sqrt{\1}", text)
    while True:
        new_text = SIMPLE_SLASH_FRAC_RE.sub(r"\\frac{\1}{\2}", text)
        if new_text == text:
            break
        text = new_text
    if re.fullmatch(r"[A-Za-z]", text):
        text = text.upper()
    return text or None


def normalize_amc_source_answer(value: Any) -> str | None:
    return normalize_answer(value)


def _clean_candidate(value: str) -> str:
    text = value.strip()
    text = text.splitlines()[0].strip()
    text = re.split(r"(?:\n\n|(?<!\d)\.(?:\s|$))", text, maxsplit=1)[0].strip()
    return text


def extract_final_answer(text: str) -> dict[str, Any]:
    text = text or ""
    boxed = boxed_contents(text)
    if boxed:
        return {
            "answer": boxed[-1],
            "parser_strategy": "last_boxed",
            "parser_error": None,
            "boxed_candidates": boxed,
        }

    answer_line_candidates = []
    for match in ANSWER_LINE_RE.finditer(text):
        raw = _clean_candidate(match.group(1))
        answer_line_candidates.append(raw)
    for item in reversed(answer_line_candidates):
        if normalize_answer(item) is not None:
            return {
                "answer": item,
                "parser_strategy": "explicit_final_answer_line",
                "parser_error": None,
                "boxed_candidates": boxed,
                "answer_line_candidates": answer_line_candidates,
            }

    return {
        "answer": None,
        "parser_strategy": "failure",
        "parser_error": "no_boxed_or_explicit_final_answer",
        "boxed_candidates": boxed,
        "answer_line_candidates": answer_line_candidates,
    }


def parse_amc_source_answer(text: str) -> dict[str, Any]:
    extracted = extract_final_answer(text)
    parsed_answer = normalize_answer(extracted["answer"])
    return {
        "parsed_answer": parsed_answer,
        "raw_answer": extracted["answer"],
        "canonical_answer_key": parsed_answer,
        "normalizer_version": NORMALIZER_VERSION,
        "parser_strategy": extracted["parser_strategy"] if parsed_answer is not None else "failure",
        "parser_error": extracted["parser_error"] if parsed_answer is None else None,
        "boxed_candidates": extracted.get("boxed_candidates", []),
        "answer_line_candidates": extracted.get("answer_line_candidates", []),
    }


def score_response(generated_text: str, gold_answer: str) -> dict[str, Any]:
    parsed = parse_amc_source_answer(generated_text)
    gold_key = normalize_answer(gold_answer)
    pred_key = parsed["canonical_answer_key"]
    return {
        **parsed,
        "gold_canonical_answer_key": gold_key,
        "correct": pred_key is not None and gold_key is not None and pred_key == gold_key,
    }


def majority_vote_canonical(parsed_answer_keys: list[str | None]) -> dict[str, Any]:
    valid = [key for key in parsed_answer_keys if key is not None]
    if not valid:
        return {"prediction": None, "tie": False, "votes": {}, "valid_votes": 0}
    counts = Counter(valid)
    top = max(counts.values())
    winners = sorted(key for key, count in counts.items() if count == top)
    return {
        "prediction": winners[0] if len(winners) == 1 else None,
        "tie": len(winners) > 1,
        "votes": dict(sorted(counts.items())),
        "valid_votes": len(valid),
    }


def score_majority(parsed_answer_keys: list[str | None], gold_answer: str) -> dict[str, Any]:
    vote = majority_vote_canonical(parsed_answer_keys)
    gold_key = normalize_answer(gold_answer)
    prediction = vote["prediction"]
    return {
        **vote,
        "gold_canonical_answer_key": gold_key,
        "correct": prediction is not None and gold_key is not None and prediction == gold_key,
    }
