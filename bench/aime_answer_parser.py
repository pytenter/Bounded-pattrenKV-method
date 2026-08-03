from __future__ import annotations

import re
from fractions import Fraction
from typing import Any


FINAL_RE = re.compile(r"final\s+answer[^-+0-9]{0,80}([-+]?\d[\d,\s]*)", re.IGNORECASE | re.DOTALL)
INTEGER_RE = re.compile(r"(?<![\w.])-?\d[\d,\s]*(?![\w.])")
FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def _boxed_contents(text: str) -> list[str]:
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


def normalize_aime_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("$", "").replace(",", "").replace(" ", "")
    text = text.strip(".。")
    frac = FRAC_RE.fullmatch(text)
    try:
        if frac:
            val = Fraction(int(frac.group(1)), int(frac.group(2)))
        else:
            text = text.replace("\\", "")
            if "/" in text and re.fullmatch(r"[-+]?\d+/[-+]?\d+", text):
                val = Fraction(text)
            elif re.fullmatch(r"[-+]?\d+", text):
                val = Fraction(int(text), 1)
            else:
                return None
    except Exception:
        return None
    if val.denominator != 1:
        return None
    number = int(val)
    if not (0 <= number <= 999):
        return None
    return str(number)


def parse_aime_answer(text: str) -> dict[str, Any]:
    text = text or ""
    boxed = _boxed_contents(text)
    boxed_candidates = []
    for item in boxed:
        parsed = normalize_aime_answer(item)
        boxed_candidates.append({"raw": item, "parsed": parsed})
    for item in reversed(boxed_candidates):
        if item["parsed"] is not None:
            return {
                "parsed_answer": item["parsed"],
                "parser_strategy": "boxed",
                "parser_error": None,
                "boxed_candidates": boxed_candidates,
            }

    final_matches = FINAL_RE.findall(text)
    for item in reversed(final_matches):
        parsed = normalize_aime_answer(item)
        if parsed is not None:
            return {
                "parsed_answer": parsed,
                "parser_strategy": "final_answer_nearby_integer",
                "parser_error": None,
                "boxed_candidates": boxed_candidates,
            }

    tail = text[-500:]
    numbers = INTEGER_RE.findall(tail)
    for item in reversed(numbers):
        parsed = normalize_aime_answer(item)
        if parsed is not None:
            return {
                "parsed_answer": parsed,
                "parser_strategy": "tail_integer",
                "parser_error": None,
                "boxed_candidates": boxed_candidates,
            }
    return {
        "parsed_answer": None,
        "parser_strategy": "failure",
        "parser_error": "no_valid_aime_integer_0_999",
        "boxed_candidates": boxed_candidates,
    }
