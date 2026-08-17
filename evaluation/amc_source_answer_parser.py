from __future__ import annotations

import re
from typing import Any


ANSWER_LINE_RE = re.compile(
    r"(?:final\s+answer|answer|therefore,\s+the\s+answer\s+is|the\s+answer\s+is)\s*(?:is|:)?\s*(.+)",
    re.IGNORECASE,
)


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


def normalize_amc_source_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("$", "")
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\left|\\right", "", text)
    text = text.strip()
    text = text.rstrip(".。")
    text = re.sub(r"\s+", "", text)
    return text or None


def _clean_candidate(value: str) -> str:
    text = value.strip()
    text = text.splitlines()[0].strip()
    text = re.split(r"(?:\n\n|(?<!\d)\.(?:\s|$))", text, maxsplit=1)[0].strip()
    return text


def parse_amc_source_answer(text: str) -> dict[str, Any]:
    text = text or ""
    boxed = [{"raw": item, "parsed": normalize_amc_source_answer(item)} for item in boxed_contents(text)]
    valid_boxed = [item for item in boxed if item["parsed"] is not None]
    if valid_boxed:
        item = valid_boxed[-1]
        return {
            "parsed_answer": item["parsed"],
            "parser_strategy": "last_boxed",
            "parser_error": None,
            "boxed_candidates": boxed,
        }

    answer_line_candidates = []
    for match in ANSWER_LINE_RE.finditer(text):
        raw = _clean_candidate(match.group(1))
        parsed = normalize_amc_source_answer(raw)
        answer_line_candidates.append({"raw": raw, "parsed": parsed})
    for item in reversed(answer_line_candidates):
        if item["parsed"] is not None:
            return {
                "parsed_answer": item["parsed"],
                "parser_strategy": "explicit_final_answer_line",
                "parser_error": None,
                "boxed_candidates": boxed,
                "answer_line_candidates": answer_line_candidates,
            }

    return {
        "parsed_answer": None,
        "parser_strategy": "failure",
        "parser_error": "no_boxed_or_explicit_final_answer",
        "boxed_candidates": boxed,
        "answer_line_candidates": answer_line_candidates,
    }
