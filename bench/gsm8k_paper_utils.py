from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


METHODS = ("fp16", "kivi_paper_g128", "patternkv_paper")
EXPECTED_GSM8K_TEST = 1319

HASH_RE = re.compile(r"####\s*([^\n\r]+)")
FINAL_RE = re.compile(r"(?:final\s+answer|answer\s*(?:is|:|=))[^-+0-9]{0,80}([-+]?\$?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


def normalize_numeric_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace("$", "").replace(",", "").rstrip(".")
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


def extract_reference(answer: str) -> str | None:
    matches = HASH_RE.findall(answer or "")
    if matches:
        return normalize_numeric_answer(matches[-1])
    nums = NUMBER_RE.findall(answer or "")
    return normalize_numeric_answer(nums[-1]) if nums else None


def parse_prediction(text: str) -> dict[str, Any]:
    text = text or ""
    final = FINAL_RE.findall(text)
    if final:
        parsed = normalize_numeric_answer(final[-1])
        if parsed is not None:
            return {"parsed_answer": parsed, "parser_strategy": "final_answer", "parser_error": None}
    tail = text[-500:]
    nums = NUMBER_RE.findall(tail)
    if nums:
        parsed = normalize_numeric_answer(nums[-1])
        if parsed is not None:
            return {"parsed_answer": parsed, "parser_strategy": "tail_number", "parser_error": None}
    return {"parsed_answer": None, "parser_strategy": "failure", "parser_error": "no_numeric_answer"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gsm8k(path: Path = Path("datasets/gsm8k/gsm8k_test.jsonl")) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != EXPECTED_GSM8K_TEST:
        raise ValueError(f"expected {EXPECTED_GSM8K_TEST} GSM8K rows, got {len(rows)}")
    seen = set()
    for i, row in enumerate(rows):
        if int(row.get("problem_id", -1)) != i:
            raise ValueError(f"problem_id mismatch at row {i}")
        if not str(row.get("question") or "").strip():
            raise ValueError(f"empty question at row {i}")
        if extract_reference(str(row.get("answer") or "")) is None:
            raise ValueError(f"unparseable answer at row {i}")
        q = row["question"]
        if q in seen:
            raise ValueError(f"duplicate question at row {i}")
        seen.add(q)
    return rows


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_prompt(question: str, tokenizer) -> tuple[str, str]:
    user_prompt = f"{question}\n\nLet's think step by step."
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": user_prompt}], tokenize=False, add_generation_prompt=True)
    return rendered, user_prompt


def manifest(rows: list[dict[str, Any]], methods=METHODS, cfg_hash: str = "") -> list[dict[str, Any]]:
    out = []
    for method in methods:
        for row in rows:
            pid = int(row["problem_id"])
            out.append({"dataset": "gsm8k", "split": "test", "method": method, "problem_id": pid, "task_key": f"gsm8k:p{pid}", "config_hash": cfg_hash})
    return out


def result_path(root: Path, method: str, problem_id: int, cfg_hash: str) -> Path:
    return root / method / f"p{problem_id:04d}_{cfg_hash}.json"


def is_complete(path: Path, cfg_hash: str, retry_failed: bool = False, retry_oom: bool = False) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if row.get("config_hash") != cfg_hash:
        return False
    if row.get("stop_reason") == "oom":
        return not retry_oom
    if row.get("stop_reason") == "error" or row.get("error"):
        return not retry_failed
    required = ("problem_id", "method", "generated_text", "parsed_answer", "stop_reason", "is_correct")
    return all(k in row for k in required)


def compute_stop_state(generated_token_ids: list[int], max_new_tokens: int, eos_token_ids: list[int]) -> dict[str, Any]:
    output_tokens = len(generated_token_ids)
    last = generated_token_ids[-1] if generated_token_ids else None
    ended = bool(generated_token_ids) and last in set(eos_token_ids)
    hit = output_tokens >= max_new_tokens
    if ended:
        reason = "eos"
    elif hit:
        reason = "length"
    else:
        reason = "unknown"
    return {"last_generated_token_id": last, "eos_token_ids": sorted(set(eos_token_ids)), "ended_with_eos": ended, "hit_max_new_tokens": hit, "stop_reason": reason}


def majority_counts(values: list[bool]) -> dict[str, int]:
    c = Counter(values)
    return {"true": c[True], "false": c[False]}
