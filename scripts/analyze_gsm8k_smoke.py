#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.gsm8k_utils import BOXED_RE, parse_prediction


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_method(root: Path, method: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / method).glob("shard_*.jsonl")):
        rows.extend(read_jsonl(path))
    return sorted(rows, key=lambda r: r.get("sample_index", -1))


def sha256_or_size(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return {"path": str(path), "exists": True, "bytes": path.stat().st_size, "sha256": h.hexdigest()}
    files = sorted(p for p in path.rglob("*") if p.is_file())
    return {"path": str(path), "exists": True, "files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def repeated_sentences(text: str) -> bool:
    parts = [p.strip().lower() for p in re.split(r"(?<=[.!?])\s+", text or "") if len(p.strip()) > 20]
    counts = Counter(parts)
    return any(count >= 3 for count in counts.values())


def looping(text: str) -> bool:
    text = text or ""
    tail = text[-1800:].lower()
    markers = ["however, this is", "we need to", "this is still not", "we can set up the equation"]
    return repeated_sentences(text) or any(tail.count(marker) >= 3 for marker in markers)


def truncation_row(row: dict[str, Any]) -> dict[str, Any]:
    pred = row.get("prediction") or ""
    boxed = list(BOXED_RE.finditer(pred))
    parsed_boxed = None
    if boxed:
        parsed_boxed = parse_prediction(boxed[-1].group(0))["parsed_answer"]
    has_nan = bool(re.search(r"\b(?:nan|inf)\b", pred, re.IGNORECASE))
    return {
        "method": row.get("method"),
        "sample_index": row.get("sample_index"),
        "correct": row.get("correct"),
        "parsed_answer": row.get("parsed_answer"),
        "gold_answer": row.get("gold_answer"),
        "parser_source": row.get("parser_source"),
        "parser_failure": row.get("parser_failure"),
        "output_tokens": row.get("output_tokens"),
        "max_new_tokens": row.get("max_new_tokens"),
        "last_generated_token_id": row.get("last_generated_token_id"),
        "ended_with_eos": row.get("ended_with_eos"),
        "hit_max_new_tokens": row.get("hit_max_new_tokens", (row.get("output_tokens") or 0) >= (row.get("max_new_tokens") or 10**9)),
        "length_truncated": row.get("length_truncated"),
        "stop_reason": row.get("stop_reason"),
        "prediction_tail_500": pred[-500:],
        "has_boxed": bool(boxed),
        "last_boxed_char_pos": boxed[-1].start() if boxed else None,
        "last_boxed_parsed_answer": parsed_boxed,
        "boxed_answer_correct": parsed_boxed == str(row.get("gold_answer")) if parsed_boxed is not None else None,
        "has_repeated_sentence": repeated_sentences(pred),
        "has_looping_reasoning": looping(pred),
        "has_nan_or_inf": has_nan,
    }


def classify_kivi(row: dict[str, Any]) -> str:
    t = truncation_row(row)
    if t["has_boxed"] and t["boxed_answer_correct"]:
        return "A_correct_boxed_then_continued"
    if t["has_boxed"] and t["boxed_answer_correct"] is False:
        return "B_wrong_boxed_then_continued"
    if row.get("parsed_answer") is not None and row.get("parser_failure"):
        return "C_answer_but_parser_missed"
    if t["has_looping_reasoning"] or t["has_repeated_sentence"]:
        return "E_repetition_or_loop"
    if row.get("parsed_answer") is None:
        return "D_no_final_answer"
    return "F_other"


def method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outs = [r.get("output_tokens") or 0 for r in rows]
    eos = [r for r in rows if r.get("ended_with_eos")]
    return {
        "rows": len(rows),
        "correct": sum(1 for r in rows if r.get("correct")),
        "accuracy": round(100 * sum(1 for r in rows if r.get("correct")) / len(rows), 3) if rows else None,
        "length_truncated": sum(1 for r in rows if r.get("length_truncated")),
        "parser_failures": sum(1 for r in rows if r.get("parser_failure")),
        "errors": sum(1 for r in rows if r.get("error")),
        "avg_output_tokens": round(mean(outs), 2) if outs else None,
        "median_output_tokens": round(median(outs), 2) if outs else None,
        "p95_output_tokens": sorted(outs)[int(0.95 * (len(outs) - 1))] if outs else None,
        "max_output_tokens": max(outs) if outs else None,
        "normal_eos_rate": round(100 * len(eos) / len(rows), 3) if rows else None,
    }


def write_truncation_analysis(root: Path, md_path: Path, json_path: Path, csv_path: Path) -> None:
    data: dict[str, Any] = {"root": str(root), "methods": {}, "truncated_samples": []}
    for method in ["fp16", "kivi", "patternkv"]:
        rows = read_method(root, method)
        truncated = [truncation_row(r) for r in rows if r.get("hit_max_new_tokens") or r.get("length_truncated") or (r.get("output_tokens") or 0) >= (r.get("max_new_tokens") or 10**9)]
        data["methods"][method] = {"summary": method_summary(rows), "truncated_count": len(truncated)}
        data["truncated_samples"].extend(truncated)
        if method == "kivi":
            classes = Counter(classify_kivi(r) for r in rows if r.get("length_truncated"))
            data["methods"][method]["kivi_truncated_classes"] = dict(classes)
            data["methods"][method]["kivi_truncated_boxed"] = {
                "boxed_count": sum(1 for r in rows if r.get("length_truncated") and truncation_row(r)["has_boxed"]),
                "boxed_correct_count": sum(1 for r in rows if r.get("length_truncated") and truncation_row(r)["boxed_answer_correct"] is True),
                "boxed_wrong_count": sum(1 for r in rows if r.get("length_truncated") and truncation_row(r)["boxed_answer_correct"] is False),
                "no_boxed_count": sum(1 for r in rows if r.get("length_truncated") and not truncation_row(r)["has_boxed"]),
                "loop_or_repetition_count": sum(1 for r in rows if r.get("length_truncated") and truncation_row(r)["has_looping_reasoning"]),
                "parser_failure_count": sum(1 for r in rows if r.get("length_truncated") and r.get("parser_failure")),
            }
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = list(data["truncated_samples"][0]) if data["truncated_samples"] else ["method", "sample_index"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data["truncated_samples"])
    lines = ["# GSM8K Smoke Truncation Analysis", "", f"- root: `{root}`", ""]
    for method, item in data["methods"].items():
        lines.append(f"## {method}")
        lines.append("")
        lines.append(f"- summary: `{item['summary']}`")
        lines.append(f"- reached_limit_or_truncated_rows: `{item['truncated_count']}`")
        if method == "kivi":
            lines.append(f"- kivi_truncated_classes: `{item.get('kivi_truncated_classes')}`")
            lines.append(f"- kivi_boxed_stats: `{item.get('kivi_truncated_boxed')}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def write_compare(root_a: Path, root_b: Path, md_path: Path, json_path: Path) -> None:
    out: dict[str, Any] = {"a": str(root_a), "b": str(root_b), "methods": {}, "paired": defaultdict(list)}
    for method in ["fp16", "kivi", "patternkv"]:
        rows_a = {r["sample_index"]: r for r in read_method(root_a, method)}
        rows_b = {r["sample_index"]: r for r in read_method(root_b, method)}
        out["methods"][method] = {"a": method_summary(list(rows_a.values())), "b": method_summary(list(rows_b.values()))}
        for idx in sorted(set(rows_a) & set(rows_b)):
            a = rows_a[idx]
            b = rows_b[idx]
            out["paired"][method].append({
                "sample_index": idx,
                "same_answer": a.get("parsed_answer") == b.get("parsed_answer"),
                "correct_changed": a.get("correct") != b.get("correct"),
                "a_truncated": a.get("length_truncated"),
                "b_truncated": b.get("length_truncated"),
                "output_token_delta": (b.get("output_tokens") or 0) - (a.get("output_tokens") or 0),
                "a_has_boxed": bool(BOXED_RE.search(a.get("prediction") or "")),
                "b_final_answer": b.get("parsed_answer"),
            })
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    lines = ["# GSM8K Smoke Comparison", "", f"- A: `{root_a}`", f"- B: `{root_b}`", ""]
    lines.append("| method | A rows | A acc | A trunc | B rows | B acc | B trunc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method, item in out["methods"].items():
        a = item["a"]
        b = item["b"]
        lines.append(f"| {method} | {a['rows']} | {a['accuracy']} | {a['length_truncated']} | {b['rows']} | {b['accuracy']} | {b['length_truncated']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("truncation")
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--md", type=Path, required=True)
    a.add_argument("--json", type=Path, required=True)
    a.add_argument("--csv", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--a", type=Path, required=True)
    c.add_argument("--b", type=Path, required=True)
    c.add_argument("--md", type=Path, required=True)
    c.add_argument("--json", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "truncation":
        write_truncation_analysis(args.root, args.md, args.json, args.csv)
    else:
        write_compare(args.a, args.b, args.md, args.json)


if __name__ == "__main__":
    main()
