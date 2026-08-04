#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.gsm8k_utils import extract_gold_answer, validate_gsm8k_rows


DATASET_ID = "modelscope/gsm8k"


def iter_records(dataset):
    if hasattr(dataset, "to_list"):
        yield from dataset.to_list()
        return
    if hasattr(dataset, "__iter__"):
        yield from dataset
        return
    raise TypeError(f"Unsupported ModelScope dataset object: {type(dataset)!r}")


def load_modelscope(split: str, revision: str | None):
    try:
        from modelscope.msdatasets import MsDataset
    except Exception as exc:
        raise SystemExit(
            "modelscope is not installed in this environment. Install it in the patternkv env "
            "and rerun scripts/prepare_gsm8k_modelscope.py."
        ) from exc
    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision
    attempts = [
        {"dataset_name": DATASET_ID, "trust_remote_code": True, **kwargs},
        {"dataset_name": DATASET_ID, "subset_name": "main", "trust_remote_code": True, **kwargs},
        {"dataset_name": "swift/gsm8k", **kwargs},
    ]
    last_exc = None
    for attempt in attempts:
        try:
            return MsDataset.load(**attempt), attempt
        except Exception as exc:
            last_exc = exc
    fallback = load_modelscope_cache_fallback(split)
    if fallback is not None:
        dataset, source = fallback
        return dataset, {"fallback": source, "last_ms_dataset_error": repr(last_exc)}
    raise RuntimeError(f"Failed to load GSM8K from ModelScope; last error: {last_exc}") from last_exc


def load_modelscope_cache_fallback(split: str):
    cache_root = Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"))
    candidates = []
    candidates.extend(cache_root.glob(f"hub/datasets/swift___gsm8k/**/gsm8k-{split}.arrow"))
    candidates.extend(cache_root.glob(f"hub/datasets/**/{split}-*.parquet"))
    candidates.extend(cache_root.glob(f"hub/datasets/downloads/*"))
    for path in candidates:
        try:
            if path.suffix == ".arrow":
                from datasets import Dataset

                return Dataset.from_file(str(path)), str(path)
            if path.suffix == ".parquet":
                import pandas as pd

                return pd.read_parquet(path).to_dict("records"), str(path)
        except Exception:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("reports/gsm8k_dataset_modelscope.md"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    dataset, attempt = load_modelscope(args.split, args.revision)
    rows = []
    for i, rec in enumerate(iter_records(dataset)):
        data = dict(rec)
        question = str(data.get("question") or "").strip()
        answer = str(data.get("answer") or data.get("target") or "").strip()
        gold = extract_gold_answer(answer)
        rows.append(
            {
                "sample_index": i,
                "question": question,
                "answer": answer,
                "gold_answer": gold,
            }
        )
    issues = validate_gsm8k_rows(rows, expected=1319)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    preview = [{k: type(v).__name__ for k, v in row.items()} for row in rows[:3]]
    lines = [
        "# GSM8K ModelScope Dataset Report",
        "",
        f"Generated at: {datetime.now().isoformat()}",
        f"Dataset ID: {DATASET_ID}",
        f"Load attempt: `{attempt}`",
        f"Revision: {args.revision or 'default'}",
        f"Split: {args.split}",
        f"Samples: {len(rows)}",
        f"Local path: {args.output}",
        "Hugging Face access: not used by this script",
        "",
        "## First Three Field Structures",
        "",
        "```json",
        json.dumps(preview, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Validation",
        "",
    ]
    if issues:
        lines.extend(["FAILED", "", "```json", json.dumps(issues, indent=2, ensure_ascii=False), "```"])
    else:
        lines.append("PASS: 1319 samples, unique sample_index, non-empty question/answer, parseable gold_answer.")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": len(rows), "issues": issues}, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
