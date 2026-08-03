#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("/data/zypan/kvarn-repro/datasets/aime/aime24.jsonl"))
    p.add_argument("--output", type=Path, default=Path("datasets/aime/aime24.jsonl"))
    args = p.parse_args()
    if not args.source.exists():
        raise SystemExit("DATASET_NOT_AVAILABLE: source file missing and this script will not fabricate AIME data")
    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 30:
        raise SystemExit(f"DATASET_INVALID: expected 30 rows, found {len(rows)}")
    out = []
    for i, row in enumerate(rows):
        problem = row.get("problem") or row.get("question")
        answer = row.get("answer") or row.get("reference_answer")
        if not problem or answer is None:
            raise SystemExit(f"DATASET_INVALID: missing problem/answer at row {i}")
        out.append({"dataset": "aime24", "problem_id": i, "problem": problem, "answer": str(answer)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n", encoding="utf-8")
    meta = {
        "source": str(args.source),
        "upstream_source": "HuggingFaceH4/aime_2024",
        "source_config": None,
        "split": "train",
        "revision": "main",
        "num_examples": len(out),
        "sha256": sha256(args.output),
        "source_sha256_original_file": sha256(args.source),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.with_name("aime24_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
