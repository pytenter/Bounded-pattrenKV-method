#!/usr/bin/env python
"""Control-plane helper for pending samples and status accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from insight_wave_a_4090_utils import (
    PLAN,
    RESULT_ROOT,
    is_completed_generation,
    is_completed_observer,
    load_reference,
    pending_samples,
    result_path,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pending", "plan", "materialize"])
    parser.add_argument("--dataset")
    parser.add_argument("--task")
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference = load_reference()
    if args.command == "plan":
        payload = {"total": 140, "tasks": [{"dataset": d, "task": t, "planned": n} for d, t, n in PLAN]}
    elif args.command == "materialize":
        rows = []
        for task in ("hotpotqa", "passage_retrieval_en", "passage_retrieval_zh", "samsum", "dureader"):
            rows.extend(reference["longbench_samples"][task])
        rows.extend(
            {
                "dataset": "gsm8k",
                "task": "gsm8k",
                "problem_id": int(problem_id),
                "sample_id": "",
                "sample_index": "",
                "selection_reason": "v100_manifest",
            }
            for problem_id in reference["gsm8k_problem_ids"]
        )
        payload = {"schema_version": "insight_v2.wave_a_4090_selected_samples", "selected": rows, "count": len(rows)}
    else:
        rows = [row for row in pending_samples(reference, args.result_root) if row["dataset"] == args.dataset and row["task"] == args.task]
        payload = {"dataset": args.dataset, "task": args.task, "pending": rows, "count": len(rows)}
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
