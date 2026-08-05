#!/usr/bin/env python
"""Select fixed 25 samples for 4090 range-aware targeted collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

LONG_TASKS = ("hotpotqa", "passage_retrieval_en", "passage_retrieval_zh", "samsum", "dureader")
LONG_POSITIONS = (0, 5, 11)
GSM_POSITIONS = (0, 9, 18, 26, 35, 44, 53, 61, 70, 79)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_manifest(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text), sha256_text(text)


def _longbench_rows(manifest: dict[str, Any], task: str) -> list[dict[str, Any]]:
    rows = manifest.get("longbench_samples", {})
    if isinstance(rows, dict):
        return list(rows.get(task) or [])
    return [row for row in rows if row.get("task") == task]


def _gsm8k_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    problem_ids = list(manifest.get("gsm8k_problem_ids") or [])
    rows = []
    for pos, problem_id in enumerate(problem_ids):
        rows.append(
            {
                "dataset": "gsm8k",
                "task": "gsm8k",
                "problem_id": int(problem_id),
                "sample_id": f"gsm8k:{int(problem_id)}",
                "sample_index": pos,
                "selection_reason": "reference_manifest_order",
            }
        )
    return rows


def select_samples(manifest: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for task in LONG_TASKS:
        rows = _longbench_rows(manifest, task)
        if len(rows) != 12:
            raise ValueError(f"{task} expected 12 rows, got {len(rows)}")
        for pos in LONG_POSITIONS:
            row = dict(rows[pos])
            row.update(
                {
                    "manifest_position": pos,
                    "selection_rule": f"{task}:fixed_positions_0_5_11",
                    "source_manifest_sha256": source_sha256,
                }
            )
            selected.append(row)
    gsm_rows = _gsm8k_rows(manifest)
    if len(gsm_rows) != 80:
        raise ValueError(f"gsm8k expected 80 rows, got {len(gsm_rows)}")
    for pos in GSM_POSITIONS:
        row = dict(gsm_rows[pos])
        row.update(
            {
                "manifest_position": pos,
                "selection_rule": "gsm8k:round(i*(n-1)/9)",
                "source_manifest_sha256": source_sha256,
            }
        )
        selected.append(row)
    payload = {
        "schema_version": "insight_v2.range_aware_targeted_selection_v1",
        "reference_manifest_sha256": source_sha256,
        "selected": selected,
    }
    selected_text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    payload["selected_manifest_sha256"] = sha256_text(selected_text)
    for row in payload["selected"]:
        row["selected_manifest_sha256"] = payload["selected_manifest_sha256"]
    return payload


def markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Range-aware targeted 25-sample selection",
        "",
        f"selected_manifest_sha256: `{payload['selected_manifest_sha256']}`",
        f"reference_manifest_sha256: `{payload['reference_manifest_sha256']}`",
        "",
        "| task | positions | count |",
        "| --- | --- | ---: |",
        "| hotpotqa | 0, 5, 11 | 3 |",
        "| passage_retrieval_en | 0, 5, 11 | 3 |",
        "| passage_retrieval_zh | 0, 5, 11 | 3 |",
        "| samsum | 0, 5, 11 | 3 |",
        "| dureader | 0, 5, 11 | 3 |",
        "| gsm8k | 0, 9, 18, 26, 35, 44, 53, 61, 70, 79 | 10 |",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest, source_sha256 = read_manifest(args.reference_manifest)
    payload = select_samples(manifest, source_sha256)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "selected_25_samples.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "selected_25_samples.md").write_text(markdown_summary(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
