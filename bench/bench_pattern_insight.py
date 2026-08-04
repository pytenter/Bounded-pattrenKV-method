#!/usr/bin/env python
"""PatternKV Insight runner entrypoint.

The runner is deliberately conservative. It validates that the requested run is
for `patternkv_paper`, loads the fixed V0 selected sample list, and prepares
resume-safe per-sample output records. GPU generation hooks can be added on top
of this entrypoint without changing sample selection or baseline semantics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.config import load_standard_baselines
from insight.io import atomic_write_json


def utc_now() -> str:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    """Return current git commit."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def load_selected(path: Path, dataset: str, tasks: list[str]) -> list[dict[str, Any]]:
    """Load fixed selected samples from V0 output."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [x for x in payload.get("selected", []) if x.get("dataset") == dataset]
    if tasks:
        rows = [x for x in rows if x.get("task") in set(tasks)]
    return rows


def result_path(output_dir: Path, sample: dict[str, Any], level: str, seed: int) -> Path:
    """Return deterministic per-sample runner output path."""
    sample_key = sample.get("sample_id") or f"p{int(sample.get('problem_id')):04d}"
    task = str(sample.get("task"))
    return output_dir / str(sample.get("dataset")) / task / f"{sample_key}_{level}_seed{seed}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["longbench", "gsm8k"], required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--selected-samples-json", type=Path, default=Path("reports/insight_v1/v0/selected_samples.json"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/insight_v1"))
    parser.add_argument("--insight-output-dir", type=Path, default=Path("reports/insight_v1"))
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--insight-level", choices=["basic", "oracle", "attention"], default="basic")
    parser.add_argument("--oracle-samples-per-head", type=int, default=8)
    parser.add_argument("--layers", nargs="*", type=int, default=[])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--method", default="patternkv_paper")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method != "patternkv_paper":
        raise SystemExit("bench_pattern_insight only runs patternkv_paper in this phase")
    baselines = load_standard_baselines()
    pattern_cfg = baselines.methods["patternkv_paper"]
    assert pattern_cfg["backend_method"] == "patternkv"
    assert pattern_cfg["k_bits"] == 2 and pattern_cfg["v_bits"] == 2
    assert pattern_cfg["group_size"] == 128 and pattern_cfg["residual_length"] == 128
    assert pattern_cfg["initial_pattern_count"] == 32 and pattern_cfg["pattern_group"] == 128

    samples = load_selected(args.selected_samples_json, args.dataset, args.tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_dir = Path("run/insight_v1") / args.dataset
    status_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    for sample in samples:
        out = result_path(args.output_dir, sample, args.insight_level, args.seed)
        if args.skip_existing and out.exists():
            skipped += 1
            continue
        record = {
            "schema_version": "insight_v1.runner_stub",
            "git_commit": git_commit(),
            "config_hash": baselines.config_hash,
            "dataset": args.dataset,
            "task": sample.get("task"),
            "sample_id": sample.get("sample_id"),
            "problem_id": sample.get("problem_id"),
            "method": "patternkv_paper",
            "insight_level": args.insight_level,
            "seed": args.seed,
            "created_at": utc_now(),
            "gpu_id": args.gpu_id,
            "model_path": str(args.model_path),
            "oracle_samples_per_head": args.oracle_samples_per_head,
            "layers": args.layers,
            "dry_run": args.dry_run,
            "selection_reason": sample.get("selection_reason"),
            "status": "dry_run_prepared" if args.dry_run else "pending_generation_not_implemented",
        }
        atomic_write_json(out, record)
        completed += 1
    atomic_write_json(
        status_dir / f"{args.insight_level}_seed{args.seed}_status.json",
        {
            "schema_version": "insight_v1.status",
            "git_commit": git_commit(),
            "config_hash": baselines.config_hash,
            "dataset": args.dataset,
            "tasks": args.tasks,
            "method": "patternkv_paper",
            "insight_level": args.insight_level,
            "seed": args.seed,
            "created_at": utc_now(),
            "selected": len(samples),
            "written": completed,
            "skipped": skipped,
            "dry_run": args.dry_run,
        },
    )
    print(json.dumps({"selected": len(samples), "written": completed, "skipped": skipped, "dry_run": args.dry_run}, sort_keys=True))


if __name__ == "__main__":
    main()
