#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant"))

from bench.aime24_int2_wave1 import task_key3
from bench.aime_utils import effective_seed, load_aime24
from bench.bench_aime24_patternkv import load_model, run_task, validate_context
from quant.matmul import get_patternkv_mixed_v_counters, reset_patternkv_mixed_v_counters
from quant.patternkv_profile import profile_snapshot, reset_profile
from scripts.run_aime24_full_causal25_quality import (
    DATASET_PATH,
    make_worker_args,
    method_generation_hash,
    set_selector_task_context,
)


OUT_DIR = ROOT / "reports/system_profile_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-ids", default="0")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--physical-gpu", default="0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused"
    os.environ["PATTERNKV_PROFILE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rows = load_aime24(DATASET_PATH)
    row_by_id = {int(row["problem_id"]): row for row in rows}
    problem_ids = [int(part.strip()) for part in args.problem_ids.split(",") if part.strip()]
    wargs = make_worker_args("CAUSAL_V4_25", args.base_seed, args.physical_gpu, experiment_id="phase_s1_5_real_model_profile_smoke")
    wargs.max_new_tokens = int(args.max_new_tokens)
    model, tokenizer = load_model(wargs)
    validate_context(wargs, tokenizer, model, rows)
    generation_hash = method_generation_hash("CAUSAL_V4_25")
    out_rows = []
    reset_profile()
    reset_patternkv_mixed_v_counters()
    runtime_error = None
    try:
        for problem_id in problem_ids:
            selector_task_key = task_key3(problem_id, 0, effective_seed(args.base_seed, problem_id, 0))
            set_selector_task_context(model, selector_task_key)
            rec = run_task(wargs, model, tokenizer, row_by_id[problem_id], 0, generation_hash, "phase_s1_5_smoke")
            out_rows.append(
                {
                    "problem_id": problem_id,
                    "generated_tokens": rec.get("generated_tokens"),
                    "stop_reason": rec.get("stop_reason"),
                    "parsed_answer": rec.get("parsed_answer"),
                    "reference_answer": rec.get("reference_answer"),
                    "is_correct": bool(rec.get("is_correct")),
                    "nan_inf_detected": bool(rec.get("nan_inf_detected")),
                }
            )
    except Exception as exc:
        runtime_error = repr(exc)
    profile = profile_snapshot(reset=True)
    counters = get_patternkv_mixed_v_counters()
    passed = (
        runtime_error is None
        and bool(profile)
        and counters["mixed_v_fused_calls"] > 0
        and counters["mixed_v_reference_calls"] == 0
        and all(not row["nan_inf_detected"] and row["stop_reason"] in {"eos", "length"} for row in out_rows)
    )
    payload = {
        "backend": "fused",
        "problem_ids": problem_ids,
        "base_seed": args.base_seed,
        "max_new_tokens": args.max_new_tokens,
        "passed": passed,
        "runtime_error": runtime_error,
        "nan_inf": any(row.get("nan_inf_detected") for row in out_rows),
        "counters": counters,
        "profile_components": sorted(profile.keys()),
        "profile_records_generated": bool(profile),
        "rows": out_rows,
        "full_aime24_rerun_started": False,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "real_model_smoke.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gate_path = OUT_DIR / "final_gate.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate["real_model_smoke_completed"] = bool(passed)
        gate["real_model_smoke_profile_records_generated"] = bool(profile)
        gate["real_model_smoke_mixed_v_fused_calls"] = int(counters["mixed_v_fused_calls"])
        gate["real_model_smoke_mixed_v_reference_calls"] = int(counters["mixed_v_reference_calls"])
        gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
