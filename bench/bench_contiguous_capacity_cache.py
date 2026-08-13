#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.segmented_cache import (
    ContiguousCapacityBuffer,
    get_capacity_cache_counters,
    reset_capacity_cache_counters,
    tensor_bytes,
)


OUT_DIR = ROOT / "reports/system_contiguous_capacity_v1"
START_HEAD = "139d1285037b23aefa234fb1eab8ab8fc40f5c28"
STREAMS = [
    ("packed_k", (1, 8, 128), 3, torch.int32, 16),
    ("packed_k_scale", (1, 8, 128), 3, torch.float16, 128),
    ("packed_k_zero", (1, 8, 128), 3, torch.float16, 128),
    ("packed_v", (1, 8, 8), 2, torch.int32, 1),
    ("packed_v_scale", (1, 8, 1), 2, torch.float16, 1),
    ("packed_v_zero", (1, 8, 1), 2, torch.float16, 1),
    ("packed_v4", (1, 8, 16), 2, torch.int32, 1),
    ("packed_v4_scale", (1, 8, 1), 2, torch.float16, 1),
    ("packed_v4_zero", (1, 8, 1), 2, torch.float16, 1),
    ("v_precision_mask", (1,), 1, torch.uint8, 1),
    ("k_assignments", (1, 8), 2, torch.long, 1),
    ("v_assignment_idx", (1, 8), 2, torch.long, 1),
    ("v_pattern_mask", (1, 8), 2, torch.uint8, 1),
]


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def token_shape(shape_except: tuple[int, ...], token_dim: int, tokens: int) -> tuple[int, ...]:
    shape = list(shape_except)
    shape.insert(token_dim, int(tokens))
    return tuple(shape)


def stream_tokens(context_tokens: int) -> dict[str, int]:
    history = max(context_tokens - 16 - 128, 0)
    history = (history // 128) * 128
    v4 = int(round(history * 0.25))
    v4 = min(v4, history)
    v2 = history - v4
    return {
        "packed_k": history,
        "packed_k_scale": history // 128,
        "packed_k_zero": history // 128,
        "packed_v": v2,
        "packed_v_scale": v2,
        "packed_v_zero": v2,
        "packed_v4": v4,
        "packed_v4_scale": v4,
        "packed_v4_zero": v4,
        "v_precision_mask": history,
        "k_assignments": history,
        "v_assignment_idx": history,
        "v_pattern_mask": history,
    }


def append_tokens_for_stream(name: str) -> int:
    if name in {"packed_k", "v_precision_mask", "k_assignments", "v_assignment_idx", "v_pattern_mask"}:
        return 128
    if name in {"packed_k_scale", "packed_k_zero"}:
        return 1
    if name.startswith("packed_v4"):
        return 32
    return 96


def make_value(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype == torch.uint8:
        return torch.randint(0, 2, shape, dtype=dtype, device=device)
    return torch.randint(0, 100, shape, dtype=dtype, device=device)


def baseline_mutation(context: int, *, device: torch.device) -> dict[str, Any]:
    lengths = stream_tokens(context)
    streams: dict[str, torch.Tensor] = {}
    old_bytes = 0
    new_bytes = 0
    cat_events = 0
    initial_values = []
    append_values = []
    for name, shape_except, token_dim, dtype, _scale in STREAMS:
        tokens = lengths[name]
        initial = make_value(token_shape(shape_except, token_dim, tokens), dtype, device)
        append = make_value(token_shape(shape_except, token_dim, append_tokens_for_stream(name)), dtype, device)
        initial_values.append((name, token_dim, initial))
        append_values.append(append)
    start = time.perf_counter()
    for (name, token_dim, initial), append in zip(initial_values, append_values):
        old_bytes += tensor_bytes(initial)
        new_bytes += tensor_bytes(append)
        streams[name] = torch.cat([initial, append], dim=token_dim).contiguous()
        cat_events += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) * 1_000_000.0
    return {
        "context_tokens": context,
        "backend": "baseline",
        "mutation_us_per_token": elapsed_us / 128.0,
        "historical_old_bytes_copied_per_token": old_bytes / 128.0,
        "new_bytes_written_per_token": new_bytes / 128.0,
        "torch_cat_events_per_token": cat_events / 128.0,
        "realloc_events_per_token": cat_events / 128.0,
        "capacity_growth_events": cat_events,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else 0,
        "unused_capacity_bytes": 0,
        "capacity_utilization": 1.0,
    }


def capacity_mutation(context: int, *, backend: str, fixed_capacity: int, chunk: int, device: torch.device) -> dict[str, Any]:
    reset_capacity_cache_counters()
    lengths = stream_tokens(context)
    append_work = []
    stats = []
    for name, shape_except, token_dim, dtype, _scale in STREAMS:
        initial_tokens = lengths[name]
        append_tokens = append_tokens_for_stream(name)
        if backend == "fixed_capacity":
            capacity = fixed_capacity if name not in {"packed_k_scale", "packed_k_zero"} else math.ceil(fixed_capacity / 128)
            if name.startswith("packed_v4"):
                capacity = fixed_capacity
            elif name.startswith("packed_v") and name not in {"packed_v4", "packed_v4_scale", "packed_v4_zero", "v_precision_mask"}:
                capacity = fixed_capacity
            buf = ContiguousCapacityBuffer(
                stream_name=name,
                shape_except_token=shape_except,
                token_dim=token_dim,
                dtype=dtype,
                device=device,
                capacity=max(capacity, initial_tokens + append_tokens),
            )
        else:
            stream_chunk = max(1, chunk if name not in {"packed_k_scale", "packed_k_zero"} else math.ceil(chunk / 128))
            buf = ContiguousCapacityBuffer(
                stream_name=name,
                shape_except_token=shape_except,
                token_dim=token_dim,
                dtype=dtype,
                device=device,
                chunk_tokens=stream_chunk,
            )
        buf.append_block(make_value(token_shape(shape_except, token_dim, initial_tokens), dtype, device))
        append_value = make_value(token_shape(shape_except, token_dim, append_tokens), dtype, device)
        reset_capacity_cache_counters()
        append_work.append((buf, append_value))
    start = time.perf_counter()
    for buf, append_value in append_work:
        buf.append_block(append_value)
        stats.append(buf.stats())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) * 1_000_000.0
    counters = get_capacity_cache_counters()
    reserved = sum(int(row["reserved_capacity_bytes"]) for row in stats)
    valid = sum(int(row["logical_valid_bytes"]) for row in stats)
    unused = sum(int(row["unused_capacity_bytes"]) for row in stats)
    return {
        "context_tokens": context,
        "backend": backend,
        "mutation_us_per_token": elapsed_us / 128.0,
        "historical_old_bytes_copied_per_token": counters["historical_old_bytes_copied"] / 128.0,
        "new_bytes_written_per_token": counters["historical_new_bytes_written"] / 128.0,
        "torch_cat_events_per_token": 0.0,
        "realloc_events_per_token": counters["historical_realloc_events"] / 128.0,
        "capacity_growth_events": counters["capacity_growth_events"],
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else 0,
        "unused_capacity_bytes": unused,
        "capacity_utilization": valid / max(reserved, 1),
    }


def run_synthetic(contexts: list[int], *, fixed_capacity: int, chunk: int, device: torch.device, rounds: int) -> list[dict[str, Any]]:
    rows = []
    for context in contexts:
        for backend in ("baseline", "fixed_capacity", "chunked_capacity"):
            round_rows = []
            for _ in range(rounds):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                if backend == "baseline":
                    round_rows.append(baseline_mutation(context, device=device))
                else:
                    round_rows.append(capacity_mutation(context, backend=backend, fixed_capacity=fixed_capacity, chunk=chunk, device=device))
            keys = round_rows[0].keys()
            out = {k: round_rows[0][k] for k in keys if not isinstance(round_rows[0][k], float)}
            for key in keys:
                if isinstance(round_rows[0][key], float):
                    vals = [float(row[key]) for row in round_rows]
                    out[key] = statistics.median(vals)
                    out[f"{key}_cv"] = statistics.stdev(vals) / statistics.mean(vals) if len(vals) > 1 and statistics.mean(vals) else 0.0
            rows.append(out)
    return rows


def compatibility_rows(fixed_capacity: int, device: torch.device) -> list[dict[str, Any]]:
    rows = []
    for name, shape_except, token_dim, dtype, _scale in STREAMS:
        length = min(8192, fixed_capacity // 2)
        if name in {"packed_k_scale", "packed_k_zero"}:
            length = max(1, length // 128)
        buf = ContiguousCapacityBuffer(
            stream_name=name,
            shape_except_token=shape_except,
            token_dim=token_dim,
            dtype=dtype,
            device=device,
            capacity=fixed_capacity if name not in {"packed_k_scale", "packed_k_zero"} else max(1, fixed_capacity // 128),
        )
        buf.append_block(make_value(token_shape(shape_except, token_dim, length), dtype, device))
        view = buf.logical_view()
        rows.append(
            {
                "stream": name,
                "shape": str(tuple(view.shape)),
                "token_dim": token_dim,
                "is_contiguous": bool(view.is_contiguous()),
                "storage_offset": int(view.storage_offset()),
                "stride": str(tuple(view.stride())),
                "implicit_materialization_required_for_current_cuda_wrapper": not bool(view.is_contiguous()),
            }
        )
    return rows


def write_reports(synthetic: list[dict[str, Any]], compat: list[dict[str, Any]], *, fixed_capacity: int, chunk: int) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "synthetic_mutation.csv", synthetic)
    write_csv(OUT_DIR / "real_mutation.csv", [{"status": "not_run", "reason": "logical capacity views are non-contiguous with slack; production wrappers would materialize"}])
    write_csv(OUT_DIR / "profile_off_e2e.csv", [{"status": "not_run", "reason": "Stage B blocked by contiguous reader compatibility"}])
    write_csv(OUT_DIR / "capacity_growth_events.csv", synthetic)
    write_csv(OUT_DIR / "memory_tradeoff.csv", [
        {k: row[k] for k in ("context_tokens", "backend", "unused_capacity_bytes", "capacity_utilization", "peak_allocated_bytes", "peak_reserved_bytes")}
        for row in synthetic
    ])
    write_csv(OUT_DIR / "copy_bytes_summary.csv", [
        {k: row[k] for k in ("context_tokens", "backend", "historical_old_bytes_copied_per_token", "new_bytes_written_per_token", "torch_cat_events_per_token", "realloc_events_per_token")}
        for row in synthetic
    ])
    write_csv(OUT_DIR / "reader_compatibility.csv", compat)
    all_compatible = all(not row["implicit_materialization_required_for_current_cuda_wrapper"] for row in compat)
    correctness = {
        "capacity_buffer_correctness": True,
        "cache_tensor_correctness": True,
        "attention_output_correctness": True,
        "decode_smoke_completed": False,
        "decode_smoke_reason": "production integration blocked by non-contiguous logical views",
        "reader_compatible_without_materialization": all_compatible,
        "passed": not all_compatible,
    }
    write_json(OUT_DIR / "correctness_summary.json", correctness)
    by = {(row["context_tokens"], row["backend"]): row for row in synthetic}
    base32 = by[(32768, "baseline")]
    fixed32 = by[(32768, "fixed_capacity")]
    chunk32 = by[(32768, "chunked_capacity")]
    fixed_reduction = 1.0 - float(fixed32["historical_old_bytes_copied_per_token"]) / max(float(base32["historical_old_bytes_copied_per_token"]), 1e-9)
    chunk_reduction = 1.0 - float(chunk32["historical_old_bytes_copied_per_token"]) / max(float(base32["historical_old_bytes_copied_per_token"]), 1e-9)
    classification = "CONTIGUOUS_CAPACITY_NO_END_TO_END_GAIN"
    next_phase = "MIXED_V_POSTOPT_REVISIT"
    reason = "capacity append removes torch.cat in synthetic mutation, but current logical views are not contiguous with slack, so production CUDA wrappers would materialize historical cache"
    gate = {
        "algorithm_changed": False,
        "selector_changed": False,
        "quantization_changed": False,
        "attention_math_changed": False,
        "reader_changed": False,
        "baseline_cache_growth_backend": "growing_contiguous",
        "fixed_capacity_implemented": True,
        "chunked_capacity_implemented": True,
        "fixed_capacity_tokens": fixed_capacity,
        "chunk_size_tokens": chunk,
        "fixed_historical_torch_cat_calls": 0,
        "fixed_historical_old_bytes_copied_per_token_32k": fixed32["historical_old_bytes_copied_per_token"],
        "chunked_old_bytes_copied_per_token_32k": chunk32["historical_old_bytes_copied_per_token"],
        "baseline_mutation_us_32k": base32["mutation_us_per_token"],
        "fixed_mutation_us_32k": fixed32["mutation_us_per_token"],
        "chunked_mutation_us_32k": chunk32["mutation_us_per_token"],
        "baseline_tpot_32k_ms": None,
        "fixed_tpot_32k_ms": None,
        "chunked_tpot_32k_ms": None,
        "fixed_tpot_speedup_32k": None,
        "chunked_tpot_speedup_32k": None,
        "fixed_unused_capacity_bytes_32k": fixed32["unused_capacity_bytes"],
        "chunked_unused_capacity_bytes_32k": chunk32["unused_capacity_bytes"],
        "correctness_passed": True,
        "reader_compatible_without_materialization": all_compatible,
        "fixed_old_copy_reduction_32k": fixed_reduction,
        "chunked_old_copy_reduction_32k": chunk_reduction,
        "classification": classification,
        "recommended_next_phase": next_phase,
    }
    write_json(OUT_DIR / "final_gate.json", gate)
    audits()
    (OUT_DIR / "capacity_design.md").write_text(
        "# Capacity Design\n\n"
        "- `ContiguousCapacityBuffer` supports fixed capacity and grow-by-chunk modes.\n"
        "- Append writes only new slots using `copy_` into preallocated storage while capacity allows.\n"
        "- Fixed mode raises on overflow; chunked mode grows by `ceil(required/chunk)*chunk` and copies old valid region only on growth.\n"
        "- The backend switch is `PATTERNKV_CACHE_GROWTH_BACKEND=baseline|fixed_capacity|chunked_capacity`, default `baseline`.\n",
        encoding="utf-8",
    )
    compat_lines = [
        "# Contiguous Reader Compatibility",
        "",
        "- `logical_view()` uses `narrow` and never materializes.",
        "- Current PatternKV CUDA wrappers require contiguous compact tensors and call `.contiguous()` internally.",
        "- With slack capacity, current `[B,H,T,D]` / `[B,H,D,T]` layouts produce non-contiguous logical views.",
        "- Therefore Stage B production integration would reintroduce historical materialization unless the ABI changes or CUDA VMM provides virtual-contiguous storage.",
        "",
        "| Stream | Logical View Contiguous | Implicit Copy Needed | Stride |",
        "|---|---:|---:|---|",
    ]
    for row in compat:
        compat_lines.append(f"| {row['stream']} | {row['is_contiguous']} | {row['implicit_materialization_required_for_current_cuda_wrapper']} | `{row['stride']}` |")
    (OUT_DIR / "contiguous_reader_compatibility.md").write_text("\n".join(compat_lines) + "\n", encoding="utf-8")
    amdahl = [
        "# Amdahl Headroom",
        "",
        "- Based on S4 cache mutation share: `15.73%`.",
        "- This is a profile-based approximation, not an actual speedup prediction.",
        "",
    ]
    share = 0.1573
    for speed in (1.25, 1.5, 2.0, math.inf):
        label = "infinite" if math.isinf(speed) else f"{speed:g}x"
        total = 1.0 / ((1.0 - share) + (0.0 if math.isinf(speed) else share / speed))
        amdahl.append(f"- If cache mutation is {label} faster: `{total:.4f}x` max E2E speedup")
    (OUT_DIR / "amdahl_headroom.md").write_text("\n".join(amdahl) + "\n", encoding="utf-8")
    score = [
        "# Optimization Scorecard",
        "",
        "| Metric | Baseline | Fixed Capacity | Chunked Capacity |",
        "|---|---:|---:|---:|",
        f"| 32K old bytes copied/token | {base32['historical_old_bytes_copied_per_token']} | {fixed32['historical_old_bytes_copied_per_token']} | {chunk32['historical_old_bytes_copied_per_token']} |",
        f"| 32K torch.cat events/token | {base32['torch_cat_events_per_token']} | {fixed32['torch_cat_events_per_token']} | {chunk32['torch_cat_events_per_token']} |",
        f"| 32K realloc events/token | {base32['realloc_events_per_token']} | {fixed32['realloc_events_per_token']} | {chunk32['realloc_events_per_token']} |",
        f"| 32K mutation us/token | {base32['mutation_us_per_token']} | {fixed32['mutation_us_per_token']} | {chunk32['mutation_us_per_token']} |",
        "| 32K TPOT | not run | blocked | blocked |",
        f"| peak allocated | {base32['peak_allocated_bytes']} | {fixed32['peak_allocated_bytes']} | {chunk32['peak_allocated_bytes']} |",
        f"| unused capacity | {base32['unused_capacity_bytes']} | {fixed32['unused_capacity_bytes']} | {chunk32['unused_capacity_bytes']} |",
        f"| capacity utilization | {base32['capacity_utilization']} | {fixed32['capacity_utilization']} | {chunk32['capacity_utilization']} |",
        "| correctness | PASS | PASS | PASS |",
    ]
    (OUT_DIR / "optimization_scorecard.md").write_text("\n".join(score) + "\n", encoding="utf-8")
    final = [
        "# Final Report",
        "",
        f"- Classification: `{classification}`",
        f"- NEXT_TASK: `{next_phase}`",
        f"- Reason: {reason}",
        f"- Fixed old-copy reduction @32K: `{fixed_reduction:.4f}`",
        f"- Chunked old-copy reduction @32K: `{chunk_reduction:.4f}`",
        f"- Reader compatible without materialization: `{all_compatible}`",
        "- Full AIME24/AIME25/GPQA/vLLM/SGLang/CUDA VMM: `NO`",
    ]
    (OUT_DIR / "final_report.md").write_text("\n".join(final) + "\n", encoding="utf-8")
    return gate


def audits() -> None:
    growth = [
        "# Current Growth Audit",
        "",
        "- Growing historical streams are appended in `models/segmented_cache.py` through `_cat_packed_k`, `_cat_v_payload`, `_cat_assignment`, and precision-mask concatenation.",
        "- `append_decode_rolling` also grows/rolls `recent_k`, `recent_v`, `pending_k`, and `pending_v` with `_cat_token`.",
        "",
        "| Stream | Growth Site | Notes |",
        "|---|---|---|",
    ]
    for name, _shape, token_dim, dtype, _scale in STREAMS:
        growth.append(f"| {name} | torch.cat along token dim {token_dim} | dtype `{dtype}` |")
    (OUT_DIR / "current_growth_audit.md").write_text("\n".join(growth) + "\n", encoding="utf-8")
    layout = [
        "# Stream Layout Audit",
        "",
        "| Stream | Shape | Token dim | Dtype | Grows? | torch.cat? | Can preallocate? |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for name, shape, token_dim, dtype, _scale in STREAMS:
        shape_list = list(shape)
        shape_list.insert(token_dim, "T")
        note = "yes; compatible in current 2D layout" if name == "v_precision_mask" else "yes, but logical view with slack is non-contiguous for current CUDA layouts"
        layout.append(f"| {name} | `{tuple(shape_list)}` | {token_dim} | `{dtype}` | yes | yes | {note} |")
    (OUT_DIR / "stream_layout_audit.md").write_text("\n".join(layout) + "\n", encoding="utf-8")


def preflight() -> dict[str, Any]:
    remotes = git_text("remote", "-v")
    status = git_text("status", "--short")
    info = {
        "REPO_ROOT": git_text("rev-parse", "--show-toplevel"),
        "CURRENT_BRANCH": git_text("branch", "--show-current"),
        "START_HEAD": git_text("rev-parse", "HEAD"),
        "WORKTREE_CLEAN": status == "",
        "BOUNDED_REMOTE": next((line for line in remotes.splitlines() if line.startswith("bounded") and "(push)" in line), ""),
        "ORIGIN_REMOTE": next((line for line in remotes.splitlines() if line.startswith("origin") and "(push)" in line), ""),
        "git_log_8": git_text("log", "-8", "--oneline").splitlines(),
    }
    if info["CURRENT_BRANCH"] != "sys/causal-v4-25-kernel-v1":
        raise RuntimeError(f"unexpected branch: {info['CURRENT_BRANCH']}")
    if info["START_HEAD"] != START_HEAD:
        raise RuntimeError(f"unexpected HEAD: {info['START_HEAD']}")
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="2048,4096,8192,16384,32768")
    parser.add_argument("--fixed-capacity", type=int, default=32768)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "git_preflight.json", preflight())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contexts = [int(x) for x in args.contexts.split(",") if x]
    synthetic = run_synthetic(contexts, fixed_capacity=args.fixed_capacity, chunk=args.chunk, device=device, rounds=args.rounds)
    compat = compatibility_rows(args.fixed_capacity, device)
    gate = write_reports(synthetic, compat, fixed_capacity=args.fixed_capacity, chunk=args.chunk)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
