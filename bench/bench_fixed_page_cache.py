#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.segmented_cache import DEFAULT_PAGE_SIZE, FixedPageCacheStorage, RecentRingBuffer, tensor_bytes  # noqa: E402


OUT_DIR = ROOT / "reports/system_fixed_page_v1"
BSZ = 1
HEADS = 8
HEAD_DIM = 128


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(math.ceil(len(ordered) * p)) - 1)])


def cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    return float(statistics.stdev(values) / mean) if mean else 0.0


def history_tokens(context_tokens: int) -> int:
    return max(int(context_tokens) - 16 - 128, 0)


def empty_stream(tokens: int, tail: tuple[int, ...], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.empty((BSZ, HEADS, tokens, *tail), dtype=dtype, device=device)


def block_stream(tokens: int, tail: tuple[int, ...], *, dtype: torch.dtype, device: torch.device, fill: int) -> torch.Tensor:
    out = torch.empty((BSZ, HEADS, tokens, *tail), dtype=dtype, device=device)
    if dtype.is_floating_point:
        out.fill_(float(fill))
    else:
        out.fill_(int(fill) % 17)
    return out


def precision_block(tokens: int, *, device: torch.device, fill: int) -> torch.Tensor:
    values = (torch.arange(tokens, device=device).view(1, tokens) + fill) % 4 == 0
    return values.to(torch.uint8)


def cat_append(current: torch.Tensor | None, value: torch.Tensor, *, dim: int) -> tuple[torch.Tensor, int, int]:
    old = tensor_bytes(current)
    new = tensor_bytes(value)
    if current is None:
        return value.contiguous(), old, new
    return torch.cat([current, value], dim=dim).contiguous(), old, new


def run_contiguous_round(context: int, decode_tokens: int, *, device: torch.device) -> dict[str, Any]:
    hist = history_tokens(context)
    streams: dict[str, tuple[torch.Tensor | None, int]] = {
        "packed_v": (empty_stream(hist, (16,), dtype=torch.float16, device=device), 2),
        "packed_v_scale": (empty_stream(hist, (1,), dtype=torch.float16, device=device), 2),
        "packed_v_zero": (empty_stream(hist, (1,), dtype=torch.float16, device=device), 2),
        "v_pattern_mask": (torch.empty((BSZ, HEADS, hist), dtype=torch.uint8, device=device), 2),
        "v_assignment_idx": (torch.empty((BSZ, HEADS, hist), dtype=torch.int16, device=device), 2),
        "v_precision_mask": (torch.empty((BSZ, hist), dtype=torch.uint8, device=device), 1),
    }
    recent_k = empty_stream(128, (HEAD_DIM,), dtype=torch.float16, device=device)
    recent_v = empty_stream(128, (HEAD_DIM,), dtype=torch.float16, device=device)
    old_bytes = 0
    new_bytes = 0
    cat_events = 0
    recent_roll_copy_bytes = 0
    historical_append_copy_bytes = 0
    metadata_copy_bytes = 0
    start = time.perf_counter()
    for t in range(decode_tokens):
        k = block_stream(1, (HEAD_DIM,), dtype=torch.float16, device=device, fill=t)
        v = block_stream(1, (HEAD_DIM,), dtype=torch.float16, device=device, fill=t)
        recent_k, old, new = cat_append(recent_k, k, dim=2)
        old_bytes += old
        new_bytes += new
        recent_roll_copy_bytes += old
        cat_events += int(old > 0)
        recent_v, old, new = cat_append(recent_v, v, dim=2)
        old_bytes += old
        new_bytes += new
        recent_roll_copy_bytes += old
        cat_events += int(old > 0)
        overflow = max(int(recent_k.shape[2]) - 128, 0)
        if overflow:
            recent_k = recent_k[:, :, overflow:, :].contiguous()
            recent_v = recent_v[:, :, overflow:, :].contiguous()
        if (t + 1) % 128 == 0:
            blocks = {
                "packed_v": block_stream(128, (16,), dtype=torch.float16, device=device, fill=t),
                "packed_v_scale": block_stream(128, (1,), dtype=torch.float16, device=device, fill=t),
                "packed_v_zero": block_stream(128, (1,), dtype=torch.float16, device=device, fill=t),
                "v_pattern_mask": torch.ones((BSZ, HEADS, 128), dtype=torch.uint8, device=device),
                "v_assignment_idx": torch.full((BSZ, HEADS, 128), t % 16, dtype=torch.int16, device=device),
                "v_precision_mask": precision_block(128, device=device, fill=t),
            }
            for name, block in blocks.items():
                current, dim = streams[name]
                result, old, new = cat_append(current, block, dim=dim)
                streams[name] = (result, dim)
                old_bytes += old
                new_bytes += new
                cat_events += int(old > 0)
                if name in {"packed_v", "packed_v_scale", "packed_v_zero"}:
                    historical_append_copy_bytes += old
                else:
                    metadata_copy_bytes += old
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1_000_000.0
    peak_alloc = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved() if device.type == "cuda" else 0
    return {
        "context_tokens": context,
        "decode_tokens": decode_tokens,
        "backend": "contiguous",
        "mutation_us_per_token": elapsed / decode_tokens,
        "torch_cat_events_per_token": cat_events / decode_tokens,
        "old_bytes_copied_per_token": old_bytes / decode_tokens,
        "new_bytes_written_per_token": new_bytes / decode_tokens,
        "recent_roll_copy_bytes_per_token": recent_roll_copy_bytes / decode_tokens,
        "historical_append_copy_bytes_per_token": historical_append_copy_bytes / decode_tokens,
        "metadata_copy_bytes_per_token": metadata_copy_bytes / decode_tokens,
        "page_allocations": 0,
        "page_crossings": 0,
        "peak_allocated_bytes": int(peak_alloc),
        "peak_reserved_bytes": int(peak_reserved),
        "fragmentation_bytes": 0,
    }


def run_paged_round(context: int, decode_tokens: int, *, device: torch.device) -> dict[str, Any]:
    hist = history_tokens(context)
    storage = FixedPageCacheStorage(page_size=DEFAULT_PAGE_SIZE)
    initial = {
        "packed_v": block_stream(hist, (16,), dtype=torch.float16, device=device, fill=0),
        "packed_v_scale": block_stream(hist, (1,), dtype=torch.float16, device=device, fill=0),
        "packed_v_zero": block_stream(hist, (1,), dtype=torch.float16, device=device, fill=0),
        "v_pattern_mask": torch.ones((BSZ, HEADS, hist), dtype=torch.uint8, device=device),
        "v_assignment_idx": torch.zeros((BSZ, HEADS, hist), dtype=torch.int16, device=device),
        "v_precision_mask": precision_block(hist, device=device, fill=0),
    }
    for name, value in initial.items():
        storage.append_stream(name, value)
    recent_k = RecentRingBuffer(capacity=128, stream="recent_k")
    recent_v = RecentRingBuffer(capacity=128, stream="recent_v")
    recent_k.append_block(block_stream(128, (HEAD_DIM,), dtype=torch.float16, device=device, fill=0))
    recent_v.append_block(block_stream(128, (HEAD_DIM,), dtype=torch.float16, device=device, fill=0))
    torch_cat_events = 0
    new_bytes = 0
    old_bytes = 0
    start = time.perf_counter()
    for t in range(decode_tokens):
        k = block_stream(1, (HEAD_DIM,), dtype=torch.float16, device=device, fill=t)
        v = block_stream(1, (HEAD_DIM,), dtype=torch.float16, device=device, fill=t)
        flushed_k = recent_k.append_block(k)
        flushed_v = recent_v.append_block(v)
        new_bytes += tensor_bytes(k) + tensor_bytes(v)
        old_bytes += tensor_bytes(flushed_k) + tensor_bytes(flushed_v)
        if (t + 1) % 128 == 0:
            blocks = {
                "packed_v": block_stream(128, (16,), dtype=torch.float16, device=device, fill=t),
                "packed_v_scale": block_stream(128, (1,), dtype=torch.float16, device=device, fill=t),
                "packed_v_zero": block_stream(128, (1,), dtype=torch.float16, device=device, fill=t),
                "v_pattern_mask": torch.ones((BSZ, HEADS, 128), dtype=torch.uint8, device=device),
                "v_assignment_idx": torch.full((BSZ, HEADS, 128), t % 16, dtype=torch.int16, device=device),
                "v_precision_mask": precision_block(128, device=device, fill=t),
            }
            for name, block in blocks.items():
                storage.append_stream(name, block)
                new_bytes += tensor_bytes(block)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1_000_000.0
    stats = storage.stats()
    page_allocations = sum(int(row["page_allocations"]) for row in stats)
    page_crossings = sum(int(row["page_crossings"]) for row in stats)
    fragmentation = sum(int(row["fragmentation_bytes"]) for row in stats)
    peak_alloc = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved() if device.type == "cuda" else 0
    return {
        "context_tokens": context,
        "decode_tokens": decode_tokens,
        "backend": "paged",
        "mutation_us_per_token": elapsed / decode_tokens,
        "torch_cat_events_per_token": torch_cat_events / decode_tokens,
        "old_bytes_copied_per_token": old_bytes / decode_tokens,
        "new_bytes_written_per_token": new_bytes / decode_tokens,
        "recent_roll_copy_bytes_per_token": old_bytes / decode_tokens,
        "historical_append_copy_bytes_per_token": 0,
        "metadata_copy_bytes_per_token": 0,
        "page_allocations": page_allocations,
        "page_crossings": page_crossings,
        "peak_allocated_bytes": int(peak_alloc),
        "peak_reserved_bytes": int(peak_reserved),
        "fragmentation_bytes": fragmentation,
    }


def run_rounds(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contexts = [int(x) for x in args.contexts.split(",") if x]
    baseline_rows: list[dict[str, Any]] = []
    paged_rows: list[dict[str, Any]] = []
    for context in contexts:
        for backend, fn, out_rows in (
            ("contiguous", run_contiguous_round, baseline_rows),
            ("paged", run_paged_round, paged_rows),
        ):
            for warm in range(args.warmup):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                fn(context, args.decode_tokens, device=device)
            round_values = []
            for round_idx in range(args.rounds):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                row = fn(context, args.decode_tokens, device=device)
                row.update({"round": round_idx})
                out_rows.append(row)
                round_values.append(float(row["mutation_us_per_token"]))
            summary = {k: out_rows[-1][k] for k in out_rows[-1] if k not in {"round", "mutation_us_per_token"}}
            summary.update(
                {
                    "backend": backend,
                    "round": "summary",
                    "mutation_us_per_token": float(statistics.median(round_values)),
                    "mutation_us_mean": float(statistics.mean(round_values)),
                    "mutation_us_p90": percentile(round_values, 0.90),
                    "round_cv": cv(round_values),
                    "stability": "STABLE" if cv(round_values) <= 0.05 else "UNSTABLE",
                }
            )
            out_rows.append(summary)
    return baseline_rows, paged_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="2048,4096,8192,16384,32768")
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline, paged = run_rounds(args)
    write_csv(OUT_DIR / "baseline_mutation.csv", baseline)
    write_csv(OUT_DIR / "paged_mutation.csv", paged)
    bsum = [r for r in baseline if r["round"] == "summary"]
    psum = [r for r in paged if r["round"] == "summary"]
    copy_rows = []
    alloc_rows = []
    frag_rows = []
    mem_rows = []
    for b, p in zip(bsum, psum):
        copy_rows.append(
            {
                "context_tokens": b["context_tokens"],
                "baseline_old_bytes_copied_per_token": b["old_bytes_copied_per_token"],
                "paged_old_bytes_copied_per_token": p["old_bytes_copied_per_token"],
                "copy_reduction_fraction": 1.0 - float(p["old_bytes_copied_per_token"]) / max(float(b["old_bytes_copied_per_token"]), 1.0),
                "baseline_new_bytes_written_per_token": b["new_bytes_written_per_token"],
                "paged_new_bytes_written_per_token": p["new_bytes_written_per_token"],
            }
        )
        alloc_rows.append(
            {
                "context_tokens": p["context_tokens"],
                "page_allocations_per_128_decode": p["page_allocations"],
                "page_crossings": p["page_crossings"],
            }
        )
        frag_rows.append(
            {
                "context_tokens": p["context_tokens"],
                "page_size": DEFAULT_PAGE_SIZE,
                "fragmentation_bytes": p["fragmentation_bytes"],
                "fragmentation_fraction_estimate": float(p["fragmentation_bytes"]) / max(float(p["peak_allocated_bytes"]), 1.0),
            }
        )
        mem_rows.append(
            {
                "context_tokens": b["context_tokens"],
                "baseline_peak_allocated_bytes": b["peak_allocated_bytes"],
                "paged_peak_allocated_bytes": p["peak_allocated_bytes"],
                "baseline_peak_reserved_bytes": b["peak_reserved_bytes"],
                "paged_peak_reserved_bytes": p["peak_reserved_bytes"],
            }
        )
    write_csv(OUT_DIR / "copy_bytes_summary.csv", copy_rows)
    write_csv(OUT_DIR / "page_allocation_stats.csv", alloc_rows)
    write_csv(OUT_DIR / "fragmentation.csv", frag_rows)
    write_csv(OUT_DIR / "memory_summary.csv", mem_rows)
    write_csv(OUT_DIR / "e2e_summary.csv", [{"status": "NOT_RUN", "reason": "S3-1 retained a storage-ABI milestone; page-native attention reader is not implemented, so E2E would materialize pages and was skipped."}])
    write_json(
        OUT_DIR / "benchmark_environment.json",
        {
            "device": str(torch.cuda.get_device_name(torch.cuda.current_device())) if torch.cuda.is_available() else "cpu",
            "torch_version": torch.__version__,
            "page_size": DEFAULT_PAGE_SIZE,
            "decode_tokens": args.decode_tokens,
            "rounds": args.rounds,
            "warmup": args.warmup,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
