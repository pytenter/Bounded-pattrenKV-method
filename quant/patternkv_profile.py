from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import torch


_COUNTERS: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0.0, "tokens": 0.0, "bytes": 0.0})
_CUDA_EVENTS: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
_CPU_US: dict[str, float] = defaultdict(float)
_TEMP_ALLOCS: dict[str, dict[str, Any]] = {}
_CACHE_MUTATIONS: dict[str, dict[str, float]] = defaultdict(
    lambda: {
        "calls": 0.0,
        "append_bytes": 0.0,
        "old_bytes": 0.0,
        "estimated_copy_bytes": 0.0,
        "largest_result_bytes": 0.0,
    }
)


def profile_enabled() -> bool:
    system_flag = os.environ.get("PATTERNKV_SYSTEM_PROFILE")
    if system_flag is not None:
        return system_flag.strip().lower() in {"1", "true", "yes", "on"}
    return os.environ.get("PATTERNKV_PROFILE", "0").strip().lower() in {"1", "true", "yes", "on"}


def reset_profile() -> None:
    _COUNTERS.clear()
    _CUDA_EVENTS.clear()
    _CPU_US.clear()
    _TEMP_ALLOCS.clear()
    _CACHE_MUTATIONS.clear()


def record_counter(component: str, *, calls: int = 1, tokens: int = 0, bytes_copied: int = 0) -> None:
    if not profile_enabled():
        return
    rec = _COUNTERS[component]
    rec["calls"] += float(calls)
    rec["tokens"] += float(tokens)
    if component.endswith("_largest_bytes"):
        rec["bytes"] = max(float(rec["bytes"]), float(bytes_copied))
    else:
        rec["bytes"] += float(bytes_copied)


def tensor_bytes(value: torch.Tensor | None) -> int:
    return 0 if value is None else int(value.numel() * value.element_size())


def record_temp_allocation(name: str, value: torch.Tensor | None) -> None:
    if not profile_enabled() or value is None:
        return
    bytes_value = tensor_bytes(value)
    rec = _TEMP_ALLOCS.setdefault(
        name,
        {
            "tensor": name,
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "bytes": 0.0,
            "calls": 0.0,
        },
    )
    rec["calls"] += 1.0
    rec["bytes"] += float(bytes_value)


def record_cache_mutation(category: str, old_value: torch.Tensor | None, append_value: torch.Tensor | None, result_value: torch.Tensor | None) -> None:
    if not profile_enabled():
        return
    old_bytes = tensor_bytes(old_value)
    append_bytes = tensor_bytes(append_value)
    result_bytes = tensor_bytes(result_value)
    rec = _CACHE_MUTATIONS[category]
    rec["calls"] += 1.0
    rec["old_bytes"] += float(old_bytes)
    rec["append_bytes"] += float(append_bytes)
    rec["estimated_copy_bytes"] += float(old_bytes + append_bytes)
    rec["largest_result_bytes"] = max(float(rec["largest_result_bytes"]), float(result_bytes))


@contextmanager
def profile_range(component: str, *, tokens: int = 0, bytes_copied: int = 0):
    if not profile_enabled():
        yield
        return
    record_counter(component, tokens=tokens, bytes_copied=bytes_copied)
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            _CUDA_EVENTS[component].append((start, end))
    else:
        start_cpu = time.perf_counter()
        try:
            yield
        finally:
            _CPU_US[component] += (time.perf_counter() - start_cpu) * 1_000_000.0


def profile_snapshot(*, reset: bool = False) -> dict[str, dict[str, float]]:
    if torch.cuda.is_available() and _CUDA_EVENTS:
        torch.cuda.synchronize()
    out: dict[str, dict[str, float]] = {}
    components = set(_COUNTERS) | set(_CUDA_EVENTS) | set(_CPU_US)
    for component in sorted(components):
        counter = _COUNTERS.get(component, {})
        cuda_us = 0.0
        for start, end in _CUDA_EVENTS.get(component, []):
            cuda_us += float(start.elapsed_time(end) * 1000.0)
        total_us = cuda_us + float(_CPU_US.get(component, 0.0))
        calls = float(counter.get("calls", 0.0))
        if calls == 0 and (component in _CUDA_EVENTS or component in _CPU_US):
            calls = float(len(_CUDA_EVENTS.get(component, [])) or 1)
        out[component] = {
            "calls": calls,
            "total_us": total_us,
            "mean_us": total_us / calls if calls else 0.0,
            "tokens": float(counter.get("tokens", 0.0)),
            "bytes": float(counter.get("bytes", 0.0)),
        }
    if reset:
        reset_profile()
    return out


def temp_allocation_snapshot(*, decode_tokens: int = 1) -> list[dict[str, Any]]:
    rows = []
    for name, rec in sorted(_TEMP_ALLOCS.items()):
        rows.append(
            {
                "tensor": name,
                "shape": "x".join(str(x) for x in rec["shape"]),
                "dtype": rec["dtype"],
                "bytes": int(rec["bytes"]),
                "calls": int(rec["calls"]),
                "bytes_per_decode_token": float(rec["bytes"]) / max(int(decode_tokens), 1),
            }
        )
    return rows


def cache_mutation_snapshot() -> list[dict[str, Any]]:
    rows = []
    for category, rec in sorted(_CACHE_MUTATIONS.items()):
        calls = int(rec["calls"])
        estimated = float(rec["estimated_copy_bytes"])
        rows.append(
            {
                "category": category,
                "calls": calls,
                "append_bytes": int(rec["append_bytes"]),
                "old_bytes": int(rec["old_bytes"]),
                "estimated_copy_bytes": int(estimated),
                "largest_result_bytes": int(rec["largest_result_bytes"]),
                "mean_copy_bytes": estimated / max(calls, 1),
            }
        )
    return rows


def merge_profile_rows(snapshot: dict[str, dict[str, float]], *, decode_tokens: int, decode_total_us: float) -> list[dict[str, Any]]:
    rows = []
    denom = max(float(decode_total_us), 1e-9)
    for component, rec in sorted(snapshot.items()):
        calls = float(rec.get("calls", 0.0))
        rows.append(
            {
                "component": component,
                "calls": int(calls),
                "calls_per_generated_token": calls / max(int(decode_tokens), 1),
                "total_us": float(rec.get("total_us", 0.0)),
                "mean_us": float(rec.get("mean_us", 0.0)),
                "percent_decode_time": float(rec.get("total_us", 0.0)) * 100.0 / denom,
                "tokens": int(rec.get("tokens", 0.0)),
                "bytes": int(rec.get("bytes", 0.0)),
            }
        )
    return rows
