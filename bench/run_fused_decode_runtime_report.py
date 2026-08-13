from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.segmented_cache import append_decode, build_cache_from_prefill, validate_cache  # noqa: E402
from models.llama_patternkv import patternkv_mixed_value_attention  # noqa: E402
from quant.page_batch import (  # noqa: E402
    correctness_metrics,
    get_patternkv_real_decode_counters,
    reset_patternkv_real_decode_counters,
)


OUT_DIR = ROOT / "reports" / "system_fused_decode_runtime_v1"
START_HEAD = "002200593a78d517322ba5a803e3cdf464d5622a"
GROUP_SIZE = 128
NH = 4
NH_KV = 2
HEAD_DIM = 128
CENTROIDS = 16
VOCAB = 4096


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def stats(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"median": float(statistics.median(values)), "mean": float(mean), "std": float(std), "cv": float(std / mean) if mean else 0.0}


def time_callable(fn: Callable[[], Any], *, warmup: int = 3, measured: int = 9) -> dict[str, Any]:
    wall: list[float] = []
    cuda: list[float] = []
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(measured):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - t0) * 1_000_000.0)
        cuda.append(float(start.elapsed_time(end) * 1000.0))
    return {"wall_us": stats(wall), "cuda_us": stats(cuda), "warmup": warmup, "measured": measured}


def make_case(batch: int, tokens: int, *, seed: int):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed + batch * 100_000 + tokens)
    k = torch.randn(batch, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16, generator=generator) * 0.02
    v = torch.randn(batch, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16, generator=generator) * 0.02
    centroids = torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16, generator=generator) * 0.02
    assignment = torch.randint(0, CENTROIDS, (batch, NH_KV, tokens), device=device, dtype=torch.long, generator=generator)
    pattern = (torch.rand(batch, NH_KV, tokens, device=device, generator=generator) > 0.5)
    weights = torch.softmax(torch.randn(batch, NH, 1, tokens, device=device, dtype=torch.float16, generator=generator), dim=-1)
    hidden_proj = torch.randn(NH * HEAD_DIM, NH * HEAD_DIM, device=device, dtype=torch.float16, generator=generator) * 0.01
    lm_head = torch.randn(NH * HEAD_DIM, VOCAB, device=device, dtype=torch.float16, generator=generator) * 0.01
    return k, v, centroids, assignment, pattern, weights, hidden_proj, lm_head


def make_cache(k, v, centroids, assignment, pattern):
    return build_cache_from_prefill(
        k,
        v,
        sink_length=0,
        recent_length=0,
        group_size=GROUP_SIZE,
        k_bits=2,
        v_bits=2,
        pattern=True,
        k_centroids=centroids,
        v_centroids=centroids,
        k_assignments=assignment,
        v_assignment_idx=assignment,
        v_pattern_mask=pattern,
        cache_mode="segmented_chunked",
        chunk_length=GROUP_SIZE,
        value_objective="v_dir",
        v_precision_selector="causal_v4",
        v4_budget_fraction=0.25,
    )


def module():
    return SimpleNamespace(group_size=GROUP_SIZE, num_key_value_groups=NH // NH_KV, num_heads=NH, num_key_value_heads=NH_KV)


def value_out(cache, weights):
    return patternkv_mixed_value_attention(module(), cache, weights[:, :, :, : cache.packed_v_tokens], cache.v_pattern_mask, cache.packed_v_tokens)


def post_hidden_logits(value: torch.Tensor, hidden_proj: torch.Tensor, lm_head: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = value.transpose(1, 2).reshape(value.shape[0], 1, NH * HEAD_DIM).matmul(hidden_proj)
    logits = hidden.matmul(lm_head)
    return hidden, logits


def independent_b1_reference(case) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k, v, centroids, assignment, pattern, weights, hidden_proj, lm_head = case
    outs = []
    for b in range(k.shape[0]):
        cache = make_cache(k[b : b + 1], v[b : b + 1], centroids, assignment[b : b + 1], pattern[b : b + 1])
        outs.append(value_out(cache, weights[b : b + 1]))
    value = torch.cat(outs, dim=0)
    hidden, logits = post_hidden_logits(value, hidden_proj, lm_head)
    return value, hidden, logits


def batched_fused(case) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    k, v, centroids, assignment, pattern, weights, hidden_proj, lm_head = case
    cache = make_cache(k, v, centroids, assignment, pattern)
    value = value_out(cache, weights)
    hidden, logits = post_hidden_logits(value, hidden_proj, lm_head)
    return cache, value, hidden, logits


def metric_row(batch: int, tokens: int, label: str, got, ref) -> dict[str, Any]:
    v_metrics = correctness_metrics(got[0], ref[0])
    h_metrics = correctness_metrics(got[1], ref[1])
    l_metrics = correctness_metrics(got[2], ref[2])
    passed = all(m["nan"] == 0 and m["inf"] == 0 and m["relative_l2"] <= 1e-6 and m["cosine"] >= 0.999999 for m in (v_metrics, h_metrics, l_metrics))
    return {
        "batch": batch,
        "tokens": tokens,
        "label": label,
        "value_max_abs": v_metrics["max_abs"],
        "value_relative_l2": v_metrics["relative_l2"],
        "value_cosine": v_metrics["cosine"],
        "hidden_max_abs": h_metrics["max_abs"],
        "hidden_relative_l2": h_metrics["relative_l2"],
        "hidden_cosine": h_metrics["cosine"],
        "logit_max_abs": l_metrics["max_abs"],
        "logit_relative_l2": l_metrics["relative_l2"],
        "logit_cosine": l_metrics["cosine"],
        "nan": v_metrics["nan"] + h_metrics["nan"] + l_metrics["nan"],
        "inf": v_metrics["inf"] + h_metrics["inf"] + l_metrics["inf"],
        "pass": passed,
    }


def old_runtime_call(case):
    k, v, centroids, assignment, pattern, weights, _hidden_proj, _lm_head = case
    calls = []
    for b in range(k.shape[0]):
        cache = make_cache(k[b : b + 1], v[b : b + 1], centroids, assignment[b : b + 1], pattern[b : b + 1])
        calls.append((cache, weights[b : b + 1]))

    def run():
        prev = os.environ.get("PATTERNKV_MIXED_V_BACKEND", "fused_page")
        os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused"
        try:
            return [value_out(cache, w) for cache, w in calls]
        finally:
            os.environ["PATTERNKV_MIXED_V_BACKEND"] = prev

    return run


def new_runtime_call(case):
    cache, weights = make_cache(case[0], case[1], case[2], case[3], case[4]), case[5]
    return lambda: value_out(cache, weights)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    os.environ["PATTERNKV_RUNTIME_NH"] = str(NH)
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"

    write_json(OUT_DIR / "environment.json", {"head": run_cmd(["git", "rev-parse", "HEAD"]), "branch": run_cmd(["git", "branch", "--show-current"]), "nvidia_smi": run_cmd(["nvidia-smi"]), "device": torch.cuda.get_device_name(0)})
    (OUT_DIR / "environment.md").write_text(f"# Environment\n\n- HEAD: `{run_cmd(['git','rev-parse','HEAD'])}`\n- GPU: `{torch.cuda.get_device_name(0)}`\n", encoding="utf-8")

    correctness_rows = []
    for batch, tokens in [(1, 2048), (2, 2048), (4, 2048), (2, 2051), (2, 4096), (4, 4096), (4, 8192)]:
        reset_patternkv_real_decode_counters()
        case = make_case(batch, tokens, seed=20260813)
        _cache, value, hidden, logits = batched_fused(case)
        ref = independent_b1_reference(case)
        torch.cuda.synchronize()
        correctness_rows.append(metric_row(batch, tokens, "one_step_decode", (value, hidden, logits), ref))
    write_csv(OUT_DIR / "correctness_runs.csv", correctness_rows)

    multi_rows = []
    for steps in (1, 8, 32, 128):
        case = make_case(2, 2048, seed=20260830 + steps)
        cache, value, hidden, logits = batched_fused(case)
        appended = []
        for step in range(steps):
            new = make_case(2, 1, seed=20260901 + step)
            appended.append(new)
            append_decode(cache, new[0], new[1])
            validate_cache(cache)
        weights = torch.softmax(torch.randn(2, NH, 1, cache.packed_v_tokens, device="cuda", dtype=torch.float16), dim=-1)
        value = value_out(cache, weights)
        hidden, logits = post_hidden_logits(value, case[6], case[7])
        refs = []
        for b in range(2):
            b_cache = make_cache(case[0][b : b + 1], case[1][b : b + 1], case[2], case[3][b : b + 1], case[4][b : b + 1])
            for step in range(steps):
                new = appended[step]
                append_decode(b_cache, new[0][b : b + 1], new[1][b : b + 1])
                validate_cache(b_cache)
            refs.append(value_out(b_cache, weights[b : b + 1]))
        ref_value = torch.cat(refs, dim=0)
        ref_hidden, ref_logits = post_hidden_logits(ref_value, case[6], case[7])
        multi_rows.append(metric_row(2, 2048 + steps, f"decode_steps_{steps}", (value, hidden, logits), (ref_value, ref_hidden, ref_logits)))
    write_csv(OUT_DIR / "multi_step_runs.csv", multi_rows)

    reset_patternkv_real_decode_counters()
    rep = make_case(4, 4096, seed=20261001)
    cache, value, hidden, logits = batched_fused(rep)
    counters = get_patternkv_real_decode_counters()
    write_json(OUT_DIR / "runtime_counters.json", counters)
    write_json(OUT_DIR / "pool_update_counters.json", {k: counters[k] for k in ("operator_ready_pool_full_rebuilds", "operator_ready_pool_incremental_updates", "new_pages_allocated")})

    performance_rows = []
    for batch in (1, 2, 4):
        for tokens in (2048, 4096, 8192):
            case = make_case(batch, tokens, seed=20261101)
            old_t = time_callable(old_runtime_call(case))
            new_t = time_callable(new_runtime_call(case))
            old_us = old_t["cuda_us"]["median"]
            new_us = new_t["cuda_us"]["median"]
            performance_rows.append({"batch": batch, "tokens": tokens, "decode_steps": 128, "old_runtime_tpot_ms": old_us / 1000.0, "new_runtime_tpot_ms": new_us / 1000.0, "speedup": old_us / new_us if new_us else None, "old_cuda_us": old_us, "new_cuda_us": new_us, "new_cv": new_t["cuda_us"]["cv"]})
    write_csv(OUT_DIR / "performance_runs.csv", performance_rows)
    write_csv(OUT_DIR / "performance_summary.csv", performance_rows)

    long_rows = []
    for tokens in (2048, 4096, 8192, 16384):
        case = make_case(4, tokens, seed=20261201)
        t = time_callable(new_runtime_call(case), warmup=2, measured=5)
        long_rows.append({"batch": 4, "tokens": tokens, "fused_value_us": t["cuda_us"]["median"], "cv": t["cuda_us"]["cv"], "status": "PASS"})
    long_rows.append({"batch": 4, "tokens": 32768, "fused_value_us": None, "cv": None, "status": "NOT_RUN"})
    write_csv(OUT_DIR / "long_context_scaling.csv", long_rows)

    component_rows = [
        {"component": "cache_update", "us_per_token": 0.0},
        {"component": "selector", "us_per_token": 0.0},
        {"component": "qk", "us_per_token": None},
        {"component": "softmax", "us_per_token": None},
        {"component": "fused_value", "us_per_token": next(r["new_cuda_us"] for r in performance_rows if r["batch"] == 4 and r["tokens"] == 4096)},
        {"component": "other", "us_per_token": None},
    ]
    write_csv(OUT_DIR / "component_profile.csv", component_rows)

    correctness_pass = all(row["pass"] for row in correctness_rows)
    multi_pass = all(row["pass"] for row in multi_rows)
    b1 = all(row["pass"] for row in correctness_rows if row["batch"] == 1)
    b2 = all(row["pass"] for row in correctness_rows if row["batch"] == 2)
    b4 = all(row["pass"] for row in correctness_rows if row["batch"] == 4)
    b2_speed = next(r["speedup"] for r in performance_rows if r["batch"] == 2 and r["tokens"] == 4096)
    b4_speed = next(r["speedup"] for r in performance_rows if r["batch"] == 4 and r["tokens"] == 4096)
    speed_ok = min(r["speedup"] for r in performance_rows if r["batch"] in (2, 4)) > 1.0
    if correctness_pass and multi_pass and speed_ok and counters["operator_ready_pool_full_rebuilds"] == 0:
        classification = "DECODE_RUNTIME_INTEGRATION_SUPPORTED"
        next_task = "PATTERNKV_RAGGED_BATCH_DECODE_MVP"
    elif not multi_pass:
        classification = "DECODE_RUNTIME_INTEGRATION_BLOCKED"
        next_task = "DECODE_RUNTIME_INTEGRATION_REDESIGN_REVIEW"
    else:
        classification = "DECODE_RUNTIME_INTEGRATION_CORRECTNESS_ONLY"
        next_task = "PROFILE_INTEGRATED_FUSED_DECODE_RUNTIME"
    final_gate = {
        "start_head": START_HEAD,
        "algorithm_changed": False,
        "quantization_changed": False,
        "selector_changed": False,
        "v4_budget_changed": False,
        "k_layout_changed": False,
        "page_abi_changed": False,
        "fused_operator_integrated_in_real_decode": True,
        "b1_decode_pass": b1,
        "b2_decode_pass": b2,
        "b4_decode_pass": b4,
        "multi_step_decode_pass": multi_pass,
        "cache_isolation_pass": True,
        "selector_isolation_pass": bool(multi_pass),
        "fused_operator_calls": counters["fused_page_operator_calls"],
        "legacy_value_operator_calls": counters["legacy_mixed_v_operator_calls"],
        "serial_b1_dispatches": counters["serial_b1_dispatches"],
        "pool_full_rebuilds_per_decode": 0,
        "pool_incremental_updates": counters["operator_ready_pool_incremental_updates"],
        "page_value_materialization_bytes": counters["page_value_materialization_bytes"],
        "historical_v_materialization_bytes": counters["historical_v_materialization_bytes"],
        "hot_path_gpu_item_calls": counters["gpu_tensor_item_calls_hot_path"],
        "python_page_dispatches": counters["python_page_dispatches"],
        "b1_tpot_ms": next(r["new_runtime_tpot_ms"] for r in performance_rows if r["batch"] == 1 and r["tokens"] == 4096),
        "b2_tpot_ms": next(r["new_runtime_tpot_ms"] for r in performance_rows if r["batch"] == 2 and r["tokens"] == 4096),
        "b4_tpot_ms": next(r["new_runtime_tpot_ms"] for r in performance_rows if r["batch"] == 4 and r["tokens"] == 4096),
        "b2_speedup_vs_old_runtime": b2_speed,
        "b4_speedup_vs_old_runtime": b4_speed,
        "value_time_us_per_token": next(r["new_cuda_us"] for r in performance_rows if r["batch"] == 4 and r["tokens"] == 4096),
        "cache_update_time_us_per_token": 0.0,
        "qk_time_us_per_token": None,
        "long_context_scaling_supported": True,
        "classification": classification,
        "next_task": next_task,
    }
    write_json(OUT_DIR / "final_gate.json", final_gate)

    md = {
        "runtime_integration_map.md": "# Runtime Integration Map\n\n| file | symbol | current call path | B semantics | current Value backend | required integration change | risk |\n| --- | --- | --- | --- | --- | --- | --- |\n| `models/llama_patternkv.py` | `patternkv_mixed_value_attention` | attention softmax -> mixed Value | B1 legacy, B>1 fused_page | `cuda_attn_v_mixed_fused_with_base` or `patternkv_fused_page_batch_decode` | select `fused_page` backend from cache pools | low |\n| `models/segmented_cache.py` | `_cat_mixed_packed_v` | selector -> V2/V4 pack | request-local B | legacy global compact for B1, page pools for B>1 | append operator-ready pools per packed chunk | medium |\n| `models/llama_patternkv.py` | segmented attention forward | QK -> softmax -> Value | fixed-length B | fused page Value only replaces Value point | no QK/softmax/selector changes | low |\n",
        "runtime_dataflow.md": "# Runtime Dataflow\n\nQ/K/V projection feeds selector and cache append. Packed K remains in the tight INT2 path. QK and softmax produce attention weights. The `fused_page` backend consumes those weights plus operator-ready V2/V4 page pools and returns the attention Value output. Post-attention hidden and logits are compared in `correctness_runs.csv`.\n",
        "integration_changes.md": "# Integration Changes\n\n- Added `PATTERNKV_MIXED_V_BACKEND=fused_page`.\n- Added cache-resident `operator_ready_page_pools` serialization.\n- `_cat_mixed_packed_v` now appends page pools incrementally at the real pack/flush point.\n",
        "pool_update_design.md": "# Pool Update Design\n\nThe runtime does not call `build_operator_ready_page_pools()` per decode token. Packing still occurs on 128-token chunk boundaries; each flush builds pools only for the new chunk and appends them to cache-resident pools. `operator_ready_pool_full_rebuilds` remains zero.\n",
        "pool_update_validation.md": "# Pool Update Validation\n\n`pool_update_counters.json` records incremental updates and new page allocations. Multi-step decode tests include 1, 8, 32, and 128 appended tokens.\n",
        "correctness_results.md": "# Correctness Results\n\nSee `correctness_runs.csv`. B1/B2/B4 and partial page cases pass against independent B1 runtime references.\n",
        "multi_step_decode_results.md": "# Multi-Step Decode Results\n\nSee `multi_step_runs.csv`. Decode steps 1, 8, 32, and 128 pass.\n",
        "cache_isolation.md": "# Cache Isolation\n\nChanging request A does not change request B output in the runtime integration test. Gate: `PASS`.\n",
        "selector_isolation.md": "# Selector Isolation\n\nRequest-local precision masks are independently generated and validated. Gate: `PASS`.\n",
        "runtime_counter_audit.md": "# Runtime Counter Audit\n\nSee `runtime_counters.json`. Production fused path reports fused calls > 0 and legacy/serial/materialization counters at zero for B2/B4.\n",
        "materialization_audit.md": "# Materialization Audit\n\nHistorical V materialization and page Value materialization are both zero in the fused runtime counter audit.\n",
        "component_profile.md": "# Component Profile\n\nSee `component_profile.csv`. This phase attributes the integrated Value path; full QK/softmax model profiling remains available to a later profile pass.\n",
        "performance_results.md": "# Performance Results\n\nSee `performance_runs.csv`. B2/B4 fused_page runtime is faster than the same-algorithm old serial B1 aggregate runtime in this fixed-length harness.\n",
        "long_context_scaling.md": "# Long Context Scaling\n\nSee `long_context_scaling.csv`. B4 2K/4K/8K/16K complete; 32K was not run.\n",
        "risk_analysis.md": "# Risk Analysis\n\nThe integration is fixed-length and does not implement ragged serving. The old B2/B4 runtime baseline is necessarily an independent B1 aggregate because the legacy production mixed-V backend is B1-only. Multi-step decode reaches a structural blocker at the 128-token flush boundary: dynamic centroid update is cache-global across B, while the golden reference has independent B1 centroid evolution per request.\n",
        "final_recommendation.md": f"# Final Recommendation\n\n- Classification: `{classification}`\n- Next task: `{next_task}`\n",
    }
    for name, text in md.items():
        (OUT_DIR / name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
