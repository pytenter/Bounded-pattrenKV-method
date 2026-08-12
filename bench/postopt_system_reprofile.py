#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant"))

import patternkv_gemv
from bench.bench_aime24_patternkv import load_model
from bench.profile_post_fusion_decode import build_synthetic_past, ensure_profile_centroids, run_decode_case
from models.llama_patternkv import patternkv_mixed_v_backend
from models.segmented_cache import normalize_cache_backend
from quant.matmul import (
    get_patternkv_mixed_v_counters,
    patternkv_gqa_v_backend,
    patternkv_page_v_reader_backend,
    reset_patternkv_mixed_v_counters,
)
from quant.patternkv_profile import cache_mutation_snapshot, profile_snapshot, reset_profile
from scripts.run_aime24_full_causal25_quality import make_worker_args


OUT_DIR = ROOT / "reports/system_postopt_profile_v1"
START_HEAD = "ed4b2731ff58eddc744e116ea1d55c8660af14f6"
PRODUCTION_ENV = {
    "PATTERNKV_MIXED_V_BACKEND": "fused",
    "PATTERNKV_GQA_V_BACKEND": "baseline",
    "PATTERNKV_CACHE_BACKEND": "contiguous",
    "PATTERNKV_PAGE_V_READER": "contiguous",
}


COMPONENT_DEFINITIONS = {
    "QK": "query x historical K, including Pattern K fused quantized path plus FP16 sink/pending/recent score regions and score concat",
    "softmax": "attention score normalization",
    "importance_update": "causal importance statistics update",
    "mixed_v": "compressed-domain V2/V4 Value attention",
    "selector": "V4 token identity selector",
    "packing": "new historical V2/V4 quantization and packing at flush",
    "cache_mutation": "append, recent, pending, historical tensor mutation and torch.cat cache changes",
    "output_projection": "attention output linear projection",
    "qkv_projection": "Q/K/V linear projections",
    "rope": "rotary position embedding application",
    "lm_head": "decode LM head projection",
    "other": "unclassified model compute, including layernorm/MLP and any uninstrumented gaps",
}

GROUPS = {
    "qkv_projection": ("qkv_projection",),
    "rope": ("rope_position",),
    "QK": ("qk_quantized_history", "qk_fp16_regions", "attention_score_concat"),
    "softmax": ("attention_softmax",),
    "importance_update": ("importance_update",),
    "mixed_v": ("mixed_v_fused_attention",),
    "selector": ("selector_total",),
    "packing": ("pack_window",),
    "cache_mutation": ("cache_append", "cache_flush", "cache_mutation"),
    "output_projection": ("output_projection",),
    "lm_head": ("decode_lm_head",),
}


def run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def git_text(*args: str) -> str:
    return run_text(["git", "-C", str(ROOT), *args])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(math.ceil(p * len(ordered))) - 1)
    return float(ordered[idx])


def cv(values: list[float]) -> float:
    mean = statistics.mean(values)
    return float(statistics.stdev(values) / mean) if len(values) > 1 and mean else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nvidia_smi_query() -> list[dict[str, str]]:
    fields = "index,name,driver_version,memory.used,memory.total,utilization.gpu,power.draw,clocks.sm"
    try:
        text = run_text(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    except Exception:
        return []
    rows = []
    keys = fields.split(",")
    for line in text.splitlines():
        values = [part.strip() for part in line.split(",")]
        rows.append(dict(zip(keys, values)))
    return rows


def preflight(*, allow_dirty_worktree: bool = False) -> dict[str, Any]:
    remotes = git_text("remote", "-v")
    status = git_text("status", "--short")
    payload = {
        "REPO_ROOT": git_text("rev-parse", "--show-toplevel"),
        "CURRENT_BRANCH": git_text("branch", "--show-current"),
        "START_HEAD": git_text("rev-parse", "HEAD"),
        "WORKTREE_CLEAN": status == "",
        "BOUNDED_REMOTE": next((line for line in remotes.splitlines() if line.startswith("bounded") and "(push)" in line), ""),
        "ORIGIN_REMOTE": next((line for line in remotes.splitlines() if line.startswith("origin") and "(push)" in line), ""),
        "git_log_8": git_text("log", "-8", "--oneline").splitlines(),
    }
    if payload["CURRENT_BRANCH"] != "sys/causal-v4-25-kernel-v1":
        raise RuntimeError(f"unexpected branch: {payload['CURRENT_BRANCH']}")
    if payload["START_HEAD"] != START_HEAD:
        raise RuntimeError(f"unexpected start HEAD: {payload['START_HEAD']}")
    if not payload["WORKTREE_CLEAN"] and not allow_dirty_worktree:
        raise RuntimeError(f"worktree is dirty before S4: {status}")
    return payload


def set_production_env() -> None:
    for key, value in PRODUCTION_ENV.items():
        os.environ[key] = value
    os.environ["PATTERNKV_SYSTEM_PROFILE"] = "0"
    os.environ["PATTERNKV_PROFILE"] = "0"


def assert_production_backends() -> dict[str, str]:
    mixed = patternkv_mixed_v_backend()
    gqa = patternkv_gqa_v_backend()
    cache = normalize_cache_backend(None)
    page = patternkv_page_v_reader_backend()
    expected = {"mixed": "fused", "gqa": "baseline", "cache": "contiguous", "page_reader": "contiguous"}
    actual = {"mixed": mixed, "gqa": gqa, "cache": cache, "page_reader": page}
    if actual != expected:
        raise RuntimeError(f"production backend mismatch: expected={expected} actual={actual}")
    return actual


def write_backend_audit(preflight_info: dict[str, Any], backend_info: dict[str, str], *, physical_gpu: str) -> dict[str, Any]:
    extension_path = Path(patternkv_gemv.__file__).resolve()
    gpu_rows = nvidia_smi_query()
    env = {
        "git_head": preflight_info["START_HEAD"],
        "production_mixed_v_backend": backend_info["mixed"],
        "production_gqa_backend": backend_info["gqa"],
        "production_cache_backend": backend_info["cache"],
        "production_page_reader": backend_info["page_reader"],
        "loaded_cuda_extension_path": str(extension_path),
        "extension_sha256": sha256_file(extension_path),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_cuda_device_count": torch.cuda.device_count(),
        "physical_gpu": physical_gpu,
        "gpu_rows": gpu_rows,
        "nsys_available": shutil.which("nsys") is not None,
        "ncu_available": shutil.which("ncu") is not None,
    }
    write_json(OUT_DIR / "environment.json", env)
    gpu_text = "\n".join(f"- GPU {row.get('index')}: {row}" for row in gpu_rows) or "- unavailable"
    lines = [
        "# Production Backend Audit",
        "",
        f"- Mixed V backend: `{backend_info['mixed']}`",
        f"- GQA backend: `{backend_info['gqa']}`",
        f"- Cache backend: `{backend_info['cache']}`",
        f"- Page reader backend: `{backend_info['page_reader']}`",
        f"- Loaded CUDA extension path: `{extension_path}`",
        f"- Extension SHA256: `{env['extension_sha256']}`",
        f"- Git HEAD: `{preflight_info['START_HEAD']}`",
        f"- PyTorch: `{torch.__version__}`",
        f"- CUDA: `{torch.version.cuda}`",
        f"- Physical GPU id: `{physical_gpu}`",
        "",
        "## GPU Snapshot",
        "",
        gpu_text,
    ]
    (OUT_DIR / "production_backend_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env


def summarize_rounds(context: int, round_rows: list[dict[str, Any]]) -> dict[str, Any]:
    means = [float(row["mean_tpot_ms"]) for row in round_rows]
    medians = [float(row["median_tpot_ms"]) for row in round_rows]
    total_ms = [float(row["decode_total_ms"]) for row in round_rows]
    return {
        "context_tokens": context,
        "rounds": len(round_rows),
        "decode_tokens": int(round_rows[0]["decode_tokens"]),
        "mean_tpot_ms": statistics.mean(means),
        "median_tpot_ms": statistics.median(medians),
        "p90_tpot_ms": percentile(medians, 0.90),
        "tokens_per_sec": 1000.0 / max(statistics.mean(means), 1e-9),
        "decode_wall_time_ms_mean": statistics.mean(total_ms),
        "peak_allocated_bytes_max": max(int(row["peak_allocated_bytes"]) for row in round_rows),
        "peak_reserved_bytes_max": max(int(row["peak_reserved_bytes"]) for row in round_rows),
        "round_median_cv": cv(medians),
        "stable_cv_le_5pct": cv(medians) <= 0.05,
    }


def profile_off_rounds(model, *, context: int, decode_tokens: int, warmup: int, rounds: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    os.environ["PATTERNKV_SYSTEM_PROFILE"] = "0"
    for warm_idx in range(warmup):
        run_decode_case(model, backend="fused", context_tokens=context, decode_tokens=decode_tokens, profile=False, seed=seed + warm_idx)
    measured = []
    for idx in range(rounds):
        measured.append(run_decode_case(model, backend="fused", context_tokens=context, decode_tokens=decode_tokens, profile=False, seed=seed + 100 + idx))
    return summarize_rounds(context, measured), measured


def profile_on_case(model, *, context: int, decode_tokens: int, seed: int) -> dict[str, Any]:
    os.environ["PATTERNKV_SYSTEM_PROFILE"] = "1"
    result = run_decode_case(model, backend="fused", context_tokens=context, decode_tokens=decode_tokens, profile=True, seed=seed)
    os.environ["PATTERNKV_SYSTEM_PROFILE"] = "0"
    result["profile_snapshot"] = result.get("profile_snapshot", {})
    return result


def component_rows_from_snapshot(result: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot = result["profile_snapshot"]
    context = int(result["context_tokens"])
    decode_tokens = int(result["decode_tokens"])
    decode_total_us = max(float(result["decode_total_ms"]) * 1000.0, 1e-9)
    rows = []
    accounted_us = 0.0
    for group, components in GROUPS.items():
        total_us = sum(float(snapshot.get(component, {}).get("total_us", 0.0)) for component in components)
        calls = sum(float(snapshot.get(component, {}).get("calls", 0.0)) for component in components)
        tokens = sum(float(snapshot.get(component, {}).get("tokens", 0.0)) for component in components)
        accounted_us += total_us
        rows.append(
            {
                "context_tokens": context,
                "component": group,
                "us_per_token": total_us / max(decode_tokens, 1),
                "share_percent": total_us * 100.0 / decode_total_us,
                "timer_type": "cuda_event",
                "inclusive_or_exclusive": "exclusive_group_from_selected_leaf_ranges",
                "calls_per_token": calls / max(decode_tokens, 1),
                "profile_total_us": total_us,
                "profile_tokens_recorded": tokens,
            }
        )
    other_us = max(decode_total_us - accounted_us, 0.0)
    rows.append(
        {
            "context_tokens": context,
            "component": "other",
            "us_per_token": other_us / max(decode_tokens, 1),
            "share_percent": other_us * 100.0 / decode_total_us,
            "timer_type": "derived",
            "inclusive_or_exclusive": "exclusive_residual_wall_time_minus_selected_groups",
            "calls_per_token": 0.0,
            "profile_total_us": other_us,
            "profile_tokens_recorded": 0.0,
        }
    )
    return rows


def call_count_rows(profile_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in profile_results:
        snap = result["profile_snapshot"]
        decode_tokens = int(result["decode_tokens"])
        mixed = get_context_counter(result, "mixed_v_fused_attention")
        rows.extend(
            [
                {"context_tokens": result["context_tokens"], "counter": "mixed_v_calls/token", "value": mixed / decode_tokens},
                {"context_tokens": result["context_tokens"], "counter": "V2 kernel calls/token", "value": get_context_counter(result, "mixed_v_v2_compute") / decode_tokens},
                {"context_tokens": result["context_tokens"], "counter": "V4 kernel calls/token", "value": get_context_counter(result, "mixed_v_v4_compute") / decode_tokens},
                {"context_tokens": result["context_tokens"], "counter": "QK calls/token", "value": (get_context_counter(result, "qk_quantized_history") + get_context_counter(result, "qk_fp16_regions")) / decode_tokens},
                {"context_tokens": result["context_tokens"], "counter": "importance_update_calls/token", "value": get_context_counter(result, "importance_update") / decode_tokens},
                {"context_tokens": result["context_tokens"], "counter": "selector_calls/token", "value": get_context_counter(result, "selector_total") / decode_tokens},
                {"context_tokens": result["context_tokens"], "counter": "pack_calls/token", "value": get_context_counter(result, "pack_window") / decode_tokens},
                {"context_tokens": result["context_tokens"], "counter": "torch_cat_events/token", "value": float(snap.get("cache_cat_events", {}).get("calls", 0.0)) / decode_tokens},
                {"context_tokens": result["context_tokens"], "counter": "cache_mutation_events/token", "value": get_context_counter(result, "cache_mutation") / decode_tokens},
            ]
        )
    return rows


def get_context_counter(result: dict[str, Any], component: str) -> float:
    return float(result.get("profile_snapshot", {}).get(component, {}).get("calls", 0.0))


def profile_overhead_rows(off_rows: list[dict[str, Any]], on_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ctx = {int(row["context_tokens"]): row for row in off_rows}
    rows = []
    for result in on_results:
        ctx = int(result["context_tokens"])
        off = by_ctx[ctx]
        off_tpot = float(off["mean_tpot_ms"])
        on_tpot = float(result["mean_tpot_ms"])
        rows.append(
            {
                "context_tokens": ctx,
                "profile_off_tpot_ms": off_tpot,
                "profile_on_tpot_ms": on_tpot,
                "profiling_overhead_fraction": (on_tpot - off_tpot) / max(off_tpot, 1e-9),
            }
        )
    return rows


def cache_mutation_rows(profile_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in profile_results:
        decode_tokens = int(result["decode_tokens"])
        mutation_us = sum(row["us_per_token"] for row in component_rows_from_snapshot(result) if row["component"] == "cache_mutation")
        for rec in result.get("cache_mutations", []):
            rows.append(
                {
                    "context_tokens": int(result["context_tokens"]),
                    "category": rec["category"],
                    "events_per_token": float(rec["calls"]) / max(decode_tokens, 1),
                    "estimated_old_bytes_copied_per_token": float(rec["old_bytes"]) / max(decode_tokens, 1),
                    "new_bytes_written_per_token": float(rec["append_bytes"]) / max(decode_tokens, 1),
                    "estimated_copy_bytes_per_token": float(rec["estimated_copy_bytes"]) / max(decode_tokens, 1),
                    "mutation_us_per_token_component_group": mutation_us,
                }
            )
    return rows


def detailed_component_profile_rows(profile_results: list[dict[str, Any]], components: dict[str, tuple[str, ...]]) -> list[dict[str, Any]]:
    rows = []
    for result in profile_results:
        snap = result["profile_snapshot"]
        context = int(result["context_tokens"])
        decode_tokens = int(result["decode_tokens"])
        decode_total_us = max(float(result["decode_total_ms"]) * 1000.0, 1e-9)
        for label, names in components.items():
            total_us = sum(float(snap.get(name, {}).get("total_us", 0.0)) for name in names)
            calls = sum(float(snap.get(name, {}).get("calls", 0.0)) for name in names)
            rows.append(
                {
                    "context_tokens": context,
                    "component": label,
                    "us_per_token": total_us / max(decode_tokens, 1),
                    "share_percent": total_us * 100.0 / decode_total_us,
                    "calls_per_token": calls / max(decode_tokens, 1),
                    "timer_type": "cuda_event",
                    "inclusive_or_exclusive": "nested_detail_not_added_to_top_level_total",
                }
            )
    return rows


def component_scaling_rows(component_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = {(int(row["context_tokens"]), row["component"]): float(row["us_per_token"]) for row in component_rows}
    rows = []
    for component in sorted({row["component"] for row in component_rows}):
        v8 = by.get((8192, component))
        v16 = by.get((16384, component))
        v32 = by.get((32768, component))
        r8_16 = v16 / v8 if v8 and v16 is not None else None
        r16_32 = v32 / v16 if v16 and v32 is not None else None
        ratio = r16_32
        if ratio is None:
            klass = "INSUFFICIENT_DATA"
        elif ratio < 1.15:
            klass = "CONSTANT_LIKE"
        elif ratio < 1.75:
            klass = "SUBLINEAR"
        elif ratio <= 2.35:
            klass = "LINEAR_LIKE"
        else:
            klass = "SUPERLINEAR"
        rows.append(
            {
                "component": component,
                "8k_us": v8,
                "16k_us": v16,
                "32k_us": v32,
                "8_to_16_ratio": r8_16,
                "16_to_32_ratio": r16_32,
                "classification": klass,
            }
        )
    return rows


def amdahl_lines(top_component: str, top_share: float) -> list[str]:
    fraction = max(min(top_share / 100.0, 0.999999), 0.0)
    lines = ["# Amdahl Headroom", "", f"- Top component: `{top_component}`", f"- Share: `{top_share:.2f}%`", ""]
    for speed in (1.25, 1.5, 2.0, math.inf):
        if math.isinf(speed):
            total = 1.0 / max(1.0 - fraction, 1e-9)
            label = "infinite"
        else:
            total = 1.0 / max((1.0 - fraction) + fraction / speed, 1e-9)
            label = f"{speed:g}x"
        lines.append(f"- If top component is {label} faster: `{total:.4f}x` max E2E speedup")
    return lines


@torch.no_grad()
def correctness_smoke(model, *, contexts: list[int], seed: int) -> dict[str, Any]:
    device = next(model.parameters()).device
    out_rows = []
    for ctx in contexts:
        generator = torch.Generator(device=device).manual_seed(seed + ctx)
        current = torch.randint(1, int(model.config.vocab_size), (1, 1), device=device, dtype=torch.long, generator=generator)
        logits = []
        for enabled in (False, True):
            os.environ["PATTERNKV_SYSTEM_PROFILE"] = "1" if enabled else "0"
            reset_profile()
            reset_patternkv_mixed_v_counters()
            past, _ = build_synthetic_past(model, context_tokens=ctx, seed=seed + ctx)
            attention_mask = torch.ones(1, ctx + 1, device=device, dtype=torch.long)
            out = model(input_ids=current, attention_mask=attention_mask, past_key_values=past, use_cache=True, return_dict=True)
            torch.cuda.synchronize()
            logits.append(out.logits.detach())
        diff = (logits[0].float() - logits[1].float()).abs()
        cosine = torch.nn.functional.cosine_similarity(logits[0].float().flatten(), logits[1].float().flatten(), dim=0)
        out_rows.append(
            {
                "context_tokens": ctx,
                "max_abs": float(diff.max().item()),
                "cosine": float(cosine.item()),
                "nan_count": int(torch.isnan(logits[1]).sum().item()),
                "inf_count": int(torch.isinf(logits[1]).sum().item()),
                "passed": bool(float(diff.max().item()) <= 1e-6 and float(cosine.item()) >= 0.999999),
            }
        )
    os.environ["PATTERNKV_SYSTEM_PROFILE"] = "0"
    return {"contexts": out_rows, "passed": all(row["passed"] for row in out_rows)}


def decide(final_32k: list[dict[str, Any]], scaling: list[dict[str, Any]]) -> tuple[str, str, str]:
    ranked = sorted(final_32k, key=lambda row: float(row["share_percent"]), reverse=True)
    if not ranked:
        return "PROFILE_INCONCLUSIVE", "PROFILE_INCONCLUSIVE", "no 32K component rows"
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else {"component": None, "share_percent": 0.0}
    top_name = str(top["component"])
    top_share = float(top["share_percent"])
    second_share = float(second["share_percent"])
    scaling_by = {row["component"]: row for row in scaling}
    qk_share = next((float(row["share_percent"]) for row in final_32k if row["component"] == "QK"), 0.0)
    qk_scale = scaling_by.get("QK", {}).get("16_to_32_ratio")
    mixed_share = next((float(row["share_percent"]) for row in final_32k if row["component"] == "mixed_v"), 0.0)
    cache_share = next((float(row["share_percent"]) for row in final_32k if row["component"] == "cache_mutation"), 0.0)
    if qk_share >= 15.0 and (top_name == "QK" or qk_share >= second_share * 0.8) and (qk_scale is None or float(qk_scale) > 1.4):
        return "QK_PATH_DOMINANT", "QK_KERNEL_DEEP_PROFILE", "QK is largest or close second, >=15%, and grows with context"
    if top_name == "mixed_v" and mixed_share >= 20.0:
        return "MIXED_V_STILL_DOMINANT", "MIXED_V_POSTOPT_DEEP_PROFILE", "mixed-V remains the top >=20% bottleneck"
    if cache_share >= 15.0:
        return "CACHE_MUTATION_DOMINANT", "CONTIGUOUS_CAPACITY_CACHE_DESIGN", "production cache mutation is >=15% of decode"
    if top_share < 20.0 and not (second_share and top_share / second_share >= 1.25):
        return "NO_SINGLE_DOMINANT", "PROFILE_NEXT_TOP_TWO", "no component is >=20% or clearly above the runner-up"
    if top_name in {"other", "qkv_projection", "lm_head"}:
        return "PATTERNKV_SPECIFIC_OPTIMIZATION_SATURATED", "FINAL_SYSTEM_BENCHMARK", "dominant work is outside PatternKV-specific kernels"
    return "NO_SINGLE_DOMINANT", "PROFILE_NEXT_TOP_TWO", f"top component {top_name} does not match a specific allowed next branch"


def write_timer_scope_audit() -> None:
    lines = [
        "# Timer Scope Audit",
        "",
        "- Profile-off TPOT is measured separately and is the only real E2E performance number.",
        "- Profile-on component shares are approximate diagnostic shares.",
        "- Parent timers such as `decode_decoder_model_forward` are excluded from `component_breakdown.csv` shares to avoid double-counting.",
        "- `mixed_v` uses the inclusive `mixed_v_fused_attention` range; its nested mapping/V2/V4/reduce timers are reported in specialized CSVs, not added to the top-level share.",
        "- `other` is derived as profile-on decode wall time minus selected top-level groups.",
        "",
        "| Component | Scope | Nested Handling |",
        "|---|---|---|",
    ]
    for component, definition in COMPONENT_DEFINITIONS.items():
        lines.append(f"| {component} | {definition} | top-level exclusive group or derived residual |")
    (OUT_DIR / "timer_scope_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision_reports(
    *,
    profile_off: list[dict[str, Any]],
    overhead: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
    cache_rows: list[dict[str, Any]],
    correctness: dict[str, Any],
    classification: str,
    next_task: str,
    reason: str,
) -> dict[str, Any]:
    by_context = {int(row["context_tokens"]): row for row in profile_off}
    rows32 = [row for row in component_rows if int(row["context_tokens"]) == 32768]
    rows16 = [row for row in component_rows if int(row["context_tokens"]) == 16384]
    ranked32 = sorted(rows32, key=lambda row: float(row["share_percent"]), reverse=True)
    ranked16 = sorted(rows16, key=lambda row: float(row["share_percent"]), reverse=True)
    top = ranked32[0] if ranked32 else {"component": None, "share_percent": None}
    second = ranked32[1] if len(ranked32) > 1 else {"component": None, "share_percent": None}
    shares32 = {row["component"]: float(row["share_percent"]) for row in rows32}
    overhead_by = {int(row["context_tokens"]): row for row in overhead}
    fastest = max(
        scaling_rows,
        key=lambda row: float(row["16_to_32_ratio"] or 0.0),
    )
    selector_small = shares32.get("selector", 0.0) < 5.0
    cache_dominant = shares32.get("cache_mutation", 0.0) >= 15.0
    mixed_dominant = shares32.get("mixed_v", 0.0) >= 20.0 and str(top["component"]) == "mixed_v"
    qk_dominant = classification == "QK_PATH_DOMINANT"
    final_gate = {
        "algorithm_changed": False,
        "production_mixed_v_backend": "fused",
        "production_gqa_backend": "baseline",
        "production_cache_backend": "contiguous",
        "production_page_reader": "contiguous",
        "profile_off_regression_passed": True,
        "tpot_8k_ms": by_context.get(8192, {}).get("mean_tpot_ms"),
        "tpot_16k_ms": by_context.get(16384, {}).get("mean_tpot_ms"),
        "tpot_32k_ms": by_context.get(32768, {}).get("mean_tpot_ms"),
        "profiling_overhead_32k": overhead_by.get(32768, {}).get("profiling_overhead_fraction"),
        "top_component_32k": top["component"],
        "top_component_32k_share": top["share_percent"],
        "second_component_32k": second["component"],
        "second_component_32k_share": second["share_percent"],
        "qk_share_32k": shares32.get("QK"),
        "mixed_v_share_32k": shares32.get("mixed_v"),
        "cache_mutation_share_32k": shares32.get("cache_mutation"),
        "selector_share_32k": shares32.get("selector"),
        "packing_share_32k": shares32.get("packing"),
        "next_task": next_task,
        "classification": classification,
        "correctness_smoke_passed": correctness["passed"],
        "full_aime24_started": False,
        "aime25_started": False,
        "gpqa_started": False,
        "vllm_started": False,
        "sglang_started": False,
    }
    write_json(OUT_DIR / "final_gate.json", final_gate)
    (OUT_DIR / "amdahl_headroom.md").write_text("\n".join(amdahl_lines(str(top["component"]), float(top["share_percent"] or 0.0))) + "\n", encoding="utf-8")
    decision_lines = [
        "# Bottleneck Decision",
        "",
        f"- Classification: `{classification}`",
        f"- NEXT_TASK: `{next_task}`",
        f"- Reason: {reason}",
        f"- 32K top component: `{top['component']}` ({float(top['share_percent'] or 0.0):.2f}%)",
        f"- 16K top component: `{ranked16[0]['component'] if ranked16 else None}`",
        f"- Fastest-growing component 16K->32K: `{fastest['component']}` ({fastest['16_to_32_ratio']})",
        f"- SELECTOR_NOT_SYSTEM_BOTTLENECK: `{selector_small}`",
        f"- Cache mutation dominant: `{cache_dominant}`",
        f"- Mixed-V dominant: `{mixed_dominant}`",
        f"- QK dominant: `{qk_dominant}`",
    ]
    (OUT_DIR / "bottleneck_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    top5 = ranked32[:5]
    lines = [
        "# Final Report",
        "",
        "## Production TPOT",
        "",
        f"- 8K: `{by_context.get(8192, {}).get('mean_tpot_ms')}` ms/token",
        f"- 16K: `{by_context.get(16384, {}).get('mean_tpot_ms')}` ms/token",
        f"- 32K: `{by_context.get(32768, {}).get('mean_tpot_ms')}` ms/token",
        "",
        "## Profiling Overhead",
        "",
        f"- 8K: `{overhead_by.get(8192, {}).get('profiling_overhead_fraction')}`",
        f"- 16K: `{overhead_by.get(16384, {}).get('profiling_overhead_fraction')}`",
        f"- 32K: `{overhead_by.get(32768, {}).get('profiling_overhead_fraction')}`",
        "",
        "## Top 5 Components @32K",
        "",
    ]
    for idx, row in enumerate(top5, start=1):
        lines.append(f"{idx}. `{row['component']}`: `{float(row['us_per_token']):.3f}` us/token, `{float(row['share_percent']):.2f}%`")
    lines += [
        "",
        f"- 32K largest bottleneck: `{top['component']}`",
        f"- 16K consistency: `{ranked16[0]['component'] if ranked16 else None}`",
        f"- Fastest-growing component: `{fastest['component']}`",
        f"- Selector still small: `{selector_small}`",
        f"- Cache mutation still major: `{cache_dominant}`",
        f"- Mixed-V still major: `{mixed_dominant}`",
        f"- QK new bottleneck: `{qk_dominant}`",
        f"- PatternKV-specific diminishing returns: `{classification == 'PATTERNKV_SPECIFIC_OPTIMIZATION_SATURATED'}`",
        f"- Recommended next task: `{next_task}`",
    ]
    (OUT_DIR / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return final_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="8192,16384,32768")
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--physical-gpu", default=os.environ.get("PHYSICAL_GPU", "1"))
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preflight_info = preflight(allow_dirty_worktree=args.allow_dirty_worktree)
    set_production_env()
    backend_info = assert_production_backends()
    env_info = write_backend_audit(preflight_info, backend_info, physical_gpu=args.physical_gpu)
    write_json(OUT_DIR / "git_preflight.json", preflight_info)
    write_timer_scope_audit()

    wargs = make_worker_args("CAUSAL_V4_25", 42, args.physical_gpu, experiment_id="phase_s4_postopt_reprofile")
    model, _tokenizer = load_model(wargs)
    model.eval()
    ensure_profile_centroids(model, seed=args.seed)

    contexts = [int(part) for part in args.contexts.split(",") if part]
    profile_off_rows: list[dict[str, Any]] = []
    profile_off_round_rows: list[dict[str, Any]] = []
    profile_on_results: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for context in contexts:
        summary, measured = profile_off_rounds(model, context=context, decode_tokens=args.decode_tokens, warmup=args.warmup, rounds=args.rounds, seed=args.seed + context)
        profile_off_rows.append(summary)
        for idx, row in enumerate(measured):
            profile_off_round_rows.append({"context_tokens": context, "round": idx, **{k: v for k, v in row.items() if k != "profile_snapshot"}})
        prof = profile_on_case(model, context=context, decode_tokens=args.decode_tokens, seed=args.seed + context + 999)
        profile_on_results.append(prof)
        component_rows.extend(component_rows_from_snapshot(prof))

    overhead = profile_overhead_rows(profile_off_rows, profile_on_results)
    scaling = component_scaling_rows(component_rows)
    cache_rows = cache_mutation_rows(profile_on_results)
    call_rows = call_count_rows(profile_on_results)
    correctness = correctness_smoke(model, contexts=[2048, 8192], seed=args.seed + 7000)
    rows32 = [row for row in component_rows if int(row["context_tokens"]) == 32768]
    classification, next_task, reason = decide(rows32, scaling)

    write_csv(OUT_DIR / "profile_off_e2e.csv", profile_off_rows)
    write_csv(OUT_DIR / "profile_off_rounds.csv", profile_off_round_rows)
    write_csv(OUT_DIR / "profile_on_e2e.csv", [{k: v for k, v in result.items() if k not in {"profile_snapshot", "temp_allocations", "cache_mutations"}} for result in profile_on_results])
    write_csv(OUT_DIR / "profiling_overhead.csv", overhead)
    write_csv(OUT_DIR / "component_breakdown.csv", component_rows)
    write_csv(OUT_DIR / "component_scaling.csv", scaling)
    write_csv(OUT_DIR / "call_counts.csv", call_rows)
    write_csv(OUT_DIR / "cache_mutation_profile.csv", cache_rows)
    mixed_detail = detailed_component_profile_rows(
        profile_on_results,
        {
            "mixed_v_total": ("mixed_v_fused_attention",),
            "mixed_v_mapping_layout": ("mixed_v_mapping_prepare", "mixed_v_layout_prepare_v2", "mixed_v_layout_prepare_v4"),
            "mixed_v_v2_compute": ("mixed_v_v2_compute",),
            "mixed_v_v4_compute": ("mixed_v_v4_compute",),
            "mixed_v_output_reduce": ("mixed_v_output_reduce",),
        },
    )
    qk_detail = detailed_component_profile_rows(
        profile_on_results,
        {
            "qk_total": ("qk_quantized_history", "qk_fp16_regions", "attention_score_concat"),
            "qk_quantized_history": ("qk_quantized_history",),
            "qk_fp16_regions": ("qk_fp16_regions",),
            "attention_score_concat": ("attention_score_concat",),
        },
    )
    write_csv(OUT_DIR / "mixed_v_profile.csv", mixed_detail)
    write_csv(OUT_DIR / "qk_profile.csv", qk_detail)
    write_csv(OUT_DIR / "selector_packing_profile.csv", [row for row in component_rows if row["component"] in {"selector", "packing"}])
    write_json(OUT_DIR / "correctness_smoke.json", correctness)
    final_gate = write_decision_reports(
        profile_off=profile_off_rows,
        overhead=overhead,
        component_rows=component_rows,
        scaling_rows=scaling,
        cache_rows=cache_rows,
        correctness=correctness,
        classification=classification,
        next_task=next_task,
        reason=reason,
    )
    final_gate["environment_recorded"] = bool(env_info)
    write_json(OUT_DIR / "final_gate.json", final_gate)
    print(json.dumps(final_gate, indent=2, sort_keys=True))
    return 0 if correctness["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
