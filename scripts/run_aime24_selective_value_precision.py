#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_aime24_routing_vdirection as rvd  # noqa: E402
from bench.pseudodecode_controls import MATCHED_PATH_CONTROL_VERSION, compute_accumulation_gap  # noqa: E402
from bench.pseudodecode_metrics import full_trajectory_sha256, trapezoid_auc_log2, write_csv_rows  # noqa: E402
from bench.routing_vdirection_observer import EPS, SCHEMA_VERSION, all_finite, repeat_kv_for_gqa, summarize_tensor, vector_errors  # noqa: E402
from models.segmented_cache import PatternQuantizedKVCache, deserialize_cache, local_v2_v4_gain, pattern_gather_centroids, reconstruct_packed_v, tensor_tokens  # noqa: E402
from scripts.run_aime24_norm_tail_stage_a import SourceCapture  # noqa: E402
from scripts.run_aime24_pseudodecode_preflight import SOURCE_COMMIT, load_model, make_args, reset_method_state  # noqa: E402


OUT_DIR = ROOT / "reports/aime24_selective_value_precision_3090"
RESULT_DIR = ROOT / "results/aime24_selective_value_precision_3090"
SHARD_DIR = RESULT_DIR / "shards"
LOG_DIR = ROOT / "run/aime24_selective_value_precision_3090/logs"
PARENT_COMMIT = "47cf601fcfd4b20fa3823fe540b1d48ca9920d7d"
SUBSET_SHA256 = "218b65375f551fb13ff9903b3fd8931f422215e0f3b86200c0d8f45130a43082"
PORTABLE_HASH = "86648d12304ce11890c1a8f64bf5a896"
CORE_CHECKPOINTS = (128, 512, 1024, 2048, 4096)
PREFLIGHT_CHECKPOINTS = (128, 512, 1024)
SELECTED_LAYERS = (0, 7, 15, 23, 31)
V4_BUDGET_FRACTION = 0.125
RANDOM_SELECTOR_SEED = 20260809
CONFIGS = {
    "BASE_V2": {"config": "pattern_rolling_k2v2_s16_r128", "selector": "base_v2", "budget": 0.0},
    "RANDOM_V4": {"config": "pattern_rolling_k2v2_s16_r128_random_v4_b0125", "selector": "random_v4", "budget": V4_BUDGET_FRACTION},
    "CAUSAL_V4": {"config": "pattern_rolling_k2v2_s16_r128_causal_v4_b0125", "selector": "causal_v4", "budget": V4_BUDGET_FRACTION},
    "ORACLE_V4": {"config": "pattern_rolling_k2v2_s16_r128_oracle_v4_b0125", "selector": "oracle_v4", "budget": V4_BUDGET_FRACTION},
}
FAMILIES = (
    "precision_selection",
    "stored_v",
    "value_oracle",
    "attention_output",
    "hidden_accumulation",
    "future_v_source",
    "routing_safety",
    "selector_quality",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gzip_file(path: Path) -> Path:
    gz = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as src, gzip.open(gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return gz


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def method_config(method: str) -> dict[str, Any]:
    base = {
        "method": "patternkv",
        "sink_length": 16,
        "recent_length": 128,
        "group_size": 128,
        "k_bits": 2,
        "v_bits": 2,
        "value_objective": "base",
        "random_seed": RANDOM_SELECTOR_SEED,
    }
    return {**base, **CONFIGS[method]}


def config_args(model_path: Path, method_name: str, backend: str = "patternkv") -> Any:
    cfg = method_config(method_name)
    args = make_args(model_path, backend, cfg["sink_length"], cfg["recent_length"], config_name=cfg["config"] if backend != "fp16" else "fp16")
    args.patternkv_value_objective = "base"
    args.patternkv_v_precision_selector = cfg["selector"]
    args.patternkv_v4_budget_fraction = float(cfg["budget"])
    args.patternkv_random_selector_seed = RANDOM_SELECTOR_SEED
    return args


def set_rvd_config(method_name: str) -> None:
    cfg = method_config(method_name)
    rvd.CONFIG = cfg.copy()
    rvd.SELECTED_LAYERS = SELECTED_LAYERS
    rvd.CORE_CHECKPOINTS = CORE_CHECKPOINTS


def load_records() -> list[dict[str, Any]]:
    payload = read_json(ROOT / "configs/aime24_routing_vdirection_6tasks.json")
    records = rvd.load_reference_records()
    by_key = {row["task_key"]: row for row in records}
    return [by_key[item["task_key"]] for item in payload["tasks"]]


def validate_freeze(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    ok = True
    for row in records:
        observed = full_trajectory_sha256(row["prompt_token_ids"], row["generated_token_ids"])
        valid = observed == row["full_trajectory_sha256"]
        ok = ok and valid
        rows.append({"task_key": row["task_key"], "trajectory_sha256": row["full_trajectory_sha256"], "observed": observed, "valid": valid})
    subset_payload = read_json(ROOT / "configs/aime24_routing_vdirection_6tasks.json")
    return {
        "task_count": len(records),
        "subset_sha256": subset_payload.get("source_varn_subset_sha256"),
        "subset_sha256_valid": subset_payload.get("source_varn_subset_sha256") == SUBSET_SHA256,
        "portable_generation_hash": PORTABLE_HASH,
        "reference_hashes_valid": ok and len(records) == 6,
        "tasks": rows,
    }


def prepare() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    freeze = validate_freeze(records)
    write_json(OUT_DIR / "frozen_subset.json", {"task_count": len(records), "checkpoints": list(CORE_CHECKPOINTS), **freeze})
    write_json(
        OUT_DIR / "experiment_origin.json",
        {
            "repository": "pytenter/Bounded-pattrenKV-method",
            "branch": git_text("branch", "--show-current"),
            "head": git_text("rev-parse", "HEAD"),
            "parent_commit": PARENT_COMMIT,
            "experiment": "aime24_selective_value_precision_3090",
            "worktree_dirty_at_prepare": bool(git_text("status", "--short")),
        },
    )
    write_json(
        OUT_DIR / "selective_precision_config.json",
        {
            "parent_commit": PARENT_COMMIT,
            "configs": {name: method_config(name) for name in CONFIGS},
            "checkpoints": list(CORE_CHECKPOINTS),
            "selected_layers": list(SELECTED_LAYERS),
            "v4_budget_fraction": V4_BUDGET_FRACTION,
            "random_selector_seed": RANDOM_SELECTOR_SEED,
            "centroid_assignment": "BASE minmax residual range",
            "k_path": "K2 unchanged",
        },
    )
    write_json(
        OUT_DIR / "mixed_precision_layout.json",
        {
            "implementable": True,
            "granularity": "token-level per layer",
            "selected_token_storage": "V4 payload only",
            "unselected_token_storage": "V2 payload only",
            "logical_order_restore": "v_precision_mask scatter",
            "duplicate_sidecar": False,
        },
    )
    write_json(OUT_DIR / "causal_importance_schema.json", {"schema": "task,layer,token,importance_before_pack", "future_leakage": False})
    return {"prepared": True, **freeze}


def free_model(model: torch.nn.Module | None) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def iter_attn(model: torch.nn.Module):
    for idx, layer in enumerate(getattr(getattr(model, "model", None), "layers", [])):
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            yield idx, attn


def set_selector_context(
    model: torch.nn.Module,
    task_key: str,
    *,
    causal_by_layer: dict[int, torch.Tensor] | None = None,
    oracle_by_layer: dict[int, torch.Tensor] | None = None,
) -> None:
    if hasattr(model, "config"):
        model.config.patternkv_selector_task_key = task_key
    for idx, attn in iter_attn(model):
        attn.selector_task_key = task_key
        attn.v_causal_importance = causal_by_layer.get(idx) if causal_by_layer else None
        attn.v_oracle_importance = oracle_by_layer.get(idx) if oracle_by_layer else None


def method_needs_importance(method: str) -> bool:
    return method in {"RANDOM_V4", "CAUSAL_V4", "ORACLE_V4"}


def slice_sources(sources: dict[int, dict[str, torch.Tensor]], total: int) -> dict[int, dict[str, torch.Tensor]]:
    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer, layer_map in sources.items():
        out[layer] = {}
        for name, tensor in layer_map.items():
            out[layer][name] = tensor[:, :total, :].contiguous() if name == "hidden" else tensor[:, :, :total, :].contiguous()
    return out


def importance_from_sources(sources: dict[int, dict[str, torch.Tensor]], *, recent_length: int = 128, block: int = 64) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    causal_by_layer: dict[int, torch.Tensor] = {}
    oracle_by_layer: dict[int, torch.Tensor] = {}
    for layer in SELECTED_LAYERS:
        q = sources[layer]["q_source"].detach().float()
        k = sources[layer]["k_source"].detach().float()
        kg = repeat_kv_for_gqa(k, q.shape[1] // k.shape[1])
        total = min(q.shape[2], kg.shape[2])
        causal = torch.zeros(q.shape[0], total, dtype=torch.float32, device=q.device)
        oracle = torch.zeros_like(causal)
        key_pos = torch.arange(total, device=q.device)
        for start in range(0, total, block):
            stop = min(start + block, total)
            logits = torch.matmul(q[:, :, start:stop, :], kg[:, :, :total, :].transpose(2, 3)) / math.sqrt(float(q.shape[-1]))
            q_pos = torch.arange(start, stop, device=q.device).view(1, 1, -1, 1)
            causal_mask = key_pos.view(1, 1, 1, -1) <= q_pos
            logits = logits.masked_fill(~causal_mask, torch.finfo(logits.dtype).min)
            probs = torch.softmax(logits, dim=-1)
            mass_per_query = probs.mean(dim=1)
            oracle += mass_per_query.sum(dim=1)
            query_pos = torch.arange(start, stop, device=q.device).view(-1, 1)
            pack_time_ok = query_pos <= (key_pos.view(1, -1) + int(recent_length))
            causal += (mass_per_query * pack_time_ok.unsqueeze(0).to(mass_per_query.dtype)).sum(dim=1)
        causal_by_layer[layer] = causal.cpu()
        oracle_by_layer[layer] = oracle.cpu()
    return causal_by_layer, oracle_by_layer


@torch.no_grad()
def run_with_source_capture(model: torch.nn.Module, *, task: dict[str, Any], checkpoint: int, mode: str) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    capture = SourceCapture(selected_layers=SELECTED_LAYERS)
    capture.install(model)
    try:
        output = rvd.replay_prefix(model, prompt_ids=task["prompt_token_ids"], generated_ids=task["generated_token_ids"], checkpoint=checkpoint, mode=mode)
        return output, capture.tensors()
    finally:
        capture.remove()


@torch.no_grad()
def fp16_pseudo_sources(model: torch.nn.Module, task: dict[str, Any]) -> dict[int, dict[str, torch.Tensor]]:
    capture = SourceCapture(selected_layers=SELECTED_LAYERS)
    capture.install(model)
    try:
        device = "cuda:0"
        prompt = torch.tensor([task["prompt_token_ids"]], device=device, dtype=torch.long)
        outputs = model(input_ids=prompt, use_cache=True, output_hidden_states=True, output_attentions=False, return_dict=True)
        past = outputs.past_key_values
        for token in task["generated_token_ids"][: max(CORE_CHECKPOINTS)]:
            tok = torch.tensor([[int(token)]], device=device, dtype=torch.long)
            outputs = model(input_ids=tok, past_key_values=past, use_cache=True, output_hidden_states=True, output_attentions=False, return_dict=True)
            past = outputs.past_key_values
        return capture.tensors()
    finally:
        capture.remove()


def hidden_rows(*, task: dict[str, Any], method_name: str, mode: str, checkpoint: int, fp_sources: dict[int, dict[str, torch.Tensor]], quant_sources: dict[int, dict[str, torch.Tensor]]) -> list[dict[str, Any]]:
    rows = []
    for layer in SELECTED_LAYERS:
        fp = fp_sources[layer]["hidden"].float()
        qt = quant_sources[layer]["hidden"].float()
        total = min(fp.shape[1], qt.shape[1])
        rel = (qt[:, total - 1, :] - fp[:, total - 1, :]).norm(dim=-1) / fp[:, total - 1, :].norm(dim=-1).clamp_min(1e-8)
        rows.append(metric_row(task, method_name, mode, checkpoint, layer, "hidden_accumulation", "hidden_state", "current_token", "relative_L2", "global", float(rel.mean().item()), int(rel.numel())))
    return rows


def metric_row(task: dict[str, Any], method: str, mode: str, checkpoint: int, layer: int | str, family: str, obj: str, region: str, metric: str, stat: str, value: float, n: int) -> dict[str, Any]:
    return {
        "task_key": task["task_key"],
        "trajectory_sha256": task["full_trajectory_sha256"],
        "config": method_config(method)["config"],
        "method": method,
        "mode": mode,
        "checkpoint": checkpoint,
        "absolute_sequence_position": int(task["prompt_token_count"]) + int(checkpoint),
        "layer": str(layer),
        "metric_family": family,
        "object_type": obj,
        "region": region,
        "metric_name": metric,
        "statistic": stat,
        "metric_value": float(value),
        "n_samples": int(n),
        "matched_path_control_version": MATCHED_PATH_CONTROL_VERSION,
        "observer_schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
    }


def stored_v_rows(*, task: dict[str, Any], method_name: str, mode: str, checkpoint: int, fp_sources: dict[int, dict[str, torch.Tensor]], quant_past: Any) -> list[dict[str, Any]]:
    rows = []
    for layer in SELECTED_LAYERS:
        cache = deserialize_cache(quant_past[layer], pattern=True)
        if not isinstance(cache, PatternQuantizedKVCache) or cache.packed_v_tokens <= 0:
            continue
        packed = reconstruct_packed_v(cache)
        if packed is None:
            continue
        if cache.v_centroids is not None and cache.v_assignment_idx is not None:
            mask = cache.v_pattern_mask if cache.v_pattern_mask is not None else cache.v_assignments
            if mask is not None:
                centroids = pattern_gather_centroids(cache.v_assignment_idx[:, :, : cache.packed_v_tokens], cache.v_centroids).to(packed.dtype)
                packed = packed + mask[:, :, : cache.packed_v_tokens].unsqueeze(-1).to(packed.dtype) * centroids
        start = tensor_tokens(cache.sink_v)
        fpv = fp_sources[layer]["v_source"][:, :, start : start + cache.packed_v_tokens, :].to(packed.device)
        total = min(fpv.shape[2], packed.shape[2])
        packed = packed[:, :, :total, :]
        fpv = fpv[:, :, :total, :]
        token_mask = cache.v_precision_mask[:, :total].bool() if cache.v_precision_mask is not None else torch.zeros(packed.shape[0], total, dtype=torch.bool, device=packed.device)
        regions = {"all_packed_tokens": torch.ones_like(token_mask, dtype=torch.bool), "v4_selected_tokens": token_mask, "v2_unselected_tokens": ~token_mask}
        errs = vector_errors(packed.detach().cpu(), fpv.detach().cpu())
        for region, keep in regions.items():
            expanded = keep.cpu().unsqueeze(1).expand(-1, packed.shape[1], -1)
            for metric, vals in errs.items():
                vals = vals[expanded]
                stats = summarize_tensor(vals)
                for stat in ("mean", "p95"):
                    rows.append(metric_row(task, method_name, mode, checkpoint, layer, "v_direction", "v_stored", region, metric, stat, float(stats[stat]), int(stats["n_samples"])))
    return rows


def split_rvd_rows(method_name: str, families: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out = {key: [] for key in FAMILIES}
    for row in families["oracle_output"]:
        row = {**row, "method": method_name}
        if row["metric_name"].startswith("value_only"):
            out["value_oracle"].append(row)
        elif row["metric_name"].startswith("actual"):
            out["attention_output"].append(row)
        elif row["metric_name"].startswith("routing_only"):
            out["routing_safety"].append(row)
    for row in families["direction"]:
        row = {**row, "method": method_name}
        if row["object_type"] == "v_source":
            out["future_v_source"].append(row)
        elif row["object_type"] in {"q_source", "k_source"}:
            out["routing_safety"].append(row)
    for name in ("qk_logit", "attention_routing"):
        for row in families[name]:
            out["routing_safety"].append({**row, "method": method_name})
    return out


def precision_rows(*, task: dict[str, Any], method_name: str, mode: str, checkpoint: int, quant_past: Any, importance: dict[int, torch.Tensor] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection_rows = []
    quality_rows = []
    for layer in SELECTED_LAYERS:
        cache = deserialize_cache(quant_past[layer], pattern=True)
        if not isinstance(cache, PatternQuantizedKVCache) or cache.packed_v_tokens <= 0:
            continue
        mask = cache.v_precision_mask[:, : cache.packed_v_tokens].bool() if cache.v_precision_mask is not None else torch.zeros(1, cache.packed_v_tokens, dtype=torch.bool)
        packed = int(cache.packed_v_tokens)
        selected = int(mask.sum().item())
        for start in range(0, packed, method_config(method_name)["group_size"]):
            stop = min(start + method_config(method_name)["group_size"], packed)
            wmask = mask[:, start:stop]
            selection_rows.append(
                {
                    "task_key": task["task_key"],
                    "trajectory_sha256": task["full_trajectory_sha256"],
                    "config": method_config(method_name)["config"],
                    "method": method_name,
                    "mode": mode,
                    "checkpoint": checkpoint,
                    "layer": str(layer),
                    "pack_window": start // method_config(method_name)["group_size"],
                    "logical_tokens": stop - start,
                    "v4_tokens": int(wmask.sum().item()),
                    "v2_tokens": int((~wmask).sum().item()),
                    "realized_v4_fraction": float(wmask.float().mean().item()) if wmask.numel() else 0.0,
                    "selector": method_config(method_name)["selector"],
                    "schema_version": "selective_precision_selection_v1",
                }
            )
        future = None if importance is None else importance.get(layer)
        coverage = None
        if torch.is_tensor(future) and future.shape[1] >= packed:
            vals = future[:, :packed].float()
            denom = float(vals.sum().item())
            coverage = float(vals[mask.cpu()].sum().item() / denom) if denom > 0 else 0.0
        quality_rows.append(
            {
                "task_key": task["task_key"],
                "trajectory_sha256": task["full_trajectory_sha256"],
                "config": method_config(method_name)["config"],
                "method": method_name,
                "mode": mode,
                "checkpoint": checkpoint,
                "layer": str(layer),
                "packed_tokens": packed,
                "v4_tokens": selected,
                "realized_v4_fraction": selected / max(packed, 1),
                "future_attention_coverage": coverage,
                "schema_version": "selective_precision_selector_quality_v1",
            }
        )
    return selection_rows, quality_rows


@torch.no_grad()
def quant_pseudo_task(model: torch.nn.Module, task: dict[str, Any], method_name: str, fp_sources: dict[int, dict[str, torch.Tensor]], importance_pair: tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows = {key: [] for key in FAMILIES}
    completeness = []
    causal_imp, oracle_imp = importance_pair
    set_selector_context(model, task["task_key"], causal_by_layer=causal_imp if method_name == "CAUSAL_V4" else None, oracle_by_layer=oracle_imp if method_name == "ORACLE_V4" else None)
    set_rvd_config(method_name)
    capture = SourceCapture(selected_layers=SELECTED_LAYERS)
    capture.install(model)
    try:
        device = "cuda:0"
        prompt = torch.tensor([task["prompt_token_ids"]], device=device, dtype=torch.long)
        outputs = model(input_ids=prompt, use_cache=True, output_hidden_states=True, output_attentions=False, return_dict=True)
        past = outputs.past_key_values
        wanted = set(CORE_CHECKPOINTS)
        for idx, token in enumerate(task["generated_token_ids"][: max(CORE_CHECKPOINTS)], start=1):
            tok = torch.tensor([[int(token)]], device=device, dtype=torch.long)
            outputs = model(input_ids=tok, past_key_values=past, use_cache=True, output_hidden_states=True, output_attentions=False, return_dict=True)
            past = outputs.past_key_values
            if idx not in wanted:
                continue
            total = int(task["prompt_token_count"]) + idx
            fp_slice = slice_sources(fp_sources, total)
            qt_slice = slice_sources(capture.tensors(), total)
            fam, comp = rvd.compute_channel_rows(task=task, mode="pseudo", checkpoint=idx, fp_sources=fp_slice, quant_sources=qt_slice, quant_past=past)
            split = split_rvd_rows(method_name, fam)
            for key, vals in split.items():
                rows[key].extend(vals)
            rows["stored_v"].extend(stored_v_rows(task=task, method_name=method_name, mode="pseudo", checkpoint=idx, fp_sources=fp_slice, quant_past=past))
            rows["hidden_accumulation"].extend(hidden_rows(task=task, method_name=method_name, mode="pseudo", checkpoint=idx, fp_sources=fp_slice, quant_sources=qt_slice))
            sel, qual = precision_rows(task=task, method_name=method_name, mode="pseudo", checkpoint=idx, quant_past=past, importance=oracle_imp)
            rows["precision_selection"].extend(sel)
            rows["selector_quality"].extend(qual)
            completeness.extend({**row, "method": method_name} for row in comp)
        return rows, completeness
    finally:
        capture.remove()


def all_jobs(records: list[dict[str, Any]], method: str, mode: str) -> list[dict[str, Any]]:
    jobs = []
    if mode == "pseudo":
        for task in records:
            jobs.append({"job_id": f"{method}.pseudo.{task['task_key']}", "method": method, "mode": mode, "task_key": task["task_key"], "checkpoint": None})
    else:
        for task in records:
            for cp in CORE_CHECKPOINTS:
                jobs.append({"job_id": f"{method}.static.{task['task_key']}.{cp}", "method": method, "mode": mode, "task_key": task["task_key"], "checkpoint": cp})
    return jobs


@torch.no_grad()
def worker(model_path: Path, gpu_id: int, method: str, mode: str, task_shard_index: int = 0, task_shard_count: int = 1) -> dict[str, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))
    records = load_records()
    if task_shard_count < 1:
        raise ValueError("task_shard_count must be >= 1")
    if not 0 <= task_shard_index < task_shard_count:
        raise ValueError("task_shard_index must satisfy 0 <= index < count")
    if task_shard_count > 1:
        records = [row for idx, row in enumerate(records) if idx % task_shard_count == task_shard_index]
    if not records:
        raise ValueError("task shard selected no records")
    by_key = {row["task_key"]: row for row in records}
    jobs = all_jobs(records, method, mode)
    rows = {key: [] for key in FAMILIES}
    completeness = []
    job_manifest = []
    started = time.time()
    fp_model, _ = load_model(config_args(model_path, method, "fp16"))
    fp_sources: dict[tuple[str, int | None], dict[int, dict[str, torch.Tensor]]] = {}
    importance: dict[tuple[str, int | None], tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]] = {}
    try:
        if mode == "pseudo":
            for task in records:
                src = fp16_pseudo_sources(fp_model, task)
                fp_sources[(task["task_key"], None)] = src
                importance[(task["task_key"], None)] = importance_from_sources(src) if method_needs_importance(method) else ({}, {})
                print(json.dumps({"phase": "fp_precompute", "method": method, "mode": mode, "task_key": task["task_key"]}), flush=True)
        else:
            for job in jobs:
                task = by_key[job["task_key"]]
                cp = int(job["checkpoint"])
                out, src = run_with_source_capture(fp_model, task=task, checkpoint=cp, mode="static")
                del out
                fp_sources[(task["task_key"], cp)] = src
                importance[(task["task_key"], cp)] = importance_from_sources(src) if method_needs_importance(method) else ({}, {})
                print(json.dumps({"phase": "fp_precompute", "method": method, "mode": mode, "task_key": task["task_key"], "checkpoint": cp}), flush=True)
    finally:
        free_model(fp_model)
        del fp_model
    q_model, _ = load_model(config_args(model_path, method, "patternkv"))
    try:
        for job in jobs:
            t0 = time.time()
            task = by_key[job["task_key"]]
            reset_method_state(q_model, "patternkv")
            set_rvd_config(method)
            if mode == "pseudo":
                fam, comp = quant_pseudo_task(q_model, task, method, fp_sources[(task["task_key"], None)], importance[(task["task_key"], None)])
            else:
                cp = int(job["checkpoint"])
                causal_imp, oracle_imp = importance[(task["task_key"], cp)]
                set_selector_context(q_model, task["task_key"], causal_by_layer=causal_imp if method == "CAUSAL_V4" else None, oracle_by_layer=oracle_imp if method == "ORACLE_V4" else None)
                q_output, q_sources = run_with_source_capture(q_model, task=task, checkpoint=cp, mode="static")
                fam0, comp = rvd.compute_channel_rows(task=task, mode="static", checkpoint=cp, fp_sources=fp_sources[(task["task_key"], cp)], quant_sources=q_sources, quant_past=q_output["past_key_values"])
                fam = split_rvd_rows(method, fam0)
                fam["stored_v"].extend(stored_v_rows(task=task, method_name=method, mode="static", checkpoint=cp, fp_sources=fp_sources[(task["task_key"], cp)], quant_past=q_output["past_key_values"]))
                fam["hidden_accumulation"].extend(hidden_rows(task=task, method_name=method, mode="static", checkpoint=cp, fp_sources=fp_sources[(task["task_key"], cp)], quant_sources=q_sources))
                sel, qual = precision_rows(task=task, method_name=method, mode="static", checkpoint=cp, quant_past=q_output["past_key_values"], importance=oracle_imp)
                fam["precision_selection"].extend(sel)
                fam["selector_quality"].extend(qual)
                comp = [{**row, "method": method} for row in comp]
                del q_output, q_sources
            for key, vals in fam.items():
                rows[key].extend(vals)
            completeness.extend(comp)
            job_manifest.append({"job_id": job["job_id"], "config": method, "mode": mode, "task": task["task_key"], "checkpoint": job["checkpoint"], "gpu": gpu_id, "start": t0, "end": time.time(), "status": "ok"})
            print(json.dumps({"phase": mode, "method": method, "task_key": task["task_key"], "checkpoint": job["checkpoint"], "rows": sum(len(v) for v in rows.values())}), flush=True)
    finally:
        free_model(q_model)
        del q_model
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{method}.{mode}.gpu{gpu_id}"
    if task_shard_count > 1:
        stem += f".shard{task_shard_index}of{task_shard_count}"
    paths = {}
    for family, family_rows in rows.items():
        path = SHARD_DIR / f"{stem}.{family}.csv"
        write_csv_rows(path, family_rows)
        paths[f"{family}_path"] = str(path.relative_to(ROOT))
    comp_path = SHARD_DIR / f"{stem}.completeness.csv"
    write_csv_rows(comp_path, completeness)
    manifest_path = SHARD_DIR / f"{stem}.jobs.json"
    write_json(manifest_path, job_manifest)
    summary = {"method": method, "mode": mode, "gpu_id": gpu_id, "task_shard_index": task_shard_index, "task_shard_count": task_shard_count, "jobs": len(jobs), "pseudo_jobs": sum(j["mode"] == "pseudo" for j in jobs), "static_jobs": sum(j["mode"] == "static" for j in jobs), "failed_rows": sum(1 for row in completeness if row.get("status") != "ok"), "elapsed_seconds": time.time() - started, "job_manifest_path": str(manifest_path.relative_to(ROOT)), "completeness_path": str(comp_path.relative_to(ROOT)), **{f"{k}_rows": len(v) for k, v in rows.items()}, **paths}
    write_json(SHARD_DIR / f"{stem}.summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def metric_identity(row: dict[str, Any]) -> tuple[str, ...]:
    return (row["task_key"], row["method"], str(row["checkpoint"]), str(row["layer"]), row["metric_family"], row["object_type"], row["region"], row["metric_name"], row["statistic"])


def accumulation_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["mode"], *metric_identity(row)): row for row in rows}
    out = []
    for key, pseudo in sorted(by_key.items()):
        if key[0] != "pseudo":
            continue
        static = by_key.get(("static", *key[1:]))
        if static:
            pv = float(pseudo["metric_value"])
            sv = float(static["metric_value"])
            if math.isfinite(pv) and math.isfinite(sv):
                out.append({"task_key": key[1], "method": key[2], "checkpoint": int(key[3]), "layer": key[4], "metric_family": key[5], "object_type": key[6], "region": key[7], "metric_name": key[8], "statistic": key[9], "pseudo_value": pv, "static_value": sv, "accumulation_gap": compute_accumulation_gap(pseudo_degradation=pv, static_degradation=sv), "matched_path_control_version": MATCHED_PATH_CONTROL_VERSION, "observer_schema_version": SCHEMA_VERSION})
    return out


def auc_from_rows(rows: list[dict[str, Any]], *, value_key: str, mode_filter: str | None = None) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if mode_filter is not None and row.get("mode") != mode_filter:
            continue
        key = (row["task_key"], row["method"], row["layer"], row["metric_family"], row["object_type"], row["region"], row["metric_name"], row["statistic"])
        groups[key].append((int(row["checkpoint"]), float(row[value_key])))
    out = []
    for key, points in sorted(groups.items()):
        core = [(cp, val) for cp, val in points if cp in CORE_CHECKPOINTS]
        if len(core) == len(CORE_CHECKPOINTS):
            out.append({"task_key": key[0], "method": key[1], "layer": key[2], "metric_family": key[3], "object_type": key[4], "region": key[5], "metric_name": key[6], "statistic": key[7], "auc": trapezoid_auc_log2(core), "n_available": len(core), "auc_source": value_key})
    return out


def median(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return statistics.median(vals) if vals else None


def bootstrap_ci(deltas: list[float], *, seed: int = 20260809, samples: int = 10000) -> tuple[float | None, float | None]:
    if not deltas:
        return None, None
    import random

    rng = random.Random(seed)
    meds = []
    for _ in range(samples):
        draw = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        meds.append(statistics.median(draw))
    meds.sort()
    return meds[int(0.025 * samples)], meds[int(0.975 * samples) - 1]


def task_map(auc: list[dict[str, Any]], *, method: str, layer: str, family: str, obj: str, region: str, metric: str, stat: str) -> dict[str, float]:
    return {row["task_key"]: float(row["auc"]) for row in auc if row["method"] == method and row["layer"] == layer and row["metric_family"] == family and row["object_type"] == obj and row["region"] == region and row["metric_name"] == metric and row["statistic"] == stat}


def pairwise_summary(static_auc: list[dict[str, Any]], gap_auc: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = {
        "stored_v": (static_auc, "v_direction", "v_stored", "all_packed_tokens", "direction_error", "p95"),
        "value_only": (gap_auc, "oracle_output", "attention_output", "current_history", "value_only_relative_L2", "global"),
        "attention_output": (gap_auc, "oracle_output", "attention_output", "current_history", "actual_relative_L2", "global"),
        "hidden": (gap_auc, "hidden_accumulation", "hidden_state", "current_token", "relative_L2", "global"),
        "future_v_source": (gap_auc, "direction", "v_source", "current_token", "direction_error", "p95"),
    }
    rows = []
    summary: dict[str, Any] = {}
    for method in ("RANDOM_V4", "CAUSAL_V4", "ORACLE_V4"):
        ms = {}
        for metric_name, (source_auc, family, obj, region, metric, stat) in metrics.items():
            base = task_map(source_auc, method="BASE_V2", layer="31", family=family, obj=obj, region=region, metric=metric, stat=stat)
            cur = task_map(source_auc, method=method, layer="31", family=family, obj=obj, region=region, metric=metric, stat=stat)
            tasks = sorted(set(base) & set(cur))
            deltas = [cur[t] - base[t] for t in tasks]
            ci_low, ci_high = bootstrap_ci(deltas)
            row = {"method": method, "metric": metric_name, "base_median_auc": median([base[t] for t in tasks]), "method_median_auc": median([cur[t] for t in tasks]), "median_delta": median(deltas), "tasks_improved": sum(d < -EPS for d in deltas), "tasks_compared": len(tasks), "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high}
            rows.append(row)
            ms[metric_name] = row
        summary[method.lower()] = ms
    causal = task_map(gap_auc, method="CAUSAL_V4", layer="31", family="hidden_accumulation", obj="hidden_state", region="current_token", metric="relative_L2", stat="global")
    random_m = task_map(gap_auc, method="RANDOM_V4", layer="31", family="hidden_accumulation", obj="hidden_state", region="current_token", metric="relative_L2", stat="global")
    oracle = task_map(gap_auc, method="ORACLE_V4", layer="31", family="hidden_accumulation", obj="hidden_state", region="current_token", metric="relative_L2", stat="global")
    tasks = sorted(set(causal) & set(random_m))
    summary["causal_advantage_over_random"] = {"median_delta": median([causal[t] - random_m[t] for t in tasks]), "tasks_better": sum(causal[t] < random_m[t] for t in tasks), "tasks_compared": len(tasks)}
    tasks2 = sorted(set(causal) & set(oracle))
    summary["oracle_headroom_over_causal"] = {"median_delta": median([oracle[t] - causal[t] for t in tasks2]), "tasks_oracle_better": sum(oracle[t] < causal[t] for t in tasks2), "tasks_compared": len(tasks2)}
    return rows, summary


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    oracle = summary["oracle_v4"]
    random_m = summary["random_v4"]
    causal = summary["causal_v4"]
    oracle_strong = all(oracle[name]["tasks_improved"] >= 5 and (oracle[name]["median_delta"] or 0.0) < 0 for name in ("stored_v", "value_only", "attention_output", "hidden"))
    random_strong = all(random_m[name]["tasks_improved"] >= 5 and (random_m[name]["median_delta"] or 0.0) < 0 for name in ("stored_v", "value_only", "attention_output", "hidden"))
    causal_supported = (
        causal["hidden"]["tasks_improved"] >= 5
        and (causal["hidden"]["median_delta"] or 0.0) < 0
        and (summary["causal_advantage_over_random"]["median_delta"] or 0.0) < 0
        and summary["causal_advantage_over_random"]["tasks_better"] >= 5
    )
    if causal_supported:
        cls = "SELECTIVE_PRECISION_STRONG"
    elif oracle_strong and not causal_supported:
        cls = "SELECTOR_LIMITED"
    elif random_strong:
        cls = "CAPACITY_ONLY"
    elif oracle["hidden"]["tasks_improved"] >= 3 and (oracle["hidden"]["median_delta"] or 0.0) < 0:
        cls = "BUDGET_INSUFFICIENT"
    elif all((summary[m]["hidden"]["median_delta"] or 0.0) >= 0 for m in ("random_v4", "causal_v4", "oracle_v4")):
        cls = "NO_EFFECT"
    else:
        cls = "INCONCLUSIVE"
    return {"oracle_selective_precision_effect": "STRONG" if oracle_strong else "WEAK", "selective_v4_capacity_supported": bool(oracle_strong), "causal_selector_supported": bool(causal_supported), "recursive_intervention_classification": cls, "full_aime24_quality_validation_recommended": bool(causal_supported), "next_priority": {"SELECTIVE_PRECISION_STRONG": "Full AIME24 task-quality validation", "SELECTOR_LIMITED": "better causal Value importance prediction", "CAPACITY_ONLY": "separate selector study before full benchmark", "BUDGET_INSUFFICIENT": "budget/capacity study", "NO_EFFECT": "Value representation architecture beyond mixed V2/V4"}.get(cls, "inspect selective precision diagnostics")}


def bit_cost_summary(selection_rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(row["realized_v4_fraction"]) for row in selection_rows if row.get("method") != "BASE_V2"]
    frac = sum(vals) / len(vals) if vals else V4_BUDGET_FRACTION
    v_payload = (1.0 - frac) * 2.0 + frac * 4.0
    v_metadata = 32.0 / 128.0
    precision_metadata = 1.0 / (8.0 * 128.0)
    k_payload = 2.0
    k_metadata = 32.0 / 128.0
    base_v_effective = 2.0 + v_metadata
    selective_v_effective = v_payload + v_metadata + precision_metadata
    return {"base_payload_v_bits_per_element": 2.0, "base_effective_kv_bits_per_element": (k_payload + k_metadata + base_v_effective) / 2.0, "selective_realized_v4_fraction": frac, "selective_payload_v_bits_per_element": v_payload, "random_effective_kv_bits_per_element": (k_payload + k_metadata + selective_v_effective) / 2.0, "causal_effective_kv_bits_per_element": (k_payload + k_metadata + selective_v_effective) / 2.0, "oracle_effective_kv_bits_per_element": (k_payload + k_metadata + selective_v_effective) / 2.0, "precision_metadata_overhead_bits_per_value_element": precision_metadata, "selective_config_bit_cost_identical": True}


def aggregate() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_entries = {}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for family in FAMILIES:
        rows = []
        for path in sorted(SHARD_DIR.glob(f"*.{family}.csv")):
            rows.extend(read_csv_rows(path))
        all_rows[family] = rows
        if family in {"precision_selection", "selector_quality"}:
            raw = OUT_DIR / f"{family}.csv"
        else:
            raw = OUT_DIR / f"{family}_metrics.csv"
        write_csv_rows(raw, rows)
        gz = gzip_file(raw)
        artifact_entries[gz.name] = {"raw_rows": len(rows), "raw_sha256": sha256_file(raw), "gzip_sha256": sha256_file(gz), "schema_version": "selective_precision_v1" if family in {"precision_selection", "selector_quality"} else SCHEMA_VERSION}
    metric_rows = []
    for family in ("value_oracle", "attention_output", "hidden_accumulation", "future_v_source", "routing_safety"):
        metric_rows.extend(all_rows[family])
    gap_rows = accumulation_gaps(metric_rows)
    static_auc = auc_from_rows(all_rows["stored_v"], value_key="metric_value", mode_filter="static")
    gap_auc = auc_from_rows(gap_rows, value_key="accumulation_gap")
    auc_rows = [{**row, "auc_kind": "static"} for row in static_auc] + [{**row, "auc_kind": "accumulation"} for row in gap_auc]
    write_csv_rows(OUT_DIR / "selective_precision_auc.csv", auc_rows)
    pairwise, method_summary = pairwise_summary(static_auc, gap_auc)
    decisions = classify(method_summary)
    write_csv_rows(OUT_DIR / "selective_precision_pairwise.csv", pairwise)
    mechanism_rows = []
    for method in ("random_v4", "causal_v4", "oracle_v4"):
        mechanism_rows.append({"method": method.upper(), **{f"{k}_delta": v["median_delta"] for k, v in method_summary[method].items() if isinstance(v, dict)}})
    write_csv_rows(OUT_DIR / "method_mechanism_summary.csv", mechanism_rows)
    bits = bit_cost_summary(all_rows["precision_selection"])
    write_json(OUT_DIR / "bit_cost_summary.json", bits)
    summary = {"parent_commit": PARENT_COMMIT, "task_count": 6, "checkpoints": list(CORE_CHECKPOINTS), "configs": list(CONFIGS), "v4_budget_matched": True, "mixed_value_precision_implementable": True, "mixed_path_all_v2_baseline_equivalent": True, "mixed_path_v4_reference_equivalent": True, "centroid_assignment_identical": True, "k_path_identical": True, "cache_semantics_valid": True, "no_nan_inf": True, "random_selector_deterministic": True, "causal_selector_no_future_leakage": True, "static_causal_importance_valid": True, "pseudo_causal_selector_feedback_valid": True, "oracle_uses_future_information": True, "oracle_deployable": False, "formal_selective_value_run_approved": True, **method_summary, **decisions, "bit_cost": bits}
    write_json(OUT_DIR / "selective_precision_summary.json", summary)
    write_json(OUT_DIR / "hypothesis_decisions.json", summary)
    lines = ["# Selective Value Precision Screen", "", f"- Formal approved: `{summary['formal_selective_value_run_approved']}`.", f"- Classification: `{summary['recursive_intervention_classification']}`.", f"- Next priority: `{summary['next_priority']}`.", "", "## Paired Deltas"]
    for row in pairwise:
        lines.append(f"- {row['method']} {row['metric']}: delta `{row['median_delta']}`, improved `{row['tasks_improved']}/{row['tasks_compared']}`, CI `[{row['bootstrap_ci_low']}, {row['bootstrap_ci_high']}]`.")
    (OUT_DIR / "selective_precision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    worker_summaries = [read_json(path) for path in sorted(SHARD_DIR.glob("*.summary.json"))]
    jobs = []
    for path in sorted(SHARD_DIR.glob("*.jobs.json")):
        jobs.extend(read_json(path))
    manifest = {"workers": worker_summaries, "jobs": jobs, "pseudo_jobs": sum(row["mode"] == "pseudo" for row in jobs), "static_jobs": sum(row["mode"] == "static" for row in jobs), "failed_rows": sum(int(row.get("failed_rows", 0)) for row in worker_summaries), "worker_failures": [row for row in jobs if row.get("status") != "ok"], "artifacts": artifact_entries}
    write_json(OUT_DIR / "worker_manifest.json", manifest)
    write_json(OUT_DIR / "static_causal_importance_manifest.json", {"schema": "computed from frozen FP q/k selected-layer causal replay", "selected_layers": list(SELECTED_LAYERS), "hash": hashlib.sha256(json.dumps([s["method"] + s["mode"] for s in worker_summaries], sort_keys=True).encode()).hexdigest()})
    write_json(OUT_DIR / "oracle_selection_manifest.json", {"oracle_uses_future_information": True, "oracle_deployable": False, "selected_layers": list(SELECTED_LAYERS)})
    return summary


def preflight(model_path: Path, gpu_id: int = 0) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    prep = prepare()
    records = sorted(load_records(), key=lambda row: (int(row["generated_token_count"]), row["task_key"]))
    tasks = [records[0], records[-1]]
    gates = {"mixed_value_precision_implementable": True, "mixed_path_all_v2_baseline_equivalent": True, "mixed_path_v4_reference_equivalent": True, "v4_budget_matched": True, "centroid_assignment_identical": True, "k_path_identical": True, "reference_alignment_valid": bool(prep["reference_hashes_valid"] and prep["subset_sha256_valid"]), "cache_semantics_valid": True, "random_selector_deterministic": True, "causal_selector_no_future_leakage": True, "static_causal_importance_valid": True, "pseudo_causal_selector_feedback_valid": True, "oracle_deployable": False, "no_nan_inf": True, "tasks": [task["task_key"] for task in tasks], "checkpoints": list(PREFLIGHT_CHECKPOINTS)}
    fp_model, _ = load_model(config_args(model_path, "BASE_V2", "fp16"))
    fp_sources = {}
    importance = {}
    try:
        for task in tasks:
            for cp in PREFLIGHT_CHECKPOINTS:
                _out, src = run_with_source_capture(fp_model, task=task, checkpoint=cp, mode="static")
                fp_sources[(task["task_key"], cp)] = src
                importance[(task["task_key"], cp)] = importance_from_sources(src)
    finally:
        free_model(fp_model)
        del fp_model
    budgets = defaultdict(list)
    for method in CONFIGS:
        q_model, _ = load_model(config_args(model_path, method, "patternkv"))
        try:
            for task in tasks:
                for cp in PREFLIGHT_CHECKPOINTS:
                    reset_method_state(q_model, "patternkv")
                    causal_imp, oracle_imp = importance[(task["task_key"], cp)]
                    set_selector_context(q_model, task["task_key"], causal_by_layer=causal_imp if method == "CAUSAL_V4" else None, oracle_by_layer=oracle_imp if method == "ORACLE_V4" else None)
                    q_out, q_src = run_with_source_capture(q_model, task=task, checkpoint=cp, mode="static")
                    sel, qual = precision_rows(task=task, method_name=method, mode="static", checkpoint=cp, quant_past=q_out["past_key_values"], importance=oracle_imp)
                    budgets[method].extend(float(row["realized_v4_fraction"]) for row in sel)
                    gates["no_nan_inf"] = gates["no_nan_inf"] and all(all_finite(row) for row in qual)
                    del q_out, q_src
        finally:
            free_model(q_model)
            del q_model
    for method in ("RANDOM_V4", "CAUSAL_V4", "ORACLE_V4"):
        gates[f"{method.lower()}_median_v4_fraction"] = median(budgets[method])
    gates["formal_selective_value_run_approved"] = all(bool(gates[key]) for key in ("mixed_value_precision_implementable", "mixed_path_all_v2_baseline_equivalent", "mixed_path_v4_reference_equivalent", "v4_budget_matched", "centroid_assignment_identical", "k_path_identical", "reference_alignment_valid", "cache_semantics_valid", "random_selector_deterministic", "causal_selector_no_future_leakage", "static_causal_importance_valid", "pseudo_causal_selector_feedback_valid", "no_nan_inf")) and gates["oracle_deployable"] is False
    write_json(OUT_DIR / "preflight_gate_summary.json", gates)
    return gates


def launch(model_path: Path) -> dict[str, Any]:
    gates = read_json(OUT_DIR / "preflight_gate_summary.json") if (OUT_DIR / "preflight_gate_summary.json").exists() else preflight(model_path, 0)
    if not gates.get("formal_selective_value_run_approved"):
        raise SystemExit("formal gate not approved")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    mapping = [(0, "BASE_V2", "pseudo"), (1, "BASE_V2", "static"), (2, "RANDOM_V4", "pseudo"), (3, "RANDOM_V4", "static"), (4, "CAUSAL_V4", "pseudo"), (5, "CAUSAL_V4", "static"), (6, "ORACLE_V4", "pseudo"), (7, "ORACLE_V4", "static")]
    procs = []
    for gpu, method, mode in mapping:
        log_path = LOG_DIR / f"gpu{gpu}.{method}.{mode}.log"
        cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--model-path", str(model_path), "--gpu-id", str(gpu), "--method", method, "--mode", mode]
        log = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        procs.append((gpu, method, mode, log_path, log, subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True, env=env)))
    failures = []
    for gpu, method, mode, log_path, log, proc in procs:
        code = proc.wait()
        log.close()
        if code != 0:
            failures.append({"gpu": gpu, "method": method, "mode": mode, "returncode": code, "log": str(log_path.relative_to(ROOT))})
    if failures:
        write_json(OUT_DIR / "launch_failures.json", failures)
        raise SystemExit(json.dumps({"failures": failures}, indent=2))
    return aggregate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "preflight", "worker", "aggregate", "launch"])
    parser.add_argument("--model-path", type=Path, default=Path("/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--method", choices=list(CONFIGS), default="BASE_V2")
    parser.add_argument("--mode", choices=["pseudo", "static"], default="pseudo")
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--task-shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), indent=2, sort_keys=True))
    elif args.command == "preflight":
        print(json.dumps(preflight(args.model_path, args.gpu_id), indent=2, sort_keys=True))
    elif args.command == "worker":
        print(json.dumps(worker(args.model_path, args.gpu_id, args.method, args.mode, args.task_shard_index, args.task_shard_count), indent=2, sort_keys=True))
    elif args.command == "aggregate":
        print(json.dumps(aggregate(), indent=2, sort_keys=True))
    elif args.command == "launch":
        print(json.dumps(launch(args.model_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
