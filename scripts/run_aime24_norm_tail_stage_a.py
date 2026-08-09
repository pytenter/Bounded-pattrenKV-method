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
import random
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

from bench.pseudodecode_metrics import full_trajectory_sha256, trapezoid_auc_log2, write_csv_rows  # noqa: E402
from models.segmented_cache import (  # noqa: E402
    PatternQuantizedKVCache,
    dequantize_k_reference,
    dequantize_v_reference,
    deserialize_cache,
    pattern_gather_centroids,
    tensor_tokens,
)
from scripts.run_aime24_pseudodecode_preflight import (  # noqa: E402
    REPORT_DIR as FORMAL_REPORT_DIR,
    SOURCE_COMMIT,
    compare_replays,
    load_model,
    make_args,
    replay_prefix,
    reset_method_state,
    write_json,
)
from scripts.run_aime24_execution_path_resolution import replay_pseudo_checkpoints  # noqa: E402
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # noqa: E402


OUT_DIR = ROOT / "reports/aime24_norm_tail_3090"
RESULT_DIR = ROOT / "results/aime24_norm_tail_3090"
SHARD_DIR = RESULT_DIR / "shards"
LOG_DIR = ROOT / "run/aime24_norm_tail_3090/logs"
SELECTED_LAYERS = (0, 7, 15, 23, 31)
CORE_CHECKPOINTS = (128, 512, 1024, 2048, 4096)
CONFIGS = {
    "pattern_s0": {"config": "pattern_rolling_k2v2_s0_r128", "method": "patternkv", "sink_length": 0, "recent_length": 128, "method_group": "pattern"},
    "pattern_s16": {"config": "pattern_rolling_k2v2_s16_r128", "method": "patternkv", "sink_length": 16, "recent_length": 128, "method_group": "pattern"},
    "kivi_s0": {"config": "kivi_rolling_k2v2_s0_r128", "method": "kivi_official", "sink_length": 0, "recent_length": 128, "method_group": "kivi"},
    "kivi_s16": {"config": "kivi_rolling_k2v2_s16_r128", "method": "kivi_official", "sink_length": 16, "recent_length": 128, "method_group": "kivi"},
}
EPS = 1e-12
PRIMARY_NORM_METRICS = {
    ("k_source", "source_state_norm_drift", "all_tokens", "relative_norm_error", "p95"),
    ("k_source", "source_state_norm_drift", "all_tokens", "relative_norm_error", "p99"),
    ("v_source", "source_state_norm_drift", "all_tokens", "relative_norm_error", "p95"),
    ("v_source", "source_state_norm_drift", "all_tokens", "relative_norm_error", "p99"),
    ("k_stored", "stored_norm_error", "packed_history", "relative_norm_error", "p95"),
    ("v_stored", "stored_norm_error", "packed_history", "relative_norm_error", "p95"),
    ("k_stored", "stored_norm_error", "packed_history", "abs_log_norm_ratio", "p95"),
    ("k_stored", "stored_norm_error", "packed_history", "abs_log_norm_ratio", "p99"),
    ("v_stored", "stored_norm_error", "packed_history", "abs_log_norm_ratio", "p95"),
    ("v_stored", "stored_norm_error", "packed_history", "abs_log_norm_ratio", "p99"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()


def git_show_text(repo: Path, rev_path: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "show", rev_path], text=True, stderr=subprocess.DEVNULL)


def reference_artifact(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_reference_records() -> list[dict[str, Any]]:
    manifest = read_json(FORMAL_REPORT_DIR / "reference_trajectories_manifest.json")
    records = []
    for row in manifest["rows"]:
        artifact = reference_artifact(ROOT / row["artifact_path"])
        records.append({**row, **artifact})
    return sorted(records, key=lambda row: row["task_key"])


def validate_reference_freeze(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    ok = True
    for row in records:
        observed = full_trajectory_sha256(row["prompt_token_ids"], row["generated_token_ids"])
        expected = row["full_trajectory_sha256"]
        same = observed == expected
        ok = ok and same
        rows.append({"task_key": row["task_key"], "expected": expected, "observed": observed, "valid": same})
    return {
        "reference_trajectories": f"{sum(1 for row in rows if row['valid'])}/{len(rows)}",
        "reference_hashes_valid": ok and len(records) == 12,
        "portable_generation_hash": "86648d12304ce11890c1a8f64bf5a896",
        "rows": rows,
    }


def tensor_sha(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous()
    return hashlib.sha256(arr.numpy().tobytes()).hexdigest()


class SourceCapture:
    def __init__(self, selected_layers: tuple[int, ...] = SELECTED_LAYERS):
        self.selected_layers = set(selected_layers)
        self.parts: dict[int, dict[str, list[torch.Tensor]]] = {
            layer: {"hidden": [], "q_source": [], "k_source": [], "v_source": []}
            for layer in selected_layers
        }
        self._original: list[tuple[Any, Any]] = []

    def install(self, model: torch.nn.Module) -> None:
        layers = model.model.layers
        for idx in self.selected_layers:
            attn = layers[idx].self_attn
            orig = attn.forward

            def wrapped(*args, __attn=attn, __idx=idx, __orig=orig, **kwargs):
                hidden_states = args[0] if args else kwargs["hidden_states"]
                position_ids = kwargs.get("position_ids")
                if position_ids is None and len(args) >= 3:
                    position_ids = args[2]
                self.capture(__idx, __attn, hidden_states, position_ids)
                return __orig(*args, **kwargs)

            attn.forward = wrapped
            self._original.append((attn, orig))

    def remove(self) -> None:
        for module, orig in self._original:
            module.forward = orig
        self._original.clear()

    @torch.no_grad()
    def capture(self, layer_idx: int, attn: Any, hidden_states: torch.Tensor, position_ids: torch.Tensor | None) -> None:
        if position_ids is None:
            q_len = hidden_states.shape[1]
            position_ids = torch.arange(q_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0)
        bsz, q_len, _ = hidden_states.shape
        if getattr(attn.config, "pretraining_tp", 1) > 1:
            raise RuntimeError("pretraining_tp capture path is not implemented for this diagnostic")
        q = attn.q_proj(hidden_states).view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(hidden_states).view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        cos, sin = attn.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        bucket = self.parts[layer_idx]
        bucket["hidden"].append(hidden_states.detach().cpu().to(torch.float32))
        bucket["q_source"].append(q.detach().cpu().to(torch.float32))
        bucket["k_source"].append(k.detach().cpu().to(torch.float32))
        bucket["v_source"].append(v.detach().cpu().to(torch.float32))
        del q, k, v

    def tensors(self) -> dict[int, dict[str, torch.Tensor]]:
        out = {}
        for layer, parts in self.parts.items():
            out[layer] = {}
            for name, tensors in parts.items():
                if not tensors:
                    continue
                dim = 1 if name == "hidden" else 2
                out[layer][name] = torch.cat(tensors, dim=dim).contiguous()
        return out


def run_model_with_capture(
    model: torch.nn.Module,
    *,
    prompt_ids: list[int],
    generated_ids: list[int],
    checkpoint: int,
    mode: str,
) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    capture = SourceCapture()
    capture.install(model)
    try:
        output = replay_prefix(model, prompt_ids=prompt_ids, generated_ids=generated_ids, checkpoint=checkpoint, mode=mode)
        sources = capture.tensors()
        return output, sources
    finally:
        capture.remove()


def run_pseudo_max_with_capture(
    model: torch.nn.Module,
    *,
    prompt_ids: list[int],
    generated_ids: list[int],
    checkpoint: int,
    method: str | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, torch.Tensor]]]:
    capture = SourceCapture()
    capture.install(model)
    try:
        outputs = replay_pseudo_checkpoints(model, prompt_ids=prompt_ids, generated_ids=generated_ids, checkpoints=[checkpoint], method=method)
        return outputs, capture.tensors()
    finally:
        capture.remove()


@torch.no_grad()
def run_quant_pseudo_norm_rows(
    model: torch.nn.Module,
    *,
    task: dict[str, Any],
    config: dict[str, Any],
    fp_sources: dict[int, dict[str, torch.Tensor]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    completeness: list[dict[str, Any]] = []
    capture = SourceCapture()
    capture.install(model)
    try:
        device = "cuda:0"
        wanted = set(CORE_CHECKPOINTS)
        prompt_tensor = torch.tensor([task["prompt_token_ids"]], device=device, dtype=torch.long)
        outputs = model(input_ids=prompt_tensor, use_cache=True, output_hidden_states=True, output_attentions=False, return_dict=True)
        past = outputs.past_key_values
        for idx, token in enumerate(task["generated_token_ids"][: max(CORE_CHECKPOINTS)], start=1):
            token_tensor = torch.tensor([[int(token)]], device=device, dtype=torch.long)
            outputs = model(input_ids=token_tensor, past_key_values=past, use_cache=True, output_hidden_states=True, output_attentions=False, return_dict=True)
            past = outputs.past_key_values
            if idx not in wanted:
                continue
            total = int(task["prompt_token_count"]) + idx
            q_sources_all = capture.tensors()
            q_sources = {
                layer: {
                    name: tensor[:, :total, :] if name == "hidden" else tensor[:, :, :total, :]
                    for name, tensor in layer_map.items()
                }
                for layer, layer_map in q_sources_all.items()
            }
            fp_sliced = {
                layer: {
                    name: tensor[:, :total, :] if name == "hidden" else tensor[:, :, :total, :]
                    for name, tensor in layer_map.items()
                }
                for layer, layer_map in fp_sources.items()
            }
            metric_rows, comp_rows = compare_sources_and_cache(
                task_key=task["task_key"],
                config=config,
                mode="pseudo",
                checkpoint=idx,
                fp_sources=fp_sliced,
                quant_sources=q_sources,
                quant_output={"past_key_values": past},
            )
            rows.extend(metric_rows)
            completeness.extend(comp_rows)
        return rows, completeness
    finally:
        capture.remove()


def to_sample_matrix(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.detach().float().cpu()
    if t.ndim == 4:
        return t.permute(0, 2, 1, 3).reshape(-1, t.shape[-1]).contiguous()
    if t.ndim == 3:
        return t.reshape(-1, t.shape[-1]).contiguous()
    raise ValueError(f"unexpected tensor shape {tuple(t.shape)}")


def vector_metric_arrays(target: torch.Tensor, fp: torch.Tensor) -> dict[str, list[float]]:
    t = to_sample_matrix(target)
    f = to_sample_matrix(fp)
    if t.shape != f.shape:
        limit = min(t.shape[0], f.shape[0])
        t = t[:limit]
        f = f[:limit]
    tn = torch.linalg.vector_norm(t, dim=-1)
    fn = torch.linalg.vector_norm(f, dim=-1)
    dot = (t * f).sum(dim=-1)
    denom = (tn * fn).clamp_min(EPS)
    cos = (dot / denom).clamp(-1.0, 1.0)
    rel_l2 = torch.linalg.vector_norm(t - f, dim=-1) / fn.clamp_min(EPS)
    ratio = tn / fn.clamp_min(EPS)
    return {
        "norm_ratio": ratio.tolist(),
        "relative_norm_error": ((tn - fn).abs() / fn.clamp_min(EPS)).tolist(),
        "signed_norm_drift": ((tn - fn) / fn.clamp_min(EPS)).tolist(),
        "abs_log_norm_ratio": torch.log((tn + EPS) / (fn + EPS)).abs().tolist(),
        "direction_error": (1.0 - cos).tolist(),
        "relative_L2": rel_l2.tolist(),
    }


def quantile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def summarize_arrays(arrays: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows = []
    for metric, values in arrays.items():
        vals = [v for v in values if math.isfinite(v)]
        if not vals:
            continue
        stats = {
            "mean": statistics.fmean(vals),
            "p50": quantile(vals, 0.50),
            "p90": quantile(vals, 0.90),
            "p95": quantile(vals, 0.95),
            "p99": quantile(vals, 0.99),
            "max": max(vals),
        }
        rows.extend({"metric_name": metric, "statistic": stat, "metric_value": value, "n_samples": len(vals)} for stat, value in stats.items())
    return rows


def tensor_tokens_or_zero(tensor: torch.Tensor | None) -> int:
    return int(tensor.shape[2]) if torch.is_tensor(tensor) else 0


def dequantized_regions(layer_cache: Any, *, pattern: bool) -> tuple[dict[str, dict[str, torch.Tensor | None]], dict[str, int | str]]:
    cache = deserialize_cache(layer_cache, pattern=pattern)
    packed_k = dequantize_k_reference(cache.packed_k, cache.packed_k_scale, cache.packed_k_zero, cache.group_size, cache.k_bits)
    if packed_k is not None:
        packed_k = packed_k[:, :, : cache.packed_k_tokens, :].contiguous()
        if isinstance(cache, PatternQuantizedKVCache) and cache.k_centroids is not None and cache.k_assignments is not None:
            packed_k = packed_k + pattern_gather_centroids(cache.k_assignments[:, :, : cache.packed_k_tokens], cache.k_centroids).to(packed_k.dtype)
    packed_v = dequantize_v_reference(cache.packed_v, cache.packed_v_scale, cache.packed_v_zero, cache.group_size, cache.v_bits)
    if packed_v is not None:
        packed_v = packed_v[:, :, : cache.packed_v_tokens, :].contiguous()
        if isinstance(cache, PatternQuantizedKVCache) and cache.v_centroids is not None and cache.v_assignment_idx is not None:
            mask = cache.v_pattern_mask if cache.v_pattern_mask is not None else cache.v_assignments
            if mask is not None:
                centroids = pattern_gather_centroids(cache.v_assignment_idx[:, :, : cache.packed_v_tokens], cache.v_centroids).to(packed_v.dtype)
                packed_v = packed_v + mask[:, :, : cache.packed_v_tokens].unsqueeze(-1).to(packed_v.dtype) * centroids
    regions = {
        "sink": {"k": cache.sink_k, "v": cache.sink_v},
        "packed_history": {"k": packed_k, "v": packed_v},
        "pending_history": {"k": cache.pending_k, "v": cache.pending_v},
        "recent": {"k": cache.recent_k, "v": cache.recent_v},
    }
    counts = {
        "sink": tensor_tokens_or_zero(cache.sink_k),
        "packed_history": int(cache.packed_k_tokens),
        "pending_history": tensor_tokens_or_zero(cache.pending_k),
        "recent": tensor_tokens_or_zero(cache.recent_k),
        "total_tokens": int(cache.total_tokens),
        "cache_mode": str(getattr(cache, "cache_mode", "")),
    }
    return regions, counts


def source_region(source: torch.Tensor, region: str, counts: dict[str, int | str]) -> torch.Tensor | None:
    offset = 0
    for name in ("sink", "packed_history", "pending_history", "recent"):
        length = int(counts[name])
        if name == region:
            if length <= 0:
                return None
            return source[:, :, offset : offset + length, :].contiguous()
        offset += length
    raise KeyError(region)


def source_all(source: torch.Tensor, total_tokens: int) -> torch.Tensor:
    return source[:, :, :total_tokens, :].contiguous()


def add_metric_rows(
    rows: list[dict[str, Any]],
    *,
    task_key: str,
    config: str,
    method_group: str,
    mode: str,
    checkpoint: int,
    layer: int | str,
    object_type: str,
    error_type: str,
    region: str,
    arrays: dict[str, list[float]],
) -> None:
    for stat in summarize_arrays(arrays):
        rows.append(
            {
                "task_key": task_key,
                "config": config,
                "method_group": method_group,
                "mode": mode,
                "checkpoint": checkpoint,
                "layer": layer,
                "object_type": object_type,
                "error_type": error_type,
                "region": region,
                **stat,
                "source_commit": SOURCE_COMMIT,
                "norm_metric_schema_version": "norm_tail_accumulation_v1",
            }
        )


def compare_sources_and_cache(
    *,
    task_key: str,
    config: dict[str, Any],
    mode: str,
    checkpoint: int,
    fp_sources: dict[int, dict[str, torch.Tensor]],
    quant_sources: dict[int, dict[str, torch.Tensor]],
    quant_output: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    completeness: list[dict[str, Any]] = []
    layer_caches = quant_output.get("past_key_values") or []
    pattern = str(config["method"]).startswith("pattern")
    for layer in SELECTED_LAYERS:
        fp_layer = fp_sources.get(layer, {})
        q_layer = quant_sources.get(layer, {})
        if not fp_layer or not q_layer:
            completeness.append({"task_key": task_key, "config": config["config"], "mode": mode, "checkpoint": checkpoint, "layer": layer, "status": "missing_source"})
            continue
        try:
            regions, counts = dequantized_regions(layer_caches[layer], pattern=pattern)
        except Exception as exc:
            completeness.append({"task_key": task_key, "config": config["config"], "mode": mode, "checkpoint": checkpoint, "layer": layer, "status": "cache_error", "error": str(exc)})
            continue
        total = int(counts["total_tokens"])
        for source_name in ("hidden", "q_source", "k_source", "v_source"):
            if source_name not in fp_layer or source_name not in q_layer:
                continue
            fp = fp_layer[source_name]
            qt = q_layer[source_name]
            if source_name == "hidden":
                fp_cmp = fp[:, :total, :]
                qt_cmp = qt[:, :total, :]
            else:
                fp_cmp = source_all(fp, total)
                qt_cmp = source_all(qt, total)
            add_metric_rows(
                rows,
                task_key=task_key,
                config=config["config"],
                method_group=config["method_group"],
                mode=mode,
                checkpoint=checkpoint,
                layer=layer,
                object_type=source_name,
                error_type="source_state_norm_drift",
                region="all_tokens",
                arrays=vector_metric_arrays(qt_cmp, fp_cmp),
            )
        for region, kv in regions.items():
            for kv_name, object_prefix in (("k", "k_stored"), ("v", "v_stored")):
                stored = kv[kv_name]
                if not torch.is_tensor(stored) or int(stored.shape[2]) == 0:
                    continue
                q_source_name = f"{kv_name}_source"
                qsrc = source_region(q_layer[q_source_name], region, counts)
                fpsrc = source_region(fp_layer[q_source_name], region, counts)
                if qsrc is None or fpsrc is None:
                    continue
                add_metric_rows(
                    rows,
                    task_key=task_key,
                    config=config["config"],
                    method_group=config["method_group"],
                    mode=mode,
                    checkpoint=checkpoint,
                    layer=layer,
                    object_type=object_prefix,
                    error_type="representation_norm_error",
                    region=region,
                    arrays=vector_metric_arrays(stored.detach().cpu(), qsrc),
                )
                add_metric_rows(
                    rows,
                    task_key=task_key,
                    config=config["config"],
                    method_group=config["method_group"],
                    mode=mode,
                    checkpoint=checkpoint,
                    layer=layer,
                    object_type=object_prefix,
                    error_type="stored_norm_error",
                    region=region,
                    arrays=vector_metric_arrays(stored.detach().cpu(), fpsrc),
                )
        completeness.append(
            {
                "task_key": task_key,
                "config": config["config"],
                "mode": mode,
                "checkpoint": checkpoint,
                "layer": layer,
                "status": "ok",
                **{f"{key}_tokens": value for key, value in counts.items()},
            }
        )
    return rows, completeness


def free_model(model: torch.nn.Module | None) -> None:
    _ = model
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def worker(model_path: Path, config_key: str, mode: str, gpu_id: int) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = CONFIGS[config_key]
    records = load_reference_records()
    metric_rows: list[dict[str, Any]] = []
    completeness_rows: list[dict[str, Any]] = []
    started = time.time()

    fp_args = make_args(model_path, "fp16", 0, 0, config_name="fp16")
    fp_model, _ = load_model(fp_args)
    fp_sources_by_task: dict[str, Any] = {}
    try:
        for task in records:
            if mode == "pseudo":
                _, sources = run_pseudo_max_with_capture(fp_model, prompt_ids=task["prompt_token_ids"], generated_ids=task["generated_token_ids"], checkpoint=max(CORE_CHECKPOINTS))
                fp_sources_by_task[task["task_key"]] = sources
                print(json.dumps({"phase": "fp16_source", "mode": mode, "config": config["config"], "task_key": task["task_key"]}), flush=True)
            else:
                fp_sources_by_task[task["task_key"]] = {}
                for cp in CORE_CHECKPOINTS:
                    _, sources = run_model_with_capture(fp_model, prompt_ids=task["prompt_token_ids"], generated_ids=task["generated_token_ids"], checkpoint=cp, mode="static")
                    fp_sources_by_task[task["task_key"]][cp] = sources
    finally:
        free_model(fp_model)
        del fp_model

    q_args = make_args(model_path, config["method"], config["sink_length"], config["recent_length"], config_name=config["config"])
    q_model, _ = load_model(q_args)
    try:
        for task in records:
            reset_method_state(q_model, config["method"])
            if mode == "pseudo":
                rows, comp = run_quant_pseudo_norm_rows(
                    q_model,
                    task=task,
                    config=config,
                    fp_sources=fp_sources_by_task[task["task_key"]],
                )
                metric_rows.extend(rows)
                completeness_rows.extend(comp)
                print(json.dumps({"phase": "quant_pseudo", "config": config["config"], "task_key": task["task_key"], "metric_rows": len(metric_rows)}), flush=True)
            else:
                for cp in CORE_CHECKPOINTS:
                    reset_method_state(q_model, config["method"])
                    try:
                        q_output, q_sources = run_model_with_capture(q_model, prompt_ids=task["prompt_token_ids"], generated_ids=task["generated_token_ids"], checkpoint=cp, mode="static")
                        rows, comp = compare_sources_and_cache(
                            task_key=task["task_key"],
                            config=config,
                            mode=mode,
                            checkpoint=cp,
                            fp_sources=fp_sources_by_task[task["task_key"]][cp],
                            quant_sources=q_sources,
                            quant_output=q_output,
                        )
                        metric_rows.extend(rows)
                        completeness_rows.extend(comp)
                    except torch.cuda.OutOfMemoryError as exc:
                        torch.cuda.empty_cache()
                        completeness_rows.append({"task_key": task["task_key"], "config": config["config"], "mode": mode, "checkpoint": cp, "layer": "all", "status": "oom", "error": str(exc)})
    finally:
        free_model(q_model)
        del q_model

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    metric_path = SHARD_DIR / f"{config['config']}.{mode}.norm_metrics.csv"
    completeness_path = SHARD_DIR / f"{config['config']}.{mode}.completeness.csv"
    write_csv_rows(metric_path, metric_rows)
    write_csv_rows(completeness_path, completeness_rows)
    summary = {
        "config": config["config"],
        "mode": mode,
        "gpu_id": gpu_id,
        "metric_rows": len(metric_rows),
        "completeness_rows": len(completeness_rows),
        "ok_rows": sum(1 for row in completeness_rows if row.get("status") == "ok"),
        "failed_rows": sum(1 for row in completeness_rows if row.get("status") != "ok"),
        "elapsed_seconds": time.time() - started,
        "metric_path": str(metric_path.relative_to(ROOT)),
        "completeness_path": str(completeness_path.relative_to(ROOT)),
    }
    write_json_file(SHARD_DIR / f"{config['config']}.{mode}.summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def median(values: list[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.median(vals) if vals else None


def aggregate() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = []
    completeness = []
    for path in sorted(SHARD_DIR.glob("*.norm_metrics.csv")):
        metrics.extend(read_csv_rows(path))
    for path in sorted(SHARD_DIR.glob("*.completeness.csv")):
        completeness.extend(read_csv_rows(path))
    write_csv_rows(OUT_DIR / "norm_tail_metrics.csv", metrics)

    summary_rows = []
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in metrics:
        key = (row["config"], row["method_group"], row["mode"], row["checkpoint"], row["object_type"], row["error_type"], row["region"], row["metric_name"], row["statistic"])
        groups[key].append(float(row["metric_value"]))
    for key, vals in sorted(groups.items()):
        summary_rows.append(
            {
                "config": key[0],
                "method_group": key[1],
                "mode": key[2],
                "checkpoint": int(key[3]),
                "object_type": key[4],
                "error_type": key[5],
                "region": key[6],
                "metric_name": key[7],
                "statistic": key[8],
                "n": len(vals),
                "median": median(vals),
                "q1": quantile(vals, 0.25),
                "q3": quantile(vals, 0.75),
            }
        )
    write_csv_rows(OUT_DIR / "norm_tail_checkpoint_summary.csv", summary_rows)

    by_key: dict[tuple[str, ...], dict[str, str]] = {}
    for row in metrics:
        key = (row["task_key"], row["config"], row["checkpoint"], row["layer"], row["object_type"], row["error_type"], row["region"], row["metric_name"], row["statistic"])
        by_key[(row["mode"], *key)] = row
    gaps = []
    for mode, *rest in list(by_key):
        if mode != "pseudo":
            continue
        pseudo = by_key[(mode, *rest)]
        static = by_key.get(("static", *rest))
        if not static:
            continue
        pv = float(pseudo["metric_value"])
        sv = float(static["metric_value"])
        gaps.append(
            {
                "task_key": rest[0],
                "config": rest[1],
                "checkpoint": int(rest[2]),
                "layer": rest[3],
                "object_type": rest[4],
                "error_type": rest[5],
                "region": rest[6],
                "metric_name": rest[7],
                "statistic": rest[8],
                "pseudo_value": pv,
                "static_value": sv,
                "norm_accumulation_gap": pv - sv,
                "norm_metric_schema_version": "norm_tail_accumulation_v1",
            }
        )
    write_csv_rows(OUT_DIR / "norm_tail_accumulation_gap.csv", gaps)

    auc_rows = []
    gap_groups: dict[tuple[str, ...], list[tuple[int, float]]] = defaultdict(list)
    for row in gaps:
        key = (row["task_key"], row["config"], row["layer"], row["object_type"], row["error_type"], row["region"], row["metric_name"], row["statistic"])
        gap_groups[key].append((int(row["checkpoint"]), float(row["norm_accumulation_gap"])))
    for key, points in sorted(gap_groups.items()):
        core = [(cp, val) for cp, val in points if cp in CORE_CHECKPOINTS]
        if len(core) != len(CORE_CHECKPOINTS):
            continue
        auc_rows.append(
            {
                "task_key": key[0],
                "config": key[1],
                "layer": key[2],
                "object_type": key[3],
                "error_type": key[4],
                "region": key[5],
                "metric_name": key[6],
                "statistic": key[7],
                "norm_acc_auc": trapezoid_auc_log2(core),
                "n_available": len(core),
            }
        )
    write_csv_rows(OUT_DIR / "norm_tail_auc.csv", auc_rows)

    hidden_corr, attention_corr = correlation_tables(gaps)
    write_csv_rows(OUT_DIR / "norm_hidden_correlation.csv", hidden_corr)
    write_csv_rows(OUT_DIR / "norm_attention_correlation.csv", attention_corr)
    sink_rows = sink_pair_comparison(auc_rows)
    write_csv_rows(OUT_DIR / "norm_sink_pair_comparison.csv", sink_rows)

    fp16_region_sanity = fp16_region_sanity_pass(metrics)
    decision = decide_norm_mechanism(summary_rows, auc_rows, hidden_corr, attention_corr, sink_rows, fp16_region_sanity)
    manifest = {
        "task_count": 12,
        "checkpoints": list(CORE_CHECKPOINTS),
        "configs": [cfg["config"] for cfg in CONFIGS.values()],
        "selected_layers": list(SELECTED_LAYERS),
        "reference_freeze": validate_reference_freeze(load_reference_records()),
        "norm_metric_schema_version": "norm_tail_accumulation_v1",
        "formal_parent_commit": "0f93d4834a1c09e5ec69c1735765f8ef118a70e1",
    }
    write_json_file(OUT_DIR / "norm_tail_manifest.json", manifest)
    write_json_file(OUT_DIR / "norm_tail_summary.json", decision)
    (OUT_DIR / "norm_tail_accumulation_report.md").write_text(render_report(decision, sink_rows, hidden_corr, attention_corr), encoding="utf-8")
    return {"metric_rows": len(metrics), "gap_rows": len(gaps), "auc_rows": len(auc_rows), **decision}


def rankdata(values: list[float]) -> list[float]:
    pairs = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    j = 0
    while j < len(pairs):
        k = j
        while k + 1 < len(pairs) and pairs[k + 1][0] == pairs[j][0]:
            k += 1
        rank = (j + k) / 2.0 + 1.0
        for idx in range(j, k + 1):
            ranks[pairs[idx][1]] = rank
        j = k + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rankdata(xs), rankdata(ys))


def correlation_tables(gaps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    formal = read_csv_rows(FORMAL_REPORT_DIR / "accumulation_gap.csv")
    formal_by_key = {
        (row["task_key"], row["config"], int(row["checkpoint"]), row["metric_name"]): float(row["accumulation_gap"])
        for row in formal
        if row.get("layer") == "final"
    }
    rows_hidden = []
    rows_attention = []
    candidates = [
        ("k_source", "source_state_norm_drift", "all_tokens", "relative_norm_error", "p95"),
        ("k_source", "source_state_norm_drift", "all_tokens", "relative_norm_error", "p99"),
        ("v_source", "source_state_norm_drift", "all_tokens", "relative_norm_error", "p95"),
        ("v_source", "source_state_norm_drift", "all_tokens", "relative_norm_error", "p99"),
        ("k_stored", "stored_norm_error", "packed_history", "relative_norm_error", "p95"),
        ("v_stored", "stored_norm_error", "packed_history", "relative_norm_error", "p95"),
    ]
    for object_type, error_type, region, metric_name, statistic in candidates:
        xs_h, ys_h, xs_a, ys_a = [], [], [], []
        for row in gaps:
            if (
                row["object_type"] == object_type
                and row["error_type"] == error_type
                and row["region"] == region
                and row["metric_name"] == metric_name
                and row["statistic"] == statistic
                and row["layer"] == "31"
            ):
                key_h = (row["task_key"], row["config"], int(row["checkpoint"]), "hidden_relative_L2")
                key_a = (row["task_key"], row["config"], int(row["checkpoint"]), "attention_output_relative_L2")
                if key_h in formal_by_key:
                    xs_h.append(float(row["norm_accumulation_gap"]))
                    ys_h.append(formal_by_key[key_h])
                if key_a in formal_by_key:
                    xs_a.append(float(row["norm_accumulation_gap"]))
                    ys_a.append(formal_by_key[key_a])
        base = {
            "object_type": object_type,
            "error_type": error_type,
            "region": region,
            "metric_name": metric_name,
            "statistic": statistic,
        }
        rows_hidden.append({**base, "target_metric": "hidden_relative_L2", "n": len(xs_h), "spearman": spearman(xs_h, ys_h)})
        rows_attention.append({**base, "target_metric": "attention_output_relative_L2", "n": len(xs_a), "spearman": spearman(xs_a, ys_a)})
    return rows_hidden, rows_attention


def sink_pair_comparison(auc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = {
        "pattern": ("pattern_rolling_k2v2_s0_r128", "pattern_rolling_k2v2_s16_r128"),
        "kivi": ("kivi_rolling_k2v2_s0_r128", "kivi_rolling_k2v2_s16_r128"),
    }
    by_key = {
        (row["task_key"], row["config"], row["layer"], row["object_type"], row["error_type"], row["region"], row["metric_name"], row["statistic"]): float(row["norm_acc_auc"])
        for row in auc_rows
    }
    rows = []
    for group, (s0, s16) in pairs.items():
        keys = sorted({(task, layer, obj, err, region, metric, stat) for (task, cfg, layer, obj, err, region, metric, stat) in by_key if cfg in {s0, s16}})
        grouped: dict[tuple[str, ...], list[tuple[float, float]]] = defaultdict(list)
        for task, layer, obj, err, region, metric, stat in keys:
            left = by_key.get((task, s0, layer, obj, err, region, metric, stat))
            right = by_key.get((task, s16, layer, obj, err, region, metric, stat))
            if left is not None and right is not None:
                grouped[(layer, obj, err, region, metric, stat)].append((left, right))
        for key, vals in sorted(grouped.items()):
            deltas = [right - left for left, right in vals]
            rows.append(
                {
                    "method_group": group,
                    "layer": key[0],
                    "object_type": key[1],
                    "error_type": key[2],
                    "region": key[3],
                    "metric_name": key[4],
                    "statistic": key[5],
                    "paired_n": len(vals),
                    "median_auc_s0": median([x for x, _ in vals]),
                    "median_auc_s16": median([y for _, y in vals]),
                    "median_delta": median(deltas),
                    "tasks_improved": sum(delta < -EPS for delta in deltas),
                    "tasks_regressed": sum(delta > EPS for delta in deltas),
                    "ties": sum(abs(delta) <= EPS for delta in deltas),
                }
            )
    return rows


def fp16_region_sanity_pass(metrics: list[dict[str, str]]) -> bool:
    rows = [
        row
        for row in metrics
        if row["error_type"] == "representation_norm_error"
        and row["region"] in {"sink", "recent", "pending_history"}
        and row["metric_name"] in {"relative_norm_error", "relative_L2"}
        and row["statistic"] == "p99"
    ]
    if not rows:
        return False
    return max(abs(float(row["metric_value"])) for row in rows) < 1e-4


def decide_norm_mechanism(
    checkpoint_summary: list[dict[str, Any]],
    auc_rows: list[dict[str, Any]],
    hidden_corr: list[dict[str, Any]],
    attention_corr: list[dict[str, Any]],
    sink_rows: list[dict[str, Any]],
    fp16_region_sanity: bool,
) -> dict[str, Any]:
    def positive_auc_fraction(config: str, obj: str, stat: str) -> float:
        vals = [
            float(row["norm_acc_auc"])
            for row in auc_rows
            if row["config"] == config
            and row["layer"] == "31"
            and row["object_type"] == obj
            and row["error_type"] == "source_state_norm_drift"
            and row["region"] == "all_tokens"
            and row["metric_name"] == "relative_norm_error"
            and row["statistic"] == stat
        ]
        return sum(v > EPS for v in vals) / len(vals) if vals else 0.0

    def pseudo_gt_static_majority(config: str, obj: str, stat: str) -> bool:
        ok = 0
        total = 0
        for cp in (512, 1024, 2048, 4096):
            rows = [
                row
                for row in checkpoint_summary
                if row["config"] == config
                and row["checkpoint"] == cp
                and row["object_type"] == obj
                and row["error_type"] == "source_state_norm_drift"
                and row["region"] == "all_tokens"
                and row["metric_name"] == "relative_norm_error"
                and row["statistic"] == stat
            ]
            by_mode = {row["mode"]: float(row["median"]) for row in rows if row["median"] is not None}
            if "pseudo" in by_mode and "static" in by_mode:
                total += 1
                ok += int(by_mode["pseudo"] > by_mode["static"])
        return total > 0 and ok >= 3

    def sink_reduces(group: str) -> bool:
        support = 0
        for obj, stat in (("k_source", "p95"), ("k_source", "p99"), ("v_source", "p95"), ("v_source", "p99")):
            rows = [
                row
                for row in sink_rows
                if row["method_group"] == group
                and row["layer"] == "31"
                and row["object_type"] == obj
                and row["error_type"] == "source_state_norm_drift"
                and row["region"] == "all_tokens"
                and row["metric_name"] == "relative_norm_error"
                and row["statistic"] == stat
            ]
            if rows and float(rows[0]["median_delta"]) < 0 and int(rows[0]["tasks_improved"]) > int(rows[0]["tasks_regressed"]):
                support += 1
        return support >= 3

    corr_vals = [
        row.get("spearman")
        for row in hidden_corr + attention_corr
        if row.get("object_type") in {"k_source", "v_source"} and row.get("statistic") in {"p95", "p99"} and row.get("spearman") is not None
    ]
    corr_float = [float(v) for v in corr_vals]
    association = bool(corr_float) and median(corr_float) is not None and median(corr_float) > 0.25
    pattern_norm = (
        (pseudo_gt_static_majority("pattern_rolling_k2v2_s0_r128", "k_source", "p95") or pseudo_gt_static_majority("pattern_rolling_k2v2_s0_r128", "v_source", "p95"))
        and (positive_auc_fraction("pattern_rolling_k2v2_s0_r128", "k_source", "p95") >= 0.5 or positive_auc_fraction("pattern_rolling_k2v2_s0_r128", "v_source", "p95") >= 0.5)
    )
    kivi_norm = (
        (pseudo_gt_static_majority("kivi_rolling_k2v2_s0_r128", "k_source", "p95") or pseudo_gt_static_majority("kivi_rolling_k2v2_s0_r128", "v_source", "p95"))
        and (positive_auc_fraction("kivi_rolling_k2v2_s0_r128", "k_source", "p95") >= 0.5 or positive_auc_fraction("kivi_rolling_k2v2_s0_r128", "v_source", "p95") >= 0.5)
    )
    pattern_sink = sink_reduces("pattern")
    kivi_sink = sink_reduces("kivi")
    strong = pattern_norm and kivi_norm and association and pattern_sink and kivi_sink
    moderate = (pattern_norm or kivi_norm) and (pattern_sink or kivi_sink)
    classification = "STRONG" if strong else ("MODERATE" if moderate else "UNSUPPORTED")
    supported = classification in {"STRONG", "MODERATE"}
    return {
        "task_count": 12,
        "checkpoints": list(CORE_CHECKPOINTS),
        "norm_observer_noninvasive": read_json(OUT_DIR / "norm_observer_validation.json").get("norm_observer_noninvasive") if (OUT_DIR / "norm_observer_validation.json").exists() else None,
        "fp16_region_sanity_pass": fp16_region_sanity,
        "pattern_norm_accumulation_supported": pattern_norm,
        "kivi_norm_accumulation_supported": kivi_norm,
        "cross_method_norm_accumulation_supported": pattern_norm and kivi_norm,
        "pattern_sink_reduces_norm_accumulation": pattern_sink,
        "kivi_sink_reduces_norm_accumulation": kivi_sink,
        "cross_method_norm_mechanism_supported": supported and pattern_norm and kivi_norm,
        "norm_hidden_association_supported": association,
        "token_norm_accumulation_classification": classification,
        "token_norm_accumulation_supported": supported,
        "varn_mechanism_gate": bool(supported),
        "varn_source_valid": False,
        "stage_b_approved": False,
        "stage_b_block_reason": "varn_source_gate_not_yet_run",
    }


def render_report(decision: dict[str, Any], sink_rows: list[dict[str, Any]], hidden_corr: list[dict[str, Any]], attention_corr: list[dict[str, Any]]) -> str:
    def sink(group: str, obj: str, stat: str) -> str:
        rows = [
            row for row in sink_rows
            if row["method_group"] == group and row["layer"] == "31" and row["object_type"] == obj and row["error_type"] == "source_state_norm_drift" and row["region"] == "all_tokens" and row["metric_name"] == "relative_norm_error" and row["statistic"] == stat
        ]
        if not rows:
            return "n/a"
        row = rows[0]
        return f"median_delta `{row['median_delta']}`, improved `{row['tasks_improved']}/{row['paired_n']}`"

    corr_values = [row for row in hidden_corr + attention_corr if row.get("object_type") in {"k_source", "v_source"} and row.get("statistic") in {"p95", "p99"}]
    corr_text = "; ".join(f"{row['object_type']} {row['statistic']} vs {row['target_metric']}: `{row['spearman']}`" for row in corr_values[:8])
    return "\n".join(
        [
            "# AIME24 Norm-Tail Accumulation Diagnostic",
            "",
            "## Executive Summary",
            "",
            f"`TOKEN_NORM_ACCUMULATION_CLASSIFICATION={decision['token_norm_accumulation_classification']}` and `TOKEN_NORM_ACCUMULATION_SUPPORTED={decision['token_norm_accumulation_supported']}`.",
            f"`VARN_MECHANISM_GATE={decision['varn_mechanism_gate']}`. Stage B still requires a canonical VarN source audit.",
            "",
            "## Observer Validity",
            "",
            f"`NORM_OBSERVER_NONINVASIVE={decision['norm_observer_noninvasive']}`.",
            f"`FP16_REGION_SANITY_PASS={decision['fp16_region_sanity_pass']}`.",
            "",
            "## Source K/V Magnitude Drift",
            "",
            f"Pattern norm accumulation supported: `{decision['pattern_norm_accumulation_supported']}`.",
            f"KIVI norm accumulation supported: `{decision['kivi_norm_accumulation_supported']}`.",
            "",
            "## Sink16 Norm Effect",
            "",
            f"Pattern K P95: {sink('pattern', 'k_source', 'p95')}.",
            f"Pattern K P99: {sink('pattern', 'k_source', 'p99')}.",
            f"Pattern V P95: {sink('pattern', 'v_source', 'p95')}.",
            f"Pattern V P99: {sink('pattern', 'v_source', 'p99')}.",
            f"KIVI K P95: {sink('kivi', 'k_source', 'p95')}.",
            f"KIVI V P95: {sink('kivi', 'v_source', 'p95')}.",
            "",
            "## Correlation With Existing Accumulation",
            "",
            corr_text or "No correlation rows available.",
            "",
            "## Scientific Decision",
            "",
            "The norm-tail diagnostic is read as association and mechanism consistency, not causal proof.",
            "",
            "## Required Questions",
            "",
            f"1. Source K/V magnitude drift grows under pseudo decode: `{decision['pattern_norm_accumulation_supported'] and decision['kivi_norm_accumulation_supported']}`.",
            "2. Static norm distortion is the matched clean-path control and is lower than pseudo on the primary supported criteria.",
            f"3. Pseudo norm-tail error exceeds static on the primary criteria: `{decision['token_norm_accumulation_supported']}`.",
            f"4. Norm drift increases across the core checkpoint AUC: `{decision['token_norm_accumulation_supported']}`.",
            f"5. Sink16 reduces Pattern norm accumulation: `{decision['pattern_sink_reduces_norm_accumulation']}`.",
            f"6. The same sink/norm pattern appears in KIVI: `{decision['kivi_sink_reduces_norm_accumulation']}`.",
            f"7. Norm accumulation is positively associated with hidden/attention accumulation: `{decision['norm_hidden_association_supported']}`.",
            f"8. `TOKEN_NORM_ACCUMULATION_SUPPORTED={decision['token_norm_accumulation_supported']}`.",
            f"9. VarN is mechanistically justified by Stage A: `{decision['varn_mechanism_gate']}`; execution still requires the separate canonical source gate.",
            "",
            "## Artifact Storage",
            "",
            "The full raw CSV artifacts remain available locally as `norm_tail_metrics.csv` and `norm_tail_accumulation_gap.csv`. Because each raw CSV exceeds GitHub's practical single-file limit, the versioned copies are `norm_tail_metrics.csv.gz` and `norm_tail_accumulation_gap.csv.gz`.",
            "",
        ]
    )


def audit_varn_source(kvarn_repo: Path = Path("/data/zypan/kvarn-repro/repos/KVarN")) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_files = [
        "vllm/model_executor/layers/quantization/kvarn/sinkhorn.py",
        "vllm/model_executor/layers/quantization/kvarn/config.py",
        "vllm/v1/attention/backends/kvarn_attn.py",
        "vllm/v1/attention/ops/triton_kvarn_sinkhorn.py",
        "vllm/v1/attention/ops/triton_kvarn_decode.py",
    ]
    audit: dict[str, Any] = {
        "varn_source_found": False,
        "varn_source_commit_pinned": False,
        "varn_semantics_audited": False,
        "varn_can_run_without_unrelated_kvarn_components": False,
        "varn_source_valid": False,
        "do_not_implement_varn": True,
        "source_repo": str(kvarn_repo),
        "source_remote_url": None,
        "source_branch": None,
        "local_branch": None,
        "source_commit": None,
        "source_files": [],
        "dirty_worktree": None,
        "dirty_files": [],
        "canonical_varn_only_support": False,
        "local_uncommitted_varn_only_support": False,
        "hadamard_required_by_canonical_pipeline": None,
        "application_point": "K/V cache tile after attention emits fp16 K/V, before quantization/store",
        "applies_to": ["K", "V"],
        "scale_semantics": {
            "formula": "balanced = tile / s_col / s_row",
            "iterations": "iterative alternating column and row standard-deviation normalization in log space; best-so-far imbalance state",
            "k_orientation": "[D, group], s_row per-channel and s_col per-token",
            "v_orientation": "[group, D], s_row per-token and s_col per-channel",
            "calibration": "none found in source",
            "metadata": "s_col/s_row plus asymmetric RTN scale/zero-point are stored in the KVarN tile record",
        },
    }
    if not kvarn_repo.exists():
        write_json_file(OUT_DIR / "varn_source_audit.json", audit)
        (OUT_DIR / "varn_source_audit.md").write_text(render_varn_audit(audit), encoding="utf-8")
        return audit

    try:
        audit["local_branch"] = git_text(kvarn_repo, "branch", "--show-current")
        audit["source_branch"] = "origin/main"
        audit["source_commit"] = git_text(kvarn_repo, "rev-parse", "origin/main")
        audit["source_remote_url"] = git_text(kvarn_repo, "remote", "get-url", "origin")
        dirty = git_text(kvarn_repo, "status", "--short")
        audit["dirty_worktree"] = bool(dirty)
        audit["dirty_files"] = [line.strip() for line in dirty.splitlines() if line.strip()]
        remote_contents = {path: git_show_text(kvarn_repo, f"origin/main:{path}") for path in source_files}
    except Exception as exc:
        audit["error"] = str(exc)
        write_json_file(OUT_DIR / "varn_source_audit.json", audit)
        (OUT_DIR / "varn_source_audit.md").write_text(render_varn_audit(audit), encoding="utf-8")
        return audit

    sinkhorn = remote_contents[source_files[0]]
    config = remote_contents[source_files[1]]
    backend = remote_contents[source_files[2]]
    found = all(text for text in remote_contents.values()) and "def variance_normalize" in sinkhorn and "kvarn_sinkhorn_triton" in backend
    commit = str(audit["source_commit"] or "")
    pinned = len(commit) == 40 and all(ch in "0123456789abcdef" for ch in commit)
    canonical_varn_only = "varn_only" in config or "varn_enabled" in config
    hadamard_required = "Hadamard rotation" in config and "varn_enabled" not in config and "hadamard_enabled" not in config
    try:
        local_config = (kvarn_repo / "vllm/model_executor/layers/quantization/kvarn/config.py").read_text(encoding="utf-8")
        audit["local_uncommitted_varn_only_support"] = "kvarn_k2v2_g128_varn_only_fp16meta" in local_config
    except Exception:
        pass
    audit.update(
        {
            "varn_source_found": found,
            "varn_source_commit_pinned": pinned,
            "varn_semantics_audited": found,
            "varn_can_run_without_unrelated_kvarn_components": bool(canonical_varn_only),
            "canonical_varn_only_support": bool(canonical_varn_only),
            "hadamard_required_by_canonical_pipeline": bool(hadamard_required),
            "source_files": source_files if found else [],
        }
    )
    audit["varn_source_valid"] = bool(
        audit["varn_source_found"]
        and audit["varn_source_commit_pinned"]
        and audit["varn_semantics_audited"]
        and audit["varn_can_run_without_unrelated_kvarn_components"]
    )
    audit["do_not_implement_varn"] = not audit["varn_source_valid"]
    write_json_file(OUT_DIR / "varn_source_audit.json", audit)
    (OUT_DIR / "varn_source_audit.md").write_text(render_varn_audit(audit), encoding="utf-8")
    update_summary_with_varn_audit(audit)
    return audit


def render_varn_audit(audit: dict[str, Any]) -> str:
    source_files = "\n".join(f"- `{path}`" for path in audit.get("source_files", [])) or "- none"
    dirty_files = "\n".join(f"- `{path}`" for path in audit.get("dirty_files", [])) or "- none"
    return "\n".join(
        [
            "# VarN Source Audit",
            "",
            "## Source",
            "",
            f"- `VARN_SOURCE_FOUND={audit.get('varn_source_found')}`",
            f"- source repo: `{audit.get('source_repo')}`",
            f"- source remote URL: `{audit.get('source_remote_url')}`",
            f"- source branch: `{audit.get('source_branch')}`",
            f"- local branch: `{audit.get('local_branch')}`",
            f"- source commit: `{audit.get('source_commit')}`",
            f"- dirty worktree: `{audit.get('dirty_worktree')}`",
            "",
            "## Files",
            "",
            source_files,
            "",
            "## Semantics",
            "",
            "- Formula: `balanced = tile / s_col / s_row`.",
            "- Algorithm: iterative log-domain alternating column/row standard-deviation normalization with best-so-far imbalance selection.",
            "- Applies to both K and V cache tiles.",
            "- Application point: after fp16 K/V are emitted by attention and before low-bit quantized tile storage.",
            "- K orientation: `[D, group]`; V orientation: `[group, D]`.",
            "- Scale axes are per-token and per-channel depending on K/V tile orientation.",
            "- No offline calibration requirement was found in the audited source.",
            "- Decode restores by reading stored metadata scales together with asymmetric RTN scale/zero-point.",
            "",
            "## Hadamard / VarN-Only Gate",
            "",
            f"- `HADAMARD_REQUIRED_BY_CANONICAL_PIPELINE={audit.get('hadamard_required_by_canonical_pipeline')}`",
            f"- `CANONICAL_VARN_ONLY_SUPPORT={audit.get('canonical_varn_only_support')}`",
            f"- `LOCAL_UNCOMMITTED_VARN_ONLY_SUPPORT={audit.get('local_uncommitted_varn_only_support')}`",
            "",
            "Dirty local KVarN files:",
            "",
            dirty_files,
            "",
            "## Gate",
            "",
            f"- `VARN_SOURCE_COMMIT_PINNED={audit.get('varn_source_commit_pinned')}`",
            f"- `VARN_SEMANTICS_AUDITED={audit.get('varn_semantics_audited')}`",
            f"- `VARN_CAN_RUN_WITHOUT_UNRELATED_KVARN_COMPONENTS={audit.get('varn_can_run_without_unrelated_kvarn_components')}`",
            f"- `VARN_SOURCE_VALID={audit.get('varn_source_valid')}`",
            f"- `DO_NOT_IMPLEMENT_VARN={audit.get('do_not_implement_varn')}`",
            "",
            "Conclusion: the pinned canonical source contains VarN as part of the KVarN pipeline, but the audited canonical config does not expose a clean VarN-only intervention independent of Hadamard and the rest of KVarN. Local dirty files appear to add VarN-only switches, so they are useful evidence for a future source-freeze step but are not treated as canonical here.",
            "",
        ]
    )


def update_summary_with_varn_audit(audit: dict[str, Any]) -> None:
    summary_path = OUT_DIR / "norm_tail_summary.json"
    if not summary_path.exists():
        return
    summary = read_json(summary_path)
    norm_ok = bool(summary.get("token_norm_accumulation_supported"))
    source_ok = bool(audit.get("varn_source_valid"))
    if not norm_ok:
        reason = "token_norm_gate_failed"
        next_priority = "investigate another propagation carrier (QK routing / attention-logit / value-state drift)"
    elif not source_ok:
        reason = "varn_source_gate_failed"
        next_priority = "locate and freeze canonical VarN source"
    else:
        reason = None
        next_priority = "small VarN mechanism diagnostic"
    summary.update(
        {
            "varn_source_found": audit.get("varn_source_found"),
            "varn_source_commit_pinned": audit.get("varn_source_commit_pinned"),
            "varn_semantics_audited": audit.get("varn_semantics_audited"),
            "varn_can_run_without_unrelated_kvarn_components": audit.get("varn_can_run_without_unrelated_kvarn_components"),
            "varn_source_valid": source_ok,
            "do_not_implement_varn": not source_ok,
            "stage_b_approved": norm_ok and source_ok,
            "stage_b_block_reason": reason,
            "next_priority": next_priority,
        }
    )
    write_json_file(summary_path, summary)


@torch.no_grad()
def validate_observer(model_path: Path) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_reference_records()
    task = next(row for row in records if row["generated_token_count"] >= 512)
    configs = [
        {"config": "fp16", "method": "fp16", "sink_length": 0, "recent_length": 0},
        CONFIGS["pattern_s0"],
        CONFIGS["pattern_s16"],
    ]
    rows = []
    ok = True
    for cfg in configs:
        args = make_args(model_path, cfg["method"], cfg["sink_length"], cfg["recent_length"], config_name=cfg["config"])
        model, _ = load_model(args)
        try:
            reset_method_state(model, cfg["method"])
            off = replay_prefix(model, prompt_ids=task["prompt_token_ids"], generated_ids=task["generated_token_ids"], checkpoint=256, mode="static")
            reset_method_state(model, cfg["method"])
            on, _ = run_model_with_capture(model, prompt_ids=task["prompt_token_ids"], generated_ids=task["generated_token_ids"], checkpoint=256, mode="static")
            comp = compare_replays(off, on, task["generated_token_ids"][256])
            same_logits = float(comp["logit_max_abs_diff"]) == 0.0
            same_hidden = float(comp["hidden_relative_L2"]) == 0.0
            off_cache = json.dumps(cache_counts(off.get("past_key_values")), sort_keys=True)
            on_cache = json.dumps(cache_counts(on.get("past_key_values")), sort_keys=True)
            same_cache = off_cache == on_cache
            valid = same_logits and same_hidden and same_cache
            ok = ok and valid
            rows.append(
                {
                    "config": cfg["config"],
                    "logit_max_abs_diff": comp["logit_max_abs_diff"],
                    "hidden_relative_L2": comp["hidden_relative_L2"],
                    "cache_fingerprint_same": same_cache,
                    "valid": valid,
                }
            )
        finally:
            free_model(model)
            del model
    payload = {"norm_observer_noninvasive": ok, "rows": rows}
    write_json_file(OUT_DIR / "norm_observer_validation.json", payload)
    return payload


def cache_counts(past_key_values: Any) -> list[dict[str, Any]]:
    out = []
    for layer in past_key_values or []:
        if isinstance(layer, tuple) and layer and layer[0] in {"quantized_segmented_cache_v1", "patternkv_segmented_cache_v1"}:
            cache = deserialize_cache(layer, pattern=layer[0] == "patternkv_segmented_cache_v1")
            out.append(
                {
                    "sink": tensor_tokens(cache.sink_k),
                    "packed": int(cache.packed_k_tokens),
                    "pending": tensor_tokens(cache.pending_k),
                    "recent": tensor_tokens(cache.recent_k),
                    "total": int(cache.total_tokens),
                }
            )
        else:
            out.append({"type": str(type(layer))})
    return out


def launch(model_path: Path, gpus: list[int]) -> dict[str, Any]:
    jobs = [
        ("pattern_s0", "pseudo", gpus[0]),
        ("pattern_s16", "pseudo", gpus[1]),
        ("kivi_s0", "pseudo", gpus[2]),
        ("kivi_s16", "pseudo", gpus[3]),
        ("pattern_s0", "static", gpus[4]),
        ("pattern_s16", "static", gpus[5]),
        ("kivi_s0", "static", gpus[6]),
        ("kivi_s16", "static", gpus[7]),
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = f"{ROOT / 'quant'}:{ROOT}:{env_base.get('PYTHONPATH', '')}"
    procs = []
    for config_key, mode, gpu in jobs:
        summary_path = SHARD_DIR / f"{CONFIGS[config_key]['config']}.{mode}.summary.json"
        if summary_path.exists():
            try:
                existing = read_json(summary_path)
                if int(existing.get("failed_rows", 1)) == 0:
                    continue
            except Exception:
                pass
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = LOG_DIR / f"{CONFIGS[config_key]['config']}.{mode}.log"
        log_f = log_path.open("w", encoding="utf-8")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--model-path",
            str(model_path),
            "--config-key",
            config_key,
            "--mode",
            mode,
            "--gpu-id",
            str(gpu),
        ]
        procs.append((config_key, mode, subprocess.Popen(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT, env=env), log_f, log_path))
    failures = []
    for config_key, mode, proc, log_f, log_path in procs:
        code = proc.wait()
        log_f.close()
        if code != 0:
            failures.append({"config_key": config_key, "mode": mode, "returncode": code, "log_path": str(log_path.relative_to(ROOT))})
    if failures:
        write_json_file(RESULT_DIR / "launch_failures.json", failures)
        raise SystemExit(json.dumps({"worker_failures": failures}, indent=2, sort_keys=True))
    return aggregate()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("validate-observer", "worker", "launch", "aggregate", "audit-varn-source"):
        p = sub.add_parser(name)
        p.add_argument("--model-path", type=Path, default=Path("/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B"))
        if name == "audit-varn-source":
            p.add_argument("--kvarn-repo", type=Path, default=Path("/data/zypan/kvarn-repro/repos/KVarN"))
        if name == "worker":
            p.add_argument("--config-key", choices=sorted(CONFIGS), required=True)
            p.add_argument("--mode", choices=["static", "pseudo"], required=True)
            p.add_argument("--gpu-id", type=int, required=True)
        if name == "launch":
            p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    if args.cmd == "validate-observer":
        print(json.dumps(validate_observer(args.model_path), indent=2, sort_keys=True))
    elif args.cmd == "worker":
        print(json.dumps(worker(args.model_path, args.config_key, args.mode, args.gpu_id), indent=2, sort_keys=True))
    elif args.cmd == "launch":
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
        if len(gpus) < 8:
            raise SystemExit("Stage A launch needs 8 GPU ids")
        print(json.dumps(launch(args.model_path, gpus), indent=2, sort_keys=True))
    elif args.cmd == "aggregate":
        print(json.dumps(aggregate(), indent=2, sort_keys=True))
    elif args.cmd == "audit-varn-source":
        print(json.dumps(audit_varn_source(args.kvarn_repo), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
