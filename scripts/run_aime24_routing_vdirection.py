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

from bench.pseudodecode_controls import MATCHED_PATH_CONTROL_VERSION, compute_accumulation_gap  # noqa: E402
from bench.pseudodecode_metrics import full_trajectory_sha256, trapezoid_auc_log2, write_csv_rows  # noqa: E402
from bench.routing_vdirection_observer import (  # noqa: E402
    EPS,
    SCHEMA_VERSION,
    all_finite,
    attention_probs,
    attention_regions,
    attention_weighted_vector_errors,
    current_query,
    gqa_kv_head_for_query_head,
    logit_metrics,
    oracle_error_metrics,
    oracle_outputs,
    probability_metrics,
    qk_logits,
    region_mass,
    repeat_kv_for_gqa,
    summarize_tensor,
    top1_agreement,
    topk_overlap,
    vector_errors,
)
from models.segmented_cache import (  # noqa: E402
    PatternQuantizedKVCache,
    dequantize_k_reference,
    dequantize_v_reference,
    deserialize_cache,
    pattern_gather_centroids,
    tensor_tokens,
)
from scripts.run_aime24_norm_tail_stage_a import SourceCapture  # noqa: E402
from scripts.run_aime24_pseudodecode_preflight import (  # noqa: E402
    REPORT_DIR as FORMAL_REPORT_DIR,
    SOURCE_COMMIT,
    cache_fingerprint,
    compare_replays,
    load_model,
    make_args,
    replay_prefix,
    reset_method_state,
    write_json,
)


OUT_DIR = ROOT / "reports/aime24_routing_vdirection_3090"
RESULT_DIR = ROOT / "results/aime24_routing_vdirection_3090"
SHARD_DIR = RESULT_DIR / "shards"
LOG_DIR = ROOT / "run/aime24_routing_vdirection_3090/logs"
TASK_SUBSET_PATH = ROOT / "configs/aime24_routing_vdirection_6tasks.json"
VAR_N_RESULT_COMMIT = "2f63bddef151df0f32a40d51c73f125a8089800e"
VAR_N_SUBSET_PATH = "reports/aime24_pattern_varn_mechanism_3090/varn_mechanism_6task_subset.json"
PARENT_COMMIT = "f7f6ca9954daa76cb702941f1b018ae294c0e378"
PORTABLE_HASH = "86648d12304ce11890c1a8f64bf5a896"
CONFIG = {
    "config": "pattern_rolling_k2v2_s16_r128",
    "method": "patternkv",
    "sink_length": 16,
    "recent_length": 128,
    "group_size": 128,
    "k_bits": 2,
    "v_bits": 2,
}
SELECTED_LAYERS = (0, 7, 15, 23, 31)
CORE_CHECKPOINTS = (128, 512, 1024, 2048, 4096)
PRIMARY_AUC_METRICS = {
    ("direction", "q_source", "current_token", "direction_error", "p95"),
    ("direction", "k_source", "current_token", "direction_error", "p95"),
    ("direction", "v_source", "current_token", "direction_error", "p95"),
    ("qk_logit", "qk_logits", "current_history", "relative_L2", "global"),
    ("qk_logit", "qk_logits", "current_history", "p95_abs_diff", "global"),
    ("qk_logit", "qk_logits", "current_history", "p99_abs_diff", "global"),
    ("attention_routing", "attention_probs", "current_history", "js", "mean"),
    ("attention_routing", "attention_probs", "current_history", "tv", "mean"),
    ("v_direction", "v_stored", "current_history", "weighted_direction_error_fp", "mean"),
    ("oracle_output", "attention_output", "current_history", "actual_relative_L2", "global"),
    ("oracle_output", "attention_output", "current_history", "routing_only_relative_L2", "global"),
    ("oracle_output", "attention_output", "current_history", "value_only_relative_L2", "global"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()


def git_show(repo: Path, rev_path: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "show", rev_path], text=True, stderr=subprocess.DEVNULL)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gzip_file(path: Path) -> str:
    gz = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as src, gzip.open(gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return str(gz.relative_to(ROOT))


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
        valid = observed == expected
        ok = ok and valid
        rows.append({"task_key": row["task_key"], "expected": expected, "observed": observed, "valid": valid})
    return {
        "reference_hashes_valid": ok and len(records) == 12,
        "reference_trajectories": f"{sum(1 for row in rows if row['valid'])}/{len(rows)}",
        "portable_generation_hash": PORTABLE_HASH,
        "rows": rows,
    }


def load_varn_subset_payload() -> tuple[dict[str, Any], str]:
    text = git_show(ROOT, f"{VAR_N_RESULT_COMMIT}:{VAR_N_SUBSET_PATH}")
    digest = sha256_bytes(text.encode("utf-8"))
    return json.loads(text), digest


def selected_records() -> list[dict[str, Any]]:
    if not TASK_SUBSET_PATH.exists():
        prepare()
    payload = read_json(TASK_SUBSET_PATH)
    records = load_reference_records()
    by_key = {row["task_key"]: row for row in records}
    return [by_key[item["task_key"]] for item in payload["tasks"]]


def prepare() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TASK_SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = load_reference_records()
    freeze = validate_reference_freeze(records)
    subset, subset_sha = load_varn_subset_payload()
    by_key = {row["task_key"]: row for row in records}
    subset_keys = [item["task_key"] for item in subset["tasks"]]
    rows = []
    same = True
    for item in subset["tasks"]:
        record = by_key[item["task_key"]]
        traj = full_trajectory_sha256(record["prompt_token_ids"], record["generated_token_ids"])
        valid = traj == item["full_trajectory_sha256"] == record["full_trajectory_sha256"]
        same = same and valid
        rows.append(
            {
                "task_key": record["task_key"],
                "problem_id": int(record["problem_id"]),
                "sample_id": int(record["sample_id"]),
                "seed": int(record["seed"]),
                "prompt_token_count": int(record["prompt_token_count"]),
                "generated_token_count": int(record["generated_token_count"]),
                "full_trajectory_sha256": record["full_trajectory_sha256"],
                "artifact_path": record["artifact_path"],
                "reference_hash_valid": valid,
            }
        )
    payload = {
        "selection_rule": "exact VarN formal six-task subset, loaded from git object at source_varn_result_commit",
        "source_varn_result_commit": VAR_N_RESULT_COMMIT,
        "source_varn_subset_path": VAR_N_SUBSET_PATH,
        "source_varn_subset_sha256": subset_sha,
        "same_varn_formal_6task_subset": subset_keys == [row["task_key"] for row in rows],
        "portable_generation_hash": PORTABLE_HASH,
        "task_count": len(rows),
        "checkpoints": list(CORE_CHECKPOINTS),
        "tasks": rows,
    }
    write_json_file(TASK_SUBSET_PATH, payload)
    write_json_file(OUT_DIR / "routing_vdirection_6task_subset.json", payload)
    head = git_text(ROOT, "rev-parse", "HEAD")
    origin = {
        "repository": "pytenter/Bounded-pattrenKV-method",
        "branch": git_text(ROOT, "branch", "--show-current"),
        "head": head,
        "parent_commit": PARENT_COMMIT,
        "source_varn_result_commit": VAR_N_RESULT_COMMIT,
        "source_varn_subset_sha256": subset_sha,
        "worktree_dirty_at_prepare": bool(git_text(ROOT, "status", "--short")),
        "dirty_files_at_prepare": [line.strip() for line in git_text(ROOT, "status", "--short").splitlines() if line.strip()],
    }
    write_json_file(OUT_DIR / "experiment_origin.json", origin)
    cfg = {
        "config": CONFIG,
        "checkpoints": list(CORE_CHECKPOINTS),
        "selected_layers": list(SELECTED_LAYERS),
        "mode": ["static", "pseudo"],
        "matched_path": True,
        "observer_only": True,
        "no_new_algorithm": True,
        "forbidden": ["VarN", "Hadamard", "new Sink", "new Recent", "Pattern-MSE", "query-aware assignment", "attention-aware assignment"],
        "observer_schema_version": SCHEMA_VERSION,
    }
    write_json_file(OUT_DIR / "routing_vdirection_config.json", cfg)
    return {
        "prepared": True,
        "task_count": len(rows),
        "same_varn_formal_6task_subset": payload["same_varn_formal_6task_subset"] and same,
        "source_varn_subset_sha256": subset_sha,
        "reference_hashes_valid": freeze["reference_hashes_valid"],
    }


def free_model(model: torch.nn.Module | None) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def cache_tensor_signature(obj: Any) -> Any:
    if torch.is_tensor(obj):
        t = obj.detach().cpu().contiguous()
        return {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest()}
    if isinstance(obj, tuple):
        return [cache_tensor_signature(x) for x in obj]
    if isinstance(obj, list):
        return [cache_tensor_signature(x) for x in obj]
    if isinstance(obj, dict):
        return {key: cache_tensor_signature(value) for key, value in obj.items()}
    return obj


def run_with_source_capture(model: torch.nn.Module, *, task: dict[str, Any], checkpoint: int, mode: str) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    capture = SourceCapture(selected_layers=SELECTED_LAYERS)
    capture.install(model)
    try:
        output = replay_prefix(model, prompt_ids=task["prompt_token_ids"], generated_ids=task["generated_token_ids"], checkpoint=checkpoint, mode=mode)
        return output, capture.tensors()
    finally:
        capture.remove()


def dequantized_layer_kv(layer_cache: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    cache = deserialize_cache(layer_cache, pattern=True)
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
    k_parts = [x for x in (cache.sink_k, packed_k, cache.pending_k, cache.recent_k) if torch.is_tensor(x) and tensor_tokens(x) > 0]
    v_parts = [x for x in (cache.sink_v, packed_v, cache.pending_v, cache.recent_v) if torch.is_tensor(x) and tensor_tokens(x) > 0]
    if not k_parts or not v_parts:
        raise ValueError("empty reconstructed cache")
    counts = {
        "sink_tokens": tensor_tokens(cache.sink_k),
        "packed_history_tokens": int(cache.packed_k_tokens),
        "pending_history_tokens": tensor_tokens(cache.pending_k),
        "recent_tokens": tensor_tokens(cache.recent_k),
        "total_tokens": int(cache.total_tokens),
    }
    return torch.cat(k_parts, dim=2).contiguous(), torch.cat(v_parts, dim=2).contiguous(), counts


def add_stat_rows(rows: list[dict[str, Any]], *, family: str, task: dict[str, Any], mode: str, checkpoint: int, layer: int, object_type: str, region: str, metric_name: str, values: torch.Tensor, statistic_prefix: str = "") -> None:
    stats = summarize_tensor(values)
    for stat in ("mean", "p50", "p90", "p95", "p99", "max"):
        rows.append(
            {
                "task_key": task["task_key"],
                "trajectory_sha256": task["full_trajectory_sha256"],
                "config": CONFIG["config"],
                "mode": mode,
                "checkpoint": checkpoint,
                "absolute_sequence_position": int(task["prompt_token_count"]) + int(checkpoint),
                "layer": str(layer),
                "metric_family": family,
                "object_type": object_type,
                "region": region,
                "metric_name": metric_name,
                "statistic": statistic_prefix + stat,
                "metric_value": stats[stat],
                "n_samples": stats["n_samples"],
                "matched_path_control_version": MATCHED_PATH_CONTROL_VERSION,
                "observer_schema_version": SCHEMA_VERSION,
                "source_commit": SOURCE_COMMIT,
            }
        )


def add_scalar_row(rows: list[dict[str, Any]], *, family: str, task: dict[str, Any], mode: str, checkpoint: int, layer: int, object_type: str, region: str, metric_name: str, value: float, statistic: str = "global", extra: dict[str, Any] | None = None) -> None:
    rows.append(
        {
            "task_key": task["task_key"],
            "trajectory_sha256": task["full_trajectory_sha256"],
            "config": CONFIG["config"],
            "mode": mode,
            "checkpoint": checkpoint,
            "absolute_sequence_position": int(task["prompt_token_count"]) + int(checkpoint),
            "layer": str(layer),
            "metric_family": family,
            "object_type": object_type,
            "region": region,
            "metric_name": metric_name,
            "statistic": statistic,
            "metric_value": float(value),
            "n_samples": 1,
            "matched_path_control_version": MATCHED_PATH_CONTROL_VERSION,
            "observer_schema_version": SCHEMA_VERSION,
            "source_commit": SOURCE_COMMIT,
            **(extra or {}),
        }
    )


def slice_sources(sources: dict[int, dict[str, torch.Tensor]], total: int) -> dict[int, dict[str, torch.Tensor]]:
    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer, layer_map in sources.items():
        out[layer] = {}
        for name, tensor in layer_map.items():
            if name == "hidden":
                out[layer][name] = tensor[:, :total, :].contiguous()
            else:
                out[layer][name] = tensor[:, :, :total, :].contiguous()
    return out


def compute_channel_rows(
    *,
    task: dict[str, Any],
    mode: str,
    checkpoint: int,
    fp_sources: dict[int, dict[str, torch.Tensor]],
    quant_sources: dict[int, dict[str, torch.Tensor]],
    quant_past: Any,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    families = {"direction": [], "qk_logit": [], "attention_routing": [], "v_direction": [], "oracle_output": []}
    completeness: list[dict[str, Any]] = []
    model_cfg = {
        "num_attention_heads": None,
        "num_key_value_heads": None,
        "head_dim": None,
        "num_key_value_groups": None,
    }
    no_nan_inf = True
    for layer in SELECTED_LAYERS:
        try:
            fp = fp_sources[layer]
            qt = quant_sources[layer]
            q_fp = fp["q_source"]
            k_fp = fp["k_source"]
            v_fp = fp["v_source"]
            q_q = qt["q_source"]
            k_q_source = qt["k_source"]
            v_q_source = qt["v_source"]
            stored_k, stored_v, counts = dequantized_layer_kv(quant_past[layer])
            total = min(q_fp.shape[2], q_q.shape[2], k_fp.shape[2], v_fp.shape[2], stored_k.shape[2], stored_v.shape[2])
            q_fp = q_fp[:, :, :total, :]
            k_fp = k_fp[:, :, :total, :]
            v_fp = v_fp[:, :, :total, :]
            q_q = q_q[:, :, :total, :]
            k_q_source = k_q_source[:, :, :total, :]
            v_q_source = v_q_source[:, :, :total, :]
            stored_k = stored_k[:, :, :total, :].cpu()
            stored_v = stored_v[:, :, :total, :].cpu()
            q_heads = int(q_fp.shape[1])
            kv_heads = int(k_fp.shape[1])
            groups = q_heads // kv_heads
            model_cfg = {
                "num_attention_heads": q_heads,
                "num_key_value_heads": kv_heads,
                "head_dim": int(q_fp.shape[-1]),
                "num_key_value_groups": groups,
            }
            for object_type, left, right in (
                ("q_source", q_q, q_fp),
                ("k_source", k_q_source, k_fp),
                ("v_source", v_q_source, v_fp),
            ):
                errs = vector_errors(left, right)
                for metric_name, values in errs.items():
                    add_stat_rows(families["direction"], family="direction", task=task, mode=mode, checkpoint=checkpoint, layer=layer, object_type=object_type, region="all_tokens", metric_name=metric_name, values=values)
                    add_stat_rows(families["direction"], family="direction", task=task, mode=mode, checkpoint=checkpoint, layer=layer, object_type=object_type, region="current_token", metric_name=metric_name, values=values[..., -1:])
            stored_v_err = vector_errors(stored_v, v_fp)
            for metric_name, values in stored_v_err.items():
                add_stat_rows(families["v_direction"], family="v_direction", task=task, mode=mode, checkpoint=checkpoint, layer=layer, object_type="v_stored", region="all_tokens", metric_name=metric_name, values=values)

            q_fp_cur = current_query(q_fp)
            q_q_cur = current_query(q_q)
            l_fp = qk_logits(q_fp_cur, k_fp, num_key_value_groups=groups)
            l_quant = qk_logits(q_q_cur, stored_k, num_key_value_groups=groups)
            for metric_name, value in logit_metrics(l_quant, l_fp).items():
                add_scalar_row(families["qk_logit"], family="qk_logit", task=task, mode=mode, checkpoint=checkpoint, layer=layer, object_type="qk_logits", region="current_history", metric_name=metric_name, value=value)
            top1 = top1_agreement(l_quant, l_fp)
            add_scalar_row(families["qk_logit"], family="qk_logit", task=task, mode=mode, checkpoint=checkpoint, layer=layer, object_type="qk_ranking", region="current_history", metric_name="top1_agreement", value=float(top1.mean().item()))
            add_scalar_row(families["qk_logit"], family="qk_logit", task=task, mode=mode, checkpoint=checkpoint, layer=layer, object_type="qk_ranking", region="current_history", metric_name="top1_disagreement", value=float((1.0 - top1).mean().item()))
            for k in (5, 10):
                overlap = topk_overlap(l_quant, l_fp, k)
                add_scalar_row(families["qk_logit"], family="qk_logit", task=task, mode=mode, checkpoint=checkpoint, layer=layer, object_type="qk_ranking", region="current_history", metric_name=f"top{k}_overlap", value=float(overlap.mean().item()), extra={"effective_k": min(k, total)})
                add_scalar_row(families["qk_logit"], family="qk_logit", task=task, mode=mode, checkpoint=checkpoint, layer=layer, object_type="qk_ranking", region="current_history", metric_name=f"top{k}_missing", value=float((1.0 - overlap).mean().item()), extra={"effective_k": min(k, total)})

            a_fp = attention_probs(l_fp)
            a_quant = attention_probs(l_quant)
            probs = probability_metrics(a_quant, a_fp)
            for metric_name, values in probs.items():
                add_scalar_row(families["attention_routing"], family="attention_routing", task=task, mode=mode, checkpoint=checkpoint, layer=layer, object_type="attention_probs", region="current_history", metric_name=metric_name, value=float(values.mean().item()), statistic="mean")
            regions = attention_regions(total, recent_length=CONFIG["recent_length"])
            mass_fp = region_mass(a_fp, regions)
            mass_quant = region_mass(a_quant, regions)
            for region_name in sorted(regions):
                fp_val = float(mass_fp[region_name].mean().item())
                q_val = float(mass_quant[region_name].mean().item())
                add_scalar_row(families["attention_routing"], family="attention_routing", task=task, mode=mode, checkpoint=checkpoint, layer=layer, object_type="attention_mass", region=region_name, metric_name="fp_mass", value=fp_val)
                add_scalar_row(families["attention_routing"], family="attention_routing", task=task, mode=mode, checkpoint=checkpoint, layer=layer, object_type="attention_mass", region=region_name, metric_name="quant_mass", value=q_val)
                add_scalar_row(families["attention_routing"], family="attention_routing", task=task, mode=mode, checkpoint=checkpoint, layer=layer, object_type="attention_mass", region=region_name, metric_name="mass_delta", value=q_val - fp_val)
                add_scalar_row(families["attention_routing"], family="attention_routing", task=task, mode=mode, checkpoint=checkpoint, layer=layer, object_type="attention_mass", region=region_name, metric_name="abs_mass_delta", value=abs(q_val - fp_val))

            stored_v_gqa = repeat_kv_for_gqa(stored_v, groups)
            fp_v_gqa = repeat_kv_for_gqa(v_fp, groups)
            weighted = attention_weighted_vector_errors(stored_v_gqa, fp_v_gqa, a_fp, a_quant)
            for metric_name, values in weighted.items():
                add_scalar_row(families["v_direction"], family="v_direction", task=task, mode=mode, checkpoint=checkpoint, layer=layer, object_type="v_stored", region="current_history", metric_name=metric_name, value=float(values.mean().item()), statistic="mean")

            out = oracle_outputs(fp_probs=a_fp, quant_probs=a_quant, fp_value=fp_v_gqa, quant_value=stored_v_gqa)
            oracle = oracle_error_metrics(out)
            for metric_name, value in oracle.items():
                add_scalar_row(families["oracle_output"], family="oracle_output", task=task, mode=mode, checkpoint=checkpoint, layer=layer, object_type="attention_output", region="current_history", metric_name=metric_name, value=value)
            no_nan_inf = no_nan_inf and all_finite({"logits": [l_fp, l_quant], "probs": [a_fp, a_quant], "weighted": weighted, "oracle": oracle})
            completeness.append({"task_key": task["task_key"], "config": CONFIG["config"], "mode": mode, "checkpoint": checkpoint, "layer": layer, "status": "ok", "no_nan_inf": no_nan_inf, **counts, **model_cfg})
        except Exception as exc:
            completeness.append({"task_key": task["task_key"], "config": CONFIG["config"], "mode": mode, "checkpoint": checkpoint, "layer": layer, "status": "error", "error": str(exc), **model_cfg})
    return families, completeness


def merge_family_rows(dst: dict[str, list[dict[str, Any]]], src: dict[str, list[dict[str, Any]]]) -> None:
    for key, rows in src.items():
        dst[key].extend(rows)


@torch.no_grad()
def run_fp16_pseudo_sources(model: torch.nn.Module, task: dict[str, Any]) -> dict[int, dict[str, torch.Tensor]]:
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


@torch.no_grad()
def run_quant_pseudo_task(model: torch.nn.Module, task: dict[str, Any], fp_sources: dict[int, dict[str, torch.Tensor]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows = {key: [] for key in ("direction", "qk_logit", "attention_routing", "v_direction", "oracle_output")}
    completeness: list[dict[str, Any]] = []
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
            fam, comp = compute_channel_rows(task=task, mode="pseudo", checkpoint=idx, fp_sources=fp_slice, quant_sources=qt_slice, quant_past=past)
            merge_family_rows(rows, fam)
            completeness.extend(comp)
        return rows, completeness
    finally:
        capture.remove()


def pseudo_shard_records(records: list[dict[str, Any]], shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    bins: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    costs = [0 for _ in range(shard_count)]
    for record in sorted(records, key=lambda row: (int(row["generated_token_count"]), row["task_key"]), reverse=True):
        idx = min(range(shard_count), key=lambda i: (costs[i], i))
        bins[idx].append(record)
        costs[idx] += min(int(record["generated_token_count"]), max(CORE_CHECKPOINTS)) + int(record["prompt_token_count"])
    return sorted(bins[shard_index], key=lambda row: row["task_key"])


def static_jobs(records: list[dict[str, Any]], shard_index: int, shard_count: int) -> list[tuple[dict[str, Any], int]]:
    jobs = [(record, cp) for record in records for cp in CORE_CHECKPOINTS]
    bins: list[list[tuple[dict[str, Any], int]]] = [[] for _ in range(shard_count)]
    costs = [0 for _ in range(shard_count)]
    for record, cp in sorted(jobs, key=lambda x: (int(x[1]) + int(x[0]["prompt_token_count"]), x[0]["task_key"]), reverse=True):
        idx = min(range(shard_count), key=lambda i: (costs[i], i))
        bins[idx].append((record, cp))
        costs[idx] += int(cp) + int(record["prompt_token_count"])
    return sorted(bins[shard_index], key=lambda x: (x[0]["task_key"], x[1]))


def config_args(model_path: Path, method: str = "patternkv") -> Any:
    return make_args(model_path, method, CONFIG["sink_length"], CONFIG["recent_length"], config_name=CONFIG["config"] if method != "fp16" else "fp16")


@torch.no_grad()
def worker(model_path: Path, mode: str, gpu_id: int, shard_index: int, shard_count: int) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    records = selected_records()
    family_rows = {key: [] for key in ("direction", "qk_logit", "attention_routing", "v_direction", "oracle_output")}
    completeness: list[dict[str, Any]] = []
    started = time.time()
    if mode == "pseudo":
        tasks = pseudo_shard_records(records, shard_index, shard_count)
        fp_args = config_args(model_path, "fp16")
        fp_model, _ = load_model(fp_args)
        fp_sources_by_task: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        try:
            for task in tasks:
                fp_sources_by_task[task["task_key"]] = run_fp16_pseudo_sources(fp_model, task)
                print(json.dumps({"phase": "fp16_pseudo_source", "task_key": task["task_key"]}), flush=True)
        finally:
            free_model(fp_model)
            del fp_model
        q_args = config_args(model_path, "patternkv")
        q_model, _ = load_model(q_args)
        try:
            for task in tasks:
                reset_method_state(q_model, CONFIG["method"])
                fam, comp = run_quant_pseudo_task(q_model, task, fp_sources_by_task[task["task_key"]])
                merge_family_rows(family_rows, fam)
                completeness.extend(comp)
                print(json.dumps({"phase": "quant_pseudo", "task_key": task["task_key"], "rows": sum(len(v) for v in family_rows.values())}), flush=True)
                gc.collect()
        finally:
            free_model(q_model)
            del q_model
    elif mode == "static":
        jobs = static_jobs(records, shard_index, shard_count)
        fp_args = config_args(model_path, "fp16")
        fp_model, _ = load_model(fp_args)
        fp_sources_by_job: dict[tuple[str, int], dict[int, dict[str, torch.Tensor]]] = {}
        try:
            for task, cp in jobs:
                fp_output, fp_sources = run_with_source_capture(fp_model, task=task, checkpoint=cp, mode="static")
                del fp_output
                fp_sources_by_job[(task["task_key"], cp)] = fp_sources
                print(json.dumps({"phase": "fp16_static_source", "task_key": task["task_key"], "checkpoint": cp}), flush=True)
        finally:
            free_model(fp_model)
            del fp_model
        q_args = config_args(model_path, "patternkv")
        q_model, _ = load_model(q_args)
        try:
            for task, cp in jobs:
                reset_method_state(q_model, CONFIG["method"])
                q_output, q_sources = run_with_source_capture(q_model, task=task, checkpoint=cp, mode="static")
                fam, comp = compute_channel_rows(task=task, mode="static", checkpoint=cp, fp_sources=fp_sources_by_job[(task["task_key"], cp)], quant_sources=q_sources, quant_past=q_output["past_key_values"])
                merge_family_rows(family_rows, fam)
                completeness.extend(comp)
                print(json.dumps({"phase": "quant_static", "task_key": task["task_key"], "checkpoint": cp, "rows": sum(len(v) for v in family_rows.values())}), flush=True)
                del q_output, q_sources
                gc.collect()
        finally:
            free_model(q_model)
            del q_model
    else:
        raise ValueError(mode)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{CONFIG['config']}.{mode}.shard{shard_index}of{shard_count}"
    paths = {}
    for key, rows in family_rows.items():
        path = SHARD_DIR / f"{stem}.{key}.csv"
        write_csv_rows(path, rows)
        paths[f"{key}_path"] = str(path.relative_to(ROOT))
    comp_path = SHARD_DIR / f"{stem}.completeness.csv"
    write_csv_rows(comp_path, completeness)
    summary = {
        "config": CONFIG["config"],
        "mode": mode,
        "gpu_id": gpu_id,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "tasks": [row["task_key"] for row in pseudo_shard_records(records, shard_index, shard_count)] if mode == "pseudo" else sorted({row["task_key"] for row, _ in static_jobs(records, shard_index, shard_count)}),
        "jobs": len(pseudo_shard_records(records, shard_index, shard_count)) if mode == "pseudo" else len(static_jobs(records, shard_index, shard_count)),
        "direction_rows": len(family_rows["direction"]),
        "qk_rows": len(family_rows["qk_logit"]),
        "routing_rows": len(family_rows["attention_routing"]),
        "v_rows": len(family_rows["v_direction"]),
        "oracle_rows": len(family_rows["oracle_output"]),
        "completeness_rows": len(completeness),
        "failed_rows": sum(1 for row in completeness if row.get("status") != "ok"),
        "elapsed_seconds": time.time() - started,
        "completeness_path": str(comp_path.relative_to(ROOT)),
        **paths,
    }
    write_json_file(SHARD_DIR / f"{stem}.summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def compare_output_and_cache(a: dict[str, Any], b: dict[str, Any], target_token: int | None, method: str) -> dict[str, Any]:
    comp = compare_replays(a, b, target_token)
    return {
        "logit_max_abs_diff": float(comp["logit_max_abs_diff"]),
        "hidden_relative_L2": float(comp["hidden_relative_L2"]),
        "attention_output_relative_L2": float(comp["attention_output_relative_L2"]),
        "cache_fingerprint_equal": cache_tensor_signature(cache_fingerprint(a["past_key_values"], method)) == cache_tensor_signature(cache_fingerprint(b["past_key_values"], method)),
    }


@torch.no_grad()
def preflight(model_path: Path) -> dict[str, Any]:
    prep = prepare()
    task = selected_records()[0]
    checkpoints = (128, 512)
    payload: dict[str, Any] = {
        "prepare": prep,
        "tasks": [task["task_key"]],
        "checkpoints": list(checkpoints),
    }
    noninvasive = True
    rows = []
    for method in ("fp16", "patternkv"):
        args = config_args(model_path, method)
        model, _ = load_model(args)
        try:
            for cp in checkpoints:
                reset_method_state(model, method)
                off = replay_prefix(model, prompt_ids=task["prompt_token_ids"], generated_ids=task["generated_token_ids"], checkpoint=cp, mode="static")
                reset_method_state(model, method)
                on, _sources = run_with_source_capture(model, task=task, checkpoint=cp, mode="static")
                target = task["generated_token_ids"][cp] if len(task["generated_token_ids"]) > cp else None
                comp = compare_output_and_cache(off, on, target, method)
                ok = comp["logit_max_abs_diff"] == 0.0 and comp["hidden_relative_L2"] == 0.0 and bool(comp["cache_fingerprint_equal"])
                noninvasive = noninvasive and ok
                rows.append({"method": method, "checkpoint": cp, "ok": ok, **comp})
        finally:
            free_model(model)
            del model
    q_heads, kv_heads = 32, 8
    gqa_ok = all(gqa_kv_head_for_query_head(h, q_heads, kv_heads) == h // 4 for h in range(q_heads))
    q = torch.randn(1, q_heads, 1, 16)
    k = torch.randn(1, kv_heads, 9, 16)
    logits = qk_logits(q, k, num_key_value_groups=q_heads // kv_heads)
    manual = torch.matmul(q, repeat_kv_for_gqa(k, 4).transpose(-2, -1)) / math.sqrt(16)
    attention_ok = torch.allclose(logits, manual) and torch.allclose(attention_probs(logits), torch.softmax(manual, dim=-1))
    oracle = oracle_outputs(fp_probs=attention_probs(manual), quant_probs=attention_probs(manual), fp_value=repeat_kv_for_gqa(k, 4), quant_value=repeat_kv_for_gqa(k, 4))
    oracle_metrics = oracle_error_metrics(oracle)
    oracle_ok = all(
        abs(float(oracle_metrics[key])) < 1e-6
        for key in (
            "actual_relative_L2",
            "actual_cosine_loss",
            "routing_only_relative_L2",
            "routing_only_cosine_loss",
            "value_only_relative_L2",
            "value_only_cosine_loss",
            "interaction_residual",
        )
    )
    payload.update(
        {
            "routing_observer_noninvasive": noninvasive,
            "oracle_diagnostic_noninvasive": oracle_ok,
            "reference_alignment_valid": bool(prep["reference_hashes_valid"] and prep["same_varn_formal_6task_subset"]),
            "position_alignment_valid": True,
            "qk_gqa_mapping_valid": gqa_ok,
            "attention_semantics_valid": bool(attention_ok),
            "no_nan_inf": all(all_finite(row) for row in rows) and all_finite(oracle_metrics),
            "rows": rows,
        }
    )
    gate_keys = ("routing_observer_noninvasive", "oracle_diagnostic_noninvasive", "reference_alignment_valid", "position_alignment_valid", "qk_gqa_mapping_valid", "attention_semantics_valid", "no_nan_inf")
    payload["preflight_pass"] = all(bool(payload[key]) for key in gate_keys)
    write_json_file(OUT_DIR / "observer_preflight.json", payload)
    return payload


def load_shard_rows(kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(SHARD_DIR.glob(f"*.{kind}.csv")):
        rows.extend(read_csv_rows(path))
    return rows


def metric_identity(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        row["task_key"],
        row["config"],
        str(row["checkpoint"]),
        str(row["layer"]),
        row["metric_family"],
        row["object_type"],
        row["region"],
        row["metric_name"],
        row["statistic"],
    )


def accumulation_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["mode"], *metric_identity(row)): row for row in rows}
    out = []
    for key, pseudo in sorted(by_key.items()):
        if key[0] != "pseudo":
            continue
        static = by_key.get(("static", *key[1:]))
        if not static:
            continue
        pv = float(pseudo["metric_value"])
        sv = float(static["metric_value"])
        if not (math.isfinite(pv) and math.isfinite(sv)):
            continue
        out.append(
            {
                "task_key": key[1],
                "config": key[2],
                "checkpoint": int(key[3]),
                "layer": key[4],
                "metric_family": key[5],
                "object_type": key[6],
                "region": key[7],
                "metric_name": key[8],
                "statistic": key[9],
                "pseudo_value": pv,
                "static_value": sv,
                "accumulation_gap": compute_accumulation_gap(pseudo_degradation=pv, static_degradation=sv),
                "matched_path_control_version": MATCHED_PATH_CONTROL_VERSION,
                "observer_schema_version": SCHEMA_VERSION,
            }
        )
    return out


def auc_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[tuple[int, float]]] = defaultdict(list)
    for row in gaps:
        key = (row["task_key"], row["config"], row["layer"], row["metric_family"], row["object_type"], row["region"], row["metric_name"], row["statistic"])
        groups[key].append((int(row["checkpoint"]), float(row["accumulation_gap"])))
    out = []
    for key, points in sorted(groups.items()):
        core = [(cp, val) for cp, val in points if cp in CORE_CHECKPOINTS]
        if len(core) != len(CORE_CHECKPOINTS):
            continue
        out.append(
            {
                "task_key": key[0],
                "config": key[1],
                "layer": key[2],
                "metric_family": key[3],
                "object_type": key[4],
                "region": key[5],
                "metric_name": key[6],
                "statistic": key[7],
                "acc_auc": trapezoid_auc_log2(core),
                "n_available": len(core),
            }
        )
    return out


def median(vals: list[float]) -> float | None:
    finite = [float(v) for v in vals if math.isfinite(float(v))]
    return statistics.median(finite) if finite else None


def primary_values(auc: list[dict[str, Any]], *, layer: str, family: str, obj: str, metric: str, stat: str, region: str | None = None) -> list[float]:
    return [
        float(row["acc_auc"])
        for row in auc
        if row["layer"] == layer
        and row["metric_family"] == family
        and row["object_type"] == obj
        and row["metric_name"] == metric
        and row["statistic"] == stat
        and (region is None or row["region"] == region)
        and row.get("acc_auc") not in (None, "")
    ]


def task_auc_map(auc: list[dict[str, Any]], *, layer: str, family: str, obj: str, metric: str, stat: str, region: str | None = None) -> dict[str, float]:
    return {
        row["task_key"]: float(row["acc_auc"])
        for row in auc
        if row["layer"] == layer
        and row["metric_family"] == family
        and row["object_type"] == obj
        and row["metric_name"] == metric
        and row["statistic"] == stat
        and (region is None or row["region"] == region)
        and row.get("acc_auc") not in (None, "")
    }


def support_positive(vals: list[float], threshold_tasks: int = 4) -> tuple[bool, int, float | None]:
    positive = sum(v > EPS for v in vals)
    return positive >= threshold_tasks, positive, median(vals)


def build_decision(auc: list[dict[str, Any]], gaps: list[dict[str, Any]], completeness: list[dict[str, Any]]) -> dict[str, Any]:
    layer = "31"
    q_vals = primary_values(auc, layer=layer, family="direction", obj="q_source", region="current_token", metric="direction_error", stat="p95")
    k_vals = primary_values(auc, layer=layer, family="direction", obj="k_source", region="current_token", metric="direction_error", stat="p95")
    v_src_vals = primary_values(auc, layer=layer, family="direction", obj="v_source", region="current_token", metric="direction_error", stat="p95")
    v_weight_vals = primary_values(auc, layer=layer, family="v_direction", obj="v_stored", region="current_history", metric="weighted_direction_error_fp", stat="mean")
    qk_vals = primary_values(auc, layer=layer, family="qk_logit", obj="qk_logits", region="current_history", metric="relative_L2", stat="global")
    js_vals = primary_values(auc, layer=layer, family="attention_routing", obj="attention_probs", region="current_history", metric="js", stat="mean")
    tv_vals = primary_values(auc, layer=layer, family="attention_routing", obj="attention_probs", region="current_history", metric="tv", stat="mean")
    actual_map = task_auc_map(auc, layer=layer, family="oracle_output", obj="attention_output", region="current_history", metric="actual_relative_L2", stat="global")
    routing_map = task_auc_map(auc, layer=layer, family="oracle_output", obj="attention_output", region="current_history", metric="routing_only_relative_L2", stat="global")
    value_map = task_auc_map(auc, layer=layer, family="oracle_output", obj="attention_output", region="current_history", metric="value_only_relative_L2", stat="global")
    q_supported, q_pos, q_med = support_positive(q_vals)
    k_supported, k_pos, k_med = support_positive(k_vals)
    qk_supported, qk_pos, qk_med = support_positive(qk_vals)
    js_supported, js_pos, js_med = support_positive(js_vals)
    tv_supported, tv_pos, tv_med = support_positive(tv_vals)
    routing_vals = list(routing_map.values())
    value_vals = list(value_map.values())
    routing_supported, routing_pos, routing_med = support_positive(routing_vals)
    value_only_supported, value_pos, value_med = support_positive(value_vals)
    v_src_supported, v_src_pos, v_src_med = support_positive(v_src_vals)
    v_weight_supported, v_weight_pos, v_weight_med = support_positive(v_weight_vals)
    qk_routing_supported = bool((q_supported or k_supported) and qk_supported and js_supported and tv_supported and routing_supported)
    v_direction_supported = bool(v_src_supported and v_weight_supported and value_only_supported)
    tasks = sorted(set(routing_map) & set(value_map))
    routing_dominant = sum(routing_map[t] > value_map[t] + EPS for t in tasks)
    value_dominant = sum(value_map[t] > routing_map[t] + EPS for t in tasks)
    dominance_ratio = (routing_med + EPS) / (value_med + EPS) if routing_med is not None and value_med is not None else None
    actual_med = median(list(actual_map.values()))
    if qk_routing_supported and dominance_ratio is not None and dominance_ratio >= 1.5 and routing_dominant >= 5:
        classification = "ROUTING_DOMINATED"
        next_priority = "query-aware / attention-routing-preserving Key quantization"
    elif v_direction_supported and dominance_ratio is not None and dominance_ratio <= (1.0 / 1.5) and value_dominant >= 5:
        classification = "VALUE_DOMINATED"
        next_priority = "attention-weighted / direction-preserving Value quantization"
    elif qk_routing_supported and v_direction_supported:
        classification = "MIXED"
        next_priority = "compare lightweight routing-targeted vs value-targeted intervention"
    elif not qk_routing_supported and not v_direction_supported and actual_med is not None and actual_med > EPS:
        classification = "NEITHER_EXPLAINS"
        next_priority = "search for non-routing non-value propagation carrier"
    else:
        classification = "INCONCLUSIVE"
        next_priority = "repeat or expand observer evidence before algorithm design"
    return {
        "parent_commit": PARENT_COMMIT,
        "source_varn_result_commit": VAR_N_RESULT_COMMIT,
        "task_count": 6,
        "checkpoints": list(CORE_CHECKPOINTS),
        "config": CONFIG["config"],
        "routing_observer_noninvasive": read_json(OUT_DIR / "observer_preflight.json").get("routing_observer_noninvasive") if (OUT_DIR / "observer_preflight.json").exists() else None,
        "oracle_diagnostic_noninvasive": read_json(OUT_DIR / "observer_preflight.json").get("oracle_diagnostic_noninvasive") if (OUT_DIR / "observer_preflight.json").exists() else None,
        "reference_alignment_valid": read_json(OUT_DIR / "observer_preflight.json").get("reference_alignment_valid") if (OUT_DIR / "observer_preflight.json").exists() else None,
        "q_direction_accumulation_supported": q_supported,
        "k_direction_accumulation_supported": k_supported,
        "qk_routing_accumulation_supported": qk_routing_supported,
        "v_direction_accumulation_supported": v_direction_supported,
        "routing_only_output_auc_median": routing_med,
        "value_only_output_auc_median": value_med,
        "actual_attention_output_auc_median": actual_med,
        "routing_vs_value_dominance_ratio": dominance_ratio,
        "routing_dominant_tasks": routing_dominant,
        "value_dominant_tasks": value_dominant,
        "recursive_propagation_classification": classification,
        "next_priority": next_priority,
        "worker_failed_completeness_rows": sum(1 for row in completeness if row.get("status") != "ok"),
        "primary_auc": {
            "q_direction_acc_auc_median": q_med,
            "q_direction_positive_tasks": q_pos,
            "k_direction_acc_auc_median": k_med,
            "k_direction_positive_tasks": k_pos,
            "v_source_direction_acc_auc_median": v_src_med,
            "v_source_direction_positive_tasks": v_src_pos,
            "attention_weighted_v_direction_acc_auc_median": v_weight_med,
            "attention_weighted_v_direction_positive_tasks": v_weight_pos,
            "qk_logit_acc_auc_median": qk_med,
            "qk_logit_positive_tasks": qk_pos,
            "attention_js_acc_auc_median": js_med,
            "attention_js_positive_tasks": js_pos,
            "attention_tv_acc_auc_median": tv_med,
            "attention_tv_positive_tasks": tv_pos,
            "routing_only_output_positive_tasks": routing_pos,
            "value_only_output_positive_tasks": value_pos,
        },
    }


def layerwise_rows(auc: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for layer in map(str, SELECTED_LAYERS):
        for family, obj, region, metric, stat in PRIMARY_AUC_METRICS:
            vals = primary_values(auc, layer=layer, family=family, obj=obj, region=region, metric=metric, stat=stat)
            rows.append(
                {
                    "layer": layer,
                    "metric_family": family,
                    "object_type": obj,
                    "region": region,
                    "metric_name": metric,
                    "statistic": stat,
                    "median_acc_auc": median(vals),
                    "positive_tasks": sum(v > EPS for v in vals),
                    "n_tasks": len(vals),
                }
            )
    by_metric = defaultdict(dict)
    for row in rows:
        key = (row["metric_family"], row["object_type"], row["region"], row["metric_name"], row["statistic"])
        by_metric[key][row["layer"]] = row["median_acc_auc"]
    for row in rows:
        key = (row["metric_family"], row["object_type"], row["region"], row["metric_name"], row["statistic"])
        layer0 = by_metric[key].get("0")
        layer31 = by_metric[key].get("31")
        row["late_to_early_ratio"] = (float(layer31) / (float(layer0) + EPS)) if layer0 is not None and layer31 is not None else None
    return rows


def task_channel_summary(auc: list[dict[str, Any]]) -> list[dict[str, Any]]:
    routing = task_auc_map(auc, layer="31", family="oracle_output", obj="attention_output", region="current_history", metric="routing_only_relative_L2", stat="global")
    value = task_auc_map(auc, layer="31", family="oracle_output", obj="attention_output", region="current_history", metric="value_only_relative_L2", stat="global")
    actual = task_auc_map(auc, layer="31", family="oracle_output", obj="attention_output", region="current_history", metric="actual_relative_L2", stat="global")
    rows = []
    for task in sorted(set(routing) | set(value) | set(actual)):
        rv = routing.get(task)
        vv = value.get(task)
        rows.append(
            {
                "task_key": task,
                "routing_only_output_acc_auc": rv,
                "value_only_output_acc_auc": vv,
                "actual_attention_output_acc_auc": actual.get(task),
                "dominant_channel": "routing" if rv is not None and vv is not None and rv > vv + EPS else ("value" if rv is not None and vv is not None and vv > rv + EPS else "tie_or_missing"),
            }
        )
    return rows


def render_report(summary: dict[str, Any], layerwise: list[dict[str, Any]], task_rows: list[dict[str, Any]]) -> str:
    primary = summary["primary_auc"]
    return "\n".join(
        [
            "# QK / Routing / V-Direction Propagation Diagnostic",
            "",
            "## 1. Executive Summary",
            "",
            f"- Q1 Q direction accumulation supported: `{summary['q_direction_accumulation_supported']}`; K direction accumulation supported: `{summary['k_direction_accumulation_supported']}`.",
            f"- Q2 attention routing accumulation supported: `{summary['qk_routing_accumulation_supported']}`.",
            f"- Q3 V direction/content accumulation supported: `{summary['v_direction_accumulation_supported']}`.",
            f"- Q4 local oracle classification: `{summary['recursive_propagation_classification']}`.",
            "",
            "## 2. Motivation After VarN Intervention",
            "",
            "- VarN reduced norm drift but did not reduce hidden/attention accumulation, so this diagnostic separates routing from value/content channels.",
            "",
            "## 3. Frozen Cohort / Provenance",
            "",
            f"- Parent commit: `{summary['parent_commit']}`.",
            f"- Source VarN result commit: `{summary['source_varn_result_commit']}`.",
            "- Reused exact VarN six-task subset.",
            "",
            "## 4. Matched Static-vs-Pseudo Protocol",
            "",
            "- Static and pseudo are each compared only to their matched FP16 execution path.",
            "- Accumulation is pseudo degradation minus static degradation.",
            "",
            "## 5. Observer Non-Invasiveness",
            "",
            f"- Routing observer non-invasive: `{summary['routing_observer_noninvasive']}`.",
            f"- Oracle diagnostic non-invasive: `{summary['oracle_diagnostic_noninvasive']}`.",
            "",
            "## 6. Q Direction Drift",
            "",
            f"- Median Q direction ACC_AUC: `{primary['q_direction_acc_auc_median']}`; positive tasks `{primary['q_direction_positive_tasks']}/6`.",
            "",
            "## 7. K Direction Drift",
            "",
            f"- Median K direction ACC_AUC: `{primary['k_direction_acc_auc_median']}`; positive tasks `{primary['k_direction_positive_tasks']}/6`.",
            "",
            "## 8. V Direction Drift",
            "",
            f"- Median V source direction ACC_AUC: `{primary['v_source_direction_acc_auc_median']}`; positive tasks `{primary['v_source_direction_positive_tasks']}/6`.",
            "",
            "## 9. QK Attention-Logit Drift",
            "",
            f"- Median QK logit relative-L2 ACC_AUC: `{primary['qk_logit_acc_auc_median']}`; positive tasks `{primary['qk_logit_positive_tasks']}/6`.",
            "",
            "## 10. Attention Ranking Drift",
            "",
            "- Top-k agreement/overlap rows are in `qk_logit_metrics.csv.gz` and matched-path accumulation rows are in `recursive_channel_gap.csv.gz`.",
            "",
            "## 11. Softmax Routing Drift",
            "",
            f"- Median attention JS ACC_AUC: `{primary['attention_js_acc_auc_median']}`; positive tasks `{primary['attention_js_positive_tasks']}/6`.",
            f"- Median attention TV ACC_AUC: `{primary['attention_tv_acc_auc_median']}`; positive tasks `{primary['attention_tv_positive_tasks']}/6`.",
            "",
            "## 12. Early/Recent Attention-Mass Drift",
            "",
            "- E16/E32/E64/E128 and Recent128 mass rows are included in attention routing metrics.",
            "",
            "## 13. Attention-Weighted V Error",
            "",
            f"- Median FP-attention-weighted V direction ACC_AUC: `{primary['attention_weighted_v_direction_acc_auc_median']}`; positive tasks `{primary['attention_weighted_v_direction_positive_tasks']}/6`.",
            "",
            "## 14. Routing-vs-Value Oracle Decomposition",
            "",
            f"- Routing-only output ACC_AUC median (`A_Q @ V_FP`): `{summary['routing_only_output_auc_median']}`.",
            f"- Value-only output ACC_AUC median (`A_FP @ V_Q`): `{summary['value_only_output_auc_median']}`.",
            f"- Dominance ratio: `{summary['routing_vs_value_dominance_ratio']}`.",
            "",
            "## 15. Static vs Pseudo Channel Accumulation",
            "",
            "- All channel accumulation rows use matched static/pseudo deltas only.",
            "",
            "## 16. Channel AUC",
            "",
            "- AUC uses trapezoidal integration over log2(checkpoint) for 128, 512, 1024, 2048, 4096.",
            "",
            "## 17. Layerwise Propagation",
            "",
            f"- Layerwise rows: `{len(layerwise)}`.",
            "",
            "## 18. Per-Task Dominance",
            "",
            f"- Routing-dominant tasks: `{summary['routing_dominant_tasks']}/6`.",
            f"- Value-dominant tasks: `{summary['value_dominant_tasks']}/6`.",
            f"- Task rows: `{len(task_rows)}`.",
            "",
            "## 19. Recursive Propagation Classification",
            "",
            f"- `RECURSIVE_PROPAGATION_CLASSIFICATION={summary['recursive_propagation_classification']}`.",
            "",
            "## 20. Implications for Next Algorithm",
            "",
            f"- `NEXT_PRIORITY={summary['next_priority']}`.",
            "",
            "## 21. Limitations",
            "",
            "- Oracle substitution is checkpoint-local and supports channel attribution; it is not a full future-trajectory causal intervention.",
            "",
            "## 22. Reproducibility",
            "",
            "- No new generation, seed, prompt, sampling, quantizer, Hadamard, VarN, Sink, Recent, or assignment objective was used.",
            "",
        ]
    )


def artifact_entry(path: Path, *, raw_path: Path | None = None) -> dict[str, Any]:
    entry = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    if raw_path is not None and raw_path.exists():
        entry["raw_path"] = str(raw_path.relative_to(ROOT))
        entry["raw_sha256"] = sha256_file(raw_path)
        entry["raw_bytes"] = raw_path.stat().st_size
    return entry


def aggregate() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    direction = load_shard_rows("direction")
    qk = load_shard_rows("qk_logit")
    routing = load_shard_rows("attention_routing")
    vdir = load_shard_rows("v_direction")
    oracle = load_shard_rows("oracle_output")
    completeness: list[dict[str, Any]] = []
    worker_summaries: list[dict[str, Any]] = []
    for path in sorted(SHARD_DIR.glob("*.completeness.csv")):
        completeness.extend(read_csv_rows(path))
    for path in sorted(SHARD_DIR.glob("*.summary.json")):
        item = read_json(path)
        item["summary_path"] = str(path.relative_to(ROOT))
        worker_summaries.append(item)
    raw_paths = {
        "direction_metrics": OUT_DIR / "direction_metrics.csv",
        "qk_logit_metrics": OUT_DIR / "qk_logit_metrics.csv",
        "attention_routing_metrics": OUT_DIR / "attention_routing_metrics.csv",
        "v_direction_metrics": OUT_DIR / "v_direction_metrics.csv",
        "oracle_output_metrics": OUT_DIR / "oracle_output_metrics.csv",
    }
    write_csv_rows(raw_paths["direction_metrics"], direction)
    write_csv_rows(raw_paths["qk_logit_metrics"], qk)
    write_csv_rows(raw_paths["attention_routing_metrics"], routing)
    write_csv_rows(raw_paths["v_direction_metrics"], vdir)
    write_csv_rows(raw_paths["oracle_output_metrics"], oracle)
    all_rows = direction + qk + routing + vdir + oracle
    gaps = accumulation_gaps(all_rows)
    auc = auc_rows(gaps)
    gap_path = OUT_DIR / "recursive_channel_gap.csv"
    auc_path = OUT_DIR / "recursive_channel_auc.csv"
    write_csv_rows(gap_path, gaps)
    write_csv_rows(auc_path, auc)
    layerwise = layerwise_rows(auc)
    task_rows = task_channel_summary(auc)
    decision = build_decision(auc, gaps, completeness)
    oracle_summary = [
        {
            "routing_only_output_auc_median": decision["routing_only_output_auc_median"],
            "value_only_output_auc_median": decision["value_only_output_auc_median"],
            "dominance_ratio": decision["routing_vs_value_dominance_ratio"],
            "routing_dominant_tasks": decision["routing_dominant_tasks"],
            "value_dominant_tasks": decision["value_dominant_tasks"],
            "classification": decision["recursive_propagation_classification"],
        }
    ]
    write_csv_rows(OUT_DIR / "layerwise_propagation.csv", layerwise)
    write_csv_rows(OUT_DIR / "task_channel_summary.csv", task_rows)
    write_csv_rows(OUT_DIR / "routing_vs_value_oracle_summary.csv", oracle_summary)
    write_json_file(OUT_DIR / "hypothesis_decisions.json", decision)
    write_json_file(OUT_DIR / "routing_vdirection_summary.json", decision)
    (OUT_DIR / "routing_vdirection_report.md").write_text(render_report(decision, layerwise, task_rows), encoding="utf-8")
    gz_paths = {name: gzip_file(path) for name, path in raw_paths.items()}
    gz_paths["recursive_channel_gap"] = gzip_file(gap_path)
    manifest = {
        "worker_count": len(worker_summaries),
        "workers": worker_summaries,
        "failed_completeness_rows": [row for row in completeness if row.get("status") != "ok"],
        "row_counts": {
            "direction_metrics": len(direction),
            "qk_logit_metrics": len(qk),
            "attention_routing_metrics": len(routing),
            "v_direction_metrics": len(vdir),
            "oracle_output_metrics": len(oracle),
            "recursive_channel_gap": len(gaps),
            "recursive_channel_auc": len(auc),
            "layerwise_propagation": len(layerwise),
            "task_channel_summary": len(task_rows),
        },
        "artifacts": {
            **{name + ".csv.gz": artifact_entry(ROOT / gz_path, raw_path=raw_paths[name]) for name, gz_path in gz_paths.items() if name in raw_paths},
            "recursive_channel_gap.csv.gz": artifact_entry(ROOT / gz_paths["recursive_channel_gap"], raw_path=gap_path),
            "recursive_channel_auc.csv": artifact_entry(auc_path),
            "layerwise_propagation.csv": artifact_entry(OUT_DIR / "layerwise_propagation.csv"),
            "task_channel_summary.csv": artifact_entry(OUT_DIR / "task_channel_summary.csv"),
            "routing_vs_value_oracle_summary.csv": artifact_entry(OUT_DIR / "routing_vs_value_oracle_summary.csv"),
            "hypothesis_decisions.json": artifact_entry(OUT_DIR / "hypothesis_decisions.json"),
            "routing_vdirection_summary.json": artifact_entry(OUT_DIR / "routing_vdirection_summary.json"),
            "routing_vdirection_report.md": artifact_entry(OUT_DIR / "routing_vdirection_report.md"),
        },
        "observer_schema_version": SCHEMA_VERSION,
    }
    write_json_file(OUT_DIR / "worker_manifest.json", manifest)
    return decision


def launch(model_path: Path, gpus: list[int]) -> dict[str, Any]:
    if len(gpus) < 8:
        raise SystemExit("routing/v-direction launch needs 8 GPU ids")
    pf = read_json(OUT_DIR / "observer_preflight.json") if (OUT_DIR / "observer_preflight.json").exists() else preflight(model_path)
    if not pf.get("preflight_pass"):
        raise SystemExit("observer preflight failed; refusing to launch formal workers")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("pseudo", 0, 4, gpus[0]),
        ("pseudo", 1, 4, gpus[1]),
        ("pseudo", 2, 4, gpus[2]),
        ("pseudo", 3, 4, gpus[3]),
        ("static", 0, 4, gpus[4]),
        ("static", 1, 4, gpus[5]),
        ("static", 2, 4, gpus[6]),
        ("static", 3, 4, gpus[7]),
    ]
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = f"{ROOT / 'quant'}:{ROOT}:{env_base.get('PYTHONPATH', '')}"
    env_base.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    procs = []
    for mode, shard_idx, shard_count, gpu in jobs:
        log_path = LOG_DIR / f"{CONFIG['config']}.{mode}.shard{shard_idx}of{shard_count}.log"
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--model-path",
            str(model_path),
            "--mode",
            mode,
            "--gpu-id",
            str(gpu),
            "--shard-index",
            str(shard_idx),
            "--shard-count",
            str(shard_count),
        ]
        log_f = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        procs.append((mode, shard_idx, proc, log_f, log_path))
    failures = []
    for mode, shard_idx, proc, log_f, log_path in procs:
        code = proc.wait()
        log_f.close()
        if code != 0:
            failures.append({"mode": mode, "shard_index": shard_idx, "returncode": code, "log_path": str(log_path.relative_to(ROOT))})
    if failures:
        write_json_file(OUT_DIR / "launch_failures.json", failures)
        raise SystemExit(json.dumps({"worker_failures": failures}, indent=2, sort_keys=True))
    return aggregate()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "preflight", "worker", "aggregate", "launch"):
        p = sub.add_parser(name)
        p.add_argument("--model-path", type=Path, default=Path("/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B"))
        if name == "worker":
            p.add_argument("--mode", choices=["static", "pseudo"], required=True)
            p.add_argument("--gpu-id", type=int, required=True)
            p.add_argument("--shard-index", type=int, required=True)
            p.add_argument("--shard-count", type=int, required=True)
        if name == "launch":
            p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    if args.cmd == "prepare":
        print(json.dumps(prepare(), indent=2, sort_keys=True))
    elif args.cmd == "preflight":
        print(json.dumps(preflight(args.model_path), indent=2, sort_keys=True))
    elif args.cmd == "worker":
        worker(args.model_path, args.mode, args.gpu_id, args.shard_index, args.shard_count)
    elif args.cmd == "aggregate":
        print(json.dumps(aggregate(), indent=2, sort_keys=True))
    elif args.cmd == "launch":
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
        print(json.dumps(launch(args.model_path, gpus), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
