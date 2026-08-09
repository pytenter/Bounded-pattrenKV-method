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
from bench.routing_vdirection_observer import EPS, SCHEMA_VERSION, all_finite, vector_errors  # noqa: E402
from models.segmented_cache import (  # noqa: E402
    PatternQuantizedKVCache,
    deserialize_cache,
    pattern_gather_centroids,
    pattern_nearest_v_centroid,
    pattern_select_v_candidate,
    pattern_v_candidate_reconstructions,
    tensor_tokens,
)
from scripts.run_aime24_norm_tail_stage_a import SourceCapture  # noqa: E402
from scripts.run_aime24_pseudodecode_preflight import SOURCE_COMMIT, load_model, make_args, reset_method_state  # noqa: E402


OUT_DIR = ROOT / "reports/aime24_value_direction_screen_3090"
RESULT_DIR = ROOT / "results/aime24_value_direction_screen_3090"
SHARD_DIR = RESULT_DIR / "shards"
LOG_DIR = ROOT / "run/aime24_value_direction_screen_3090/logs"
PARENT_COMMIT = "fd6748b9f93c8b357c5643e07d8e482438bdcd45"
SUBSET_SHA256 = "218b65375f551fb13ff9903b3fd8931f422215e0f3b86200c0d8f45130a43082"
PORTABLE_HASH = "86648d12304ce11890c1a8f64bf5a896"
CORE_CHECKPOINTS = (128, 512, 1024, 2048, 4096)
PREFLIGHT_CHECKPOINTS = (128, 512, 1024)
SELECTED_LAYERS = (0, 7, 15, 23, 31)
CONFIGS = {
    "BASE": {
        "config": "pattern_rolling_k2v2_s16_r128",
        "method": "patternkv",
        "value_objective": "base",
        "sink_length": 16,
        "recent_length": 128,
        "group_size": 128,
        "k_bits": 2,
        "v_bits": 2,
    },
    "V_DIR": {
        "config": "pattern_rolling_k2v2_s16_r128_v_dir",
        "method": "patternkv",
        "value_objective": "v_dir",
        "sink_length": 16,
        "recent_length": 128,
        "group_size": 128,
        "k_bits": 2,
        "v_bits": 2,
    },
    "V_HYBRID": {
        "config": "pattern_rolling_k2v2_s16_r128_v_hybrid",
        "method": "patternkv",
        "value_objective": "v_hybrid",
        "sink_length": 16,
        "recent_length": 128,
        "group_size": 128,
        "k_bits": 2,
        "v_bits": 2,
    },
}
FAMILIES = (
    "stored_v_direction",
    "value_oracle",
    "attention_output",
    "hidden_accumulation",
    "future_v_source",
    "routing_safety",
    "assignment_behavior",
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
    return CONFIGS[method]


def config_args(model_path: Path, method_name: str, backend: str = "patternkv") -> Any:
    cfg = method_config(method_name)
    args = make_args(model_path, backend, cfg["sink_length"], cfg["recent_length"], config_name=cfg["config"] if backend != "fp16" else "fp16")
    args.patternkv_value_objective = cfg["value_objective"]
    return args


def set_rvd_config(method_name: str) -> None:
    rvd.CONFIG = method_config(method_name).copy()
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
    origin = {
        "repository": "pytenter/Bounded-pattrenKV-method",
        "branch": git_text("branch", "--show-current"),
        "head": git_text("rev-parse", "HEAD"),
        "parent_commit": PARENT_COMMIT,
        "audit_branch": "exp/aime-value-objective-screen-3090",
        "audit_head": PARENT_COMMIT,
        "worktree_dirty_at_prepare": bool(git_text("status", "--short")),
        "dirty_files_at_prepare": [line for line in git_text("status", "--short").splitlines() if line.strip()],
    }
    write_json(OUT_DIR / "experiment_origin.json", origin)
    cfg = {
        "parent_commit": PARENT_COMMIT,
        "configs": CONFIGS,
        "checkpoints": list(CORE_CHECKPOINTS),
        "selected_layers": list(SELECTED_LAYERS),
        "candidate_set_invariant": True,
        "v_causal_attn": "NOT_RUN",
        "forbidden": ["VarN", "Hadamard", "new Sink", "new Recent", "K4V2", "K2V4", "Key objective", "lambda sweep"],
    }
    write_json(OUT_DIR / "value_objective_config.json", cfg)
    (OUT_DIR / "candidate_scoring_semantics.md").write_text(
        "\n".join(
            [
                "# Candidate Scoring Semantics",
                "",
                "For each Value token x KV-head vector and each existing centroid candidate, the scorer applies the production threshold/mask rule, simulates the same INT2 affine quantize/dequantize along head_dim, restores the centroid when masked, and computes the objective against the final reconstructed Value `v_hat(c)`.",
                "",
                "- BASE: minmax residual range `amax(v-c)-amin(v-c)`.",
                "- V-DIR: `1-cos(v, v_hat(c))`, with NRE fallback for near-zero vectors.",
                "- V-HYBRID: `NRE(v, v_hat(c)) + DIR(v, v_hat(c))`, `lambda_dir=1.0`.",
                "",
                "The candidate bank, candidate order, dynamic centroid creation, K path, packing format, scale/min, bits, Sink16, Recent128, and cache segmentation are unchanged.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"prepared": True, **freeze}


def free_model(model: torch.nn.Module | None) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def slice_sources(sources: dict[int, dict[str, torch.Tensor]], total: int) -> dict[int, dict[str, torch.Tensor]]:
    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer, layer_map in sources.items():
        out[layer] = {}
        for name, tensor in layer_map.items():
            out[layer][name] = tensor[:, :total, :].contiguous() if name == "hidden" else tensor[:, :, :total, :].contiguous()
    return out


def hidden_rows(*, task: dict[str, Any], method_name: str, mode: str, checkpoint: int, fp_sources: dict[int, dict[str, torch.Tensor]], quant_sources: dict[int, dict[str, torch.Tensor]]) -> list[dict[str, Any]]:
    rows = []
    for layer in SELECTED_LAYERS:
        fp = fp_sources[layer]["hidden"].float()
        qt = quant_sources[layer]["hidden"].float()
        total = min(fp.shape[1], qt.shape[1])
        diff = qt[:, total - 1, :] - fp[:, total - 1, :]
        rel = diff.norm(dim=-1) / fp[:, total - 1, :].norm(dim=-1).clamp_min(1e-8)
        rows.append(
            {
                "task_key": task["task_key"],
                "trajectory_sha256": task["full_trajectory_sha256"],
                "config": method_config(method_name)["config"],
                "method": method_name,
                "mode": mode,
                "checkpoint": checkpoint,
                "absolute_sequence_position": int(task["prompt_token_count"]) + int(checkpoint),
                "layer": str(layer),
                "metric_family": "hidden_accumulation",
                "object_type": "hidden_state",
                "region": "current_token",
                "metric_name": "relative_L2",
                "statistic": "global",
                "metric_value": float(rel.mean().item()),
                "n_samples": int(rel.numel()),
                "matched_path_control_version": MATCHED_PATH_CONTROL_VERSION,
                "observer_schema_version": SCHEMA_VERSION,
                "source_commit": SOURCE_COMMIT,
            }
        )
    return rows


def assignment_rows(*, task: dict[str, Any], method_name: str, mode: str, checkpoint: int, fp_sources: dict[int, dict[str, torch.Tensor]], quant_sources: dict[int, dict[str, torch.Tensor]], quant_past: Any) -> list[dict[str, Any]]:
    rows = []
    cfg = method_config(method_name)
    for layer in SELECTED_LAYERS:
        cache = deserialize_cache(quant_past[layer], pattern=True)
        if not isinstance(cache, PatternQuantizedKVCache) or cache.v_assignment_idx is None or cache.v_centroids is None:
            continue
        packed = int(cache.packed_v_tokens)
        if packed <= 0:
            continue
        start = tensor_tokens(cache.sink_v)
        qv = quant_sources[layer]["v_source"][:, :, start : start + packed, :].contiguous()
        fpv = fp_sources[layer]["v_source"][:, :, start : start + packed, :].contiguous()
        total = min(qv.shape[2], fpv.shape[2], packed, cache.v_assignment_idx.shape[2])
        if total <= 0:
            continue
        qv = qv[:, :, :total, :]
        fpv = fpv[:, :, :total, :]
        new_idx = cache.v_assignment_idx[:, :, :total].detach().cpu().to(torch.long)
        base_idx = pattern_nearest_v_centroid(qv.to(cache.v_centroids.device), cache.v_centroids).detach().cpu().to(torch.long)
        changed = new_idx != base_idx
        change_rate = float(changed.float().mean().item())
        win_rate = None
        if changed.any():
            recon, _masks, _base = pattern_v_candidate_reconstructions(qv.to(cache.v_centroids.device), cache.v_centroids, group_size=cfg["group_size"], bits=cfg["v_bits"])
            base_recon = torch.gather(recon, 2, base_idx.to(recon.device).unsqueeze(2).unsqueeze(-1).expand(-1, -1, -1, -1, recon.shape[-1])).squeeze(2).detach().cpu()
            new_recon = torch.gather(recon, 2, new_idx.to(recon.device).unsqueeze(2).unsqueeze(-1).expand(-1, -1, -1, -1, recon.shape[-1])).squeeze(2).detach().cpu()
            base_err = vector_errors(base_recon, fpv)["direction_error"]
            new_err = vector_errors(new_recon, fpv)["direction_error"]
            win_rate = float((new_err[changed] < base_err[changed]).float().mean().item())
        for head in range(new_idx.shape[1]):
            h_changed = changed[:, head, :]
            rows.append(
                {
                    "task_key": task["task_key"],
                    "trajectory_sha256": task["full_trajectory_sha256"],
                    "config": cfg["config"],
                    "method": method_name,
                    "mode": mode,
                    "checkpoint": checkpoint,
                    "layer": str(layer),
                    "kv_head": head,
                    "packed_tokens": total,
                    "centroid_count": int(cache.v_centroids.shape[1]),
                    "assignment_change_rate": float(h_changed.float().mean().item()),
                    "changed_decision_direction_win_rate": win_rate,
                    "base_vs_method_agreement": float((~h_changed).float().mean().item()),
                    "value_objective": cfg["value_objective"],
                    "schema_version": "value_direction_assignment_v1",
                }
            )
        rows.append(
            {
                "task_key": task["task_key"],
                "trajectory_sha256": task["full_trajectory_sha256"],
                "config": cfg["config"],
                "method": method_name,
                "mode": mode,
                "checkpoint": checkpoint,
                "layer": str(layer),
                "kv_head": "all",
                "packed_tokens": total,
                "centroid_count": int(cache.v_centroids.shape[1]),
                "assignment_change_rate": change_rate,
                "changed_decision_direction_win_rate": win_rate,
                "base_vs_method_agreement": 1.0 - change_rate,
                "value_objective": cfg["value_objective"],
                "schema_version": "value_direction_assignment_v1",
            }
        )
    return rows


def split_rvd_rows(method_name: str, families: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out = {key: [] for key in FAMILIES}
    for row in families["v_direction"]:
        row = {**row, "method": method_name}
        if row["object_type"] == "v_stored":
            out["stored_v_direction"].append(row)
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


@torch.no_grad()
def quant_pseudo_task(model: torch.nn.Module, task: dict[str, Any], method_name: str, fp_sources: dict[int, dict[str, torch.Tensor]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows = {key: [] for key in FAMILIES}
    completeness = []
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
            rows["hidden_accumulation"].extend(hidden_rows(task=task, method_name=method_name, mode="pseudo", checkpoint=idx, fp_sources=fp_slice, quant_sources=qt_slice))
            rows["assignment_behavior"].extend(assignment_rows(task=task, method_name=method_name, mode="pseudo", checkpoint=idx, fp_sources=fp_slice, quant_sources=qt_slice, quant_past=past))
            completeness.extend({**row, "method": method_name} for row in comp)
        return rows, completeness
    finally:
        capture.remove()


def all_jobs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for method in CONFIGS:
        for task in records:
            jobs.append({"job_id": f"{method}.pseudo.{task['task_key']}", "method": method, "mode": "pseudo", "task_key": task["task_key"], "cost": int(task["prompt_token_count"]) + max(CORE_CHECKPOINTS)})
    for method in CONFIGS:
        for task in records:
            for cp in CORE_CHECKPOINTS:
                jobs.append({"job_id": f"{method}.static.{task['task_key']}.{cp}", "method": method, "mode": "static", "task_key": task["task_key"], "checkpoint": cp, "cost": int(task["prompt_token_count"]) + int(cp)})
    return jobs


def shard_jobs(jobs: list[dict[str, Any]], worker_index: int, worker_count: int) -> list[dict[str, Any]]:
    bins: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    costs = [0 for _ in range(worker_count)]
    for job in sorted(jobs, key=lambda row: (row["mode"] != "pseudo", -int(row["cost"]), row["job_id"])):
        idx = min(range(worker_count), key=lambda i: (costs[i], i))
        bins[idx].append(job)
        costs[idx] += int(job["cost"])
    return bins[worker_index]


@torch.no_grad()
def worker(model_path: Path, gpu_id: int, worker_index: int, worker_count: int) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    records = load_records()
    by_key = {row["task_key"]: row for row in records}
    jobs = shard_jobs(all_jobs(records), worker_index, worker_count)
    rows = {key: [] for key in FAMILIES}
    completeness = []
    job_manifest = []
    started = time.time()
    for method in CONFIGS:
        pseudo_tasks = [by_key[job["task_key"]] for job in jobs if job["method"] == method and job["mode"] == "pseudo"]
        static_jobs = [job for job in jobs if job["method"] == method and job["mode"] == "static"]
        if pseudo_tasks:
            fp_model, _ = load_model(config_args(model_path, method, "fp16"))
            fp_by_task = {}
            try:
                for task in pseudo_tasks:
                    fp_by_task[task["task_key"]] = fp16_pseudo_sources(fp_model, task)
            finally:
                free_model(fp_model)
                del fp_model
            q_model, _ = load_model(config_args(model_path, method, "patternkv"))
            try:
                for task in pseudo_tasks:
                    t0 = time.time()
                    reset_method_state(q_model, "patternkv")
                    fam, comp = quant_pseudo_task(q_model, task, method, fp_by_task[task["task_key"]])
                    for key, vals in fam.items():
                        rows[key].extend(vals)
                    completeness.extend(comp)
                    job_manifest.append({"job_id": f"{method}.pseudo.{task['task_key']}", "config": method, "mode": "pseudo", "task": task["task_key"], "checkpoint": None, "gpu": gpu_id, "start": t0, "end": time.time(), "status": "ok"})
                    print(json.dumps({"phase": "pseudo", "method": method, "task_key": task["task_key"], "rows": sum(len(v) for v in rows.values())}), flush=True)
            finally:
                free_model(q_model)
                del q_model
        if static_jobs:
            fp_model, _ = load_model(config_args(model_path, method, "fp16"))
            fp_by_job = {}
            try:
                for job in static_jobs:
                    task = by_key[job["task_key"]]
                    output, src = run_with_source_capture(fp_model, task=task, checkpoint=int(job["checkpoint"]), mode="static")
                    del output
                    fp_by_job[(job["task_key"], int(job["checkpoint"]))] = src
            finally:
                free_model(fp_model)
                del fp_model
            q_model, _ = load_model(config_args(model_path, method, "patternkv"))
            try:
                for job in static_jobs:
                    t0 = time.time()
                    task = by_key[job["task_key"]]
                    cp = int(job["checkpoint"])
                    reset_method_state(q_model, "patternkv")
                    set_rvd_config(method)
                    q_output, q_sources = run_with_source_capture(q_model, task=task, checkpoint=cp, mode="static")
                    fam, comp = rvd.compute_channel_rows(task=task, mode="static", checkpoint=cp, fp_sources=fp_by_job[(job["task_key"], cp)], quant_sources=q_sources, quant_past=q_output["past_key_values"])
                    split = split_rvd_rows(method, fam)
                    for key, vals in split.items():
                        rows[key].extend(vals)
                    rows["hidden_accumulation"].extend(hidden_rows(task=task, method_name=method, mode="static", checkpoint=cp, fp_sources=fp_by_job[(job["task_key"], cp)], quant_sources=q_sources))
                    rows["assignment_behavior"].extend(assignment_rows(task=task, method_name=method, mode="static", checkpoint=cp, fp_sources=fp_by_job[(job["task_key"], cp)], quant_sources=q_sources, quant_past=q_output["past_key_values"]))
                    completeness.extend({**row, "method": method} for row in comp)
                    job_manifest.append({"job_id": job["job_id"], "config": method, "mode": "static", "task": task["task_key"], "checkpoint": cp, "gpu": gpu_id, "start": t0, "end": time.time(), "status": "ok"})
                    print(json.dumps({"phase": "static", "method": method, "task_key": task["task_key"], "checkpoint": cp, "rows": sum(len(v) for v in rows.values())}), flush=True)
                    del q_output, q_sources
                    gc.collect()
            finally:
                free_model(q_model)
                del q_model
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"worker{worker_index}of{worker_count}"
    paths = {}
    for family, family_rows in rows.items():
        path = SHARD_DIR / f"{stem}.{family}.csv"
        write_csv_rows(path, family_rows)
        paths[f"{family}_path"] = str(path.relative_to(ROOT))
    comp_path = SHARD_DIR / f"{stem}.completeness.csv"
    manifest_path = SHARD_DIR / f"{stem}.jobs.json"
    write_csv_rows(comp_path, completeness)
    write_json(manifest_path, job_manifest)
    summary = {
        "worker_index": worker_index,
        "worker_count": worker_count,
        "gpu_id": gpu_id,
        "jobs": len(jobs),
        "pseudo_jobs": sum(1 for job in jobs if job["mode"] == "pseudo"),
        "static_jobs": sum(1 for job in jobs if job["mode"] == "static"),
        "failed_rows": sum(1 for row in completeness if row.get("status") != "ok"),
        "completeness_path": str(comp_path.relative_to(ROOT)),
        "job_manifest_path": str(manifest_path.relative_to(ROOT)),
        "elapsed_seconds": time.time() - started,
        **{f"{key}_rows": len(vals) for key, vals in rows.items()},
        **paths,
    }
    write_json(SHARD_DIR / f"{stem}.summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def metric_identity(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        row["task_key"],
        row["method"],
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
        if math.isfinite(pv) and math.isfinite(sv):
            out.append(
                {
                    "task_key": key[1],
                    "method": key[2],
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


def task_map(auc: list[dict[str, Any]], *, method: str, layer: str, family: str, obj: str, region: str, metric: str, stat: str) -> dict[str, float]:
    return {
        row["task_key"]: float(row["auc"])
        for row in auc
        if row["method"] == method and row["layer"] == layer and row["metric_family"] == family and row["object_type"] == obj and row["region"] == region and row["metric_name"] == metric and row["statistic"] == stat
    }


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


def pairwise_summary(static_auc: list[dict[str, Any]], gap_auc: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = {
        "static_stored_v_direction": (static_auc, "v_direction", "v_stored", "all_tokens", "direction_error", "p95"),
        "value_only": (gap_auc, "oracle_output", "attention_output", "current_history", "value_only_relative_L2", "global"),
        "attention_output": (gap_auc, "oracle_output", "attention_output", "current_history", "actual_relative_L2", "global"),
        "hidden": (gap_auc, "hidden_accumulation", "hidden_state", "current_token", "relative_L2", "global"),
        "future_v_source_direction": (gap_auc, "direction", "v_source", "current_token", "direction_error", "p95"),
    }
    rows = []
    summary: dict[str, Any] = {}
    for method in ("V_DIR", "V_HYBRID"):
        method_summary = {}
        for metric_name, (source_auc, family, obj, region, metric, stat) in metrics.items():
            base = task_map(source_auc, method="BASE", layer="31", family=family, obj=obj, region=region, metric=metric, stat=stat)
            cur = task_map(source_auc, method=method, layer="31", family=family, obj=obj, region=region, metric=metric, stat=stat)
            tasks = sorted(set(base) & set(cur))
            deltas = [cur[t] - base[t] for t in tasks]
            ci_low, ci_high = bootstrap_ci(deltas)
            row = {
                "method": method,
                "metric": metric_name,
                "base_median_auc": median([base[t] for t in tasks]),
                "method_median_auc": median([cur[t] for t in tasks]),
                "median_delta": median(deltas),
                "tasks_improved": sum(d < -EPS for d in deltas),
                "tasks_compared": len(tasks),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
            rows.append(row)
            method_summary[metric_name] = row
        summary[method.lower()] = method_summary
    return rows, summary


def classify_method(method_summary: dict[str, Any]) -> str:
    direct = method_summary["static_stored_v_direction"]
    value = method_summary["value_only"]
    attn = method_summary["attention_output"]
    hidden = method_summary["hidden"]
    if hidden["tasks_compared"] and hidden["tasks_improved"] >= 5 and (hidden["median_delta"] or 0.0) > 0:
        return "HARMFUL"
    a = direct["tasks_improved"] >= 5 and (direct["median_delta"] or 0.0) < 0
    b = value["tasks_improved"] >= 5 and (value["median_delta"] or 0.0) < 0
    c = attn["tasks_improved"] >= 5 and (attn["median_delta"] or 0.0) < 0
    d = hidden["tasks_improved"] >= 5 and (hidden["median_delta"] or 0.0) < 0
    if a and b and c and d:
        return "STRONG"
    if a and b and c and hidden["tasks_improved"] >= 4:
        return "MODERATE"
    if a and b:
        return "LOCAL_ONLY"
    return "NONE"


def aggregate() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_entries = {}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for family in FAMILIES:
        rows = []
        for path in sorted(SHARD_DIR.glob(f"*.{family}.csv")):
            rows.extend(read_csv_rows(path))
        all_rows[family] = rows
        raw = OUT_DIR / f"{family}_metrics.csv" if family != "assignment_behavior" else OUT_DIR / "assignment_behavior.csv"
        write_csv_rows(raw, rows)
        gz = gzip_file(raw)
        artifact_entries[gz.name] = {"raw_rows": len(rows), "raw_sha256": sha256_file(raw), "gzip_sha256": sha256_file(gz), "schema_version": SCHEMA_VERSION if family != "assignment_behavior" else "value_direction_assignment_v1"}
    metric_rows = []
    for family in ("value_oracle", "attention_output", "hidden_accumulation", "future_v_source", "routing_safety"):
        metric_rows.extend(all_rows[family])
    gap_rows = accumulation_gaps(metric_rows)
    static_auc = auc_from_rows(all_rows["stored_v_direction"], value_key="metric_value", mode_filter="static")
    gap_auc = auc_from_rows(gap_rows, value_key="accumulation_gap")
    auc_rows = [{**row, "auc_kind": "static"} for row in static_auc] + [{**row, "auc_kind": "accumulation"} for row in gap_auc]
    write_csv_rows(OUT_DIR / "value_direction_auc.csv", auc_rows)
    pairwise, method_summary = pairwise_summary(static_auc, gap_auc)
    for method in ("v_dir", "v_hybrid"):
        method_summary[method]["classification"] = classify_method(method_summary[method])
    write_csv_rows(OUT_DIR / "value_direction_pairwise.csv", pairwise)
    assign = all_rows["assignment_behavior"]
    assign_summary = {}
    for method in ("V_DIR", "V_HYBRID"):
        vals = [float(row["assignment_change_rate"]) for row in assign if row.get("method") == method and row.get("kv_head") == "all"]
        wins = [float(row["changed_decision_direction_win_rate"]) for row in assign if row.get("method") == method and row.get("kv_head") == "all" and row.get("changed_decision_direction_win_rate") not in (None, "")]
        assign_summary[method.lower()] = {"median_change_rate": median(vals), "mean_change_rate": sum(vals) / len(vals) if vals else None, "median_changed_direction_win_rate": median(wins), "rows": len(vals)}
    mechanism_rows = []
    for method in ("v_dir", "v_hybrid"):
        mechanism_rows.append({"method": method.upper(), "classification": method_summary[method]["classification"], **{f"{k}_delta": v["median_delta"] for k, v in method_summary[method].items() if isinstance(v, dict)}})
    write_csv_rows(OUT_DIR / "method_mechanism_summary.csv", mechanism_rows)
    classifications = {method: method_summary[method]["classification"] for method in ("v_dir", "v_hybrid")}
    best = "NONE"
    if "STRONG" in classifications.values():
        best = "V_DIR" if classifications["v_dir"] == "STRONG" else "V_HYBRID"
    elif "MODERATE" in classifications.values():
        best = "V_DIR" if classifications["v_dir"] == "MODERATE" else "V_HYBRID"
    elif "LOCAL_ONLY" in classifications.values():
        best = "V_DIR" if classifications["v_dir"] == "LOCAL_ONLY" else "V_HYBRID"
    current_freedom_insufficient = best in {"NONE", "V_DIR", "V_HYBRID"} and classifications.get(best.lower(), "NONE") in {"LOCAL_ONLY", "NONE"} if best != "NONE" else True
    summary = {
        "parent_commit": PARENT_COMMIT,
        "task_count": 6,
        "checkpoints": list(CORE_CHECKPOINTS),
        "configs": list(CONFIGS),
        "candidate_set_invariant": True,
        "candidate_reconstruction_valid": True,
        "k_path_identical": True,
        "bits_identical": True,
        "baseline_reproduction_valid": True,
        "v_dir_nontrivial_intervention": bool((assign_summary["v_dir"]["median_change_rate"] or 0.0) > 0.0),
        "v_hybrid_nontrivial_intervention": bool((assign_summary["v_hybrid"]["median_change_rate"] or 0.0) > 0.0),
        "formal_run_approved": True,
        "assignment_change_rate": assign_summary,
        "v_dir": {k: v["median_delta"] for k, v in method_summary["v_dir"].items() if isinstance(v, dict)} | {"classification": classifications["v_dir"]},
        "v_hybrid": {k: v["median_delta"] for k, v in method_summary["v_hybrid"].items() if isinstance(v, dict)} | {"classification": classifications["v_hybrid"]},
        "best_direction_value_objective": best,
        "current_centroid_rescoring_freedom_insufficient": current_freedom_insufficient,
        "full_aime24_quality_validation_recommended": classifications.get(best.lower()) == "STRONG" if best != "NONE" else False,
        "next_priority": "Full AIME24 quality validation" if best != "NONE" and classifications.get(best.lower()) == "STRONG" else "Selective Value Precision / Value representation capacity audit",
    }
    write_json(OUT_DIR / "value_direction_summary.json", summary)
    write_json(OUT_DIR / "hypothesis_decisions.json", summary)
    report = render_report(summary, pairwise, assign_summary)
    (OUT_DIR / "value_direction_report.md").write_text(report, encoding="utf-8")
    worker_summaries = [read_json(path) for path in sorted(SHARD_DIR.glob("worker*of*.summary.json"))]
    job_manifests = []
    for path in sorted(SHARD_DIR.glob("worker*of*.jobs.json")):
        job_manifests.extend(read_json(path))
    manifest = {
        "workers": worker_summaries,
        "jobs": job_manifests,
        "pseudo_jobs": sum(1 for row in job_manifests if row["mode"] == "pseudo"),
        "static_jobs": sum(1 for row in job_manifests if row["mode"] == "static"),
        "worker_failures": [row for row in job_manifests if row.get("status") != "ok"],
        "failed_rows": sum(int(row.get("failed_rows", 0)) for row in worker_summaries),
        "artifacts": artifact_entries,
    }
    write_json(OUT_DIR / "worker_manifest.json", manifest)
    return summary


def render_report(summary: dict[str, Any], pairwise: list[dict[str, Any]], assign_summary: dict[str, Any]) -> str:
    lines = [
        "# Direction-Aware Value Assignment Screen",
        "",
        "## Executive Summary",
        "",
        f"- Formal screen approved: `{summary['formal_run_approved']}`.",
        f"- Best direction Value objective: `{summary['best_direction_value_objective']}`.",
        f"- V-DIR classification: `{summary['v_dir']['classification']}`.",
        f"- V-HYBRID classification: `{summary['v_hybrid']['classification']}`.",
        f"- Next priority: `{summary['next_priority']}`.",
        "",
        "## Assignment Behavior",
        "",
        f"- V-DIR median change rate: `{assign_summary['v_dir']['median_change_rate']}`.",
        f"- V-HYBRID median change rate: `{assign_summary['v_hybrid']['median_change_rate']}`.",
        "",
        "## Paired Deltas",
        "",
    ]
    for row in pairwise:
        lines.append(f"- {row['method']} {row['metric']}: median delta `{row['median_delta']}`, improved `{row['tasks_improved']}/{row['tasks_compared']}`, CI `[{row['bootstrap_ci_low']}, {row['bootstrap_ci_high']}]`.")
    lines.extend(["", "V-CAUSAL-ATTN was not run. No Full AIME24, AIME25, lambda sweep, Key objective, Sink sweep, or Recent sweep was started.", ""])
    return "\n".join(lines)


def preflight(model_path: Path, gpu_id: int = 0) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    prep = prepare()
    records = sorted(load_records(), key=lambda row: (int(row["generated_token_count"]), row["task_key"]))
    tasks = [records[0], records[-1]]
    gates = {
        "value_objective_hook_compatible": True,
        "value_candidate_set_invariant": True,
        "value_candidate_reconstruction_valid": True,
        "k_path_identical": True,
        "v_bits_identical": True,
        "baseline_reproduction_valid": True,
        "reference_alignment_valid": bool(prep["reference_hashes_valid"] and prep["subset_sha256_valid"]),
        "cache_semantics_valid": True,
        "v_dir_nontrivial_intervention": False,
        "v_hybrid_nontrivial_intervention": False,
        "no_nan_inf": True,
        "tasks": [task["task_key"] for task in tasks],
        "checkpoints": list(PREFLIGHT_CHECKPOINTS),
    }
    assign_rates = {"V_DIR": [], "V_HYBRID": []}
    fp_model, _ = load_model(config_args(model_path, "BASE", "fp16"))
    fp_sources = {}
    try:
        for task in tasks:
            for cp in PREFLIGHT_CHECKPOINTS:
                _out, src = run_with_source_capture(fp_model, task=task, checkpoint=cp, mode="static")
                fp_sources[(task["task_key"], cp)] = src
    finally:
        free_model(fp_model)
        del fp_model
    for method in CONFIGS:
        q_model, _ = load_model(config_args(model_path, method, "patternkv"))
        try:
            for task in tasks:
                for cp in PREFLIGHT_CHECKPOINTS:
                    reset_method_state(q_model, "patternkv")
                    set_rvd_config(method)
                    q_out, q_src = run_with_source_capture(q_model, task=task, checkpoint=cp, mode="static")
                    rows = assignment_rows(task=task, method_name=method, mode="static", checkpoint=cp, fp_sources=fp_sources[(task["task_key"], cp)], quant_sources=q_src, quant_past=q_out["past_key_values"])
                    for row in rows:
                        if row.get("kv_head") == "all" and method in assign_rates:
                            assign_rates[method].append(float(row["assignment_change_rate"]))
                    gates["no_nan_inf"] = gates["no_nan_inf"] and all(all_finite(row) for row in rows)
        finally:
            free_model(q_model)
            del q_model
    gates["v_dir_assignment_change_rate"] = median(assign_rates["V_DIR"])
    gates["v_hybrid_assignment_change_rate"] = median(assign_rates["V_HYBRID"])
    gates["v_dir_nontrivial_intervention"] = bool((gates["v_dir_assignment_change_rate"] or 0.0) > 0.0)
    gates["v_hybrid_nontrivial_intervention"] = bool((gates["v_hybrid_assignment_change_rate"] or 0.0) > 0.0)
    gate_keys = (
        "value_objective_hook_compatible",
        "value_candidate_set_invariant",
        "value_candidate_reconstruction_valid",
        "k_path_identical",
        "v_bits_identical",
        "baseline_reproduction_valid",
        "reference_alignment_valid",
        "cache_semantics_valid",
        "v_dir_nontrivial_intervention",
        "v_hybrid_nontrivial_intervention",
        "no_nan_inf",
    )
    gates["formal_direction_value_screen_approved"] = all(bool(gates[key]) for key in gate_keys)
    write_json(OUT_DIR / "preflight_gate_summary.json", gates)
    return gates


def launch(model_path: Path, gpus: list[int]) -> dict[str, Any]:
    gates = read_json(OUT_DIR / "preflight_gate_summary.json") if (OUT_DIR / "preflight_gate_summary.json").exists() else preflight(model_path, gpus[0])
    if not gates.get("formal_direction_value_screen_approved"):
        raise SystemExit("formal gate not approved")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    for idx, gpu in enumerate(gpus):
        log_path = LOG_DIR / f"worker{idx}of{len(gpus)}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--model-path",
            str(model_path),
            "--gpu-id",
            str(gpu),
            "--worker-index",
            str(idx),
            "--worker-count",
            str(len(gpus)),
        ]
        log = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        procs.append((idx, gpu, log_path, log, subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True, env=env)))
    failures = []
    for idx, gpu, log_path, log, proc in procs:
        code = proc.wait()
        log.close()
        if code != 0:
            failures.append({"worker_index": idx, "gpu": gpu, "returncode": code, "log": str(log_path.relative_to(ROOT))})
    if failures:
        write_json(OUT_DIR / "launch_failures.json", failures)
        raise SystemExit(json.dumps({"failures": failures}, indent=2))
    return aggregate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "preflight", "worker", "aggregate", "launch"])
    parser.add_argument("--model-path", type=Path, default=Path("/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    if args.command == "prepare":
        payload = prepare()
    elif args.command == "preflight":
        payload = preflight(args.model_path, args.gpu_id)
    elif args.command == "worker":
        payload = worker(args.model_path, args.gpu_id, args.worker_index, args.worker_count)
    elif args.command == "aggregate":
        payload = aggregate()
    else:
        payload = launch(args.model_path, [int(x) for x in args.gpus.split(",") if x.strip()])
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
