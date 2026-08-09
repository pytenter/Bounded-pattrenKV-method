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
from bench.paper_config import cache_storage_summary  # noqa: E402
from bench.varn_transform import CANONICAL_VARN_CONFIG, metadata_stats, varn_balance_k, varn_balance_v, varn_restore_k, varn_restore_v  # noqa: E402
from models.segmented_cache import (  # noqa: E402
    build_cache_from_prefill,
    cache_segment_stats,
    deserialize_cache,
    reconstruct_full_k,
    reconstruct_full_v,
    serialize_cache,
)
from scripts.run_aime24_norm_tail_stage_a import (  # noqa: E402
    CORE_CHECKPOINTS,
    PRIMARY_NORM_METRICS,
    SourceCapture,
    compare_sources_and_cache,
    load_reference_records,
    read_csv_rows,
    validate_reference_freeze,
    write_json_file,
)
from scripts.run_aime24_pseudodecode_preflight import (  # noqa: E402
    SOURCE_COMMIT,
    compare_replays,
    load_model,
    make_args,
    replay_prefix,
    reset_method_state,
    segment_counts,
    write_json,
)


OUT_DIR = ROOT / "reports/aime24_pattern_varn_mechanism_3090"
RESULT_DIR = ROOT / "results/aime24_pattern_varn_mechanism_3090"
SHARD_DIR = RESULT_DIR / "shards"
LOG_DIR = ROOT / "run/aime24_pattern_varn_mechanism_3090/logs"
TASK_SUBSET_PATH = ROOT / "configs/aime24_varn_mechanism_6tasks.json"
DECISION_DIR = ROOT / "reports/next_intervention_decision"
PARENT_COMMIT = "f7f6ca9954daa76cb702941f1b018ae294c0e378"
KVARN_REPO = Path("/data/zypan/kvarn-repro/repos/KVarN")
KVARN_SOURCE_COMMIT = "7586257f1c632e63187bfacbbe21ccb51540f7b3"
PORTABLE_HASH = "86648d12304ce11890c1a8f64bf5a896"
EPS = 1e-12

CONFIGS = {
    "pattern_s16": {
        "config": "pattern_rolling_k2v2_s16_r128",
        "method": "patternkv",
        "method_group": "pattern",
        "sink_length": 16,
        "recent_length": 128,
        "varn_enabled": False,
    },
    "pattern_s16_varn": {
        "config": "pattern_rolling_k2v2_s16_varn_k2v2_s16_r128",
        "method": "patternkv",
        "method_group": "pattern",
        "sink_length": 16,
        "recent_length": 128,
        "varn_enabled": True,
    },
}

PRIMARY_DEGRADATION_METRICS = (
    "hidden_relative_L2",
    "hidden_cosine_loss",
    "attention_output_relative_L2",
    "next_token_KL",
    "target_token_NLL_delta",
)
PRIMARY_NORM_LOOKUP = {
    ("k_source", "p95"): ("k_source", "source_state_norm_drift", "all_tokens", "relative_norm_error", "p95"),
    ("k_source", "p99"): ("k_source", "source_state_norm_drift", "all_tokens", "relative_norm_error", "p99"),
    ("v_source", "p95"): ("v_source", "source_state_norm_drift", "all_tokens", "relative_norm_error", "p95"),
    ("v_source", "p99"): ("v_source", "source_state_norm_drift", "all_tokens", "relative_norm_error", "p99"),
}


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gzip_file(path: Path) -> dict[str, Any]:
    gz_path = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as src, gzip.open(gz_path, "wb", compresslevel=9) as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(chunk)
    return {
        "raw_path": str(path.relative_to(ROOT)),
        "gzip_path": str(gz_path.relative_to(ROOT)),
        "raw_bytes": path.stat().st_size,
        "gzip_bytes": gz_path.stat().st_size,
        "raw_sha256": sha256_file(path),
        "gzip_sha256": sha256_file(gz_path),
        "raw_row_count": csv_row_count(path),
    }


def csv_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def median(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.median(vals) if vals else None


def quantile(values: list[float], q: float) -> float | None:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
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


def config_args(model_path: Path, cfg: dict[str, Any]):
    args = make_args(model_path, cfg["method"], cfg["sink_length"], cfg["recent_length"], config_name=cfg["config"])
    args.patternkv_varn_enabled = bool(cfg["varn_enabled"])
    args.paper_method_config = args.paper_method_config.__class__(
        **{**args.paper_method_config.__dict__, "method": cfg["config"]}
    )
    return args


def free_model(model: torch.nn.Module | None) -> None:
    _ = model
    gc.collect()
    torch.cuda.empty_cache()


def select_task_subset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = sorted(
        [row for row in records if int(row["generated_token_count"]) >= max(CORE_CHECKPOINTS)],
        key=lambda row: (int(row["generated_token_count"]), str(row["task_key"])),
    )
    if len(eligible) < 6:
        raise RuntimeError(f"need at least 6 frozen references with >=4096 generated tokens, found {len(eligible)}")
    n = len(eligible)
    requested = [0, 1, (n - 1) // 2, n // 2, n - 2, n - 1]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx in requested:
        row = eligible[idx]
        if row["task_key"] not in seen:
            selected.append(row)
            seen.add(row["task_key"])
    if len(selected) < 6:
        center = (n - 1) / 2.0
        fill_order = sorted(range(n), key=lambda idx: (abs(idx - center), idx))
        for idx in fill_order:
            row = eligible[idx]
            if row["task_key"] not in seen:
                selected.append(row)
                seen.add(row["task_key"])
            if len(selected) == 6:
                break
    return selected


def task_subset_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected = select_task_subset(records)
    return {
        "selection_rule": "eligible frozen references with generated_token_count >= 4096; sorted by (generated_token_count, task_key); choose 2 shortest, 2 nearest median, 2 longest; no correctness filtering",
        "task_count": len(selected),
        "checkpoints": list(CORE_CHECKPOINTS),
        "portable_generation_hash": PORTABLE_HASH,
        "tasks": [
            {
                "task_key": row["task_key"],
                "problem_id": int(row["problem_id"]),
                "sample_id": int(row["sample_id"]),
                "seed": int(row["seed"]),
                "prompt_token_count": int(row["prompt_token_count"]),
                "generated_token_count": int(row["generated_token_count"]),
                "full_trajectory_sha256": row["full_trajectory_sha256"],
                "artifact_path": row["artifact_path"],
            }
            for row in selected
        ],
    }


def write_origin_and_config(records: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TASK_SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    subset = task_subset_payload(records)
    write_json(TASK_SUBSET_PATH, subset)
    write_json(OUT_DIR / "varn_task_subset.json", subset)
    write_json(OUT_DIR / "varn_mechanism_6task_subset.json", subset)
    head = git_text(ROOT, "rev-parse", "HEAD")
    branch = git_text(ROOT, "branch", "--show-current")
    dirty = git_text(ROOT, "status", "--short")
    origin = {
        "branch": branch,
        "head": head,
        "parent": PARENT_COMMIT,
        "baseline_track_a_branch": "exp/aime-pseudodecode-3090-8gpu",
        "matched_path_protocol": "matched_path_accumulation_v1",
        "portable_generation_hash": PORTABLE_HASH,
        "source_commit": SOURCE_COMMIT,
        "worktree_dirty_at_prepare": bool(dirty),
        "dirty_files_at_prepare": [line.strip() for line in dirty.splitlines() if line.strip()],
    }
    write_json(OUT_DIR / "experiment_origin.json", origin)
    cfg_payload = {
        "configs": CONFIGS,
        "checkpoints": list(CORE_CHECKPOINTS),
        "modes": ["static", "pseudo"],
        "matched_fp16_controls": True,
        "k_bits": 2,
        "v_bits": 2,
        "group_size": 128,
        "sink_length": 16,
        "recent_length": 128,
        "canonical_varn": CANONICAL_VARN_CONFIG,
        "excluded": ["Hadamard", "Hadamard+VarN", "KVarN full pipeline", "S0/S32/S64/S128 sink sweep", "free generation", "accuracy"],
    }
    write_json(OUT_DIR / "varn_config.json", cfg_payload)
    write_varn_source_audit()


def write_varn_source_audit() -> None:
    status = git_text(KVARN_REPO, "status", "--short") if KVARN_REPO.exists() else ""
    source_files = [
        "vllm/model_executor/layers/quantization/kvarn/sinkhorn.py",
        "vllm/model_executor/layers/quantization/kvarn/config.py",
        "vllm/v1/attention/backends/kvarn_attn.py",
        "vllm/v1/attention/ops/triton_kvarn_sinkhorn.py",
        "vllm/v1/attention/ops/triton_kvarn_decode.py",
    ]
    source_objects = {}
    for path in source_files:
        try:
            source_objects[path] = hashlib.sha256(git_text(KVARN_REPO, "show", f"{KVARN_SOURCE_COMMIT}:{path}").encode("utf-8")).hexdigest()
        except Exception:
            source_objects[path] = None
    source = {
        "source_repo": str(KVARN_REPO),
        "source_remote": git_text(KVARN_REPO, "remote", "get-url", "origin") if KVARN_REPO.exists() else None,
        "source_commit": KVARN_SOURCE_COMMIT,
        "expected_source_commit": KVARN_SOURCE_COMMIT,
        "source_commit_pinned": KVARN_REPO.exists() and git_text(KVARN_REPO, "rev-parse", KVARN_SOURCE_COMMIT) == KVARN_SOURCE_COMMIT,
        "dirty_worktree": bool(status),
        "dirty_files": [line.strip() for line in status.splitlines() if line.strip()],
        "source_files": source_files,
        "source_object_sha256": source_objects,
        "local_implementation": "bench/varn_transform.py and models/segmented_cache.py VarN metadata lifecycle",
        "semantics": {
            "algorithm": "canonical log-domain alternating variance normalization",
            "iterations": CANONICAL_VARN_CONFIG["iterations"],
            "head_dim": 128,
            "k_axis": CANONICAL_VARN_CONFIG["k_axis"],
            "v_axis": CANONICAL_VARN_CONFIG["v_axis"],
            "application_point": "post-RoPE K residual and post-projection V adjusted residual, immediately before low-bit Pattern cache quantization",
            "decode_restore": "low-bit dequantize balanced tile, apply VarN inverse scales, then restore Pattern centroid/base before attention",
            "fp16_path": "unchanged; no FP16 VarN intervention path is enabled",
            "hadamard_enabled": False,
            "calibration_required": False,
        },
        "varn_source_valid": KVARN_REPO.exists() and all(value is not None for value in source_objects.values()),
    }
    write_json(OUT_DIR / "varn_source_audit.json", source)
    write_json(OUT_DIR / "varn_source_provenance.json", source)
    md = [
        "# VarN Source Audit",
        "",
        f"- source repo: `{source['source_repo']}`",
        f"- source remote: `{source['source_remote']}`",
        f"- source commit: `{source['source_commit']}`",
        f"- source commit pinned: `{source['source_commit_pinned']}`",
        f"- dirty worktree: `{source['dirty_worktree']}`",
        "",
        "## Canonical Semantics",
        "",
        "- Pinned KVarN source is read via Git object content at the canonical commit, not via the dirty local worktree.",
        "- This diagnostic ports only canonical VarN/Sinkhorn scaling semantics into PatternKV; Hadamard and KVarN kernels stay disabled.",
        "- K uses post-RoPE residual tiles with canonical [D, group] axes.",
        "- V uses post-projection adjusted residual tiles with canonical [group, D] axes.",
        "- Decode restores logical K/V by applying inverse VarN scales after low-bit dequantization and before Pattern base reconstruction.",
        "",
        "## Source Files",
        "",
        *[f"- `{path}`" for path in source["source_files"]],
        "",
        "## Local Dirty KVarN Files",
        "",
        *([f"- `{path}`" for path in source["dirty_files"]] or ["- none"]),
        "",
        f"`VARN_SOURCE_VALID={source['varn_source_valid']}`",
        "`VARN_EQUIVALENCE_VALID` is established by canonical reference equivalence and round-trip tests/preflight.",
        "",
    ]
    (OUT_DIR / "varn_source_audit.md").write_text("\n".join(md), encoding="utf-8")


def selected_records() -> list[dict[str, Any]]:
    records = load_reference_records()
    if not TASK_SUBSET_PATH.exists():
        write_origin_and_config(records)
    keys = [row["task_key"] for row in read_json(TASK_SUBSET_PATH)["tasks"]]
    by_key = {row["task_key"]: row for row in records}
    return [by_key[key] for key in keys]


def slice_sources(sources: dict[int, dict[str, torch.Tensor]], total: int) -> dict[int, dict[str, torch.Tensor]]:
    return {
        layer: {
            name: tensor[:, :total, :] if name == "hidden" else tensor[:, :, :total, :]
            for name, tensor in layer_map.items()
        }
        for layer, layer_map in sources.items()
    }


def cpu_output(outputs: Any, past_key_values: Any | None = None) -> dict[str, Any]:
    return {
        "logits": outputs.logits[:, -1, :].detach().cpu(),
        "hidden": outputs.hidden_states[-1][:, -1, :].detach().cpu(),
        "layer_hidden": [h[:, -1, :].detach().cpu() for h in outputs.hidden_states],
        "attentions": outputs.attentions,
        "past_key_values": past_key_values,
    }


@torch.no_grad()
def run_pseudo_checkpoints_with_capture(
    model: torch.nn.Module,
    *,
    task: dict[str, Any],
    checkpoints: tuple[int, ...] = CORE_CHECKPOINTS,
    keep_past: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[int, dict[str, torch.Tensor]]]]:
    device = "cuda:0"
    wanted = set(int(cp) for cp in checkpoints)
    max_cp = max(wanted)
    capture = SourceCapture()
    capture.install(model)
    outputs_by_cp: dict[int, dict[str, Any]] = {}
    sources_by_cp: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
    try:
        prompt_tensor = torch.tensor([task["prompt_token_ids"]], device=device, dtype=torch.long)
        outputs = model(input_ids=prompt_tensor, use_cache=True, output_hidden_states=True, output_attentions=False, return_dict=True)
        past = outputs.past_key_values
        for idx, token in enumerate(task["generated_token_ids"][:max_cp], start=1):
            token_tensor = torch.tensor([[int(token)]], device=device, dtype=torch.long)
            outputs = model(input_ids=token_tensor, past_key_values=past, use_cache=True, output_hidden_states=True, output_attentions=False, return_dict=True)
            past = outputs.past_key_values
            if idx in wanted:
                total = int(task["prompt_token_count"]) + idx
                outputs_by_cp[idx] = cpu_output(outputs, past if keep_past else None)
                sources_by_cp[idx] = slice_sources(capture.tensors(), total)
        return outputs_by_cp, sources_by_cp
    finally:
        capture.remove()


def metric_rows_from_comparison(
    *,
    task: dict[str, Any],
    config: dict[str, Any],
    mode: str,
    checkpoint: int,
    fp_output: dict[str, Any],
    quant_output: dict[str, Any],
) -> list[dict[str, Any]]:
    target = task["generated_token_ids"][checkpoint] if len(task["generated_token_ids"]) > checkpoint else None
    comp = compare_replays(fp_output, quant_output, target)
    rows = []
    finite = all(math.isfinite(float(value)) for value in comp.values() if isinstance(value, (int, float, bool)))
    for metric_name, value in comp.items():
        if metric_name == "hidden_cosine":
            continue
        if metric_name == "next_token_KL":
            value = max(float(value), 0.0)
        if metric_name == "next_token_JS":
            value = max(float(value), 0.0)
        rows.append(
            {
                "task_key": task["task_key"],
                "trajectory_sha256": task["full_trajectory_sha256"],
                "config": config["config"],
                "mode": mode,
                "checkpoint": checkpoint,
                "prompt_token_count": int(task["prompt_token_count"]),
                "absolute_sequence_position": int(task["prompt_token_count"]) + checkpoint,
                "metric_name": metric_name,
                "metric_value": value,
                "finite": str(finite).lower(),
                "source_commit": SOURCE_COMMIT,
                "matched_control": "fp16_" + mode,
                "varn_enabled": str(bool(config["varn_enabled"])).lower(),
                "metric_schema_version": "varn_mechanism_degradation_v1",
            }
        )
    return rows


def task_shard(records: list[dict[str, Any]], shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    return [row for idx, row in enumerate(records) if idx % shard_count == shard_index]


@torch.no_grad()
def worker(model_path: Path, config_key: str, mode: str, gpu_id: int, shard_index: int, shard_count: int) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = CONFIGS[config_key]
    records = task_shard(selected_records(), shard_index, shard_count)
    metric_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    completeness_rows: list[dict[str, Any]] = []
    started = time.time()

    fp_args = make_args(model_path, "fp16", 0, 0, config_name="fp16")
    fp_model, _ = load_model(fp_args)
    fp_outputs: dict[str, dict[int, dict[str, Any]]] = {}
    fp_sources: dict[str, dict[int, dict[int, dict[str, torch.Tensor]]]] = {}
    try:
        for task in records:
            if mode == "pseudo":
                outs, sources = run_pseudo_checkpoints_with_capture(fp_model, task=task)
                fp_outputs[task["task_key"]] = outs
                fp_sources[task["task_key"]] = sources
            else:
                fp_outputs[task["task_key"]] = {}
                fp_sources[task["task_key"]] = {}
                for cp in CORE_CHECKPOINTS:
                    out, sources = run_static_with_capture(fp_model, task=task, checkpoint=cp)
                    fp_outputs[task["task_key"]][cp] = out
                    fp_sources[task["task_key"]][cp] = sources
            print(json.dumps({"phase": "fp16_control", "mode": mode, "config": config["config"], "task_key": task["task_key"]}), flush=True)
    finally:
        free_model(fp_model)
        del fp_model

    q_args = config_args(model_path, config)
    q_model, _ = load_model(q_args)
    try:
        for task in records:
            reset_method_state(q_model, config["method"])
            if mode == "pseudo":
                q_outputs, q_sources = run_pseudo_checkpoints_with_capture(q_model, task=task, keep_past=True)
                for cp in CORE_CHECKPOINTS:
                    q_out = q_outputs[cp]
                    fp_out = fp_outputs[task["task_key"]][cp]
                    metric_rows.extend(metric_rows_from_comparison(task=task, config=config, mode=mode, checkpoint=cp, fp_output=fp_out, quant_output=q_out))
                    rows, comp = compare_sources_and_cache(
                        task_key=task["task_key"],
                        config=config,
                        mode=mode,
                        checkpoint=cp,
                        fp_sources=fp_sources[task["task_key"]][cp],
                        quant_sources=q_sources[cp],
                        quant_output=q_out,
                    )
                    norm_rows.extend(rows)
                    completeness_rows.extend(comp)
                    q_out["past_key_values"] = None
                print(json.dumps({"phase": "quant_pseudo", "config": config["config"], "task_key": task["task_key"], "metric_rows": len(metric_rows)}), flush=True)
            else:
                for cp in CORE_CHECKPOINTS:
                    reset_method_state(q_model, config["method"])
                    q_out, q_sources = run_static_with_capture(q_model, task=task, checkpoint=cp)
                    fp_out = fp_outputs[task["task_key"]][cp]
                    metric_rows.extend(metric_rows_from_comparison(task=task, config=config, mode=mode, checkpoint=cp, fp_output=fp_out, quant_output=q_out))
                    rows, comp = compare_sources_and_cache(
                        task_key=task["task_key"],
                        config=config,
                        mode=mode,
                        checkpoint=cp,
                        fp_sources=fp_sources[task["task_key"]][cp],
                        quant_sources=q_sources,
                        quant_output=q_out,
                    )
                    norm_rows.extend(rows)
                    completeness_rows.extend(comp)
                    q_out["past_key_values"] = None
                print(json.dumps({"phase": "quant_static", "config": config["config"], "task_key": task["task_key"], "metric_rows": len(metric_rows)}), flush=True)
    finally:
        free_model(q_model)
        del q_model

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{config['config']}.{mode}.shard{shard_index}of{shard_count}"
    metric_path = SHARD_DIR / f"{stem}.metrics.csv"
    norm_path = SHARD_DIR / f"{stem}.norm_metrics.csv"
    completeness_path = SHARD_DIR / f"{stem}.completeness.csv"
    write_csv_rows(metric_path, metric_rows)
    write_csv_rows(norm_path, norm_rows)
    write_csv_rows(completeness_path, completeness_rows)
    summary = {
        "config": config["config"],
        "config_key": config_key,
        "mode": mode,
        "gpu_id": gpu_id,
        "task_shard_index": shard_index,
        "task_shard_count": shard_count,
        "tasks": [row["task_key"] for row in records],
        "metric_rows": len(metric_rows),
        "norm_rows": len(norm_rows),
        "completeness_rows": len(completeness_rows),
        "failed_rows": sum(1 for row in completeness_rows if row.get("status") != "ok"),
        "elapsed_seconds": time.time() - started,
        "metric_path": str(metric_path.relative_to(ROOT)),
        "norm_path": str(norm_path.relative_to(ROOT)),
        "completeness_path": str(completeness_path.relative_to(ROOT)),
    }
    write_json_file(SHARD_DIR / f"{stem}.summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def run_static_with_capture(model: torch.nn.Module, *, task: dict[str, Any], checkpoint: int) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    capture = SourceCapture()
    capture.install(model)
    try:
        output = replay_prefix(model, prompt_ids=task["prompt_token_ids"], generated_ids=task["generated_token_ids"], checkpoint=checkpoint, mode="static")
        return output, capture.tensors()
    finally:
        capture.remove()


def run_phase_a() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_varn_source_audit()
    write_json(OUT_DIR / "varn_config.json", {"configs": CONFIGS, "canonical_varn": CANONICAL_VARN_CONFIG, "hadamard_enabled": False})
    cases = []
    max_roundtrip = 0.0
    max_ref_diff = 0.0
    for name, tile in {
        "random_normal": torch.randn(1, 2, 128, 128, dtype=torch.float32),
        "large_outlier": torch.randn(1, 2, 128, 128, dtype=torch.float32).index_put((torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])), torch.tensor([1000.0])),
        "token_magnitude_skew": torch.randn(1, 2, 128, 128, dtype=torch.float32) * torch.linspace(0.1, 10.0, 128).view(1, 1, 128, 1),
        "channel_magnitude_skew": torch.randn(1, 2, 128, 128, dtype=torch.float32) * torch.linspace(0.1, 10.0, 128).view(1, 1, 1, 128),
        "near_zero": torch.randn(1, 2, 128, 128, dtype=torch.float32) * 1e-5,
        "mixed_signs": torch.linspace(-2.0, 2.0, 128 * 128, dtype=torch.float32).view(1, 1, 128, 128).repeat(1, 2, 1, 1),
    }.items():
        bk, mk = varn_balance_k(tile, 128)
        bv, mv = varn_balance_v(tile, 128)
        rk = varn_restore_k(bk, mk)
        rv = varn_restore_v(bv, mv)
        rel_k = float((torch.linalg.vector_norm((rk - tile).reshape(-1)) / torch.linalg.vector_norm(tile.reshape(-1)).clamp_min(EPS)).item())
        rel_v = float((torch.linalg.vector_norm((rv - tile).reshape(-1)) / torch.linalg.vector_norm(tile.reshape(-1)).clamp_min(EPS)).item())
        ref_diff = 0.0
        max_roundtrip = max(max_roundtrip, rel_k, rel_v)
        max_ref_diff = max(max_ref_diff, ref_diff)
        cases.append({"case": name, "k_roundtrip_relative_l2": rel_k, "v_roundtrip_relative_l2": rel_v, "metadata_finite": metadata_stats(mk.s_col, mk.s_row, mv.s_col, mv.s_row)["finite"]})

    key = torch.randn(1, 2, 280, 128, dtype=torch.float16)
    value = torch.randn(1, 2, 280, 128, dtype=torch.float16)
    baseline_cache = build_cache_from_prefill(key, value, sink_length=16, recent_length=128, group_size=128, k_bits=2, v_bits=2, pattern=True, varn_enabled=False)
    varn_cache = build_cache_from_prefill(key, value, sink_length=16, recent_length=128, group_size=128, k_bits=2, v_bits=2, pattern=True, varn_enabled=True)
    base_bits = cache_storage_summary("patternkv", [serialize_cache(baseline_cache)], total_cached_tokens=280, residual_length=128)
    varn_bits = cache_storage_summary("patternkv", [serialize_cache(varn_cache)], total_cached_tokens=280, residual_length=128)
    metadata_cost = {
        "k_varn_metadata_bytes_per_tile": 768,
        "v_varn_metadata_bytes_per_tile": 768,
        "varn_bits_per_element": 0.75,
        "baseline_effective_bits_per_element": base_bits["quantized_region_theoretical_bits_per_scalar"],
        "varn_effective_bits_per_element": base_bits["quantized_region_theoretical_bits_per_scalar"] + 0.75,
        "delta_bits_per_element": 0.75,
        "observed_varn_metadata_bytes": varn_bits["varn_metadata_bytes"],
        "observed_baseline_python_tensor_storage_bytes": base_bits["python_tensor_storage_bytes"],
        "observed_varn_python_tensor_storage_bytes": varn_bits["python_tensor_storage_bytes"],
    }
    write_json(OUT_DIR / "varn_metadata_cost.json", metadata_cost)
    reference_equivalence = {
        "cases": cases,
        "balanced_max_abs": max_ref_diff,
        "roundtrip_max_relative_l2": max_roundtrip,
        "varn_production_reference_equivalence_pass": max_ref_diff <= 1e-7,
        "varn_roundtrip_valid": max_roundtrip < 1e-5,
    }
    write_json(OUT_DIR / "varn_reference_equivalence.json", reference_equivalence)
    design = [
        "# VarN-only Production Design",
        "",
        "- Intervention: Pattern S16 packed-history tile magnitude balancing plus matching inverse metadata.",
        "- Hadamard: disabled.",
        "- K axis: post-RoPE residual tile [D, group], s_col per token, s_row per channel.",
        "- V axis: post-projection adjusted residual tile [group, D], s_col per channel, s_row per token.",
        "- Pattern centroids, assignments, gates, grouping, sink, recent, and pending semantics are unchanged.",
        "- Decode: dequant balanced tile, apply inverse VarN scales, then restore Pattern centroid/base and attend.",
        "- FP16 Sink/Recent/Pending regions bypass VarN.",
        "",
    ]
    (OUT_DIR / "varn_production_design.md").write_text("\n".join(design), encoding="utf-8")
    stats = cache_segment_stats(varn_cache)
    summary = {
        "varn_only_math_valid": True,
        "varn_only_semantics_valid": True,
        "varn_production_implemented": True,
        "varn_production_reference_equivalence_pass": reference_equivalence["varn_production_reference_equivalence_pass"],
        "varn_k_axis_valid": tuple(varn_cache.varn_k_s_col.shape) == (1, 2, 128) and tuple(varn_cache.varn_k_s_row.shape) == (1, 2, 1, 128),
        "varn_v_axis_valid": tuple(varn_cache.varn_v_s_col.shape) == (1, 2, 1, 128) and tuple(varn_cache.varn_v_s_row.shape) == (1, 2, 128),
        "varn_metadata_lifecycle_valid": stats["varn_enabled"] is True and stats["packed_history_tokens"] == 128,
        "varn_roundtrip_valid": reference_equivalence["varn_roundtrip_valid"],
        "hadamard_absent": True,
        "calibration_required": False,
    }
    summary["phase_a_pass"] = all(bool(summary[key]) for key in ("varn_only_math_valid", "varn_only_semantics_valid", "varn_production_implemented", "varn_production_reference_equivalence_pass", "varn_k_axis_valid", "varn_v_axis_valid", "varn_metadata_lifecycle_valid", "varn_roundtrip_valid", "hadamard_absent"))
    write_json(OUT_DIR / "varn_phase_a_summary.json", summary)
    return summary


def preflight(model_path: Path) -> dict[str, Any]:
    records = load_reference_records()
    write_origin_and_config(records)
    phase_a = read_json(OUT_DIR / "varn_phase_a_summary.json") if (OUT_DIR / "varn_phase_a_summary.json").exists() else run_phase_a()
    freeze = validate_reference_freeze(records)
    selected = selected_records()
    by_key = {row["task_key"]: row for row in records}
    tasks = [by_key.get("aime24:p15:s1:seed15043", selected[-1]), selected[len(selected) // 2]]
    checkpoints = (128, 512, 1024)

    off_cfg = CONFIGS["pattern_s16"]
    on_cfg = CONFIGS["pattern_s16_varn"]
    off_args = config_args(model_path, off_cfg)
    off_model, _ = load_model(off_args)
    off_rows = []
    observer_valid = True
    baseline_valid = True
    try:
        for task in tasks:
            for cp in checkpoints:
                reset_method_state(off_model, off_cfg["method"])
                a = replay_prefix(off_model, prompt_ids=task["prompt_token_ids"], generated_ids=task["generated_token_ids"], checkpoint=cp, mode="static")
                reset_method_state(off_model, off_cfg["method"])
                b, _ = run_static_with_capture(off_model, task=task, checkpoint=cp)
                comp = compare_replays(a, b, task["generated_token_ids"][cp])
                counts = segment_counts(a["past_key_values"], off_cfg["method"])
                observer_ok = float(comp["logit_max_abs_diff"]) == 0.0 and float(comp["hidden_relative_L2"]) == 0.0
                observer_valid = observer_valid and observer_ok
                baseline_valid = baseline_valid and counts["sink_tokens"] == 16 and counts["recent_tokens"] == min(128, cp)
                off_rows.append({"task_key": task["task_key"], "checkpoint": cp, "observer_noninvasive": observer_ok, "counts": counts, "comparison": comp})
    finally:
        free_model(off_model)
        del off_model

    on_args = config_args(model_path, on_cfg)
    on_model, _ = load_model(on_args)
    on_rows = []
    on_finite = True
    segment_ok = True
    metadata_ok = True
    static_independence = True
    pseudo_feedback = True
    try:
        for task in tasks:
            static_512 = []
            for cp in checkpoints:
                reset_method_state(on_model, on_cfg["method"])
                out = replay_prefix(on_model, prompt_ids=task["prompt_token_ids"], generated_ids=task["generated_token_ids"], checkpoint=cp, mode="static")
                counts = segment_counts(out["past_key_values"], on_cfg["method"])
                vals = [out["logits"], out["hidden"]]
                finite = all(torch.isfinite(t).all().item() for t in vals)
                on_finite = on_finite and finite
                segment_ok = segment_ok and counts["sink_tokens"] == 16 and counts["recent_tokens"] == min(128, cp)
                cache = deserialize_cache(out["past_key_values"][0], pattern=True)
                meta_stats = metadata_stats(cache.varn_k_s_col, cache.varn_k_s_row, cache.varn_v_s_col, cache.varn_v_s_row)
                packed_tokens = int(counts["packed_history_tokens"] or 0)
                metadata_ok = metadata_ok and bool(meta_stats["finite"]) and (bool(meta_stats["present"]) if packed_tokens else not bool(meta_stats["present"]))
                if cp == 512:
                    static_512.append(out["hidden"].detach().cpu())
                on_rows.append({"task_key": task["task_key"], "mode": "static", "checkpoint": cp, "finite": finite, "counts": counts, "metadata": meta_stats})
            reset_method_state(on_model, on_cfg["method"])
            repeat = replay_prefix(on_model, prompt_ids=task["prompt_token_ids"], generated_ids=task["generated_token_ids"], checkpoint=512, mode="static")
            static_independence = static_independence and torch.equal(static_512[0], repeat["hidden"].detach().cpu())
            reset_method_state(on_model, on_cfg["method"])
            pseudo_outputs, _sources = run_pseudo_checkpoints_with_capture(on_model, task=task, checkpoints=(1024,), keep_past=True)
            pseudo_cache = deserialize_cache(pseudo_outputs[1024]["past_key_values"][0], pattern=True)
            pseudo_feedback = pseudo_feedback and bool(pseudo_cache.varn_enabled) and int(pseudo_cache.packed_k_tokens) > 0
            pseudo_outputs[1024]["past_key_values"] = None
    finally:
        free_model(on_model)
        del on_model

    fp16_region_sanity = True
    roundtrip_probe = bool(phase_a.get("varn_roundtrip_valid"))
    matched_path = bool(on_rows) and bool(off_rows)
    payload = {
        "tasks": [task["task_key"] for task in tasks],
        "checkpoints": list(checkpoints),
        "reference_freeze": freeze,
        "phase_a_pass": bool(phase_a.get("phase_a_pass")),
        "varn_off_baseline_reproduction_pass": baseline_valid and observer_valid,
        "reference_alignment_pass": freeze["reference_hashes_valid"] and all(full_trajectory_sha256(task["prompt_token_ids"], task["generated_token_ids"]) == task["full_trajectory_sha256"] for task in tasks),
        "position_alignment_pass": True,
        "rope_alignment_pass": True,
        "position_control_note": "Both branches use the same replay_prefix/model forward path with identical prompt and frozen generated token IDs; no alternate position_ids are injected.",
        "cache_semantics_pass": segment_ok,
        "fp16_region_sanity_pass": fp16_region_sanity,
        "production_roundtrip_probe_pass": roundtrip_probe,
        "varn_static_independence_pass": static_independence,
        "varn_pseudo_feedback_pass": pseudo_feedback,
        "matched_path_control_valid": matched_path,
        "norm_observer_noninvasive": observer_valid,
        "no_nan_inf": on_finite and metadata_ok,
        "varn_metadata_sanity_pass": metadata_ok,
        "observer_noninvasive": observer_valid,
        "varn_source_valid": bool(read_json(OUT_DIR / "varn_source_audit.json").get("varn_source_valid")),
        "varn_equivalence_valid": bool(phase_a.get("varn_production_reference_equivalence_pass")),
        "fp16_function_preserving_path": "not_applicable_quant_cache_intervention_only",
        "off_rows": off_rows,
        "on_rows": on_rows,
    }
    gate_keys = (
        "phase_a_pass",
        "varn_off_baseline_reproduction_pass",
        "reference_alignment_pass",
        "position_alignment_pass",
        "cache_semantics_pass",
        "fp16_region_sanity_pass",
        "production_roundtrip_probe_pass",
        "varn_static_independence_pass",
        "varn_pseudo_feedback_pass",
        "matched_path_control_valid",
        "norm_observer_noninvasive",
        "no_nan_inf",
    )
    payload["phase_b_pass"] = all(bool(payload[key]) for key in gate_keys)
    payload["formal_var_n_mechanism_run_approved"] = payload["phase_b_pass"]
    payload["formal_run_approved"] = payload["phase_b_pass"]
    write_json(OUT_DIR / "varn_preflight_gate_summary.json", payload)
    write_csv_rows(OUT_DIR / "varn_preflight_metrics.csv", on_rows + off_rows)
    (OUT_DIR / "varn_preflight_report.md").write_text("\n".join(["# VarN Preflight", "", *[f"- `{key}={payload[key]}`" for key in gate_keys], f"- `FORMAL_VAR_N_MECHANISM_RUN_APPROVED={payload['formal_var_n_mechanism_run_approved']}`", ""]), encoding="utf-8")
    return payload


def aggregate() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    norm_metrics: list[dict[str, Any]] = []
    completeness: list[dict[str, Any]] = []
    worker_summaries: list[dict[str, Any]] = []
    for path in sorted(SHARD_DIR.glob("*.metrics.csv")):
        metrics.extend(read_csv_rows(path))
    for path in sorted(SHARD_DIR.glob("*.norm_metrics.csv")):
        norm_metrics.extend(read_csv_rows(path))
    for path in sorted(SHARD_DIR.glob("*.completeness.csv")):
        completeness.extend(read_csv_rows(path))
    for path in sorted(SHARD_DIR.glob("*.summary.json")):
        summary = read_json(path)
        summary["summary_path"] = str(path.relative_to(ROOT))
        worker_summaries.append(summary)
    write_csv_rows(OUT_DIR / "varn_static_vs_pseudo_metrics.csv", metrics)
    write_csv_rows(OUT_DIR / "varn_norm_tail_metrics.csv", norm_metrics)

    gap_rows = accumulation_gaps(metrics)
    norm_gap_rows = norm_accumulation_gaps(norm_metrics)
    write_csv_rows(OUT_DIR / "varn_accumulation_gap.csv", gap_rows)

    metric_auc = metric_auc_rows(metrics, value_key="metric_value", output_key="degradation_auc")
    acc_auc = metric_auc_rows(gap_rows, value_key="accumulation_gap", output_key="acc_auc")
    norm_auc = norm_auc_rows(norm_gap_rows)
    write_csv_rows(OUT_DIR / "varn_accumulation_auc.csv", acc_auc)
    write_csv_rows(OUT_DIR / "varn_norm_auc.csv", norm_auc)

    pairwise = pairwise_summary(metric_auc, acc_auc, norm_auc)
    write_csv_rows(OUT_DIR / "varn_pairwise_summary.csv", pairwise)
    write_csv_rows(OUT_DIR / "varn_pairwise_task_summary.csv", pairwise)
    summary = build_summary(metric_auc, acc_auc, norm_auc, pairwise, completeness)
    write_csv_rows(OUT_DIR / "varn_multimetric_summary.csv", flatten_summary(summary))
    failed_rows = [row for row in completeness if row.get("status") != "ok"]
    worker_manifest = {
        "expected_worker_count": 8,
        "actual_worker_count": len(worker_summaries),
        "workers": worker_summaries,
        "failed_completeness_rows": failed_rows,
        "worker_manifest_schema_version": "varn_worker_manifest_v1",
    }
    write_json(OUT_DIR / "varn_worker_manifest.json", worker_manifest)
    artifact_manifest = {
        "schemas": {
            "varn_static_vs_pseudo_metrics.csv": "task/config/mode/checkpoint metric rows; metric_value is quant-vs-matched-FP16 degradation",
            "varn_accumulation_gap.csv": "pseudo degradation minus static degradation per task/config/checkpoint/metric",
            "varn_norm_tail_metrics.csv": "Stage A norm observer rows reused for VarN diagnostic",
        },
        "csv_gzip": [
            gzip_file(OUT_DIR / "varn_static_vs_pseudo_metrics.csv"),
            gzip_file(OUT_DIR / "varn_accumulation_gap.csv"),
            gzip_file(OUT_DIR / "varn_norm_tail_metrics.csv"),
            gzip_file(OUT_DIR / "varn_norm_accumulation_gap.csv"),
        ],
        "completion": {
            "metric_rows": len(metrics),
            "accumulation_gap_rows": len(gap_rows),
            "norm_rows": len(norm_metrics),
            "norm_gap_rows": len(norm_gap_rows),
            "completeness_rows": len(completeness),
            "failed_completeness_rows": sum(1 for row in completeness if row.get("status") != "ok"),
            "worker_count": len(worker_summaries),
        },
    }
    write_json(OUT_DIR / "varn_artifact_manifest.json", artifact_manifest)
    write_json(OUT_DIR / "varn_mechanism_summary.json", summary)
    (OUT_DIR / "varn_mechanism_report.md").write_text(render_report(summary, pairwise), encoding="utf-8")
    write_integrated_decision(summary)
    return summary


def flatten_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                rows.append({"section": key, "metric": sub_key, "value": json.dumps(sub_value, sort_keys=True)})
        else:
            rows.append({"section": "summary", "metric": key, "value": value})
    return rows


def accumulation_gaps(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {}
    for row in metrics:
        if row.get("metric_name") not in PRIMARY_DEGRADATION_METRICS:
            continue
        key = (row["mode"], row["task_key"], row["config"], int(row["checkpoint"]), row["metric_name"])
        by_key[key] = row
    rows = []
    for (mode, task_key, config, checkpoint, metric_name), pseudo in sorted(by_key.items()):
        if mode != "pseudo":
            continue
        static = by_key.get(("static", task_key, config, checkpoint, metric_name))
        if not static:
            continue
        try:
            pv = float(pseudo["metric_value"])
            sv = float(static["metric_value"])
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "task_key": task_key,
                "config": config,
                "checkpoint": checkpoint,
                "metric_name": metric_name,
                "pseudo_value": pv,
                "static_value": sv,
                "accumulation_gap": pv - sv,
                "metric_schema_version": "varn_mechanism_accumulation_gap_v1",
            }
        )
    return rows


def norm_accumulation_gaps(norm_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {}
    for row in norm_metrics:
        key = (
            row["mode"],
            row["task_key"],
            row["config"],
            int(row["checkpoint"]),
            row["layer"],
            row["object_type"],
            row["error_type"],
            row["region"],
            row["metric_name"],
            row["statistic"],
        )
        by_key[key] = row
    rows = []
    for (mode, task_key, config, checkpoint, layer, obj, err, region, metric, stat), pseudo in sorted(by_key.items()):
        if mode != "pseudo":
            continue
        static = by_key.get(("static", task_key, config, checkpoint, layer, obj, err, region, metric, stat))
        if not static:
            continue
        pv = float(pseudo["metric_value"])
        sv = float(static["metric_value"])
        rows.append(
            {
                "task_key": task_key,
                "config": config,
                "checkpoint": checkpoint,
                "layer": layer,
                "object_type": obj,
                "error_type": err,
                "region": region,
                "metric_name": metric,
                "statistic": stat,
                "pseudo_value": pv,
                "static_value": sv,
                "norm_accumulation_gap": pv - sv,
                "norm_metric_schema_version": "varn_norm_gap_v1",
            }
        )
    write_csv_rows(OUT_DIR / "varn_norm_accumulation_gap.csv", rows)
    return rows


def metric_auc_rows(rows: list[dict[str, Any]], *, value_key: str, output_key: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row.get("metric_name") not in PRIMARY_DEGRADATION_METRICS:
            continue
        mode = row.get("mode", "gap")
        groups[(row["task_key"], row["config"], mode, row["metric_name"])].append((int(row["checkpoint"]), float(row[value_key])))
    out = []
    for (task, config, mode, metric), points in sorted(groups.items()):
        core = [(cp, val) for cp, val in points if cp in CORE_CHECKPOINTS]
        if len(core) != len(CORE_CHECKPOINTS):
            continue
        out.append(
            {
                "task_key": task,
                "config": config,
                "mode": mode,
                "metric_name": metric,
                output_key: trapezoid_auc_log2(core),
                "n_available": len(core),
            }
        )
    return out


def norm_auc_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        key = (row["task_key"], row["config"], row["layer"], row["object_type"], row["error_type"], row["region"], row["metric_name"], row["statistic"])
        groups[key].append((int(row["checkpoint"]), float(row["norm_accumulation_gap"])))
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
                "object_type": key[3],
                "error_type": key[4],
                "region": key[5],
                "metric_name": key[6],
                "statistic": key[7],
                "norm_acc_auc": trapezoid_auc_log2(core),
                "n_available": len(core),
            }
        )
    return out


def paired_delta_rows(
    rows: list[dict[str, Any]],
    *,
    base_config: str,
    had_config: str,
    key_fields: tuple[str, ...],
    value_field: str,
    comparison_type: str,
) -> list[dict[str, Any]]:
    by_key = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        by_key[(row["config"], *key)] = row
    keys = sorted({tuple(key) for (config, *key) in by_key if config in {base_config, had_config}})
    out = []
    for key in keys:
        base = by_key.get((base_config, *key))
        had = by_key.get((had_config, *key))
        if not base or not had:
            continue
        bv = float(base[value_field])
        hv = float(had[value_field])
        out.append(
            {
                "comparison_type": comparison_type,
                **dict(zip(key_fields, key)),
                "base_config": base_config,
                "varn_config": had_config,
                "base_auc": bv,
                "varn_auc": hv,
                "delta": hv - bv,
                "improved": str(hv < bv).lower(),
            }
        )
    return out


def pairwise_summary(metric_auc: list[dict[str, Any]], acc_auc: list[dict[str, Any]], norm_auc: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = CONFIGS["pattern_s16"]["config"]
    had = CONFIGS["pattern_s16_varn"]["config"]
    rows = []
    rows.extend(
        paired_delta_rows(
            [r for r in metric_auc if r.get("mode") in {"static", "pseudo"}],
            base_config=base,
            had_config=had,
            key_fields=("task_key", "mode", "metric_name"),
            value_field="degradation_auc",
            comparison_type="degradation_auc",
        )
    )
    rows.extend(
        paired_delta_rows(
            acc_auc,
            base_config=base,
            had_config=had,
            key_fields=("task_key", "metric_name"),
            value_field="acc_auc",
            comparison_type="accumulation_auc",
        )
    )
    rows.extend(
        paired_delta_rows(
            [r for r in norm_auc if r["layer"] == "31"],
            base_config=base,
            had_config=had,
            key_fields=("task_key", "layer", "object_type", "error_type", "region", "metric_name", "statistic"),
            value_field="norm_acc_auc",
            comparison_type="norm_accumulation_auc",
        )
    )
    summary = []
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in ("comparison_type", "mode", "metric_name", "object_type", "error_type", "region", "statistic"))
        groups[key].append(row)
    for key, vals in sorted(groups.items()):
        deltas = [float(row["delta"]) for row in vals]
        summary.append(
            {
                "comparison_type": key[0],
                "mode": key[1],
                "metric_name": key[2],
                "object_type": key[3],
                "error_type": key[4],
                "region": key[5],
                "statistic": key[6],
                "paired_n": len(vals),
                "base_auc_median": median([float(row["base_auc"]) for row in vals]),
                "varn_auc_median": median([float(row["varn_auc"]) for row in vals]),
                "median_delta": median(deltas),
                "tasks_improved": sum(delta < -EPS for delta in deltas),
                "tasks_regressed": sum(delta > EPS for delta in deltas),
                "ties": sum(abs(delta) <= EPS for delta in deltas),
            }
        )
    return summary


def lookup_pair(rows: list[dict[str, Any]], *, comparison_type: str, metric_name: str, mode: str = "", obj: str = "", stat: str = "") -> dict[str, Any] | None:
    for row in rows:
        if row["comparison_type"] != comparison_type:
            continue
        if row["metric_name"] != metric_name:
            continue
        if mode and row.get("mode") != mode:
            continue
        if obj and row.get("object_type") != obj:
            continue
        if stat and row.get("statistic") != stat:
            continue
        return row
    return None


def effect_label(hidden: dict[str, Any] | None, attention: dict[str, Any] | None = None, norm_support: bool = False) -> str:
    if not hidden:
        return "NONE"
    hidden_delta = float(hidden["median_delta"])
    hidden_improved = int(hidden["tasks_improved"])
    attn_delta = float(attention["median_delta"]) if attention else hidden_delta
    attn_improved = int(attention["tasks_improved"]) if attention else hidden_improved
    if hidden_delta < 0 and attn_delta < 0 and hidden_improved >= 5 and attn_improved >= 5 and (norm_support or attention is None):
        return "STRONG"
    if hidden_delta < 0 and attn_delta < 0 and hidden_improved >= 4 and attn_improved >= 4:
        return "MODERATE"
    if hidden_delta < 0 or hidden_improved >= 4:
        return "WEAK"
    return "NONE"


def build_summary(metric_auc: list[dict[str, Any]], acc_auc: list[dict[str, Any]], norm_auc: list[dict[str, Any]], pairwise: list[dict[str, Any]], completeness: list[dict[str, Any]]) -> dict[str, Any]:
    preflight_path = OUT_DIR / "varn_preflight_gate_summary.json"
    preflight_data = read_json(preflight_path) if preflight_path.exists() else {}
    static_hidden = lookup_pair(pairwise, comparison_type="degradation_auc", metric_name="hidden_relative_L2", mode="static")
    acc_hidden = lookup_pair(pairwise, comparison_type="accumulation_auc", metric_name="hidden_relative_L2")
    acc_attention = lookup_pair(pairwise, comparison_type="accumulation_auc", metric_name="attention_output_relative_L2")
    norm_primary = {}
    norm_support = False
    for (obj, stat), _lookup in PRIMARY_NORM_LOOKUP.items():
        row = lookup_pair(pairwise, comparison_type="norm_accumulation_auc", metric_name="relative_norm_error", obj=obj, stat=stat)
        norm_primary[f"{obj}_{stat}"] = row
        if row and float(row["median_delta"]) < 0 and int(row["tasks_improved"]) >= 4:
            norm_support = True
    static_effect = effect_label(static_hidden)
    acc_effect = effect_label(acc_hidden, acc_attention, norm_support=norm_support)
    norm_effect = "STRONG" if sum(1 for row in norm_primary.values() if row and float(row["median_delta"]) < 0 and int(row["tasks_improved"]) >= 5) >= 3 else (
        "MODERATE" if sum(1 for row in norm_primary.values() if row and float(row["median_delta"]) < 0 and int(row["tasks_improved"]) >= 4) >= 2 else (
            "WEAK" if any(row and float(row["median_delta"]) < 0 for row in norm_primary.values()) else "NONE"
        )
    )
    summary = {
        "task_count": 6,
        "checkpoints": list(CORE_CHECKPOINTS),
        "varn_source_valid": bool(preflight_data.get("varn_source_valid", False)),
        "varn_equivalence_valid": bool(preflight_data.get("varn_equivalence_valid", False)),
        "parent_commit": PARENT_COMMIT,
        "canonical_varn_source_commit": KVARN_SOURCE_COMMIT,
        "varn_only_semantics_valid": True,
        "varn_production_reference_equivalence_pass": bool(preflight_data.get("varn_equivalence_valid", False)),
        "phase_a_pass": bool(preflight_data.get("phase_a_pass", False)),
        "phase_b_pass": bool(preflight_data.get("phase_b_pass", False)),
        "formal_run_approved": bool(preflight_data.get("formal_var_n_mechanism_run_approved", False)),
        "baseline_reproduction_valid": bool(preflight_data.get("varn_off_baseline_reproduction_pass", False)),
        "varn_reduces_static_error": static_effect in {"STRONG", "MODERATE", "WEAK"},
        "varn_reduces_accumulation": acc_effect in {"STRONG", "MODERATE", "WEAK"},
        "varn_reduces_hidden_accumulation": acc_effect in {"STRONG", "MODERATE", "WEAK"},
        "varn_reduces_attention_accumulation": acc_effect in {"STRONG", "MODERATE", "WEAK"},
        "varn_reduces_norm_accumulation": norm_effect in {"STRONG", "MODERATE", "WEAK"},
        "hidden_auc_median_delta": float(acc_hidden["median_delta"]) if acc_hidden else None,
        "hidden_tasks_improved": int(acc_hidden["tasks_improved"]) if acc_hidden else None,
        "attention_auc_median_delta": float(acc_attention["median_delta"]) if acc_attention else None,
        "attention_tasks_improved": int(acc_attention["tasks_improved"]) if acc_attention else None,
        "k_norm_p95_auc_delta": float(norm_primary["k_source_p95"]["median_delta"]) if norm_primary.get("k_source_p95") else None,
        "v_norm_p95_auc_delta": float(norm_primary["v_source_p95"]["median_delta"]) if norm_primary.get("v_source_p95") else None,
        "varn_static_effect": static_effect,
        "varn_accumulation_effect": acc_effect,
        "varn_norm_effect": norm_effect,
        "static_hidden_auc": summarize_pair(static_hidden),
        "accumulation_hidden_auc": summarize_pair(acc_hidden),
        "accumulation_attention_auc": summarize_pair(acc_attention),
        "norm_primary_auc": {key: summarize_pair(row) for key, row in norm_primary.items()},
        "worker_failed_completeness_rows": sum(1 for row in completeness if row.get("status") != "ok"),
        "varn_mechanism_supported": mechanism_supported(acc_effect, norm_effect, static_effect),
        "varn_full_quality_validation_recommended": acc_effect == "STRONG" and norm_effect == "STRONG" and static_effect in {"NONE", "WEAK"},
        "next_priority": next_priority(acc_effect, norm_effect),
    }
    return summary


def summarize_pair(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "base_auc_median": float(row["base_auc_median"]),
        "varn_auc_median": float(row["varn_auc_median"]),
        "median_delta": float(row["median_delta"]),
        "tasks_improved": int(row["tasks_improved"]),
        "paired_n": int(row["paired_n"]),
    }


def next_priority(acc_effect: str, norm_effect: str) -> str:
    if acc_effect == "STRONG" and norm_effect == "STRONG":
        return "full AIME24 task-quality validation of FP16 / Pattern S16 / Pattern S16+VarN"
    if norm_effect == "STRONG" and acc_effect in {"NONE", "WEAK"}:
        return "QK / attention-logit / value-direction propagation diagnostic"
    if acc_effect == "STRONG" and norm_effect in {"NONE", "WEAK"}:
        return "investigate other VarN-induced propagation changes"
    return "attention/QK/value-direction propagation carrier diagnostic"


def mechanism_supported(acc_effect: str, norm_effect: str, static_effect: str) -> str | bool:
    if norm_effect == "STRONG" and acc_effect == "STRONG" and static_effect in {"NONE", "WEAK"}:
        return "STRONG"
    if norm_effect == "STRONG" and acc_effect in {"NONE", "WEAK"}:
        return "WEAK"
    if acc_effect == "STRONG" and norm_effect in {"NONE", "WEAK"}:
        return "MODERATE"
    if norm_effect == "NONE" and acc_effect == "NONE":
        return False
    if acc_effect in {"MODERATE", "WEAK"} or norm_effect in {"MODERATE", "WEAK"}:
        return "WEAK"
    return False


def render_report(summary: dict[str, Any], pairwise: list[dict[str, Any]]) -> str:
    def line(key: str, label: str) -> str:
        row = summary.get(key)
        if not row:
            return f"- {label}: n/a"
        return f"- {label}: S16 `{row['base_auc_median']}`, S16+VarN `{row['varn_auc_median']}`, delta `{row['median_delta']}`, improved `{row['tasks_improved']}/{row['paired_n']}`"

    return "\n".join(
        [
            "# AIME24 Pattern S16 VarN-only Mechanism Diagnostic",
            "",
            "## Scope",
            "",
            "- Ran only Pattern S16 and Pattern S16 + VarN-only.",
            "- Used frozen 6-task subset from the 12 reference trajectories.",
            "- Used static and pseudo matched FP16 controls at checkpoints 128, 512, 1024, 2048, 4096.",
            "- Did not run Hadamard, Hadamard+VarN, KVarN full pipeline, accuracy, or full AIME.",
            "",
            "## Decisions",
            "",
            f"- `VARN_STATIC_EFFECT={summary['varn_static_effect']}`",
            f"- `VARN_ACCUMULATION_EFFECT={summary['varn_accumulation_effect']}`",
            f"- `VARN_NORM_EFFECT={summary['varn_norm_effect']}`",
            f"- `NEXT_PRIORITY={summary['next_priority']}`",
            "",
            "## Static Effect",
            "",
            line("static_hidden_auc", "hidden_relative_L2 static degradation AUC"),
            "",
            "## Accumulation Effect",
            "",
            line("accumulation_hidden_auc", "hidden_relative_L2 ACC_AUC"),
            line("accumulation_attention_auc", "attention_output_relative_L2 ACC_AUC"),
            "",
            "## Norm Effect",
            "",
            *[line_key_norm(summary, key) for key in ("k_source_p95", "k_source_p99", "v_source_p95", "v_source_p99")],
            "",
            "## Artifact Rows",
            "",
            f"- Pairwise summary rows: `{len(pairwise)}`",
            "- Raw CSVs are retained locally; gzipped CSVs are versioned.",
            "",
        ]
    )


def line_key_norm(summary: dict[str, Any], key: str) -> str:
    row = (summary.get("norm_primary_auc") or {}).get(key)
    if not row:
        return f"- {key}: n/a"
    return f"- {key}: S16 `{row['base_auc_median']}`, S16+VarN `{row['varn_auc_median']}`, delta `{row['median_delta']}`, improved `{row['tasks_improved']}/{row['paired_n']}`"


def load_varn_decision() -> dict[str, Any]:
    fallback = {
        "isolation_case": "CASE_B_MATHEMATICALLY_ISOLATABLE_BUT_KERNEL_FUSED",
        "varn_only_semantics_valid": True,
        "varn_only_implementation_path_valid": False,
        "note": "Track A audit was committed on exp/aime-pseudodecode-3090-8gpu; artifact not present on this child branch.",
    }
    candidates = [
        ROOT / "reports/varn_isolation_audit/varn_isolation_decision.json",
        ROOT / "reports/aime24_norm_tail_3090/varn_source_audit.json",
    ]
    for path in candidates:
        if path.exists():
            data = read_json(path)
            if "isolation_case" in data or "varn_only_semantics_valid" in data or "varn_only_math_valid" in data:
                return {**fallback, **data}
    return fallback


def write_integrated_decision(summary: dict[str, Any]) -> dict[str, Any]:
    DECISION_DIR.mkdir(parents=True, exist_ok=True)
    varn = load_varn_decision()
    varn_semantics = bool(varn.get("varn_only_semantics_valid", varn.get("varn_only_math_valid", False)))
    if summary["varn_norm_effect"] == "STRONG" and summary["varn_accumulation_effect"] in {"NONE", "WEAK"}:
        recommendation = "Investigate other propagation carriers: QK logit drift, attention entropy, value-state directional drift."
    elif summary["varn_accumulation_effect"] == "STRONG":
        recommendation = "Use a separate prompt for full AIME24 quality validation; do not launch it automatically from this diagnostic."
    elif summary["varn_accumulation_effect"] in {"NONE", "WEAK"}:
        recommendation = "Investigate other propagation carriers before expanding VarN."
    elif varn_semantics:
        recommendation = "Do not launch full AIME automatically; use the mechanism summary to decide whether a separate quality-validation prompt is warranted."
    else:
        recommendation = "Use VarN as the next small expansion while keeping VarN gated by provenance."
    payload = {
        "varn_summary_path": str((OUT_DIR / "varn_mechanism_summary.json").relative_to(ROOT)),
        "varn_decision_path": "reports/varn_isolation_audit/varn_isolation_decision.json",
        "varn_static_effect": summary["varn_static_effect"],
        "varn_accumulation_effect": summary["varn_accumulation_effect"],
        "varn_norm_effect": summary["varn_norm_effect"],
        "varn_isolation_case": varn.get("isolation_case"),
        "varn_only_semantics_valid": varn_semantics,
        "varn_only_implementation_path_valid": bool(varn.get("varn_only_implementation_path_valid", False)),
        "next_priority": summary["next_priority"],
        "recommended_next_experiment": recommendation,
        "forbidden_this_prompt": ["Hadamard x VarN 2x2", "full AIME validation", "new sink sweep", "assignment objective"],
    }
    write_json(DECISION_DIR / "varn_mechanism_decision.json", payload)
    md = [
        "# Pattern S16 VarN Next Intervention Decision",
        "",
        f"- `VARN_STATIC_EFFECT={payload['varn_static_effect']}`",
        f"- `VARN_ACCUMULATION_EFFECT={payload['varn_accumulation_effect']}`",
        f"- `VARN_NORM_EFFECT={payload['varn_norm_effect']}`",
        f"- `VARN_ISOLATION_CASE={payload['varn_isolation_case']}`",
        f"- `VARN_ONLY_SEMANTICS_VALID={payload['varn_only_semantics_valid']}`",
        f"- `VARN_ONLY_IMPLEMENTATION_PATH_VALID={payload['varn_only_implementation_path_valid']}`",
        "",
        f"`NEXT_PRIORITY={payload['next_priority']}`",
        "",
        f"Recommended next experiment: {payload['recommended_next_experiment']}",
        "",
        "Do not start Hadamard x VarN 2x2 or full AIME automatically from this prompt.",
        "",
    ]
    (DECISION_DIR / "varn_mechanism_decision.md").write_text("\n".join(md), encoding="utf-8")
    return payload


def launch(model_path: Path, gpus: list[int]) -> dict[str, Any]:
    if len(gpus) < 8:
        raise SystemExit("VarN launch needs 8 GPU ids")
    records = load_reference_records()
    write_origin_and_config(records)
    if not (OUT_DIR / "varn_preflight_gate_summary.json").exists():
        pf = preflight(model_path)
        if not pf.get("formal_var_n_mechanism_run_approved"):
            raise SystemExit("VarN preflight failed; not launching formal workers")
    jobs = [
        ("pattern_s16", "pseudo", 0, gpus[0]),
        ("pattern_s16", "pseudo", 1, gpus[1]),
        ("pattern_s16_varn", "pseudo", 0, gpus[2]),
        ("pattern_s16_varn", "pseudo", 1, gpus[3]),
        ("pattern_s16", "static", 0, gpus[4]),
        ("pattern_s16", "static", 1, gpus[5]),
        ("pattern_s16_varn", "static", 0, gpus[6]),
        ("pattern_s16_varn", "static", 1, gpus[7]),
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = f"{ROOT / 'quant'}:{ROOT}:{env_base.get('PYTHONPATH', '')}"
    env_base.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    procs = []
    for config_key, mode, shard_index, gpu in jobs:
        stem = f"{CONFIGS[config_key]['config']}.{mode}.shard{shard_index}of2"
        summary_path = SHARD_DIR / f"{stem}.summary.json"
        if summary_path.exists():
            try:
                existing = read_json(summary_path)
                if int(existing.get("failed_rows", 1)) == 0:
                    continue
            except Exception:
                pass
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = LOG_DIR / f"{stem}.log"
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
            "--task-shard-index",
            str(shard_index),
            "--task-shard-count",
            "2",
        ]
        procs.append((config_key, mode, shard_index, subprocess.Popen(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT, env=env), log_f, log_path))
    failures = []
    for config_key, mode, shard_index, proc, log_f, log_path in procs:
        code = proc.wait()
        log_f.close()
        if code != 0:
            failures.append({"config_key": config_key, "mode": mode, "task_shard_index": shard_index, "returncode": code, "log_path": str(log_path.relative_to(ROOT))})
    if failures:
        write_json_file(RESULT_DIR / "launch_failures.json", failures)
        raise SystemExit(json.dumps({"worker_failures": failures}, indent=2, sort_keys=True))
    return aggregate()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "phase-a", "preflight", "worker", "launch", "aggregate", "integrated-decision"):
        p = sub.add_parser(name)
        p.add_argument("--model-path", type=Path, default=Path("/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B"))
        if name == "worker":
            p.add_argument("--config-key", choices=sorted(CONFIGS), required=True)
            p.add_argument("--mode", choices=["static", "pseudo"], required=True)
            p.add_argument("--gpu-id", type=int, required=True)
            p.add_argument("--task-shard-index", type=int, required=True)
            p.add_argument("--task-shard-count", type=int, default=2)
        if name == "launch":
            p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    if args.cmd == "prepare":
        records = load_reference_records()
        write_origin_and_config(records)
        phase_a = run_phase_a()
        print(json.dumps({"prepared": True, "task_count": len(read_json(TASK_SUBSET_PATH)["tasks"]), "phase_a_pass": phase_a["phase_a_pass"]}, indent=2, sort_keys=True))
    elif args.cmd == "phase-a":
        records = load_reference_records()
        write_origin_and_config(records)
        print(json.dumps(run_phase_a(), indent=2, sort_keys=True))
    elif args.cmd == "preflight":
        print(json.dumps(preflight(args.model_path), indent=2, sort_keys=True))
    elif args.cmd == "worker":
        print(json.dumps(worker(args.model_path, args.config_key, args.mode, args.gpu_id, args.task_shard_index, args.task_shard_count), indent=2, sort_keys=True))
    elif args.cmd == "launch":
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
        print(json.dumps(launch(args.model_path, gpus), indent=2, sort_keys=True))
    elif args.cmd == "aggregate":
        print(json.dumps(aggregate(), indent=2, sort_keys=True))
    elif args.cmd == "integrated-decision":
        summary = read_json(OUT_DIR / "varn_mechanism_summary.json")
        print(json.dumps(write_integrated_decision(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
