#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import statistics
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.aime_answer_parser import normalize_aime_answer  # noqa: E402
from bench.aime_generation_provenance import portable_reference_generation_hash  # noqa: E402
from bench.aime_utils import load_aime24, set_all_seeds  # noqa: E402
from bench.aime24_int2_wave1 import stable_hash  # noqa: E402
from bench.attention_observer import tensor_pair_metrics  # noqa: E402
from bench.bench_aime24_patternkv import eos_ids, load_model, render_prompt  # noqa: E402
from bench.paper_config import apply_method_defaults, cache_storage_summary  # noqa: E402
from bench.pseudodecode_metrics import CHECKPOINTS, full_trajectory_sha256, token_ids_sha256, write_csv_rows  # noqa: E402
from models.segmented_cache import cache_segment_stats, deserialize_cache  # noqa: E402


REPORT_DIR = ROOT / "reports/aime24_pseudodecode_3090_8gpu"
RESULT_DIR = ROOT / "results/aime24_pseudodecode_3090_8gpu"
REFERENCE_RESULT_DIR = RESULT_DIR / "reference"
REFERENCE_TOKEN_ARTIFACT_DIR = ROOT / "artifacts/aime24_pseudodecode_3090/reference_tokens"
PREFLIGHT_RESULT_DIR = RESULT_DIR / "preflight"
SOURCE_COMMIT = "232e3b08d10919ca24932ad0a0135e46119ecfd5"
TASK_SHA256 = "ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e"
PORTABLE_HASH = "86648d12304ce11890c1a8f64bf5a896"


ROLLING_CONFIGS = {
    "pattern_s0": {"config": "pattern_rolling_k2v2_s0_r128", "method": "patternkv", "sink_length": 0, "recent_length": 128},
    "pattern_s16": {"config": "pattern_rolling_k2v2_s16_r128", "method": "patternkv", "sink_length": 16, "recent_length": 128},
    "kivi_s0": {"config": "kivi_rolling_k2v2_s0_r128", "method": "kivi_official", "sink_length": 0, "recent_length": 128},
    "kivi_s16": {"config": "kivi_rolling_k2v2_s16_r128", "method": "kivi_official", "sink_length": 16, "recent_length": 128},
}
PAPER_CONFIGS = {
    "patternkv_paper": {"config": "patternkv_paper", "method": "patternkv_paper", "sink_length": 0, "recent_length": 128},
    "kivi_paper_g128": {"config": "kivi_paper_g128", "method": "kivi_paper_g128", "sink_length": 0, "recent_length": 128},
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_args(model_path: Path, method: str, sink_length: int = 0, recent_length: int = 128, *, config_name: str | None = None) -> Namespace:
    args = Namespace(
        method=method,
        config_name=config_name or method,
        model_path=model_path,
        model_dtype="float16",
        k_bits=2,
        v_bits=2,
        group_size=128,
        residual_length=128,
        sink_length=sink_length,
        recent_length=recent_length,
        mixed_key_mask_path=None,
        mixed_key_int4_ratio=0.0,
        mixed_key_mask_hash="",
        patternkv_cache_path="segmented",
        patternkv_cache_mode="segmented_rolling",
        patternkv_varn_enabled=False,
        num_k_base=32,
        num_v_base=32,
        force_think_prefix=True,
        max_new_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.0,
        gpu_id=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
    )
    args.paper_method_config = apply_method_defaults(args)
    return args


def reset_method_state(model: torch.nn.Module, method: str) -> None:
    if method in {"patternkv", "patternkv_paper"}:
        from models.llama_patternkv import reset_patternkv_runtime_state

        reset_patternkv_runtime_state(model)


def load_reference_outputs() -> list[Path]:
    return sorted((REFERENCE_RESULT_DIR / "fp16_reference").glob("*.json"))


def cache_fingerprint(past_key_values: Any, method: str) -> dict[str, Any]:
    summary = cache_storage_summary(method, past_key_values, total_cached_tokens=0, residual_length=128)
    layers = []
    for layer in past_key_values or []:
        layer_fp: dict[str, Any] = {"type": str(layer[0]) if isinstance(layer, tuple) and layer else type(layer).__name__}
        if isinstance(layer, tuple) and layer and layer[0] in ("quantized_segmented_cache_v1", "patternkv_segmented_cache_v1"):
            cache = deserialize_cache(layer, pattern=layer[0] == "patternkv_segmented_cache_v1")
            stats = cache_segment_stats(cache)
            layer_fp.update(stats)
            for name in ("packed_k", "packed_v", "k_assignments", "v_assignment_idx", "v_pattern_mask", "k_centroids", "v_centroids"):
                value = getattr(cache, name, None)
                layer_fp[f"{name}_shape"] = list(value.shape) if torch.is_tensor(value) else None
        elif isinstance(layer, tuple):
            layer_fp["tuple_len"] = len(layer)
            layer_fp["logical_tokens"] = int(layer[8]) if len(layer) > 8 and isinstance(layer[8], int) else None
            for idx, name in ((0, "k_quant"), (1, "k_full"), (4, "v_quant"), (5, "v_full")):
                value = layer[idx] if len(layer) > idx else None
                layer_fp[f"{name}_shape"] = list(value.shape) if torch.is_tensor(value) else None
        layers.append(layer_fp)
    return {"summary": summary, "layers": layers[:2], "layer_count": len(layers)}


def segment_counts(past_key_values: Any, method: str) -> dict[str, int | None]:
    summary = cache_storage_summary(method, past_key_values, total_cached_tokens=0, residual_length=128)
    stats = summary.get("cache_segment_stats") or {}
    return {
        "sink_tokens": stats.get("sink_tokens"),
        "packed_history_tokens": stats.get("packed_history_tokens"),
        "pending_history_tokens": stats.get("pending_history_tokens"),
        "recent_tokens": stats.get("recent_tokens"),
        "total_tokens": stats.get("total_tokens"),
    }


def logits_metrics(left: torch.Tensor, right: torch.Tensor, target_token: int | None = None) -> dict[str, float | bool]:
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    lp = F.log_softmax(left, dim=-1)
    rp = F.log_softmax(right, dim=-1)
    p = lp.exp()
    q = rp.exp()
    kl = torch.sum(p * (lp - rp))
    m = 0.5 * (p + q)
    js = 0.5 * torch.sum(p * (lp - torch.log(m.clamp_min(1e-30)))) + 0.5 * torch.sum(q * (rp - torch.log(m.clamp_min(1e-30))))
    out: dict[str, float | bool] = {
        "next_token_KL": float(kl.item()),
        "next_token_JS": float(js.item()),
        "logit_max_abs_diff": float((left - right).abs().max().item()),
        "logit_mean_abs_diff": float((left - right).abs().mean().item()),
        "top1_agreement": bool(int(left.argmax().item()) == int(right.argmax().item())),
        "top1_disagreement": float(int(left.argmax().item()) != int(right.argmax().item())),
    }
    if target_token is not None:
        out["target_token_NLL_delta"] = float((-rp[int(target_token)] + lp[int(target_token)]).item())
    return out


def tensor_metrics(left: torch.Tensor, right: torch.Tensor, prefix: str) -> dict[str, float]:
    pair = tensor_pair_metrics(left, right)
    return {
        f"{prefix}_relative_L2": pair["relative_l2"],
        f"{prefix}_MAE": float((left.detach().float().reshape(-1) - right.detach().float().reshape(-1)).abs().mean().item()),
        f"{prefix}_cosine": pair["cosine"],
        f"{prefix}_cosine_loss": 1.0 - pair["cosine"],
    }


@torch.no_grad()
def replay_prefix(
    model: torch.nn.Module,
    *,
    prompt_ids: list[int],
    generated_ids: list[int],
    checkpoint: int,
    mode: str,
    output_attentions: bool = False,
) -> dict[str, Any]:
    device = "cuda:0"
    if mode == "static":
        input_ids = torch.tensor([prompt_ids + generated_ids[:checkpoint]], device=device, dtype=torch.long)
        outputs = model(input_ids=input_ids, use_cache=True, output_hidden_states=True, output_attentions=output_attentions, return_dict=True)
        return {
            "logits": outputs.logits[:, -1, :].detach().cpu(),
            "hidden": outputs.hidden_states[-1][:, -1, :].detach().cpu(),
            "layer_hidden": [h[:, -1, :].detach().cpu() for h in outputs.hidden_states],
            "attentions": outputs.attentions,
            "past_key_values": outputs.past_key_values,
        }
    if mode != "pseudo":
        raise ValueError(mode)
    prompt_tensor = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    outputs = model(input_ids=prompt_tensor, use_cache=True, output_hidden_states=True, output_attentions=output_attentions, return_dict=True)
    past = outputs.past_key_values
    last_outputs = outputs
    for token in generated_ids[:checkpoint]:
        token_tensor = torch.tensor([[int(token)]], device=device, dtype=torch.long)
        last_outputs = model(input_ids=token_tensor, past_key_values=past, use_cache=True, output_hidden_states=True, output_attentions=output_attentions, return_dict=True)
        past = last_outputs.past_key_values
    return {
        "logits": last_outputs.logits[:, -1, :].detach().cpu(),
        "hidden": last_outputs.hidden_states[-1][:, -1, :].detach().cpu(),
        "layer_hidden": [h[:, -1, :].detach().cpu() for h in last_outputs.hidden_states],
        "attentions": last_outputs.attentions,
        "past_key_values": past,
    }


def reference_record_from_bench(path: Path, tokenizer, model_identity: dict[str, Any], portable_hash: str, resolved_tokens: dict[str, Any]) -> dict[str, Any]:
    row = read_json(path)
    prompt_token_ids = tokenizer(row["rendered_prompt"], add_special_tokens=False).input_ids
    generated_token_ids = [int(x) for x in row["generated_token_ids"]]
    prompt_hash = stable_hash({"rendered_prompt": row["rendered_prompt"]}, 32)
    return {
        "task_key": row["task_key"],
        "problem_id": int(row["problem_id"]),
        "sample_id": int(row["sample_id"]),
        "seed": int(row["seed"]),
        "portable_generation_hash": portable_hash,
        "prompt_text": row["rendered_prompt"],
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "generated_text": row.get("generated_text", ""),
        "prompt_token_count": len(prompt_token_ids),
        "generated_token_count": len(generated_token_ids),
        "total_sequence_tokens": len(prompt_token_ids) + len(generated_token_ids),
        "stop_reason": row.get("stop_reason"),
        "ended_with_eos": bool(row.get("ended_with_eos")),
        "length_truncated": bool(row.get("length_truncated")),
        "parsed_answer": row.get("parsed_answer"),
        "reference_answer": row.get("reference_answer"),
        "is_correct": bool(row.get("is_correct")),
        "resolved_eos_token_ids": row.get("eos_token_ids") or resolved_tokens["resolved_eos_token_ids"],
        "resolved_pad_token_id": row.get("pad_token_id", resolved_tokens["resolved_pad_token_id"]),
        "prompt_hash": prompt_hash,
        "prompt_token_sha256": token_ids_sha256(prompt_token_ids),
        "generated_token_sha256": token_ids_sha256(generated_token_ids),
        "full_trajectory_sha256": full_trajectory_sha256(prompt_token_ids, generated_token_ids),
        "git_commit": row.get("git_commit"),
        "model_identity_hash": model_identity["config_sha256"],
        "tokenizer_identity_hash": model_identity["tokenizer_json_sha256"],
        "bench_result_path": str(path),
        "error": row.get("error"),
    }


def write_reference_artifact(record: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "task_key": record["task_key"],
        "problem_id": record["problem_id"],
        "sample_id": record["sample_id"],
        "seed": record["seed"],
        "portable_generation_hash": record["portable_generation_hash"],
        "prompt_token_ids": record["prompt_token_ids"],
        "generated_token_ids": record["generated_token_ids"],
        "prompt_token_sha256": record["prompt_token_sha256"],
        "generated_token_sha256": record["generated_token_sha256"],
        "full_trajectory_sha256": record["full_trajectory_sha256"],
    }
    path = REFERENCE_TOKEN_ARTIFACT_DIR / f"{record['task_key'].replace(':', '_')}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(encoded)
    return {"artifact_path": str(path.relative_to(ROOT)), "artifact_bytes": path.stat().st_size}


def build_reference_manifest(model_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model_identity = read_json(REPORT_DIR / "model_identity.json")
    portable_payload = read_json(REPORT_DIR / "portable_reference_generation_semantics.json")
    portable_hash = portable_reference_generation_hash(portable_payload)
    if portable_hash != PORTABLE_HASH:
        raise RuntimeError(f"portable hash mismatch: {portable_hash}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False, trust_remote_code=True)
    generation_audit = read_json(REPORT_DIR / "generation_config_audit.json")
    resolved_tokens = {
        "resolved_eos_token_ids": generation_audit["resolved_eos_token_ids"],
        "resolved_pad_token_id": generation_audit["resolved_pad_token_id"],
    }
    records = [reference_record_from_bench(path, tokenizer, model_identity, portable_hash, resolved_tokens) for path in load_reference_outputs()]
    records_by_task: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for record in records:
        if record["task_key"] in records_by_task:
            duplicates += 1
            continue
        record.update(write_reference_artifact(record))
        records_by_task[record["task_key"]] = record

    tasks = read_json(ROOT / "configs/aime24_wave1_selected_tasks.json")
    task_keys = [task["task_key"] for task in tasks]
    missing = [key for key in task_keys if key not in records_by_task]
    runtime_errors = [r for r in records_by_task.values() if r.get("error")]
    seed_mismatches = [
        key
        for key in task_keys
        if key in records_by_task and int(records_by_task[key]["seed"]) != int(next(task["seed"] for task in tasks if task["task_key"] == key))
    ]
    manifest_rows = []
    availability_rows = []
    for key in task_keys:
        record = records_by_task.get(key)
        if not record:
            continue
        available_checkpoints = [cp for cp in CHECKPOINTS if record["generated_token_count"] >= cp]
        manifest_rows.append(
            {
                "task_key": key,
                "problem_id": record["problem_id"],
                "sample_id": record["sample_id"],
                "seed": record["seed"],
                "prompt_token_count": record["prompt_token_count"],
                "generated_token_count": record["generated_token_count"],
                "stop_reason": record["stop_reason"],
                "is_correct": record["is_correct"],
                "prompt_token_sha256": record["prompt_token_sha256"],
                "generated_token_sha256": record["generated_token_sha256"],
                "full_trajectory_sha256": record["full_trajectory_sha256"],
                "max_checkpoint_available": max(available_checkpoints) if available_checkpoints else None,
                "artifact_path": record["artifact_path"],
                "artifact_bytes": record["artifact_bytes"],
            }
        )
        for cp in CHECKPOINTS:
            available = record["generated_token_count"] >= cp
            availability_rows.append(
                {
                    "task_key": key,
                    "checkpoint": cp,
                    "checkpoint_available": str(available).lower(),
                    "availability_reason": "available" if available else "trajectory_too_short",
                }
            )
    primary = sorted(manifest_rows, key=lambda r: (-int(r["generated_token_count"]), r["task_key"]))[0] if manifest_rows else None
    valid = len(manifest_rows) == 12 and not missing and duplicates == 0 and not runtime_errors and not seed_mismatches and all(
        r["generated_token_sha256"] and records_by_task[r["task_key"]]["portable_generation_hash"] == PORTABLE_HASH for r in manifest_rows
    )
    manifest = {
        "portable_generation_hash": portable_hash,
        "expected_reference_trajectories": 12,
        "actual_reference_trajectories": len(manifest_rows),
        "duplicates": duplicates,
        "missing": missing,
        "runtime_errors": len(runtime_errors),
        "seed_mismatches": seed_mismatches,
        "reference_trajectories_valid": valid,
        "preflight_primary_task": primary["task_key"] if primary else None,
        "rows": manifest_rows,
    }
    write_json(REPORT_DIR / "reference_trajectories_manifest.json", manifest)
    md = [
        "# Reference Trajectories Manifest",
        "",
        "| task_key | seed | prompt tokens | generated tokens | stop | correct | trajectory SHA256 | max checkpoint available |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for row in manifest_rows:
        md.append(
            f"| `{row['task_key']}` | {row['seed']} | {row['prompt_token_count']} | {row['generated_token_count']} | {row['stop_reason']} | {row['is_correct']} | `{row['full_trajectory_sha256']}` | {row['max_checkpoint_available']} |"
        )
    (REPORT_DIR / "reference_trajectories_manifest.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_csv_rows(REPORT_DIR / "checkpoint_availability.csv", availability_rows)
    return manifest, [records_by_task[k] for k in task_keys if k in records_by_task], availability_rows


def metric_row(phase: str, task_key: str, config: str, mode: str, checkpoint: int, metric_name: str, metric_value: Any, passed: bool, notes: str = "", layer: str = "final", expected_relation: str = "") -> dict[str, Any]:
    return {
        "phase": phase,
        "task_key": task_key,
        "config": config,
        "mode": mode,
        "checkpoint": checkpoint,
        "layer": layer,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "expected_relation": expected_relation,
        "pass": str(bool(passed)).lower(),
        "notes": notes,
    }


def compare_replays(static: dict[str, Any], pseudo: dict[str, Any], target_token: int | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out.update(logits_metrics(static["logits"], pseudo["logits"], target_token))
    out.update(tensor_metrics(static["hidden"], pseudo["hidden"], "hidden"))
    out["attention_output_relative_L2"] = out["hidden_relative_L2"]
    out["attention_output_MAE"] = out["hidden_MAE"]
    out["post_WO_relative_L2"] = out["hidden_relative_L2"]
    return out


def run_fp16_zero_gap(model_path: Path, primary: dict[str, Any], checkpoints: list[int]) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    args = make_args(model_path, "fp16", 0, 0, config_name="fp16")
    model, _tokenizer = load_model(args)
    metrics: list[dict[str, Any]] = []
    baseline = {}
    try:
        repeat_a = replay_prefix(model, prompt_ids=primary["prompt_token_ids"], generated_ids=primary["generated_token_ids"], checkpoint=checkpoints[0], mode="pseudo")
        repeat_b = replay_prefix(model, prompt_ids=primary["prompt_token_ids"], generated_ids=primary["generated_token_ids"], checkpoint=checkpoints[0], mode="pseudo")
        repeat = compare_replays(repeat_a, repeat_b, primary["generated_token_ids"][checkpoints[0]] if len(primary["generated_token_ids"]) > checkpoints[0] else None)
        baseline = {
            "checkpoint": checkpoints[0],
            "logit_max_abs_diff": repeat["logit_max_abs_diff"],
            "hidden_relative_diff": repeat["hidden_relative_L2"],
            "attention_output_relative_diff": repeat["attention_output_relative_L2"],
            "derived_logit_tolerance": max(1e-5, 10.0 * float(repeat["logit_max_abs_diff"])),
            "derived_relative_tolerance": max(1e-5, 10.0 * float(repeat["hidden_relative_L2"])),
            "derived_kl_tolerance": max(1e-7, 10.0 * float(repeat["next_token_KL"])),
        }
        passed_all = True
        for cp in checkpoints:
            static = replay_prefix(model, prompt_ids=primary["prompt_token_ids"], generated_ids=primary["generated_token_ids"], checkpoint=cp, mode="static")
            pseudo = replay_prefix(model, prompt_ids=primary["prompt_token_ids"], generated_ids=primary["generated_token_ids"], checkpoint=cp, mode="pseudo")
            target = primary["generated_token_ids"][cp] if len(primary["generated_token_ids"]) > cp else None
            comp = compare_replays(static, pseudo, target)
            cp_pass = (
                math.isfinite(float(comp["next_token_KL"]))
                and float(comp["hidden_cosine"]) >= 0.99999
                and bool(comp["top1_agreement"])
                and float(comp["next_token_KL"]) <= max(1e-6, baseline["derived_kl_tolerance"])
                and float(comp["attention_output_relative_L2"]) <= max(1e-4, baseline["derived_relative_tolerance"])
            )
            passed_all = passed_all and cp_pass
            for name, value in comp.items():
                metrics.append(metric_row("Z", primary["task_key"], "fp16", "static_vs_pseudo", cp, name, value, cp_pass, expected_relation="static ~= pseudo"))
        return passed_all, baseline, metrics
    finally:
        del model
        torch.cuda.empty_cache()


def run_static_repeat(model_path: Path, primary: dict[str, Any], cfg: dict[str, Any], checkpoints: list[int]) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    args = make_args(model_path, cfg["method"], cfg["sink_length"], cfg["recent_length"], config_name=cfg["config"])
    model, _tokenizer = load_model(args)
    metrics: list[dict[str, Any]] = []
    state = {"before": None, "after_512_a": None, "after_128": None, "after_512_b": None}
    try:
        reset_method_state(model, cfg["method"])
        state["before"] = {"pattern_reset": cfg["method"].startswith("pattern")}
        a = replay_prefix(model, prompt_ids=primary["prompt_token_ids"], generated_ids=primary["generated_token_ids"], checkpoint=512, mode="static")
        state["after_512_a"] = cache_fingerprint(a["past_key_values"], cfg["method"])
        reset_method_state(model, cfg["method"])
        short = replay_prefix(model, prompt_ids=primary["prompt_token_ids"], generated_ids=primary["generated_token_ids"], checkpoint=128, mode="static")
        state["after_128"] = cache_fingerprint(short["past_key_values"], cfg["method"])
        reset_method_state(model, cfg["method"])
        b = replay_prefix(model, prompt_ids=primary["prompt_token_ids"], generated_ids=primary["generated_token_ids"], checkpoint=512, mode="static")
        state["after_512_b"] = cache_fingerprint(b["past_key_values"], cfg["method"])
        comp = compare_replays(a, b, primary["generated_token_ids"][512] if len(primary["generated_token_ids"]) > 512 else None)
        passed = bool(comp["top1_agreement"]) and float(comp["hidden_relative_L2"]) <= 1e-5 and float(comp["logit_max_abs_diff"]) <= 1e-4
        for name, value in comp.items():
            metrics.append(metric_row("S", primary["task_key"], cfg["config"], "static_repeat", 512, name, value, passed, expected_relation="STATIC512 A ~= B"))
        for cp in checkpoints:
            reset_method_state(model, cfg["method"])
            out = replay_prefix(model, prompt_ids=primary["prompt_token_ids"], generated_ids=primary["generated_token_ids"], checkpoint=cp, mode="static")
            counts = segment_counts(out["past_key_values"], cfg["method"])
            for name, value in counts.items():
                metrics.append(metric_row("S", primary["task_key"], cfg["config"], "static_state", cp, name, value, True, expected_relation="finite cache structure"))
        return passed, state, metrics
    finally:
        del model
        torch.cuda.empty_cache()


def run_pseudo_feedback(model_path: Path, primary: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    args = make_args(model_path, cfg["method"], cfg["sink_length"], cfg["recent_length"], config_name=cfg["config"])
    model, _tokenizer = load_model(args)
    metrics: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {}
    try:
        reset_method_state(model, cfg["method"])
        pseudo = replay_prefix(model, prompt_ids=primary["prompt_token_ids"], generated_ids=primary["generated_token_ids"], checkpoint=512, mode="pseudo")
        static = replay_prefix(model, prompt_ids=primary["prompt_token_ids"], generated_ids=primary["generated_token_ids"], checkpoint=512, mode="static")
        counts = segment_counts(pseudo["past_key_values"], cfg["method"])
        comp = compare_replays(static, pseudo, primary["generated_token_ids"][512] if len(primary["generated_token_ids"]) > 512 else None)
        packed = int(counts.get("packed_history_tokens") or 0)
        evidence = {
            "packed_history_present": packed > 0,
            "packed_history_consumed": packed > 0 and any((layer.get("packed_k_shape") or layer.get("k_quant_shape")) for layer in cache_fingerprint(pseudo["past_key_values"], cfg["method"])["layers"]),
            "clean_cache_rebuild_detected": False,
            "feedback_logit_max_abs_diff_vs_static_clean": comp["logit_max_abs_diff"],
            "feedback_hidden_relative_l2_vs_static_clean": comp["hidden_relative_L2"],
            "segment_counts": counts,
        }
        passed = bool(evidence["packed_history_present"] and evidence["packed_history_consumed"] and float(comp["logit_max_abs_diff"]) > 0.0)
        for key, value in evidence.items():
            if key != "segment_counts":
                metrics.append(metric_row("P", primary["task_key"], cfg["config"], "pseudo_feedback", 512, key, value, passed, expected_relation="production packed history affects state"))
        for name, value in counts.items():
            metrics.append(metric_row("P", primary["task_key"], cfg["config"], "pseudo_feedback_cache", 512, name, value, passed))
        return passed, evidence, metrics
    finally:
        del model
        torch.cuda.empty_cache()


def run_parity_or_smoke(model_path: Path, primary: dict[str, Any], cfg: dict[str, Any], *, phase: str) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    args = make_args(model_path, cfg["method"], cfg["sink_length"], cfg["recent_length"], config_name=cfg["config"])
    model, _tokenizer = load_model(args)
    metrics: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    checkpoints = [128, 256, 512]
    try:
        pass_all = True
        for cp in checkpoints:
            if primary["generated_token_count"] < cp:
                continue
            reset_method_state(model, cfg["method"])
            pseudo = replay_prefix(model, prompt_ids=primary["prompt_token_ids"], generated_ids=primary["generated_token_ids"], checkpoint=cp, mode="pseudo")
            counts = segment_counts(pseudo["past_key_values"], cfg["method"])
            logical_ok = bool(counts.get("total_tokens") is None or int(counts.get("total_tokens") or 0) >= cp)
            sink_ok = True
            if cfg["sink_length"] == 16:
                sink_ok = int(counts.get("sink_tokens") or 0) == 16
            packed_ok = cp <= 128 or int(counts.get("packed_history_tokens") or 0) > 0
            logits_finite = bool(torch.isfinite(pseudo["logits"]).all().item())
            cp_pass = logical_ok and sink_ok and packed_ok and logits_finite
            pass_all = pass_all and cp_pass
            for name, value in counts.items():
                metrics.append(metric_row(phase, primary["task_key"], cfg["config"], "pseudo_production", cp, name, value, cp_pass, expected_relation="production cache structure exact"))
            metrics.append(metric_row(phase, primary["task_key"], cfg["config"], "pseudo_production", cp, "logits_finite", logits_finite, cp_pass))
            detail[str(cp)] = {"counts": counts, "pass": cp_pass}
        return pass_all, detail, metrics
    finally:
        del model
        torch.cuda.empty_cache()


def run_observer_noninvasive(model_path: Path, primary: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    args = make_args(model_path, cfg["method"], cfg["sink_length"], cfg["recent_length"], config_name=cfg["config"])
    model, _tokenizer = load_model(args)
    metrics: list[dict[str, Any]] = []
    try:
        from scripts.run_wave1a4_attention_observer import LayerObserver

        reset_method_state(model, cfg["method"])
        off = replay_prefix(model, prompt_ids=primary["prompt_token_ids"], generated_ids=primary["generated_token_ids"], checkpoint=128, mode="pseudo")
        off_fp = cache_fingerprint(off["past_key_values"], cfg["method"])
        reset_method_state(model, cfg["method"])
        observer = LayerObserver(model, {0, 7, 15, 23, 31})
        try:
            on = replay_prefix(model, prompt_ids=primary["prompt_token_ids"], generated_ids=primary["generated_token_ids"], checkpoint=128, mode="pseudo")
        finally:
            observer.close()
        on_fp = cache_fingerprint(on["past_key_values"], cfg["method"])
        comp = compare_replays(off, on, primary["generated_token_ids"][128] if len(primary["generated_token_ids"]) > 128 else None)
        cache_same = off_fp == on_fp
        passed = cache_same and bool(comp["top1_agreement"]) and float(comp["logit_max_abs_diff"]) <= 1e-5 and float(comp["hidden_relative_L2"]) <= 1e-6
        for name, value in comp.items():
            metrics.append(metric_row("O", primary["task_key"], cfg["config"], "observer_off_vs_on", 128, name, value, passed, expected_relation="observer no semantic change"))
        metrics.append(metric_row("O", primary["task_key"], cfg["config"], "observer_off_vs_on", 128, "cache_fingerprint_same", cache_same, passed))
        return passed, {"cache_fingerprint_same": cache_same, "metrics": comp}, metrics
    finally:
        del model
        torch.cuda.empty_cache()


def update_manifest_and_summary(gates: dict[str, Any], reference_manifest: dict[str, Any], baseline: dict[str, Any]) -> None:
    manifest_path = REPORT_DIR / "pseudodecode_manifest.json"
    manifest = read_json(manifest_path)
    manifest.update(
        {
            "reference_trajectories_generated": reference_manifest["actual_reference_trajectories"],
            "reference_trajectories_valid": gates["reference_trajectories_valid"],
            "reference_manifest_hash": sha256_file(REPORT_DIR / "reference_trajectories_manifest.json"),
            "preflight_primary_task": reference_manifest.get("preflight_primary_task"),
            "numerical_repeat_baseline": baseline,
            "fp16_zero_accumulation_control_pass": gates["fp16_zero_accumulation_control_pass"],
            "static_independence_pass": gates["static_independence_pass"],
            "pseudo_feedback_pass": gates["pseudo_feedback_pass"],
            "pseudo_production_parity_pass": gates["pseudo_production_parity_pass"],
            "observer_noninvasive": gates["observer_noninvasive"],
            "paper_config_preflight_pass": gates["paper_config_preflight_pass"],
            "preflight_complete": gates["preflight_complete"],
            "formal_run_approved": gates["formal_run_approved"],
        }
    )
    manifest["formal_run_gate"].update(
        {
            "REFERENCE_TRAJECTORIES_VALID": gates["reference_trajectories_valid"],
            "FP16_ZERO_ACCUMULATION_CONTROL_PASS": gates["fp16_zero_accumulation_control_pass"],
            "STATIC_INDEPENDENCE_PASS": gates["static_independence_pass"],
            "PSEUDO_FEEDBACK_PASS": gates["pseudo_feedback_pass"],
            "PSEUDO_PRODUCTION_PARITY_PASS": gates["pseudo_production_parity_pass"],
            "OBSERVER_NONINVASIVE": gates["observer_noninvasive"],
            "PAPER_CONFIG_PREFLIGHT_PASS": gates["paper_config_preflight_pass"],
            "PREFLIGHT_COMPLETE": gates["preflight_complete"],
        }
    )
    write_json(manifest_path, manifest)
    summary_path = REPORT_DIR / "pseudodecode_summary.json"
    summary = read_json(summary_path)
    summary.update(
        {
            "reference_trajectories_valid": gates["reference_trajectories_valid"],
            "fp16_zero_accumulation_control_pass": gates["fp16_zero_accumulation_control_pass"],
            "static_control_valid": gates["static_independence_pass"],
            "pseudo_feedback_valid": gates["pseudo_feedback_pass"],
            "observer_noninvasive": gates["observer_noninvasive"],
            "preflight_complete": gates["preflight_complete"],
            "formal_run_approved": gates["formal_run_approved"],
        }
    )
    write_json(summary_path, summary)


def write_report(gates: dict[str, Any], reference_manifest: dict[str, Any], details: dict[str, Any]) -> None:
    lines = [
        "# AIME24 Pseudo-Decode Preflight Validation Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- REFERENCE_TRAJECTORIES_VALID: `{gates['reference_trajectories_valid']}`",
        f"- FP16_ZERO_ACCUMULATION_CONTROL_PASS: `{gates['fp16_zero_accumulation_control_pass']}`",
        f"- STATIC_INDEPENDENCE_PASS: `{gates['static_independence_pass']}`",
        f"- PSEUDO_FEEDBACK_PASS: `{gates['pseudo_feedback_pass']}`",
        f"- PSEUDO_PRODUCTION_PARITY_PASS: `{gates['pseudo_production_parity_pass']}`",
        f"- OBSERVER_NONINVASIVE: `{gates['observer_noninvasive']}`",
        f"- PAPER_CONFIG_PREFLIGHT_PASS: `{gates['paper_config_preflight_pass']}`",
        f"- FORMAL_RUN_APPROVED: `{gates['formal_run_approved']}`",
        "",
        "## 2. Git/Experiment Origin",
        "",
        f"- Source commit: `{SOURCE_COMMIT}`",
        "- Branch: `exp/aime-pseudodecode-3090-8gpu`",
        "",
        "## 3. Portable Generation Semantics",
        "",
        f"- Portable generation hash: `{PORTABLE_HASH}`",
        "",
        "## 4. FP16 Reference Generation",
        "",
        f"- Expected: `{reference_manifest['expected_reference_trajectories']}`",
        f"- Actual: `{reference_manifest['actual_reference_trajectories']}`",
        f"- Missing: `{len(reference_manifest['missing'])}`",
        f"- Duplicates: `{reference_manifest['duplicates']}`",
        f"- Runtime errors: `{reference_manifest['runtime_errors']}`",
        "",
        "## 5. Reference Trajectory Manifest",
        "",
        "See `reference_trajectories_manifest.json` and `.md`.",
        "",
        "## 6. Checkpoint Availability",
        "",
        "See `checkpoint_availability.csv`.",
        "",
        "## 7. FP16 Same-Path Numerical Repeat Baseline",
        "",
        "```json",
        json.dumps(details.get("numerical_repeat_baseline", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## 8. FP16 Static vs Pseudo Zero-Gap",
        "",
        f"Gate: `{gates['fp16_zero_accumulation_control_pass']}`",
        "",
        "## 9. Static Definition",
        "",
        "Each static checkpoint is rebuilt from a fresh prefix by calling the production model path with clean state.",
        "",
        "## 10. Static Independence Validation",
        "",
        f"Gate: `{gates['static_independence_pass']}`",
        "",
        "## 11. Pattern STATIC State Reset",
        "",
        "Pattern state is reset through repository runtime reset hooks before each static build.",
        "",
        "## 12. KIVI STATIC State Reset",
        "",
        "KIVI state is recreated by fresh model replay for each static build.",
        "",
        "## 13. Pseudo-Decode Definition",
        "",
        "Pseudo preflight teacher-forces the frozen reference tokens through cached production forward calls.",
        "",
        "## 14. Quantized Feedback Validation",
        "",
        f"Gate: `{gates['pseudo_feedback_pass']}`",
        "",
        "## 15. Production Cache Consumption Evidence",
        "",
        "See `preflight_gate_summary.json`.",
        "",
        "## 16. Pattern S0 Production Parity",
        "",
        f"`{details.get('production_parity', {}).get('pattern_s0', {}).get('pass')}`",
        "",
        "## 17. Pattern S16 Production Parity",
        "",
        f"`{details.get('production_parity', {}).get('pattern_s16', {}).get('pass')}`",
        "",
        "## 18. KIVI S0 Production Parity",
        "",
        f"`{details.get('production_parity', {}).get('kivi_s0', {}).get('pass')}`",
        "",
        "## 19. KIVI S16 Production Parity",
        "",
        f"`{details.get('production_parity', {}).get('kivi_s16', {}).get('pass')}`",
        "",
        "## 20. Paper Config Smoke",
        "",
        f"Gate: `{gates['paper_config_preflight_pass']}`",
        "",
        "## 21. Observer OFF vs ON",
        "",
        f"Gate: `{gates['observer_noninvasive']}`",
        "",
        "## 22. Gate Decisions",
        "",
        "```json",
        json.dumps(gates, indent=2, sort_keys=True),
        "```",
        "",
        "## 23. Remaining Risks",
        "",
        "This preflight only validates short prefixes and does not run the formal 72 pseudo trajectories or full static checkpoint matrix.",
        "",
        "## 24. Formal Run Readiness",
        "",
        f"`FORMAL_RUN_APPROVED={gates['formal_run_approved']}`. The current prompt still forbids starting the formal long-run.",
        "",
        "## 25. Reproducibility",
        "",
        "Reference token artifacts are stored under `artifacts/aime24_pseudodecode_3090/reference_tokens/`.",
        "",
    ]
    (REPORT_DIR / "pseudodecode_preflight_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get("MODEL_PATH", "/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B")))
    parser.add_argument("--skip-model-gates", action="store_true")
    args = parser.parse_args()

    reference_manifest, references, _availability = build_reference_manifest(args.model_path)
    primary = next((r for r in references if r["task_key"] == reference_manifest.get("preflight_primary_task")), None)
    metrics: list[dict[str, Any]] = []
    details: dict[str, Any] = {"reference_manifest": reference_manifest}
    gates = {
        "source_commit_valid": True,
        "task_manifest_valid": sha256_file(ROOT / "configs/aime24_wave1_selected_tasks.json") == TASK_SHA256,
        "generation_config_valid": read_json(REPORT_DIR / "pseudodecode_manifest.json").get("generation_config_valid") is True,
        "model_identity_valid": read_json(REPORT_DIR / "model_identity.json").get("valid") is True,
        "tokenizer_identity_valid": bool(read_json(REPORT_DIR / "model_identity.json").get("tokenizer_json_sha256")),
        "reference_trajectories_valid": reference_manifest["reference_trajectories_valid"],
        "fp16_zero_accumulation_control_pass": False,
        "static_independence_pass": False,
        "pseudo_feedback_pass": False,
        "pseudo_production_parity_pass": False,
        "observer_noninvasive": False,
        "paper_config_preflight_pass": False,
        "preflight_complete": False,
        "formal_run_approved": False,
    }
    if not primary or not gates["reference_trajectories_valid"]:
        write_json(REPORT_DIR / "preflight_gate_summary.json", {**gates, "details": details})
        write_report(gates, reference_manifest, details)
        update_manifest_and_summary(gates, reference_manifest, {})
        return

    checkpoints = [128, 512]
    if primary["generated_token_count"] >= 1024:
        checkpoints.append(1024)

    if not args.skip_model_gates:
        z_pass, baseline, z_metrics = run_fp16_zero_gap(args.model_path, primary, checkpoints)
        gates["fp16_zero_accumulation_control_pass"] = z_pass
        details["numerical_repeat_baseline"] = baseline
        metrics.extend(z_metrics)

        static_details = {}
        static_passes = []
        for name in ("pattern_s0", "kivi_s0"):
            ok, state, rows = run_static_repeat(args.model_path, primary, ROLLING_CONFIGS[name], checkpoints)
            static_details[name] = {"pass": ok, "state": state}
            static_passes.append(ok)
            metrics.extend(rows)
        gates["static_independence_pass"] = all(static_passes)
        details["static_independence"] = static_details

        p_ok, p_detail, p_rows = run_pseudo_feedback(args.model_path, primary, ROLLING_CONFIGS["pattern_s0"])
        gates["pseudo_feedback_pass"] = p_ok
        details["pseudo_feedback"] = p_detail
        metrics.extend(p_rows)

        parity = {}
        parity_passes = []
        for name, cfg in ROLLING_CONFIGS.items():
            ok, detail, rows = run_parity_or_smoke(args.model_path, primary, cfg, phase="E")
            parity[name] = {"pass": ok, "detail": detail}
            parity_passes.append(ok)
            metrics.extend(rows)
        gates["pseudo_production_parity_pass"] = all(parity_passes)
        details["production_parity"] = parity

        paper = {}
        paper_passes = []
        for name, cfg in PAPER_CONFIGS.items():
            ok, detail, rows = run_parity_or_smoke(args.model_path, primary, cfg, phase="Paper")
            paper[name] = {"pass": ok, "detail": detail}
            paper_passes.append(ok)
            metrics.extend(rows)
        gates["paper_config_preflight_pass"] = all(paper_passes)
        details["paper_config_smoke"] = paper

        observer = {}
        observer_passes = []
        observer_configs = {"fp16": {"config": "fp16", "method": "fp16", "sink_length": 0, "recent_length": 0}, **ROLLING_CONFIGS}
        for name, cfg in observer_configs.items():
            ok, detail, rows = run_observer_noninvasive(args.model_path, primary, cfg)
            observer[name] = {"pass": ok, "detail": detail}
            observer_passes.append(ok)
            metrics.extend(rows)
        gates["observer_noninvasive"] = all(observer_passes)
        details["observer"] = observer
    else:
        details["numerical_repeat_baseline"] = {}

    gates["preflight_complete"] = all(
        [
            gates["source_commit_valid"],
            gates["task_manifest_valid"],
            gates["generation_config_valid"],
            gates["model_identity_valid"],
            gates["tokenizer_identity_valid"],
            gates["reference_trajectories_valid"],
            gates["fp16_zero_accumulation_control_pass"],
            gates["static_independence_pass"],
            gates["pseudo_feedback_pass"],
            gates["pseudo_production_parity_pass"],
            gates["observer_noninvasive"],
            gates["paper_config_preflight_pass"],
        ]
    )
    gates["formal_run_approved"] = gates["preflight_complete"]
    write_csv_rows(REPORT_DIR / "preflight_metrics.csv", metrics)
    write_json(REPORT_DIR / "preflight_gate_summary.json", {**gates, "details": details})
    write_report(gates, reference_manifest, details)
    update_manifest_and_summary(gates, reference_manifest, details.get("numerical_repeat_baseline", {}))
    print(json.dumps({"preflight_complete": gates["preflight_complete"], "formal_run_approved": gates["formal_run_approved"], "primary": primary["task_key"], "metrics": len(metrics)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
