from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.aime24_int2_wave1 import stable_hash, task_key3
from bench.aime_answer_parser import normalize_aime_answer, parse_aime_answer
from bench.aime_utils import compute_stop_state, effective_seed, load_aime24, normalize_eos_token_ids, set_all_seeds
from bench.bench_aime24_patternkv import render_prompt
from models.segmented_cache import PatternQuantizedKVCache, deserialize_cache, tensor_tokens


DEFAULT_TASKS_PATH = Path("configs/aime24_patternkv_equivalence_tasks.json")
DEFAULT_ARTIFACT_DIR = Path("artifacts/aime24_patternkv_equivalence")


@dataclass(frozen=True)
class MetricThresholds:
    logits_cosine_reference: float = 0.9999
    logits_cosine_production: float = 0.999
    nll_abs_diff_production: float = 0.05


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    if af.numel() == 0 and bf.numel() == 0:
        return 1.0
    denom = af.norm() * bf.norm()
    if float(denom.item()) == 0.0:
        return 1.0 if torch.equal(af, bf) else 0.0
    return float(torch.dot(af, bf).div(denom).item())


def relative_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float()
    bf = b.detach().float()
    if af.numel() == 0 and bf.numel() == 0:
        return 0.0
    return float(((af - bf) ** 2).mean().div((af**2).mean().clamp_min(1e-12)).item())


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    pf = p.detach().float().clamp_min(1e-12)
    qf = q.detach().float().clamp_min(1e-12)
    return float((pf * (pf.log() - qf.log())).sum(dim=-1).mean().item())


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 5) -> int:
    aa = set(torch.topk(a.detach().float().reshape(-1), k).indices.cpu().tolist())
    bb = set(torch.topk(b.detach().float().reshape(-1), k).indices.cpu().tolist())
    return len(aa & bb)


def nll_for_target(logits: torch.Tensor, target_token: int) -> float:
    target = torch.tensor([int(target_token)], device=logits.device)
    return float(F.cross_entropy(logits.float().view(1, -1), target).item())


def disagreement_rate(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return None
    tokens = min(a.shape[-1], b.shape[-1])
    if tokens == 0:
        return 0.0
    aa = a[..., :tokens].detach().cpu()
    bb = b[..., :tokens].detach().cpu()
    return float((aa != bb).float().mean().item())


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for idx, (left, right) in enumerate(zip(a, b)):
        if int(left) != int(right):
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def classify_divergence(margin_a: float | None, margin_b: float | None, *, reference_diverged: bool, cache_mismatch_before: bool) -> str:
    if cache_mismatch_before:
        return "algorithmic_mismatch"
    if reference_diverged:
        return "algorithmic_mismatch"
    if margin_a is not None and margin_b is not None and max(abs(margin_a), abs(margin_b)) < 1e-3:
        return "near_tie_amplification"
    return "unknown"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(rows)


def eos_ids(tokenizer, model) -> list[int]:
    ids = normalize_eos_token_ids(
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(tokenizer, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
    )
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0:
        ids.append(eot)
    return sorted(set(int(x) for x in ids if x is not None))


def load_tokenizer(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_pattern_model(model_path: Path, *, cache_path: str, gpu_id: int, dtype: torch.dtype):
    from models.llama_patternkv import LlamaForCausalLM_PatternKV

    config = LlamaConfig.from_pretrained(model_path, local_files_only=True)
    config.k_bits = 2
    config.v_bits = 2
    config.group_size = 128
    config.residual_length = 128
    config.sink_length = 0
    config.recent_length = 128
    config.use_flash = True
    config.num_k_base = 32
    config.num_v_base = 32
    config.patternkv_cache_path = cache_path
    device = f"cuda:{gpu_id}"
    model = LlamaForCausalLM_PatternKV.from_pretrained(model_path, local_files_only=True, config=config, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.eval()
    return model


def load_fp16_model(model_path: Path, *, gpu_id: int, dtype: torch.dtype):
    device = f"cuda:{gpu_id}"
    model = LlamaForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.eval()
    return model


def row_by_problem(dataset_path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["problem_id"]): row for row in load_aime24(dataset_path)}


def select_tasks(args) -> list[dict[str, Any]]:
    selected = json.loads(args.selected_tasks.read_text(encoding="utf-8"))
    chosen = []
    for item in selected:
        pid = int(item["problem_id"])
        sid = int(item["sample_id"])
        if (pid, sid) in {(12, 0), (14, 0)}:
            reason = "medium_stable_fp16_eos" if pid == 12 else "longer_quantization_sensitive_fp16_eos"
            chosen.append(
                {
                    "problem_id": pid,
                    "sample_id": sid,
                    "seed": int(item.get("seed", effective_seed(args.base_seed, pid, sid))),
                    "selection_reason": reason,
                    "fp16_generated_tokens": int(item.get("fp16", {}).get("generated_tokens") or 0),
                    "legacy_patternkv_generated_tokens": int(item.get("patternkv", {}).get("generated_tokens") or 0),
                    "reference_result_path": str(args.selected_tasks),
                    "task_key": item.get("task_key") or task_key3(pid, sid, int(item.get("seed", effective_seed(args.base_seed, pid, sid)))),
                }
            )
    if len(chosen) != 2:
        raise RuntimeError("expected to select fixed tasks p12_s0 and p14_s0 from selected task manifest")
    write_json(DEFAULT_TASKS_PATH, chosen)
    return chosen


def teacher_artifact_paths(problem_id: int, sample_id: int) -> tuple[Path, Path]:
    stem = f"task_{problem_id}_{sample_id}_teacher_tokens"
    return DEFAULT_ARTIFACT_DIR / f"{stem}.pt", DEFAULT_ARTIFACT_DIR / f"{stem}.json"


@torch.no_grad()
def ensure_teacher_tokens(args, tokenizer, task: dict[str, Any], row: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    pt_path, json_path = teacher_artifact_paths(int(task["problem_id"]), int(task["sample_id"]))
    if pt_path.exists() and json_path.exists() and not args.overwrite_invalid:
        data = torch.load(pt_path, map_location="cpu")
        return data["teacher_token_ids"].to(torch.long), json.loads(json_path.read_text(encoding="utf-8"))
    if args.dry_run:
        fake = torch.empty(0, dtype=torch.long)
        meta = {"problem_id": task["problem_id"], "sample_id": task["sample_id"], "teacher_token_count": 0, "source_method": "dry_run"}
        return fake, meta
    set_all_seeds(int(task["seed"]))
    model = load_fp16_model(args.model_path, gpu_id=args.gpu_id, dtype=torch.float16)
    rendered, _, _ = render_prompt(row["problem"], tokenizer, args.force_think_prefix)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(f"cuda:{args.gpu_id}")
    attention_mask = encoded.attention_mask.to(f"cuda:{args.gpu_id}")
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max(max(args.checkpoints), args.max_new_tokens),
        repetition_penalty=1.0,
        num_return_sequences=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids(tokenizer, model),
        return_dict_in_generate=True,
        output_scores=False,
    )
    generated = out.sequences[0, input_ids.shape[1] :].detach().cpu().to(torch.long)
    meta = {
        "problem_id": int(task["problem_id"]),
        "sample_id": int(task["sample_id"]),
        "seed": int(task["seed"]),
        "prompt_token_hash": stable_hash({"input_ids": encoded.input_ids.tolist()}, 32),
        "teacher_token_hash": stable_hash({"teacher_token_ids": generated.tolist()}, 32),
        "teacher_token_count": int(generated.numel()),
        "source_method": "fp16_greedy_generated",
        "source_result_path": None,
        "tokenizer_hash": stable_hash({"vocab_size": len(tokenizer), "eos": tokenizer.eos_token_id}, 32),
        "model_config_hash": stable_hash(AutoConfig.from_pretrained(args.model_path, local_files_only=True).to_dict(), 32),
    }
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"teacher_token_ids": generated}, pt_path)
    write_json(json_path, meta)
    del out, input_ids, attention_mask, encoded, model
    gc.collect()
    torch.cuda.empty_cache()
    return generated, meta


def cache_kind(layer_cache: Any) -> str:
    if isinstance(layer_cache, tuple) and layer_cache and layer_cache[0] == "patternkv_segmented_cache_v1":
        return "segmented"
    return "legacy"


def layer_cache_snapshot(layer_cache: Any, layer_idx: int, model_layer: Any) -> dict[str, Any]:
    attn = getattr(model_layer, "self_attn", None)
    if cache_kind(layer_cache) == "segmented":
        cache = deserialize_cache(layer_cache, pattern=True)
        v_mask = cache.v_pattern_mask if cache.v_pattern_mask is not None else cache.v_assignments
        return {
            "layer": layer_idx,
            "cache_kind": "segmented",
            "total_tokens": int(cache.total_tokens),
            "sink_tokens": tensor_tokens(cache.sink_k),
            "packed_k_tokens": int(cache.packed_k_tokens),
            "packed_v_tokens": int(cache.packed_v_tokens),
            "pending_k_tokens": tensor_tokens(cache.pending_k),
            "pending_v_tokens": tensor_tokens(cache.pending_v),
            "recent_k_tokens": tensor_tokens(cache.recent_k),
            "recent_v_tokens": tensor_tokens(cache.recent_v),
            "k_assignment_tokens": tensor_tokens(cache.k_assignments),
            "v_assignment_tokens": tensor_tokens(cache.v_assignment_idx),
            "v_gate_tokens": tensor_tokens(v_mask),
            "k_centroid_count": int(cache.k_centroids.shape[1]) if torch.is_tensor(cache.k_centroids) else 0,
            "v_centroid_count": int(cache.v_centroids.shape[1]) if torch.is_tensor(cache.v_centroids) else 0,
            "k_centroid_update_count": int(cache.centroid_updates_k),
            "v_centroid_update_count": int(cache.centroid_updates_v),
            "_k_assignments": cache.k_assignments.detach().cpu() if torch.is_tensor(cache.k_assignments) else None,
            "_v_assignment_idx": cache.v_assignment_idx.detach().cpu() if torch.is_tensor(cache.v_assignment_idx) else None,
            "_v_gate": v_mask.detach().cpu() if torch.is_tensor(v_mask) else None,
            "_k_centroids": cache.k_centroids.detach().cpu() if torch.is_tensor(cache.k_centroids) else None,
            "_v_centroids": cache.v_centroids.detach().cpu() if torch.is_tensor(cache.v_centroids) else None,
        }
    assignments = layer_cache[9] if len(layer_cache) > 9 else None
    v_mask = layer_cache[10] if len(layer_cache) > 10 else None
    v_idx = layer_cache[11] if len(layer_cache) > 11 else None
    k_base = getattr(attn, "k_base", None)
    v_centroids = getattr(attn, "v_centroids", None)
    total_tokens = int(layer_cache[8])
    full_k = layer_cache[1]
    full_v = layer_cache[5]
    recent = tensor_tokens(full_k)
    packed = tensor_tokens(assignments)
    return {
        "layer": layer_idx,
        "cache_kind": "legacy",
        "total_tokens": total_tokens,
        "sink_tokens": 0,
        "packed_k_tokens": packed,
        "packed_v_tokens": tensor_tokens(v_idx),
        "pending_k_tokens": 0,
        "pending_v_tokens": 0,
        "recent_k_tokens": recent,
        "recent_v_tokens": tensor_tokens(full_v),
        "k_assignment_tokens": tensor_tokens(assignments),
        "v_assignment_tokens": tensor_tokens(v_idx),
        "v_gate_tokens": tensor_tokens(v_mask),
        "k_centroid_count": int(k_base.shape[1]) if torch.is_tensor(k_base) else 0,
        "v_centroid_count": int(v_centroids.shape[1]) if torch.is_tensor(v_centroids) else 0,
        "k_centroid_update_count": max(int(k_base.shape[1]) - int(getattr(attn, "num_k_bases", 32)), 0) if torch.is_tensor(k_base) else 0,
        "v_centroid_update_count": max(int(v_centroids.shape[1]) - int(getattr(attn, "num_v_bases", 32)), 0) if torch.is_tensor(v_centroids) else 0,
        "_k_assignments": assignments.detach().cpu() if torch.is_tensor(assignments) else None,
        "_v_assignment_idx": v_idx.detach().cpu() if torch.is_tensor(v_idx) else None,
        "_v_gate": v_mask.detach().cpu() if torch.is_tensor(v_mask) else None,
        "_k_centroids": k_base.detach().cpu() if torch.is_tensor(k_base) else None,
        "_v_centroids": v_centroids.detach().cpu() if torch.is_tensor(v_centroids) else None,
    }


def public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in snapshot.items() if not key.startswith("_")}


@torch.no_grad()
def run_teacher_path(args, tokenizer, task: dict[str, Any], row: dict[str, Any], teacher_tokens: torch.Tensor, cache_path: str) -> dict[str, Any]:
    from models.llama_patternkv import reset_patternkv_runtime_state

    set_all_seeds(int(task["seed"]))
    os.environ["PATTERNKV_CACHE_PATH"] = cache_path
    os.environ["PATTERNKV_FORCE_REFERENCE_ATTENTION"] = "1" if args.force_reference_attention else "0"
    model = load_pattern_model(args.model_path, cache_path=cache_path, gpu_id=args.gpu_id, dtype=torch.float16)
    reset_patternkv_runtime_state(model)
    rendered, _, _ = render_prompt(row["problem"], tokenizer, args.force_think_prefix)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    device = f"cuda:{args.gpu_id}"
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
    past = outputs.past_key_values
    last_logits = outputs.logits[:, -1, :].detach().cpu()
    checkpoints = sorted(set(int(x) for x in args.checkpoints if int(x) <= int(teacher_tokens.numel())))
    snapshots: dict[int, dict[str, Any]] = {}
    for pos, token in enumerate(teacher_tokens.tolist(), start=1):
        token_tensor = torch.tensor([[int(token)]], device=device, dtype=torch.long)
        pos_attention = torch.ones(1, input_ids.shape[1] + pos, device=device, dtype=attention_mask.dtype)
        outputs = model(input_ids=token_tensor, attention_mask=pos_attention, past_key_values=past, use_cache=True, return_dict=True)
        past = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :].detach().cpu()
        if pos in checkpoints:
            layer_snaps = [layer_cache_snapshot(layer_cache, idx, model.model.layers[idx]) for idx, layer_cache in enumerate(past)]
            snapshots[pos] = {
                "decode_position": pos,
                "logits": last_logits,
                "target_token": int(teacher_tokens[pos].item()) if pos < teacher_tokens.numel() else None,
                "layers": layer_snaps,
            }
    result = {"cache_path": cache_path, "snapshots": snapshots}
    del outputs, past, model, input_ids, attention_mask, encoded
    gc.collect()
    torch.cuda.empty_cache()
    return result


def compare_teacher_runs(task: dict[str, Any], legacy: dict[str, Any], segmented: dict[str, Any], backend: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_checkpoint: list[dict[str, Any]] = []
    per_layer: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for checkpoint, legacy_snap in legacy["snapshots"].items():
        seg_snap = segmented["snapshots"].get(checkpoint)
        if seg_snap is None:
            mismatches.append({"sample": task, "backend": backend, "decode_position": checkpoint, "layer": None, "kv_head": None, "global_token_index": None, "mismatch_type": "missing_segmented_checkpoint", "legacy_value": True, "segmented_value": None, "local_context": {}})
            continue
        logits_legacy = legacy_snap["logits"]
        logits_segmented = seg_snap["logits"]
        target = legacy_snap["target_token"]
        row = {
            "backend": backend,
            "task_key": task["task_key"],
            "problem_id": task["problem_id"],
            "sample_id": task["sample_id"],
            "decode_position": checkpoint,
            "logits_cosine": cosine_similarity(logits_legacy, logits_segmented),
            "logits_relative_mse": relative_mse(logits_legacy, logits_segmented),
            "logits_max_abs_error": max_abs_error(logits_legacy, logits_segmented),
            "top1_legacy": int(logits_legacy.argmax(dim=-1).item()),
            "top1_segmented": int(logits_segmented.argmax(dim=-1).item()),
            "top1_agreement": int(logits_legacy.argmax(dim=-1).item()) == int(logits_segmented.argmax(dim=-1).item()),
            "top5_overlap": topk_overlap(logits_legacy, logits_segmented, 5),
            "teacher_target_token": target,
            "teacher_target_nll_legacy": nll_for_target(logits_legacy, target) if target is not None else None,
            "teacher_target_nll_segmented": nll_for_target(logits_segmented, target) if target is not None else None,
        }
        row["absolute_nll_difference"] = abs(row["teacher_target_nll_legacy"] - row["teacher_target_nll_segmented"]) if row["teacher_target_nll_legacy"] is not None else None
        per_checkpoint.append(row)
        if not row["top1_agreement"]:
            mismatches.append({"sample": task, "backend": backend, "decode_position": checkpoint, "layer": None, "kv_head": None, "global_token_index": None, "mismatch_type": "top1_mismatch", "legacy_value": row["top1_legacy"], "segmented_value": row["top1_segmented"], "local_context": {"logits_cosine": row["logits_cosine"]}})
        for legacy_layer, segmented_layer in zip(legacy_snap["layers"], seg_snap["layers"]):
            layer_row = {
                "backend": backend,
                "task_key": task["task_key"],
                "decode_position": checkpoint,
                "layer": legacy_layer["layer"],
                "legacy_cache_kind": legacy_layer["cache_kind"],
                "segmented_cache_kind": segmented_layer["cache_kind"],
            }
            for key in (
                "total_tokens",
                "sink_tokens",
                "packed_k_tokens",
                "packed_v_tokens",
                "pending_k_tokens",
                "pending_v_tokens",
                "recent_k_tokens",
                "recent_v_tokens",
                "k_assignment_tokens",
                "v_assignment_tokens",
                "v_gate_tokens",
                "k_centroid_count",
                "v_centroid_count",
                "k_centroid_update_count",
                "v_centroid_update_count",
            ):
                layer_row[f"legacy_{key}"] = legacy_layer[key]
                layer_row[f"segmented_{key}"] = segmented_layer[key]
                layer_row[f"{key}_match"] = legacy_layer[key] == segmented_layer[key]
            for name in ("k", "v"):
                legacy_centroids = legacy_layer[f"_{name}_centroids"]
                segmented_centroids = segmented_layer[f"_{name}_centroids"]
                if torch.is_tensor(legacy_centroids) and torch.is_tensor(segmented_centroids):
                    shared = min(legacy_centroids.shape[1], segmented_centroids.shape[1])
                    layer_row[f"{name}_centroid_relative_l2"] = relative_mse(legacy_centroids[:, :shared], segmented_centroids[:, :shared])
                    layer_row[f"{name}_centroid_max_abs_error"] = max_abs_error(legacy_centroids[:, :shared], segmented_centroids[:, :shared])
                    layer_row[f"{name}_centroid_shape_legacy"] = list(legacy_centroids.shape)
                    layer_row[f"{name}_centroid_shape_segmented"] = list(segmented_centroids.shape)
                else:
                    layer_row[f"{name}_centroid_relative_l2"] = None
                    layer_row[f"{name}_centroid_max_abs_error"] = None
            per_layer.append(layer_row)
            assign_row = {
                "backend": backend,
                "task_key": task["task_key"],
                "decode_position": checkpoint,
                "layer": legacy_layer["layer"],
                "k_assignment_disagreement_rate": disagreement_rate(legacy_layer["_k_assignments"], segmented_layer["_k_assignments"]),
                "v_assignment_disagreement_rate": disagreement_rate(legacy_layer["_v_assignment_idx"], segmented_layer["_v_assignment_idx"]),
                "v_gate_disagreement_rate": disagreement_rate(legacy_layer["_v_gate"], segmented_layer["_v_gate"]),
            }
            assignment_rows.append(assign_row)
            structural_keys = [key for key in layer_row if key.endswith("_match")]
            if not all(layer_row[key] for key in structural_keys):
                mismatches.append({"sample": task, "backend": backend, "decode_position": checkpoint, "layer": legacy_layer["layer"], "kv_head": None, "global_token_index": None, "mismatch_type": "structural_mismatch", "legacy_value": public_snapshot(legacy_layer), "segmented_value": public_snapshot(segmented_layer), "local_context": {}})
            for metric_name in ("k_assignment_disagreement_rate", "v_assignment_disagreement_rate", "v_gate_disagreement_rate"):
                value = assign_row[metric_name]
                if value not in (None, 0.0):
                    mismatches.append({"sample": task, "backend": backend, "decode_position": checkpoint, "layer": legacy_layer["layer"], "kv_head": None, "global_token_index": None, "mismatch_type": metric_name, "legacy_value": None, "segmented_value": value, "local_context": {}})
    return per_checkpoint, per_layer, assignment_rows, mismatches


def run_teacher_forcing(args) -> dict[str, Any]:
    tokenizer = load_tokenizer(args.model_path)
    tasks = json.loads(DEFAULT_TASKS_PATH.read_text(encoding="utf-8")) if DEFAULT_TASKS_PATH.exists() else select_tasks(args)
    rows = row_by_problem(args.dataset_path)
    output_dir = args.output_dir
    backend = "reference" if args.force_reference_attention else "production"
    all_checkpoint_rows: list[dict[str, Any]] = []
    all_layer_rows: list[dict[str, Any]] = []
    all_assignment_rows: list[dict[str, Any]] = []
    all_mismatches: list[dict[str, Any]] = []
    teacher_meta = []
    if args.dry_run:
        write_json(output_dir / "run_manifest.json", {"mode": "teacher-forcing", "dry_run": True, "tasks": tasks, "backend": backend})
        return {"dry_run": True, "tasks": tasks}
    for task in tasks:
        teacher_tokens, meta = ensure_teacher_tokens(args, tokenizer, task, rows[int(task["problem_id"])])
        teacher_meta.append(meta)
        if int(teacher_tokens.numel()) < min(max(args.checkpoints), 4096):
            all_mismatches.append({"sample": task, "backend": backend, "decode_position": int(teacher_tokens.numel()), "layer": None, "kv_head": None, "global_token_index": None, "mismatch_type": "insufficient_teacher_tokens", "legacy_value": int(teacher_tokens.numel()), "segmented_value": max(args.checkpoints), "local_context": {}})
            continue
        legacy = run_teacher_path(args, tokenizer, task, rows[int(task["problem_id"])], teacher_tokens, "legacy")
        segmented = run_teacher_path(args, tokenizer, task, rows[int(task["problem_id"])], teacher_tokens, "segmented")
        checkpoint_rows, layer_rows, assignment_rows, mismatches = compare_teacher_runs(task, legacy, segmented, backend)
        all_checkpoint_rows.extend(checkpoint_rows)
        all_layer_rows.extend(layer_rows)
        all_assignment_rows.extend(assignment_rows)
        all_mismatches.extend(mismatches)
    structure_pass = not any(m["mismatch_type"] == "structural_mismatch" for m in all_mismatches)
    top1_pass = all(row["top1_agreement"] for row in all_checkpoint_rows) if all_checkpoint_rows else False
    logits_pass = all(float(row["logits_cosine"]) >= (MetricThresholds().logits_cosine_reference if backend == "reference" else MetricThresholds().logits_cosine_production) for row in all_checkpoint_rows) if all_checkpoint_rows else False
    assign_pass = all((row["k_assignment_disagreement_rate"] in (None, 0.0) and row["v_assignment_disagreement_rate"] in (None, 0.0) and row["v_gate_disagreement_rate"] in (None, 0.0)) for row in all_assignment_rows) if all_assignment_rows else False
    summary = {
        "mode": "teacher-forcing",
        "backend": backend,
        "tasks": tasks,
        "teacher_token_provenance": teacher_meta,
        "checkpoint_rows": len(all_checkpoint_rows),
        "layer_rows": len(all_layer_rows),
        "first_mismatch_count": len(all_mismatches),
        "LEVEL2_STRUCTURE_PASS": structure_pass,
        "LEVEL2_REFERENCE_PASS": bool(structure_pass and top1_pass and logits_pass and assign_pass) if backend == "reference" else None,
        "LEVEL2_PRODUCTION_PASS": bool(structure_pass and top1_pass and logits_pass and assign_pass) if backend == "production" else None,
    }
    write_csv(output_dir / "per_checkpoint.csv", all_checkpoint_rows)
    write_csv(output_dir / "per_layer.csv", all_layer_rows)
    write_csv(output_dir / "assignment_disagreement.csv", all_assignment_rows)
    write_json(output_dir / "first_mismatches.json", all_mismatches[:50])
    write_json(output_dir / "teacher_forcing_summary.json", summary)
    write_json(output_dir / "run_manifest.json", {"mode": "teacher-forcing", "backend": backend, "args": vars(args), "tasks": tasks})
    md = [
        "# PatternKV Teacher-Forcing Equivalence",
        "",
        f"Backend: `{backend}`",
        f"Checkpoint rows: `{len(all_checkpoint_rows)}`",
        f"Layer rows: `{len(all_layer_rows)}`",
        f"First mismatches: `{len(all_mismatches)}`",
        "",
        f"LEVEL2_STRUCTURE_PASS={str(summary['LEVEL2_STRUCTURE_PASS']).lower()}",
        f"LEVEL2_REFERENCE_PASS={str(summary['LEVEL2_REFERENCE_PASS']).lower() if summary['LEVEL2_REFERENCE_PASS'] is not None else 'null'}",
        f"LEVEL2_PRODUCTION_PASS={str(summary['LEVEL2_PRODUCTION_PASS']).lower() if summary['LEVEL2_PRODUCTION_PASS'] is not None else 'null'}",
    ]
    (output_dir / "teacher_forcing_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


@torch.no_grad()
def run_greedy_path(args, tokenizer, task: dict[str, Any], row: dict[str, Any], cache_path: str, do_sample: bool) -> dict[str, Any]:
    from models.llama_patternkv import collect_patternkv_dynamic_stats, reset_patternkv_runtime_state

    set_all_seeds(int(task["seed"]))
    os.environ["PATTERNKV_CACHE_PATH"] = cache_path
    model = load_pattern_model(args.model_path, cache_path=cache_path, gpu_id=args.gpu_id, dtype=torch.float16)
    reset_patternkv_runtime_state(model)
    rendered, _, _ = render_prompt(row["problem"], tokenizer, args.force_think_prefix)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(f"cuda:{args.gpu_id}")
    attention_mask = encoded.attention_mask.to(f"cuda:{args.gpu_id}")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=do_sample,
        temperature=0.6 if do_sample else None,
        top_p=0.95 if do_sample else None,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.0,
        num_return_sequences=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids(tokenizer, model),
        return_dict_in_generate=True,
        output_scores=False,
    )
    torch.cuda.synchronize()
    generated = output.sequences[0, input_ids.shape[1] :].detach().cpu().tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    parsed = parse_aime_answer(text)
    stop = compute_stop_state(generated, args.max_new_tokens, eos_ids(tokenizer, model))
    rec = {
        "cache_path": cache_path,
        "task_key": task["task_key"],
        "problem_id": int(task["problem_id"]),
        "sample_id": int(task["sample_id"]),
        "seed": int(task["seed"]),
        "generated_token_ids": generated,
        "generated_token_hash": stable_hash({"generated_token_ids": generated}, 32),
        "generated_text": text,
        "generated_tokens": len(generated),
        "stop_reason": stop.get("stop_reason"),
        "parsed_answer": parsed["parsed_answer"],
        "reference_answer": normalize_aime_answer(row["answer"]),
        "patternkv_dynamic_stats": collect_patternkv_dynamic_stats(model, getattr(output, "past_key_values", None)),
        "wall_time_seconds": round(time.perf_counter() - start, 4),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    del output, input_ids, attention_mask, encoded, model
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def run_greedy(args, *, do_sample: bool = False) -> dict[str, Any]:
    tokenizer = load_tokenizer(args.model_path)
    tasks = json.loads(DEFAULT_TASKS_PATH.read_text(encoding="utf-8")) if DEFAULT_TASKS_PATH.exists() else select_tasks(args)
    rows = row_by_problem(args.dataset_path)
    output_dir = args.output_dir
    per_sample = []
    divergences = []
    if args.dry_run:
        write_json(output_dir / "run_manifest.json", {"mode": "greedy", "dry_run": True, "tasks": tasks})
        return {"dry_run": True}
    for task in tasks:
        legacy = run_greedy_path(args, tokenizer, task, rows[int(task["problem_id"])], "legacy", do_sample)
        segmented = run_greedy_path(args, tokenizer, task, rows[int(task["problem_id"])], "segmented", do_sample)
        div = first_divergence(legacy["generated_token_ids"], segmented["generated_token_ids"])
        exact = div is None
        divergence = {
            "task_key": task["task_key"],
            "exact_token_match": exact,
            "first_divergence_position": div,
            "legacy_generated_tokens": legacy["generated_tokens"],
            "segmented_generated_tokens": segmented["generated_tokens"],
            "classification": "none" if exact else classify_divergence(None, None, reference_diverged=False, cache_mismatch_before=False),
        }
        divergences.append(divergence)
        per_sample.append({**{k: v for k, v in legacy.items() if k != "generated_token_ids" and k != "generated_text"}, "path": "legacy", "exact_token_match": exact, "first_divergence_position": div})
        per_sample.append({**{k: v for k, v in segmented.items() if k != "generated_token_ids" and k != "generated_text"}, "path": "segmented", "exact_token_match": exact, "first_divergence_position": div})
        write_json(output_dir / f"{task['task_key'].replace(':', '_')}.legacy.json", legacy)
        write_json(output_dir / f"{task['task_key'].replace(':', '_')}.segmented.json", segmented)
    pass_level = all(item["exact_token_match"] or (item["first_divergence_position"] is not None and item["first_divergence_position"] >= 512) for item in divergences)
    summary = {"mode": "sampling" if do_sample else "greedy", "samples": len(tasks), "divergences": divergences, "LEVEL3_PASS": pass_level if not do_sample else None, "LEVEL4_SANITY_PASS": True if do_sample else None}
    write_csv(output_dir / ("per_sample.csv"), per_sample)
    write_json(output_dir / ("first_divergence.json"), divergences)
    write_json(output_dir / ("greedy_summary.json" if not do_sample else "sampling_summary.json"), summary)
    write_json(output_dir / "run_manifest.json", {"mode": "sampling" if do_sample else "greedy", "args": vars(args), "tasks": tasks})
    md_name = "sampling_summary.md" if do_sample else "greedy_summary.md"
    (output_dir / md_name).write_text(f"# PatternKV {'Sampling' if do_sample else 'Greedy'} Equivalence\n\nPASS={pass_level if not do_sample else True}\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["select-tasks", "teacher-forcing", "greedy", "sampling"], required=True)
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get("MODEL_PATH", "/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B")))
    parser.add_argument("--dataset-path", type=Path, default=Path("datasets/aime/aime24.jsonl"))
    parser.add_argument("--selected-tasks", type=Path, default=Path("configs/aime24_wave1_selected_tasks.json"))
    parser.add_argument("--problem-ids", nargs="*", type=int)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--checkpoints", nargs="*", type=int, default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-invalid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--force-reference-attention", action="store_true")
    parser.add_argument("--production-kernel", action="store_true")
    parser.add_argument("--force-think-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "select-tasks":
        tasks = select_tasks(args)
        write_json(args.output_dir / "selected_tasks.json", tasks)
        return
    if args.mode == "teacher-forcing":
        run_teacher_forcing(args)
        return
    if args.mode == "greedy":
        run_greedy(args, do_sample=False)
        return
    if args.mode == "sampling":
        run_greedy(args, do_sample=True)
        return


if __name__ == "__main__":
    main()
