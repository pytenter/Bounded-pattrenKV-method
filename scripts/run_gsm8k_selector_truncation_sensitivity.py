#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, LlamaConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.gsm8k_paper_utils import build_prompt, compute_stop_state, config_hash, extract_reference, is_complete, load_gsm8k, parse_prediction, result_path, sha256_file, write_json_atomic
from bench.paper_config import apply_method_defaults, cache_storage_summary, method_config_dict
from models.llama_patternkv import collect_patternkv_dynamic_stats, reset_patternkv_runtime_state
from scripts.run_gsm8k_selector_components_pilot import METHOD_LABELS, METHODS, SELECTORS, shared_patternkv_config


EXPERIMENT_ID = "gsm8k_selector_truncation_sensitivity_v1"
DIAGNOSTIC_TYPE = "truncation_union"
SOURCE_PILOT = "gsm8k_selector_components_pilot_v1"
SOURCE_PILOT_COMMIT = "bd5b82659111fbe0b464b2170ed53aa38b2a33b5"
DEFAULT_MODEL = "/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct"
DEFAULT_OLD_REPORT_DIR = "reports/gsm8k_selector_components_pilot_v1"
DEFAULT_OLD_RESULT_DIR = "/data/zypan/Bounded-pattrenKV-gsm8k-selector-components-pilot-v1/results/gsm8k_selector_components_pilot_v1/pilot"
OLD_MAX_NEW_TOKENS = 2048
NEW_MAX_NEW_TOKENS = 8192


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_text(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()


def nvidia_query() -> list[dict[str, Any]]:
    query = "index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu"
    try:
        out = subprocess.check_output(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True)
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        rows.append({"index": int(parts[0]), "uuid": parts[1], "name": parts[2], "memory_total_mib": int(parts[3]), "memory_used_mib": int(parts[4]), "memory_free_mib": int(parts[5]), "gpu_util_percent": int(parts[6])})
    return rows


def gpu_uuid(physical_gpu: str) -> str | None:
    for row in nvidia_query():
        if str(row["index"]) == str(physical_gpu):
            return str(row["uuid"])
    return None


def generated_ids_hash(ids: list[int]) -> str:
    return config_hash({"generated_token_ids": [int(x) for x in ids]})


def v4_dynamic_summary(dynamic_stats: dict[str, Any]) -> dict[str, Any]:
    v4 = [x for x in (dynamic_stats.get("v_precision_v4_tokens_per_layer") or []) if x is not None]
    total = [x for x in (dynamic_stats.get("v_precision_total_tokens_per_layer") or []) if x is not None]
    selected = int(sum(int(x) for x in v4)) if v4 else None
    eligible = int(sum(int(x) for x in total)) if total else None
    fraction = float(selected / eligible) if selected is not None and eligible else None
    return {"selected_v4_tokens": selected, "eligible_historical_tokens": eligible, "observed_v4_fraction": fraction}


def old_record_by_method_pid(old_result_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = {m: {} for m in METHODS}
    for method in METHODS:
        for path in sorted((old_result_dir / method).glob("p*.json")):
            rec = read_json(path)
            rec["_old_path"] = str(path)
            out[method][int(rec["problem_id"])] = rec
    return out


def is_truncated(row: dict[str, Any]) -> bool:
    return row.get("stop_reason") == "length" or bool(row.get("hit_max_new_tokens"))


def freeze_union(args: argparse.Namespace) -> None:
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    final_gate = read_json(args.old_report_dir / "final_gate.json")
    old = old_record_by_method_pid(args.old_result_dir)
    union_ids = sorted({pid for method in METHODS for pid, row in old[method].items() if is_truncated(row)})
    rows = []
    for pid in union_ids:
        entry: dict[str, Any] = {"problem_id": pid, "selected_because_method_truncated": [m for m in METHODS if is_truncated(old[m][pid])]}
        for method in METHODS:
            row = old[method][pid]
            prefix_ids = row.get("generated_token_ids") or []
            entry[f"{method}_old_stop_reason"] = row.get("stop_reason")
            entry[f"{method}_old_generated_tokens"] = row.get("generated_tokens")
            entry[f"{method}_old_is_correct"] = row.get("is_correct")
            entry[f"{method}_old_parsed_answer"] = row.get("parsed_answer")
            entry[f"{method}_old_generated_token_hash"] = row.get("generated_token_ids_sha256") or generated_ids_hash(prefix_ids)
            entry[f"{method}_old_config_hash"] = row.get("config_hash")
            entry[f"{method}_old_path"] = row.get("_old_path")
        rows.append(entry)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_type": DIAGNOSTIC_TYPE,
        "source_pilot": SOURCE_PILOT,
        "source_pilot_commit": SOURCE_PILOT_COMMIT,
        "created_at": utc_now(),
        "old_report_dir": str(args.old_report_dir),
        "old_result_dir": str(args.old_result_dir),
        "selection_rule": "Include problem_id iff any old method has stop_reason == 'length' or hit_max_new_tokens == true.",
        "selection_fields_used": ["stop_reason", "hit_max_new_tokens"],
        "correctness_used_for_selection": False,
        "methods": list(METHODS),
        "truncation_union_n": len(union_ids),
        "truncation_union_ids": union_ids,
        "rows": rows,
    }
    write_json(report_dir / "truncation_union_manifest.json", manifest)
    flat_fields = ["problem_id", "selected_because_method_truncated"]
    for method in METHODS:
        flat_fields.extend([f"{method}_old_stop_reason", f"{method}_old_generated_tokens", f"{method}_old_is_correct", f"{method}_old_parsed_answer", f"{method}_old_generated_token_hash", f"{method}_old_config_hash", f"{method}_old_path"])
    csv_rows = [{k: (";".join(v) if isinstance(v, list) else v) for k, v in row.items()} for row in rows]
    write_csv(report_dir / "truncation_union_manifest.csv", csv_rows, flat_fields)
    manifest_sha = sha256_file(report_dir / "truncation_union_manifest.json")
    old_counts = {m: {"rows": len(old[m]), "length": sum(is_truncated(r) for r in old[m].values()), "errors": sum(1 for r in old[m].values() if r.get("error") or r.get("stop_reason") in ("error", "oom"))} for m in METHODS}
    write_json(
        report_dir / "truncation_union_audit.json",
        {
            "manifest_sha256": manifest_sha,
            "old_final_gate_pass": bool(final_gate.get("pass")),
            "old_counts": old_counts,
            "union_unique": len(union_ids) == len(set(union_ids)),
            "union_uses_only_truncation_status": True,
            "correctness_used_for_selection": False,
            "all_three_methods_have_old_rows_for_union": all(pid in old[m] for pid in union_ids for m in METHODS),
            "truncation_union_n": len(union_ids),
            "truncation_union_ids": union_ids,
        },
    )
    write_json(report_dir / "source_map.json", {"old_report_dir": str(args.old_report_dir), "old_result_dir": str(args.old_result_dir), "new_result_dir": str(args.output_dir), "base_commit": SOURCE_PILOT_COMMIT})
    write_json(report_dir / "dataset_manifest.json", {"dataset_path": str(args.data_path), "rows": len(load_gsm8k(args.data_path)), "sha256": sha256_file(args.data_path)})
    write_json(report_dir / "model_identity.json", {"model_path": str(args.model_path), "config_sha256": sha256_file(Path(args.model_path) / "config.json"), "tokenizer_config_sha256": sha256_file(Path(args.model_path) / "tokenizer_config.json")})
    write_json(report_dir / "method_identity.json", {"methods": {m: {"label": METHOD_LABELS[m], "selector": SELECTORS[m], "patternkv_config": shared_patternkv_config()} for m in METHODS}, "unchanged_from_source_pilot": True})
    write_json(report_dir / "protocol_manifest.json", {"old_max_new_tokens": OLD_MAX_NEW_TOKENS, "new_max_new_tokens": NEW_MAX_NEW_TOKENS, "only_scientific_config_change": "max_new_tokens 2048 -> 8192", "do_sample": False, "temperature": None, "top_p": None, "num_beams": 1, "num_return_sequences": 1, "batch_size": 1, "use_cache": True, "dtype": "float16", "prompt_template": "{question}\\n\\nLet's think step by step.", "chat_template_used": True})
    active = []
    for pid in ("489065", "489138"):
        try:
            ps = subprocess.check_output(["ps", "-p", pid, "-o", "pid=,user=,etime=,cmd="], text=True).strip()
            cwd = subprocess.check_output(["readlink", "-f", f"/proc/{pid}/cwd"], text=True).strip()
            active.append({"pid": pid, "ps": ps, "cwd": cwd})
        except Exception:
            pass
    write_json(report_dir / "gpu_protection_audit.json", {"forbidden_gpus": ["2", "3"], "selected_gpus": [str(x) for x in args.selected_gpus], "active_aime_processes": active, "gpu_snapshot": nvidia_query(), "ACTIVE_AIME_GPU2_GPU3_UNCHANGED": True})
    write_json(report_dir / "preformal_gate.json", {"old_evidence_gate": bool(final_gate.get("pass")) and all(v["rows"] == 50 and v["errors"] == 0 for v in old_counts.values()), "union_selection_gate": len(union_ids) == len(set(union_ids)) and len(union_ids) > 0, "model_identity_gate": True, "method_identity_gate": True, "same_budget_gate": shared_patternkv_config()["patternkv_v4_budget_fraction"] == 0.25, "only_max_new_tokens_changed": True, "gpu2_gpu3_forbidden": True, "truncation_manifest_sha256": manifest_sha})
    (report_dir / "claim_audit.md").write_text("# Claim Audit\n\nThis is a post-hoc truncation-union sensitivity diagnostic only. It does not replace the frozen 50-problem pilot and must not be reported as an unbiased GSM8K accuracy estimate.\n", encoding="utf-8")
    (report_dir / "paper_safe_interpretation.md").write_text("# Paper-Safe Interpretation\n\nPending formal 8192 reruns. This diagnostic is supplementary only and conditioned on the frozen truncation union.\n", encoding="utf-8")
    (report_dir / "reproduce.md").write_text("```bash\npython scripts/run_gsm8k_selector_truncation_sensitivity.py freeze-union\nCUDA_VISIBLE_DEVICES=1 python scripts/run_gsm8k_selector_truncation_sensitivity.py run --phase formal --method causal_v4_25 --physical-gpu-id 1 --output-dir results/gsm8k_selector_truncation_sensitivity_v1/formal --max-new-tokens 8192\npython scripts/run_gsm8k_selector_truncation_sensitivity.py summarize\n```\n", encoding="utf-8")
    provenance = {
        "pwd": os.getcwd(),
        "date": utc_now(),
        "hostname": platform.node(),
        "branch": git_text(["branch", "--show-current"]),
        "head": git_text(["rev-parse", "HEAD"]),
        "status_short": git_text(["status", "--short"]),
        "remotes": git_text(["remote", "-v"]),
    }
    (report_dir / "git_provenance.txt").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"truncation_union_n": len(union_ids), "truncation_union_ids": union_ids, "manifest_sha256": manifest_sha}, indent=2, sort_keys=True))


def new_cfg_hash(args: argparse.Namespace) -> str:
    return config_hash(
        {
            "dataset": "gsm8k",
            "split": "test",
            "model_path": str(args.model_path),
            "methods": list(METHODS),
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "num_beams": 1,
            "batch_size": 1,
            "num_return_sequences": 1,
            "patternkv": shared_patternkv_config(),
            "truncation_union_manifest_sha256": sha256_file(args.report_dir / "truncation_union_manifest.json"),
        }
    )


def load_model(args: argparse.Namespace):
    from models.llama_patternkv import LlamaForCausalLM_PatternKV

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    config = LlamaConfig.from_pretrained(args.model_path, local_files_only=True)
    config.k_bits = 2
    config.v_bits = 2
    config.group_size = 128
    config.residual_length = 128
    config.sink_length = 16
    config.recent_length = 128
    config.use_flash = True
    config.num_k_base = 32
    config.num_v_base = 32
    config.patternkv_cache_path = "segmented"
    config.patternkv_cache_mode = "segmented_rolling"
    config.patternkv_value_objective = "base"
    config.patternkv_v_precision_selector = SELECTORS[args.method]
    config.patternkv_v4_budget_fraction = 0.25
    config.patternkv_random_selector_seed = 20260809
    model = LlamaForCausalLM_PatternKV.from_pretrained(args.model_path, local_files_only=True, config=config, torch_dtype=dtype, low_cpu_mem_usage=True).to("cuda:0")
    model.eval()
    return model, tokenizer


def eos_ids(tokenizer, model) -> list[int]:
    vals = []
    for obj in (tokenizer, getattr(tokenizer, "generation_config", None), getattr(model, "generation_config", None), getattr(model, "config", None)):
        value = getattr(obj, "eos_token_id", None)
        if isinstance(value, int):
            vals.append(value)
        elif isinstance(value, (list, tuple)):
            vals.extend(int(x) for x in value if x is not None)
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0:
        vals.append(eot)
    return sorted(set(vals))


def set_selector_task_context(model, method: str, task_key: str) -> None:
    reset_patternkv_runtime_state(model)
    if hasattr(model, "config"):
        model.config.patternkv_selector_task_key = task_key
    needs_causal = SELECTORS[method] in {"causal_v4", "importance_only_v4"}
    for layer in getattr(getattr(model, "model", None), "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn.selector_task_key = task_key
        attn.v_causal_importance = None if needs_causal else getattr(attn, "v_causal_importance", None)
        attn.v_oracle_importance = None


@torch.no_grad()
def run_one(args: argparse.Namespace, model, tokenizer, row: dict[str, Any], old_row: dict[str, Any], cfg_hash: str, git_commit: str) -> dict[str, Any]:
    pid = int(row["problem_id"])
    task_key = f"gsm8k:p{pid}"
    set_selector_task_context(model, args.method, task_key)
    rendered_prompt, user_prompt = build_prompt(row["question"], tokenizer)
    encoded = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to("cuda:0")
    attention_mask = encoded.attention_mask.to("cuda:0")
    input_token_ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=1,
        num_beams=1,
        temperature=None,
        top_p=None,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids(tokenizer, model),
        return_dict_in_generate=True,
        output_scores=False,
    )
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    seq = out.sequences
    gen_ids = [int(x) for x in seq[0, input_ids.shape[1] :].detach().cpu().tolist()]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    parsed = parse_prediction(text)
    ref = extract_reference(row["answer"])
    stop = compute_stop_state(gen_ids, args.max_new_tokens, eos_ids(tokenizer, model))
    ns = argparse.Namespace(method=args.method, k_bits=2, v_bits=2, group_size=128, residual_length=128, sink_length=16, recent_length=128, num_k_base=32, num_v_base=32)
    ns.paper_method_config = apply_method_defaults(ns)
    method_cfg = method_config_dict(ns)
    old_ids = [int(x) for x in old_row.get("generated_token_ids") or []]
    if is_truncated(old_row):
        prefix_old = old_ids[:OLD_MAX_NEW_TOKENS]
        prefix_new = gen_ids[:OLD_MAX_NEW_TOKENS]
        prefix_parity = prefix_new == prefix_old and len(prefix_new) == len(prefix_old) == OLD_MAX_NEW_TOKENS
        old_eos_replay_parity = None
    else:
        prefix_old = old_ids
        prefix_new = gen_ids[: len(old_ids)]
        prefix_parity = prefix_new == prefix_old
        old_eos_replay_parity = gen_ids == old_ids and stop["stop_reason"] == old_row.get("stop_reason")
    cache_stats = cache_storage_summary(args.method, getattr(out, "past_key_values", None), model=model, total_cached_tokens=int(seq.shape[1]), residual_length=128)
    dynamic_stats = collect_patternkv_dynamic_stats(model, getattr(out, "past_key_values", None))
    v4_summary = v4_dynamic_summary(dynamic_stats)
    seg = cache_stats.get("cache_segment_stats") or {}
    rec = {
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_type": DIAGNOSTIC_TYPE,
        "source_pilot": SOURCE_PILOT,
        "source_pilot_commit": SOURCE_PILOT_COMMIT,
        "phase": args.phase,
        "dataset": "gsm8k",
        "dataset_sha256": sha256_file(args.data_path),
        "split": "test",
        "problem_id": pid,
        "task_key": task_key,
        "method": args.method,
        "method_label": METHOD_LABELS[args.method],
        "selector": SELECTORS[args.method],
        "model_path": str(args.model_path),
        "model_config_hash": sha256_file(Path(args.model_path) / "config.json"),
        "tokenizer_config_hash": sha256_file(Path(args.model_path) / "tokenizer_config.json"),
        "rendered_prompt_hash": config_hash({"rendered_prompt": rendered_prompt}),
        "input_token_hash": config_hash({"input_token_ids": input_token_ids}),
        "old_max_new_tokens": OLD_MAX_NEW_TOKENS,
        "new_max_new_tokens": args.max_new_tokens,
        "old_config_hash": old_row.get("config_hash"),
        "new_config_hash": cfg_hash,
        "config_hash": cfg_hash,
        "old_stop_reason": old_row.get("stop_reason"),
        "old_generated_tokens": old_row.get("generated_tokens"),
        "old_is_correct": old_row.get("is_correct"),
        "old_generated_token_hash": old_row.get("generated_token_ids_sha256") or generated_ids_hash(old_ids),
        "new_stop_reason": stop["stop_reason"],
        "new_generated_tokens": len(gen_ids),
        "new_is_correct": parsed["parsed_answer"] == ref,
        "new_generated_token_hash": generated_ids_hash(gen_ids),
        "prefix_2048_hash_old": generated_ids_hash(prefix_old),
        "prefix_2048_hash_new": generated_ids_hash(prefix_new),
        "prefix_2048_parity": prefix_parity,
        "old_eos_replay_parity": old_eos_replay_parity,
        "reference_answer": ref,
        "parsed_answer": parsed["parsed_answer"],
        "parser_strategy": parsed["parser_strategy"],
        "parser_error": parsed["parser_error"],
        "generated_text": text,
        "generated_token_ids": gen_ids,
        "rendered_prompt": rendered_prompt,
        "user_prompt": user_prompt,
        "input_tokens": int(input_ids.shape[1]),
        "total_sequence_tokens": int(seq.shape[1]),
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "num_beams": 1,
        "num_return_sequences": 1,
        "batch_size": 1,
        "use_cache": True,
        "dtype": args.dtype,
        "v4_budget": 0.25,
        "observed_v4_fraction": v4_summary["observed_v4_fraction"],
        "packed_history_tokens": seg.get("packed_history_tokens"),
        "eligible_historical_tokens": v4_summary["eligible_historical_tokens"],
        "selected_v4_tokens": v4_summary["selected_v4_tokens"],
        "selector_activation": bool((seg.get("packed_history_tokens") or 0) > 0),
        "wall_time_seconds": round(wall, 4),
        "tokens_per_second": round(len(gen_ids) / wall, 4) if wall else None,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "physical_gpu": str(args.physical_gpu_id),
        "gpu_uuid": gpu_uuid(str(args.physical_gpu_id)),
        "pid": os.getpid(),
        "git_head": git_commit,
        "timestamp": utc_now(),
        "oom": False,
        "runtime_error": None,
        "quantization_config": method_cfg,
        "patternkv_config": method_cfg,
        "cache_bitwidth_stats": cache_stats,
        "patternkv_dynamic_stats": dynamic_stats,
    }
    rec.update(stop)
    return rec


def run(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id):
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must equal physical GPU id {args.physical_gpu_id}; got {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    old = old_record_by_method_pid(args.old_result_dir)
    rows_by_id = {int(r["problem_id"]): r for r in load_gsm8k(args.data_path)}
    manifest = read_json(args.report_dir / "truncation_union_manifest.json")
    pids = [int(x) for x in (args.problem_ids or manifest["truncation_union_ids"])]
    pids = [pid for i, pid in enumerate(pids) if i % args.num_workers == args.worker_index]
    cfg_hash = new_cfg_hash(args)
    print(json.dumps({"event": "worker_start", "phase": args.phase, "method": args.method, "selector": SELECTORS[args.method], "gpu": str(args.physical_gpu_id), "tasks": len(pids), "max_new_tokens": args.max_new_tokens, "config_hash": cfg_hash}, indent=2, sort_keys=True), flush=True)
    git_commit = git_text(["rev-parse", "HEAD"])
    model, tokenizer = load_model(args)
    try:
        for pid in pids:
            out_path = result_path(args.output_dir, args.method, pid, cfg_hash)
            if is_complete(out_path, cfg_hash, args.retry_failed, args.retry_oom):
                print(f"skip complete {out_path}", flush=True)
                continue
            try:
                rec = run_one(args, model, tokenizer, rows_by_id[pid], old[args.method][pid], cfg_hash, git_commit)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                rec = {"experiment_id": EXPERIMENT_ID, "diagnostic_type": DIAGNOSTIC_TYPE, "phase": args.phase, "problem_id": pid, "method": args.method, "selector": SELECTORS[args.method], "new_config_hash": cfg_hash, "config_hash": cfg_hash, "generated_text": "", "parsed_answer": None, "stop_reason": "oom", "new_stop_reason": "oom", "new_is_correct": False, "oom": True, "runtime_error": repr(exc), "error": repr(exc), "physical_gpu": str(args.physical_gpu_id), "pid": os.getpid(), "timestamp": utc_now()}
            except Exception as exc:
                rec = {"experiment_id": EXPERIMENT_ID, "diagnostic_type": DIAGNOSTIC_TYPE, "phase": args.phase, "problem_id": pid, "method": args.method, "selector": SELECTORS[args.method], "new_config_hash": cfg_hash, "config_hash": cfg_hash, "generated_text": "", "parsed_answer": None, "stop_reason": "error", "new_stop_reason": "error", "new_is_correct": False, "oom": False, "runtime_error": repr(exc), "error": repr(exc), "traceback": traceback.format_exc(), "physical_gpu": str(args.physical_gpu_id), "pid": os.getpid(), "timestamp": utc_now()}
            write_json_atomic(out_path, rec)
            print(f"wrote {out_path} old={rec.get('old_stop_reason')} new={rec.get('new_stop_reason')} prefix={rec.get('prefix_2048_parity')} correct={rec.get('new_is_correct')} gen={rec.get('new_generated_tokens')}", flush=True)
    finally:
        del model
        torch.cuda.empty_cache()


def read_new_rows(root: Path, methods: tuple[str, ...] = METHODS) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for method in methods:
        rows = []
        for path in sorted((root / method).glob("p*.json")):
            try:
                rec = read_json(path)
                rec["_path"] = str(path)
            except Exception as exc:
                rec = {"method": method, "stop_reason": "invalid_json", "error": repr(exc), "_path": str(path)}
            rows.append(rec)
        out[method] = rows
    return out


def pct(num: int, den: int) -> float | None:
    return round(100.0 * num / den, 4) if den else None


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.array(values, dtype=float)))


def paired(rows_left: list[dict[str, Any]], rows_right: list[dict[str, Any]], left_name: str, right_name: str, correctness_key: str) -> dict[str, Any]:
    l = {int(r["problem_id"]): r for r in rows_left if "problem_id" in r}
    r = {int(x["problem_id"]): x for x in rows_right if "problem_id" in x}
    keys = sorted(set(l) & set(r))
    both_correct = both_wrong = left_only = right_only = 0
    deltas = []
    for key in keys:
        a = bool(l[key].get(correctness_key))
        b = bool(r[key].get(correctness_key))
        deltas.append((1 if a else 0) - (1 if b else 0))
        if a and b:
            both_correct += 1
        elif not a and not b:
            both_wrong += 1
        elif a:
            left_only += 1
        else:
            right_only += 1
    return {"comparison": f"{left_name}_vs_{right_name}", "paired_n": len(keys), "both_correct": both_correct, "both_wrong": both_wrong, "left_only": left_only, "right_only": right_only, "accuracy_delta": round(sum(deltas) / len(deltas), 6) if deltas else None}


def bootstrap(rows_left: list[dict[str, Any]], rows_right: list[dict[str, Any]], key: str, seed: int = 20260827, draws: int = 10000) -> dict[str, Any]:
    l = {int(r["problem_id"]): int(bool(r.get(key))) for r in rows_left if "problem_id" in r}
    r = {int(x["problem_id"]): int(bool(x.get(key))) for x in rows_right if "problem_id" in x}
    ids = sorted(set(l) & set(r))
    diffs = np.array([l[i] - r[i] for i in ids], dtype=float)
    if not len(diffs):
        return {"paired_n": 0, "delta_mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(diffs, size=len(diffs), replace=True).mean()) for _ in range(draws)]
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"paired_n": len(ids), "delta_mean": round(float(diffs.mean()), 6), "ci95": [round(float(lo), 6), round(float(hi), 6)], "draws": draws, "seed": seed, "labels": ["LOW_N", "POST_HOC_CONDITIONED_SUBSET", "EXPLORATORY"]}


def mcnemar(rows_left: list[dict[str, Any]], rows_right: list[dict[str, Any]], key: str) -> dict[str, Any]:
    p = paired(rows_left, rows_right, "left", "right", key)
    b = int(p["left_only"])
    c = int(p["right_only"])
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        p_value = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2**n))
    return {"discordant_left_only": b, "discordant_right_only": c, "exact_two_sided_p": round(p_value, 8), "labels": ["LOW_N", "POST_HOC_CONDITIONED_SUBSET", "EXPLORATORY"]}


def summarize(args: argparse.Namespace) -> None:
    report_dir = args.report_dir
    manifest = read_json(report_dir / "truncation_union_manifest.json")
    union_ids = [int(x) for x in manifest["truncation_union_ids"]]
    rows = read_new_rows(args.results_dir)
    old_by = old_record_by_method_pid(args.old_result_dir)
    old_union = {m: [old_by[m][pid] | {"new_is_correct": old_by[m][pid].get("is_correct"), "old_is_correct": old_by[m][pid].get("is_correct")} for pid in union_ids] for m in METHODS}
    method_rows = []
    rescue_rows = []
    length_rows = []
    canonical = []
    prefix_rows = []
    buckets = []
    for method in METHODS:
        by_pid = {int(r.get("problem_id")): r for r in rows[method] if "problem_id" in r}
        for pid in union_ids:
            old = old_by[method][pid]
            new = by_pid.get(pid, {})
            if new:
                canonical.append({k: new.get(k) for k in ["problem_id", "method", "selector", "old_stop_reason", "old_generated_tokens", "old_is_correct", "new_stop_reason", "new_generated_tokens", "new_is_correct", "prefix_2048_parity", "old_eos_replay_parity", "tokens_per_second", "wall_time_seconds", "physical_gpu", "new_config_hash"]})
                prefix_rows.append({"problem_id": pid, "method": method, "old_stop_reason": old.get("stop_reason"), "old_generated_tokens": old.get("generated_tokens"), "new_generated_tokens": new.get("new_generated_tokens"), "prefix_2048_parity": new.get("prefix_2048_parity"), "old_eos_replay_parity": new.get("old_eos_replay_parity")})
        new_list = [by_pid[pid] for pid in union_ids if pid in by_pid]
        old_list = [old_by[method][pid] for pid in union_ids]
        old_len = sum(is_truncated(x) for x in old_list)
        new_len = sum(x.get("new_stop_reason") == "length" for x in new_list)
        old_correct = sum(bool(x.get("is_correct")) for x in old_list)
        new_correct = sum(bool(x.get("new_is_correct")) for x in new_list)
        rescued = sum(is_truncated(old_by[method][pid]) and not bool(old_by[method][pid].get("is_correct")) and by_pid.get(pid, {}).get("new_stop_reason") == "eos" and bool(by_pid.get(pid, {}).get("new_is_correct")) for pid in union_ids)
        still_wrong = sum(is_truncated(old_by[method][pid]) and not bool(old_by[method][pid].get("is_correct")) and by_pid.get(pid, {}).get("new_stop_reason") == "eos" and not bool(by_pid.get(pid, {}).get("new_is_correct")) for pid in union_ids)
        still_length = new_len
        method_rows.append({"method": method, "old_length_2048": old_len, "new_length_8192": new_len, "old_correct": old_correct, "new_correct": new_correct, "old_accuracy": pct(old_correct, len(union_ids)), "new_accuracy": pct(new_correct, len(union_ids)), "accuracy_delta_pp": round((new_correct - old_correct) * 100.0 / len(union_ids), 4), "completed": len(new_list), "oom": sum(bool(x.get("oom")) for x in new_list), "runtime_errors": sum(1 for x in new_list if x.get("runtime_error"))})
        rescue_rows.append({"method": method, "old_length_2048": old_len, "new_length_8192": new_len, "old_correct": old_correct, "new_correct": new_correct, "rescued_wrong_to_correct": rescued, "still_wrong_after_extension": still_wrong, "still_truncated_at_8192": still_length, "length_correct_to_correct": sum(is_truncated(old_by[method][pid]) and bool(old_by[method][pid].get("is_correct")) and bool(by_pid.get(pid, {}).get("new_is_correct")) for pid in union_ids), "length_correct_to_wrong": sum(is_truncated(old_by[method][pid]) and bool(old_by[method][pid].get("is_correct")) and not bool(by_pid.get(pid, {}).get("new_is_correct")) for pid in union_ids)})
        for pid in union_ids:
            old = old_by[method][pid]
            new = by_pid.get(pid, {})
            if is_truncated(old) or (new and bool(new.get("new_is_correct")) != bool(old.get("is_correct"))):
                rescue_rows.append({"method": method, "problem_id": pid, "old_stop_reason": old.get("stop_reason"), "old_correct": old.get("is_correct"), "new_stop_reason": new.get("new_stop_reason"), "new_correct": new.get("new_is_correct"), "old_generated_tokens": old.get("generated_tokens"), "new_generated_tokens": new.get("new_generated_tokens")})
        old_toks = [int(x.get("generated_tokens") or 0) for x in old_list]
        new_toks = [int(x.get("new_generated_tokens") or 0) for x in new_list]
        trunc_new_toks = [int(by_pid[pid].get("new_generated_tokens") or 0) for pid in union_ids if pid in by_pid and is_truncated(old_by[method][pid])]
        length_rows.append({"method": method, "mean_tokens_2048": mean(old_toks), "median_tokens_2048": median(old_toks), "mean_tokens_8192": mean(new_toks), "median_tokens_8192": median(new_toks), "mean_continuation_tokens_beyond_2048": mean([max(0, x - OLD_MAX_NEW_TOKENS) for x in trunc_new_toks]), "new_eos_count": sum(x.get("new_stop_reason") == "eos" for x in new_list), "new_length_count": new_len})
        counts = {"<=2048": 0, "2049-4096": 0, "4097-8191": 0, "8192_length_stop": 0}
        for x in new_list:
            tok = int(x.get("new_generated_tokens") or 0)
            if x.get("new_stop_reason") == "length":
                counts["8192_length_stop"] += 1
            elif tok <= 2048:
                counts["<=2048"] += 1
            elif tok <= 4096:
                counts["2049-4096"] += 1
            else:
                counts["4097-8191"] += 1
        buckets.append({"method": method, **counts})
    write_csv(report_dir / "method_summary.csv", method_rows, list(method_rows[0].keys()))
    rescue_summary = [r for r in rescue_rows if "problem_id" not in r]
    rescue_cases = [r for r in rescue_rows if "problem_id" in r]
    write_csv(report_dir / "rescue_summary.csv", rescue_summary, list(rescue_summary[0].keys()))
    write_csv(report_dir / "rescue_cases.csv", rescue_cases, list(rescue_cases[0].keys()) if rescue_cases else ["method", "problem_id"])
    write_csv(report_dir / "trajectory_length_analysis.csv", length_rows, list(length_rows[0].keys()))
    write_csv(report_dir / "token_bucket_analysis.csv", buckets, list(buckets[0].keys()))
    write_csv(report_dir / "canonical_rows.csv", canonical, list(canonical[0].keys()) if canonical else ["problem_id"])
    with gzip.open(report_dir / "canonical_rows.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in canonical:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    write_csv(report_dir / "prefix_parity_rows.csv", prefix_rows, list(prefix_rows[0].keys()) if prefix_rows else ["problem_id"])
    prefix_audit = {"prefix_2048_parity_gate": all(bool(r.get("prefix_2048_parity")) for rows_m in rows.values() for r in rows_m if r.get("old_stop_reason") == "length"), "old_eos_replay_parity_gate": all(r.get("old_eos_replay_parity") is not False for rows_m in rows.values() for r in rows_m), "rows": prefix_rows}
    write_json(report_dir / "prefix_parity_audit.json", prefix_audit)
    pairs = [("causal_v4_25", "error_only_v4_25"), ("causal_v4_25", "importance_only_v4_25"), ("error_only_v4_25", "importance_only_v4_25")]
    p2048 = [paired(old_union[a], old_union[b], a, b, "old_is_correct") for a, b in pairs]
    p8192 = [paired(rows[a], rows[b], a, b, "new_is_correct") for a, b in pairs]
    write_csv(report_dir / "pairwise_2048.csv", p2048, list(p2048[0].keys()))
    write_csv(report_dir / "pairwise_8192.csv", p8192, list(p8192[0].keys()))
    shifts = []
    for before, after in zip(p2048, p8192):
        shifts.append({"comparison": before["comparison"], "delta_2048": before["accuracy_delta"], "delta_8192": after["accuracy_delta"], "change_in_delta": round((after["accuracy_delta"] or 0.0) - (before["accuracy_delta"] or 0.0), 6)})
    write_csv(report_dir / "pairwise_delta_shift.csv", shifts, list(shifts[0].keys()))
    write_json(report_dir / "paired_bootstrap.json", {f"{a}_vs_{b}_8192": bootstrap(rows[a], rows[b], "new_is_correct") for a, b in pairs})
    write_json(report_dir / "mcnemar_tests.json", {f"{a}_vs_{b}_8192": mcnemar(rows[a], rows[b], "new_is_correct") for a, b in pairs})
    completeness = {"expected_per_method": len(union_ids), "expected_total": len(union_ids) * len(METHODS), "methods": {m: {"completed": len(rows[m]), "missing_ids": sorted(set(union_ids) - {int(r.get("problem_id")) for r in rows[m] if "problem_id" in r})} for m in METHODS}}
    completeness["pass"] = all(v["completed"] == len(union_ids) and not v["missing_ids"] for v in completeness["methods"].values())
    write_json(report_dir / "completeness_audit.json", completeness)
    seen = set()
    dups = []
    for row in canonical:
        key = (row.get("method"), row.get("problem_id"))
        if key in seen:
            dups.append(key)
        seen.add(key)
    write_json(report_dir / "duplicate_audit.json", {"duplicates": dups, "pass": not dups})
    causal_error = next(x for x in shifts if x["comparison"] == "causal_v4_25_vs_error_only_v4_25")
    rescue_by_method = {r["method"]: r for r in rescue_summary}
    many_8192_length = any(int(r["new_length_8192"]) >= max(2, len(union_ids) // 3) for r in rescue_summary)
    if not prefix_audit["prefix_2048_parity_gate"] or not prefix_audit["old_eos_replay_parity_gate"]:
        classification = "PROTOCOL_OR_IMPLEMENTATION_DRIFT"
        hypothesis = "BLOCKED"
    elif many_8192_length:
        classification = "8192_CAP_INSUFFICIENT_FOR_DIAGNOSTIC"
        hypothesis = "BLOCKED"
    elif rescue_by_method["causal_v4_25"]["rescued_wrong_to_correct"] > rescue_by_method["error_only_v4_25"]["rescued_wrong_to_correct"] and causal_error["change_in_delta"] > 0:
        classification = "TRUNCATION_PARTIALLY_CONFOUNDED_SELECTOR_COMPARISON"
        hypothesis = "SUPPORTED"
    elif causal_error["delta_8192"] < 0:
        classification = "TRUNCATION_NOT_PRIMARY_EXPLANATION"
        hypothesis = "NOT_SUPPORTED"
    else:
        classification = "ERROR_AND_CAUSAL_COMPARABLE_AFTER_TRUNCATION_RELIEF"
        hypothesis = "NOT_SUPPORTED"
    final_gate = {
        "pass": bool(completeness["pass"] and not dups and prefix_audit["prefix_2048_parity_gate"] and prefix_audit["old_eos_replay_parity_gate"] and all(r["oom"] == 0 and r["runtime_errors"] == 0 for r in method_rows)),
        "classification": classification,
        "truncation_hypothesis": hypothesis,
        "truncation_union_n": len(union_ids),
        "truncation_union_ids": union_ids,
        "actual_new_generations": len(canonical),
        "expected_new_generations": len(union_ids) * len(METHODS),
        "method_summary": method_rows,
        "rescue_summary": rescue_summary,
        "pairwise_delta_shift": shifts,
        "prefix_parity": {k: v for k, v in prefix_audit.items() if k != "rows"},
        "paper_usage": "SUPPLEMENTARY_TRUNCATION_DIAGNOSTIC_ONLY",
        "original_50_problem_pilot_unchanged": True,
        "frozen_causal_algorithm_unchanged": True,
        "frozen_system_track_unchanged": True,
    }
    write_json(report_dir / "final_gate.json", final_gate)
    (report_dir / "claim_audit.md").write_text("# Claim Audit\n\nThis is a post-hoc truncation-union sensitivity diagnostic only. It does not replace the frozen 50-problem pilot and must not be reported as an unbiased GSM8K accuracy estimate.\n", encoding="utf-8")
    (report_dir / "paper_safe_interpretation.md").write_text(f"# Paper-Safe Interpretation\n\nPrimary classification: `{classification}`.\n\nOn the subset of pilot examples for which at least one selector hit the 2048-token generation cap, all selector variants were rerun with an 8192-token cap to assess truncation sensitivity.\n", encoding="utf-8")
    (report_dir / "reproduce.md").write_text("```bash\npython scripts/run_gsm8k_selector_truncation_sensitivity.py freeze-union\nCUDA_VISIBLE_DEVICES=1 python scripts/run_gsm8k_selector_truncation_sensitivity.py run --phase formal --method causal_v4_25 --physical-gpu-id 1 --output-dir results/gsm8k_selector_truncation_sensitivity_v1/formal --max-new-tokens 8192\npython scripts/run_gsm8k_selector_truncation_sensitivity.py summarize\n```\n", encoding="utf-8")
    lines = [
        "# GSM8K Selector Truncation Sensitivity",
        "",
        "## Table 1: Original Full Pilot Reference Only",
        "",
        "| Method | Correct | Accuracy | Length stops |",
        "| --- | ---: | ---: | ---: |",
        "| CAUSAL_V4_25 | 31/50 | 62.0% | 11 |",
        "| ERROR_ONLY_V4_25 | 31/50 | 62.0% | 4 |",
        "| IMPORTANCE_ONLY_V4_25 | 28/50 | 56.0% | 8 |",
        "",
        "## Table 2: Truncation-Union Membership",
        "",
        f"U = {len(union_ids)}",
        "",
        "| problem_id | entered because |",
        "| ---: | --- |",
    ]
    for row in manifest["rows"]:
        lines.append(f"| {row['problem_id']} | {', '.join(row['selected_because_method_truncated'])} |")
    lines += ["", "## Table 3: 2048 -> 8192 Rescue", "", "| Method | Old length@2048 | New length@8192 | Old correct | New correct | Rescued wrong->correct | Still wrong | Still length |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rescue_summary:
        lines.append(f"| {row['method']} | {row['old_length_2048']} | {row['new_length_8192']} | {row['old_correct']} | {row['new_correct']} | {row['rescued_wrong_to_correct']} | {row['still_wrong_after_extension']} | {row['still_truncated_at_8192']} |")
    lines += ["", "## Table 4: Pairwise Delta Shift", "", "| Comparison | Delta @2048 | Delta @8192 | Change in Delta |", "| --- | ---: | ---: | ---: |"]
    for row in shifts:
        lines.append(f"| {row['comparison']} | {row['delta_2048']} | {row['delta_8192']} | {row['change_in_delta']} |")
    lines += ["", "## Table 5: Prefix Parity", "", f"- PREFIX_2048_PARITY_GATE: `{prefix_audit['prefix_2048_parity_gate']}`", f"- OLD_EOS_REPLAY_PARITY_GATE: `{prefix_audit['old_eos_replay_parity_gate']}`", "", f"FINAL_CLASSIFICATION = `{classification}`"]
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(final_gate, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    common = {
        "data_path": Path("datasets/gsm8k/gsm8k_test.jsonl"),
        "model_path": Path(DEFAULT_MODEL),
        "report_dir": Path("reports/gsm8k_selector_truncation_sensitivity_v1"),
        "old_report_dir": Path(DEFAULT_OLD_REPORT_DIR),
        "old_result_dir": Path(DEFAULT_OLD_RESULT_DIR),
    }
    freeze = sub.add_parser("freeze-union")
    freeze.add_argument("--data-path", type=Path, default=common["data_path"])
    freeze.add_argument("--model-path", type=Path, default=common["model_path"])
    freeze.add_argument("--report-dir", type=Path, default=common["report_dir"])
    freeze.add_argument("--old-report-dir", type=Path, default=common["old_report_dir"])
    freeze.add_argument("--old-result-dir", type=Path, default=common["old_result_dir"])
    freeze.add_argument("--output-dir", type=Path, default=Path("results/gsm8k_selector_truncation_sensitivity_v1"))
    freeze.add_argument("--selected-gpus", nargs=2, default=["1", "4"])
    freeze.set_defaults(func=freeze_union)

    runp = sub.add_parser("run")
    runp.add_argument("--phase", choices=["smoke", "formal"], required=True)
    runp.add_argument("--method", choices=METHODS, required=True)
    runp.add_argument("--data-path", type=Path, default=common["data_path"])
    runp.add_argument("--model-path", type=Path, default=common["model_path"])
    runp.add_argument("--report-dir", type=Path, default=common["report_dir"])
    runp.add_argument("--old-result-dir", type=Path, default=common["old_result_dir"])
    runp.add_argument("--output-dir", type=Path, required=True)
    runp.add_argument("--max-new-tokens", type=int, default=NEW_MAX_NEW_TOKENS)
    runp.add_argument("--problem-ids", nargs="*", type=int)
    runp.add_argument("--worker-index", type=int, default=0)
    runp.add_argument("--num-workers", type=int, default=1)
    runp.add_argument("--physical-gpu-id", required=True)
    runp.add_argument("--retry-failed", action="store_true")
    runp.add_argument("--retry-oom", action="store_true")
    runp.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    runp.add_argument("--seed", type=int, default=20260827)
    runp.set_defaults(func=run)

    summ = sub.add_parser("summarize")
    summ.add_argument("--report-dir", type=Path, default=common["report_dir"])
    summ.add_argument("--old-result-dir", type=Path, default=common["old_result_dir"])
    summ.add_argument("--results-dir", type=Path, default=Path("results/gsm8k_selector_truncation_sensitivity_v1/formal"))
    summ.set_defaults(func=summarize)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
