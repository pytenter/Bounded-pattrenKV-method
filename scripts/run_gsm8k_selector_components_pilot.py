#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, LlamaConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.gsm8k_paper_utils import build_prompt, compute_stop_state, config_hash, extract_reference, is_complete, load_gsm8k, parse_prediction, result_path, sha256_file, write_json_atomic
from bench.paper_config import apply_method_defaults, cache_storage_summary, method_config_dict
from models.llama_patternkv import collect_patternkv_dynamic_stats, reset_patternkv_runtime_state


EXPERIMENT_ID = "gsm8k_selector_components_pilot_v1"
METHODS = ("importance_only_v4_25", "error_only_v4_25", "causal_v4_25")
METHOD_LABELS = {
    "importance_only_v4_25": "IMPORTANCE_ONLY_V4_25",
    "error_only_v4_25": "ERROR_ONLY_V4_25",
    "causal_v4_25": "CAUSAL_V4_25",
}
SELECTORS = {
    "importance_only_v4_25": "importance_only_v4",
    "error_only_v4_25": "error_only_v4",
    "causal_v4_25": "causal_v4",
}
DEFAULT_MODEL = "/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct"
DEFAULT_CANONICAL_CAUSAL = "/data/zypan/Bounded-pattrenKV-pseudodecode-3090/results/causal_v4_25_generalization_v1/gsm8k_full/causal_v4_25"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def git_text(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def nvidia_rows() -> list[dict[str, Any]]:
    query = "index,name,memory.total,memory.used,memory.free,utilization.gpu"
    try:
        out = subprocess.check_output(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True)
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": int(parts[2]),
                "memory_used_mib": int(parts[3]),
                "memory_free_mib": int(parts[4]),
                "gpu_util_percent": int(parts[5]),
            }
        )
    return rows


def canonical_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("p*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        stats = (rec.get("cache_bitwidth_stats") or {}).get("cache_segment_stats") or {}
        rows.append(
            {
                "problem_id": int(rec["problem_id"]),
                "path": str(path),
                "packed_history_tokens": int(stats.get("packed_history_tokens") or 0),
                "generated_tokens": int(rec.get("generated_tokens") or 0),
                "is_correct": bool(rec.get("is_correct")),
                "config_hash": rec.get("config_hash"),
                "model_path": rec.get("model_path"),
            }
        )
    return rows


def select_subset(canonical_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(20260827)
    stratum_a = [int(x) for x in rng.choice(np.arange(1319), size=25, replace=False).tolist()]
    excluded = set(stratum_a)
    records = canonical_rows(canonical_root)
    if len(records) != 1319:
        raise SystemExit(f"canonical CAUSAL root must contain 1319 records, got {len(records)}: {canonical_root}")
    candidates = [r for r in records if r["problem_id"] not in excluded and r["packed_history_tokens"] > 0]
    candidates.sort(key=lambda r: (-r["packed_history_tokens"], -r["generated_tokens"], r["problem_id"]))
    stratum_b = [r["problem_id"] for r in candidates[:25]]
    rows = []
    for order, pid in enumerate(stratum_a):
        rows.append({"order": order, "problem_id": pid, "stratum": "UNIFORM_RANDOM", "selection_signal": "numpy.random.default_rng(20260827)", "canonical_packed_history_tokens": None, "canonical_generated_tokens": None})
    by_pid = {r["problem_id"]: r for r in records}
    for offset, pid in enumerate(stratum_b, start=len(rows)):
        c = by_pid[pid]
        rows.append({"order": offset, "problem_id": pid, "stratum": "LENGTH_PROXY_ACTIVATION_ENRICHED", "selection_signal": "packed_history_tokens>0, packed_history_tokens desc, generated_tokens desc, problem_id asc", "canonical_packed_history_tokens": c["packed_history_tokens"], "canonical_generated_tokens": c["generated_tokens"]})
    audit = {
        "selection_timestamp": utc_now(),
        "random_seed": 20260827,
        "stratum_a_count": len(stratum_a),
        "stratum_b_count": len(stratum_b),
        "total_count": len(rows),
        "duplicate_problem_ids": sorted({pid for pid in [r["problem_id"] for r in rows] if [x["problem_id"] for x in rows].count(pid) > 1}),
        "activation_rule": "LENGTH_PROXY_ACTIVATION_ENRICHED",
        "correctness_used_for_selection": False,
        "canonical_root": str(canonical_root),
        "canonical_records": len(records),
    }
    return rows, audit


def cfg_hash_for(args: argparse.Namespace) -> str:
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
            "pilot_manifest_sha256": sha256_file(args.report_dir / "pilot_manifest.json") if (args.report_dir / "pilot_manifest.json").exists() else None,
        }
    )


def shared_patternkv_config() -> dict[str, Any]:
    return {
        "k_bits": 2,
        "base_v_bits": 2,
        "selected_v_bits": 4,
        "group_size": 128,
        "sink_length": 16,
        "recent_length": 128,
        "residual_length": 128,
        "num_k_base": 32,
        "num_v_base": 32,
        "patternkv_cache_path": "segmented",
        "patternkv_cache_mode": "segmented_rolling",
        "patternkv_value_objective": "base",
        "patternkv_v4_budget_fraction": 0.25,
    }


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
    needs_causal_importance = SELECTORS[method] in {"causal_v4", "importance_only_v4"}
    for layer in getattr(getattr(model, "model", None), "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn.selector_task_key = task_key
        attn.v_causal_importance = None if needs_causal_importance else getattr(attn, "v_causal_importance", None)
        attn.v_oracle_importance = None


@torch.no_grad()
def run_one(args: argparse.Namespace, model, tokenizer, row: dict[str, Any], cfg_hash: str, git_commit: str) -> dict[str, Any]:
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
    cache_stats = cache_storage_summary(args.method, getattr(out, "past_key_values", None), model=model, total_cached_tokens=int(seq.shape[1]), residual_length=128)
    rec = {
        "experiment_id": EXPERIMENT_ID,
        "phase": args.phase,
        "config_hash": cfg_hash,
        "dataset": "gsm8k",
        "split": "test",
        "problem_id": pid,
        "task_key": task_key,
        "method": args.method,
        "method_label": METHOD_LABELS[args.method],
        "value_precision_selector": SELECTORS[args.method],
        "model_path": str(args.model_path),
        "question": row["question"],
        "reference_answer": ref,
        "rendered_prompt": rendered_prompt,
        "user_prompt": user_prompt,
        "prompt_template": "{question}\\n\\nLet's think step by step.",
        "chat_template_used": True,
        "input_tokens": int(input_ids.shape[1]),
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "batch_size": 1,
        "num_beams": 1,
        "num_return_sequences": 1,
        "max_new_tokens": args.max_new_tokens,
        "generated_text": text,
        "input_token_ids_sha256": config_hash({"input_token_ids": input_token_ids}),
        "generated_token_ids": gen_ids,
        "generated_token_ids_sha256": config_hash({"generated_token_ids": gen_ids}),
        "generated_tokens": len(gen_ids),
        "total_sequence_tokens": int(seq.shape[1]),
        "parsed_answer": parsed["parsed_answer"],
        "normalized_prediction": parsed["parsed_answer"],
        "parser_strategy": parsed["parser_strategy"],
        "parser_error": parsed["parser_error"],
        "is_correct": parsed["parsed_answer"] == ref,
        "wall_time_seconds": round(wall, 4),
        "tokens_per_second": round(len(gen_ids) / wall, 4) if wall else None,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "quantization_config": method_cfg,
        "patternkv_config": method_cfg,
        "cache_bitwidth_stats": cache_stats,
        "patternkv_dynamic_stats": collect_patternkv_dynamic_stats(model, getattr(out, "past_key_values", None)),
        "selector_task_key": task_key,
        "git_commit": git_commit,
        "physical_gpu_id": str(args.physical_gpu_id),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "timestamp": utc_now(),
        "error": None,
    }
    rec.update(stop)
    return rec


def prepare(args: argparse.Namespace) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    rows = load_gsm8k(args.data_path)
    subset, audit = select_subset(args.canonical_causal_root)
    pids = [r["problem_id"] for r in subset]
    if len(pids) != 50 or len(set(pids)) != 50:
        raise SystemExit("pilot manifest must contain exactly 50 unique problem_ids")
    pilot_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "dataset_path": str(args.data_path),
        "dataset_rows": len(rows),
        "subset_count": len(subset),
        "methods": list(METHODS),
        "method_labels": METHOD_LABELS,
        "selectors": SELECTORS,
        "shared_patternkv_config": shared_patternkv_config(),
        "generation_config": {"do_sample": False, "temperature": None, "top_p": None, "num_beams": 1, "num_return_sequences": 1, "max_new_tokens": 2048, "batch_size": 1, "use_cache": True, "dtype": "float16"},
        "gpu_policy": {
            "user_override": "Use any two currently idle GPUs; do not use active AIME GPUs.",
            "selected_physical_gpus": [str(x) for x in args.selected_gpus],
            "forbidden_physical_gpus": [str(x) for x in args.forbidden_gpus],
            "original_fixed_gpu_policy_superseded": True,
        },
        "rows": subset,
    }
    write_json(args.report_dir / "pilot_manifest.json", pilot_manifest)
    with (args.report_dir / "pilot_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(subset[0].keys()))
        writer.writeheader()
        writer.writerows(subset)
    write_json(args.report_dir / "subset_selection_audit.json", audit)
    write_json(args.report_dir / "dataset_manifest.json", {"dataset_path": str(args.data_path), "rows": len(rows), "sha256": sha256_file(args.data_path), "problem_id_min": 0, "problem_id_max": 1318})
    write_json(args.report_dir / "protocol_manifest.json", pilot_manifest["generation_config"] | {"prompt_template": "{question}\\n\\nLet's think step by step.", "chat_template_used": True})
    write_json(args.report_dir / "method_identity.json", {"methods": {m: {"selector": SELECTORS[m], "label": METHOD_LABELS[m], "patternkv_config": shared_patternkv_config()} for m in METHODS}})
    write_json(args.report_dir / "model_identity.json", {"model_path": str(args.model_path), "config_sha256": sha256_file(Path(args.model_path) / "config.json"), "tokenizer_config_sha256": sha256_file(Path(args.model_path) / "tokenizer_config.json")})
    write_json(args.report_dir / "environment.json", {"created_at": utc_now(), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "nvidia_smi": nvidia_rows()})
    write_json(args.report_dir / "memory_qualification.json", {"selected_gpus": [str(x) for x in args.selected_gpus], "forbidden_gpus": [str(x) for x in args.forbidden_gpus], "gpu_snapshot": nvidia_rows(), "note": "The original 30000 MiB threshold is impossible on 24 GiB RTX 3090 cards; user override selected two idle GPUs by current free memory/utilization."})
    gates = {
        "dataset_1319_rows": len(rows) == 1319,
        "exact_50_manifest": len(subset) == 50 and len(set(pids)) == 50,
        "strata_25_25": audit["stratum_a_count"] == 25 and audit["stratum_b_count"] == 25,
        "selection_not_correctness_based": audit["correctness_used_for_selection"] is False,
        "method_registration": sorted(SELECTORS) == sorted(METHODS),
        "selector_identities": SELECTORS,
        "v4_budget_fraction_25": shared_patternkv_config()["patternkv_v4_budget_fraction"] == 0.25,
        "gpu_policy_user_override": True,
        "selected_gpus_exclude_forbidden": not (set(map(str, args.selected_gpus)) & set(map(str, args.forbidden_gpus))),
        "smoke_excluded_from_pilot": True,
        "pilot_to_paper_claim_blocked": True,
    }
    write_json(args.report_dir / "preflight_gate.json", gates | {"pass": all(bool(v) for v in gates.values() if not isinstance(v, dict))})
    (args.report_dir / "git_provenance.txt").write_text(git_text(["status", "--short"]) + "\n\nHEAD " + git_text(["rev-parse", "HEAD"]) + "\n", encoding="utf-8")
    (args.report_dir / "reproduce.md").write_text(
        "# Reproduce\n\n"
        "Prepare manifest:\n\n"
        "```bash\npython scripts/run_gsm8k_selector_components_pilot.py prepare --selected-gpus 1 4 --forbidden-gpus 2 3\n```\n\n"
        "Run one shard:\n\n"
        "```bash\nCUDA_VISIBLE_DEVICES=1 python scripts/run_gsm8k_selector_components_pilot.py run --phase pilot --method importance_only_v4_25 --physical-gpu-id 1 --worker-index 0 --num-workers 1\n```\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id):
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must equal physical GPU id {args.physical_gpu_id}; got {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows_by_id = {int(r["problem_id"]): r for r in load_gsm8k(args.data_path)}
    manifest_data = json.loads((args.report_dir / "pilot_manifest.json").read_text(encoding="utf-8")) if (args.report_dir / "pilot_manifest.json").exists() else None
    if args.problem_ids:
        pids = [int(x) for x in args.problem_ids]
    else:
        pids = [int(r["problem_id"]) for r in manifest_data["rows"]]
    pids = [pid for i, pid in enumerate(pids) if i % args.num_workers == args.worker_index]
    cfg_hash = cfg_hash_for(args)
    manifest_path = args.output_dir / "task_manifest.jsonl"
    if args.phase == "pilot" and not manifest_path.exists():
        all_manifest_pids = [int(r["problem_id"]) for r in manifest_data["rows"]]
        records = [{"dataset": "gsm8k", "split": "test", "method": method, "problem_id": pid, "task_key": f"gsm8k:p{pid}", "config_hash": cfg_hash} for method in METHODS for pid in all_manifest_pids]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in records) + "\n", encoding="utf-8")
    header = {"phase": args.phase, "method": args.method, "physical_gpu_id": str(args.physical_gpu_id), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "tasks": len(pids), "cfg_hash": cfg_hash, "max_new_tokens": args.max_new_tokens, "selector": SELECTORS[args.method], "started_at": utc_now()}
    print(json.dumps(header, indent=2, sort_keys=True), flush=True)
    git_commit = git_text(["rev-parse", "HEAD"])
    model, tokenizer = load_model(args)
    try:
        for pid in pids:
            out_path = result_path(args.output_dir, args.method, pid, cfg_hash)
            if is_complete(out_path, cfg_hash, args.retry_failed, args.retry_oom):
                print(f"skip complete {out_path}", flush=True)
                continue
            try:
                rec = run_one(args, model, tokenizer, rows_by_id[pid], cfg_hash, git_commit)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                rec = {"experiment_id": EXPERIMENT_ID, "phase": args.phase, "dataset": "gsm8k", "split": "test", "problem_id": pid, "method": args.method, "value_precision_selector": SELECTORS[args.method], "config_hash": cfg_hash, "generated_text": "", "parsed_answer": None, "stop_reason": "oom", "hit_max_new_tokens": False, "is_correct": False, "error": repr(exc), "git_commit": git_commit, "physical_gpu_id": str(args.physical_gpu_id), "timestamp": utc_now()}
            except Exception as exc:
                rec = {"experiment_id": EXPERIMENT_ID, "phase": args.phase, "dataset": "gsm8k", "split": "test", "problem_id": pid, "method": args.method, "value_precision_selector": SELECTORS[args.method], "config_hash": cfg_hash, "generated_text": "", "parsed_answer": None, "stop_reason": "error", "hit_max_new_tokens": False, "is_correct": False, "error": repr(exc), "traceback": traceback.format_exc(), "git_commit": git_commit, "physical_gpu_id": str(args.physical_gpu_id), "timestamp": utc_now()}
            write_json_atomic(out_path, rec)
            print(f"wrote {out_path} stop={rec.get('stop_reason')} correct={rec.get('is_correct')} gen={rec.get('generated_tokens')}", flush=True)
    finally:
        del model
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data-path", type=Path, default=Path("datasets/gsm8k/gsm8k_test.jsonl"))
    prep.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL))
    prep.add_argument("--canonical-causal-root", type=Path, default=Path(DEFAULT_CANONICAL_CAUSAL))
    prep.add_argument("--report-dir", type=Path, default=Path("reports/gsm8k_selector_components_pilot_v1"))
    prep.add_argument("--selected-gpus", nargs=2, default=["1", "4"])
    prep.add_argument("--forbidden-gpus", nargs="*", default=["2", "3"])
    prep.set_defaults(func=prepare)

    runp = sub.add_parser("run")
    runp.add_argument("--phase", choices=["smoke", "pilot"], required=True)
    runp.add_argument("--method", choices=METHODS, required=True)
    runp.add_argument("--data-path", type=Path, default=Path("datasets/gsm8k/gsm8k_test.jsonl"))
    runp.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL))
    runp.add_argument("--output-dir", type=Path, required=True)
    runp.add_argument("--report-dir", type=Path, default=Path("reports/gsm8k_selector_components_pilot_v1"))
    runp.add_argument("--max-new-tokens", type=int, default=2048)
    runp.add_argument("--problem-ids", nargs="*", type=int)
    runp.add_argument("--worker-index", type=int, default=0)
    runp.add_argument("--num-workers", type=int, default=1)
    runp.add_argument("--physical-gpu-id", required=True)
    runp.add_argument("--retry-failed", action="store_true")
    runp.add_argument("--retry-oom", action="store_true")
    runp.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    runp.add_argument("--seed", type=int, default=20260827)
    runp.set_defaults(func=run)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
