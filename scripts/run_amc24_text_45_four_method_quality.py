#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.aime24_int2_wave1 import stable_hash  # noqa: E402
from bench.aime_utils import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    compute_stop_state,
    config_hash,
    normalize_eos_token_ids,
    set_all_seeds,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from bench.paper_config import apply_method_defaults, cache_storage_summary, method_config_dict  # noqa: E402
from evaluation.amc_source_answer_parser import (  # noqa: E402
    NORMALIZER_VERSION,
    normalize_answer,
    parse_amc_source_answer,
    score_majority,
)
from scripts.run_aime24_value_capacity_budget import effective_kv_bits  # noqa: E402


BENCHMARK_ID = "amc24_text_45"
EXPECTED_DATASET_SHA256 = "59a7450d9e480a41aa0d9db6dc2d89d16b1188cdf9a1ea8fd12e19dd2033c4b9"
EXPECTED_HEAD = "4ef0e19c12a69082a82aabf207ce06f6e172caa0"
FROZEN_RELEASE_SHA = "8d60485b5d2c93b7c1d478efc449de56d28159c3"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B"))
DATASET_PATH = ROOT / "datasets/amc24_text_45/amc24_text_45.jsonl"
PROTOCOL_PATH = ROOT / "datasets/amc24_text_45/protocol.json"
REPORT_DIR = ROOT / "reports/amc24_text_45_four_method_quality_v1"
RESULT_DIR = ROOT / "results/amc24_text_45_four_method_quality_v1"
LOG_DIR = ROOT / "run/amc24_text_45_four_method_quality_v1/logs"
PROGRESS_DIR = RESULT_DIR / "progress"
WORK_MANIFEST_PATH = REPORT_DIR / "work_manifest.json"
SHARD_PLAN_PATH = REPORT_DIR / "sharding_plan.json"
RESUME_VALIDATION_PATH = REPORT_DIR / "resume_validation.json"
FORMAL_CONFIG_VERSION = "amc24_text_45_four_method_quality_v1"
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_RESAMPLES = 10000
SEEDS = (42, 43, 44, 45, 46, 47, 48, 49)
METHOD_ORDER = ("FP16", "KIVI", "PatternKV", "CAUSAL")
METHOD_SLUG = {"FP16": "fp16", "KIVI": "kivi", "PatternKV": "patternkv", "CAUSAL": "causal_v4_25"}
DISPLAY_METHOD = {"FP16": "FP16", "KIVI": "KIVI", "PatternKV": "PatternKV", "CAUSAL": "CAUSAL-V4@25%"}
SMOKE_PROBLEM_IDS = ("12A_01", "12A_02")
SUBSET_PROBLEM_IDS = ("12A_01", "12A_02", "12A_03", "12A_04", "12A_05")
FEASIBILITY_PROBLEM_IDS = ("12A_01",)

METHOD_CONFIGS: dict[str, dict[str, Any]] = {
    "FP16": {
        "backend_method": "fp16",
        "method_arg": "fp16",
        "config_name": "fp16",
        "k_bits": 16,
        "v_bits": 16,
        "group_size": 0,
        "residual_length": 0,
        "sink_length": 0,
        "recent_length": 0,
        "selector": None,
        "v4_budget_fraction": None,
    },
    "KIVI": {
        "backend_method": "kivi_official",
        "method_arg": "kivi_paper_g128",
        "config_name": "kivi_paper_g128",
        "k_bits": 2,
        "v_bits": 2,
        "group_size": 128,
        "residual_length": 128,
        "sink_length": 0,
        "recent_length": 128,
        "selector": None,
        "v4_budget_fraction": None,
    },
    "PatternKV": {
        "backend_method": "patternkv",
        "method_arg": "patternkv_paper",
        "config_name": "patternkv_paper",
        "k_bits": 2,
        "v_bits": 2,
        "group_size": 128,
        "residual_length": 128,
        "sink_length": 0,
        "recent_length": 128,
        "selector": "base_v2",
        "v4_budget_fraction": 0.0,
        "num_k_base": 32,
        "num_v_base": 32,
    },
    "CAUSAL": {
        "backend_method": "patternkv",
        "method_arg": "patternkv",
        "config_name": "causal_v4_25",
        "k_bits": 2,
        "v_bits": 2,
        "group_size": 128,
        "residual_length": 128,
        "sink_length": 16,
        "recent_length": 128,
        "selector": "causal_v4",
        "v4_budget_fraction": 0.25,
        "num_k_base": 32,
        "num_v_base": 32,
    },
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()


def git_success(*args: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0


def head_descends_from_protocol_checkpoint() -> bool:
    head = git_text("rev-parse", "HEAD")
    return head == EXPECTED_HEAD or git_success("merge-base", "--is-ancestor", EXPECTED_HEAD, "HEAD")


def load_dataset() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 45:
        raise ValueError(f"AMC24-Text dataset must contain 45 rows, found {len(rows)}")
    ids = [str(r["problem_id"]) for r in rows]
    if len(set(ids)) != 45:
        raise ValueError("AMC24-Text problem IDs are not unique")
    if sha256_file(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise ValueError("AMC24-Text dataset SHA mismatch")
    for row in rows:
        if normalize_answer(row.get("answer")) is None:
            raise ValueError(f"gold answer does not normalize: {row.get('problem_id')}")
    return rows


def render_prompt(problem: str, tokenizer) -> tuple[str, str]:
    user_prompt = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": user_prompt}], tokenize=False, add_generation_prompt=True)
    rendered += "<think>\n"
    return rendered, user_prompt


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


def make_worker_args(method: str, physical_gpu: str) -> SimpleNamespace:
    cfg = METHOD_CONFIGS[method]
    args = SimpleNamespace()
    args.method = cfg["method_arg"]
    args.model_path = MODEL_PATH
    args.model_dtype = "float16"
    args.do_sample = True
    args.temperature = DEFAULT_TEMPERATURE
    args.top_p = DEFAULT_TOP_P
    args.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
    args.repetition_penalty = 1.0
    args.gpu_id = str(physical_gpu)
    args.k_bits = cfg["k_bits"]
    args.v_bits = cfg["v_bits"]
    args.group_size = cfg["group_size"]
    args.residual_length = cfg["residual_length"]
    args.sink_length = cfg["sink_length"]
    args.recent_length = cfg["recent_length"]
    args.config_name = cfg["config_name"]
    args.mixed_key_mask_path = None
    args.mixed_key_int4_ratio = 0.0
    args.mixed_key_mask_hash = ""
    args.patternkv_cache_path = "segmented"
    args.patternkv_cache_mode = "segmented_rolling"
    args.num_k_base = cfg.get("num_k_base", 32)
    args.num_v_base = cfg.get("num_v_base", 32)
    args.patternkv_value_objective = "base"
    args.patternkv_v_precision_selector = cfg.get("selector") or "base_v2"
    args.patternkv_v4_budget_fraction = float(cfg.get("v4_budget_fraction") or 0.0)
    args.patternkv_random_selector_seed = 20260809
    args.paper_method_config = apply_method_defaults(args)
    return args


def load_model(args):
    dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    backend = args.paper_method_config.backend_method
    if backend == "fp16":
        model = LlamaForCausalLM.from_pretrained(args.model_path, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True).to("cuda:0")
    elif backend == "kivi_official":
        from models.llama_kivi import LlamaForCausalLM_KIVI

        config = LlamaConfig.from_pretrained(args.model_path, local_files_only=True)
        config.k_bits = args.k_bits
        config.v_bits = args.v_bits
        config.group_size = args.group_size
        config.residual_length = args.residual_length
        config.sink_length = args.sink_length
        config.recent_length = args.recent_length
        config.use_flash = True
        model = LlamaForCausalLM_KIVI.from_pretrained(args.model_path, local_files_only=True, config=config, torch_dtype=dtype, low_cpu_mem_usage=True).to("cuda:0")
    elif backend == "patternkv":
        from models.llama_patternkv import LlamaForCausalLM_PatternKV

        config = LlamaConfig.from_pretrained(args.model_path, local_files_only=True)
        config.k_bits = args.k_bits
        config.v_bits = args.v_bits
        config.group_size = args.group_size
        config.residual_length = args.residual_length
        config.sink_length = args.sink_length
        config.recent_length = args.recent_length
        config.patternkv_cache_path = args.patternkv_cache_path
        config.patternkv_cache_mode = args.patternkv_cache_mode
        config.patternkv_value_objective = args.patternkv_value_objective
        config.patternkv_v_precision_selector = args.patternkv_v_precision_selector
        config.patternkv_v4_budget_fraction = args.patternkv_v4_budget_fraction
        config.patternkv_random_selector_seed = args.patternkv_random_selector_seed
        config.use_flash = True
        config.num_k_base = args.num_k_base
        config.num_v_base = args.num_v_base
        model = LlamaForCausalLM_PatternKV.from_pretrained(args.model_path, local_files_only=True, config=config, torch_dtype=dtype, low_cpu_mem_usage=True).to("cuda:0")
    else:
        raise ValueError(f"unsupported backend: {backend}")
    model.eval()
    return model, tokenizer


def set_selector_task_context(model: Any, selector_task_key: str) -> None:
    if hasattr(model, "config"):
        model.config.patternkv_selector_task_key = selector_task_key
    for layer in getattr(getattr(model, "model", None), "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            attn.selector_task_key = selector_task_key
            attn.v_causal_importance = None
            attn.v_oracle_importance = None


def method_identity(method: str, model: Any, args: SimpleNamespace) -> dict[str, Any]:
    cls = type(model).__name__
    cfg = method_config_dict(args)
    evidence = {
        "method": method,
        "display_method": DISPLAY_METHOD[method],
        "model_class": cls,
        "backend_method": args.paper_method_config.backend_method,
        "method_config": cfg,
        "active": False,
    }
    if method == "FP16":
        evidence["active"] = cls == "LlamaForCausalLM" and args.paper_method_config.backend_method == "fp16"
    elif method == "KIVI":
        evidence["active"] = "KIVI" in cls and args.paper_method_config.backend_method == "kivi_official" and cfg["k_bits"] == 2 and cfg["v_bits"] == 2 and cfg["group_size"] == 128 and cfg["residual_length"] == 128
    elif method == "PatternKV":
        evidence["active"] = "PatternKV" in cls and args.patternkv_v_precision_selector == "base_v2" and args.patternkv_v4_budget_fraction == 0.0 and args.num_k_base == 32 and args.num_v_base == 32
    elif method == "CAUSAL":
        evidence["active"] = "PatternKV" in cls and args.patternkv_v_precision_selector == "causal_v4" and args.patternkv_v4_budget_fraction == 0.25 and args.sink_length == 16 and args.recent_length == 128
    return evidence


def method_generation_hash(method: str) -> str:
    return config_hash({"version": FORMAL_CONFIG_VERSION, "method": method, "method_config": METHOD_CONFIGS[method], "sampling": frozen_generation_config()})


def formal_config_hash() -> str:
    return config_hash({"version": FORMAL_CONFIG_VERSION, "dataset_sha": EXPECTED_DATASET_SHA256, "methods": METHOD_CONFIGS, "seeds": SEEDS, "normalizer": NORMALIZER_VERSION})


def frozen_generation_config() -> dict[str, Any]:
    return {
        "prompt": "{problem}\\n\\nPlease reason step by step, and put your final answer within \\\\boxed{}.",
        "chat_template": "tokenizer.apply_chat_template(..., add_generation_prompt=True)",
        "assistant_suffix": "<think>\\n",
        "do_sample": True,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "top_k": None,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "responses_per_problem": 8,
        "seeds": list(SEEDS),
    }


def result_path(phase: str, method: str, problem_id: str, response_id: int) -> Path:
    return RESULT_DIR / phase / METHOD_SLUG[method] / f"{problem_id}" / f"r{response_id:02d}.json"


def raw_path(phase: str, method: str, problem_id: str, response_id: int) -> Path:
    return RESULT_DIR / phase / METHOD_SLUG[method] / f"{problem_id}" / f"r{response_id:02d}.txt"


def task_key(method: str, problem_id: str, response_id: int) -> str:
    return f"{BENCHMARK_ID}:{method}:{problem_id}:r{response_id}"


def identity_hash(value: str) -> int:
    return int(stable_hash({"identity": value}, 16), 16)


def formal_identity_records() -> list[dict[str, Any]]:
    rows = []
    for method in METHOD_ORDER:
        for row in load_dataset():
            problem_id = str(row["problem_id"])
            for response_id, seed in enumerate(SEEDS):
                identity = {
                    "benchmark": BENCHMARK_ID,
                    "method": method,
                    "problem_id": problem_id,
                    "response_id": response_id,
                    "seed": seed,
                    "dataset_sha256": EXPECTED_DATASET_SHA256,
                    "protocol_hash": formal_config_hash(),
                    "method_config_hash": config_hash(METHOD_CONFIGS[method]),
                    "status": "PENDING",
                    "attempt_count": 0,
                    "output_path": str(result_path("formal", method, problem_id, response_id).relative_to(ROOT)),
                    "task_key": task_key(method, problem_id, response_id),
                }
                rows.append(identity)
    return rows


def deterministic_shard_id(method: str, problem_id: str, response_id: int, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    key = f"{method}:{problem_id}:r{response_id}"
    return identity_hash(key) % num_shards


def apply_shard_filter(records: list[dict[str, Any]], *, method: str, shard_id: int | None, num_shards: int | None) -> list[dict[str, Any]]:
    if shard_id is None or num_shards is None or num_shards <= 1:
        return records
    return [row for row in records if deterministic_shard_id(method, str(row["problem_id"]), int(row["response_id"]), num_shards) == shard_id]


def progress_path(phase: str, method: str) -> Path:
    return PROGRESS_DIR / phase / f"{method}.json"


def gpu_runtime_snapshot(physical_gpu: str) -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for line in out.splitlines():
            idx, used, util = [part.strip() for part in line.split(",")]
            if idx == str(physical_gpu):
                return {"physical_gpu": str(physical_gpu), "memory_used_mib": int(used), "utilization_gpu_percent": int(util)}
    except Exception:
        pass
    return {"physical_gpu": str(physical_gpu), "memory_used_mib": None, "utilization_gpu_percent": None}


def write_progress(
    *,
    phase: str,
    method: str,
    physical_gpu: str,
    started_at: str,
    status: str,
    problem_id: str | None,
    response_id: int | None,
    seed: int | None,
    generated_tokens: int | None,
    wall_time_seconds: float | None,
    stop_reason: str | None,
    parser_failed: bool | None,
    truncated: bool | None,
    oom: bool | None,
    runtime_error: str | None,
) -> None:
    path = progress_path(phase, method)
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "phase": phase,
        "method": method,
        "physical_gpu": str(physical_gpu),
        "status": status,
        "problem_id": problem_id,
        "response_id": response_id,
        "seed": seed,
        "generated_tokens": generated_tokens,
        "wall_time_seconds": wall_time_seconds,
        "stop_reason": stop_reason,
        "parser_failed": parser_failed,
        "truncated": truncated,
        "oom": oom,
        "runtime_error": runtime_error,
        "gpu_snapshot": gpu_runtime_snapshot(physical_gpu),
        "started_at": started_at,
        "last_progress_at": utc_now(),
    }
    write_json(path, payload)


def update_work_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entries = formal_identity_records()
    by_key = {(r["method"], str(r["problem_id"]), int(r["response_id"])): r for r in rows}
    for entry in entries:
        key = (entry["method"], entry["problem_id"], int(entry["response_id"]))
        rec = by_key.get(key)
        if rec is None:
            continue
        entry["status"] = "COMPLETE" if rec.get("status") == "completed" else "ERROR"
        if rec.get("oom"):
            entry["status"] = "OOM"
        elif rec.get("status") == "failed" and rec.get("runtime_error"):
            entry["status"] = "ERROR"
        entry["attempt_count"] = int(rec.get("retry_count") or 0) + 1
        entry["output_path"] = rec.get("raw_generation_path") or entry["output_path"]
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "total_identities": len(entries),
        "generated_at": utc_now(),
        "statuses": {
            "PENDING": sum(entry["status"] == "PENDING" for entry in entries),
            "COMPLETE": sum(entry["status"] == "COMPLETE" for entry in entries),
            "ERROR": sum(entry["status"] == "ERROR" for entry in entries),
            "OOM": sum(entry["status"] == "OOM" for entry in entries),
            "INCOMPATIBLE": sum(entry["status"] == "INCOMPATIBLE" for entry in entries),
            "RUNNING": sum(entry["status"] == "RUNNING" for entry in entries),
        },
        "identities": entries,
    }
    write_json(WORK_MANIFEST_PATH, payload)
    return payload


def shard_plan(num_shards: int = 2) -> dict[str, Any]:
    plan = {}
    for method in METHOD_ORDER:
        all_records = [
            {"problem_id": row["problem_id"], "response_id": response_id}
            for row in load_dataset()
            for response_id, _seed in enumerate(SEEDS)
        ]
        counts = []
        for shard_id in range(num_shards):
            shard_rows = apply_shard_filter(all_records, method=method, shard_id=shard_id, num_shards=num_shards)
            counts.append({"shard_id": shard_id, "count": len(shard_rows), "sample_keys": [f"{method}:{r['problem_id']}:r{r['response_id']}" for r in shard_rows[:5]]})
        plan[method] = {"num_shards": num_shards, "shards": counts}
    payload = {"benchmark_id": BENCHMARK_ID, "generated_at": utc_now(), "methods": plan}
    write_json(SHARD_PLAN_PATH, payload)
    return payload


def is_current_record(path: Path, *, phase: str, method: str, problem_id: str, response_id: int) -> bool:
    if not path.exists():
        return False
    try:
        rec = read_json(path)
    except Exception:
        return False
    return (
        rec.get("status") == "completed"
        and rec.get("phase") == phase
        and rec.get("benchmark_id") == BENCHMARK_ID
        and rec.get("method") == method
        and rec.get("problem_id") == problem_id
        and int(rec.get("response_id", -1)) == response_id
        and rec.get("dataset_sha256") == EXPECTED_DATASET_SHA256
        and rec.get("formal_config_hash") == formal_config_hash()
        and rec.get("generation_config_hash") == method_generation_hash(method)
        and rec.get("normalizer_version") == NORMALIZER_VERSION
    )


def v4_stats(rec: dict[str, Any]) -> dict[str, Any]:
    stats = rec.get("patternkv_dynamic_stats") or {}
    selected = stats.get("v_precision_v4_tokens_per_layer") or []
    total = stats.get("v_precision_total_tokens_per_layer") or []
    rows = []
    num = den = 0
    for layer, (v4, tot) in enumerate(zip(selected, total)):
        v4_i = int(v4 or 0)
        tot_i = int(tot or 0)
        rows.append({"layer": layer, "v4_tokens": v4_i, "total_tokens": tot_i, "fraction": v4_i / tot_i if tot_i else None})
        num += v4_i
        den += tot_i
    return {"v4_tokens": num, "total_tokens": den, "fraction": num / den if den else None, "by_layer": rows}


def effective_bits(method: str, realized_fraction: float | None) -> float:
    if method == "FP16":
        return 16.0
    if method in {"KIVI", "PatternKV"}:
        return effective_kv_bits(0.0, precision_metadata=False)
    return effective_kv_bits(0.25 if realized_fraction is None else realized_fraction, precision_metadata=True)


def validate_context(rows: list[dict[str, Any]], tokenizer, model, args: SimpleNamespace) -> dict[str, Any]:
    max_prompt_tokens = 0
    max_prompt_problem_id = None
    prompt_hashes = []
    input_hashes = []
    for row in rows:
        rendered, _ = render_prompt(row["problem"], tokenizer)
        input_ids = tokenizer(rendered, add_special_tokens=False).input_ids
        if len(input_ids) > max_prompt_tokens:
            max_prompt_tokens = len(input_ids)
            max_prompt_problem_id = row["problem_id"]
        prompt_hashes.append(hashlib.sha256(rendered.encode("utf-8")).hexdigest())
        input_hashes.append(stable_hash({"input_ids": input_ids}, 32))
    max_position = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    required = max_prompt_tokens + args.max_new_tokens
    if max_position and required > int(max_position):
        raise ValueError(f"required max_model_len {required} exceeds max_position_embeddings {max_position}")
    return {
        "max_model_len": int(max_position) if max_position else None,
        "max_prompt_tokens": max_prompt_tokens,
        "max_prompt_problem_id": max_prompt_problem_id,
        "required_sequence_length": required,
        "max_new_tokens": args.max_new_tokens,
        "unique_prompt_hashes": len(set(prompt_hashes)),
        "unique_input_token_hashes": len(set(input_hashes)),
    }


@torch.no_grad()
def run_generation(args: SimpleNamespace, model, tokenizer, row: dict[str, Any], response_id: int, seed: int, phase: str, method: str) -> dict[str, Any]:
    set_all_seeds(seed)
    if args.paper_method_config.backend_method == "patternkv":
        from models.llama_patternkv import collect_patternkv_dynamic_stats, reset_patternkv_runtime_state

        reset_patternkv_runtime_state(model)
        set_selector_task_context(model, f"{BENCHMARK_ID}:{row['problem_id']}:r{response_id}:seed{seed}")
    else:
        collect_patternkv_dynamic_stats = None
    rendered_prompt, user_prompt = render_prompt(row["problem"], tokenizer)
    encoded = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to("cuda:0")
    attention_mask = encoded.attention_mask.to("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        repetition_penalty=1.0,
        num_return_sequences=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids(tokenizer, model),
        return_dict_in_generate=True,
        output_scores=False,
    )
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    seq = output.sequences
    generated_ids = seq[0, input_ids.shape[1] :].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    parsed = parse_amc_source_answer(generated_text)
    gold_key = normalize_answer(row["answer"])
    pred_key = parsed["canonical_answer_key"]
    total_tokens = int(seq.shape[1])
    cache_stats = cache_storage_summary(args.method, getattr(output, "past_key_values", None), model=model, total_cached_tokens=total_tokens, residual_length=args.residual_length)
    dynamic_stats = collect_patternkv_dynamic_stats(model, getattr(output, "past_key_values", None)) if collect_patternkv_dynamic_stats else {}
    prompt_hash = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
    input_token_hash = stable_hash({"input_ids": input_ids.detach().cpu().tolist()}, 32)
    generated_token_hash = stable_hash({"generated_ids": generated_ids}, 32)
    stop = compute_stop_state(generated_ids, DEFAULT_MAX_NEW_TOKENS, eos_ids(tokenizer, model))
    rec = {
        "benchmark_id": BENCHMARK_ID,
        "phase": phase,
        "status": "completed",
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "protocol_version": FORMAL_CONFIG_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "model": "DeepSeek-R1-Distill-Llama-8B",
        "model_path": str(MODEL_PATH),
        "method": method,
        "display_method": DISPLAY_METHOD[method],
        "method_config": METHOD_CONFIGS[method],
        "method_config_hash": config_hash(METHOD_CONFIGS[method]),
        "generation_config_hash": method_generation_hash(method),
        "formal_config_hash": formal_config_hash(),
        "problem_id": row["problem_id"],
        "competition": row["competition"],
        "problem_number": row["problem_number"],
        "ground_truth": row["answer"],
        "canonical_ground_truth_key": gold_key,
        "response_id": response_id,
        "seed": seed,
        "task_key": task_key(method, row["problem_id"], response_id),
        "prompt_hash": prompt_hash,
        "input_token_hash": input_token_hash,
        "prompt_token_count": int(input_ids.shape[1]),
        "user_prompt_hash": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
        "generated_token_count": len(generated_ids),
        "generated_token_hash": generated_token_hash,
        "parsed_answer": parsed["raw_answer"],
        "canonical_prediction_key": pred_key,
        "parser_strategy": parsed["parser_strategy"],
        "parser_error": parsed["parser_error"],
        "parser_failed": pred_key is None,
        "correct": pred_key is not None and gold_key is not None and pred_key == gold_key,
        "wall_time_seconds": round(wall, 4),
        "tokens_per_second": round(len(generated_ids) / wall, 4) if wall > 0 else None,
        "stop_reason": stop["stop_reason"],
        "truncated": stop["length_truncated"],
        "oom": False,
        "runtime_error": None,
        "retry_count": 0,
        "physical_gpu_id": args.gpu_id,
        "logical_gpu": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "cache_bitwidth_stats": cache_stats,
        "patternkv_dynamic_stats": dynamic_stats,
        "git_commit": git_text("rev-parse", "HEAD"),
        "timestamp": utc_now(),
        **stop,
    }
    del output, seq, input_ids, attention_mask, encoded
    return rec, generated_text


def failure_record(phase: str, method: str, row: dict[str, Any], response_id: int, seed: int, physical_gpu: str, error: str, retry_count: int) -> dict[str, Any]:
    is_oom = "out of memory" in error.lower()
    return {
        "benchmark_id": BENCHMARK_ID,
        "phase": phase,
        "status": "failed",
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "protocol_version": FORMAL_CONFIG_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "model": "DeepSeek-R1-Distill-Llama-8B",
        "model_path": str(MODEL_PATH),
        "method": method,
        "display_method": DISPLAY_METHOD[method],
        "method_config": METHOD_CONFIGS[method],
        "method_config_hash": config_hash(METHOD_CONFIGS[method]),
        "generation_config_hash": method_generation_hash(method),
        "formal_config_hash": formal_config_hash(),
        "problem_id": row["problem_id"],
        "competition": row["competition"],
        "problem_number": row["problem_number"],
        "ground_truth": row["answer"],
        "canonical_ground_truth_key": normalize_answer(row["answer"]),
        "response_id": response_id,
        "seed": seed,
        "task_key": task_key(method, row["problem_id"], response_id),
        "prompt_hash": None,
        "prompt_token_count": 0,
        "generated_token_count": 0,
        "parsed_answer": None,
        "canonical_prediction_key": None,
        "parser_failed": True,
        "correct": False,
        "stop_reason": "oom" if is_oom else "error",
        "truncated": False,
        "oom": is_oom,
        "runtime_error": error,
        "retry_count": retry_count,
        "physical_gpu_id": physical_gpu,
        "logical_gpu": "cuda:0",
        "git_commit": git_text("rev-parse", "HEAD"),
        "timestamp": utc_now(),
    }


def phase_problem_ids(phase: str, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    if phase == "smoke":
        return list(SMOKE_PROBLEM_IDS)
    if phase == "subset":
        return list(SUBSET_PROBLEM_IDS)
    if phase == "feasibility":
        return list(FEASIBILITY_PROBLEM_IDS)
    return [str(r["problem_id"]) for r in load_dataset()]


def phase_response_ids(phase: str, requested: list[int] | None) -> list[int]:
    if requested is not None:
        return requested
    if phase == "smoke":
        return [0]
    if phase == "subset":
        return [0, 1]
    if phase == "feasibility":
        return [0]
    return list(range(8))


def run_worker(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rows = load_dataset()
    row_by_id = {str(r["problem_id"]): r for r in rows}
    problem_ids = phase_problem_ids(args.phase, args.problem_ids.split(",") if args.problem_ids else None)
    response_ids = phase_response_ids(args.phase, [int(x) for x in args.response_ids.split(",")] if args.response_ids else None)
    method = args.method
    wargs = make_worker_args(method, args.physical_gpu)
    model = tokenizer = None
    status = "completed"
    failure = None
    started = time.time()
    started_at = utc_now()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "worker_start", "phase": args.phase, "method": method, "gpu": args.physical_gpu, "problems": problem_ids, "response_ids": response_ids}, sort_keys=True), flush=True)
    try:
        model, tokenizer = load_model(wargs)
        ident = method_identity(method, model, wargs)
        if not ident["active"]:
            raise RuntimeError(f"method identity gate failed: {ident}")
        context = validate_context(rows, tokenizer, model, wargs)
        write_json(REPORT_DIR / "method_identity_runtime" / f"{args.phase}_{METHOD_SLUG[method]}.json", {"identity": ident, "context": context, "timestamp": utc_now()})
        for problem_id in problem_ids:
            for response_id in response_ids:
                seed = SEEDS[response_id]
                if getattr(args, "num_shards", None) and getattr(args, "shard_id", None) is not None:
                    if deterministic_shard_id(method, str(problem_id), response_id, int(args.num_shards)) != int(args.shard_id):
                        continue
                out = result_path(args.phase, method, problem_id, response_id)
                if is_current_record(out, phase=args.phase, method=method, problem_id=problem_id, response_id=response_id):
                    print(json.dumps({"event": "skip", "method": method, "problem_id": problem_id, "response_id": response_id}, sort_keys=True), flush=True)
                    continue
                last_error = None
                for attempt in (0, 1):
                    try:
                        stop_event = threading.Event()

                        def heartbeat_loop() -> None:
                            while not stop_event.wait(120):
                                write_progress(
                                    phase=args.phase,
                                    method=method,
                                    physical_gpu=args.physical_gpu,
                                    started_at=started_at,
                                    status="running",
                                    problem_id=problem_id,
                                    response_id=response_id,
                                    seed=seed,
                                    generated_tokens=None,
                                    wall_time_seconds=round(time.time() - started, 3),
                                    stop_reason=None,
                                    parser_failed=None,
                                    truncated=None,
                                    oom=None,
                                    runtime_error=None,
                                )

                        heartbeat = threading.Thread(target=heartbeat_loop, daemon=True)
                        heartbeat.start()
                        write_progress(
                            phase=args.phase,
                            method=method,
                            physical_gpu=args.physical_gpu,
                            started_at=started_at,
                            status="running",
                            problem_id=problem_id,
                            response_id=response_id,
                            seed=seed,
                            generated_tokens=None,
                            wall_time_seconds=round(time.time() - started, 3),
                            stop_reason=None,
                            parser_failed=None,
                            truncated=None,
                            oom=None,
                            runtime_error=None,
                        )
                        rec, text = run_generation(wargs, model, tokenizer, row_by_id[problem_id], response_id, seed, args.phase, method)
                        rec["retry_count"] = attempt
                        raw = raw_path(args.phase, method, problem_id, response_id)
                        raw.parent.mkdir(parents=True, exist_ok=True)
                        raw.write_text(text, encoding="utf-8")
                        rec["raw_generation_path"] = str(raw.relative_to(ROOT))
                        rec["raw_generation_sha256"] = sha256_file(raw)
                        write_json_atomic(out, rec)
                        if args.phase == "formal":
                            update_work_manifest(collect_records("formal"))
                        write_progress(
                            phase=args.phase,
                            method=method,
                            physical_gpu=args.physical_gpu,
                            started_at=started_at,
                            status="complete",
                            problem_id=problem_id,
                            response_id=response_id,
                            seed=seed,
                            generated_tokens=int(rec.get("generated_token_count") or 0),
                            wall_time_seconds=float(rec.get("wall_time_seconds") or 0.0),
                            stop_reason=str(rec.get("stop_reason")),
                            parser_failed=bool(rec.get("parser_failed")),
                            truncated=bool(rec.get("truncated")),
                            oom=bool(rec.get("oom")),
                            runtime_error=None,
                        )
                        print(json.dumps({"event": "wrote", "method": method, "problem_id": problem_id, "response_id": response_id, "correct": rec["correct"], "tokens": rec["generated_token_count"], "stop": rec["stop_reason"]}, sort_keys=True), flush=True)
                        stop_event.set()
                        heartbeat.join(timeout=1.0)
                        break
                    except torch.cuda.OutOfMemoryError as exc:
                        torch.cuda.empty_cache()
                        last_error = repr(exc)
                    except Exception as exc:  # noqa: BLE001
                        last_error = repr(exc) + "\n" + traceback.format_exc()
                    finally:
                        try:
                            stop_event.set()
                            heartbeat.join(timeout=1.0)
                        except Exception:
                            pass
                    if attempt == 1:
                        rec = failure_record(args.phase, method, row_by_id[problem_id], response_id, seed, str(args.physical_gpu), str(last_error), attempt)
                        write_json_atomic(out, rec)
                        if args.phase == "formal":
                            update_work_manifest(collect_records("formal"))
                        write_progress(
                            phase=args.phase,
                            method=method,
                            physical_gpu=args.physical_gpu,
                            started_at=started_at,
                            status="error",
                            problem_id=problem_id,
                            response_id=response_id,
                            seed=seed,
                            generated_tokens=0,
                            wall_time_seconds=round(time.time() - started, 3),
                            stop_reason=rec.get("stop_reason"),
                            parser_failed=True,
                            truncated=False,
                            oom=bool(rec.get("oom")),
                            runtime_error=str(last_error),
                        )
                        status = "failed"
                        failure = str(last_error)
                        print(json.dumps({"event": "failed", "method": method, "problem_id": problem_id, "response_id": response_id, "error": str(last_error)[:500]}, sort_keys=True), flush=True)
                    else:
                        print(json.dumps({"event": "retry", "method": method, "problem_id": problem_id, "response_id": response_id, "error": str(last_error)[:500]}, sort_keys=True), flush=True)
    finally:
        try:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if args.phase == "formal":
            update_work_manifest(collect_records("formal"))
        write_worker_manifest({"phase": args.phase, "method": method, "physical_gpu": str(args.physical_gpu), "status": status, "failure": failure, "pid": os.getpid(), "started_at_unix": started, "ended_at_unix": time.time(), "runtime_seconds": round(time.time() - started, 3)})


def write_worker_manifest(row: dict[str, Any]) -> None:
    path = REPORT_DIR / "worker_manifest.json"
    rows = []
    if path.exists():
        try:
            rows = list(read_json(path).get("workers", []))
        except Exception:
            rows = []
    rows = [r for r in rows if not (r.get("phase") == row.get("phase") and r.get("method") == row.get("method") and r.get("pid") == row.get("pid"))]
    rows.append(row)
    write_json(path, {"updated_at": utc_now(), "workers": rows})


def query_gpu_rows() -> list[dict[str, Any]]:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        idx, name, used, total, util = [part.strip() for part in line.split(",")]
        rows.append({"index": idx, "name": name, "memory_used_mib": int(used), "memory_total_mib": int(total), "utilization_gpu_percent": int(util)})
    return rows


def query_compute_gpus() -> set[str]:
    try:
        out = subprocess.check_output(["nvidia-smi", "pmon", "-c", "1"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return set()
    busy = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[2] == "C":
            busy.add(parts[0])
    return busy


def select_gpus(requested: list[str] | None) -> tuple[list[str], dict[str, Any]]:
    gpu_rows = query_gpu_rows()
    busy = query_compute_gpus()
    requested_set = set(requested or [])
    idle = []
    for row in gpu_rows:
        idx = row["index"]
        ok = "RTX 3090" in row["name"] and (not requested_set or idx in requested_set) and idx not in busy and row["memory_used_mib"] <= 96 and row["utilization_gpu_percent"] <= 5
        row["foreign_compute_process"] = idx in busy
        row["idle_by_gate"] = ok
        if ok:
            idle.append(idx)
    return idle[:4], {"gpus": gpu_rows, "compute_busy_gpus": sorted(busy), "selected": idle[:4], "requested": requested or []}


def model_manifest() -> dict[str, Any]:
    files = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        path = MODEL_PATH / name
        if path.exists():
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    shards = sorted(MODEL_PATH.glob("*.safetensors")) + sorted(MODEL_PATH.glob("pytorch_model*.bin"))
    cfg = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=False, trust_remote_code=True)
    return {
        "path": str(MODEL_PATH),
        "identity": MODEL_PATH.name,
        "exists": MODEL_PATH.exists(),
        "model_type": getattr(cfg, "model_type", None),
        "architectures": getattr(cfg, "architectures", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "tokenizer_class": type(tok).__name__,
        "tokenizer_vocab_size": len(tok),
        "model_file_count": len(shards),
        "model_file_bytes": sum(p.stat().st_size for p in shards),
        "files": files,
        "valid": MODEL_PATH.exists() and (MODEL_PATH / "config.json").exists() and len(shards) > 0,
    }


def prompt_identity_manifest() -> dict[str, Any]:
    rows = load_dataset()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=False, trust_remote_code=True)
    problem = rows[0]["problem"]
    items = {}
    for method in METHOD_ORDER:
        rendered, user_prompt = render_prompt(problem, tok)
        ids = tok(rendered, add_special_tokens=False).input_ids
        items[method] = {
            "prompt_hash": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "input_token_hash": stable_hash({"input_ids": ids}, 32),
            "prompt_token_count": len(ids),
            "user_prompt_hash": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
        }
    return {
        "fixed_problem_id": rows[0]["problem_id"],
        "by_method": items,
        "prompt_hash_identical": len({v["prompt_hash"] for v in items.values()}) == 1,
        "input_token_hash_identical": len({v["input_token_hash"] for v in items.values()}) == 1,
        "prompt_token_count_identical": len({v["prompt_token_count"] for v in items.values()}) == 1,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    selected, gpu_snapshot = select_gpus(args.gpus.split(",") if args.gpus else None)
    protocol = read_json(PROTOCOL_PATH)
    dataset_sha = sha256_file(DATASET_PATH)
    release_sha = git_text("rev-parse", "release/causal-v4-25-system-final")
    prompt_identity = prompt_identity_manifest()
    work_manifest = update_work_manifest(collect_records("formal"))
    shards = shard_plan(getattr(args, "num_shards", 2))
    payload = {
        "branch": git_text("branch", "--show-current"),
        "head": git_text("rev-parse", "HEAD"),
        "expected_head": EXPECTED_HEAD,
        "head_descends_from_expected_protocol_checkpoint": head_descends_from_protocol_checkpoint(),
        "frozen_release_sha": release_sha,
        "release_unchanged": release_sha == FROZEN_RELEASE_SHA,
        "dataset_sha256": dataset_sha,
        "dataset_sha_match": dataset_sha == EXPECTED_DATASET_SHA256,
        "problem_count": len(load_dataset()),
        "protocol_version": FORMAL_CONFIG_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "protocol_manifest": protocol,
        "model": model_manifest(),
        "generation": frozen_generation_config(),
        "max_model_len": model_manifest().get("max_position_embeddings"),
        "method_configs": METHOD_CONFIGS,
        "method_config_hashes": {m: config_hash(METHOD_CONFIGS[m]) for m in METHOD_ORDER},
        "formal_config_hash": formal_config_hash(),
        "method_generation_hashes": {m: method_generation_hash(m) for m in METHOD_ORDER},
        "gpu_snapshot": gpu_snapshot,
        "gpu_mapping": {
            "FP16": selected[0] if len(selected) > 0 else None,
            "KIVI": selected[1] if len(selected) > 1 else None,
            "PatternKV": selected[2] if len(selected) > 2 else None,
            "CAUSAL": selected[3] if len(selected) > 3 else None,
        },
        "work_manifest": {
            "path": str(WORK_MANIFEST_PATH.relative_to(ROOT)),
            "total_identities": work_manifest["total_identities"],
            "statuses": work_manifest["statuses"],
        },
        "sharding_plan": {
            "path": str(SHARD_PLAN_PATH.relative_to(ROOT)),
            "methods": {method: {"num_shards": shards["methods"][method]["num_shards"], "shards": shards["methods"][method]["shards"]} for method in METHOD_ORDER},
        },
        "prompt_identity": prompt_identity,
        "sampling_assertions": {
            "temperature": DEFAULT_TEMPERATURE == 0.6,
            "top_p": DEFAULT_TOP_P == 0.95,
            "top_k": None,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS == 32768,
            "seeds": list(SEEDS) == [42, 43, 44, 45, 46, 47, 48, 49],
            "responses_per_problem": 8,
        },
        "runtime_versions": runtime_versions(),
        "timestamp": utc_now(),
    }
    payload["preflight_pass"] = (
        payload["branch"] == "sys/causal-v4-25-kernel-v1"
        and payload["head_descends_from_expected_protocol_checkpoint"]
        and payload["release_unchanged"]
        and payload["dataset_sha_match"]
        and payload["problem_count"] == 45
        and payload["normalizer_version"] == "amc24_text_normalizer_v1"
        and payload["model"]["valid"]
        and len(selected) == 4
        and prompt_identity["prompt_hash_identical"]
        and prompt_identity["input_token_hash_identical"]
        and all(v is True or v is None or isinstance(v, int) for v in payload["sampling_assertions"].values())
    )
    write_json(REPORT_DIR / "preflight.json", payload)
    write_json(REPORT_DIR / "gpu_mapping.json", payload["gpu_mapping"])
    write_json(RESUME_VALIDATION_PATH, {"work_manifest": payload["work_manifest"], "sharding_plan": payload["sharding_plan"], "timestamp": utc_now()})
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def runtime_versions() -> dict[str, Any]:
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    return {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers_version, "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda}


def launch(phase: str, selected: list[str], *, detach: bool = False, shard_id: int | None = None, num_shards: int | None = None, methods: list[str] | None = None) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    commands = []
    launch_methods = methods or list(METHOD_ORDER)
    for method, gpu in zip(launch_methods, selected):
        log_path = LOG_DIR / f"{phase}_{METHOD_SLUG[method]}.log"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--phase", phase, "--method", method, "--physical-gpu", str(gpu)]
        if shard_id is not None and num_shards is not None:
            cmd.extend(["--shard-id", str(shard_id), "--num-shards", str(num_shards)])
        commands.append({"method": method, "physical_gpu": gpu, "log": str(log_path.relative_to(ROOT)), "cmd": cmd})
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        log = log_path.open("a", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=detach), log))
    write_json(LOG_DIR / f"{phase}_launcher_manifest.json", {"phase": phase, "commands": commands, "detach": detach, "timestamp": utc_now()})
    if detach:
        write_json(LOG_DIR / f"{phase}_detached_pids.json", {"phase": phase, "pids": [proc.pid for proc, _ in procs], "timestamp": utc_now()})
        for _, log in procs:
            log.close()
        return 0
    rc = 0
    for proc, log in procs:
        rc = max(rc, proc.wait())
        log.close()
    return rc


def collect_records(phase: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((RESULT_DIR / phase).glob("*/*/r*.json")):
        try:
            rec = read_json(path)
            if rec.get("formal_config_hash") == formal_config_hash():
                rows.append(rec)
        except Exception:
            continue
    return rows


def expected_keys(phase: str) -> set[tuple[str, str, int]]:
    problems = phase_problem_ids(phase, None)
    response_ids = phase_response_ids(phase, None)
    return {(method, pid, rid) for method in METHOD_ORDER for pid in problems for rid in response_ids}


def completeness(phase: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = expected_keys(phase)
    accounted = {(r["method"], str(r["problem_id"]), int(r["response_id"])) for r in rows}
    completed = {(r["method"], str(r["problem_id"]), int(r["response_id"])) for r in rows if r.get("status") == "completed"}
    return {
        "phase": phase,
        "expected": len(expected),
        "accounted": len(accounted),
        "completed": len(completed),
        "missing": [{"method": m, "problem_id": p, "response_id": r} for m, p, r in sorted(expected - accounted)],
        "runtime_errors": sum(bool(r.get("runtime_error")) or r.get("status") == "failed" for r in rows),
        "oom": sum(bool(r.get("oom")) for r in rows),
        "complete": len(accounted) == len(expected) and not (expected - accounted),
    }


def majority_for_problem(rows: list[dict[str, Any]], gold: str) -> dict[str, Any]:
    by_response = {int(r["response_id"]): r for r in rows}
    keys = [by_response.get(i, {}).get("canonical_prediction_key") for i in range(8)]
    return score_majority(keys, gold)


def method_summary(rows: list[dict[str, Any]], dataset_rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    method_rows = [r for r in rows if r.get("method") == method]
    completed = [r for r in method_rows if r.get("status") == "completed"]
    tokens = [int(r.get("generated_token_count") or 0) for r in completed]
    maj_correct = 0
    for row in dataset_rows:
        qrows = [r for r in completed if r.get("problem_id") == row["problem_id"]]
        maj_correct += bool(majority_for_problem(qrows, row["answer"])["correct"])
    correct = sum(bool(r.get("correct")) for r in completed)
    return {
        "method": method,
        "display_method": DISPLAY_METHOD[method],
        "problems": 45,
        "responses_per_problem": 8,
        "generations": len(method_rows),
        "completed": len(completed),
        "correct_responses": correct,
        "Avg@8": correct / 360,
        "Maj@8_correct": maj_correct,
        "Maj@8": maj_correct / 45,
        "mean_generated_tokens": statistics.mean(tokens) if tokens else None,
        "median_generated_tokens": statistics.median(tokens) if tokens else None,
        "p95_generated_tokens": percentile(tokens, 0.95),
        "max_generated_tokens": max(tokens) if tokens else None,
        "mean_prompt_tokens": statistics.mean([int(r.get("prompt_token_count") or 0) for r in completed]) if completed else None,
        "truncations": sum(bool(r.get("truncated")) for r in completed),
        "parser_failures": sum(bool(r.get("parser_failed")) for r in completed),
        "eos_stops": sum(r.get("stop_reason") == "eos" for r in completed),
        "length_stops": sum(r.get("stop_reason") == "length" for r in completed),
        "OOM": sum(bool(r.get("oom")) for r in method_rows),
        "runtime_errors": sum(bool(r.get("runtime_error")) or r.get("status") == "failed" for r in method_rows),
        "retries": sum(int(r.get("retry_count") or 0) for r in method_rows),
    }


def percentile(values: list[int | float], q: float) -> float | None:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def per_seed_summary(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for method in METHOD_ORDER:
        out[method] = []
        for response_id, seed in enumerate(SEEDS):
            subset = [r for r in rows if r.get("method") == method and int(r.get("response_id", -1)) == response_id and r.get("status") == "completed"]
            correct = sum(bool(r.get("correct")) for r in subset)
            out[method].append({"response_id": response_id, "seed": seed, "correct": correct, "total": len(subset), "accuracy": correct / len(subset) if subset else None})
    return out


def per_question_summary(rows: list[dict[str, Any]], dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in dataset_rows:
        item: dict[str, Any] = {"problem_id": row["problem_id"], "competition": row["competition"], "problem_number": row["problem_number"], "ground_truth": row["answer"], "canonical_ground_truth_key": normalize_answer(row["answer"])}
        for method in METHOD_ORDER:
            qrows = sorted([r for r in rows if r.get("method") == method and r.get("problem_id") == row["problem_id"] and r.get("status") == "completed"], key=lambda r: int(r["response_id"]))
            maj = majority_for_problem(qrows, row["answer"])
            item[method] = {
                "parsed_answers": [r.get("parsed_answer") for r in qrows],
                "canonical_answers": [r.get("canonical_prediction_key") for r in qrows],
                "correct_bits": [bool(r.get("correct")) for r in qrows],
                "mean_correct": sum(bool(r.get("correct")) for r in qrows) / 8,
                "majority_prediction": maj["prediction"],
                "majority_correct": bool(maj["correct"]),
                "majority_tie": bool(maj["tie"]),
                "majority_votes": maj["votes"],
                "parser_failure_count": sum(bool(r.get("parser_failed")) for r in qrows),
                "truncation_count": sum(bool(r.get("truncated")) for r in qrows),
            }
        out.append(item)
    return out


def paired_bootstrap(question_rows: list[dict[str, Any]], metric_key: str) -> dict[str, Any]:
    rng = random.Random(BOOTSTRAP_SEED)
    out = {"bootstrap_unit": "question", "seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES}
    for baseline in ("FP16", "KIVI", "PatternKV"):
        vals = [float(q["CAUSAL"][metric_key]) - float(q[baseline][metric_key]) for q in question_rows]
        draws = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            draws.append(sum(vals[rng.randrange(len(vals))] for _ in range(len(vals))) / len(vals))
        draws.sort()
        out[f"CAUSAL_vs_{baseline}"] = {
            "mean_difference": statistics.mean(vals),
            "bootstrap_mean": statistics.mean(draws),
            "ci95_low": draws[int(0.025 * BOOTSTRAP_RESAMPLES)],
            "ci95_high": draws[int(0.975 * BOOTSTRAP_RESAMPLES) - 1],
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_phase(phase: str = "formal") -> dict[str, Any]:
    dataset_rows = load_dataset()
    rows = collect_records(phase)
    comp = completeness(phase, rows)
    full = {method: method_summary(rows, dataset_rows, method) for method in METHOD_ORDER}
    seed = per_seed_summary(rows)
    question = per_question_summary(rows, dataset_rows)
    avg_boot = paired_bootstrap(question, "mean_correct") if phase == "formal" and comp["complete"] else {}
    maj_boot = paired_bootstrap(question, "majority_correct") if phase == "formal" and comp["complete"] else {}
    deltas_avg = {f"CAUSAL_vs_{m}": full["CAUSAL"]["Avg@8"] - full[m]["Avg@8"] for m in ("FP16", "KIVI", "PatternKV")}
    deltas_maj = {f"CAUSAL_vs_{m}": full["CAUSAL"]["Maj@8"] - full[m]["Maj@8"] for m in ("FP16", "KIVI", "PatternKV")}
    summary = {
        "phase": phase,
        "benchmark_id": BENCHMARK_ID,
        "dataset_sha256": sha256_file(DATASET_PATH),
        "normalizer_version": NORMALIZER_VERSION,
        "completion": comp,
        "methods": full,
        "deltas_avg8": deltas_avg,
        "deltas_maj8": deltas_maj,
        "total_accounted": comp["accounted"],
        "timestamp": utc_now(),
    }
    if phase == "formal":
        write_json(REPORT_DIR / "completion_manifest.json", comp)
        write_json(REPORT_DIR / "full_summary.json", summary)
        write_json(REPORT_DIR / "per_seed_summary.json", seed)
        write_json(REPORT_DIR / "per_question_summary.json", question)
        write_json(REPORT_DIR / "paired_statistics.json", {"Avg@8": avg_boot, "Maj@8": maj_boot})
        write_tables(summary, avg_boot, maj_boot)
        write_analysis(rows, question, summary)
        write_final_gate(summary, avg_boot, maj_boot)
    else:
        write_json(REPORT_DIR / f"{phase}_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def format_pct(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.2f}%"


def write_tables(summary: dict[str, Any], avg_boot: dict[str, Any], maj_boot: dict[str, Any]) -> None:
    lines = ["# AMC24-Text Four-Method Quality", "", "| Method | Avg@8 | Maj@8 | Correct responses | Maj correct |", "|---|---:|---:|---:|---:|"]
    for method in METHOD_ORDER:
        row = summary["methods"][method]
        lines.append(f"| {DISPLAY_METHOD[method]} | {format_pct(row['Avg@8'])} | {format_pct(row['Maj@8'])} | {row['correct_responses']}/360 | {row['Maj@8_correct']}/45 |")
    (REPORT_DIR / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    delta = ["# AMC24-Text CAUSAL Deltas", "", "| Comparison | Delta Avg@8 | Delta Maj@8 | Avg 95% CI | Maj 95% CI |", "|---|---:|---:|---:|---:|"]
    for base in ("FP16", "KIVI", "PatternKV"):
        key = f"CAUSAL_vs_{base}"
        avg = avg_boot.get(key, {})
        maj = maj_boot.get(key, {})
        delta.append(f"| CAUSAL - {base} | {summary['deltas_avg8'][key]:.6f} | {summary['deltas_maj8'][key]:.6f} | [{avg.get('ci95_low')}, {avg.get('ci95_high')}] | [{maj.get('ci95_low')}, {maj.get('ci95_high')}] |")
    (REPORT_DIR / "delta_table.md").write_text("\n".join(delta) + "\n", encoding="utf-8")


def write_analysis(rows: list[dict[str, Any]], question: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    token_lines = ["# Token Length Analysis", "", "| Method | Mean | Median | P95 | Max | Truncated correct | Truncated incorrect |", "|---|---:|---:|---:|---:|---:|---:|"]
    parser_lines = ["# Parser Failure Analysis", ""]
    for method in METHOD_ORDER:
        mrows = [r for r in rows if r.get("method") == method and r.get("status") == "completed"]
        trunc_correct = sum(bool(r.get("truncated")) and bool(r.get("correct")) for r in mrows)
        trunc_incorrect = sum(bool(r.get("truncated")) and not bool(r.get("correct")) for r in mrows)
        m = summary["methods"][method]
        token_lines.append(f"| {DISPLAY_METHOD[method]} | {m['mean_generated_tokens']} | {m['median_generated_tokens']} | {m['p95_generated_tokens']} | {m['max_generated_tokens']} | {trunc_correct} | {trunc_incorrect} |")
        fails = [r for r in mrows if r.get("parser_failed")]
        parser_lines.append(f"## {DISPLAY_METHOD[method]}")
        parser_lines.append(f"- Parser failures: `{len(fails)}`")
        parser_lines.append(f"- IDs: `{[(r.get('problem_id'), r.get('response_id')) for r in fails]}`")
    (REPORT_DIR / "token_length_analysis.md").write_text("\n".join(token_lines) + "\n", encoding="utf-8")
    (REPORT_DIR / "parser_failure_analysis.md").write_text("\n".join(parser_lines) + "\n", encoding="utf-8")
    claim = claim_audit(summary)
    write_json(REPORT_DIR / "claim_audit.json", claim)
    (REPORT_DIR / "claim_audit.md").write_text(claim["markdown"], encoding="utf-8")
    readme = "# AMC24-Text Four-Method Quality V1\n\nFormal preregistered AMC24-Text quality evaluation for FP16, KIVI, PatternKV, and CAUSAL-V4@25%.\n\nRaw per-sample records are under `results/amc24_text_45_four_method_quality_v1/`; reports in this directory are generated from those records.\n"
    (REPORT_DIR / "README.md").write_text(readme, encoding="utf-8")
    method_lines = ["# Method Identity", ""]
    for path in sorted((REPORT_DIR / "method_identity_runtime").glob("*.json")):
        payload = read_json(path)
        ident = payload["identity"]
        method_lines.append(f"- {ident['display_method']}: active=`{ident['active']}`, backend=`{ident['backend_method']}`, class=`{ident['model_class']}`")
    (REPORT_DIR / "method_identity.md").write_text("\n".join(method_lines) + "\n", encoding="utf-8")


def claim_audit(summary: dict[str, Any]) -> dict[str, Any]:
    methods = summary["methods"]
    avg_causal = methods["CAUSAL"]["Avg@8"]
    maj_causal = methods["CAUSAL"]["Maj@8"]
    avg_pattern = methods["PatternKV"]["Avg@8"]
    maj_pattern = methods["PatternKV"]["Maj@8"]
    avg_kivi = methods["KIVI"]["Avg@8"]
    maj_kivi = methods["KIVI"]["Maj@8"]
    avg_fp16 = methods["FP16"]["Avg@8"]
    maj_fp16 = methods["FP16"]["Maj@8"]
    classification = "AMC24_TEXT_45_FOUR_METHOD_QUALITY_V1_SUPPORTED" if avg_causal > avg_pattern and maj_causal >= maj_pattern and avg_causal > avg_kivi and maj_causal >= maj_kivi else "AMC24_TEXT_45_FOUR_METHOD_QUALITY_V1_MIXED" if avg_causal >= avg_pattern or maj_causal >= maj_pattern else "AMC24_TEXT_45_FOUR_METHOD_QUALITY_V1_NOT_SUPPORTED"
    lines = [
        "# Claim Audit",
        "",
        f"- CAUSAL Avg@8 > PatternKV: `{avg_causal > avg_pattern}`",
        f"- CAUSAL Maj@8 > PatternKV: `{maj_causal > maj_pattern}`",
        f"- CAUSAL Avg@8 > KIVI: `{avg_causal > avg_kivi}`",
        f"- CAUSAL Maj@8 > KIVI: `{maj_causal > maj_kivi}`",
        f"- CAUSAL Avg@8 gap to FP16: `{avg_causal - avg_fp16:.6f}`",
        f"- CAUSAL Maj@8 gap to FP16: `{maj_causal - maj_fp16:.6f}`",
        f"- Classification: `{classification}`",
        "",
        "Paper-safe wording: AMC24-Text uses a public 45-problem text-only subset and deterministic preregistered scoring; these are our same-harness results, not a reproduction of PatternKV's unpublished exact AMC24 subset.",
    ]
    return {"classification": classification, "markdown": "\n".join(lines) + "\n", "main_paper_recommendation": "MAIN PAPER" if classification.endswith("SUPPORTED") else "APPENDIX_OR_LIMITATION"}


def write_final_gate(summary: dict[str, Any], avg_boot: dict[str, Any], maj_boot: dict[str, Any]) -> None:
    claim = claim_audit(summary)
    gate = {
        "classification": claim["classification"],
        "scoring_oracle_frozen": NORMALIZER_VERSION == "amc24_text_normalizer_v1",
        "dataset_sha_match": summary["dataset_sha256"] == EXPECTED_DATASET_SHA256,
        "generation_protocol_unchanged": True,
        "formal_matrix_complete": summary["completion"]["complete"] and summary["completion"]["accounted"] == 1440,
        "methods": list(METHOD_ORDER),
        "total_accounted": summary["completion"]["accounted"],
        "paired_bootstrap": {"Avg@8": avg_boot, "Maj@8": maj_boot},
        "next_task_recommendation": "PAPER_SELECTOR_COMPONENT_ABLATION_AIME24_V1",
        "timestamp": utc_now(),
    }
    write_json(REPORT_DIR / "final_gate.json", gate)


def status() -> None:
    payload = {phase: completeness(phase, collect_records(phase)) for phase in ("smoke", "subset", "formal")}
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--launch-smoke", action="store_true")
    parser.add_argument("--launch-subset", action="store_true")
    parser.add_argument("--launch-feasibility", action="store_true")
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--aggregate", choices=["smoke", "subset", "feasibility", "formal"])
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--phase", choices=["smoke", "subset", "feasibility", "formal"], default="formal")
    parser.add_argument("--method", choices=METHOD_ORDER)
    parser.add_argument("--physical-gpu")
    parser.add_argument("--problem-ids")
    parser.add_argument("--response-ids")
    parser.add_argument("--gpus", help="Comma-separated physical GPU IDs in FP16,KIVI,PatternKV,CAUSAL order.")
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--num-shards", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        if not args.method or args.physical_gpu is None:
            raise SystemExit("--worker requires --method and --physical-gpu")
        run_worker(args)
        return
    if args.preflight:
        preflight(args)
        return
    if args.launch_smoke or args.launch_subset or args.launch_feasibility or args.launch_formal:
        pf = read_json(REPORT_DIR / "preflight.json")
        if not pf.get("preflight_pass"):
            raise SystemExit("preflight_pass is false; refusing to launch")
        selected = [str(pf["gpu_mapping"][m]) for m in METHOD_ORDER]
        phase = "smoke" if args.launch_smoke else "subset" if args.launch_subset else "feasibility" if args.launch_feasibility else "formal"
        methods = None
        if args.launch_feasibility:
            methods = ["PatternKV", "CAUSAL"]
            selected = [pf["gpu_mapping"][m] for m in methods]
        raise SystemExit(launch(phase, selected, detach=args.detach, shard_id=args.shard_id, num_shards=args.num_shards, methods=methods))
    if args.aggregate:
        aggregate_phase(args.aggregate)
        return
    if args.status:
        status()
        return
    raise SystemExit("Specify --preflight, --launch-smoke, --launch-subset, --launch-formal, --aggregate, --status, or --worker.")


if __name__ == "__main__":
    main()
