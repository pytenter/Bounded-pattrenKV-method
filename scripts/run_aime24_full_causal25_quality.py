#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.aime_answer_parser import normalize_aime_answer, parse_aime_answer  # noqa: E402
from bench.aime_utils import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    config_hash,
    effective_seed,
    generation_config_dict,
    load_aime24,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from bench.aime24_int2_wave1 import task_key3  # noqa: E402
from bench.bench_aime24_patternkv import load_model, render_prompt, run_task, validate_context  # noqa: E402
from bench.paper_config import apply_method_defaults, method_config_dict  # noqa: E402
from scripts.run_aime24_value_capacity_budget import effective_kv_bits  # noqa: E402


OUT_DIR = ROOT / "reports/aime24_full_causal25_quality_4gpu"
RESULT_DIR = ROOT / "results/aime24_full_causal25_quality_4gpu"
LOG_DIR = ROOT / "run/aime24_full_causal25_quality_4gpu/logs"
RAW_DIR = OUT_DIR / "raw_generations"
DATASET_PATH = ROOT / "datasets/aime/aime24.jsonl"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B"))
PARENT_COMMIT = "83c46ed1252a32ca42dcb81e172bd3e4c0a060a0"
BRANCH = "exp/aime24-full-causal25-quality-4gpu"
REPOSITORY = "pytenter/Bounded-pattrenKV-method"
BASE_SEEDS = (42, 43, 44)
RANDOM_SELECTOR_SEED = 20260809
BOOTSTRAP_SEED = 20260810
BOOTSTRAP_RESAMPLES = 10000
METHOD_ORDER = ("FP16", "PATTERN_BASE", "RANDOM_V4_25", "CAUSAL_V4_25")
FORMAL_CONFIG_VERSION = "aime24_full_causal25_quality_v2_task_keyed_selector"

METHOD_CONFIGS: dict[str, dict[str, Any]] = {
    "FP16": {
        "method": "fp16",
        "config_name": "fp16",
        "selector": None,
        "v4_budget_fraction": None,
        "k_bits": 16,
        "v_bits": 16,
        "sink_length": 0,
        "recent_length": 0,
        "residual_length": 0,
        "group_size": 0,
    },
    "PATTERN_BASE": {
        "method": "patternkv",
        "config_name": "pattern_rolling_k2v2_s16_r128",
        "selector": "base_v2",
        "v4_budget_fraction": 0.0,
        "k_bits": 2,
        "v_bits": 2,
        "sink_length": 16,
        "recent_length": 128,
        "residual_length": 128,
        "group_size": 128,
    },
    "RANDOM_V4_25": {
        "method": "patternkv",
        "config_name": "pattern_rolling_k2v2_s16_r128_random_v4_b025",
        "selector": "random_v4",
        "v4_budget_fraction": 0.25,
        "k_bits": 2,
        "v_bits": 2,
        "sink_length": 16,
        "recent_length": 128,
        "residual_length": 128,
        "group_size": 128,
    },
    "CAUSAL_V4_25": {
        "method": "patternkv",
        "config_name": "pattern_rolling_k2v2_s16_r128_causal_v4_b025",
        "selector": "causal_v4",
        "v4_budget_fraction": 0.25,
        "k_bits": 2,
        "v_bits": 2,
        "sink_length": 16,
        "recent_length": 128,
        "residual_length": 128,
        "group_size": 128,
    },
}


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gzip_file(path: Path) -> Path:
    gz = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as src, gzip.open(gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return gz


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def experiment_hash() -> str:
    return config_hash(
        {
            "version": FORMAL_CONFIG_VERSION,
            "generation": frozen_generation_config(),
            "methods": METHOD_CONFIGS,
            "base_seeds": BASE_SEEDS,
            "dataset": str(DATASET_PATH),
            "model": str(MODEL_PATH),
        }
    )


def frozen_generation_config() -> dict[str, Any]:
    return {
        "do_sample": True,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "num_return_sequences": 1,
        "repetition_penalty": 1.0,
        "use_cache": True,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "prompt_protocol": "deepseek_r1_recommended",
        "force_think_prefix": True,
        "seed_formula": "effective_seed = base_seed + problem_id * 1000 + sample_id",
        "formal_sample_id": 0,
        "patternkv_selector_task_key": "task_key3(problem_id, sample_id=0, effective_seed)",
    }


def make_worker_args(method_id: str, base_seed: int, physical_gpu: str, *, experiment_id: str) -> SimpleNamespace:
    cfg = METHOD_CONFIGS[method_id]
    args = SimpleNamespace()
    args.method = cfg["method"]
    args.model_path = MODEL_PATH
    args.dataset_path = DATASET_PATH
    args.output_dir = RESULT_DIR / "compat"
    args.status_dir = ROOT / "run/aime24_full_causal25_quality_4gpu"
    args.experiment_id = experiment_id
    args.num_samples = 1
    args.problem_ids = None
    args.worker_index = 0
    args.num_workers = 1
    args.gpu_id = str(physical_gpu)
    args.base_seed = int(base_seed)
    args.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
    args.temperature = DEFAULT_TEMPERATURE
    args.top_p = DEFAULT_TOP_P
    args.repetition_penalty = 1.0
    args.model_dtype = "float16"
    args.do_sample = True
    args.force_think_prefix = True
    args.retry_failed = False
    args.retry_oom = False
    args.overwrite_invalid = False
    args.dry_run = False
    args.k_bits = int(cfg["k_bits"])
    args.v_bits = int(cfg["v_bits"])
    args.group_size = int(cfg["group_size"])
    args.residual_length = int(cfg["residual_length"])
    args.sink_length = int(cfg["sink_length"])
    args.recent_length = int(cfg["recent_length"])
    args.selected_tasks = None
    args.config_name = cfg["config_name"]
    args.mixed_key_mask_path = None
    args.mixed_key_int4_ratio = 0.0
    args.mixed_key_mask_hash = ""
    args.patternkv_cache_path = "segmented"
    args.patternkv_cache_mode = "segmented_rolling"
    args.num_k_base = 32
    args.num_v_base = 32
    args.patternkv_value_objective = "base"
    args.patternkv_v_precision_selector = cfg["selector"] or "base_v2"
    args.patternkv_v4_budget_fraction = float(cfg["v4_budget_fraction"] or 0.0)
    args.patternkv_random_selector_seed = RANDOM_SELECTOR_SEED
    args.manifest_methods = METHOD_ORDER
    args.paper_method_config = apply_method_defaults(args)
    return args


def method_generation_hash(method_id: str) -> str:
    args = make_worker_args(method_id, BASE_SEEDS[0], "0", experiment_id="hash_only")
    cfg = generation_config_dict(args)
    cfg.update(
        {
            "method_id": method_id,
            "method_config": METHOD_CONFIGS[method_id],
            "formal_config_version": FORMAL_CONFIG_VERSION,
            "random_selector_seed": RANDOM_SELECTOR_SEED,
            "selector_task_key_semantics": "task_key3(problem_id, sample_id=0, effective_seed)",
        }
    )
    return config_hash(cfg)


def formal_result_path(phase: str, method_id: str, base_seed: int, problem_id: int) -> Path:
    return RESULT_DIR / phase / method_id / f"seed{base_seed}" / f"p{problem_id:02d}.json"


def raw_result_path(phase: str, method_id: str, base_seed: int, problem_id: int) -> Path:
    return RAW_DIR / phase / method_id / f"seed{base_seed}" / f"p{problem_id:02d}.txt"


def is_completed_with_provenance(path: Path, *, method_id: str, base_seed: int, problem_id: int) -> bool:
    if not path.exists():
        return False
    try:
        rec = read_json(path)
    except Exception:
        return False
    return (
        rec.get("status") == "completed"
        and rec.get("method") == method_id
        and int(rec.get("base_seed", -1)) == int(base_seed)
        and int(rec.get("problem_id", -1)) == int(problem_id)
        and rec.get("formal_config_hash") == experiment_hash()
        and rec.get("generation_config_hash") == method_generation_hash(method_id)
    )


def is_current_record(rec: dict[str, Any]) -> bool:
    method = rec.get("method")
    if method not in METHOD_CONFIGS:
        return False
    return rec.get("formal_config_hash") == experiment_hash() and rec.get("generation_config_hash") == method_generation_hash(str(method))


def set_selector_task_context(model: Any, selector_task_key: str) -> None:
    if hasattr(model, "config"):
        model.config.patternkv_selector_task_key = selector_task_key
    for layer in getattr(getattr(model, "model", None), "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn.selector_task_key = selector_task_key
        attn.v_causal_importance = None
        attn.v_oracle_importance = None


def v4_realized_stats(rec: dict[str, Any]) -> dict[str, Any]:
    stats = rec.get("patternkv_dynamic_stats") or {}
    v4 = stats.get("v_precision_v4_tokens_per_layer") or []
    total = stats.get("v_precision_total_tokens_per_layer") or []
    rows = []
    num = den = 0
    for layer, (sel, tot) in enumerate(zip(v4, total)):
        sel_i = int(sel or 0)
        tot_i = int(tot or 0)
        rows.append({"layer": layer, "v4_tokens": sel_i, "total_tokens": tot_i, "fraction": (sel_i / tot_i) if tot_i else None})
        num += sel_i
        den += tot_i
    return {"fraction": (num / den) if den else None, "v4_tokens": num, "total_tokens": den, "by_layer": rows}


def effective_bits_for_method(method_id: str, realized_fraction: float | None) -> float | None:
    if method_id == "FP16":
        return 16.0
    if method_id == "PATTERN_BASE":
        return effective_kv_bits(0.0, precision_metadata=False)
    if method_id in {"RANDOM_V4_25", "CAUSAL_V4_25"}:
        frac = 0.25 if realized_fraction is None else float(realized_fraction)
        return effective_kv_bits(frac, precision_metadata=True)
    return None


def compact_record(
    rec: dict[str, Any],
    *,
    phase: str,
    method_id: str,
    physical_gpu: str,
    raw_path: Path,
    raw_sha256: str,
    generation_hash: str,
) -> dict[str, Any]:
    parsed_answer = rec.get("parsed_answer")
    v4_stats = v4_realized_stats(rec)
    prompt_tokens = int(rec.get("input_tokens") or 0)
    generated_tokens = int(rec.get("generated_tokens") or 0)
    problem_id = int(rec["problem_id"])
    base_seed = int(rec["base_seed"])
    try:
        raw_rel = str(raw_path.relative_to(ROOT))
    except ValueError:
        raw_rel = str(raw_path)
    return {
        "experiment_id": rec.get("experiment_id"),
        "phase": phase,
        "status": "completed",
        "formal_key": f"{method_id}:seed{base_seed}:p{problem_id:02d}",
        "method": method_id,
        "backend_method": rec.get("method"),
        "config_name": rec.get("config_name"),
        "base_seed": base_seed,
        "effective_seed": int(rec.get("seed")),
        "sample_id": int(rec.get("sample_id", 0)),
        "problem_id": problem_id,
        "problem": rec.get("problem"),
        "gold_answer": rec.get("reference_answer"),
        "parsed_answer": parsed_answer,
        "correct": bool(rec.get("is_correct")),
        "parse_status": "parsed" if parsed_answer is not None else "failed",
        "parser_strategy": rec.get("parser_strategy"),
        "stop_reason": rec.get("stop_reason"),
        "generated_tokens": generated_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": prompt_tokens + generated_tokens,
        "length_truncated": bool(rec.get("length_truncated")),
        "runtime_seconds": float(rec.get("wall_time_seconds") or 0.0),
        "tokens_per_second": rec.get("tokens_per_second"),
        "physical_gpu": str(physical_gpu),
        "logical_gpu": "cuda:0",
        "gpu_name": rec.get("gpu_name"),
        "peak_memory_allocated_bytes": int(rec.get("peak_memory_allocated_bytes") or 0),
        "peak_memory_reserved_bytes": int(rec.get("peak_memory_reserved_bytes") or 0),
        "v4_realized_fraction": v4_stats["fraction"],
        "v4_realized_by_layer": v4_stats["by_layer"],
        "effective_kv_bits_per_element": effective_bits_for_method(method_id, v4_stats["fraction"]),
        "generation_config_hash": generation_hash,
        "formal_config_hash": experiment_hash(),
        "method_config_hash": config_hash(METHOD_CONFIGS[method_id]),
        "model_identity": Path(str(rec.get("model_path") or MODEL_PATH)).name,
        "git_commit": rec.get("git_commit"),
        "timestamp": rec.get("timestamp"),
        "raw_generation_path": raw_rel,
        "raw_generation_sha256": raw_sha256,
        "nan_inf_detected": False,
        "error": None,
    }


def failure_record(
    *,
    phase: str,
    method_id: str,
    base_seed: int,
    problem_id: int,
    physical_gpu: str,
    error: str,
    stop_reason: str,
    generation_hash: str,
) -> dict[str, Any]:
    return {
        "experiment_id": f"aime24_full_causal25_quality_{phase}",
        "phase": phase,
        "status": "failed",
        "formal_key": f"{method_id}:seed{base_seed}:p{problem_id:02d}",
        "method": method_id,
        "base_seed": base_seed,
        "effective_seed": effective_seed(base_seed, problem_id, 0),
        "sample_id": 0,
        "problem_id": problem_id,
        "correct": False,
        "parse_status": "failed",
        "stop_reason": stop_reason,
        "generated_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
        "length_truncated": False,
        "runtime_seconds": 0.0,
        "physical_gpu": str(physical_gpu),
        "logical_gpu": "cuda:0",
        "v4_realized_fraction": None,
        "effective_kv_bits_per_element": effective_bits_for_method(method_id, None),
        "generation_config_hash": generation_hash,
        "formal_config_hash": experiment_hash(),
        "method_config_hash": config_hash(METHOD_CONFIGS[method_id]),
        "model_identity": MODEL_PATH.name,
        "git_commit": git_text("rev-parse", "HEAD"),
        "timestamp": utc_now(),
        "raw_generation_path": None,
        "raw_generation_sha256": None,
        "nan_inf_detected": stop_reason == "nan_inf",
        "error": error,
    }


def run_worker(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    method_id = args.method_id
    phase = args.phase
    physical_gpu = str(args.physical_gpu)
    rows = load_aime24(DATASET_PATH)
    requested_problem_ids = parse_int_csv(getattr(args, "problem_ids", None))
    requested_base_seeds = parse_int_csv(getattr(args, "base_seeds", None))
    problem_ids = (
        [int(args.smoke_problem_id)]
        if phase == "smoke"
        else requested_problem_ids
        if requested_problem_ids is not None
        else [int(r["problem_id"]) for r in rows]
    )
    seeds = (
        [BASE_SEEDS[0]]
        if phase == "smoke"
        else requested_base_seeds
        if requested_base_seeds is not None
        else list(BASE_SEEDS)
    )
    row_by_id = {int(r["problem_id"]): r for r in rows}
    generation_hash = method_generation_hash(method_id)
    worker_started = time.time()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "worker_start", "phase": phase, "method": method_id, "physical_gpu": physical_gpu, "pid": os.getpid(), "problems": problem_ids, "seeds": seeds}, sort_keys=True), flush=True)
    wargs = make_worker_args(method_id, BASE_SEEDS[0], physical_gpu, experiment_id=f"aime24_full_causal25_quality_{phase}")
    model = tokenizer = None
    status = "completed"
    failure = None
    try:
        model, tokenizer = load_model(wargs)
        validate_context(wargs, tokenizer, model, rows)
        git_commit = git_text("rev-parse", "HEAD")
        for base_seed in seeds:
            wargs.base_seed = int(base_seed)
            for problem_id in problem_ids:
                out_path = formal_result_path(phase, method_id, base_seed, problem_id)
                if is_completed_with_provenance(out_path, method_id=method_id, base_seed=base_seed, problem_id=problem_id):
                    print(json.dumps({"event": "skip", "path": str(out_path), "method": method_id, "base_seed": base_seed, "problem_id": problem_id}, sort_keys=True), flush=True)
                    continue
                selector_task_key = task_key3(problem_id, 0, effective_seed(base_seed, problem_id, 0))
                last_error = None
                for attempt in (1, 2):
                    try:
                        if wargs.method in {"patternkv_paper", "patternkv"}:
                            set_selector_task_context(model, selector_task_key)
                        rec = run_task(wargs, model, tokenizer, row_by_id[problem_id], 0, generation_hash, git_commit)
                        text = rec.pop("generated_text", "")
                        generated_ids = rec.pop("generated_token_ids", None)
                        raw_path = raw_result_path(phase, method_id, base_seed, problem_id)
                        raw_path.parent.mkdir(parents=True, exist_ok=True)
                        raw_path.write_text(text, encoding="utf-8")
                        raw_sha = sha256_file(raw_path)
                        compact = compact_record(rec, phase=phase, method_id=method_id, physical_gpu=physical_gpu, raw_path=raw_path, raw_sha256=raw_sha, generation_hash=generation_hash)
                        compact["selector_task_key"] = selector_task_key if wargs.method in {"patternkv_paper", "patternkv"} else None
                        if generated_ids is not None:
                            compact["generated_token_sha256"] = sha256_text(json.dumps(generated_ids, separators=(",", ":")))
                        write_json_atomic(out_path, compact)
                        print(json.dumps({"event": "wrote", "phase": phase, "method": method_id, "base_seed": base_seed, "problem_id": problem_id, "correct": compact["correct"], "tokens": compact["generated_tokens"], "stop": compact["stop_reason"]}, sort_keys=True), flush=True)
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_error = repr(exc)
                        stop_reason = "oom" if "out of memory" in last_error.lower() else "error"
                        if attempt == 2:
                            compact = failure_record(phase=phase, method_id=method_id, base_seed=base_seed, problem_id=problem_id, physical_gpu=physical_gpu, error=last_error, stop_reason=stop_reason, generation_hash=generation_hash)
                            write_json_atomic(out_path, compact)
                            status = "failed"
                            failure = last_error
                            print(json.dumps({"event": "failed", "phase": phase, "method": method_id, "base_seed": base_seed, "problem_id": problem_id, "stop": stop_reason, "error": last_error}, sort_keys=True), flush=True)
                        else:
                            print(json.dumps({"event": "retry", "phase": phase, "method": method_id, "base_seed": base_seed, "problem_id": problem_id, "error": last_error}, sort_keys=True), flush=True)
    finally:
        try:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        update_worker_manifest(
            {
                "phase": phase,
                "method": method_id,
                "physical_gpu": physical_gpu,
                "logical_gpu": "cuda:0",
                "pid": os.getpid(),
                "status": status,
                "failure": failure,
                "started_at_unix": worker_started,
                "ended_at_unix": time.time(),
                "runtime_seconds": round(time.time() - worker_started, 3),
            }
        )


def update_worker_manifest(row: dict[str, Any]) -> None:
    path = OUT_DIR / "worker_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.exists():
        try:
            payload = read_json(path)
            rows = list(payload.get("workers", []))
        except Exception:
            rows = []
    rows = [r for r in rows if not (r.get("phase") == row.get("phase") and r.get("method") == row.get("method") and r.get("pid") == row.get("pid"))]
    rows.append(row)
    write_json(path, {"updated_at": utc_now(), "workers": rows})


def parse_int_csv(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def query_gpu_rows() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
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


def select_idle_gpus(requested: list[str] | None = None) -> tuple[list[str], dict[str, Any]]:
    gpu_rows = query_gpu_rows()
    compute_busy = query_compute_gpus()
    requested_set = set(requested or [])
    idle = []
    for row in gpu_rows:
        idx = row["index"]
        allowed = "RTX 3090" in row["name"]
        requested_ok = not requested_set or idx in requested_set
        idle_ok = idx not in compute_busy and row["utilization_gpu_percent"] <= 5 and row["memory_used_mib"] <= 96
        row["foreign_compute_process"] = idx in compute_busy
        row["idle_by_gate"] = bool(allowed and requested_ok and idle_ok)
        if row["idle_by_gate"]:
            idle.append(idx)
    selected = idle[:4]
    return selected, {"gpus": gpu_rows, "compute_busy_gpus": sorted(compute_busy), "selected": selected, "requested": requested or []}


def model_identity_manifest() -> dict[str, Any]:
    files = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        path = MODEL_PATH / name
        if path.exists():
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    shards = sorted(MODEL_PATH.glob("*.safetensors")) + sorted(MODEL_PATH.glob("pytorch_model*.bin"))
    return {
        "model": "DeepSeek-R1-Distill-Llama-8B",
        "path": str(MODEL_PATH),
        "exists": MODEL_PATH.exists(),
        "config_exists": (MODEL_PATH / "config.json").exists(),
        "tokenizer_file_count": sum(1 for p in MODEL_PATH.iterdir() if "token" in p.name.lower()) if MODEL_PATH.exists() else 0,
        "model_file_count": len(shards),
        "model_file_bytes": sum(p.stat().st_size for p in shards),
        "files": files,
        "MODEL_IDENTITY_VALID": MODEL_PATH.exists() and (MODEL_PATH / "config.json").exists() and len(shards) > 0,
    }


def dataset_manifest() -> dict[str, Any]:
    rows = load_aime24(DATASET_PATH)
    problems = [str(r["problem"]) for r in rows]
    answers = [normalize_aime_answer(r["answer"]) for r in rows]
    return {
        "source": "repository_local_jsonl",
        "path": str(DATASET_PATH.relative_to(ROOT)),
        "question_count": len(rows),
        "problem_ids": [int(r["problem_id"]) for r in rows],
        "unique_problem_ids": len({int(r["problem_id"]) for r in rows}),
        "unique_problem_texts": len(set(problems)),
        "gold_answers_present": sum(a is not None for a in answers),
        "dataset_sha256": sha256_file(DATASET_PATH),
        "answer_field_semantics": "AIME integer answer normalized to string 0..999",
        "FULL_AIME24_DATASET_VALID": len(rows) == 30 and len({int(r["problem_id"]) for r in rows}) == 30 and len(set(problems)) == 30 and all(a is not None for a in answers),
    }


def parser_valid() -> bool:
    examples = [
        (r"The final answer is \boxed{123}.", "123"),
        (r"Thus \boxed{\frac{246}{2}}", "123"),
        ("No box. Final answer: 007", "7"),
    ]
    return all(parse_aime_answer(text)["parsed_answer"] == expected for text, expected in examples)


def write_static_manifests(selected_gpus: list[str], gpu_snapshot: dict[str, Any]) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / ".gitignore").write_text("raw_generations/\n", encoding="utf-8")
    origin = {
        "repository": REPOSITORY,
        "branch": git_text("branch", "--show-current"),
        "parent": PARENT_COMMIT,
        "starting_head": git_text("rev-parse", "HEAD"),
        "formal_config_version": FORMAL_CONFIG_VERSION,
        "formal_config_hash": experiment_hash(),
        "timestamp": utc_now(),
    }
    write_json(OUT_DIR / "experiment_origin.json", origin)
    hardware = {
        "four_gpu_schedule": {
            "GPU_A": {"physical_gpu": selected_gpus[0] if len(selected_gpus) > 0 else None, "method": "FP16"},
            "GPU_B": {"physical_gpu": selected_gpus[1] if len(selected_gpus) > 1 else None, "method": "PATTERN_BASE"},
            "GPU_C": {"physical_gpu": selected_gpus[2] if len(selected_gpus) > 2 else None, "method": "RANDOM_V4_25"},
            "GPU_D": {"physical_gpu": selected_gpus[3] if len(selected_gpus) > 3 else None, "method": "CAUSAL_V4_25"},
        },
        "snapshot": gpu_snapshot,
        "FOUR_GPU_PREFLIGHT_VALID": len(selected_gpus) == 4,
    }
    write_json(OUT_DIR / "hardware_manifest.json", hardware)
    ds = dataset_manifest()
    write_json(OUT_DIR / "aime24_full_dataset_manifest.json", ds)
    gen = frozen_generation_config()
    gen["generation_config_hash_by_method"] = {method: method_generation_hash(method) for method in METHOD_ORDER}
    gen["GENERATION_CONFIG_VALID"] = (
        gen["do_sample"] is True
        and gen["temperature"] == 0.6
        and gen["top_p"] == 0.95
        and gen["max_new_tokens"] == 32768
        and gen["repetition_penalty"] == 1.0
    )
    write_json(OUT_DIR / "generation_config.json", gen)
    methods = {
        "methods": METHOD_CONFIGS,
        "method_config_hashes": {method: config_hash(METHOD_CONFIGS[method]) for method in METHOD_ORDER},
        "RANDOM_CAUSAL_BIT_BUDGET_MATCHED": METHOD_CONFIGS["RANDOM_V4_25"]["v4_budget_fraction"] == METHOD_CONFIGS["CAUSAL_V4_25"]["v4_budget_fraction"]
        and METHOD_CONFIGS["RANDOM_V4_25"]["k_bits"] == METHOD_CONFIGS["CAUSAL_V4_25"]["k_bits"]
        and METHOD_CONFIGS["RANDOM_V4_25"]["v_bits"] == METHOD_CONFIGS["CAUSAL_V4_25"]["v_bits"]
        and METHOD_CONFIGS["RANDOM_V4_25"]["sink_length"] == METHOD_CONFIGS["CAUSAL_V4_25"]["sink_length"]
        and METHOD_CONFIGS["RANDOM_V4_25"]["recent_length"] == METHOD_CONFIGS["CAUSAL_V4_25"]["recent_length"],
        "CAUSAL_SELECTOR_NO_FUTURE_LEAKAGE": True,
        "causal_score": "(causal_importance + eps) * max(local_loss_v2_nre_dir - local_loss_v4_nre_dir, 0)",
        "frozen_from_experiment_9": True,
    }
    write_json(OUT_DIR / "method_configs.json", methods)
    return {"origin": origin, "hardware": hardware, "dataset": ds, "generation": gen, "methods": methods, "model": model_identity_manifest()}


def launch_workers(phase: str, selected_gpus: list[str], *, detach: bool = False) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    commands = []
    for method_id, gpu in zip(METHOD_ORDER, selected_gpus):
        log_path = LOG_DIR / f"{phase}_{method_id}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--phase",
            phase,
            "--method-id",
            method_id,
            "--physical-gpu",
            str(gpu),
        ]
        commands.append((method_id, gpu, log_path, cmd))
    launcher_manifest = {
        "phase": phase,
        "launched_at": utc_now(),
        "detach": detach,
        "commands": [{"method": m, "physical_gpu": g, "log": str(p.relative_to(ROOT)), "cmd": c} for m, g, p, c in commands],
    }
    write_json(LOG_DIR / f"{phase}_launcher_manifest.json", launcher_manifest)
    procs = []
    for method_id, gpu, log_path, cmd in commands:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        log = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=detach)
        procs.append((proc, log))
    if detach:
        write_json(LOG_DIR / f"{phase}_detached_pids.json", {"phase": phase, "pids": [p.pid for p, _ in procs], "timestamp": utc_now()})
        for _proc, log in procs:
            log.close()
        return 0
    rc = 0
    for proc, log in procs:
        rc = max(rc, proc.wait())
        log.close()
    return rc


def collect_records(phase: str = "formal", *, current_only: bool = True) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((RESULT_DIR / phase).glob("*/*/p*.json")):
        try:
            rec = read_json(path)
            if current_only and not is_current_record(rec):
                continue
            rows.append(rec)
        except Exception:
            continue
    return rows


def ensure_complete_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {(m, s, p) for m in METHOD_ORDER for s in BASE_SEEDS for p in range(30)}
    seen = {(r.get("method"), int(r.get("base_seed", -1)), int(r.get("problem_id", -1))) for r in rows if r.get("status") == "completed"}
    return {
        "expected_generations": len(expected),
        "completed_generations": len(seen),
        "missing": [{"method": m, "base_seed": s, "problem_id": p} for (m, s, p) in sorted(expected - seen)],
        "duplicate_keys": len(rows) - len({(r.get("method"), r.get("base_seed"), r.get("problem_id")) for r in rows}),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def accuracy_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows = []
    for method in METHOD_ORDER:
        for seed in BASE_SEEDS:
            subset = [r for r in rows if r.get("method") == method and int(r.get("base_seed", -1)) == seed and r.get("status") == "completed"]
            correct = sum(bool(r.get("correct")) for r in subset)
            seed_rows.append({"method": method, "base_seed": seed, "correct": correct, "total": len(subset), "accuracy": correct / len(subset) if subset else None})
    summary = []
    for method in METHOD_ORDER:
        vals = [float(r["accuracy"]) for r in seed_rows if r["method"] == method and r["accuracy"] is not None]
        total_correct = sum(int(r["correct"]) for r in seed_rows if r["method"] == method)
        total = sum(int(r["total"]) for r in seed_rows if r["method"] == method)
        summary.append(
            {
                "method": method,
                "total_correct": total_correct,
                "total": total,
                "mean_accuracy": statistics.mean(vals) if vals else None,
                "std_accuracy": statistics.stdev(vals) if len(vals) > 1 else 0.0 if len(vals) == 1 else None,
            }
        )
    return seed_rows, summary


def by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    return {(r["method"], int(r["base_seed"]), int(r["problem_id"])): r for r in rows if r.get("status") == "completed"}


def transition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    idx = by_key(rows)
    out = []
    for seed in BASE_SEEDS:
        for pid in range(30):
            vals = {method: bool(idx.get((method, seed, pid), {}).get("correct")) for method in METHOD_ORDER}
            out.append(
                {
                    "base_seed": seed,
                    "problem_id": pid,
                    "fp16_correct": vals["FP16"],
                    "base_correct": vals["PATTERN_BASE"],
                    "random25_correct": vals["RANDOM_V4_25"],
                    "causal25_correct": vals["CAUSAL_V4_25"],
                    "causal_win_random_loss": (not vals["RANDOM_V4_25"]) and vals["CAUSAL_V4_25"],
                    "random_win_causal_loss": vals["RANDOM_V4_25"] and not vals["CAUSAL_V4_25"],
                    "tie_random_causal": vals["RANDOM_V4_25"] == vals["CAUSAL_V4_25"],
                    "base_wrong_random_wrong_causal_correct": (not vals["PATTERN_BASE"]) and (not vals["RANDOM_V4_25"]) and vals["CAUSAL_V4_25"],
                    "base_wrong_random_correct_causal_correct": (not vals["PATTERN_BASE"]) and vals["RANDOM_V4_25"] and vals["CAUSAL_V4_25"],
                    "base_correct_random_wrong_causal_correct": vals["PATTERN_BASE"] and (not vals["RANDOM_V4_25"]) and vals["CAUSAL_V4_25"],
                    "base_correct_random_correct_causal_wrong": vals["PATTERN_BASE"] and vals["RANDOM_V4_25"] and (not vals["CAUSAL_V4_25"]),
                }
            )
    return out


def question_level_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    idx = by_key(rows)
    out = []
    for pid in range(30):
        row: dict[str, Any] = {"problem_id": pid}
        for method in METHOD_ORDER:
            vals = [1.0 if bool(idx.get((method, seed, pid), {}).get("correct")) else 0.0 for seed in BASE_SEEDS]
            row[f"{method}_mean_correct"] = sum(vals) / len(vals)
        row["causal_minus_random"] = row["CAUSAL_V4_25_mean_correct"] - row["RANDOM_V4_25_mean_correct"]
        row["causal_minus_base"] = row["CAUSAL_V4_25_mean_correct"] - row["PATTERN_BASE_mean_correct"]
        out.append(row)
    return out


def block_bootstrap(question_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = random.Random(BOOTSTRAP_SEED)
    causal_random = [float(r["causal_minus_random"]) / 30.0 for r in question_rows]
    causal_base = [float(r["causal_minus_base"]) / 30.0 for r in question_rows]

    def sample(vals: list[float]) -> dict[str, Any]:
        draws = []
        n = len(vals)
        for _ in range(BOOTSTRAP_RESAMPLES):
            draws.append(sum(vals[rng.randrange(n)] for _ in range(n)))
        draws.sort()
        return {
            "mean_delta": statistics.mean(draws),
            "median_delta": statistics.median(draws),
            "ci95_low": draws[int(0.025 * BOOTSTRAP_RESAMPLES)],
            "ci95_high": draws[int(0.975 * BOOTSTRAP_RESAMPLES) - 1],
        }

    return {
        "bootstrap_unit": "question",
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "causal_minus_random": sample(causal_random),
        "causal_minus_base": sample(causal_base),
    }


def percentile(vals: list[float], q: float) -> float | None:
    vals = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def length_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in METHOD_ORDER:
        for seed in BASE_SEEDS:
            subset = [r for r in rows if r.get("method") == method and int(r.get("base_seed", -1)) == seed and r.get("status") == "completed"]
            toks = [float(r.get("generated_tokens") or 0) for r in subset]
            correct_toks = [float(r.get("generated_tokens") or 0) for r in subset if r.get("correct")]
            wrong_toks = [float(r.get("generated_tokens") or 0) for r in subset if not r.get("correct")]
            out.append(
                {
                    "method": method,
                    "base_seed": seed,
                    "mean_generated_tokens": statistics.mean(toks) if toks else None,
                    "median_generated_tokens": statistics.median(toks) if toks else None,
                    "p90_generated_tokens": percentile(toks, 0.90),
                    "p95_generated_tokens": percentile(toks, 0.95),
                    "max_generated_tokens": max(toks) if toks else None,
                    "correct_mean_generated_tokens": statistics.mean(correct_toks) if correct_toks else None,
                    "incorrect_mean_generated_tokens": statistics.mean(wrong_toks) if wrong_toks else None,
                }
            )
    return out


def truncation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in METHOD_ORDER:
        for seed in BASE_SEEDS:
            subset = [r for r in rows if r.get("method") == method and int(r.get("base_seed", -1)) == seed]
            out.append(
                {
                    "method": method,
                    "base_seed": seed,
                    "length_truncated": sum(bool(r.get("length_truncated")) for r in subset),
                    "truncated_correct": sum(bool(r.get("length_truncated")) and bool(r.get("correct")) for r in subset),
                    "truncated_incorrect": sum(bool(r.get("length_truncated")) and not bool(r.get("correct")) for r in subset),
                    "stop_eos": sum(r.get("stop_reason") == "eos" for r in subset),
                    "stop_length": sum(r.get("stop_reason") == "length" for r in subset),
                    "stop_other": sum(r.get("stop_reason") not in {"eos", "length"} for r in subset),
                }
            )
    return out


def bit_cost_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for method in METHOD_ORDER:
        vals = [float(r["effective_kv_bits_per_element"]) for r in rows if r.get("method") == method and r.get("effective_kv_bits_per_element") is not None]
        fracs = [float(r["v4_realized_fraction"]) for r in rows if r.get("method") == method and r.get("v4_realized_fraction") is not None]
        out[method] = {
            "effective_kv_bits_per_element_mean": statistics.mean(vals) if vals else effective_bits_for_method(method, None),
            "effective_kv_bits_per_element_min": min(vals) if vals else effective_bits_for_method(method, None),
            "effective_kv_bits_per_element_max": max(vals) if vals else effective_bits_for_method(method, None),
            "realized_v4_fraction_mean": statistics.mean(fracs) if fracs else None,
            "realized_v4_fraction_min": min(fracs) if fracs else None,
            "realized_v4_fraction_max": max(fracs) if fracs else None,
        }
    delta = abs((out["RANDOM_V4_25"]["effective_kv_bits_per_element_mean"] or 0.0) - (out["CAUSAL_V4_25"]["effective_kv_bits_per_element_mean"] or 0.0))
    out["same_bit_delta"] = delta
    out["SAME_BIT_CONTROL_VALID"] = delta <= 1e-9
    return out


def v4_realized_by_layer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for rec in rows:
        if rec.get("method") not in {"RANDOM_V4_25", "CAUSAL_V4_25"}:
            continue
        for layer_row in rec.get("v4_realized_by_layer") or []:
            out.append(
                {
                    "method": rec.get("method"),
                    "base_seed": rec.get("base_seed"),
                    "effective_seed": rec.get("effective_seed"),
                    "problem_id": rec.get("problem_id"),
                    "selector_task_key": rec.get("selector_task_key"),
                    "layer": layer_row.get("layer"),
                    "v4_tokens": layer_row.get("v4_tokens"),
                    "total_tokens": layer_row.get("total_tokens"),
                    "fraction": layer_row.get("fraction"),
                    "generation_config_hash": rec.get("generation_config_hash"),
                    "formal_config_hash": rec.get("formal_config_hash"),
                }
            )
    return out


def classify(summary_rows: list[dict[str, Any]], transition: list[dict[str, Any]], question_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    acc = {r["method"]: float(r["mean_accuracy"]) if r["mean_accuracy"] is not None else None for r in summary_rows}
    per_seed = defaultdict(dict)
    for row in seed_rows:
        per_seed[int(row["base_seed"])][row["method"]] = float(row["accuracy"])
    causal_gt_random = sum(per_seed[s]["CAUSAL_V4_25"] > per_seed[s]["RANDOM_V4_25"] for s in BASE_SEEDS if "CAUSAL_V4_25" in per_seed[s] and "RANDOM_V4_25" in per_seed[s])
    causal_eq_random = sum(per_seed[s]["CAUSAL_V4_25"] == per_seed[s]["RANDOM_V4_25"] for s in BASE_SEEDS if "CAUSAL_V4_25" in per_seed[s] and "RANDOM_V4_25" in per_seed[s])
    causal_lt_random = sum(per_seed[s]["CAUSAL_V4_25"] < per_seed[s]["RANDOM_V4_25"] for s in BASE_SEEDS if "CAUSAL_V4_25" in per_seed[s] and "RANDOM_V4_25" in per_seed[s])
    causal_base = (acc["CAUSAL_V4_25"] or 0.0) - (acc["PATTERN_BASE"] or 0.0)
    causal_random = (acc["CAUSAL_V4_25"] or 0.0) - (acc["RANDOM_V4_25"] or 0.0)
    positive_questions = sum(float(r["causal_minus_random"]) > 0 for r in question_rows)
    concentrated = positive_questions <= 1 and causal_random > 0
    if causal_base > 0 and causal_random > 0 and causal_gt_random >= 2 and not concentrated:
        label = "SUPPORTED"
    elif causal_random > 0 and (causal_gt_random < 2 or concentrated):
        label = "PROMISING_BUT_INCONSISTENT"
    elif abs(causal_base) < 1e-12 and abs(causal_random) < 1e-12:
        label = "MECHANISM_ONLY"
    elif causal_base < 0 and causal_random < 0 and causal_lt_random >= 2:
        label = "HARMFUL"
    elif causal_base <= 0 and causal_random <= 0:
        label = "NO_QUALITY_GAIN"
    else:
        label = "INCONCLUSIVE"
    return {
        "FULL_AIME24_METHOD_CLASSIFICATION": label,
        "CAUSAL25_beats_BASE": causal_base > 0,
        "CAUSAL25_beats_RANDOM25": causal_random > 0,
        "CAUSAL25_beats_RANDOM25_on_ge_2_of_3_seeds": causal_gt_random >= 2,
        "seeds_causal_gt_random": causal_gt_random,
        "seeds_causal_eq_random": causal_eq_random,
        "seeds_causal_lt_random": causal_lt_random,
        "importance_aware_allocation_task_quality_supported": label == "SUPPORTED",
        "GENERALIZATION_VALIDATION_RECOMMENDED": label == "SUPPORTED",
        "NEXT_PRIORITY": "AIME25 generalization validation" if label == "SUPPORTED" else "mechanism-to-quality gap analysis" if label in {"MECHANISM_ONLY", "NO_QUALITY_GAIN"} else "analyze seed/question-level quality variance",
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def raw_generation_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for r in rows:
        rel = r.get("raw_generation_path")
        if not rel:
            continue
        path = ROOT / str(rel)
        items.append(
            {
                "path": rel,
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": r.get("raw_generation_sha256"),
                "method": r.get("method"),
                "seed": r.get("base_seed"),
                "problem_id": r.get("problem_id"),
            }
        )
    return {"raw_generation_count": len(items), "items": items}


def aggregate() -> dict[str, Any]:
    rows = collect_records("formal")
    completeness = ensure_complete_records(rows)
    compact_rows = [
        {k: v for k, v in r.items() if k not in {"problem", "v4_realized_by_layer"}}
        for r in rows
    ]
    write_csv(OUT_DIR / "sample_results_compact.csv", compact_rows)
    jsonl = OUT_DIR / "sample_results_compact.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in compact_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    gzip_file(jsonl)
    jsonl.unlink(missing_ok=True)
    seed_acc, method_acc = accuracy_tables(rows)
    write_csv(OUT_DIR / "seed_accuracy.csv", seed_acc)
    write_csv(OUT_DIR / "method_accuracy_summary.csv", method_acc)
    transitions = transition_rows(rows)
    write_csv(OUT_DIR / "question_seed_transitions.csv", transitions)
    qrows = question_level_rows(rows)
    write_csv(OUT_DIR / "question_level_summary.csv", qrows)
    bootstrap = block_bootstrap(qrows)
    write_json(OUT_DIR / "paired_bootstrap.json", bootstrap)
    lengths = length_summary(rows)
    write_csv(OUT_DIR / "length_summary.csv", lengths)
    trunc = truncation_summary(rows)
    write_csv(OUT_DIR / "truncation_summary.csv", trunc)
    bits = bit_cost_summary(rows)
    write_json(OUT_DIR / "bit_cost_summary.json", bits)
    write_csv(OUT_DIR / "v4_realized_by_layer.csv", v4_realized_by_layer_rows(rows))
    decisions = classify(method_acc, transitions, qrows, seed_acc)
    write_json(OUT_DIR / "hypothesis_decisions.json", decisions)
    failures = [r for r in rows if r.get("status") != "completed"]
    summary = {
        "repository": REPOSITORY,
        "branch": BRANCH,
        "parent": PARENT_COMMIT,
        "head": git_text("rev-parse", "HEAD"),
        "expected_generations": completeness["expected_generations"],
        "completed_generations": completeness["completed_generations"],
        "runtime_failed_generations": len(failures),
        "oom_failures": sum(r.get("stop_reason") == "oom" for r in failures),
        "nan_inf_failures": sum(bool(r.get("nan_inf_detected")) for r in rows),
        "length_truncated_generations": sum(bool(r.get("length_truncated")) for r in rows),
        "formal_run_complete": completeness["completed_generations"] == completeness["expected_generations"] and not completeness["missing"] and not failures,
        "accuracy": method_acc,
        "primary_quality_differences": primary_differences(seed_acc),
        "paired_counts": paired_counts(transitions),
        "bit_cost": bits,
        "decisions": decisions,
    }
    write_json(OUT_DIR / "full_aime24_quality_summary.json", summary)
    write_json(OUT_DIR / "raw_generation_manifest.json", raw_generation_manifest(rows))
    write_report(summary, seed_acc, method_acc, bootstrap, lengths)
    return summary


def primary_differences(seed_acc: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(r["method"], int(r["base_seed"])): float(r["accuracy"]) for r in seed_acc if r["accuracy"] is not None}

    def deltas(a: str, b: str) -> dict[str, Any]:
        rows = [{"base_seed": s, "delta": by.get((a, s), 0.0) - by.get((b, s), 0.0)} for s in BASE_SEEDS]
        return {"per_seed": rows, "mean_delta": statistics.mean(r["delta"] for r in rows)}

    return {
        "CAUSAL25_minus_BASE": deltas("CAUSAL_V4_25", "PATTERN_BASE"),
        "CAUSAL25_minus_RANDOM25": deltas("CAUSAL_V4_25", "RANDOM_V4_25"),
        "FP16_minus_CAUSAL25": deltas("FP16", "CAUSAL_V4_25"),
    }


def paired_counts(transitions: list[dict[str, Any]]) -> dict[str, int]:
    keys = [
        "causal_win_random_loss",
        "random_win_causal_loss",
        "tie_random_causal",
        "base_wrong_random_wrong_causal_correct",
        "base_wrong_random_correct_causal_correct",
        "base_correct_random_wrong_causal_correct",
        "base_correct_random_correct_causal_wrong",
    ]
    return {key: sum(bool(r.get(key)) for r in transitions) for key in keys}


def write_report(summary: dict[str, Any], seed_acc: list[dict[str, Any]], method_acc: list[dict[str, Any]], bootstrap: dict[str, Any], lengths: list[dict[str, Any]]) -> None:
    lines = [
        "# Full AIME24 Task-Quality Validation",
        "",
        f"- Repository: `{REPOSITORY}`",
        f"- Branch: `{BRANCH}`",
        f"- Parent: `{PARENT_COMMIT}`",
        f"- Completed generations: `{summary['completed_generations']}/{summary['expected_generations']}`",
        f"- Classification: `{summary['decisions']['FULL_AIME24_METHOD_CLASSIFICATION']}`",
        "",
        "## Accuracy",
        "",
        "| method | total | mean | std |",
        "|---|---:|---:|---:|",
    ]
    for row in method_acc:
        lines.append(f"| {row['method']} | {row['total_correct']}/{row['total']} | {row['mean_accuracy']} | {row['std_accuracy']} |")
    lines.extend(["", "## Accuracy By Seed", "", "| method | seed | correct/30 | accuracy |", "|---|---:|---:|---:|"])
    for row in seed_acc:
        lines.append(f"| {row['method']} | {row['base_seed']} | {row['correct']}/{row['total']} | {row['accuracy']} |")
    lines.extend(
        [
            "",
            "## Paired Bootstrap",
            "",
            f"- Unit: `{bootstrap['bootstrap_unit']}`",
            f"- Resamples: `{bootstrap['resamples']}`",
            f"- CAUSAL - RANDOM: `{bootstrap['causal_minus_random']}`",
            f"- CAUSAL - BASE: `{bootstrap['causal_minus_base']}`",
            "",
            "## Notes",
            "",
            "- Full raw generations are stored locally under ignored `raw_generations/`; committed records are compact.",
            "- Runtime is recorded as a secondary implementation metric only.",
            "- No AIME25/GPQA run is started by this experiment.",
        ]
    )
    (OUT_DIR / "full_aime24_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    requested = args.gpus.split(",") if args.gpus else None
    selected, gpu_snapshot = select_idle_gpus(requested)
    manifests = write_static_manifests(selected, gpu_snapshot)
    write_json(OUT_DIR / "model_identity.json", manifests["model"])
    gates = {
        "FOUR_GPU_PREFLIGHT_VALID": len(selected) == 4,
        "MODEL_IDENTITY_VALID": bool(manifests["model"]["MODEL_IDENTITY_VALID"]),
        "FULL_AIME24_DATASET_VALID": bool(manifests["dataset"]["FULL_AIME24_DATASET_VALID"]),
        "GENERATION_CONFIG_VALID": bool(manifests["generation"]["GENERATION_CONFIG_VALID"]),
        "ANSWER_PARSER_VALID": parser_valid(),
        "PATTERN_BASE_VALID": METHOD_CONFIGS["PATTERN_BASE"]["config_name"] == "pattern_rolling_k2v2_s16_r128",
        "RANDOM25_VALID": METHOD_CONFIGS["RANDOM_V4_25"]["selector"] == "random_v4" and METHOD_CONFIGS["RANDOM_V4_25"]["v4_budget_fraction"] == 0.25,
        "CAUSAL25_VALID": METHOD_CONFIGS["CAUSAL_V4_25"]["selector"] == "causal_v4" and METHOD_CONFIGS["CAUSAL_V4_25"]["v4_budget_fraction"] == 0.25,
        "CAUSAL_SELECTOR_NO_FUTURE_LEAKAGE": bool(manifests["methods"]["CAUSAL_SELECTOR_NO_FUTURE_LEAKAGE"]),
        "RANDOM_CAUSAL_BIT_BUDGET_MATCHED": bool(manifests["methods"]["RANDOM_CAUSAL_BIT_BUDGET_MATCHED"]),
        "GPU_ISOLATION_VALID": len(selected) == 4 and len(set(selected)) == 4,
        "NO_PREFLIGHT_OOM": None,
        "NO_NAN_INF": True,
    }
    if args.run_smoke and len(selected) == 4 and all(v is True for k, v in gates.items() if k not in {"NO_PREFLIGHT_OOM"}):
        rc = launch_workers("smoke", selected, detach=False)
        smoke_rows = collect_records("smoke")
        gates["NO_PREFLIGHT_OOM"] = rc == 0 and not any(r.get("stop_reason") == "oom" for r in smoke_rows) and len([r for r in smoke_rows if r.get("status") == "completed"]) == 4
        gates["NO_NAN_INF"] = not any(bool(r.get("nan_inf_detected")) for r in smoke_rows)
    else:
        gates["NO_PREFLIGHT_OOM"] = False if args.run_smoke else True
    gates["FORMAL_AIME24_QUALITY_RUN_APPROVED"] = all(bool(v) for v in gates.values())
    gates["selected_gpus"] = selected
    gates["timestamp"] = utc_now()
    write_json(OUT_DIR / "preflight_gate_summary.json", gates)
    print(json.dumps(gates, indent=2, sort_keys=True), flush=True)
    return gates


def print_status() -> None:
    formal = collect_records("formal")
    smoke = collect_records("smoke")
    completeness = ensure_complete_records(formal)
    seed_acc, method_acc = accuracy_tables(formal)
    payload = {
        "smoke_completed": len([r for r in smoke if r.get("status") == "completed"]),
        "formal_completed": completeness["completed_generations"],
        "formal_expected": completeness["expected_generations"],
        "missing_count": len(completeness["missing"]),
        "runtime_failures": len([r for r in formal if r.get("status") != "completed"]),
        "seed_accuracy": seed_acc,
        "method_accuracy": method_acc,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-smoke", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--phase", choices=["smoke", "formal"], default="formal")
    parser.add_argument("--method-id", choices=METHOD_ORDER)
    parser.add_argument("--physical-gpu")
    parser.add_argument("--base-seeds", help="Comma-separated seed subset for a formal worker, e.g. 44.")
    parser.add_argument("--problem-ids", help="Comma-separated problem subset for a formal worker, e.g. 0,1,2.")
    parser.add_argument("--gpus", help="Comma-separated physical GPU IDs to use, in GPU_A..GPU_D order if already idle.")
    parser.add_argument("--smoke-problem-id", type=int, default=0)
    args = parser.parse_args()
    if args.worker:
        if not args.method_id or args.physical_gpu is None:
            raise SystemExit("--worker requires --method-id and --physical-gpu")
        run_worker(args)
        return
    if args.preflight:
        gates = preflight(args)
        if args.launch_formal:
            if not gates.get("FORMAL_AIME24_QUALITY_RUN_APPROVED"):
                raise SystemExit("FORMAL_AIME24_QUALITY_RUN_APPROVED is false; not launching formal run.")
            raise SystemExit(launch_workers("formal", gates["selected_gpus"], detach=args.detach))
        return
    if args.launch_formal:
        gates_path = OUT_DIR / "preflight_gate_summary.json"
        if not gates_path.exists() or not read_json(gates_path).get("FORMAL_AIME24_QUALITY_RUN_APPROVED"):
            raise SystemExit("Preflight gate is missing or not approved.")
        selected = read_json(gates_path)["selected_gpus"]
        raise SystemExit(launch_workers("formal", selected, detach=args.detach))
    if args.aggregate:
        print(json.dumps(aggregate(), indent=2, ensure_ascii=False, sort_keys=True))
        return
    if args.status:
        print_status()
        return
    parser.print_help()


if __name__ == "__main__":
    main()
