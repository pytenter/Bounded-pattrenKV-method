#!/usr/bin/env python
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench._longbench_scorer import score_example, score_subtask
from bench.bench_longbench_patternkv import (
    SKIP_CHAT,
    build_prompt,
    load_model,
    load_task,
    patternkv_evidence,
    sample_id,
)
from bench.longbench_config import LONGBENCH_PIN, MAX_NEW_TOKENS, METRIC_NAMES, PROMPT_TEMPLATES, SUBTASKS, expected_samples
from bench.paper_config import apply_method_defaults, cache_storage_summary, method_config_dict
from models.llama_patternkv import reset_patternkv_runtime_state

EXPERIMENT_NAME = "longbench_paper_v2_8k_single4090"
EXPERIMENT_ID = "paper_v2_8k_single4090"
BASELINE_METHODS = ("fp16", "kivi_paper_g128", "patternkv_paper")
METHODS = BASELINE_METHODS + ("causal_v4_25",)
CONFIG_PATH = ROOT / "configs/longbench_paper_v2_8k_single4090.yaml"
PROMPT_PATH = ROOT / "bench/longbench_config/dataset2prompt.json"
MAXLEN_PATH = ROOT / "bench/longbench_config/dataset2maxlen.json"
SCORER_PATH = ROOT / "bench/_longbench_scorer.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def gpu_info() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        name, uuid = [x.strip() for x in out.split(",", 1)]
    except Exception:
        name, uuid = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), None
    return {"gpu_name": name, "gpu_uuid": uuid}


def config_hash(max_input_length: int) -> str:
    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "max_input_length": max_input_length,
        "methods": BASELINE_METHODS,
        "tasks": SUBTASKS,
        "prompt_sha256": sha256_file(PROMPT_PATH),
        "maxlen_sha256": sha256_file(MAXLEN_PATH),
        "scorer_sha256": sha256_file(SCORER_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl_safely(path: Path) -> tuple[list[dict], int]:
    rows: list[dict] = []
    bad = 0
    if not path.exists():
        return rows, bad
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def atomic_append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows, bad = read_jsonl_safely(path)
        if bad:
            raise RuntimeError(f"{path} has {bad} damaged JSON line(s); refusing to rewrite it")
        rows.append(record)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def row_is_complete(row: dict, cfg_hash: str) -> bool:
    if row.get("config_hash") != cfg_hash:
        return False
    if row.get("stop_reason") in ("oom", "error"):
        return False
    required = ("experiment_id", "method", "task", "sample_id", "prediction", "metric_name")
    return all(row.get(key) is not None for key in required)


def output_path(output_dir: Path, method: str, task: str) -> Path:
    return output_dir / method / f"{task}.jsonl"


def done_sample_ids(path: Path, cfg_hash: str) -> set[str]:
    rows, bad = read_jsonl_safely(path)
    if bad:
        return set()
    return {str(row.get("sample_id")) for row in rows if row_is_complete(row, cfg_hash)}


def truncate_to_limit(prompt: str, tokenizer, limit: int, add_special_tokens: bool) -> tuple[str, int]:
    toks = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if len(toks) <= limit:
        return prompt, len(toks)
    half = limit // 2
    kept = toks[:half] + toks[-(limit - half):]
    return tokenizer.decode(kept, skip_special_tokens=True), len(kept)


def encode_prompt(ex: dict, tokenizer, task: str, max_input_length: int) -> tuple[dict, Any, Any]:
    prompt, prompt_stats = build_prompt(ex, tokenizer, task, max_input_length, True)
    add_special = task in SKIP_CHAT
    prompt, final_len = truncate_to_limit(prompt, tokenizer, max_input_length, add_special)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special)
    if int(encoded.input_ids.shape[1]) > max_input_length:
        prompt, final_len = truncate_to_limit(prompt, tokenizer, max_input_length - 1, add_special)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special)
    info = {
        "prompt": prompt,
        "raw_input_tokens": prompt_stats["raw_prompt_tokens"],
        "truncated_input_tokens": prompt_stats["truncated_chat_tokens"] or prompt_stats["truncated_prompt_tokens"],
        "was_truncated": bool(prompt_stats["prompt_truncated_to_max_input"] or int(encoded.input_ids.shape[1]) < prompt_stats["raw_prompt_tokens"]),
        "input_tokens_after_special_tokens": int(encoded.input_ids.shape[1]),
    }
    return info, encoded.input_ids.to("cuda:0"), encoded.attention_mask.to("cuda:0")


def generation_stop(output_ids: list[int], max_new_tokens: int, eos_id: int | None) -> tuple[str, bool]:
    if output_ids and eos_id is not None and output_ids[-1] == eos_id:
        return "eos", False
    if len(output_ids) >= max_new_tokens:
        return "length", True
    return "unknown", False


def method_args(method: str, model_path: str, max_input_length: int, output_dir: Path, status_dir: Path, data_dir: Path) -> Namespace:
    args = Namespace(
        method=method,
        model_path=model_path,
        data_dir=str(data_dir),
        output_dir=output_dir,
        status_dir=status_dir,
        mode=EXPERIMENT_ID,
        gpu_id="0",
        max_input_length=max_input_length,
        dtype="float16",
        attn_implementation="flash_attention_2",
        seed=0,
        k_bits=2,
        v_bits=2,
        group_size=128,
        residual_length=128,
        num_k_base=32,
        num_v_base=32,
        kvtuner_flex_root="/data/zypan/Block-kvcache-experiment/third_party/kvtuner/flexible_quant",
        skip_existing=True,
        instruct_model=True,
        tasks=list(SUBTASKS),
        num_samples=0,
    )
    args.paper_method_config = apply_method_defaults(args)
    return args


def set_selector_task_context(model, selector_task_key: str) -> None:
    reset_patternkv_runtime_state(model)
    if hasattr(model, "config"):
        model.config.patternkv_selector_task_key = selector_task_key
    for layer in getattr(getattr(model, "model", None), "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn.selector_task_key = selector_task_key
        attn.v_causal_importance = None
        attn.v_oracle_importance = None


@torch.no_grad()
def run_one(model, tokenizer, args: Namespace, task: str, index: int, ex: dict, dataset_size: int, hashes: dict, cfg_hash: str, gpu: dict) -> dict:
    sid = sample_id(task, index, ex)
    if getattr(args, "patternkv_v_precision_selector", None) == "causal_v4":
        set_selector_task_context(model, sid)
    t0 = time.perf_counter()
    last_op = "encode_prompt"
    prompt_info = {}
    try:
        prompt_info, input_ids, attention_mask = encode_prompt(ex, tokenizer, task, args.max_input_length)
        input_token_ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]
        input_token_ids_sha256 = sha256_text(json.dumps(input_token_ids, separators=(",", ":")))
        if prompt_info["input_tokens_after_special_tokens"] > args.max_input_length:
            raise RuntimeError(f"encoded prompt exceeds max_input_length: {prompt_info['input_tokens_after_special_tokens']} > {args.max_input_length}")
        last_op = "generate"
        torch.cuda.reset_peak_memory_stats()
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS[task],
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        torch.cuda.synchronize()
        seq = output.sequences
        output_ids = [int(x) for x in seq[0, input_ids.shape[1]:].detach().cpu().tolist()]
        generated_token_ids_sha256 = sha256_text(json.dumps(output_ids, separators=(",", ":")))
        generated = tokenizer.decode(seq[0, input_ids.shape[1]:], skip_special_tokens=True)
        stop_reason, hit_max = generation_stop(output_ids, MAX_NEW_TOKENS[task], tokenizer.eos_token_id)
        refs = list(ex.get("answers") or [])
        all_classes = list(ex.get("all_classes") or [])
        score = score_example(task, generated, refs, all_classes=all_classes or None)
        cache_stats = cache_storage_summary(
            args.method,
            getattr(output, "past_key_values", None),
            model=model,
            total_cached_tokens=int(seq.shape[1]),
            residual_length=args.residual_length,
        )
        runtime_evidence = {}
        if args.method in ("patternkv_paper", "causal_v4_25"):
            runtime_evidence["patternkv_runtime_evidence"] = patternkv_evidence(model, getattr(output, "past_key_values", None))
        if args.method == "kivi_paper_g128":
            runtime_evidence["kivi_runtime_evidence"] = {
                "method": args.method,
                "model_class": "LlamaForCausalLM_KIVI",
                "k_bits": args.k_bits,
                "v_bits": args.v_bits,
                "group_size": args.group_size,
                "residual_length": args.residual_length,
                "persistent_key_heads": cache_stats.get("persistent_key_heads"),
                "persistent_value_heads": cache_stats.get("persistent_value_heads"),
            }
        rec = base_record(args, task, sid, index, dataset_size, hashes, cfg_hash, gpu, prompt_info, refs)
        rec.update(
            {
                "generated_text": generated,
                "input_token_ids_sha256": input_token_ids_sha256,
                "generated_token_ids": output_ids,
                "generated_token_ids_sha256": generated_token_ids_sha256,
                "generated_tokens": len(output_ids),
                "prediction": generated,
                "score": score,
                "stop_reason": stop_reason,
                "hit_max_new_tokens": hit_max,
                "wall_time_seconds": round(time.perf_counter() - t0, 4),
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "cache_bitwidth_stats": enrich_bitwidth(args.method, cache_stats, int(torch.cuda.max_memory_allocated())),
                "exception_type": None,
                "exception_message": None,
                "oom_stage": None,
                "last_successful_operation": "generate",
                **runtime_evidence,
            }
        )
        del output, seq, input_ids, attention_mask
        return rec
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        rec = base_record(args, task, sid, index, dataset_size, hashes, cfg_hash, gpu, prompt_info, list(ex.get("answers") or []))
        rec.update(error_fields("oom", exc, last_op, t0))
        return rec
    except Exception as exc:
        rec = base_record(args, task, sid, index, dataset_size, hashes, cfg_hash, gpu, prompt_info, list(ex.get("answers") or []))
        rec.update(error_fields("error", exc, last_op, t0))
        return rec


def base_record(args, task: str, sid: str, index: int, dataset_size: int, hashes: dict, cfg_hash: str, gpu: dict, prompt_info: dict, refs: list) -> dict:
    method_cfg = method_config_dict(args)
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "config_hash": cfg_hash,
        "git_commit": hashes["git_commit"],
        "model_path": args.model_path,
        "model_config_sha256": hashes["model_config_sha256"],
        "tokenizer_config_sha256": hashes["tokenizer_config_sha256"],
        "gpu_name": gpu.get("gpu_name"),
        "gpu_uuid": gpu.get("gpu_uuid"),
        "method": args.method,
        "task": task,
        "sample_id": sid,
        "sample_index": index,
        "dataset_size": dataset_size,
        "prompt_template_hash": sha256_text(PROMPT_TEMPLATES[task]),
        "max_gen": MAX_NEW_TOKENS[task],
        "max_input_length": args.max_input_length,
        "raw_input_tokens": prompt_info.get("raw_input_tokens"),
        "truncated_input_tokens": prompt_info.get("truncated_input_tokens"),
        "was_truncated": prompt_info.get("was_truncated"),
        "input_tokens_after_special_tokens": prompt_info.get("input_tokens_after_special_tokens"),
        "generated_text": "",
        "generated_tokens": 0,
        "stop_reason": "unknown",
        "hit_max_new_tokens": False,
        "wall_time_seconds": None,
        "peak_memory_allocated_bytes": None,
        "peak_memory_reserved_bytes": None,
        "prediction": "",
        "reference": refs,
        "answers": refs,
        "all_classes": [],
        "metric_name": METRIC_NAMES[task],
        "metric": METRIC_NAMES[task],
        "quantization_config": method_cfg,
        "patternkv_config": method_cfg if args.method in ("patternkv_paper", "causal_v4_25") else None,
        "selector_task_key": sid if getattr(args, "patternkv_v_precision_selector", None) == "causal_v4" else None,
        "paper_config_snapshot": method_cfg,
        "cache_bitwidth_stats": None,
        "timestamp": utc_now(),
    }


def error_fields(reason: str, exc: Exception, last_op: str, t0: float) -> dict:
    return {
        "stop_reason": reason,
        "wall_time_seconds": round(time.perf_counter() - t0, 4),
        "exception_type": exc.__class__.__name__,
        "exception_message": str(exc),
        "oom_stage": last_op if reason == "oom" else None,
        "last_successful_operation": last_op,
        "error": repr(exc),
    }


def enrich_bitwidth(method: str, stats: dict, peak: int) -> dict:
    payload = 16 if method == "fp16" else 2
    affine = None if method == "fp16" else 2.25
    out = dict(stats or {})
    out.update(
        {
            "payload_bits": payload,
            "quantized_region_affine_bits": affine,
            "paper_theoretical_average_bits": None,
            "compact_storage_bits": None,
            "python_tensor_storage_bits": None,
            "actual_peak_gpu_memory": peak,
            "missing_bitwidth_reason": "Full average requires assignments, masks, centroids, residual window and task-specific cache lengths; bytes are recorded where available.",
        }
    )
    if stats and stats.get("python_tensor_storage_bytes") is not None:
        out["python_tensor_storage_bits"] = int(stats["python_tensor_storage_bytes"]) * 8
    return out


def plan(args) -> dict:
    data_dir = Path(args.data_dir)
    task_filter = parse_filter(args.task_filter)
    method_filter = parse_filter(args.method_filter)
    tasks = [t for t in SUBTASKS if not task_filter or t in task_filter]
    methods = [m for m in METHODS if not method_filter or m in method_filter]
    unknown_tasks = sorted(task_filter - set(SUBTASKS))
    unknown_methods = sorted(method_filter - set(METHODS))
    if unknown_tasks:
        raise SystemExit(f"Unknown LongBench task(s) in TASK_FILTER: {unknown_tasks}")
    if unknown_methods:
        raise SystemExit(f"Unknown method(s) in METHOD_FILTER: {unknown_methods}")
    counts = {}
    samples = {}
    for task in tasks:
        try:
            data = load_task(task, 0, data_dir)
            filtered = sample_indices(data, args.sample_filter)
        except PermissionError:
            filtered = list(range(expected_samples(task)))
        counts[task] = len(filtered)
        samples[task] = filtered
    return {
        "methods": methods,
        "tasks": tasks,
        "counts": counts,
        "samples": samples,
        "planned_per_method": sum(counts.values()),
        "planned_total": sum(counts.values()) * len(methods),
    }


def parse_filter(value: str | None) -> set[str]:
    if not value:
        return set()
    normalized = value.replace(",", " ")
    return {part.strip() for part in normalized.split() if part.strip()}


def sample_indices(data: list[dict], sample_filter: str | None) -> list[int]:
    if not sample_filter:
        return list(range(len(data)))
    selected = set()
    for part in sample_filter.replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            if start.strip().isdigit() and end.strip().isdigit():
                selected.update(range(int(start), int(end) + 1))
                continue
        if part.isdigit():
            selected.add(int(part))
    return [i for i in range(len(data)) if i in selected]


def write_status(path: Path, status: dict) -> None:
    write_json(path, {**status, "updated_at": utc_now()})


def run(args) -> None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices or "," in visible_devices:
        raise SystemExit("This runner requires exactly one visible CUDA device; the model uses cuda:0 inside the process")
    if args.max_input_length != 8192:
        raise SystemExit("MAX_INPUT_LENGTH must be 8192 for this 8K experiment")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg_hash = config_hash(args.max_input_length)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    status_dir = Path(args.status_dir)
    hashes = {
        "git_commit": git_commit(),
        "model_config_sha256": sha256_file(Path(args.model_path) / "config.json"),
        "tokenizer_config_sha256": sha256_file(Path(args.model_path) / "tokenizer_config.json"),
        "generation_config_sha256": sha256_file(Path(args.model_path) / "generation_config.json"),
        "prompt_sha256": sha256_file(PROMPT_PATH),
        "maxlen_sha256": sha256_file(MAXLEN_PATH),
        "scorer_sha256": sha256_file(SCORER_PATH),
    }
    p = plan(args)
    header = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "config_hash": cfg_hash,
        "max_input_length": args.max_input_length,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_count_used": 1,
        "model_path": args.model_path,
        "git_commit": hashes["git_commit"],
        "methods": p["methods"],
        "tasks": p["tasks"],
        "task_counts": p["counts"],
        "task_max_gen": {task: MAX_NEW_TOKENS[task] for task in p["tasks"]},
        "task_metrics": {task: METRIC_NAMES[task] for task in p["tasks"]},
        "planned_per_method": p["planned_per_method"],
        "planned_total": p["planned_total"],
        "output_dir": str(output_dir),
        "resume": args.resume,
        "retry_failed": args.retry_failed,
        "retry_oom": args.retry_oom,
        "oom_policy": "record and continue; no input-cap fallback",
        "hashes": hashes,
    }
    print(json.dumps(header, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    if args.dry_run:
        return
    gpu = gpu_info()
    for method in p["methods"]:
        phase_start = utc_now()
        margs = method_args(method, args.model_path, args.max_input_length, output_dir, status_dir, data_dir)
        print("[PaperConfigCheck] " + json.dumps(method_config_dict(margs), ensure_ascii=False, sort_keys=True), flush=True)
        model, tokenizer = load_model(margs)
        try:
            for task in p["tasks"]:
                data = load_task(task, 0, data_dir)
                task_out = output_path(output_dir, method, task)
                done = done_sample_ids(task_out, cfg_hash) if args.resume else set()
                for index in p["samples"][task]:
                    ex = data[index]
                    sid = sample_id(task, index, ex)
                    rows, _bad = read_jsonl_safely(task_out)
                    existing = [r for r in rows if r.get("sample_id") == sid and r.get("config_hash") == cfg_hash]
                    if sid in done and not (args.retry_failed or args.retry_oom):
                        continue
                    if existing:
                        last = existing[-1]
                        if last.get("stop_reason") == "oom" and not args.retry_oom:
                            continue
                        if last.get("stop_reason") == "error" and not args.retry_failed:
                            continue
                    status_path = status_dir / "runner.status.json"
                    write_status(
                        status_path,
                        {
                            "launcher_pid": os.getppid(),
                            "worker_pid": os.getpid(),
                            "phase": method,
                            "current_method": method,
                            "current_task": task,
                            "current_sample": sid,
                            "planned_total": p["planned_total"],
                            "output_dir": str(output_dir),
                            "phase_started_at": phase_start,
                        },
                    )
                    rec = run_one(model, tokenizer, margs, task, index, ex, len(data), hashes, cfg_hash, gpu)
                    atomic_append_jsonl(task_out, rec)
                    print(f"[{utc_now()}] wrote {method}/{task}/{sid} stop={rec.get('stop_reason')} tokens={rec.get('input_tokens_after_special_tokens')} out={task_out}", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[{utc_now()}] phase done method={method}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default="/root/Block-kvcache-experiment/data/LongBench")
    parser.add_argument("--output-dir", default="results/paper_repro_v2/longbench_full_8k_4090")
    parser.add_argument("--status-dir", default="run/paper_repro_v2/longbench_full_8k_4090")
    parser.add_argument("--max-input-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method-filter")
    parser.add_argument("--task-filter")
    parser.add_argument("--sample-filter")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-oom", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
