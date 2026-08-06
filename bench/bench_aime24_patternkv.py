import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.aime_answer_parser import normalize_aime_answer, parse_aime_answer
from bench.aime_utils import (
    DEFAULT_BASE_SEED,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    METHODS,
    build_manifest,
    config_hash,
    effective_seed,
    generation_config_dict,
    is_complete_result,
    load_aime24,
    result_path,
    search_model_candidates,
    set_all_seeds,
    shard_tasks,
    task_key,
    utc_now,
    write_json_atomic,
    compute_stop_state,
    normalize_eos_token_ids,
)
from bench.aime24_int2_wave1 import stable_hash, task_key3
from bench.paper_config import apply_method_defaults, cache_storage_summary, method_config_dict


def render_prompt(problem: str, tokenizer, force_think_prefix: bool) -> tuple[str, str, bool]:
    user_prompt = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    messages = [{"role": "user", "content": user_prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if force_think_prefix:
        rendered += "<think>\n"
    return rendered, user_prompt, True


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
        config.mixed_key_mask_path = str(args.mixed_key_mask_path or "")
        config.patternkv_cache_path = args.patternkv_cache_path
        config.use_flash = True
        config.num_k_base = args.num_k_base
        config.num_v_base = args.num_v_base
        model = LlamaForCausalLM_PatternKV.from_pretrained(args.model_path, local_files_only=True, config=config, torch_dtype=dtype, low_cpu_mem_usage=True).to("cuda:0")
    else:
        raise ValueError(f"Unsupported backend for AIME: {backend}")
    model.eval()
    return model, tokenizer


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


def validate_context(args, tokenizer, model, rows) -> dict[str, Any]:
    config = getattr(model, "config", None)
    max_pos = getattr(config, "max_position_embeddings", None) or getattr(config, "model_max_length", None)
    sample_prompt, _, _ = render_prompt(rows[0]["problem"], tokenizer, args.force_think_prefix)
    prompt_tokens = len(tokenizer(sample_prompt, add_special_tokens=False).input_ids)
    effective = prompt_tokens + args.max_new_tokens
    if max_pos and effective > int(max_pos):
        raise ValueError(f"model context too short: max_position_embeddings={max_pos}, prompt_tokens={prompt_tokens}, max_new_tokens={args.max_new_tokens}, effective={effective}")
    return {"model_max_position": max_pos, "sample_prompt_tokens": prompt_tokens, "max_new_tokens": args.max_new_tokens, "effective_max_sequence_length": effective}


@torch.no_grad()
def run_task(args, model, tokenizer, row: dict[str, Any], sample_id: int, cfg_hash: str, git_commit: str) -> dict[str, Any]:
    problem_id = int(row["problem_id"])
    seed = effective_seed(args.base_seed, problem_id, sample_id)
    set_all_seeds(seed)
    if args.method in {"patternkv_paper", "patternkv"}:
        from models.llama_patternkv import reset_patternkv_runtime_state

        reset_patternkv_runtime_state(model)
    rendered_prompt, user_prompt, chat_template_used = render_prompt(row["problem"], tokenizer, args.force_think_prefix)
    encoded = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to("cuda:0")
    attention_mask = encoded.attention_mask.to("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids(tokenizer, model),
        return_dict_in_generate=True,
        output_scores=False,
    )
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    seq = output.sequences
    generated_ids = seq[0, input_ids.shape[1] :].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    stop = compute_stop_state(generated_ids, args.max_new_tokens, eos_ids(tokenizer, model))
    parsed = parse_aime_answer(generated_text)
    ref = normalize_aime_answer(row["answer"])
    total_tokens = int(seq.shape[1])
    cache_stats = cache_storage_summary(args.method, getattr(output, "past_key_values", None), model=model, total_cached_tokens=total_tokens, residual_length=args.residual_length)
    patternkv_dynamic_stats = {}
    if args.method in {"patternkv_paper", "patternkv"}:
        from models.llama_patternkv import collect_patternkv_dynamic_stats

        patternkv_dynamic_stats = collect_patternkv_dynamic_stats(model, getattr(output, "past_key_values", None))
    cache_segment_stats = cache_stats.get("cache_segment_stats") or {
        "sink_tokens": 0,
        "packed_history_tokens": 0,
        "pending_history_tokens": 0,
        "recent_tokens": 0,
        "total_tokens": total_tokens,
        "k_assignment_tokens": None,
        "v_assignment_tokens": None,
    }
    prompt_hash = stable_hash({"rendered_prompt": rendered_prompt}, 32)
    input_token_hash = stable_hash({"input_ids": input_ids.detach().cpu().tolist()}, 32)
    generated_token_hash = stable_hash({"generated_ids": generated_ids}, 32)
    rec = {
        "experiment_id": args.experiment_id,
        "dataset": "aime24",
        "model_path": str(args.model_path),
        "model_name": Path(args.model_path).name,
        "method": args.method,
        "config_name": args.config_name or args.method,
        "problem_id": problem_id,
        "sample_id": sample_id,
        "task_key": task_key3(problem_id, sample_id, seed),
        "seed": seed,
        "base_seed": args.base_seed,
        "config_hash": cfg_hash,
        "problem": row["problem"],
        "raw_problem": row["problem"],
        "reference_answer": ref,
        "rendered_prompt": rendered_prompt,
        "prompt_hash": prompt_hash,
        "input_token_hash": input_token_hash,
        "prompt_protocol": "deepseek_r1_recommended",
        "chat_template_used": chat_template_used,
        "force_think_prefix": args.force_think_prefix,
        "input_tokens": int(input_ids.shape[1]),
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "generated_text": generated_text,
        "generated_token_ids": generated_ids,
        "generated_token_hash": generated_token_hash,
        "generated_tokens": len(generated_ids),
        "total_sequence_tokens": total_tokens,
        "parsed_answer": parsed["parsed_answer"],
        "parser_strategy": parsed["parser_strategy"],
        "parser_error": parsed["parser_error"],
        "boxed_candidates": parsed["boxed_candidates"],
        "is_correct": parsed["parsed_answer"] == ref,
        "wall_time_seconds": round(wall, 4),
        "prefill_time_seconds": None,
        "decode_time_seconds": None,
        "tokens_per_second": round(len(generated_ids) / wall, 4) if wall > 0 else None,
        "gpu_id": args.gpu_id,
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "quantization_config": method_config_dict(args),
        "patternkv_config": method_config_dict(args) if args.method == "patternkv_paper" else {},
        "patternkv_cache_path": args.patternkv_cache_path if args.method in {"patternkv_paper", "patternkv"} else None,
        "cache_bitwidth_stats": cache_stats,
        "patternkv_dynamic_stats": patternkv_dynamic_stats,
        "cache_segment_stats": cache_segment_stats,
        "git_commit": git_commit,
        "timestamp": utc_now(),
        "error": None,
        "sink_length": args.sink_length,
        "recent_length": args.recent_length,
        "k_bits": args.k_bits,
        "v_bits": args.v_bits,
        "mixed_precision_ratio": args.mixed_key_int4_ratio,
        "selected_mask_hash": args.mixed_key_mask_hash,
        "effective_bitwidth_statistics": {},
    }
    rec.update(stop)
    del output, seq, input_ids, attention_mask, encoded
    return rec


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    p.add_argument("--dataset-path", type=Path, default=Path("datasets/aime/aime24.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("results/paper_repro_v2/aime24_budget_n2"))
    p.add_argument("--status-dir", type=Path, default=Path("run/paper_repro_v2/aime24_budget_n2"))
    p.add_argument("--experiment-id", default="aime24_budget_n2")
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--problem-ids", nargs="*", type=int)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--gpu-id", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--model-dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force-think-prefix", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--retry-oom", action="store_true")
    p.add_argument("--overwrite-invalid", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--k-bits", type=int, default=2)
    p.add_argument("--v-bits", type=int, default=2)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--residual-length", type=int, default=128)
    p.add_argument("--sink-length", type=int, default=0)
    p.add_argument("--recent-length", type=int, default=None)
    p.add_argument("--selected-tasks", type=Path)
    p.add_argument("--config-name")
    p.add_argument("--mixed-key-mask-path", type=Path)
    p.add_argument("--mixed-key-int4-ratio", type=float, default=0.0)
    p.add_argument("--mixed-key-mask-hash", default="")
    p.add_argument("--patternkv-cache-path", choices=["legacy", "segmented"], default=os.environ.get("PATTERNKV_CACHE_PATH", "segmented"))
    p.add_argument("--num-k-base", type=int, default=32)
    p.add_argument("--num-v-base", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.recent_length is None:
        args.recent_length = args.residual_length
    args.residual_length = args.recent_length
    args.paper_method_config = apply_method_defaults(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.status_dir.mkdir(parents=True, exist_ok=True)
    rows = load_aime24(args.dataset_path)
    if args.problem_ids:
        rows = [r for r in rows if int(r["problem_id"]) in set(args.problem_ids)]
    args.manifest_methods = METHODS
    cfg = generation_config_dict(args)
    cfg.update(
        {
            "config_name": args.config_name or args.method,
            "method": args.method,
            "k_bits": args.k_bits,
            "v_bits": args.v_bits,
            "group_size": args.group_size,
            "sink_length": args.sink_length,
            "recent_length": args.recent_length,
            "mixed_key_mask_path": str(args.mixed_key_mask_path or ""),
            "mixed_key_int4_ratio": args.mixed_key_int4_ratio,
            "mixed_key_mask_hash": args.mixed_key_mask_hash,
            "patternkv_cache_path": args.patternkv_cache_path if args.method in {"patternkv_paper", "patternkv"} else None,
        }
    )
    cfg_hash = config_hash(cfg)
    manifest_methods = [args.config_name or args.method]
    if args.selected_tasks and args.selected_tasks.exists():
        selected = json.loads(args.selected_tasks.read_text(encoding="utf-8"))
        manifest = []
        for item in selected:
            pid = int(item["problem_id"])
            sid = int(item["sample_id"])
            seed = int(item.get("seed", effective_seed(args.base_seed, pid, sid)))
            manifest.append({"dataset": "aime24", "method": args.method, "config_name": args.config_name or args.method, "problem_id": pid, "sample_id": sid, "task_key": task_key3(pid, sid, seed), "seed": seed, "config_hash": cfg_hash})
    else:
        manifest = build_manifest(load_aime24(args.dataset_path), manifest_methods, args.num_samples, args.base_seed, cfg_hash)
    manifest_path = args.output_dir / "task_manifest.jsonl"
    manifest_mismatch = False
    if manifest_path.exists():
        try:
            first = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
            manifest_mismatch = first.get("config_hash") != cfg_hash
        except Exception:
            manifest_mismatch = True
    if not manifest_path.exists() or args.overwrite_invalid or manifest_mismatch:
        manifest_path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in manifest) + "\n", encoding="utf-8")
    tasks = [x for x in manifest if int(x["problem_id"]) in {int(r["problem_id"]) for r in rows}]
    tasks = shard_tasks(tasks, args.worker_index, args.num_workers)
    git_commit = os.popen("git rev-parse HEAD").read().strip()
    header = {"pid": os.getpid(), "method": args.method, "gpu_id": args.gpu_id, "tasks": len(tasks), "config_hash": cfg_hash, "generation_config": cfg, "prompt_protocol": "deepseek_r1_recommended", "system_prompt": "none", "final_answer_instruction": "boxed"}
    print(json.dumps(header, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    print("[PaperConfigCheck] " + json.dumps(method_config_dict(args), ensure_ascii=False, sort_keys=True), flush=True)
    if args.dry_run:
        return
    if not args.model_path:
        candidates = search_model_candidates()
        raise SystemExit("MODEL_PATH is not set. DeepSeek-R1-Distill-Llama-8B candidates: " + json.dumps(candidates, ensure_ascii=False))
    args.model_path = Path(args.model_path)
    model, tokenizer = load_model(args)
    context_info = validate_context(args, tokenizer, model, load_aime24(args.dataset_path))
    print("[AIMEContextCheck] " + json.dumps(context_info, ensure_ascii=False, sort_keys=True), flush=True)
    row_by_id = {int(r["problem_id"]): r for r in load_aime24(args.dataset_path)}
    try:
        for task in tasks:
            path = result_path(args.output_dir, args.config_name or args.method, int(task["problem_id"]), int(task["sample_id"]), cfg_hash)
            if is_complete_result(path, cfg_hash, args.retry_failed, args.retry_oom):
                print(f"[{utc_now()}] skip complete {path}", flush=True)
                continue
            try:
                rec = run_task(args, model, tokenizer, row_by_id[int(task["problem_id"])], int(task["sample_id"]), cfg_hash, git_commit)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                rec = {**task, "experiment_id": args.experiment_id, "model_path": str(args.model_path), "method": args.method, "stop_reason": "oom", "error": repr(exc), "generated_text": "", "parsed_answer": None, "parser_strategy": "failure", "parser_error": "oom", "git_commit": git_commit, "timestamp": utc_now()}
            except Exception as exc:
                rec = {**task, "experiment_id": args.experiment_id, "model_path": str(args.model_path), "method": args.method, "stop_reason": "error", "error": repr(exc), "traceback": traceback.format_exc(), "generated_text": "", "parsed_answer": None, "parser_strategy": "failure", "parser_error": repr(exc), "git_commit": git_commit, "timestamp": utc_now()}
            write_json_atomic(path, rec)
            print(f"[{utc_now()}] wrote {path} stop={rec.get('stop_reason')} correct={rec.get('is_correct')}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
