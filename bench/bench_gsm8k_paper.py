import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.gsm8k_paper_utils import (
    METHODS,
    build_prompt,
    compute_stop_state,
    config_hash,
    extract_reference,
    is_complete,
    load_gsm8k,
    manifest,
    parse_prediction,
    result_path,
    write_json_atomic,
)
from bench.paper_config import apply_method_defaults, cache_storage_summary, method_config_dict


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


def load_model(args):
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
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
        config.use_flash = True
        model = LlamaForCausalLM_KIVI.from_pretrained(args.model_path, local_files_only=True, config=config, torch_dtype=dtype, low_cpu_mem_usage=True).to("cuda:0")
    elif backend == "patternkv":
        from models.llama_patternkv import LlamaForCausalLM_PatternKV

        config = LlamaConfig.from_pretrained(args.model_path, local_files_only=True)
        config.k_bits = args.k_bits
        config.v_bits = args.v_bits
        config.group_size = args.group_size
        config.residual_length = args.residual_length
        config.use_flash = True
        config.num_k_base = args.num_k_base
        config.num_v_base = args.num_v_base
        model = LlamaForCausalLM_PatternKV.from_pretrained(args.model_path, local_files_only=True, config=config, torch_dtype=dtype, low_cpu_mem_usage=True).to("cuda:0")
    else:
        raise ValueError(f"Unsupported backend {backend}")
    model.eval()
    return model, tokenizer


@torch.no_grad()
def run_one(args, model, tokenizer, row, cfg_hash, git_commit):
    rendered_prompt, user_prompt = build_prompt(row["question"], tokenizer)
    encoded = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to("cuda:0")
    attention_mask = encoded.attention_mask.to("cuda:0")
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
    gen_ids = seq[0, input_ids.shape[1] :].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    parsed = parse_prediction(text)
    ref = extract_reference(row["answer"])
    stop = compute_stop_state(gen_ids, args.max_new_tokens, eos_ids(tokenizer, model))
    cache_stats = cache_storage_summary(args.method, getattr(out, "past_key_values", None), model=model, total_cached_tokens=int(seq.shape[1]), residual_length=args.residual_length)
    rec = {
        "experiment_id": args.experiment_id,
        "config_hash": cfg_hash,
        "dataset": "gsm8k",
        "split": "test",
        "problem_id": int(row["problem_id"]),
        "task_key": f"gsm8k:p{int(row['problem_id'])}",
        "method": args.method,
        "model_path": str(args.model_path),
        "question": row["question"],
        "reference_answer": ref,
        "rendered_prompt": rendered_prompt,
        "prompt_protocol": "zero-shot-cot",
        "prompt_source": "reproduction-choice",
        "input_tokens": int(input_ids.shape[1]),
        "do_sample": False,
        "batch_size": 1,
        "num_return_sequences": 1,
        "max_new_tokens": args.max_new_tokens,
        "generated_text": text,
        "raw_prediction": text,
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
        "quantization_config": method_config_dict(args),
        "patternkv_config": method_config_dict(args) if args.method == "patternkv_paper" else {},
        "cache_bitwidth_stats": cache_stats,
        "git_commit": git_commit,
        "gpu_id": args.gpu_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "error": None,
    }
    rec.update(stop)
    del out, seq, input_ids, attention_mask, encoded
    return rec


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct"))
    p.add_argument("--data-path", type=Path, default=Path("datasets/gsm8k/gsm8k_test.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("results/paper_repro_v2/gsm8k_full"))
    p.add_argument("--status-dir", type=Path, default=Path("run/paper_repro_v2/gsm8k_full"))
    p.add_argument("--experiment-id", default="gsm8k_full_paper")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--problem-ids", nargs="*", type=int)
    p.add_argument("--gpu-id", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--retry-oom", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--k-bits", type=int, default=2)
    p.add_argument("--v-bits", type=int, default=2)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--residual-length", type=int, default=128)
    p.add_argument("--num-k-base", type=int, default=32)
    p.add_argument("--num-v-base", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    args.paper_method_config = apply_method_defaults(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.status_dir.mkdir(parents=True, exist_ok=True)
    rows = load_gsm8k(args.data_path)
    if args.problem_ids:
        rows = [r for r in rows if int(r["problem_id"]) in set(args.problem_ids)]
    cfg = {"dataset": "gsm8k", "split": "test", "model_path": str(args.model_path), "method": args.method, "max_new_tokens": args.max_new_tokens, "do_sample": False, "batch_size": 1, "num_return_sequences": 1, "paper_method_config": method_config_dict(args)}
    cfg_hash = config_hash(cfg)
    all_manifest = manifest(load_gsm8k(args.data_path), METHODS, cfg_hash)
    manifest_path = args.output_dir / "task_manifest.jsonl"
    if not manifest_path.exists():
        manifest_path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in all_manifest) + "\n", encoding="utf-8")
    tasks = [r for i, r in enumerate(rows) if i % args.num_workers == args.worker_index]
    header = {"dataset": "gsm8k", "split": "test", "prompt_protocol": "zero-shot-cot", "prompt_source": "reproduction-choice", "do_sample": False, "batch_size": 1, "max_new_tokens": args.max_new_tokens, "method": args.method, "config_hash": cfg_hash, "tasks": len(tasks), "paper_config": method_config_dict(args)}
    print(json.dumps(header, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    print("[PaperConfigCheck] " + json.dumps(method_config_dict(args), ensure_ascii=False, sort_keys=True), flush=True)
    if args.dry_run:
        return
    git_commit = os.popen("git rev-parse HEAD").read().strip()
    model, tokenizer = load_model(args)
    try:
        for row in tasks:
            pid = int(row["problem_id"])
            path = result_path(args.output_dir, args.method, pid, cfg_hash)
            if is_complete(path, cfg_hash, args.retry_failed, args.retry_oom):
                print(f"skip complete {path}", flush=True)
                continue
            try:
                rec = run_one(args, model, tokenizer, row, cfg_hash, git_commit)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                rec = {"dataset": "gsm8k", "split": "test", "problem_id": pid, "method": args.method, "config_hash": cfg_hash, "generated_text": "", "parsed_answer": None, "stop_reason": "oom", "hit_max_new_tokens": False, "is_correct": False, "error": repr(exc), "git_commit": git_commit, "gpu_id": args.gpu_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            except Exception as exc:
                rec = {"dataset": "gsm8k", "split": "test", "problem_id": pid, "method": args.method, "config_hash": cfg_hash, "generated_text": "", "parsed_answer": None, "stop_reason": "error", "hit_max_new_tokens": False, "is_correct": False, "error": repr(exc), "traceback": traceback.format_exc(), "git_commit": git_commit, "gpu_id": args.gpu_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            write_json_atomic(path, rec)
            print(f"wrote {path} stop={rec.get('stop_reason')} correct={rec.get('is_correct')}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
