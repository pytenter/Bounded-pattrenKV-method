#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.paper_config import cache_storage_summary, method_config_dict
from scripts.run_longbench_paper_8k_single4090 import (
    encode_prompt,
    gpu_info,
    load_model,
    load_task,
    method_args,
    sample_id,
    set_selector_task_context,
)
from bench.longbench_config import MAX_NEW_TOKENS


def mem() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0}
    torch.cuda.synchronize()
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }


def peak() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0}
    torch.cuda.synchronize()
    return {
        "allocated": int(torch.cuda.max_memory_allocated()),
        "reserved": int(torch.cuda.max_memory_reserved()),
    }


def classify_decode_oom(generated: int) -> str:
    return "FIRST_DECODE" if generated == 0 else "DECODE_GROWTH"


def model_step(model, input_ids, attention_mask, past_key_values=None):
    out = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )
    logits = model.lm_head(out.last_hidden_state[:, -1:, :]).float()
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del logits
    return next_id, out.past_key_values


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise SystemExit("Expose exactly one GPU via CUDA_VISIBLE_DEVICES for this diagnostic.")

    out: dict[str, Any] = {
        "task": args.task,
        "sample_index": args.sample_index,
        "method": args.method,
        "max_new_tokens": args.max_new_tokens,
        "gpu": gpu_info(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "oom_stage": None,
        "generated_tokens_before_oom": 0,
        "error": None,
        "events": {},
    }
    torch.cuda.empty_cache()
    out["events"]["before_model"] = mem()

    margs = method_args(args.method, args.model_path, args.max_input_length, args.output_dir, args.status_dir, args.data_dir)
    out["quantization_config"] = method_config_dict(margs)
    try:
        model, tokenizer = load_model(margs)
        out["events"]["after_model_load"] = mem()
    except torch.cuda.OutOfMemoryError as exc:
        out["oom_stage"] = "MODEL_LOAD"
        out["error"] = repr(exc)
        out["events"]["after_oom"] = mem()
        return out

    try:
        data = load_task(args.task, 0, Path(args.data_dir))
        ex = data[args.sample_index]
        sid = sample_id(args.task, args.sample_index, ex)
        out["sample_id"] = sid
        prompt_info, input_ids, attention_mask = encode_prompt(ex, tokenizer, args.task, args.max_input_length)
        out["raw_input_tokens"] = prompt_info.get("raw_input_tokens")
        out["tokenized_input_length"] = int(input_ids.shape[1])
        out["prompt_truncated_to_max_input"] = bool(prompt_info.get("was_truncated"))
        out["official_max_new_tokens"] = MAX_NEW_TOKENS[args.task]
        out["events"]["before_prefill"] = mem()
        if getattr(margs, "patternkv_v_precision_selector", None) == "causal_v4":
            set_selector_task_context(model, sid)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            next_input, past = model_step(model, input_ids, attention_mask)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            out["oom_stage"] = "PREFILL"
            out["error"] = repr(exc)
            out["events"]["peak_during_prefill"] = peak()
            out["events"]["after_prefill_oom"] = mem()
            return out
        out["prefill_seconds"] = round(time.perf_counter() - t0, 4)
        out["events"]["peak_during_prefill"] = peak()
        out["events"]["after_prefill"] = mem()
        out["prefill_cache_stats"] = cache_storage_summary(
            margs.method,
            past,
            model=model,
            total_cached_tokens=int(input_ids.shape[1]),
            residual_length=margs.residual_length,
        )

        generated = 0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        out["events"]["before_decode"] = mem()
        decode_t0 = time.perf_counter()
        for _ in range(int(args.max_new_tokens)):
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                dim=1,
            )
            try:
                next_input, past = model_step(model, next_input, attention_mask, past)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                out["oom_stage"] = classify_decode_oom(generated)
                out["generated_tokens_before_oom"] = generated
                out["error"] = repr(exc)
                out["events"]["peak_during_decode"] = peak()
                out["events"]["after_decode_oom"] = mem()
                return out
            generated += 1
        out["generated_tokens"] = generated
        out["decode_seconds"] = round(time.perf_counter() - decode_t0, 4)
        out["events"]["peak_during_decode"] = peak()
        out["events"]["after_decode"] = mem()
        out["decode_cache_stats"] = cache_storage_summary(
            margs.method,
            past,
            model=model,
            total_cached_tokens=int(input_ids.shape[1]) + generated,
            residual_length=margs.residual_length,
        )
        out["oom_stage"] = None
        return out
    except Exception as exc:
        out["oom_stage"] = "UNKNOWN"
        out["error"] = repr(exc)
        out["events"]["after_error"] = mem()
        return out
    finally:
        try:
            del model
        except UnboundLocalError:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["causal_v4_25", "patternkv_paper"], required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--sample-index", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--model-path", default="/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct")
    p.add_argument("--data-dir", default="/data/zypan/Block-kvcache-experiment/data/LongBench")
    p.add_argument("--output-dir", type=Path, default=Path("results/causal_v4_25_generalization_v1/longbench_diagnostic_tmp"))
    p.add_argument("--status-dir", type=Path, default=Path("run/causal_v4_25_generalization_v1/longbench_diagnostic_tmp"))
    p.add_argument("--max-input-length", type=int, default=8192)
    p.add_argument("--report-path", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
