import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


def _dtype(name: str):
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _prompt(path: str | None, min_tokens: int | None, tokenizer) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    base = (
        "You are validating a deterministic language model inference path. "
        "Continue the following technical note with concise, ordinary prose. "
        "Do not end early. PatternKV quantizes historical key and value cache blocks while keeping a residual window. "
    )
    text = base
    if min_tokens:
        while len(tokenizer(text, return_tensors="pt").input_ids[0]) < min_tokens:
            text += base
    return text


def _layer_stats(model) -> dict:
    stats = {"layers": []}
    for i, layer in enumerate(getattr(model.model, "layers", [])):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        item = {"layer": i}
        for name in ("k_base", "v_centroids"):
            tensor = getattr(attn, name, None)
            if tensor is not None:
                item[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        stats["layers"].append(item)
    return stats


def _cache_stats(past_key_values) -> dict:
    out = {"layers": []}
    if past_key_values is None:
        return out
    for i, pkv in enumerate(past_key_values):
        item = {"layer": i}
        names = [
            "key_states_quant_trans",
            "key_states_full",
            "key_scale_trans",
            "key_mn_trans",
            "value_states_quant",
            "value_states_full",
            "value_scale",
            "value_mn",
            "kv_seq_len",
            "k_assignments",
            "v_mask",
            "v_assignments_idx",
        ]
        for name, value in zip(names, pkv):
            if torch.is_tensor(value):
                item[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
                if name in ("k_assignments", "v_assignments_idx"):
                    item[name]["min"] = int(value.min().item()) if value.numel() else None
                    item[name]["max"] = int(value.max().item()) if value.numel() else None
                if name == "v_mask":
                    item[name]["mean"] = float(value.float().mean().item()) if value.numel() else None
            else:
                item[name] = value
        out["layers"].append(item)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--method", choices=["fp16", "patternkv"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--k-bits", type=int, default=2)
    parser.add_argument("--v-bits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument("--num-k-base", type=int, default=32)
    parser.add_argument("--num-v-base", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prompt-file")
    parser.add_argument("--min-input-tokens", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    dtype = _dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    min_tokens = args.min_input_tokens
    if min_tokens is None and args.method == "patternkv" and args.max_new_tokens >= 128:
        min_tokens = 256
    prompt = _prompt(args.prompt_file, min_tokens, tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt")
    inputs = encoded.input_ids.to(args.device)
    attention_mask = encoded.attention_mask.to(args.device)

    result = {
        "method": args.method,
        "model_path": args.model_path,
        "input_tokens": int(inputs.shape[1]),
        "config": vars(args),
        "oom": False,
        "error": None,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    model = None
    try:
        if args.method == "fp16":
            model = LlamaForCausalLM.from_pretrained(
                args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True
            ).to(args.device)
        else:
            from models.llama_patternkv import LlamaForCausalLM_PatternKV

            config = LlamaConfig.from_pretrained(args.model_path)
            config.k_bits = args.k_bits
            config.v_bits = args.v_bits
            config.group_size = args.group_size
            config.residual_length = args.residual_length
            config.use_flash = True
            config.num_k_base = args.num_k_base
            config.num_v_base = args.num_v_base
            model = LlamaForCausalLM_PatternKV.from_pretrained(
                args.model_path, config=config, torch_dtype=dtype, low_cpu_mem_usage=True
            ).to(args.device)
        model.eval()
        start.record()
        wall0 = time.perf_counter()
        output = model.generate(
            inputs,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.max_new_tokens if args.max_new_tokens >= args.residual_length else 0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
        )
        end.record()
        torch.cuda.synchronize()
        result["wall_clock_latency_s"] = time.perf_counter() - wall0
        result["cuda_event_latency_ms"] = start.elapsed_time(end)
        seq = output.sequences
        result["output_tokens"] = int(seq.shape[1] - inputs.shape[1])
        result["stop_reason"] = "max_new_tokens" if result["output_tokens"] == args.max_new_tokens else "early_stop"
        result["output_text"] = tokenizer.decode(seq[0, inputs.shape[1] :], skip_special_tokens=True)
        if args.method == "patternkv":
            result["patternkv_layers"] = _layer_stats(model)
            result["patternkv_cache"] = _cache_stats(getattr(output, "past_key_values", None))
    except torch.cuda.OutOfMemoryError as exc:
        result["oom"] = True
        result["error"] = repr(exc)
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        Path(args.output_json).parent.mkdir(exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        if args.method == "patternkv" and os.getenv("PATTERNKV_DEBUG_STATS") == "1":
            Path("results/patternkv_debug_stats.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if result["error"]:
        raise SystemExit(result["error"])


if __name__ == "__main__":
    main()
