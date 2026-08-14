import argparse
import gc
import json
import math
import os
import random
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers.cache_utils as hf_cache_utils
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench._longbench_scorer import score_example, score_subtask
from bench.longbench_config import DEFAULT_INPUT_CAP, LONGBENCH_PIN, MAX_NEW_TOKENS, METRIC_NAMES, PROMPT_TEMPLATES, SUBTASKS
from bench.paper_config import apply_method_defaults, cache_storage_summary, method_config_dict


TASKS = SUBTASKS
SKIP_CHAT = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
CACHE_NAMES = (
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
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not json serializable: {type(obj)!r}")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def existing_ids(path: Path) -> set[str]:
    ids = set()
    for rec in read_jsonl(path):
        sid = rec.get("sample_id")
        if sid is not None:
            ids.add(str(sid))
    return ids


def sample_id(task: str, index: int, ex: dict) -> str:
    raw = ex.get("_id") or ex.get("id") or ex.get("sample_id") or index
    return f"{task}:{raw}"


def load_task(task: str, num_samples: int, data_dir: Path | None) -> list[dict]:
    def read_jsonl_limited(path: Path) -> list[dict]:
        data = []
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if num_samples > 0 and i >= num_samples:
                    break
                data.append(json.loads(line))
        return data

    def read_zip_limited(path: Path) -> list[dict]:
        member = f"data/{task}.jsonl"
        data = []
        with zipfile.ZipFile(path) as zf:
            with zf.open(member) as f:
                for i, raw in enumerate(f):
                    if num_samples > 0 and i >= num_samples:
                        break
                    data.append(json.loads(raw.decode("utf-8")))
        return data

    if data_dir is not None:
        for jsonl in (data_dir / f"{task}.jsonl", data_dir / "data" / f"{task}.jsonl"):
            if jsonl.exists():
                return read_jsonl_limited(jsonl)
        for zip_path in (data_dir / "data.zip", data_dir / "LongBench.zip"):
            if zip_path.exists():
                return read_zip_limited(zip_path)
    zip_path = Path(hf_hub_download(repo_id="THUDM/LongBench", repo_type="dataset", filename="data.zip"))
    return read_zip_limited(zip_path)


def truncate_middle(prompt: str, tokenizer, max_tokens: int) -> tuple[str, int, int, bool]:
    toks = tokenizer.encode(prompt, add_special_tokens=False)
    if len(toks) <= max_tokens:
        return prompt, len(toks), len(toks), False
    half = max_tokens // 2
    kept = toks[:half] + toks[-half:]
    return tokenizer.decode(toks[:half], skip_special_tokens=True) + tokenizer.decode(toks[-half:], skip_special_tokens=True), len(toks), len(kept), True


def build_prompt(ex: dict, tokenizer, task: str, max_input: int, instruct_model: bool) -> tuple[str, dict]:
    raw = PROMPT_TEMPLATES[task].format(context=ex["context"], input=ex["input"])
    prompt, raw_prompt_tokens, truncated_prompt_tokens, truncated = truncate_middle(raw, tokenizer, max_input)
    raw_chat_tokens = None
    truncated_chat_tokens = None
    if instruct_model and task not in SKIP_CHAT:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt, raw_chat_tokens, truncated_chat_tokens, truncated_after_chat = truncate_middle(prompt, tokenizer, max_input)
        truncated = truncated or truncated_after_chat
    return prompt, {
        "raw_prompt_tokens": raw_prompt_tokens,
        "truncated_prompt_tokens": truncated_prompt_tokens,
        "raw_chat_tokens": raw_chat_tokens,
        "truncated_chat_tokens": truncated_chat_tokens,
        "prompt_truncated_to_max_input": truncated,
    }


def tensor_info(value) -> dict | int | None:
    if torch.is_tensor(value):
        info = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "numel": int(value.numel()),
        }
        if value.numel() and value.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            info["min"] = int(value.min().item())
            info["max"] = int(value.max().item())
        if value.numel() and value.dtype == torch.uint8:
            info["mean"] = float(value.float().mean().item())
        return info
    if isinstance(value, int):
        return value
    return None


def patternkv_evidence(model, past_key_values) -> dict:
    evidence: dict[str, Any] = {"layers": len(past_key_values or []), "layer0": {}, "centroids": []}
    if past_key_values:
        layer0 = past_key_values[0]
        evidence["layer0"] = {name: tensor_info(value) for name, value in zip(CACHE_NAMES, layer0)}
    for i, layer in enumerate(getattr(getattr(model, "model", None), "layers", [])):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        item = {"layer": i}
        for name in ("k_base", "v_centroids"):
            value = getattr(attn, name, None)
            item[name] = tensor_info(value)
        evidence["centroids"].append(item)
    masks = []
    for pkv in past_key_values or []:
        if len(pkv) > 10 and torch.is_tensor(pkv[10]) and pkv[10].numel():
            masks.append(float(pkv[10].float().mean().item()))
    evidence["v_mask_mean"] = round(sum(masks) / len(masks), 6) if masks else None
    return evidence


def load_model(args):
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    args.cache_factory = None
    backend_method = getattr(args, "paper_method_config", None).backend_method if getattr(args, "paper_method_config", None) else args.method
    if backend_method in ("fp16", "hf_flexible_quantized_cache"):
        load_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if args.attn_implementation:
            load_kwargs["attn_implementation"] = args.attn_implementation
        model = LlamaForCausalLM.from_pretrained(args.model_path, **load_kwargs).to("cuda:0")
        if backend_method == "hf_flexible_quantized_cache":
            if not hasattr(hf_cache_utils, "is_optimum_quanto_available"):
                hf_cache_utils.is_optimum_quanto_available = lambda: False
            kvtuner_flex = Path(args.kvtuner_flex_root)
            if str(kvtuner_flex) not in sys.path:
                sys.path.insert(0, str(kvtuner_flex))
            from flexible_quant.flexible_quantized_cache import (
                FlexibleQuantizedCacheConfig,
                FlexibleVanillaQuantizedCache,
            )

            cache_config = FlexibleQuantizedCacheConfig(
                backend="vanilla",
                nbits_key=args.k_bits,
                nbits_value=args.v_bits,
                residual_length=args.residual_length,
                q_group_size=args.group_size,
                asym=True,
                axis_key=1,
                axis_value=0,
                device="cuda",
                compute_dtype=dtype,
                per_layer_quant=False,
            )
            args.cache_factory = lambda: FlexibleVanillaQuantizedCache(cache_config=cache_config)
    elif backend_method == "kivi_official":
        from models.llama_kivi import LlamaForCausalLM_KIVI

        config = LlamaConfig.from_pretrained(args.model_path)
        config.k_bits = args.k_bits
        config.v_bits = args.v_bits
        config.group_size = args.group_size
        config.residual_length = args.residual_length
        config.use_flash = True
        model = LlamaForCausalLM_KIVI.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to("cuda:0")
    else:
        from models.llama_patternkv import LlamaForCausalLM_PatternKV

        config = LlamaConfig.from_pretrained(args.model_path)
        config.k_bits = args.k_bits
        config.v_bits = args.v_bits
        config.group_size = args.group_size
        config.residual_length = args.residual_length
        config.sink_length = getattr(args, "sink_length", 0)
        config.recent_length = getattr(args, "recent_length", args.residual_length)
        config.use_flash = True
        config.num_k_base = args.num_k_base
        config.num_v_base = args.num_v_base
        config.patternkv_cache_path = getattr(args, "patternkv_cache_path", "segmented")
        config.patternkv_cache_mode = getattr(args, "patternkv_cache_mode", "segmented_rolling")
        config.patternkv_value_objective = getattr(args, "patternkv_value_objective", "base")
        config.patternkv_v_precision_selector = getattr(args, "patternkv_v_precision_selector", "base_v2")
        config.patternkv_v4_budget_fraction = float(getattr(args, "patternkv_v4_budget_fraction", 0.0))
        config.patternkv_random_selector_seed = int(getattr(args, "patternkv_random_selector_seed", 20260809))
        model = LlamaForCausalLM_PatternKV.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to("cuda:0")
    model.eval()
    return model, tokenizer


def task_status(args, status_path: Path, current_task: str | None, current_sample: str | None, started_at: str) -> None:
    completed = 0
    failures = 0
    mtimes = {}
    for task in args.tasks:
        f = args.output_dir / args.method / f"{task}.jsonl"
        rows = read_jsonl(f)
        completed += len(rows)
        failures += sum(1 for row in rows if row.get("error"))
        if f.exists():
            mtimes[task] = datetime.fromtimestamp(f.stat().st_mtime).isoformat()
    write_json(
        status_path,
        {
            "pid": os.getpid(),
            "physical_gpu_id": args.gpu_id,
            "logical_gpu_id": 0,
            "method": args.method,
            "paper_method_config": method_config_dict(args),
            "mode": args.mode,
            "tasks": args.tasks,
            "num_samples_per_task": args.num_samples,
            "total_samples": (args.num_samples * len(args.tasks)) if args.num_samples > 0 else None,
            "completed_samples": completed,
            "failures": failures,
            "current_task": current_task,
            "current_sample": current_sample,
            "result_mtimes": mtimes,
            "started_at": started_at,
            "updated_at": utc_now(),
        },
    )


@torch.no_grad()
def run_one_sample(model, tokenizer, args, task: str, index: int, ex: dict) -> dict:
    sid = sample_id(task, index, ex)
    prompt, prompt_stats = build_prompt(ex, tokenizer, task, args.max_input_length, args.instruct_model)
    add_special = not (args.instruct_model and task not in SKIP_CHAT)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special)
    input_ids = encoded.input_ids.to("cuda:0")
    attention_mask = encoded.attention_mask.to("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
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
        **({"past_key_values": args.cache_factory()} if getattr(args, "cache_factory", None) is not None else {}),
    )
    torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    seq = output.sequences
    pred = tokenizer.decode(seq[0, input_ids.shape[1] :], skip_special_tokens=True)
    total_cached_tokens = int(seq.shape[1])
    cache_stats = cache_storage_summary(
        args.method,
        getattr(output, "past_key_values", None),
        model=model,
        total_cached_tokens=total_cached_tokens,
        residual_length=args.residual_length,
    )
    refs = list(ex.get("answers") or [])
    all_classes = list(ex.get("all_classes") or [])
    score = score_example(task, pred, refs, all_classes=all_classes or None)
    rec = {
        "sample_id": sid,
        "task": task,
        "method": args.method,
        "sample_index": index,
        "prediction": pred,
        "answers": refs,
        "all_classes": all_classes,
        "score": score,
        "metric": METRIC_NAMES[task],
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": int(seq.shape[1] - input_ids.shape[1]),
        "raw_input_tokens": prompt_stats["raw_prompt_tokens"],
        "truncated_input_tokens": prompt_stats["truncated_prompt_tokens"],
        "raw_chat_tokens": prompt_stats["raw_chat_tokens"],
        "truncated_chat_tokens": prompt_stats["truncated_chat_tokens"],
        "total_cached_tokens": total_cached_tokens,
        "quantized_tokens": cache_stats["quantized_tokens"],
        "fp16_residual_tokens": cache_stats["fp16_residual_tokens"],
        "max_new_tokens": MAX_NEW_TOKENS[task],
        "prompt_truncated_to_max_input": prompt_stats["prompt_truncated_to_max_input"],
        "paper_config_snapshot": method_config_dict(args),
        "cache_bitwidth_stats": cache_stats,
        "latency_s": round(latency, 4),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "error": None,
        "created_at": utc_now(),
    }
    if args.method in ("patternkv", "patternkv_paper"):
        rec["patternkv_runtime_evidence"] = patternkv_evidence(model, getattr(output, "past_key_values", None))
    if args.method == "kivi":
        rec["kivi_runtime_evidence"] = {
            "k_bits": args.k_bits,
            "v_bits": args.v_bits,
            "group_size": args.group_size,
            "residual_length": args.residual_length,
            "axis_key": 1,
            "axis_value": 0,
        }
    if args.method in ("kivi_official", "kivi_paper_g128", "kivi_original_g32"):
        rec["kivi_runtime_evidence"] = {
            "method": args.method,
            "model_class": "LlamaForCausalLM_KIVI",
            "k_bits": args.k_bits,
            "v_bits": args.v_bits,
            "group_size": args.group_size,
            "residual_length": args.residual_length,
            "use_flash": True,
            "axis_key": "per-channel: transposed K last dim is sequence length",
            "axis_value": "per-token: V last dim is head_dim",
        }
    del output, seq, input_ids, attention_mask, encoded
    return rec


def summarize_task(path: Path, task: str, args, started_at: float, expected_samples: int | None = None) -> dict:
    rows = read_jsonl(path)
    preds = [str(row.get("prediction") or "") for row in rows]
    refs = [list(row.get("answers") or []) for row in rows]
    all_classes = None
    for row in rows:
        if row.get("all_classes"):
            all_classes = list(row["all_classes"])
            break
    score = score_subtask(task, preds, refs, all_classes=all_classes) if rows else {"score": math.nan, "metric": METRIC_NAMES[task], "n": 0}
    summary = {
        "task": task,
        "method": args.method,
        "path": str(path),
        "samples": len(rows),
        "expected_samples": expected_samples if expected_samples is not None else args.num_samples,
        "failures": sum(1 for row in rows if row.get("error")),
        "empty_predictions": sum(1 for row in rows if not str(row.get("prediction") or "").strip()),
        "score": score["score"],
        "metric": score["metric"],
        "avg_input_tokens": round(sum(int(row.get("input_tokens") or 0) for row in rows) / len(rows), 2) if rows else 0,
        "avg_output_tokens": round(sum(int(row.get("output_tokens") or 0) for row in rows) / len(rows), 2) if rows else 0,
        "wall_time_s": round(time.perf_counter() - started_at, 2),
        "updated_at": utc_now(),
    }
    write_json(path.with_suffix(".summary.json"), summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["fp16", "patternkv", "patternkv_paper", "causal_v4_25", "kivi", "kivi_official", "kivi_paper_g128", "kivi_original_g32"], required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    parser.add_argument("--num-samples", type=int, required=True, help="Use <=0 for the full available LongBench split for each task.")
    parser.add_argument("--model-path", default="/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--data-dir", default=os.environ.get("LONGBENCH_DATA_DIR"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/longbench"))
    parser.add_argument("--status-dir", type=Path, default=Path("run/longbench"))
    parser.add_argument("--mode", default="manual")
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--max-input-length", type=int, default=DEFAULT_INPUT_CAP)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-bits", type=int, default=2)
    parser.add_argument("--v-bits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument("--num-k-base", type=int, default=32)
    parser.add_argument("--num-v-base", type=int, default=32)
    parser.add_argument("--kvtuner-flex-root", default="/data/zypan/Block-kvcache-experiment/third_party/kvtuner/flexible_quant")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--instruct-model", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.paper_method_config = apply_method_defaults(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir) if args.data_dir else None
    started_at = utc_now()
    status_path = args.status_dir / f"gpu{args.gpu_id}_{args.method}_{args.mode}.status.json"
    args.status_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "pid": os.getpid(),
        "physical_gpu_id": args.gpu_id,
        "logical_gpu_id": 0,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "method": args.method,
        "paper_method_config": method_config_dict(args),
        "tasks": args.tasks,
        "started_at": started_at,
        "model_path": args.model_path,
        "max_input_length": args.max_input_length,
        "longbench_pin": LONGBENCH_PIN,
        "device_policy": "CUDA_VISIBLE_DEVICES exposes one physical GPU; model.to('cuda:0'); no device_map='auto'",
    }
    print(json.dumps(header, indent=2, sort_keys=True), flush=True)
    print("[PaperConfigCheck] " + json.dumps(method_config_dict(args), ensure_ascii=False, sort_keys=True), flush=True)
    task_status(args, status_path, None, None, started_at)
    model, tokenizer = load_model(args)
    all_summaries = []
    try:
        for task in args.tasks:
            out_path = args.output_dir / args.method / f"{task}.jsonl"
            data = load_task(task, args.num_samples, data_dir)
            expected_count = len(data)
            if args.skip_existing and expected_count > 0 and len(read_jsonl(out_path)) >= expected_count:
                print(f"[{utc_now()}] skip complete {args.method}/{task}: {out_path}", flush=True)
                all_summaries.append(summarize_task(out_path, task, args, time.perf_counter(), expected_count))
                continue
            task_t0 = time.perf_counter()
            done = existing_ids(out_path)
            for i, ex in enumerate(data):
                sid = sample_id(task, i, ex)
                if sid in done:
                    continue
                task_status(args, status_path, task, sid, started_at)
                try:
                    rec = run_one_sample(model, tokenizer, args, task, i, ex)
                except Exception as exc:
                    rec = {
                        "sample_id": sid,
                        "task": task,
                        "method": args.method,
                        "sample_index": i,
                        "prediction": "",
                        "answers": list(ex.get("answers") or []),
                        "all_classes": list(ex.get("all_classes") or []),
                        "score": 0.0,
                        "metric": METRIC_NAMES[task],
                        "input_tokens": None,
                        "output_tokens": 0,
                        "max_new_tokens": MAX_NEW_TOKENS[task],
                        "paper_config_snapshot": method_config_dict(args),
                        "error": repr(exc),
                        "created_at": utc_now(),
                    }
                    print(f"[{utc_now()}] ERROR {args.method}/{task}/{sid}: {exc!r}", flush=True)
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                append_jsonl(out_path, rec)
                done.add(sid)
                gc.collect()
                torch.cuda.empty_cache()
                task_status(args, status_path, task, sid, started_at)
                print(f"[{utc_now()}] done {args.method}/{task} {len(done)}/{expected_count} sid={sid}", flush=True)
            summary = summarize_task(out_path, task, args, task_t0, expected_count)
            all_summaries.append(summary)
            print(f"[{utc_now()}] task summary {json.dumps(summary, sort_keys=True)}", flush=True)
            os.system("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")
    finally:
        task_status(args, status_path, None, None, started_at)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output_dir / args.method / f"worker_gpu{args.gpu_id}_{args.mode}.summary.json", {"worker": header, "summaries": all_summaries})


if __name__ == "__main__":
    main()
