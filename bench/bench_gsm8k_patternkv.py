import argparse
import gc
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers.cache_utils as hf_cache_utils
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.gsm8k_utils import (
    build_prompt,
    compute_stop_state,
    load_gsm8k_jsonl,
    normalize_eos_token_ids,
    parse_prediction,
    validate_gsm8k_rows,
)


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


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl_safely(path: Path) -> tuple[list[dict], int]:
    rows = []
    bad_lines = 0
    if not path.exists():
        return rows, bad_lines
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
    return rows, bad_lines


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
    evidence: dict[str, Any] = {
        "layers": len(past_key_values or []),
        "packed_k_exists": False,
        "packed_v_exists": False,
        "packed_k_dtype": None,
        "packed_v_dtype": None,
        "k_assignment_exists": False,
        "v_mask_exists": False,
        "v_assignment_idx_exists": False,
        "quantized_k_tokens": 0,
        "quantized_v_tokens": 0,
        "residual_k_tokens": 0,
        "residual_v_tokens": 0,
        "k_centroid_count_layer0": None,
        "v_centroid_count_layer0": None,
        "v_mask_mean": None,
        "layer0": {},
    }
    masks = []
    if past_key_values:
        layer0 = past_key_values[0]
        evidence["layer0"] = {name: tensor_info(value) for name, value in zip(CACHE_NAMES, layer0)}
        mapping = dict(zip(CACHE_NAMES, layer0))
        kq = mapping.get("key_states_quant_trans")
        vq = mapping.get("value_states_quant")
        kr = mapping.get("key_states_full")
        vr = mapping.get("value_states_full")
        evidence["packed_k_exists"] = torch.is_tensor(kq) and kq.numel() > 0
        evidence["packed_v_exists"] = torch.is_tensor(vq) and vq.numel() > 0
        evidence["packed_k_dtype"] = str(kq.dtype) if torch.is_tensor(kq) else None
        evidence["packed_v_dtype"] = str(vq.dtype) if torch.is_tensor(vq) else None
        evidence["k_assignment_exists"] = torch.is_tensor(mapping.get("k_assignments")) and mapping["k_assignments"].numel() > 0
        evidence["v_mask_exists"] = torch.is_tensor(mapping.get("v_mask")) and mapping["v_mask"].numel() > 0
        evidence["v_assignment_idx_exists"] = torch.is_tensor(mapping.get("v_assignments_idx")) and mapping["v_assignments_idx"].numel() > 0
        evidence["quantized_k_tokens"] = int(kq.shape[-1] * 16) if torch.is_tensor(kq) and kq.ndim >= 4 else 0
        evidence["quantized_v_tokens"] = int(vq.shape[-2]) if torch.is_tensor(vq) and vq.ndim >= 4 else 0
        evidence["residual_k_tokens"] = int(kr.shape[-2]) if torch.is_tensor(kr) and kr.ndim >= 4 else 0
        evidence["residual_v_tokens"] = int(vr.shape[-2]) if torch.is_tensor(vr) and vr.ndim >= 4 else 0
    for pkv in past_key_values or []:
        if len(pkv) > 10 and torch.is_tensor(pkv[10]) and pkv[10].numel():
            masks.append(float(pkv[10].float().mean().item()))
    evidence["v_mask_mean"] = round(sum(masks) / len(masks), 6) if masks else None
    layers = getattr(getattr(model, "model", None), "layers", [])
    if layers:
        attn0 = getattr(layers[0], "self_attn", None)
        for attr, key in (("k_base", "k_centroid_count_layer0"), ("v_centroids", "v_centroid_count_layer0")):
            value = getattr(attn0, attr, None)
            if torch.is_tensor(value) and value.ndim >= 2:
                evidence[key] = int(value.shape[-2])
    evidence["prefill_pattern_mining_executed"] = (evidence["k_centroid_count_layer0"] or 0) >= 32 and (evidence["v_centroid_count_layer0"] or 0) >= 32
    evidence["decode_pattern_update_executed"] = (evidence["k_centroid_count_layer0"] or 0) > 32 or (evidence["v_centroid_count_layer0"] or 0) > 32
    evidence["cuda_fused_path_expected"] = True
    evidence["no_full_history_fp16_fallback_in_evidence"] = evidence["packed_k_exists"] and evidence["packed_v_exists"]
    return evidence


def validate_model_path(path: Path) -> list[str]:
    required = ["config.json", "tokenizer_config.json", "model.safetensors.index.json"]
    issues = [f"missing:{name}" for name in required if not (path / name).exists()]
    if not ((path / "tokenizer.json").exists() or (path / "tokenizer.model").exists()):
        issues.append("missing:tokenizer.json_or_tokenizer.model")
    if not list(path.glob("model*.safetensors")):
        issues.append("missing:model*.safetensors")
    return issues


def load_model(args):
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    args.cache_factory = None
    if args.method in ("fp16", "kivi"):
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        if args.method == "kivi":
            if not hasattr(hf_cache_utils, "is_optimum_quanto_available"):
                hf_cache_utils.is_optimum_quanto_available = lambda: False
            flex_root = Path(args.kvtuner_flex_root)
            if str(flex_root) not in sys.path:
                sys.path.insert(0, str(flex_root))
            from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleVanillaQuantizedCache

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
            if cache_config.axis_key != 1 or cache_config.axis_value != 0:
                raise ValueError(f"Invalid KIVI fixed axes: axis_key={cache_config.axis_key}, axis_value={cache_config.axis_value}")
            args.cache_factory = lambda: FlexibleVanillaQuantizedCache(cache_config=cache_config)
    elif args.method == "kivi_official":
        from models.llama_kivi import LlamaForCausalLM_KIVI

        config = LlamaConfig.from_pretrained(args.model_path, local_files_only=True)
        config.k_bits = args.k_bits
        config.v_bits = args.v_bits
        config.group_size = args.group_size
        config.residual_length = args.residual_length
        config.use_flash = True
        model = LlamaForCausalLM_KIVI.from_pretrained(
            args.model_path,
            local_files_only=True,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to("cuda:0")
    else:
        from models.llama_patternkv import LlamaForCausalLM_PatternKV

        config = LlamaConfig.from_pretrained(args.model_path, local_files_only=True)
        config.k_bits = args.k_bits
        config.v_bits = args.v_bits
        config.group_size = args.group_size
        config.residual_length = args.residual_length
        config.use_flash = True
        config.num_k_base = args.num_k_base
        config.num_v_base = args.num_v_base
        model = LlamaForCausalLM_PatternKV.from_pretrained(
            args.model_path,
            local_files_only=True,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to("cuda:0")
    model.eval()
    return model, tokenizer


def generation_eos_token_ids(tokenizer, model=None) -> list[int]:
    eos_ids = normalize_eos_token_ids(
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(tokenizer, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
    )
    ids = list(eos_ids)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0:
        ids.append(eot_id)
    return sorted({int(x) for x in ids if x is not None})


def output_base(root: Path, mode: str) -> Path:
    if root.name == mode or root.name.startswith(f"{mode}_"):
        return root
    return root / mode


def selected_indices(num_samples: int, shard_id: int, num_shards: int) -> list[int]:
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0,{num_shards}), got {shard_id}")
    shards = np.array_split(np.arange(num_samples), num_shards)
    return [int(x) for x in shards[shard_id].tolist()]


def status(args, status_path: Path, current_index: int | None, started_at: str, out_path: Path) -> None:
    rows, bad = read_jsonl_safely(out_path)
    write_json(
        status_path,
        {
            "pid": os.getpid(),
            "physical_gpu_id": args.physical_gpu_id,
            "logical_gpu_id": 0,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "method": args.method,
            "mode": args.mode,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "num_samples": args.num_samples,
            "expected_shard_samples": len(selected_indices(args.num_samples, args.shard_id, args.num_shards)),
            "completed_samples": len(rows),
            "bad_json_lines": bad,
            "errors": sum(1 for r in rows if r.get("error")),
            "current_sample_index": current_index,
            "output_path": str(out_path),
            "started_at": started_at,
            "updated_at": utc_now(),
        },
    )


@torch.no_grad()
def run_one(model, tokenizer, args, row: dict) -> dict:
    question = row["question"]
    prompt = build_prompt(question, args.prompt_style)
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to("cuda:0")
    attention_mask = encoded.attention_mask.to("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    eos_token_ids = generation_eos_token_ids(tokenizer, model)
    t0 = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_ids,
        return_dict_in_generate=True,
        output_scores=False,
        **({"past_key_values": args.cache_factory()} if getattr(args, "cache_factory", None) is not None else {}),
    )
    torch.cuda.synchronize()
    seq = output.sequences
    output_ids = seq[0, input_ids.shape[1] :]
    prediction = tokenizer.decode(output_ids, skip_special_tokens=True)
    parsed = parse_prediction(prediction)
    gold = str(row["gold_answer"])
    correct = parsed["parsed_answer"] == gold
    generated_token_ids = [int(x) for x in output_ids.detach().cpu().tolist()]
    stop_state = compute_stop_state(generated_token_ids, args.max_new_tokens, eos_token_ids)
    rec = {
        "sample_index": int(row["sample_index"]),
        "sample_id": f"gsm8k:{row['sample_index']}",
        "method": args.method,
        "question": question,
        "prompt": prompt,
        "chat_prompt": chat_prompt,
        "prediction": prediction,
        "gold_raw": row["answer"],
        "gold_answer": gold,
        "parsed_answer": parsed["parsed_answer"],
        "correct": correct,
        "parser_source": parsed["parser_source"],
        "parser_failure": parsed["parser_failure"],
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": int(output_ids.shape[0]),
        "max_new_tokens": args.max_new_tokens,
        **stop_state,
        "latency_s": round(time.perf_counter() - t0, 4),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "error": None,
        "created_at": utc_now(),
    }
    if args.method == "patternkv":
        rec["patternkv_runtime_evidence"] = patternkv_evidence(model, getattr(output, "past_key_values", None))
    if args.method == "kivi":
        rec["kivi_runtime_evidence"] = {
            "method": "kivi_fixed",
            "k_bits": args.k_bits,
            "v_bits": args.v_bits,
            "group_size": args.group_size,
            "residual_length": args.residual_length,
            "axis_key": 1,
            "axis_value": 0,
            "asym": True,
            "dtype": args.dtype,
            "compute_dtype": "torch.float16" if args.dtype == "float16" else "torch.bfloat16",
        }
    if args.method == "kivi_official":
        rec["kivi_runtime_evidence"] = {
            "method": "kivi_official",
            "model_class": "LlamaForCausalLM_KIVI",
            "k_bits": args.k_bits,
            "v_bits": args.v_bits,
            "group_size": args.group_size,
            "residual_length": args.residual_length,
            "use_flash": True,
            "dtype": args.dtype,
            "compute_dtype": "torch.float16" if args.dtype == "float16" else "torch.bfloat16",
        }
    del output, seq, output_ids, input_ids, attention_mask, encoded
    return rec


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["fp16", "kivi", "kivi_official", "patternkv"], required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--num-samples", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=Path("results/gsm8k"))
    p.add_argument("--status-dir", type=Path, default=Path("run/gsm8k"))
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--k-bits", type=int, default=2)
    p.add_argument("--v-bits", type=int, default=2)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--residual-length", type=int, default=128)
    p.add_argument("--num-k-base", type=int, default=32)
    p.add_argument("--num-v-base", type=int, default=32)
    p.add_argument("--physical-gpu-id", required=True)
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    p.add_argument("--prompt-style", choices=["concat", "newline"], default="concat")
    p.add_argument("--kvtuner-flex-root", default=os.environ.get("KVTUNER_FLEX_ROOT", "/data/zypan/Block-kvcache-experiment/third_party/kvtuner/flexible_quant"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id):
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must equal physical GPU id {args.physical_gpu_id}; got {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_issues = validate_model_path(Path(args.model_path))
    if model_issues:
        raise SystemExit(f"Invalid local model path {args.model_path}: {model_issues}")
    data = load_gsm8k_jsonl(Path(args.data_path))
    data_issues = validate_gsm8k_rows(data, expected=1319)
    if data_issues:
        raise SystemExit(f"Invalid GSM8K data path {args.data_path}: {data_issues[:10]}")
    if args.num_samples > len(data):
        raise SystemExit(f"num_samples={args.num_samples} exceeds data size {len(data)}")
    indices = selected_indices(args.num_samples, args.shard_id, args.num_shards)
    out_path = output_base(args.output_dir, args.mode) / args.method / f"shard_{args.shard_id}.jsonl"
    status_path = output_base(args.status_dir, args.mode) / args.method / f"gpu{args.physical_gpu_id}_shard{args.shard_id}.status.json"
    rows, bad_lines = read_jsonl_safely(out_path)
    if bad_lines:
        raise SystemExit(f"{out_path} contains {bad_lines} damaged JSON line(s); preserve evidence and inspect before resuming")
    done = {str(r.get("sample_id")) for r in rows}
    started = utc_now()
    header = {
        "pid": os.getpid(),
        "physical_gpu_id": args.physical_gpu_id,
        "logical_gpu_id": 0,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "method": args.method,
        "mode": args.mode,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "indices_first_last": [indices[0], indices[-1]] if indices else None,
        "expected_shard_samples": len(indices),
        "model_path": args.model_path,
        "data_path": args.data_path,
        "max_new_tokens": args.max_new_tokens,
        "prompt_template": "{Question}Please reason step by step, and put your final answer within \\\\boxed{}.",
        "chat_template_used": True,
        "device_policy": "CUDA_VISIBLE_DEVICES exposes one physical GPU; model and tensors use cuda:0; no tensor parallel",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "started_at": started,
    }
    print(json.dumps(header, indent=2, sort_keys=True), flush=True)
    status(args, status_path, None, started, out_path)
    model, tokenizer = load_model(args)
    try:
        for idx in indices:
            row = data[idx]
            sid = f"gsm8k:{idx}"
            if args.skip_existing and sid in done:
                continue
            status(args, status_path, idx, started, out_path)
            try:
                rec = run_one(model, tokenizer, args, row)
            except torch.cuda.OutOfMemoryError as exc:
                rec = {
                    "sample_index": idx,
                    "sample_id": sid,
                    "method": args.method,
                    "question": row.get("question"),
                    "prompt": build_prompt(row.get("question", ""), args.prompt_style),
                    "prediction": "",
                    "gold_raw": row.get("answer"),
                    "gold_answer": row.get("gold_answer"),
                    "parsed_answer": None,
                    "correct": False,
                    "parser_source": "failure",
                    "parser_failure": True,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "max_new_tokens": args.max_new_tokens,
                    "last_generated_token_id": None,
                    "eos_token_ids": generation_eos_token_ids(tokenizer) if "tokenizer" in locals() else [],
                    "ended_with_eos": False,
                    "hit_max_new_tokens": False,
                    "length_truncated": False,
                    "stop_reason": "error",
                    "latency_s": 0.0,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
                    "error": str(exc),
                    "error_type": "OOM",
                    "created_at": utc_now(),
                }
                torch.cuda.empty_cache()
            except Exception as exc:
                rec = {
                    "sample_index": idx,
                    "sample_id": sid,
                    "method": args.method,
                    "question": row.get("question"),
                    "prompt": build_prompt(row.get("question", ""), args.prompt_style),
                    "prediction": "",
                    "gold_raw": row.get("answer"),
                    "gold_answer": row.get("gold_answer"),
                    "parsed_answer": None,
                    "correct": False,
                    "parser_source": "failure",
                    "parser_failure": True,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "max_new_tokens": args.max_new_tokens,
                    "last_generated_token_id": None,
                    "eos_token_ids": generation_eos_token_ids(tokenizer) if "tokenizer" in locals() else [],
                    "ended_with_eos": False,
                    "hit_max_new_tokens": False,
                    "length_truncated": False,
                    "stop_reason": "error",
                    "latency_s": 0.0,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                    "created_at": utc_now(),
                }
            append_jsonl(out_path, rec)
            done.add(sid)
            print(f"[{utc_now()}] done {args.method} {args.mode} shard={args.shard_id} {len(done)}/{len(indices)} sid={sid} correct={rec.get('correct')}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        status(args, status_path, None, started, out_path)


if __name__ == "__main__":
    main()
