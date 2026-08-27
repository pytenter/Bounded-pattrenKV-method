from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VENDOR = os.environ.get("QWEN3_TRANSFORMERS_VENDOR")
if VENDOR and VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from transformers import AutoConfig, AutoTokenizer

from models.qwen3_patternkv import Qwen3ForCausalLM_PatternKV
from models.qwen3_patternkv_system import (
    Qwen3ForCausalLM_PatternKVCompressed,
    collect_qwen3_compressed_dynamic_stats,
    get_qwen3_compressed_counters,
    reset_qwen3_compressed_counters,
)

OUT = ROOT / "reports/qwen3_v100_system_generalization_v1"
MODEL = "/home/qinch2023/modelscope_models/Qwen3-8B"
DECODE = int(os.environ.get("QWEN3_CLOSURE_DECODE", "64"))
CONTEXT = int(os.environ.get("QWEN3_CLOSURE_CONTEXT", "512"))

BASES = [
    "Mathematics proof sketch: define a sequence and reason about modular arithmetic carefully. ",
    "System benchmark prompt: explain cache compression, attention, and deterministic greedy decoding. ",
    "Long context QA: a researcher compares two inference backends and records every generated token. ",
]

CFG_VALUES = dict(
    k_bits=2,
    v_bits=2,
    group_size=128,
    sink_length=16,
    recent_length=128,
    residual_length=128,
    num_k_base=32,
    num_v_base=32,
    patternkv_cache_mode="segmented_rolling",
    patternkv_value_objective="base",
    patternkv_v_precision_selector="causal_v4",
    patternkv_v4_budget_fraction=0.25,
    patternkv_random_selector_seed=20260809,
)


def make_config(task: str):
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False, attn_implementation="eager")
    for key, value in CFG_VALUES.items():
        setattr(config, key, value)
    setattr(config, "patternkv_selector_task_key", task)
    return config


def prompt_ids(tokenizer) -> list[torch.Tensor]:
    out = []
    for base in BASES:
        ids = tokenizer(base * 160, return_tensors="pt", add_special_tokens=False).input_ids[:, :CONTEXT]
        if ids.shape[1] < CONTEXT:
            raise RuntimeError(f"prompt too short: {ids.shape[1]} < {CONTEXT}")
        out.append(ids)
    return out


def run_model(cls: Any, label: str, prompts: list[torch.Tensor]) -> list[dict[str, Any]]:
    model = cls.from_pretrained(
        MODEL,
        local_files_only=True,
        config=make_config(label),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda:0").eval()
    runs = []
    for prompt_index, ids_cpu in enumerate(prompts):
        ids = ids_cpu.to("cuda:0")
        tokens: list[int] = []
        logits_cpu: list[torch.Tensor] = []
        nan_inf = False
        stats = {}
        counters = {}
        if label == "compressed":
            reset_qwen3_compressed_counters()
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, return_dict=True)
            for _step in range(DECODE):
                logits = out.logits[:, -1, :].detach().float().cpu()
                nan_inf = nan_inf or (not bool(torch.isfinite(logits).all()))
                nxt = logits.argmax(dim=-1)
                tokens.append(int(nxt[0]))
                logits_cpu.append(logits.squeeze(0))
                out = model(input_ids=nxt.view(1, 1).to("cuda:0"), past_key_values=out.past_key_values, use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        if label == "compressed":
            stats = collect_qwen3_compressed_dynamic_stats(model, out.past_key_values)["cache_segment_stats_per_layer"][0]
            counters = get_qwen3_compressed_counters()
        runs.append({
            "prompt_index": prompt_index,
            "tokens": tokens,
            "logits": logits_cpu,
            "nan_inf": nan_inf,
            "stats0": stats,
            "counters": counters,
        })
    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    return runs


def b1_reference_vs_compressed() -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts = prompt_ids(tokenizer)
    rec: dict[str, Any] = {"context": CONTEXT, "decode": DECODE, "backend_mode": "CUDA_COMPRESSED_LEGACY"}
    ref = run_model(Qwen3ForCausalLM_PatternKV, "reference", prompts)
    comp = run_model(Qwen3ForCausalLM_PatternKVCompressed, "compressed", prompts)
    rows = []
    all_match = True
    max_rel = 0.0
    max_abs = 0.0
    first = None
    nan_inf = False
    for ref_run, comp_run in zip(ref, comp):
        nan_inf = nan_inf or bool(ref_run["nan_inf"]) or bool(comp_run["nan_inf"])
        for step, (ref_logits, comp_logits) in enumerate(zip(ref_run["logits"], comp_run["logits"])):
            diff = (ref_logits - comp_logits).float()
            rel = float(diff.norm() / ref_logits.float().norm().clamp_min(1e-12))
            max_abs_step = float(diff.abs().max())
            ref_token = int(ref_run["tokens"][step])
            comp_token = int(comp_run["tokens"][step])
            match = ref_token == comp_token
            all_match = all_match and match
            max_rel = max(max_rel, rel)
            max_abs = max(max_abs, max_abs_step)
            if not match and first is None:
                first = {
                    "prompt_index": int(ref_run["prompt_index"]),
                    "step": step,
                    "reference_token": ref_token,
                    "compressed_token": comp_token,
                    "logits_rel_l2": rel,
                    "logits_max_abs": max_abs_step,
                    "classification": "TOKEN_TOP1_DIVERGENCE",
                }
            rows.append({
                "prompt_index": int(ref_run["prompt_index"]),
                "step": step,
                "reference_token": ref_token,
                "compressed_token": comp_token,
                "top1_match": match,
                "logits_rel_l2": rel,
                "logits_max_abs": max_abs_step,
            })
    rec.update({
        "status": "PASS" if all_match and not nan_inf else "FAIL",
        "token_parity": bool(all_match),
        "top1_parity": bool(all_match),
        "no_nan_inf": not nan_inf,
        "max_logits_rel_l2": max_rel,
        "max_logits_max_abs": max_abs,
        "first_divergence": first,
        "rows": rows,
        "compressed_final_counters": [run["counters"] for run in comp],
        "compressed_final_stats0": [run["stats0"] for run in comp],
    })
    return rec


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        rec = b1_reference_vs_compressed()
    except Exception as exc:
        rec = {"status": "ERROR", "error": repr(exc), "context": CONTEXT, "decode": DECODE}
    (OUT / "b1_reference_vs_compressed_raw.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    summary = {key: rec.get(key) for key in ["status", "token_parity", "top1_parity", "no_nan_inf", "max_logits_rel_l2", "max_logits_max_abs", "first_divergence", "error"]}
    (OUT / "b1_reference_vs_compressed.md").write_text("# B1 Reference vs Compressed\n\n```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```\n")
    if rec.get("first_divergence"):
        (OUT / "first_divergence_trace.json").write_text(json.dumps(rec["first_divergence"], indent=2, sort_keys=True) + "\n")
        (OUT / "first_divergence_trace.md").write_text("# First Divergence Trace\n\n```json\n" + json.dumps(rec["first_divergence"], indent=2, sort_keys=True) + "\n```\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if rec.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
