from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.segmented_cache import deserialize_cache
from quant.page_batch import (
    get_patternkv_page_batch_counters,
    get_patternkv_real_decode_counters,
    reset_patternkv_page_batch_counters,
    reset_patternkv_real_decode_counters,
)


MODEL_PATH = Path("/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B")
REPORT_DIR = REPO_ROOT / "reports/system_actual_model_fixed_batch_v1"
START_HEAD = "970d1bad623e016d876fe460cc2fdef0bf01a005"


PROMPTS = [
    "Request A: explain why matrix multiplication is associative in two concise steps.",
    "Request B: derive the area of a circle using a limiting polygon argument.",
    "Request C: write a tiny proof that every even square is divisible by four.",
    "Request D: compare breadth first search and depth first search for shortest paths.",
]


@dataclass(frozen=True)
class SmokeCase:
    name: str
    batch: int
    context: int
    decode: int
    kind: str


def smoke_cases(matrix: str) -> list[SmokeCase]:
    stage1 = [
        SmokeCase("b1_ctx512_d1", 1, 512, 1, "stage1"),
        SmokeCase("b2_ctx512_d1", 2, 512, 1, "stage1"),
        SmokeCase("b2_ctx512_d8", 2, 512, 8, "stage1"),
        SmokeCase("b2_ctx512_d32", 2, 512, 32, "stage1"),
    ]
    stage2 = [
        SmokeCase("b1_ctx2048_d1", 1, 2048, 1, "stage2"),
        SmokeCase("b2_ctx2048_d1", 2, 2048, 1, "stage2"),
        SmokeCase("b4_ctx2048_d1", 4, 2048, 1, "stage2"),
        SmokeCase("b2_ctx2048_d8", 2, 2048, 8, "stage2"),
        SmokeCase("b4_ctx2048_d8", 4, 2048, 8, "stage2"),
        SmokeCase("b2_ctx2048_d32", 2, 2048, 32, "stage2"),
        SmokeCase("b4_ctx2048_d32", 4, 2048, 32, "stage2"),
    ]
    boundary = [
        SmokeCase("b2_ctx2048_d127", 2, 2048, 127, "boundary"),
        SmokeCase("b2_ctx2048_d128", 2, 2048, 128, "boundary"),
        SmokeCase("b2_ctx2048_d129", 2, 2048, 129, "boundary"),
        SmokeCase("b4_ctx2048_d127", 4, 2048, 127, "boundary"),
        SmokeCase("b4_ctx2048_d128", 4, 2048, 128, "boundary"),
        SmokeCase("b4_ctx2048_d129", 4, 2048, 129, "boundary"),
    ]
    optional = [
        SmokeCase("b2_ctx4096_d32", 2, 4096, 32, "optional_4k"),
        SmokeCase("b4_ctx4096_d32", 4, 4096, 32, "optional_4k"),
    ]
    if matrix == "stage1":
        return stage1
    if matrix == "stage2":
        return stage1 + stage2
    if matrix == "boundary":
        return stage1 + stage2 + boundary
    if matrix == "full":
        return stage1 + stage2 + boundary + optional
    raise ValueError(f"unknown matrix: {matrix}")


def tensor_metrics(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float | int]:
    got_f = got.detach().float()
    ref_f = ref.detach().float()
    diff = got_f - ref_f
    ref_norm = torch.linalg.vector_norm(ref_f).clamp_min(1e-12)
    got_norm = torch.linalg.vector_norm(got_f).clamp_min(1e-12)
    rel = torch.linalg.vector_norm(diff) / ref_norm
    cosine = torch.sum(got_f * ref_f) / (got_norm * ref_norm)
    return {
        "max_abs": float(diff.abs().max().item()),
        "relative_l2": float(rel.item()),
        "cosine": float(cosine.item()),
        "nan": int(torch.isnan(got_f).sum().item()),
        "inf": int(torch.isinf(got_f).sum().item()),
    }


def make_fixed_inputs(tokenizer: Any, batch: int, context: int, device: torch.device) -> torch.Tensor:
    rows = []
    bos = int(tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1)
    for prompt in PROMPTS[:batch]:
        body = tokenizer.encode(prompt, add_special_tokens=False)
        if not body:
            body = [bos]
        tokens = [bos]
        while len(tokens) < context:
            tokens.extend(body)
        rows.append(tokens[:context])
    return torch.tensor(rows, dtype=torch.long, device=device)


def load_model(dtype: torch.dtype, device: torch.device):
    from transformers import AutoTokenizer, LlamaConfig
    from models.llama_patternkv import LlamaForCausalLM_PatternKV

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=False, trust_remote_code=True)
    config = LlamaConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    config.k_bits = 2
    config.v_bits = 2
    config.group_size = 128
    config.residual_length = 128
    config.sink_length = 16
    config.recent_length = 128
    config.use_flash = True
    config.num_k_base = 32
    config.num_v_base = 32
    config.patternkv_cache_path = "segmented"
    config.patternkv_cache_mode = "segmented_rolling"
    config.patternkv_value_objective = "v_dir"
    config.patternkv_v_precision_selector = "causal_v4"
    config.patternkv_v4_budget_fraction = 0.25
    model = LlamaForCausalLM_PatternKV.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tokenizer, config, model


def greedy_run(model: Any, input_ids: torch.Tensor, decode_steps: int) -> dict[str, Any]:
    reset_patternkv_page_batch_counters()
    reset_patternkv_real_decode_counters()
    torch.cuda.reset_peak_memory_stats(input_ids.device)
    started = time.perf_counter()
    records = []
    with torch.inference_mode():
        prefill = model(input_ids=input_ids, use_cache=True, output_hidden_states=True, return_dict=True)
        past = prefill.past_key_values
        prefill_logits = prefill.logits[:, -1, :].detach()
        prefill_hidden = prefill.hidden_states[-1][:, -1, :].detach()
        next_token = prefill_logits.argmax(dim=-1)
        records.append({"step": 0, "phase": "prefill", "logits": prefill_logits, "hidden": prefill_hidden, "token": next_token})
        for step in range(1, decode_steps + 1):
            out = model(input_ids=next_token[:, None], past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].detach()
            hidden = out.hidden_states[-1][:, -1, :].detach()
            next_token = logits.argmax(dim=-1)
            records.append({"step": step, "phase": "decode", "logits": logits, "hidden": hidden, "token": next_token})
    torch.cuda.synchronize(input_ids.device)
    elapsed = time.perf_counter() - started
    return {
        "records": records,
        "past_key_values": past,
        "elapsed_s": elapsed,
        "peak_memory_allocated": int(torch.cuda.max_memory_allocated(input_ids.device)),
        "peak_memory_reserved": int(torch.cuda.max_memory_reserved(input_ids.device)),
        "page_counters": get_patternkv_page_batch_counters(),
        "real_counters": get_patternkv_real_decode_counters(),
    }


def cache_state_summary(past_key_values: Any, row: int | None = None) -> dict[str, Any]:
    layers = []
    for layer_idx, layer_cache in enumerate(past_key_values or []):
        cache = deserialize_cache(layer_cache, pattern=True)
        item: dict[str, Any] = {
            "layer": layer_idx,
            "total_tokens": int(cache.total_tokens),
            "packed_k_tokens": int(cache.packed_k_tokens),
            "packed_v_tokens": int(cache.packed_v_tokens),
            "packed_v4_tokens": int(getattr(cache, "packed_v4_tokens", 0) or 0),
        }
        pool = getattr(cache, "centroid_state_pool", None)
        slots = getattr(cache, "centroid_state_indices", None)
        if pool is not None and slots is not None:
            slot = int(slots[row].item()) if row is not None else None
            if slot is None:
                item["centroid_state_indices"] = [int(x) for x in slots.detach().cpu().tolist()]
                item["k_counts"] = [int(x) for x in pool.k_counts[slots.long()].detach().cpu().tolist()]
                item["v_counts"] = [int(x) for x in pool.v_counts[slots.long()].detach().cpu().tolist()]
                item["update_counts_k"] = [int(x) for x in pool.update_counts_k[slots.long()].detach().cpu().tolist()]
                item["update_counts_v"] = [int(x) for x in pool.update_counts_v[slots.long()].detach().cpu().tolist()]
            else:
                item["centroid_state_slot"] = slot
                item["k_count"] = int(pool.k_counts[slot].item())
                item["v_count"] = int(pool.v_counts[slot].item())
                item["update_count_k"] = int(pool.update_counts_k[slot].item())
                item["update_count_v"] = int(pool.update_counts_v[slot].item())
        if cache.v_precision_mask is not None:
            mask = cache.v_precision_mask[row : row + 1] if row is not None else cache.v_precision_mask
            item["v_precision_mask_shape"] = list(mask.shape)
            item["v_precision_v4"] = int(mask.bool().sum().item())
        layers.append(item)
    return {"layers": layers}


def structural_match(batch_past: Any, refs: list[Any], batch: int) -> tuple[bool, str]:
    for row in range(batch):
        b_summary = cache_state_summary(batch_past, row=row)["layers"]
        r_summary = cache_state_summary(refs[row], row=0)["layers"]
        for layer_idx, (b_layer, r_layer) in enumerate(zip(b_summary, r_summary)):
            keys = ["total_tokens", "packed_k_tokens", "packed_v_tokens", "k_count", "v_count", "update_count_k", "update_count_v", "v_precision_v4"]
            for key in keys:
                if b_layer.get(key) != r_layer.get(key):
                    return False, f"row={row} layer={b_layer['layer']} key={key} batched={b_layer.get(key)} ref={r_layer.get(key)}"
            b_cache = deserialize_cache(batch_past[layer_idx], pattern=True)
            r_cache = deserialize_cache(refs[row][layer_idx], pattern=True)
            if b_cache.k_assignments is not None and r_cache.k_assignments is not None:
                if not torch.equal(b_cache.k_assignments[row : row + 1], r_cache.k_assignments):
                    return False, f"row={row} layer={layer_idx} k_assignments differ"
            if b_cache.v_assignment_idx is not None and r_cache.v_assignment_idx is not None:
                if not torch.equal(b_cache.v_assignment_idx[row : row + 1], r_cache.v_assignment_idx):
                    return False, f"row={row} layer={layer_idx} v_assignment_idx differ"
            if b_cache.v_pattern_mask is not None and r_cache.v_pattern_mask is not None:
                if not torch.equal(b_cache.v_pattern_mask[row : row + 1], r_cache.v_pattern_mask):
                    return False, f"row={row} layer={layer_idx} v_pattern_mask differ"
            if b_cache.v_precision_mask is not None and r_cache.v_precision_mask is not None:
                if not torch.equal(b_cache.v_precision_mask[row : row + 1], r_cache.v_precision_mask):
                    return False, f"row={row} layer={layer_idx} v_precision_mask differ"
            b_pool = getattr(b_cache, "centroid_state_pool", None)
            b_slots = getattr(b_cache, "centroid_state_indices", None)
            r_pool = getattr(r_cache, "centroid_state_pool", None)
            r_slots = getattr(r_cache, "centroid_state_indices", None)
            if b_pool is not None and b_slots is not None and r_pool is not None and r_slots is not None:
                b_slot = int(b_slots[row].item())
                r_slot = int(r_slots[0].item())
                b_k_count = int(b_pool.k_counts[b_slot].item())
                r_k_count = int(r_pool.k_counts[r_slot].item())
                b_v_count = int(b_pool.v_counts[b_slot].item())
                r_v_count = int(r_pool.v_counts[r_slot].item())
                if not torch.equal(b_pool.k_centroid_pool[b_slot, :, :b_k_count, :], r_pool.k_centroid_pool[r_slot, :, :r_k_count, :]):
                    return False, f"row={row} layer={layer_idx} k_centroid_values differ"
                if not torch.equal(b_pool.v_centroid_pool[b_slot, :, :b_v_count, :], r_pool.v_centroid_pool[r_slot, :, :r_v_count, :]):
                    return False, f"row={row} layer={layer_idx} v_centroid_values differ"
    return True, ""


def run_case(model: Any, tokenizer: Any, case: SmokeCase, device: torch.device) -> dict[str, Any]:
    input_ids = make_fixed_inputs(tokenizer, case.batch, case.context, device)
    refs = []
    for row in range(case.batch):
        refs.append(greedy_run(model, input_ids[row : row + 1], case.decode))
        torch.cuda.empty_cache()
    batched = greedy_run(model, input_ids, case.decode)
    max_hidden_rel = 0.0
    max_logit_rel = 0.0
    min_logit_cos = 1.0
    token_matches = 0
    token_total = 0
    numerical_rows = []
    for row in range(case.batch):
        for step_idx, record in enumerate(batched["records"]):
            ref_record = refs[row]["records"][step_idx]
            hm = tensor_metrics(record["hidden"][row], ref_record["hidden"][0])
            lm = tensor_metrics(record["logits"][row], ref_record["logits"][0])
            token_match = int(record["token"][row].item()) == int(ref_record["token"][0].item())
            token_matches += int(token_match)
            token_total += 1
            max_hidden_rel = max(max_hidden_rel, float(hm["relative_l2"]))
            max_logit_rel = max(max_logit_rel, float(lm["relative_l2"]))
            min_logit_cos = min(min_logit_cos, float(lm["cosine"]))
            numerical_rows.append(
                {
                    "case": case.name,
                    "batch": case.batch,
                    "row": row,
                    "step": record["step"],
                    "phase": record["phase"],
                    "hidden_relative_l2": hm["relative_l2"],
                    "logit_relative_l2": lm["relative_l2"],
                    "logit_cosine": lm["cosine"],
                    "logit_max_abs": lm["max_abs"],
                    "token_match": token_match,
                    "batched_token": int(record["token"][row].item()),
                    "ref_token": int(ref_record["token"][0].item()),
                }
            )
    state_ok, state_reason = structural_match(batched["past_key_values"], [ref["past_key_values"] for ref in refs], case.batch)
    counters = batched["real_counters"]
    true_batch_ok = case.batch == 1 or (counters.get("serial_b1_dispatches", 0) == 0 and counters.get("fused_page_operator_calls", 0) > 0)
    hard_counters_ok = (
        counters.get("serial_b1_dispatches", 0) == 0
        and counters.get("legacy_mixed_v_operator_calls", 0) == 0
        and counters.get("historical_v_materialization_bytes", 0) == 0
        and counters.get("page_value_materialization_bytes", 0) == 0
        and counters.get("operator_ready_pool_full_rebuilds", 0) == 0
        and counters.get("gpu_tensor_item_calls_hot_path", 0) == 0
        and counters.get("python_page_dispatches", 0) == 0
    )
    pass_case = state_ok and true_batch_ok and hard_counters_ok and token_matches == token_total and max_logit_rel < 5e-2
    return {
        "case": case.__dict__,
        "pass": bool(pass_case),
        "state_ok": bool(state_ok),
        "state_reason": state_reason,
        "true_batch_ok": bool(true_batch_ok),
        "hard_counters_ok": bool(hard_counters_ok),
        "max_hidden_relative_l2": max_hidden_rel,
        "max_logit_relative_l2": max_logit_rel,
        "min_logit_cosine": min_logit_cos,
        "argmax_token_match_rate": token_matches / token_total if token_total else 0.0,
        "elapsed_s": batched["elapsed_s"],
        "tpot_ms": (batched["elapsed_s"] * 1000.0 / max(case.decode, 1)),
        "aggregate_tokens_s": (case.batch * case.decode / batched["elapsed_s"]) if batched["elapsed_s"] > 0 else None,
        "peak_memory_allocated": batched["peak_memory_allocated"],
        "peak_memory_reserved": batched["peak_memory_reserved"],
        "real_counters": batched["real_counters"],
        "page_counters": batched["page_counters"],
        "numerical_rows": numerical_rows,
        "cache_summary": cache_state_summary(batched["past_key_values"]),
    }


def model_config_dict(config: Any, dtype: torch.dtype) -> dict[str, Any]:
    return {
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "hidden_size": int(config.hidden_size),
        "head_dim": int(config.hidden_size // config.num_attention_heads),
        "vocab_size": int(config.vocab_size),
        "max_position_embeddings": int(config.max_position_embeddings),
        "torch_dtype": str(dtype).replace("torch.", ""),
        "rope_scaling": getattr(config, "rope_scaling", None),
        "rope_theta": float(getattr(config, "rope_theta", 0.0)),
        "k_bits": 2,
        "v_bits": 2,
        "group_size": 128,
        "sink_length": 16,
        "recent_length": 128,
        "residual_length": 128,
        "num_k_base": 32,
        "num_v_base": 32,
    }


def centroid_storage(config: Any) -> dict[str, int]:
    hkv = int(config.num_key_value_heads)
    dim = int(config.hidden_size // config.num_attention_heads)
    bases = 32
    elem = 2
    static_per_request = 2 * hkv * bases * dim * elem
    max_dynamic = 512
    dynamic_per_request = 2 * hkv * max_dynamic * dim * elem
    counter_per_request = 5 * 4 + 1
    def dynamic_for(tokens: int) -> int:
        return 2 * hkv * (tokens // 128) * dim * elem
    return {
        "static_centroid_bytes_per_request": static_per_request,
        "dynamic_centroid_bytes_per_request": dynamic_per_request,
        "counter_metadata_bytes_per_request": counter_per_request,
        "b1_total": static_per_request + dynamic_per_request + counter_per_request,
        "b2_total": 2 * (static_per_request + dynamic_per_request + counter_per_request),
        "b4_total": 4 * (static_per_request + dynamic_per_request + counter_per_request),
        "projected_4k_dynamic_per_request": dynamic_for(4096),
        "projected_16k_dynamic_per_request": dynamic_for(16384),
        "projected_32k_dynamic_per_request": dynamic_for(32768),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_reports(results: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = results["model_config"]
    cases = results["cases"]
    numerical = [row for case in cases for row in case.get("numerical_rows", [])]
    prefill_rows = [row for row in numerical if row["phase"] == "prefill"]
    decode_rows = [row for row in numerical if row["phase"] == "decode"]
    boundary_rows = [case for case in cases if case["case"]["kind"] == "boundary"]
    perf_rows = [
        {
            "case": case["case"]["name"],
            "batch": case["case"]["batch"],
            "context": case["case"]["context"],
            "decode": case["case"]["decode"],
            "elapsed_s": case.get("elapsed_s"),
            "tpot_ms": case.get("tpot_ms"),
            "aggregate_tokens_s": case.get("aggregate_tokens_s"),
        }
        for case in cases
    ]
    memory_rows = [
        {
            "case": case["case"]["name"],
            "batch": case["case"]["batch"],
            "context": case["case"]["context"],
            "decode": case["case"]["decode"],
            "peak_memory_allocated": case.get("peak_memory_allocated"),
            "peak_memory_reserved": case.get("peak_memory_reserved"),
        }
        for case in cases
    ]
    counters = {
        "real": [case.get("real_counters", {}) | {"case": case["case"]["name"]} for case in cases],
        "page": [case.get("page_counters", {}) | {"case": case["case"]["name"]} for case in cases],
    }
    all_pass = bool(cases) and all(case.get("pass") for case in cases)
    b1_pass = any(case["case"]["batch"] == 1 and case.get("pass") for case in cases)
    b2_pass = any(case["case"]["batch"] == 2 and case.get("pass") for case in cases)
    b4_pass = any(case["case"]["batch"] == 4 and case.get("pass") for case in cases)
    step_pass = {
        step: any(case["case"]["decode"] == step and case.get("pass") for case in cases)
        for step in (127, 128, 129)
    }
    max_hidden = max([case.get("max_hidden_relative_l2", 0.0) for case in cases] or [None])
    max_logit = max([case.get("max_logit_relative_l2", 0.0) for case in cases] or [None])
    min_cos = min([case.get("min_logit_cosine", 1.0) for case in cases] or [None])
    match_rates = [case.get("argmax_token_match_rate", 0.0) for case in cases]
    classification = "ACTUAL_MODEL_FIXED_BATCH_SUPPORTED" if all_pass and b1_pass and b2_pass and b4_pass and step_pass[128] and step_pass[129] else "ACTUAL_MODEL_FIXED_BATCH_INTEGRATION_BLOCKED"
    if results.get("failure"):
        failure_text = str(results.get("failure"))
        if any(marker in failure_text for marker in ("centroid", "assignments", "v_precision_mask", "v_pattern_mask", "v_assignment_idx")):
            classification = "ACTUAL_MODEL_FIXED_BATCH_STATE_DIVERGENCE"
        else:
            classification = "ACTUAL_MODEL_FIXED_BATCH_INTEGRATION_BLOCKED"
    failure_text = str(results.get("failure") or "")
    layer_match = re.search(r"layer=(\d+)", failure_text)
    component = "UNKNOWN"
    if "k_assignments" in failure_text:
        component = "K_ASSIGNMENT_STATE"
    elif "v_assignment_idx" in failure_text:
        component = "V_ASSIGNMENT_STATE"
    elif "v_precision_mask" in failure_text:
        component = "SELECTOR_PRECISION_STATE"
    elif "centroid" in failure_text:
        component = "CENTROID_STATE"
    final_gate = {
        "start_head": START_HEAD,
        "model_path": str(MODEL_PATH),
        "actual_model_loaded": bool(results.get("actual_model_loaded")),
        "synthetic_model_used_for_primary_evidence": False,
        "algorithm_changed": False,
        "quantization_changed": False,
        "selector_changed": False,
        "v4_budget_changed": False,
        "k_payload_layout_changed": False,
        "v_page_abi_changed": False,
        "fused_value_arithmetic_changed": False,
        "centroid_state_architecture_changed": bool(results.get("centroid_state_architecture_changed", False)),
        "model_num_layers": cfg.get("num_hidden_layers"),
        "model_num_attention_heads": cfg.get("num_attention_heads"),
        "model_num_key_value_heads": cfg.get("num_key_value_heads"),
        "model_head_dim": cfg.get("head_dim"),
        "b1_pass": b1_pass,
        "b2_pass": b2_pass,
        "b4_pass": b4_pass,
        "prefill_pass": bool(prefill_rows) and all(row["token_match"] for row in prefill_rows),
        "multi_step_decode_pass": bool(decode_rows) and all(case.get("pass") for case in cases if case["case"]["decode"] >= 8),
        "step_127_pass": step_pass[127],
        "step_128_pass": step_pass[128],
        "step_129_pass": step_pass[129],
        "centroid_isolation_pass": bool(cases) and all(case.get("state_ok") for case in cases),
        "selector_isolation_pass": bool(cases) and all(case.get("state_ok") for case in cases),
        "cache_isolation_pass": bool(cases) and all(case.get("state_ok") for case in cases),
        "true_batched_execution": bool(cases) and all(case.get("true_batch_ok") for case in cases if case["case"]["batch"] > 1),
        "serial_b1_dispatches": sum(case.get("real_counters", {}).get("serial_b1_dispatches", 0) for case in cases),
        "legacy_value_operator_calls": sum(case.get("real_counters", {}).get("legacy_mixed_v_operator_calls", 0) for case in cases),
        "fused_page_operator_calls": sum(case.get("real_counters", {}).get("fused_page_operator_calls", 0) for case in cases),
        "historical_v_materialization_bytes": sum(case.get("real_counters", {}).get("historical_v_materialization_bytes", 0) for case in cases),
        "page_v_materialization_bytes": sum(case.get("real_counters", {}).get("page_value_materialization_bytes", 0) for case in cases),
        "pool_full_rebuilds_per_decode": sum(case.get("real_counters", {}).get("operator_ready_pool_full_rebuilds", 0) for case in cases),
        "hot_path_gpu_item_calls": sum(case.get("real_counters", {}).get("gpu_tensor_item_calls_hot_path", 0) for case in cases),
        "python_page_dispatches": sum(case.get("real_counters", {}).get("python_page_dispatches", 0) for case in cases),
        "max_hidden_relative_l2": max_hidden,
        "max_logit_relative_l2": max_logit,
        "min_logit_cosine": min_cos,
        "argmax_token_match_rate": min(match_rates) if match_rates else None,
        "actual_model_centroid_bytes_per_request": results["centroid_storage"].get("dynamic_centroid_bytes_per_request"),
        "b1_tpot_ms": next((row["tpot_ms"] for row in perf_rows if row["batch"] == 1), None),
        "b2_tpot_ms": next((row["tpot_ms"] for row in perf_rows if row["batch"] == 2), None),
        "b4_tpot_ms": next((row["tpot_ms"] for row in perf_rows if row["batch"] == 4), None),
        "peak_memory_b1_bytes": next((row["peak_memory_allocated"] for row in memory_rows if row["batch"] == 1), None),
        "peak_memory_b2_bytes": next((row["peak_memory_allocated"] for row in memory_rows if row["batch"] == 2), None),
        "peak_memory_b4_bytes": next((row["peak_memory_allocated"] for row in memory_rows if row["batch"] == 4), None),
        "first_divergence_step": None if all_pass else results.get("first_divergence_step", "UNKNOWN"),
        "first_divergence_layer": None if all_pass else (int(layer_match.group(1)) if layer_match else "UNKNOWN"),
        "first_divergence_component": None if all_pass else component,
        "classification": classification,
        "next_task": "PATTERNKV_RAGGED_BATCH_DECODE_MVP" if classification == "ACTUAL_MODEL_FIXED_BATCH_SUPPORTED" else ("TRACE_ACTUAL_MODEL_REQUEST_STATE_DIVERGENCE" if classification == "ACTUAL_MODEL_FIXED_BATCH_STATE_DIVERGENCE" else "ACTUAL_MODEL_BATCH_INTEGRATION_REDESIGN_REVIEW"),
    }
    write_json(REPORT_DIR / "model_config.json", cfg)
    write_json(REPORT_DIR / "smoke_cases.json", [case["case"] | {"pass": case.get("pass")} for case in cases])
    write_json(REPORT_DIR / "runtime_counters.json", counters)
    write_json(REPORT_DIR / "actual_model_centroid_storage.json", results["centroid_storage"])
    write_json(REPORT_DIR / "final_gate.json", final_gate)
    write_json(REPORT_DIR / "results.json", results)
    write_csv(REPORT_DIR / "prefill_runs.csv", prefill_rows, ["case", "batch", "row", "step", "hidden_relative_l2", "logit_relative_l2", "logit_cosine", "token_match", "batched_token", "ref_token"])
    write_csv(REPORT_DIR / "decode_runs.csv", decode_rows, ["case", "batch", "row", "step", "hidden_relative_l2", "logit_relative_l2", "logit_cosine", "token_match", "batched_token", "ref_token"])
    write_csv(REPORT_DIR / "flush_boundary_runs.csv", [{"case": c["case"]["name"], "batch": c["case"]["batch"], "decode": c["case"]["decode"], "pass": c.get("pass"), "state_ok": c.get("state_ok")} for c in boundary_rows], ["case", "batch", "decode", "pass", "state_ok"])
    write_csv(REPORT_DIR / "numerical_comparison.csv", numerical, ["case", "batch", "row", "step", "phase", "hidden_relative_l2", "logit_relative_l2", "logit_cosine", "logit_max_abs", "token_match", "batched_token", "ref_token"])
    write_csv(REPORT_DIR / "memory_runs.csv", memory_rows, ["case", "batch", "context", "decode", "peak_memory_allocated", "peak_memory_reserved"])
    write_csv(REPORT_DIR / "performance_runs.csv", perf_rows, ["case", "batch", "context", "decode", "elapsed_s", "tpot_ms", "aggregate_tokens_s"])

    docs = {
        "environment.md": f"# Environment\n\nRepo: pytenter/Bounded-pattrenKV-method\n\nLocal: {REPO_ROOT}\n\nModel path: {MODEL_PATH}\n\nCUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}\n",
        "model_config.md": "# Model Config\n\n```json\n" + json.dumps(cfg, indent=2, sort_keys=True) + "\n```\n",
        "actual_model_integration_map.md": "# Actual Model Integration Map\n\n`LlamaForCausalLM_PatternKV.forward` calls `LlamaModel_PatternKV`, whose decoder layers call `LlamaFlashAttention_PatternKV`. Prefill creates PatternKV segmented caches through `build_cache_from_prefill`. Decode enters `deserialize_cache -> append_decode -> QK -> softmax -> patternkv_mixed_value_attention`; `PATTERNKV_MIXED_V_BACKEND=fused_page` dispatches to the fused page operator. Batch dimension is preserved in hidden states, cache tensors, `centroid_state_indices`, page pools, and fused Value weights.\n",
        "smoke_methodology.md": "# Smoke Methodology\n\nThe runner loads the actual DeepSeek model and compares fixed-length batched greedy decode rows against independent B1 runs for the same deterministic token sequences. Primary evidence uses real embeddings, projections, RoPE, QK, PatternKV cache, fused mixed-V Value, MLP, LM head, and logits.\n",
        "prompt_and_token_protocol.md": "# Prompt And Token Protocol\n\nFour distinct prompts are tokenized, repeated deterministically, and truncated to the requested fixed context length. Rows are not copies of one prompt.\n",
        "prefill_validation.md": f"# Prefill Validation\n\nRows: {len(prefill_rows)}\n",
        "decode_validation.md": f"# Decode Validation\n\nRows: {len(decode_rows)}\n",
        "flush_boundary_validation.md": "# Flush Boundary Validation\n\nSee `flush_boundary_runs.csv`.\n",
        "centroid_state_validation.md": "# Centroid State Validation\n\nState summaries compare centroid counts and update counts for batched rows against independent B1 cache state.\n",
        "selector_state_validation.md": "# Selector State Validation\n\nV precision mask counts are included in structural state comparison.\n",
        "cache_isolation.md": "# Cache Isolation\n\nThe same row-vs-independent B1 cache summaries are used as the fixed-batch cache isolation gate.\n",
        "true_batch_audit.md": "# True Batch Audit\n\nThe runner records fused page calls and serial B1 dispatch counters for each batched case.\n",
        "materialization_audit.md": "# Materialization Audit\n\nHistorical and page V materialization counters are recorded in `runtime_counters.json` and summarized in `final_gate.json`.\n",
        "runtime_counters.md": "# Runtime Counters\n\nSee `runtime_counters.json`.\n",
        "memory_accounting.md": "# Memory Accounting\n\nSee `memory_runs.csv`.\n",
        "performance_sanity.md": "# Performance Sanity\n\nSmoke timings are decode sanity numbers only, not formal serving benchmarks. See `performance_runs.csv`.\n",
        "failure_localization.md": f"# Failure Localization\n\nfirst_divergence_step={final_gate['first_divergence_step']}\n\nfirst_divergence_layer={final_gate['first_divergence_layer']}\n\nfirst_divergence_component={final_gate['first_divergence_component']}\n",
        "risk_analysis.md": "# Risk Analysis\n\nThe smoke is fixed-length only and intentionally does not cover ragged or continuous batching.\n",
        "final_recommendation.md": f"# Final Recommendation\n\nCLASSIFICATION={final_gate['classification']}\n\nNEXT_TASK={final_gate['next_task']}\n",
    }
    for name, text in docs.items():
        (REPORT_DIR / name).write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", choices=["stage1", "stage2", "boundary", "full"], default="stage1")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stop-on-fail", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    os.environ.setdefault("PATTERNKV_CENTROID_MAX_SLOTS", "16")
    os.environ.setdefault("PATTERNKV_CENTROID_MAX_DYNAMIC", "512")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = torch.device(args.device)
    results: dict[str, Any] = {
        "actual_model_loaded": False,
        "synthetic_model_used_for_primary_evidence": False,
        "centroid_state_architecture_changed": True,
        "cases": [],
        "failure": None,
    }
    try:
        tokenizer, config, model = load_model(dtype, device)
        results["actual_model_loaded"] = True
        results["model_config"] = model_config_dict(config, dtype)
        results["centroid_storage"] = centroid_storage(config)
        for case in smoke_cases(args.matrix):
            try:
                case_result = run_case(model, tokenizer, case, device)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                case_result = {"case": case.__dict__, "pass": False, "oom": True, "error": repr(exc)}
            except Exception as exc:
                case_result = {"case": case.__dict__, "pass": False, "error": repr(exc)}
            results["cases"].append(case_result)
            if not case_result.get("pass"):
                results["failure"] = case_result.get("error") or case_result.get("state_reason") or "case failed"
                results["first_divergence_step"] = case.decode
                results["first_divergence_component"] = "UNKNOWN"
                if args.stop_on_fail:
                    break
    except Exception as exc:
        results["failure"] = repr(exc)
        results["model_config"] = {}
        results["centroid_storage"] = {}
    write_reports(results)


if __name__ == "__main__":
    main()
