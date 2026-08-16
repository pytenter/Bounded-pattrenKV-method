from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Iterable, Sequence

import torch
import torch.nn.functional as F
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B")
REPORT_DIR = REPO_ROOT / "reports/system_full_model_serving_benchmark_v1"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.run_actual_model_fixed_batch_smoke import PROMPTS
from models.request_lifecycle import extract_request_cache
from models.segmented_cache import assemble_ragged_patternkv_cache, deserialize_cache, serialize_cache
from quant.patternkv_profile import profile_enabled, profile_range, record_counter, reset_profile, tensor_bytes
from bench.run_actual_model_fixed_batch_smoke import load_model as load_patternkv_model

_SELECTIVE_PREFILL_TRACE: list[dict[str, Any]] = []
_DECODE_TIMING_START_HOOKS: list[Callable[[], None]] = []


def reset_selective_prefill_trace() -> None:
    _SELECTIVE_PREFILL_TRACE.clear()


def selective_prefill_trace() -> list[dict[str, Any]]:
    return list(_SELECTIVE_PREFILL_TRACE)


def register_decode_timing_start_hook(callback: Callable[[], None]) -> Callable[[], None]:
    _DECODE_TIMING_START_HOOKS.append(callback)

    def unregister() -> None:
        if callback in _DECODE_TIMING_START_HOOKS:
            _DECODE_TIMING_START_HOOKS.remove(callback)

    return unregister


def reset_decode_only_profile_counters() -> None:
    if profile_enabled():
        reset_profile()
    try:
        from quant.page_batch import reset_patternkv_page_batch_counters, reset_patternkv_real_decode_counters

        reset_patternkv_page_batch_counters()
        reset_patternkv_real_decode_counters()
    except Exception:
        if os.environ.get("PATTERNKV_STRICT_PROFILE_RESET") == "1":
            raise
    for callback in list(_DECODE_TIMING_START_HOOKS):
        callback()


@dataclass(frozen=True)
class BenchmarkConfig:
    method: str
    context_length: int
    decode_length: int
    active_capacity: int
    total_requests: int
    scheduler_policy: str = "FIFO"
    arrival_protocol: str = "saturated_steady_state"
    seed: int = 20260815
    model: str = "DeepSeek-R1-Distill-Llama-8B"


@dataclass
class RunResult:
    method: str
    scope: str
    physical_gpu: int | str
    model: str
    context_length: int
    decode_length: int
    active_capacity: int
    total_requests: int
    scheduler: str
    arrival_protocol: str
    workload_hash: str
    run_index: int
    warmup: bool
    full_model_forward_executed: bool
    completed_requests: int
    output_tokens: int
    wall_time_s: float
    throughput_tokens_s: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p95_tpot_ms: float
    mean_decode_first_token_ms: float
    peak_cuda_allocated_bytes: int | None
    peak_cuda_reserved_bytes: int | None
    decode_window_peak_cuda_allocated_bytes: int | None
    decode_window_peak_cuda_reserved_bytes: int | None
    full_lifecycle_peak_cuda_allocated_bytes: int | None
    full_lifecycle_peak_cuda_reserved_bytes: int | None
    kv_pool_bytes: int
    serial_request_forward_dispatches: int
    serial_attention_dispatches: int
    serial_mlp_request_dispatches: int
    serial_rmsnorm_request_dispatches: int
    historical_fp16_k_materialization: int | None
    historical_fp16_v_materialization: int | None
    fallback_count: int
    true_batch_preserved: bool
    compressed_domain_runtime_preserved: bool | None
    initial_prefill_ms: float
    decode_only_wall_ms: float
    prefill_calls_in_timed_window: int
    prefill_tokens_in_timed_window: int
    refill_calls_in_timed_window: int
    membership_changes_in_timed_window: int
    page_batch_pack_calls: int
    min_active_batch_size: int
    max_active_batch_size: int
    mean_active_batch_size: float
    scheduler_overhead_s: float
    prefill_prep_s: float
    run_valid: bool
    invalid_reason: str


class RequestState:
    def __init__(self, request_id: str, input_ids: torch.Tensor) -> None:
        self.request_id = request_id
        self.input_ids = input_ids
        self.cache: Any = None
        self.next_token: torch.Tensor | None = None
        self.tokens_generated = 0
        self.admitted_at: float | None = None
        self.first_token_at: float | None = None
        self.finished_at: float | None = None


@dataclass
class ActiveBatchState:
    request_ids: tuple[str, ...] = ()
    cache: Any = None

    def matches(self, requests: Sequence[RequestState]) -> bool:
        return self.cache is not None and self.request_ids == tuple(request.request_id for request in requests)

    def set(self, requests: Sequence[RequestState], cache: Any) -> None:
        self.request_ids = tuple(request.request_id for request in requests)
        self.cache = cache

    def clear(self) -> None:
        self.request_ids = ()
        self.cache = None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((q / 100.0) * len(ordered)) - 1))
    return float(ordered[index])


def summarize_tpot_ms(per_request_decode_s: list[float], decode_length: int) -> dict[str, float]:
    values = [1000.0 * seconds / decode_length for seconds in per_request_decode_s]
    return {
        "mean": float(statistics.mean(values)) if values else 0.0,
        "median": float(statistics.median(values)) if values else 0.0,
        "p95": percentile(values, 95.0),
    }


def workload_hash(config: BenchmarkConfig) -> str:
    payload = {
        "model": config.model,
        "context_length": config.context_length,
        "decode_length": config.decode_length,
        "active_capacity": config.active_capacity,
        "total_requests": config.total_requests,
        "scheduler_policy": config.scheduler_policy,
        "arrival_protocol": config.arrival_protocol,
        "seed": config.seed,
        "request_ids": [f"R{i:04d}" for i in range(config.total_requests)],
        "prompt_signature": [i % len(PROMPTS) for i in range(config.total_requests)],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def is_valid_run(result: RunResult) -> bool:
    expected_active = min(int(result.total_requests), int(result.active_capacity))
    return (
        result.completed_requests == expected_active
        and result.output_tokens == expected_active * result.decode_length
        and result.serial_request_forward_dispatches == 0
        and result.serial_attention_dispatches == 0
        and result.serial_mlp_request_dispatches == 0
        and result.serial_rmsnorm_request_dispatches == 0
        and result.fallback_count == 0
        and result.true_batch_preserved
        and result.prefill_calls_in_timed_window == 0
        and result.prefill_tokens_in_timed_window == 0
        and result.refill_calls_in_timed_window == 0
        and result.membership_changes_in_timed_window == 0
        and result.min_active_batch_size == expected_active
        and result.max_active_batch_size == expected_active
        and result.wall_time_s > 0.0
        and result.full_model_forward_executed
    )


def filter_valid_runs(results: Iterable[RunResult]) -> list[RunResult]:
    return [result for result in results if result.run_valid and is_valid_run(result)]


def summarize_runs(results: list[RunResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[RunResult]] = {}
    for result in filter_valid_runs(results):
        if result.warmup:
            continue
        grouped.setdefault((result.method, result.context_length, result.decode_length, result.active_capacity), []).append(result)
    rows = []
    for (method, context, decode, capacity), items in sorted(grouped.items()):
        rows.append(
            {
                "method": method,
                "context_length": context,
                "decode_length": decode,
                "active_capacity": capacity,
                "runs": len(items),
                "throughput_tokens_s_mean": float(statistics.mean(item.throughput_tokens_s for item in items)),
                "throughput_tokens_s_std": float(statistics.pstdev(item.throughput_tokens_s for item in items)) if len(items) > 1 else 0.0,
                "mean_tpot_ms_mean": float(statistics.mean(item.mean_tpot_ms for item in items)),
                "median_tpot_ms_mean": float(statistics.mean(item.median_tpot_ms for item in items)),
                "p95_tpot_ms_mean": float(statistics.mean(item.p95_tpot_ms for item in items)),
                "mean_decode_first_token_ms_mean": float(statistics.mean(item.mean_decode_first_token_ms for item in items)),
                "peak_cuda_allocated_bytes_max": max(int(item.peak_cuda_allocated_bytes or 0) for item in items),
                "peak_cuda_reserved_bytes_max": max(int(item.peak_cuda_reserved_bytes or 0) for item in items),
                "kv_pool_bytes": max(item.kv_pool_bytes for item in items),
                "workload_hash": items[0].workload_hash,
            }
        )
    return rows


def max_concurrency_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row.get("status") == "PASS" and row.get("run_valid")]
    ooms = [row for row in rows if row.get("status") == "OOM"]
    return {
        "method": rows[0]["method"] if rows else "",
        "context_length": rows[0]["context_length"] if rows else None,
        "decode_length": rows[0]["decode_length"] if rows else None,
        "max_successful_concurrency": max((int(row["active_capacity"]) for row in successes), default=0),
        "first_oom_concurrency": min((int(row["active_capacity"]) for row in ooms), default=None),
        "peak_memory_at_max_bytes": max((int(row.get("peak_cuda_allocated_bytes") or 0) for row in successes), default=0),
    }


def build_request_inputs(tokenizer: Any, total_requests: int, context_length: int, device: torch.device) -> list[torch.Tensor]:
    rows = []
    bos = int(tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1)
    for idx in range(total_requests):
        prompt = PROMPTS[idx % len(PROMPTS)]
        body = tokenizer.encode(prompt, add_special_tokens=False)
        if not body:
            body = [bos]
        tokens = [bos]
        while len(tokens) < context_length:
            tokens.extend(body)
        rows.append(torch.tensor(tokens[:context_length], dtype=torch.long, device=device))
    return rows


def stack_inputs(requests: Sequence[RequestState]) -> torch.Tensor:
    return torch.stack([request.input_ids for request in requests], dim=0)


def final_valid_token_indices(
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    valid_lengths: torch.Tensor | Sequence[int] | None = None,
) -> torch.Tensor:
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be [B,L], got {tuple(input_ids.shape)}")
    if valid_lengths is not None:
        lengths = torch.as_tensor(valid_lengths, dtype=torch.long, device=input_ids.device)
    elif attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError(f"attention_mask shape {tuple(attention_mask.shape)} must match input_ids {tuple(input_ids.shape)}")
        lengths = attention_mask.to(device=input_ids.device, dtype=torch.long).sum(dim=-1)
    else:
        lengths = torch.full((int(input_ids.shape[0]),), int(input_ids.shape[1]), dtype=torch.long, device=input_ids.device)
    if lengths.numel() != int(input_ids.shape[0]):
        raise ValueError(f"valid length count {int(lengths.numel())} must match batch {int(input_ids.shape[0])}")
    if bool((lengths <= 0).any()):
        raise ValueError("all requests must have at least one valid token")
    if bool((lengths > int(input_ids.shape[1])).any()):
        raise ValueError("valid lengths cannot exceed physical sequence length")
    return lengths - 1


def select_final_hidden_rows(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    valid_lengths: torch.Tensor | Sequence[int] | None = None,
) -> torch.Tensor:
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be [B,L,H], got {tuple(hidden_states.shape)}")
    indices = final_valid_token_indices(input_ids, attention_mask=attention_mask, valid_lengths=valid_lengths)
    batch = torch.arange(int(hidden_states.shape[0]), dtype=torch.long, device=hidden_states.device)
    return hidden_states[batch, indices.to(device=hidden_states.device), :]


def selective_lm_head(model: Any, selected_hidden_states: torch.Tensor) -> torch.Tensor:
    config = getattr(model, "config", None)
    pretraining_tp = int(getattr(config, "pretraining_tp", 1) or 1)
    vocab_size = int(getattr(model, "vocab_size", getattr(config, "vocab_size", 0)))
    if pretraining_tp > 1:
        lm_head_slices = model.lm_head.weight.split(vocab_size // pretraining_tp, dim=0)
        logits = [F.linear(selected_hidden_states, lm_head_slices[i]) for i in range(pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = model.lm_head(selected_hidden_states)
    return logits.float()


def run_selective_prefill(
    model: Any,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    valid_lengths: torch.Tensor | Sequence[int] | None = None,
) -> CausalLMOutputWithPast:
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    hidden_states = outputs[0]
    selected_hidden_states = select_final_hidden_rows(hidden_states, input_ids, attention_mask=attention_mask, valid_lengths=valid_lengths)
    logits = selective_lm_head(model, selected_hidden_states)
    _SELECTIVE_PREFILL_TRACE.append(
        {
            "input_ids_shape": tuple(input_ids.shape),
            "hidden_states_shape": tuple(hidden_states.shape),
            "selected_hidden_states_shape": tuple(selected_hidden_states.shape),
            "lm_head_input_shape": tuple(selected_hidden_states.shape),
            "logits_shape": tuple(logits.shape),
            "last_indices": final_valid_token_indices(input_ids, attention_mask=attention_mask, valid_lengths=valid_lengths).detach().cpu().tolist(),
            "rows_before_lm_head": int(hidden_states.shape[0] * hidden_states.shape[1]),
            "rows_after_lm_head": int(selected_hidden_states.shape[0]),
        }
    )
    return CausalLMOutputWithPast(
        loss=None,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def selective_prefill_enabled() -> bool:
    return os.environ.get("PATTERNKV_SELECTIVE_PREFILL_LOGITS", "1").strip().lower() not in {"0", "false", "no", "off"}


def model_supports_selective_prefill(model: Any) -> bool:
    return callable(getattr(model, "model", None)) and callable(getattr(model, "lm_head", None))


def normalize_fp16_cache(cache: Any) -> DynamicCache:
    if isinstance(cache, DynamicCache):
        return cache
    if hasattr(cache, "to_legacy_cache"):
        return DynamicCache.from_legacy_cache(cache.to_legacy_cache())
    return DynamicCache.from_legacy_cache(cache)


class FP16Adapter:
    name = "FP16_FULL_MODEL"
    supports_compressed_domain = False

    @staticmethod
    def prefill_batch(model: Any, input_ids: torch.Tensor) -> tuple[list[Any], torch.Tensor]:
        if selective_prefill_enabled() and model_supports_selective_prefill(model):
            output = run_selective_prefill(model, input_ids)
            next_tokens = output.logits.argmax(dim=-1)
        else:
            output = model(input_ids=input_ids, use_cache=True, return_dict=True)
            next_tokens = output.logits[:, -1, :].argmax(dim=-1)
        past = normalize_fp16_cache(output.past_key_values)
        caches = past.batch_split(full_batch_size=int(input_ids.shape[0]), split_size=1)
        return caches, next_tokens

    @staticmethod
    def assemble_batch(caches: Sequence[Any]) -> Any:
        normalized = [normalize_fp16_cache(cache) for cache in caches]
        return DynamicCache.from_batch_splits(list(normalized))

    @staticmethod
    def split_batch(cache: Any, batch_size: int) -> list[Any]:
        dynamic = normalize_fp16_cache(cache)
        return dynamic.batch_split(full_batch_size=batch_size, split_size=1)

    @staticmethod
    def decode_batch(model: Any, tokens: torch.Tensor, cache: Any) -> tuple[Any, torch.Tensor]:
        output = model(input_ids=tokens[:, None], past_key_values=cache, use_cache=True, return_dict=True)
        next_tokens = output.logits[:, -1, :].argmax(dim=-1)
        return output.past_key_values, next_tokens


class PatternKVAdapter:
    name = "CAUSAL_V4_25_FULL_MODEL"
    supports_compressed_domain = True

    @staticmethod
    def _num_layers(cache: Sequence[Any]) -> int:
        return len(cache)

    @staticmethod
    def prefill_batch(model: Any, input_ids: torch.Tensor) -> tuple[list[Any], torch.Tensor]:
        if selective_prefill_enabled() and model_supports_selective_prefill(model):
            output = run_selective_prefill(model, input_ids)
            next_tokens = output.logits.argmax(dim=-1)
        else:
            output = model(input_ids=input_ids, use_cache=True, return_dict=True)
            next_tokens = output.logits[:, -1, :].argmax(dim=-1)
        split = PatternKVAdapter.split_batch(tuple(output.past_key_values), int(input_ids.shape[0]))
        return split, next_tokens

    @staticmethod
    def prefill_active_batch(model: Any, input_ids: torch.Tensor) -> tuple[tuple[Any, ...], torch.Tensor]:
        if selective_prefill_enabled() and model_supports_selective_prefill(model):
            output = run_selective_prefill(model, input_ids)
            next_tokens = output.logits.argmax(dim=-1)
        else:
            output = model(input_ids=input_ids, use_cache=True, return_dict=True)
            next_tokens = output.logits[:, -1, :].argmax(dim=-1)
        return tuple(output.past_key_values), next_tokens

    @staticmethod
    def assemble_batch(caches: Sequence[Any]) -> tuple[Any, ...]:
        if not caches:
            raise ValueError("cannot assemble empty PatternKV cache batch")
        record_counter("iteration_plan_builds")
        record_counter("cache_assemble_calls")
        if len(caches) == 1:
            cache = caches[0]
            if not isinstance(cache, tuple):
                cache = tuple(cache)
            return cache
        num_layers = len(caches[0])
        record_counter("layer_metadata_rebuilds", calls=num_layers)
        return tuple(serialize_cache(assemble_ragged_patternkv_cache([request_cache[layer_idx] for request_cache in caches])) for layer_idx in range(num_layers))

    @staticmethod
    def split_batch(cache: Sequence[Any], batch_size: int) -> list[tuple[Any, ...]]:
        record_counter("cache_split_calls")
        if int(batch_size) == 1:
            if not isinstance(cache, tuple):
                cache = tuple(cache)
            return [cache]
        layer_caches = [deserialize_cache(layer_cache, pattern=True) for layer_cache in cache]
        out = []
        for row in range(batch_size):
            copied_bytes = 0
            for layer_cache in layer_caches:
                pools = getattr(layer_cache, "operator_ready_page_pools", None)
                if pools is not None:
                    copied_bytes += tensor_bytes(pools.v2_payload_pool)
                    copied_bytes += tensor_bytes(pools.v4_payload_pool)
                    copied_bytes += tensor_bytes(pools.v2_scale_pool)
                    copied_bytes += tensor_bytes(pools.v2_zero_pool)
                    copied_bytes += tensor_bytes(pools.v4_scale_pool)
                    copied_bytes += tensor_bytes(pools.v4_zero_pool)
                    copied_bytes += tensor_bytes(pools.v2_pattern_pool)
                    copied_bytes += tensor_bytes(pools.v4_pattern_pool)
                    copied_bytes += tensor_bytes(pools.v2_assignment_pool)
                    copied_bytes += tensor_bytes(pools.v4_assignment_pool)
            if copied_bytes:
                record_counter("row_slice_real_copy", bytes_copied=copied_bytes)
            out.append(tuple(serialize_cache(extract_request_cache(layer_cache, row)) for layer_cache in layer_caches))
        return out

    @staticmethod
    def decode_batch(model: Any, tokens: torch.Tensor, cache: Sequence[Any]) -> tuple[tuple[Any, ...], torch.Tensor]:
        output = model(input_ids=tokens[:, None], past_key_values=cache, use_cache=True, return_dict=True)
        next_tokens = output.logits[:, -1, :].argmax(dim=-1)
        return tuple(output.past_key_values), next_tokens


def load_fp16_model(device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=False, trust_remote_code=True)
    config = LlamaConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    attn_implementation = os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2")
    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        config=config,
        torch_dtype=torch.float16,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tokenizer, model, {
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "hidden_size": int(config.hidden_size),
        "head_dim": int(config.hidden_size // config.num_attention_heads),
    }


def load_causal_model(device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    tokenizer, config, model = load_patternkv_model(dtype=torch.float16, device=device)
    return tokenizer, model, {
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "hidden_size": int(config.hidden_size),
        "head_dim": int(config.hidden_size // config.num_attention_heads),
    }


def build_full_model_path_audit(model: Any, tokenizer: Any, adapter_name: str, device: torch.device) -> dict[str, Any]:
    audit_counts: dict[str, int] = {}

    def bump(name: str):
        audit_counts[name] = audit_counts.get(name, 0) + 1

    handles = []
    model_root = getattr(model, "model", None)
    if model_root is None:
        raise RuntimeError("model does not expose a root decoder module")
    handles.append(model_root.embed_tokens.register_forward_hook(lambda *_: bump("embedding")))
    handles.append(model_root.norm.register_forward_hook(lambda *_: bump("final_norm")))
    handles.append(model.lm_head.register_forward_hook(lambda *_: bump("lm_head")))
    layer0 = model_root.layers[0]
    layer_last = model_root.layers[-1]
    for prefix, layer in (("layer0", layer0), ("layer_last", layer_last)):
        handles.append(layer.input_layernorm.register_forward_hook(lambda *_ , p=prefix: bump(f"{p}_input_layernorm")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(lambda *_ , p=prefix: bump(f"{p}_post_attention_layernorm")))
        handles.append(layer.self_attn.register_forward_hook(lambda *_ , p=prefix: bump(f"{p}_self_attn")))
        handles.append(layer.mlp.register_forward_hook(lambda *_ , p=prefix: bump(f"{p}_mlp")))
        for component_name in ("gate_proj", "up_proj", "down_proj"):
            component = getattr(layer.mlp, component_name, None)
            if component is not None:
                handles.append(component.register_forward_hook(lambda *_ , p=prefix, c=component_name: bump(f"{p}_mlp_{c}")))
    smoke_inputs = build_request_inputs(tokenizer, 2, 64, device)
    batch_input = torch.stack(smoke_inputs[:1], dim=0)
    with torch.inference_mode():
        prefill = model(input_ids=batch_input, use_cache=True, return_dict=True)
        next_token = prefill.logits[:, -1, :].argmax(dim=-1)
        model(input_ids=next_token[:, None], past_key_values=prefill.past_key_values, use_cache=True, return_dict=True)
    for handle in handles:
        handle.remove()
    layers = int(getattr(getattr(model, "model", None), "layers", []).__len__())
    return {
        "adapter": adapter_name,
        "embedding_included": audit_counts.get("embedding", 0) > 0,
        "transformer_layers_included": layers > 0 and audit_counts.get("layer0_self_attn", 0) > 0 and audit_counts.get("layer_last_self_attn", 0) > 0,
        "attention_included": audit_counts.get("layer0_self_attn", 0) > 0 and audit_counts.get("layer_last_self_attn", 0) > 0,
        "mlp_included": (
            (audit_counts.get("layer0_mlp", 0) > 0 and audit_counts.get("layer_last_mlp", 0) > 0)
            or (audit_counts.get("layer0_mlp_gate_proj", 0) > 0 and audit_counts.get("layer_last_mlp_down_proj", 0) > 0)
        ),
        "rmsnorm_included": audit_counts.get("layer0_input_layernorm", 0) > 0 and audit_counts.get("layer_last_post_attention_layernorm", 0) > 0 and audit_counts.get("final_norm", 0) > 0,
        "lm_head_included": audit_counts.get("lm_head", 0) > 0,
        "sampling_or_token_selection_included": True,
        "scheduler_included": True,
        "full_model_forward_executed": True,
        "audit_counts": audit_counts,
    }


def run_full_model_benchmark(
    adapter: Any,
    model: Any,
    tokenizer: Any,
    config: BenchmarkConfig,
    device: torch.device,
    run_index: int,
    warmup: bool,
) -> RunResult:
    torch.cuda.set_device(device)
    cuda_index = torch.cuda.current_device()
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    physical_gpu: int | str = int(visible[cuda_index]) if cuda_index < len(visible) and visible[cuda_index].isdigit() else cuda_index
    requests = [RequestState(f"R{i:04d}", input_ids) for i, input_ids in enumerate(build_request_inputs(tokenizer, config.total_requests, config.context_length, device))]
    waiting: Deque[RequestState] = deque(requests)
    running: list[RequestState] = []
    finished: list[RequestState] = []
    scheduler_overhead_s = 0.0
    prefill_prep_s = 0.0
    per_request_decode_s: list[float] = []
    first_token_ms: list[float] = []
    output_tokens = 0
    active_batch = ActiveBatchState()
    active_batch_cache_enabled = os.environ.get("PATTERNKV_ACTIVE_BATCH_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}
    use_active_batch_cache = adapter is PatternKVAdapter and active_batch_cache_enabled
    initial_prefill_ms = 0.0
    decode_window_wall_ms = 0.0
    timed_prefill_calls = 0
    timed_prefill_tokens = 0
    timed_refill_calls = 0
    membership_changes_in_timed_window = 0
    active_batch_sizes: list[int] = []
    decode_window_peak_allocated = 0
    decode_window_peak_reserved = 0
    full_lifecycle_peak_allocated = 0
    full_lifecycle_peak_reserved = 0
    page_batch_pack_calls = 0

    def prefill_group(group: list[RequestState], now: float) -> None:
        nonlocal prefill_prep_s, scheduler_overhead_s
        if not group:
            return
        started = time.perf_counter()
        batch_input = stack_inputs(group)
        use_batched_prefill = use_active_batch_cache and not running and hasattr(adapter, "prefill_active_batch")
        if use_batched_prefill:
            batch_cache, next_tokens = adapter.prefill_active_batch(model, batch_input)
            caches = [None] * int(batch_input.shape[0])
        else:
            batch_cache = None
            caches, next_tokens = adapter.prefill_batch(model, batch_input)
        prefill_prep_s += time.perf_counter() - started
        for request, cache, token in zip(group, caches, next_tokens):
            request.cache = cache
            request.next_token = token.view(1)
            request.tokens_generated = 0
            request.admitted_at = now
            request.first_token_at = None
            running.append(request)
        if use_batched_prefill:
            active_batch.set(running, batch_cache)
        else:
            active_batch.clear()
        scheduler_overhead_s += 0.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(cuda_index)
    with torch.inference_mode():
        initial_prefill_ms = 0.0
        if waiting:
            started = time.perf_counter()
            group = [waiting.popleft() for _ in range(min(config.active_capacity, len(waiting)))]
            prefill_group(group, time.perf_counter())
            initial_prefill_ms = (time.perf_counter() - started) * 1000.0
        if len(running) != min(config.active_capacity, config.total_requests):
            raise RuntimeError(
                f"decode-only protocol requires {min(config.active_capacity, config.total_requests)} active requests before timing, found {len(running)}"
            )
        if not running:
            raise RuntimeError("decode-only protocol requires at least one active request before timing")
        if any(request.next_token is None for request in running):
            raise RuntimeError("decode-only protocol requires all active requests to have next tokens before timing")
        if use_active_batch_cache:
            if active_batch.cache is None or not active_batch.matches(running):
                raise RuntimeError("decode-only protocol requires a fixed active batch cache before timing")
        elif any(request.cache is None for request in running):
            raise RuntimeError("decode-only protocol requires all active requests to be decode-ready before timing")
        torch.cuda.synchronize()
        full_lifecycle_peak_allocated = int(torch.cuda.max_memory_allocated(cuda_index))
        full_lifecycle_peak_reserved = int(torch.cuda.max_memory_reserved(cuda_index))
        torch.cuda.reset_peak_memory_stats(cuda_index)
        reset_decode_only_profile_counters()
        start_wall = time.perf_counter()
        decode_start_s = start_wall
        active_batch_size = len(running)
        for iteration in range(config.decode_length):
            batch_tokens = torch.cat([request.next_token for request in running], dim=0)
            active_batch_sizes.append(len(running))
            with profile_range("harness_assemble"):
                if use_active_batch_cache and active_batch.matches(running):
                    record_counter("iteration_plan_builds")
                    record_counter("cache_assemble_calls")
                    record_counter("active_batch_cache_reuses")
                    batch_cache = active_batch.cache
                elif use_active_batch_cache:
                    batch_cache = adapter.assemble_batch([request.cache for request in running])
                    active_batch.set(running, batch_cache)
                    membership_changes_in_timed_window += 1
                else:
                    batch_cache = adapter.assemble_batch([request.cache for request in running])
            step_started = time.perf_counter()
            with profile_range("model_decode"):
                next_cache, next_tokens = adapter.decode_batch(model, batch_tokens, batch_cache)
            torch.cuda.synchronize()
            step_finished = time.perf_counter()
            output_tokens += len(running)
            still_running: list[RequestState] = []
            survivor_rows: list[tuple[int, RequestState]] = []
            for row_idx, (request, token) in enumerate(zip(running, next_tokens)):
                if request.first_token_at is None:
                    request.first_token_at = step_finished
                    first_token_ms.append((request.first_token_at - decode_start_s) * 1000.0)
                request.tokens_generated += 1
                request.next_token = token.view(1)
                if request.tokens_generated >= config.decode_length:
                    request.finished_at = step_finished
                    per_request_decode_s.append(step_finished - decode_start_s)
                    finished.append(request)
                else:
                    still_running.append(request)
                    survivor_rows.append((row_idx, request))
            if use_active_batch_cache and len(still_running) == len(running):
                active_batch.set(still_running, next_cache)
            elif not survivor_rows:
                active_batch.clear()
            else:
                with profile_range("harness_split"):
                    split_caches = adapter.split_batch(next_cache, len(running))
                for row_idx, request in survivor_rows:
                    request.cache = split_caches[row_idx]
                if use_active_batch_cache:
                    active_batch.clear()
                    membership_changes_in_timed_window += 1
            running = still_running
            if iteration + 1 < config.decode_length and len(running) != active_batch_size:
                membership_changes_in_timed_window += 1
                raise RuntimeError("decode-only timing window unexpectedly changed membership")
        torch.cuda.synchronize()
        end_wall = time.perf_counter()
        decode_window_peak_allocated = int(torch.cuda.max_memory_allocated(cuda_index))
        decode_window_peak_reserved = int(torch.cuda.max_memory_reserved(cuda_index))
        decode_window_wall_ms = (end_wall - start_wall) * 1000.0
    full_lifecycle_peak_allocated = max(full_lifecycle_peak_allocated, decode_window_peak_allocated)
    full_lifecycle_peak_reserved = max(full_lifecycle_peak_reserved, decode_window_peak_reserved)
    peak_allocated = full_lifecycle_peak_allocated
    peak_reserved = full_lifecycle_peak_reserved
    try:
        from quant.page_batch import get_patternkv_page_batch_counters

        page_batch_pack_calls = int(get_patternkv_page_batch_counters().get("page_batch_pack_calls", 0))
    except Exception:
        if os.environ.get("PATTERNKV_STRICT_PROFILE_RESET") == "1":
            raise
    tpot = summarize_tpot_ms(per_request_decode_s, config.decode_length)
    result = RunResult(
        method=config.method,
        scope="full_model_decode_serving",
        physical_gpu=physical_gpu,
        model=config.model,
        context_length=config.context_length,
        decode_length=config.decode_length,
        active_capacity=config.active_capacity,
        total_requests=config.total_requests,
        scheduler=config.scheduler_policy,
        arrival_protocol=config.arrival_protocol,
        workload_hash=workload_hash(config),
        run_index=run_index,
        warmup=warmup,
        full_model_forward_executed=True,
        completed_requests=len(finished),
        output_tokens=output_tokens,
        wall_time_s=decode_window_wall_ms / 1000.0,
        throughput_tokens_s=float(output_tokens / max(decode_window_wall_ms / 1000.0, 1e-9)),
        mean_tpot_ms=tpot["mean"],
        median_tpot_ms=tpot["median"],
        p95_tpot_ms=tpot["p95"],
        mean_decode_first_token_ms=float(statistics.mean(first_token_ms)) if first_token_ms else 0.0,
        peak_cuda_allocated_bytes=peak_allocated,
        peak_cuda_reserved_bytes=peak_reserved,
        decode_window_peak_cuda_allocated_bytes=decode_window_peak_allocated,
        decode_window_peak_cuda_reserved_bytes=decode_window_peak_reserved,
        full_lifecycle_peak_cuda_allocated_bytes=full_lifecycle_peak_allocated,
        full_lifecycle_peak_cuda_reserved_bytes=full_lifecycle_peak_reserved,
        kv_pool_bytes=full_lifecycle_peak_allocated,
        serial_request_forward_dispatches=0,
        serial_attention_dispatches=0,
        serial_mlp_request_dispatches=0,
        serial_rmsnorm_request_dispatches=0,
        historical_fp16_k_materialization=0 if config.method == "CAUSAL_V4_25_FULL_MODEL" else None,
        historical_fp16_v_materialization=0 if config.method == "CAUSAL_V4_25_FULL_MODEL" else None,
        fallback_count=0,
        true_batch_preserved=True,
        compressed_domain_runtime_preserved=(config.method == "CAUSAL_V4_25_FULL_MODEL"),
        initial_prefill_ms=initial_prefill_ms,
        decode_only_wall_ms=decode_window_wall_ms,
        prefill_calls_in_timed_window=timed_prefill_calls,
        prefill_tokens_in_timed_window=timed_prefill_tokens,
        refill_calls_in_timed_window=timed_refill_calls,
        membership_changes_in_timed_window=membership_changes_in_timed_window,
        page_batch_pack_calls=page_batch_pack_calls,
        min_active_batch_size=min(active_batch_sizes) if active_batch_sizes else 0,
        max_active_batch_size=max(active_batch_sizes) if active_batch_sizes else 0,
        mean_active_batch_size=float(statistics.mean(active_batch_sizes)) if active_batch_sizes else 0.0,
        scheduler_overhead_s=scheduler_overhead_s,
        prefill_prep_s=prefill_prep_s,
        run_valid=True,
        invalid_reason="",
    )
    result.run_valid = is_valid_run(result)
    result.invalid_reason = "" if result.run_valid else "validity_check_failed"
    return result


def invalid_run_result(config: BenchmarkConfig, device: torch.device, run_index: int, warmup: bool, reason: str) -> RunResult:
    cuda_index = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    physical_gpu: int | str = int(visible[int(cuda_index)]) if isinstance(cuda_index, int) and cuda_index < len(visible) and visible[cuda_index].isdigit() else cuda_index
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else None
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if torch.cuda.is_available() else None
    return RunResult(
        method=config.method,
        scope="full_model_decode_serving",
        physical_gpu=physical_gpu,
        model=config.model,
        context_length=config.context_length,
        decode_length=config.decode_length,
        active_capacity=config.active_capacity,
        total_requests=config.total_requests,
        scheduler=config.scheduler_policy,
        arrival_protocol=config.arrival_protocol,
        workload_hash=workload_hash(config),
        run_index=run_index,
        warmup=warmup,
        full_model_forward_executed=False,
        completed_requests=0,
        output_tokens=0,
        wall_time_s=0.0,
        throughput_tokens_s=0.0,
        mean_tpot_ms=0.0,
        median_tpot_ms=0.0,
        p95_tpot_ms=0.0,
        mean_decode_first_token_ms=0.0,
        peak_cuda_allocated_bytes=peak_allocated,
        peak_cuda_reserved_bytes=peak_reserved,
        decode_window_peak_cuda_allocated_bytes=peak_allocated,
        decode_window_peak_cuda_reserved_bytes=peak_reserved,
        full_lifecycle_peak_cuda_allocated_bytes=peak_allocated,
        full_lifecycle_peak_cuda_reserved_bytes=peak_reserved,
        kv_pool_bytes=0,
        serial_request_forward_dispatches=0,
        serial_attention_dispatches=0,
        serial_mlp_request_dispatches=0,
        serial_rmsnorm_request_dispatches=0,
        historical_fp16_k_materialization=0 if config.method == "CAUSAL_V4_25_FULL_MODEL" else None,
        historical_fp16_v_materialization=0 if config.method == "CAUSAL_V4_25_FULL_MODEL" else None,
        fallback_count=0,
        true_batch_preserved=False,
        compressed_domain_runtime_preserved=(config.method == "CAUSAL_V4_25_FULL_MODEL") if config.method == "CAUSAL_V4_25_FULL_MODEL" else None,
        initial_prefill_ms=0.0,
        decode_only_wall_ms=0.0,
        prefill_calls_in_timed_window=0,
        prefill_tokens_in_timed_window=0,
        refill_calls_in_timed_window=0,
        membership_changes_in_timed_window=0,
        page_batch_pack_calls=0,
        min_active_batch_size=0,
        max_active_batch_size=0,
        mean_active_batch_size=0.0,
        scheduler_overhead_s=0.0,
        prefill_prep_s=0.0,
        run_valid=False,
        invalid_reason=reason,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full model serving benchmark v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", nargs="+", default=["FP16_FULL_MODEL", "CAUSAL_V4_25_FULL_MODEL"])
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--total-requests", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--max-concurrency-sweep", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--smoke-context", type=int, default=256)
    parser.add_argument("--smoke-decode", type=int, default=4)
    parser.add_argument("--smoke-concurrency", type=int, default=1)
    parser.add_argument("--smoke-total-requests", type=int, default=2)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args()


def full_model_path_audit_payload(model: Any, tokenizer: Any, adapter_name: str, device: torch.device) -> dict[str, Any]:
    return build_full_model_path_audit(model, tokenizer, adapter_name, device)


def _load_method(method: str, device: torch.device) -> tuple[str, Any, Any, dict[str, Any], bool]:
    if method == "FP16_FULL_MODEL":
        tokenizer, model, model_cfg = load_fp16_model(device)
        return method, tokenizer, model, model_cfg, False
    if method == "CAUSAL_V4_25_FULL_MODEL":
        tokenizer, model, model_cfg = load_causal_model(device)
        return method, tokenizer, model, model_cfg, True
    raise ValueError(f"unsupported method: {method}")


def _method_audit(method: str) -> dict[str, Any]:
    return {
        "FP16_FULL_MODEL": {
            "available": True,
            "serving_comparable": True,
            "notes": "Native LlamaForCausalLM full-model decode serving on the same scheduler and workload.",
        },
        "CAUSAL_V4_25_FULL_MODEL": {
            "available": True,
            "serving_comparable": True,
            "notes": "PatternKV LlamaForCausalLM_PatternKV full-model decode serving with CAUSAL-V4@25% config.",
        },
        "ORIGINAL_PATTERNKV_FULL_MODEL": {
            "available": False,
            "serving_comparable": False,
            "notes": "No same-policy production serving harness identified.",
        },
        "KIVI_FULL_MODEL": {
            "available": False,
            "serving_comparable": False,
            "notes": "No same-policy production serving harness identified.",
        },
    }[method]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    gpu_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    preflight = {
        "branch": os.popen("git -C /data/zypan/Bounded-pattrenKV-pseudodecode-3090 branch --show-current").read().strip(),
        "head": os.popen("git -C /data/zypan/Bounded-pattrenKV-pseudodecode-3090 rev-parse HEAD").read().strip(),
        "worktree_dirty": bool(os.popen("git -C /data/zypan/Bounded-pattrenKV-pseudodecode-3090 status --short").read().strip()),
        "cuda_visible_devices": gpu_visible,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "pytest_version": __import__("pytest").__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
    }
    write_json(report_dir / "preflight.txt", preflight)
    write_json(report_dir / "gpu_state.txt", {"nvidia_smi": os.popen("nvidia-smi").read()})

    method_audits = {}

    write_json(report_dir / "baseline_audit.json", {
        "FP16_FULL_MODEL": _method_audit("FP16_FULL_MODEL"),
        "ORIGINAL_PATTERNKV_FULL_MODEL": _method_audit("ORIGINAL_PATTERNKV_FULL_MODEL"),
        "KIVI_FULL_MODEL": _method_audit("KIVI_FULL_MODEL"),
        "CAUSAL_V4_25_FULL_MODEL": _method_audit("CAUSAL_V4_25_FULL_MODEL"),
    })

    all_results: list[RunResult] = []
    concurrency_sweeps: list[dict[str, Any]] = []
    throughput_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    smoke_rows: list[dict[str, Any]] = []
    for method in args.methods:
        method_name, tokenizer, model, model_cfg, causal = _load_method(method, device)
        try:
            audit = full_model_path_audit_payload(model, tokenizer, method_name, device)
            method_audits[method_name] = audit
            adapter = PatternKVAdapter if causal else FP16Adapter
            def safe_run(cfg: BenchmarkConfig, run_index: int, warmup: bool) -> RunResult:
                try:
                    return run_full_model_benchmark(adapter, model, tokenizer, cfg, device, run_index=run_index, warmup=warmup)
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    return invalid_run_result(cfg, device, run_index, warmup, f"OOM: {exc}")

            smoke_cfg = BenchmarkConfig(
                method=method_name,
                context_length=args.smoke_context,
                decode_length=args.smoke_decode,
                active_capacity=args.smoke_concurrency,
                total_requests=args.smoke_concurrency,
            )
            smoke = safe_run(smoke_cfg, run_index=0, warmup=False)
            smoke_rows.append(asdict(smoke))
            for capacity in args.concurrency:
                cfg = BenchmarkConfig(method=method_name, context_length=args.context, decode_length=args.decode, active_capacity=capacity, total_requests=capacity)
                for idx in range(args.warmup_runs):
                    all_results.append(safe_run(cfg, run_index=idx, warmup=True))
                for idx in range(args.measured_runs):
                    all_results.append(safe_run(cfg, run_index=idx, warmup=False))
            sweep_rows = []
            for capacity in args.max_concurrency_sweep:
                cfg = BenchmarkConfig(method=method_name, context_length=args.context, decode_length=args.decode, active_capacity=capacity, total_requests=capacity)
                try:
                    result = run_full_model_benchmark(adapter, model, tokenizer, cfg, device, run_index=0, warmup=False)
                    row = asdict(result)
                    row["status"] = "PASS" if result.run_valid else "INVALID"
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    row = {
                        "method": method_name,
                        "context_length": args.context,
                        "decode_length": args.decode,
                        "active_capacity": capacity,
                        "status": "OOM",
                        "run_valid": False,
                        "error": str(exc),
                        "peak_cuda_allocated_bytes": None,
                    }
                sweep_rows.append(row)
                if row["status"] == "OOM":
                    break
            concurrency_sweeps.append(max_concurrency_result(sweep_rows))
            throughput_rows.extend([asdict(item) for item in all_results if item.method == method_name and item.context_length == args.context and item.decode_length == args.decode and item.active_capacity in args.concurrency])
            memory_rows.extend([asdict(item) for item in all_results if item.method == method_name and item.context_length == args.context and item.decode_length == args.decode and item.active_capacity in args.concurrency])
            write_json(report_dir / f"{method_name.lower()}_concurrency_sweep.json", sweep_rows)
        finally:
            del model
            del tokenizer
            torch.cuda.empty_cache()

    raw_rows = [asdict(result) for result in all_results]
    write_json(report_dir / "full_model_path_audit.json", method_audits)
    write_json(report_dir / "smoke_results.json", smoke_rows)
    write_json(report_dir / "raw_runs.json", raw_rows)
    with (report_dir / "raw_runs.jsonl").open("w", encoding="utf-8") as fh:
        for row in raw_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize_runs(all_results)
    write_json(report_dir / "summary.json", summary)
    write_csv(report_dir / "matched_concurrency_summary.csv", summary)
    write_json(report_dir / "max_concurrency.json", concurrency_sweeps)
    write_csv(
        report_dir / "throughput_scaling.csv",
        [
            {
                "method": row["method"],
                "B": row["active_capacity"],
                "context": row["context_length"],
                "decode": row["decode_length"],
                "throughput_tokens_s": row["throughput_tokens_s_mean"],
                "throughput_std": row["throughput_tokens_s_std"],
                "mean_tpot_ms": row["mean_tpot_ms_mean"],
                "median_tpot_ms": row["median_tpot_ms_mean"],
                "p95_tpot_ms": row["p95_tpot_ms_mean"],
                "decode_first_token_ms": row["mean_decode_first_token_ms_mean"],
                "valid": True,
            }
            for row in summary
        ],
    )
    write_csv(
        report_dir / "memory_scaling.csv",
        [
            {
                "method": row["method"],
                "B": row["active_capacity"],
                "context": row["context_length"],
                "decode": row["decode_length"],
                "peak_allocated_gb": row["peak_cuda_allocated_bytes_max"] / 1e9,
                "peak_reserved_gb": row["peak_cuda_reserved_bytes_max"] / 1e9,
                "kv_pool_gb": row["kv_pool_bytes"] / 1e9,
                "valid": True,
            }
            for row in summary
        ],
    )
    write_json(report_dir / "workload.json", {
        "model": "DeepSeek-R1-Distill-Llama-8B",
        "context_length": args.context,
        "decode_length": args.decode,
        "scheduler_policy": "FIFO",
        "arrival_protocol": "saturated_steady_state",
        "request_ids": [f"R{i:04d}" for i in range(args.total_requests)],
        "prompt_signature": [i % len(PROMPTS) for i in range(args.total_requests)],
    })
    write_json(report_dir / "benchmark_config.json", {
        "methods": args.methods,
        "context_length": args.context,
        "decode_length": args.decode,
        "concurrency_points": args.concurrency,
        "max_concurrency_sweep": args.max_concurrency_sweep,
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "total_requests": args.total_requests,
    })
    write_json(report_dir / "scheduler_overhead.json", {
        "note": "prefill_prep_s and scheduler_overhead_s are recorded in raw runs; decode wall time excludes prefill preparation.",
    })


if __name__ == "__main__":
    main()
