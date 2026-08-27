from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
for candidate in (
    ROOT / "vendor/transformers_4_51_runtime",
    Path(os.environ.get("QWEN3_TRANSFORMERS_VENDOR", "")),
    Path("/home/qinch2023/v100_aime24_aime25_quality_work/Bounded-pattrenKV-method/vendor/transformers_4_51_runtime"),
):
    if str(candidate) and candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (  # type: ignore
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)

from models.segmented_cache import (
    PatternQuantizedKVCache,
    append_decode,
    build_cache_from_prefill,
    build_k_segment_validity_mask,
    cache_segment_stats,
    k_segment_valid_lengths,
    request_invariant_full_value_attention,
    request_invariant_segmented_attention_softmax,
    update_value_causal_importance,
    value_precision_is_mixed,
    record_ragged_k_counter,
)
from quant.matmul import (
    cuda_attn_v_fused_with_base,
    cuda_attn_v_mixed_fused_with_base,
    cuda_bmm_fA_qB_outer,
    cuda_bmm_fA_qB_outer_with_base,
    fp16_tail_value_forward_cuda,
    fp16_tail_value_fusion_enabled,
)
from quant.page_batch import (
    get_patternkv_page_batch_counters,
    get_patternkv_real_decode_counters,
    patternkv_fused_page_batch_decode,
    record_patternkv_real_decode_counter,
    reset_patternkv_page_batch_counters,
    reset_patternkv_real_decode_counters,
)

_COUNTERS = {
    "historical_fp16_k_materialization_calls": 0,
    "historical_fp16_k_materialized_bytes": 0,
    "historical_fp16_v_materialization_calls": 0,
    "historical_fp16_v_materialized_bytes": 0,
    "page_local_v_materialization_calls": 0,
    "page_local_v_materialized_bytes": 0,
    "serial_request_forward_dispatches": 0,
    "serial_attention_dispatches": 0,
    "fallback_count": 0,
    "prefill_calls": 0,
    "prefill_tokens": 0,
    "refill_calls": 0,
    "membership_changes": 0,
}


def reset_qwen3_compressed_counters() -> None:
    for key in _COUNTERS:
        _COUNTERS[key] = 0
    reset_patternkv_page_batch_counters()
    reset_patternkv_real_decode_counters()


def get_qwen3_compressed_counters() -> dict[str, int]:
    out = dict(_COUNTERS)
    out.update({f"page_batch_{k}": int(v) for k, v in get_patternkv_page_batch_counters().items()})
    out.update({f"real_decode_{k}": int(v) for k, v in get_patternkv_real_decode_counters().items()})
    return out


class Qwen3PatternKVCompressedCache(Cache):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.layer_caches: list[PatternQuantizedKVCache | None] = [None] * int(config.num_hidden_layers)
        self.key_cache = self.layer_caches
        self.value_cache = self.layer_caches
        self._seen_tokens = 0

    def _initial_centroids(self, states: torch.Tensor, count: int) -> torch.Tensor | None:
        if count <= 0 or states.shape[2] <= 0:
            return None
        bsz, heads, tokens, dim = states.shape
        k = min(int(count), int(tokens))
        if k <= 0:
            return None
        idx = torch.linspace(0, tokens - 1, steps=k, device=states.device).round().long()
        # Request-local centroids: [B,H,M,D]. Do not flatten B into the centroid bank.
        return states.index_select(2, idx).contiguous()

    def update_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> None:
        layer_idx = int(layer_idx)
        if layer_idx == 0:
            self._seen_tokens += int(key_states.shape[2])
        cache = build_cache_from_prefill(
            key_states,
            value_states,
            sink_length=int(getattr(self.config, "sink_length", 16)),
            recent_length=int(getattr(self.config, "recent_length", 128)),
            group_size=int(getattr(self.config, "group_size", 128)),
            k_bits=int(getattr(self.config, "k_bits", 2)),
            v_bits=int(getattr(self.config, "v_bits", 2)),
            pattern=True,
            k_centroids=self._initial_centroids(key_states, int(getattr(self.config, "num_k_base", 32))),
            v_centroids=self._initial_centroids(value_states, int(getattr(self.config, "num_v_base", 32))),
            cache_mode=str(getattr(self.config, "patternkv_cache_mode", "segmented_rolling")),
            chunk_length=int(getattr(self.config, "group_size", 128)),
            value_objective=str(getattr(self.config, "patternkv_value_objective", "base")),
            v_precision_selector=str(getattr(self.config, "patternkv_v_precision_selector", "causal_v4")),
            v4_budget_fraction=float(getattr(self.config, "patternkv_v4_budget_fraction", 0.25)),
            random_selector_seed=int(getattr(self.config, "patternkv_random_selector_seed", 20260809)),
            selector_task_key=str(getattr(self.config, "patternkv_selector_task_key", "task")),
            selector_layer_idx=layer_idx,
        )
        self.layer_caches[layer_idx] = cache

    def append_decode(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> PatternQuantizedKVCache:
        layer_idx = int(layer_idx)
        cache = self.layer_caches[layer_idx]
        if cache is None:
            self.update_prefill(key_states, value_states, layer_idx)
        else:
            if layer_idx == 0:
                self._seen_tokens += int(key_states.shape[2])
            append_decode(cache, key_states, value_states)
        cache = self.layer_caches[layer_idx]
        if cache is None:
            raise RuntimeError(f"missing Qwen3 compressed cache for layer {layer_idx}")
        return cache

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        idx = 0 if layer_idx is None else int(layer_idx)
        cache = self.layer_caches[idx] if idx < len(self.layer_caches) else None
        return int(getattr(cache, "total_tokens", 0) or 0) if cache is not None else 0

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError("Qwen3 compressed PatternKV supports greedy true-batch decode, not beam reorder")



def patternkv_request_invariant_qk_scores(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    if query_states.dim() != 4 or key_states.dim() != 4:
        raise ValueError(f"expected query/key states [B,H,Q,D]/[B,Hkv,T,D], got {tuple(query_states.shape)} {tuple(key_states.shape)}")
    if query_states.shape[0] != key_states.shape[0] or query_states.shape[-1] != key_states.shape[-1]:
        raise ValueError(f"QK shape mismatch: query={tuple(query_states.shape)} key={tuple(key_states.shape)}")
    key_for_attention = repeat_kv(key_states, int(num_key_value_groups))
    products = query_states.unsqueeze(3) * key_for_attention.unsqueeze(2)
    return products.sum(dim=-1).contiguous()


def patternkv_page_value_attention(
    module: nn.Module,
    cache: PatternQuantizedKVCache,
    weights: torch.Tensor,
    *,
    attn_f: torch.Tensor | None = None,
    v_full: torch.Tensor | None = None,
) -> torch.Tensor:
    pools = getattr(cache, "operator_ready_page_pools", None)
    if pools is None:
        raise RuntimeError("page Value path requires operator-ready page pools")
    record_patternkv_real_decode_counter("real_decode_steps", 1)
    out = patternkv_fused_page_batch_decode(weights, pools)
    if attn_f is not None and v_full is not None:
        out = out + torch.matmul(attn_f, repeat_kv(v_full, int(module.num_key_value_groups)))
    return out


def patternkv_mixed_value_attention(
    module: nn.Module,
    cache: PatternQuantizedKVCache,
    weights: torch.Tensor,
    v_mask: torch.Tensor | None,
    quant_tokens: int,
    *,
    attn_f: torch.Tensor | None = None,
    v_full: torch.Tensor | None = None,
) -> torch.Tensor:
    pools = getattr(cache, "operator_ready_page_pools", None)
    if pools is not None and os.environ.get("QWEN3_COMPRESSED_V_BACKEND", "legacy_cuda").strip().lower() == "fused_page":
        return patternkv_page_value_attention(module, cache, weights, attn_f=attn_f, v_full=v_full)
    if cache.v_centroids is None or cache.v_assignment_idx is None or v_mask is None:
        raise RuntimeError("compressed mixed Value path requires centroid, assignment, and mask metadata")
    if cache.v_precision_mask is None:
        raise RuntimeError("compressed mixed Value path requires v_precision_mask")
    record_patternkv_real_decode_counter("legacy_mixed_v_operator_calls", 1)
    out = cuda_attn_v_mixed_fused_with_base(
        module.group_size,
        weights,
        cache.packed_v,
        cache.packed_v_scale,
        cache.packed_v_zero,
        cache.packed_v4,
        cache.packed_v4_scale,
        cache.packed_v4_zero,
        cache.v_precision_mask[:, :quant_tokens],
        cache.v_centroids,
        v_mask[:, :, :quant_tokens],
        cache.v_assignment_idx[:, :, :quant_tokens],
        nh=module.num_heads,
        nh_kv=module.num_key_value_heads,
        attn_f=attn_f,
        v_full=v_full,
        v2_mask_q=getattr(cache, "v2_pattern_mask", None),
        v2_idx_q=getattr(cache, "v2_assignment_idx", None),
        v4_mask_q=getattr(cache, "v4_pattern_mask", None),
        v4_idx_q=getattr(cache, "v4_assignment_idx", None),
    )
    return out


def patternkv_value_reader_fn(bits: int):
    if int(bits) != 2:
        raise RuntimeError("Qwen3 compressed backend expects frozen V bits=2")
    return cuda_attn_v_fused_with_base


def compressed_backend_counters_pass(counters: dict[str, int] | None = None) -> bool:
    counters = get_qwen3_compressed_counters() if counters is None else counters
    return (
        int(counters.get("historical_fp16_k_materialization_calls", 0)) == 0
        and int(counters.get("historical_fp16_k_materialized_bytes", 0)) == 0
        and int(counters.get("historical_fp16_v_materialization_calls", 0)) == 0
        and int(counters.get("historical_fp16_v_materialized_bytes", 0)) == 0
        and int(counters.get("serial_request_forward_dispatches", 0)) == 0
        and int(counters.get("serial_attention_dispatches", 0)) == 0
        and int(counters.get("fallback_count", 0)) == 0
    )


def _cache_value_parts(cache: PatternQuantizedKVCache) -> list[tuple[str, int]]:
    parts: list[tuple[str, int]] = []
    if cache.sink_k is not None:
        parts.append(("sink", int(cache.sink_k.shape[2])))
    if cache.packed_k is not None and int(cache.packed_k_tokens):
        parts.append(("packed", int(cache.packed_k_tokens)))
    if cache.pending_k is not None:
        parts.append(("pending", int(cache.pending_k.shape[2])))
    if cache.recent_k is not None:
        parts.append(("recent", int(cache.recent_k.shape[2])))
    return parts


def _compressed_attention(
    module: nn.Module,
    query_states: torch.Tensor,
    cache: PatternQuantizedKVCache,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, _heads, q_len, _head_dim = query_states.shape
    if q_len != 1:
        raise RuntimeError("Qwen3 compressed backend only handles decode q_len=1 after prefill")
    score_parts = []
    value_parts = []
    if cache.sink_k is not None:
        score_parts.append(patternkv_request_invariant_qk_scores(query_states, cache.sink_k, module.num_key_value_groups))
        value_parts.append(("sink", cache.sink_k.shape[2]))
    if cache.packed_k is not None and int(cache.packed_k_tokens):
        if cache.k_centroids is not None and cache.k_assignments is not None:
            packed_scores = cuda_bmm_fA_qB_outer_with_base(
                module.group_size,
                query_states,
                cache.packed_k,
                cache.packed_k_scale,
                cache.packed_k_zero,
                module.k_bits,
                cache.k_centroids,
                cache.k_assignments[:, :, : cache.packed_k_tokens],
                module.num_heads,
                module.num_key_value_heads,
            )
        else:
            packed_scores = cuda_bmm_fA_qB_outer(
                module.group_size,
                query_states,
                cache.packed_k,
                cache.packed_k_scale,
                cache.packed_k_zero,
                module.k_bits,
            )
        score_parts.append(packed_scores[:, :, :, : cache.packed_k_tokens])
        value_parts.append(("packed", cache.packed_k_tokens))
    if cache.pending_k is not None:
        score_parts.append(patternkv_request_invariant_qk_scores(query_states, cache.pending_k, module.num_key_value_groups))
        value_parts.append(("pending", cache.pending_k.shape[2]))
    if cache.recent_k is not None:
        score_parts.append(patternkv_request_invariant_qk_scores(query_states, cache.recent_k, module.num_key_value_groups))
        value_parts.append(("recent", cache.recent_k.shape[2]))
    if not score_parts:
        raise RuntimeError("empty compressed PatternKV attention cache")
    attn_weights = torch.cat(score_parts, dim=-1) * float(module.scaling)
    if attn_weights.size() != (bsz, module.num_heads, q_len, int(cache.total_tokens)):
        raise ValueError(f"Qwen3 compressed attention weight shape mismatch: {tuple(attn_weights.size())}, total={cache.total_tokens}")
    k_valid_mask = build_k_segment_validity_mask(cache, value_parts, device=attn_weights.device)
    if k_valid_mask is not None:
        record_ragged_k_counter("ragged_batch_forward_calls", 1)
        record_ragged_k_counter("ragged_requests_processed", int(bsz))
        record_ragged_k_counter("ragged_k_path_calls", 1)
        attn_weights = attn_weights.masked_fill(~k_valid_mask[:, None, None, :], torch.finfo(attn_weights.dtype).min)
    if attention_mask is not None:
        expected = (bsz, 1, q_len, int(cache.total_tokens))
        if attention_mask.size() != expected:
            if (
                attention_mask.dim() == 4
                and attention_mask.shape[0] == bsz
                and attention_mask.shape[1] == 1
                and attention_mask.shape[2] == q_len
                and attention_mask.shape[3] >= int(cache.total_tokens)
            ):
                total_tokens = int(cache.total_tokens)
                # Qwen3 may allocate one future causal-mask slot before the layer appends
                # the decode token to PatternKV. Keep the logical cache prefix and drop
                # future slots instead of right-aligning them into the cache.
                attention_mask = attention_mask[:, :, :, :total_tokens]
            else:
                raise ValueError(f"Qwen3 attention mask should be {expected}, got {tuple(attention_mask.size())}")
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights,
            torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device, dtype=attn_weights.dtype),
        )
    attn_weights = request_invariant_segmented_attention_softmax(attn_weights, cache, value_parts)
    update_value_causal_importance(cache, attn_weights)

    attn_output = None
    offset = 0
    tail_segments: dict[str, tuple[int, int]] = {}
    for name, length_value in value_parts:
        length = int(length_value)
        weights = attn_weights[:, :, :, offset : offset + length]
        if name == "packed":
            v_mask = cache.v_pattern_mask if getattr(cache, "v_pattern_mask", None) is not None else cache.v_assignments
            if value_precision_is_mixed(getattr(cache, "v_precision_selector", "base_v2")):
                part = patternkv_mixed_value_attention(module, cache, weights, v_mask, int(cache.packed_v_tokens))
            elif getattr(cache, "operator_ready_page_pools", None) is not None:
                part = patternkv_page_value_attention(module, cache, weights)
            else:
                v_reader = patternkv_value_reader_fn(module.v_bits)
                part = v_reader(
                    module.group_size,
                    weights,
                    cache.packed_v,
                    cache.packed_v_scale,
                    cache.packed_v_zero,
                    module.v_bits,
                    cache.v_centroids,
                    v_mask[:, :, : cache.packed_v_tokens],
                    cache.v_assignment_idx[:, :, : cache.packed_v_tokens],
                    nh=module.num_heads,
                    nh_kv=module.num_key_value_heads,
                )
            attn_output = part if attn_output is None else attn_output + part
        else:
            tail_segments[name] = (offset, length)
            if not fp16_tail_value_fusion_enabled():
                source = {"sink": cache.sink_v, "pending": cache.pending_v, "recent": cache.recent_v}[name]
                part = request_invariant_full_value_attention(
                    weights,
                    source,
                    k_segment_valid_lengths(cache, device=weights.device)[name],
                    module.num_key_value_groups,
                )
                attn_output = part if attn_output is None else attn_output + part
        offset += length
    if fp16_tail_value_fusion_enabled() and tail_segments:
        lengths = k_segment_valid_lengths(cache, device=attn_weights.device)
        empty_v = torch.empty((bsz, module.num_key_value_heads, 0, module.head_dim), dtype=attn_weights.dtype, device=attn_weights.device)
        sink_offset, sink_length = tail_segments.get("sink", (0, 0))
        pending_offset, pending_length = tail_segments.get("pending", (0, 0))
        recent_offset, recent_length = tail_segments.get("recent", (0, 0))
        tail = fp16_tail_value_forward_cuda(
            attn_weights[:, :, :, sink_offset : sink_offset + sink_length] if sink_length else attn_weights[:, :, :, :0],
            attn_weights[:, :, :, pending_offset : pending_offset + pending_length] if pending_length else attn_weights[:, :, :, :0],
            attn_weights[:, :, :, recent_offset : recent_offset + recent_length] if recent_length else attn_weights[:, :, :, :0],
            cache.sink_v if sink_length else empty_v,
            cache.pending_v if pending_length else empty_v,
            cache.recent_v if recent_length else empty_v,
            lengths.get("sink", torch.zeros((bsz,), dtype=torch.long, device=attn_weights.device)),
            lengths.get("pending", torch.zeros((bsz,), dtype=torch.long, device=attn_weights.device)),
            lengths.get("recent", torch.zeros((bsz,), dtype=torch.long, device=attn_weights.device)),
            int(module.num_key_value_groups),
        )
        attn_output = tail if attn_output is None else attn_output + tail
    if attn_output is None:
        raise RuntimeError("Qwen3 compressed Value path produced no output")
    return attn_output, attn_weights


class Qwen3Attention_PatternKVCompressed(Qwen3Attention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.group_size = int(getattr(config, "group_size", 128))
        self.k_bits = int(getattr(config, "k_bits", 2))
        self.v_bits = int(getattr(config, "v_bits", 2))
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        layer_cache = None
        compressed_decode = False
        if past_key_value is not None:
            if not isinstance(past_key_value, Qwen3PatternKVCompressedCache):
                raise TypeError(f"Qwen3 compressed PatternKV requires Qwen3PatternKVCompressedCache, got {type(past_key_value)!r}")
            if int(key_states.shape[2]) == 1 and past_key_value.layer_caches[self.layer_idx] is not None:
                layer_cache = past_key_value.append_decode(key_states, value_states, self.layer_idx)
                compressed_decode = True
            else:
                _COUNTERS["prefill_calls"] += 1
                _COUNTERS["prefill_tokens"] += int(key_states.shape[2])
                past_key_value.update_prefill(key_states, value_states, self.layer_idx)
                layer_cache = past_key_value.layer_caches[self.layer_idx]
        if compressed_decode:
            attn_output, attn_weights = _compressed_attention(self, query_states, layer_cache, attention_mask)
        else:
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                output_attentions=True,
            )
            if layer_cache is not None and attn_weights is not None:
                update_value_causal_importance(layer_cache, attn_weights.detach())
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights if kwargs.get("output_attentions", False) else None


class Qwen3Model_PatternKVCompressed(Qwen3Model):
    def __init__(self, config):
        super().__init__(config)
        for idx, layer in enumerate(self.layers):
            replacement = Qwen3Attention_PatternKVCompressed(config=config, layer_idx=idx)
            replacement.load_state_dict(layer.self_attn.state_dict(), strict=True)
            layer.self_attn = replacement


class Qwen3ForCausalLM_PatternKVCompressed(Qwen3ForCausalLM):
    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3Model_PatternKVCompressed(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, *args: Any, past_key_values: Optional[Cache] = None, use_cache: Optional[bool] = None, **kwargs: Any):
        if use_cache is None:
            use_cache = self.config.use_cache
        if use_cache:
            replace_cache = past_key_values is None
            if past_key_values is not None and past_key_values.__class__.__name__ == "DynamicCache":
                replace_cache = int(past_key_values.get_seq_length()) == 0
            if replace_cache:
                past_key_values = Qwen3PatternKVCompressedCache(self.config)
            elif not isinstance(past_key_values, Qwen3PatternKVCompressedCache):
                raise TypeError(f"Qwen3 compressed PatternKV refuses non-native cache {type(past_key_values)!r}")
        return super().forward(*args, past_key_values=past_key_values, use_cache=use_cache, **kwargs)


def collect_qwen3_compressed_dynamic_stats(model: nn.Module, past_key_values: Any = None) -> dict[str, Any]:
    cache = past_key_values
    if not isinstance(cache, Qwen3PatternKVCompressedCache):
        return {"backend": "qwen3_patternkv_compressed", "layers": 0, "counters": get_qwen3_compressed_counters()}
    stats = [cache_segment_stats(layer_cache) if layer_cache is not None else {} for layer_cache in cache.layer_caches]
    return {
        "backend": "qwen3_patternkv_compressed",
        "layers": len(stats),
        "cache_segment_stats_per_layer": stats,
        "counters": get_qwen3_compressed_counters(),
    }
