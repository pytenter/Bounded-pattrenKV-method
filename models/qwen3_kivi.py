from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor/transformers_4_51_runtime"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (  # type: ignore
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3PreTrainedModel,
    Qwen3Model,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from models.segmented_cache import (
    QuantizedKVCache,
    append_decode,
    build_cache_from_prefill,
    cache_segment_stats,
    reconstruct_full_k,
    reconstruct_full_v,
)


class Qwen3KIVICache(Cache):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.layer_caches: list[QuantizedKVCache | None] = [None] * int(config.num_hidden_layers)
        self.full_key_cache: list[torch.Tensor | None] = [None] * int(config.num_hidden_layers)
        self.full_value_cache: list[torch.Tensor | None] = [None] * int(config.num_hidden_layers)
        self.key_cache = self.layer_caches
        self.value_cache = self.layer_caches
        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer_idx = int(layer_idx)
        q_tokens = int(key_states.shape[2])
        if layer_idx == 0:
            self._seen_tokens += q_tokens
        if bool(getattr(self.config, "kivi_disable_quantization", False)):
            old_k = self.full_key_cache[layer_idx]
            old_v = self.full_value_cache[layer_idx]
            full_k = key_states if old_k is None else torch.cat([old_k, key_states], dim=2).contiguous()
            full_v = value_states if old_v is None else torch.cat([old_v, value_states], dim=2).contiguous()
            self.full_key_cache[layer_idx] = full_k
            self.full_value_cache[layer_idx] = full_v
            return full_k, full_v
        cache = self.layer_caches[layer_idx]
        if cache is None:
            cache = build_cache_from_prefill(
                key_states,
                value_states,
                sink_length=int(getattr(self.config, "sink_length", 0)),
                recent_length=int(getattr(self.config, "recent_length", 128)),
                group_size=int(getattr(self.config, "group_size", 128)),
                k_bits=int(getattr(self.config, "k_bits", 2)),
                v_bits=int(getattr(self.config, "v_bits", 2)),
                pattern=False,
                cache_mode=str(getattr(self.config, "kivi_cache_mode", "segmented_rolling")),
                chunk_length=int(getattr(self.config, "group_size", 128)),
            )
            self.layer_caches[layer_idx] = cache
            return key_states, value_states
        append_decode(cache, key_states, value_states)
        full_k = reconstruct_full_k(cache)
        full_v = reconstruct_full_v(cache)
        if full_k is None or full_v is None:
            raise RuntimeError(f"failed to reconstruct KIVI cache for layer {layer_idx}")
        return full_k, full_v

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        idx = 0 if layer_idx is None else int(layer_idx)
        if bool(getattr(self.config, "kivi_disable_quantization", False)):
            value = self.full_key_cache[idx] if idx < len(self.full_key_cache) else None
            return int(value.shape[2]) if torch.is_tensor(value) else 0
        cache = self.layer_caches[idx] if idx < len(self.layer_caches) else None
        return int(getattr(cache, "total_tokens", 0) or 0) if cache is not None else 0

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError("Qwen3 KIVI cache currently supports batch_size=1 sampling only")


class Qwen3Attention_KIVI(Qwen3Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            if not isinstance(past_key_value, Qwen3KIVICache):
                raise TypeError(f"Qwen3 KIVI requires Qwen3KIVICache, got {type(past_key_value)!r}")
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, {"cache_position": cache_position})
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            output_attentions=bool(kwargs.get("output_attentions", False)),
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights if kwargs.get("output_attentions", False) else None


class Qwen3Model_KIVI(Qwen3Model):
    def __init__(self, config):
        super().__init__(config)
        for idx, layer in enumerate(self.layers):
            replacement = Qwen3Attention_KIVI(config=config, layer_idx=idx)
            replacement.load_state_dict(layer.self_attn.state_dict(), strict=True)
            layer.self_attn = replacement


class Qwen3ForCausalLM_KIVI(Qwen3ForCausalLM):
    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3Model_KIVI(config)
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
                past_key_values = Qwen3KIVICache(self.config)
            elif not isinstance(past_key_values, Qwen3KIVICache):
                raise TypeError(f"Qwen3 KIVI refuses non-native cache {type(past_key_values)!r}")
        return super().forward(*args, past_key_values=past_key_values, use_cache=use_cache, **kwargs)


def collect_qwen3_kivi_dynamic_stats(model: nn.Module, past_key_values: Any = None) -> dict[str, Any]:
    cache = past_key_values
    if not isinstance(cache, Qwen3KIVICache):
        return {"backend": "qwen3_kivi", "layers": 0}
    if bool(getattr(cache.config, "kivi_disable_quantization", False)):
        return {"backend": "qwen3_kivi", "quantization_disabled": True, "layers": len(cache.full_key_cache)}
    stats = []
    for layer_cache in cache.layer_caches:
        stats.append(cache_segment_stats(layer_cache) if layer_cache is not None else {})
    return {"backend": "qwen3_kivi", "layers": len(stats), "cache_segment_stats_per_layer": stats}
