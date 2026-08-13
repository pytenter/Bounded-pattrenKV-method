# Actual Model Integration Map

`LlamaForCausalLM_PatternKV.forward` calls `LlamaModel_PatternKV`, whose decoder layers call `LlamaFlashAttention_PatternKV`. Prefill creates PatternKV segmented caches through `build_cache_from_prefill`. Decode enters `deserialize_cache -> append_decode -> QK -> softmax -> patternkv_mixed_value_attention`; `PATTERNKV_MIXED_V_BACKEND=fused_page` dispatches to the fused page operator. Batch dimension is preserved in hidden states, cache tensors, `centroid_state_indices`, page pools, and fused Value weights.
