# V Causal Importance Call Graph

`LlamaFlashAttention_PatternKV.forward` builds segmented decode scores from sink, packed, pending, and recent K; concatenates them in that physical segment order; masks invalid ragged tails; softmaxes over the physical attention axis; then calls `update_value_causal_importance(cache, attn_weights)`. Later V candidate selection reads `cache.v_causal_importance[:, absolute_start:absolute_start + tokens]` as a logical token-indexed vector.
