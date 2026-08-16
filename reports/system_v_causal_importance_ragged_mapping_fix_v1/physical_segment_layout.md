# Attention Physical Segment Layout

ATTENTION_PHYSICAL_LAYOUT_CONTRACT

File: `models/llama_patternkv.py`, function `LlamaFlashAttention_PatternKV.forward`, segmented decode branch.

Physical attention axis is built in this production order:

1. `sink`: `cache.sink_k.shape[2]`
2. `packed`: `cache.packed_k_tokens`
3. `pending`: `cache.pending_k.shape[2]`
4. `recent`: `cache.recent_k.shape[2]`

The tensor consumed by `update_value_causal_importance` has shape `[B, QH, Q, cache.total_tokens]` after softmax. For ragged batches, `cache.total_tokens` is the batch maximum, while request-local valid lengths come from `k_segment_valid_lengths(cache)` and `get_total_tokens_per_request(cache)`. Invalid per-row segment tails are masked before softmax, but still occupy physical offsets.
