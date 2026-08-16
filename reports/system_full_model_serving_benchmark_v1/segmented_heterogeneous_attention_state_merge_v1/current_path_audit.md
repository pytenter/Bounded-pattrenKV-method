# Current Path Audit

## Entrypoints

- Python attention entrypoint: `models/llama_patternkv.py`, segmented PatternKV decode path in `LlamaAttention_PatternKV.forward`.
- FP16 QK entrypoint: `patternkv_request_invariant_qk_scores(query_states, segment_k, num_key_value_groups)`.
- Historical INT2 QK entrypoints: `cuda_bmm_fA_qB_outer_with_base` and `cuda_bmm_fA_qB_outer` from `quant.matmul`.
- Fixed-split softmax entrypoint: `request_invariant_segmented_attention_softmax` in `models/segmented_cache.py`; CUDA path through `request_invariant_fixed_split_softmax_cuda`.
- Mixed historical Value entrypoint: `patternkv_mixed_value_attention`, dispatching to `patternkv_fused_page_batch_decode` for the fused page backend.
- FP16 Value entrypoint: `request_invariant_full_value_attention`.

## Current ABI

- `CURRENT_SCORE_ABI = score_parts: list[[B,Hq,Q,T_i]] in physical order sink, packed/history, pending, recent; then torch.cat(..., dim=-1) -> [B,Hq,Q,total_physical]`.
- `CURRENT_SOFTMAX_ABI = request_invariant_segmented_attention_softmax([B,Hq,Q,total], cache, value_parts) -> full global probabilities [B,Hq,Q,total]`.
- `CURRENT_VALUE_ABI = value backends consume probability slices [B,Hq,Q,T_i] and return normalized outputs [B,Hq,Q,D]`.
- `CURRENT_MERGE_ABI = Python tensor-level sum of per-segment normalized Value outputs`.

## Temporaries

- Old path materializes a global concatenated score tensor and a global normalized probability tensor.
- Historical K and V remain compressed-domain; no full historical FP16 K/V materialization was found or introduced.
- Ragged validity uses `k_segment_valid_lengths` and request-local physical-to-logical segment mapping.

## Target ABI

- `TARGET_STATE_ABI = SegmentedAttentionState(o,m,l)`.
- `o: [B,Hq,Q,D], float32`; `m: [B,Hq,Q], float32`; `l: [B,Hq,Q], float32`.
- Final output is `o / l` cast back to query dtype before the output projection.
- Deterministic merge order is sink -> packed/history -> pending -> recent.

