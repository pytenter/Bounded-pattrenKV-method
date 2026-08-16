# Current Heterogeneous Attention Dataflow

One decode token in `LlamaFlashAttention_PatternKV.forward` follows this production path:

1. Hidden state enters Q/K/V projections. The new K/V is appended to request-local segmented cache state.
2. QK score production is segmented:
   - FP16 sink K: `patternkv_request_invariant_qk_scores(query_states, cache.sink_k, ...)`.
   - INT2 historical K: `cuda_bmm_fA_qB_outer_with_base(...)` over `packed_k`, `packed_k_scale`, `packed_k_zero`, centroid table, and K assignments.
   - FP16 pending K: same request-invariant FP16 QK helper.
   - FP16 recent K: same request-invariant FP16 QK helper.
3. The score tensors are concatenated by `torch.cat(score_parts, dim=-1)` and scaled by `sqrt(head_dim)`.
4. Ragged valid masks are applied in physical segment layout when needed.
5. `request_invariant_segmented_attention_softmax` computes one global softmax over logical request order. The CUDA fixed-split path internally keeps `merged_max` and `merged_sum`, then writes a full probability tensor in physical segment order.
6. Value consumption is segmented after global softmax:
   - historical mixed V2/V4: `patternkv_mixed_value_attention` -> `patternkv_fused_page_batch_decode` -> `attn_v_forward_cuda_page_mixed_pool`.
   - FP16 sink/pending/recent: `request_invariant_full_value_attention` for each segment.
7. Segment outputs are tensor-level summed in Python: `attn_output = part if None else attn_output + part`.
8. The merged attention output is transposed/reshaped and passed to `o_proj`.

This is implementation category C plus tensor-level Value summation: separately compute segment scores, concatenate logits, run unified softmax, compute segment Value outputs with normalized global probabilities, then sum outputs. It is not currently FlashInfer-style `(O,m,l)` state merge.
