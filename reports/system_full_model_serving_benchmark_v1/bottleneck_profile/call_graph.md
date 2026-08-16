# Full-Model CAUSAL Call Graph

Decode token path for `CAUSAL_V4_25_FULL_MODEL`:

1. Harness stacks next token and calls `PatternKVAdapter.assemble_batch`.
2. `assemble_batch` calls `assemble_ragged_patternkv_cache` once per layer.
3. Per layer, `_cat_optional_by_batch` rebuilds K/V/cache tensors and `_merge_operator_ready_page_pools` rebuilds operator-ready page metadata/pools.
4. `model(input_ids, past_key_values=...)` executes embedding, all decoder layers, final norm, and LM head.
5. Each layer executes Q/K/V projection, RoPE, `append_decode`, QK against compressed and FP16 regions, request-invariant segmented softmax, causal-importance update, fused-page mixed Value decode, FP16 tail Value, output projection, RMSNorm, and MLP.
6. Harness calls `PatternKVAdapter.split_batch`.
7. `split_batch` deserializes each layer cache and calls `extract_request_cache`; `_slice_ragged_request_cache` and `_slice_operator_ready_page_pools_for_request` create request-local cache copies.
8. The next token is selected from LM-head logits and the loop repeats.
