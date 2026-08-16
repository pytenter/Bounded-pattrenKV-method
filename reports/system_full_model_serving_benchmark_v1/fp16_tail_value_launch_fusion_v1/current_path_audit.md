# Current Path Audit

## Scope

This audit covers the validated eager CAUSAL rolling-cache decode path in `models/llama_patternkv.py`. It starts after global fixed-split softmax has produced one normalized probability tensor and does not include QK, softmax, cache append, centroid, or historical mixed-V logic.

## Current ABI

`CURRENT_SINK_VALUE_ABI = request_invariant_full_value_attention(weights, cache.sink_v, k_segment_valid_lengths(cache)[\"sink\"], num_key_value_groups)`

`CURRENT_PENDING_VALUE_ABI = request_invariant_full_value_attention(weights, cache.pending_v, k_segment_valid_lengths(cache)[\"pending\"], num_key_value_groups)`

`CURRENT_RECENT_VALUE_ABI = request_invariant_full_value_attention(weights, cache.recent_v, k_segment_valid_lengths(cache)[\"recent\"], num_key_value_groups)`

`CURRENT_OUTPUT_MERGE = attn_output = part if attn_output is None else attn_output + part`

## Layouts

`PROBABILITY_LAYOUT = [B, Hq, Q, T]`, with production decode `Q = 1`. The tensor is the global softmax result over the physical segment order used by `value_parts`.

`VALUE_LAYOUT = [B, Hkv, L, D]` for each FP16 tail segment. For DeepSeek-R1-Distill-Llama-8B production geometry, `Hq = 32`, `Hkv = 8`, and `D = 128`.

`OUTPUT_LAYOUT = [B, Hq, Q, D]` before attention output reshape.

`GQA_MAPPING = kv_head = query_head // num_key_value_groups`, with `num_key_value_groups = Hq / Hkv = 4`.

## Segment Order And Offsets

The rolling-cache CAUSAL path builds `value_parts` in physical probability order:

1. `sink`, if `cache.sink_k` exists.
2. `packed`, if historical quantized K exists.
3. `pending`, if `cache.pending_k` exists.
4. `recent`, if `cache.recent_k` exists.

The old Value loop slices probabilities as views using `attn_weights[:, :, :, offset : offset + length]`. These slices are not explicit copies, but `request_invariant_full_value_attention` can allocate padded weights/values for non-multiple split sizes.

## Current FP16 Tail Execution

For every non-packed tail segment, Python dispatches `request_invariant_full_value_attention`. That helper supports ragged valid lengths using a per-row `valid_lengths` tensor, expands KV heads to query heads through PyTorch `expand`/`reshape`, splits over the reduction length, computes multiply/sum per split, and accumulates the split outputs.

The old path materializes one output tensor per non-empty FP16 tail segment, then adds those tensors into the running attention output. With all three tail segments present, the FP16 tail path has three logical Value calls plus two output additions. It does not concatenate FP16 Values or probabilities.

## Empty, Ragged, And Batch Behavior

Empty physical segments are skipped by the caller because they are absent from `value_parts`. `request_invariant_full_value_attention` also returns zeros if the padded valid length is zero.

Ragged behavior comes from `k_segment_valid_lengths(cache)`, which supplies request-local valid lengths for `sink`, `packed`, `pending`, and `recent`. Production batching is true batch in tensor shape `[B,...]`; the old FP16 tail Value helper operates over the whole batch and does not launch one request at a time.

## DTypes

`INPUT_DTYPE = torch.float16` for probabilities and FP16 tail Values in the CUDA production path.

`ACCUMULATOR_DTYPE = torch.float32` for the current helper's multiply/sum expression because PyTorch reductions over FP16 CUDA tensors accumulate according to PyTorch kernel semantics; the existing Python reference path in tests expects tight FP16 output equivalence at the output tensor.

`OUTPUT_DTYPE = torch.float16`.
