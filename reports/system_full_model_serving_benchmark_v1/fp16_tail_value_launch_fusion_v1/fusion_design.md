# Fusion Design

## Production Switch

`PATTERNKV_FP16_TAIL_VALUE_FUSION=0` keeps the old validated per-segment FP16 tail Value path.

`PATTERNKV_FP16_TAIL_VALUE_FUSION=1` enables one fused FP16 tail Value call for the rolling-cache CAUSAL path.

## Fused Operator ABI

`fp16_tail_value_forward_cuda(probs, sink_v, pending_v, recent_v, sink_lens, pending_lens, recent_lens, sink_offset, pending_offset, recent_offset, num_key_value_groups) -> tail_output`

Inputs:

- `probs`: `[B, Hq, 1, T]`, `float16`, contiguous, existing global softmax probabilities.
- `sink_v`: `[B, Hkv, sink_physical, D]`, `float16`, contiguous, or `[B, Hkv, 0, D]`.
- `pending_v`: `[B, Hkv, pending_physical, D]`, `float16`, contiguous, or `[B, Hkv, 0, D]`.
- `recent_v`: `[B, Hkv, recent_physical, D]`, `float16`, contiguous, or `[B, Hkv, 0, D]`.
- segment valid lengths: `[B]`, `int64`, CUDA, contiguous.
- segment offsets: physical offsets into `probs`.
- `num_key_value_groups`: GQA ratio, production value `4`.

Output:

- `tail_output`: `[B, Hq, 1, D]`, `float16`.

## Scope Preservation

The fused path starts after fixed-split global softmax. It does not change QK decomposition, score concat, softmax, historical mixed-V, selector, quantization, cache ownership, centroids, or append.

No new probability concatenation or Value concatenation is introduced. Empty segments pass zero-length tensors.

## Old Fallback

The old path remains in `models/llama_patternkv.py` and is selected by default. Counters record old segment calls and output add calls.
