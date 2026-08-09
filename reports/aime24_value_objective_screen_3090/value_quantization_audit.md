# Value Quantization Objective Audit

## Decision Path

- Production segmented PatternKV stores V in `PatternQuantizedKVCache`.
- Prefill creates the initial V centroid bank in `models/llama_patternkv.py` using `batched_kmeans_fast_compiled` on value vectors.
- Decode/rolling appends one dynamic V centroid per pack window with `pattern_chebyshev_center_per_head`.
- The clean V assignment decision point is `pattern_nearest_v_centroid` in `models/segmented_cache.py`; the legacy tuple path has the mirrored method `LlamaAttention_PatternKV._nearest_v_centroid`.
- The baseline V assignment objective is minmax range of the residual candidate: `amax(v - c) - amin(v - c)`.
- After assignment, `pattern_v_threshold_and_mask` decides whether the selected centroid is actually subtracted.
- The packed tensor is always `v_adjusted = v - mask * centroid`, quantized by `quantize_pack_v_reference` / `triton_quantize_and_pack_along_last_dim`.

## Granularity

- Logical Value unit: token x KV-head x head_dim vector.
- Current AIME24 config has `head_dim=128` and `group_size=128`, so one Value vector is one affine quantization group.
- V bitwidth is INT2 (`v_bits=2`).
- Packing format, scale, zero/min, centroid bank shape, assignment tensor shape, and mask tensor shape are independent of the scoring objective.

## Candidate Representation

- Candidate set for a Value token is the existing V centroid bank for that KV head.
- Candidate reconstruction can be interpreted as `dequant(v - mask(candidate) * candidate) + mask(candidate) * candidate`.
- V-DIR and V-HYBRID can rescore exactly this candidate set without changing bit width, packing, group size, metadata schema, sink, recent, or K path.

## MSE / Reconstruction

- Baseline does not optimize raw MSE; it chooses the centroid by minmax residual range, then uses affine INT2 packing on the chosen adjusted vector.
- Existing observer/reference code computes MSE-like diagnostics, but the production assignment is minmax range.

## Hook Compatibility

- `VALUE_OBJECTIVE_HOOK_COMPATIBLE=true` for V-DIR and V-HYBRID because they can replace only the V centroid scoring function over the same feasible candidates.
- K assignment remains `_assign_minmax_hnk`; K path is not touched.

## Causal-Attention Constraint

- Current V decisions are independent per token x KV-head vector.
- A scalar historical-attention weight `w_i` multiplies all candidate costs for token `i`; therefore it cancels out of the per-token argmin.
- A meaningful attention-weighted objective would require a coupled decision over multiple tokens, a changed grouping/packing degree of freedom, selective precision, or persistent/pack-time metadata. Those are outside this prompt.
- Therefore V-CAUSAL-ATTN is not a clean objective-rescoring intervention for the current PatternKV V quantization granularity.
