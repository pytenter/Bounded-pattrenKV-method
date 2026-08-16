# Full-Model Bottleneck Profile

## Executive Summary

At `context=2048`, `B=1`, `decode=4`, CAUSAL full-model decode takes `788.231` ms/token versus FP16 `57.773` ms/token. The incremental slowdown is `730.459` ms/token. The dominant measured causes are not MLP and not the fused page Value kernel. They are:

1. Full-model PatternKV self-attention path, especially request-invariant segmented softmax and related attention bookkeeping.
2. Harness-level per-token cache assemble/split, which rebuilds/slices every layer cache and page-pool metadata.
3. Retained centroid-state pool capacity, which dominates extra allocated memory after prefill.

## Timing Attribution

Top-level CAUSAL timing:

| Component | ms/token | Fraction total | Increment vs FP16 |
| --- | ---: | ---: | ---: |
| model decode | 502.638 | 63.768% | 446.835 ms/token |
| harness split | 162.619 | 20.631% | 162.379 ms/token |
| harness assemble | 119.335 | 15.140% | 117.844 ms/token |

Nested CAUSAL attention timing from targeted wrappers:

| Component | ms/token | Calls/token | Interpretation |
| --- | ---: | ---: | --- |
| request-invariant softmax | 100.500 | 32.0 | large per-layer Torch softmax/scatter/gather path |
| importance update | 24.443 | 32.0 | moderate per-layer cost |
| fused page Value operator | 6.283 | 32.0 | not dominant |

Harness assemble/split copies allocate about `447.5` MB/token across `1696` copy-producing calls/token. This explains `38.6%` of CAUSAL-vs-FP16 incremental runtime by top-level timing.

## Memory Attribution

At profile context 2048, retained post-prefill extra allocation is `1.338` GB. CAUSAL centroid state accounts for `1.212` GB, or `90.6%` of that retained extra memory. Page-pool bytes are only `0.058` GB, so `PAGE_POOL_OVERALLOCATION` is not supported as the primary memory root.

Decode peak extra allocation is `2.348` GB. Centroid state explains `51.6%` of peak extra allocation; transient assemble/split copies and allocator reservation account for the remaining peak behavior, but exact peak lifetime decomposition is incomplete.

## Hypothesis Results

- H1 per-layer preparation: SUPPORTED. Operator-ready page pools are merged and sliced 32 times/token.
- H2 row-slice copy: SUPPORTED. Row/cache copies are real tensor allocations, not metadata-only views.
- H3 page-pool over-allocation: NOT SUPPORTED as primary root. Page pool is about `0.058` GB.
- H4 duplicate/oversized runtime state: SUPPORTED for centroid-state overcapacity, not for dense historical FP16 KV.
- H5 temporary buffer retention: PARTIAL/INCOMPLETE. Transient copy volume is high, but retained temp allocation sites were not captured as persistent tensors.
- H6 quant/selector overhead: PARTIAL. Decode importance/softmax costs are material; selector/packing are more prefill-side than decode-dominant here.
- H7 GPU-CPU sync: NOT SUPPORTED for fused-page hot path by counters.

## Final Root Cause

Throughput bottleneck classification: `MULTI_COMPONENT`, specifically `PER_LAYER_METADATA_REBUILD_DOMINATED + ROW_SLICE_COPY_DOMINATED + COMPRESSED_ATTENTION_SOFTMAX_DOMINATED`. The fused compressed Value operator itself is not the dominant cost.

Memory bottleneck classification: `MULTI_COMPONENT`, specifically `CENTROID_STATE_POOL_OVERCAPACITY + ROW_SLICE_COPY_MEMORY + ALLOCATOR_RESERVATION`.

## Next Task

`OPTIMIZE_FULL_MODEL_CACHE_METADATA_REUSE_AND_CENTROID_CAPACITY`

This should be a separate optimization round with an A/B check. Do not change frozen CAUSAL-V4@25% algorithm semantics.
