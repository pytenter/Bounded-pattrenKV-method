# Segmented Heterogeneous Attention State Merge V1 Design

## Implementation

- Added `SegmentedAttentionState(o,m,l)` in `models/segmented_cache.py`.
- Added exact two-state merge and finalization helpers.
- Added `PATTERNKV_SEGMENTED_STATE_MERGE=0` as the explicit legacy fallback.
- Default `PATTERNKV_SEGMENTED_STATE_MERGE=1` enables the experimental state path for rolling segmented cache decode.

## Production Flow

1. Existing QK segment kernels still produce segment-local score tensors.
2. The new path reuses `score_parts`; it does not recompute QK.
3. Each segment computes local probabilities only for that segment, feeds the existing Value backend, and converts the normalized segment output into `(O_i,m_i,l_i)`.
4. States merge deterministically left-to-right in physical/request-invariant order.
5. Causal importance is updated from segment probabilities rescaled by the merged state, avoiding a global `[B,Hq,Q,total]` probability tensor.

## Non-Goals Held

- No selector, V4 ratio, quantization layout, sink/recent/residual length, centroid ownership, scheduler, CUDA graph, or low-bit kernel rewrite changed.
- Chunked-cache path remains on the validated legacy integration path.

