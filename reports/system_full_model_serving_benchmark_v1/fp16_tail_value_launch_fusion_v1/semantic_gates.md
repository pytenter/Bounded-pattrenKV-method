# Semantic Gates

- `ALGORITHM_MODIFIED = false`
- `QK_MODIFIED = false`
- `SOFTMAX_MODIFIED = false`
- `HISTORICAL_MIXED_V_MODIFIED = false`
- `CACHE_LAYOUT_MODIFIED = false`
- `CENTROID_LOGIC_MODIFIED = false`
- `STATE_MERGE_ACTIVE = false`
- `CUDAGRAPH_ACTIVE = false`
- `NO_P_CONCAT_TEMPORARY = true`
- `NO_V_CONCAT_TEMPORARY = true`
- `OLD_PATH_FALLBACK = true`

The fused operator consumes existing global probabilities and existing sink, pending, and recent Value tensors.
