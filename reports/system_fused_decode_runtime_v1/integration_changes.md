# Integration Changes

- Added `PATTERNKV_MIXED_V_BACKEND=fused_page`.
- Added cache-resident `operator_ready_page_pools` serialization.
- `_cat_mixed_packed_v` now appends page pools incrementally at the real pack/flush point.
