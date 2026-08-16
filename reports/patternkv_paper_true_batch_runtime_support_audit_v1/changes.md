# Changes

- Built operator-ready V2 page pools for non-mixed PatternKV `base_v2` history.
- Scoped that path to canonical PatternKV-paper-compatible all-V2 page-pool geometry: INT2 Value, group size 128, head dimension 128, and a paper-scale centroid bank.
- Routed non-mixed PatternKV-paper decode to the existing fused page-pool Value operator when page pools are present.
- Preserved legacy shared-centroid reader fallback only for caches without page pools.
- Fixed page-batch reference restore to select per-row centroids when using 4D centroid banks.
- Added a full-model audit runner and targeted request-local centroid regression tests.

No PatternKV-paper algorithm configuration changed. No CAUSAL selector or frozen precision semantics changed.
