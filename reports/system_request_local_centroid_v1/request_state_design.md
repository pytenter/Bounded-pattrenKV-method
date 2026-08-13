# Request State Design

`PatternKVCentroidStatePool` stores K/V centroid pools, per-slot counts, update counters, last flush position, and active lifecycle metadata. The active batch carries `centroid_state_indices[B]`; non-contiguous slot mappings and reorder are covered by tests.
