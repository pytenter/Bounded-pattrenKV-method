# Chunked Level 1 Synthetic Equivalence

Synthetic tests passed for the `segmented_chunked` cache state machine.

- Prefill cadence follows `packed=floor(T/residual_length)*residual_length` and `chunk=T%residual_length`.
- Decode cadence flushes only when the FP16 chunk buffer reaches `residual_length`.
- K/V assignment and V gate counts match packed tokens.
- Serialization preserves cache mode, chunk buffer, centroids, assignments, and V gate mask.
- Rolling-recent behavior is intentionally different and covered by a separate regression test.

CHUNKED_LEVEL1_PASS=true
