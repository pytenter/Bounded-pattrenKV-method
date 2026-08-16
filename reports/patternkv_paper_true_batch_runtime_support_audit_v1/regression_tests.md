# Regression Tests

Added `tests/test_patternkv_paper_true_batch_runtime_support.py`.

Coverage:

- B1/B2/B4 all-V2 fused page-pool output matches a request-local reference.
- B2 reorder preserves request-local outputs.
- Request-local centroid banks and assignments diverge.
- Shared row-0 centroid negative control fails for row 1.
- Page-pool dispatch counters report zero serial B1 dispatch.
- Prefill cache construction creates request-local page pools for `base_v2`.

Existing paper-baseline tests continue to check canonical method identity, PatternKV-paper selector identity, CAUSAL frozen constants, invalid-run rejection, and reporting classification.
