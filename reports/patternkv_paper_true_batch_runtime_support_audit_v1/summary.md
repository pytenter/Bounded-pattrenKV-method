# Summary

`PATTERNKV_PAPER_TRUE_BATCH_RUNTIME_SUPPORT_AUDIT_V1` is supported.

The B2 crash was caused by a reader dispatch mismatch: PatternKV-paper correctly produced request-local centroids `[B,Hkv,C,D]`, while the old non-mixed reader expected shared centroids `[Hkv,C,D]`. The existing fused page-pool Value operator already supports request-local centroids, so `base_v2` now builds all-V2 page pools and decodes through that batch-aware path.

The page-pool path is explicitly scoped to the canonical PatternKV-paper all-V2 geometry so non-paper synthetic and legacy configurations keep their previous cache representation.

Semantic gates pass for B1, B2 single-step, B2 decode-8, B2 reorder, and B4. CAUSAL B1/B2 smoke and B2 reorder non-regression pass.

Validation passes: compileall, targeted pytest (`16 passed`), full pytest (`1052 passed`), and `git diff --check`.
