# Root Cause

Classification:

```text
PATTERNKV_PAPER_BATCH_DIM_READER_SUPPORT_MISSING
```

PatternKV-paper correctly constructs request-local centroid banks and request-local assignments for B2. The old non-mixed Value reader dispatch assumed a single shared 3D centroid bank. The repository already contained a batch-aware fused page-pool Value operator, so the minimum correctness fix was to route `base_v2` all-V2 PatternKV-paper history through that operator-ready page-pool path.
