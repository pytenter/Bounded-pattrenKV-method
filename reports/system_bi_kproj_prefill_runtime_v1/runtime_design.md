# Runtime Design

The runtime calls `batch_invariant_k_projection(..., backend=PATTERNKV_BI_KPROJ_BACKEND)` for K projection during prefill. Q projection, V projection, RoPE, K-means, selector, packing, centroid state, and fused Value code are unchanged.
