# Kernel Contract

The legacy non-mixed reader wrapper `cuda_attn_v_fused_with_base` requires:

```text
v_centroids: [Hkv,C,D]
```

The existing fused page-pool CUDA operator `attn_v_forward_cuda_page_mixed_pool` already accepts:

```text
v_centroids: [Hkv,C,D] or [B,Hkv,C,D]
```

Therefore the blocker was not a fundamental kernel limitation. PatternKV-paper `base_v2` needed to package all historical V tokens as V2 pages and route decode through the already batch-aware page-pool operator.
