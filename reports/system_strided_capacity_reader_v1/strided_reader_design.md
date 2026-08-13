# Strided V2 Reader Design

- Added experimental API `cuda_attn_v_fused_with_base_strided_v2`.
- V2 only: bit width is fixed to INT2; no V4, mixed-V, K/QK, E2E decode, VMM, vLLM, or SGLang changes.
- Production default API is unchanged.
- Historical cache tensors are read through PyTorch-reported strides.
- Attention weights and centroids remain tight contiguous because they are not growing historical cache streams.
- Page lookup: NO.
- Page table: NO.
- Kernel loop bound is logical `K`, not physical capacity.
- Per-warp private histogram and lane0 centroid-table contribution are preserved by matching the production `ABLATION_LANE0_TABLE_FULL` path.
