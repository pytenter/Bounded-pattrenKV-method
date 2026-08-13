# S1 Reuse Map

| S1 component | Reuse in fused page operator |
| --- | --- |
| Packed INT2/INT4 extraction | Reused directly in CUDA compressed-domain loads |
| Affine scale/zero streams | Reused as independent V2/V4 pools |
| Pattern centroid correction | Reused per token using pattern mask and assignment pools |
| Output ABI | Preserves `[B, nh, 1, head_dim]` Value output |
