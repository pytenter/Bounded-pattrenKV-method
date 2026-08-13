# Strided V4 Design

- Added `cuda_attn_v_fused_with_base_strided_v4`.
- V4 remains an independent INT4 affine stream with its own scale/zero.
- The implementation shares the S5A-1 stride-aware addressing template with V2 and preserves the production Value attention math.
