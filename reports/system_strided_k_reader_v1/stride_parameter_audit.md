# Stride Parameter Audit

- `packed_view`: shape=(1, 8, 128, 512), stride=(2097152, 262144, 2048, 1), storage_offset=0, dtype=torch.int32
- `scale_view`: shape=(1, 8, 128, 64), stride=(262144, 32768, 256, 1), storage_offset=0, dtype=torch.float16
- `zero_view`: shape=(1, 8, 128, 64), stride=(262144, 32768, 256, 1), storage_offset=0, dtype=torch.float16
- `assignments_view`: shape=(1, 8, 8192), stride=(262144, 32768, 1), storage_offset=0, dtype=torch.int32
- `alpha`: shape=(32, 1, 128), stride=(128, 128, 1), storage_offset=0, dtype=torch.float16
- `centroids`: shape=(8, 16, 128), stride=(2048, 128, 1), storage_offset=0, dtype=torch.float16

- Packed K kernel parameters: B/H/head_dim/pack strides from `_kernel.stride(0..3)`.
- Scale kernel parameters: B/H/head_dim/group strides from `_scaling_factors.stride(0..3)`.
- Zero kernel parameters: B/H/head_dim/group strides from `_zeros.stride(0..3)`.
- Assignment kernel parameters: B/H/token strides from `_assignments.stride(0..2)`.
- Logical tokens are `_assignments.size(2)`, not physical capacity.
