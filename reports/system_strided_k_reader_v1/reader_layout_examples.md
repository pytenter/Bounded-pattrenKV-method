# Reader Layout Examples

- `packed_view`: shape=(1, 8, 128, 512), stride=(2097152, 262144, 2048, 1), storage_offset=0, dtype=torch.int32
- `scale_view`: shape=(1, 8, 128, 64), stride=(262144, 32768, 256, 1), storage_offset=0, dtype=torch.float16
- `zero_view`: shape=(1, 8, 128, 64), stride=(262144, 32768, 256, 1), storage_offset=0, dtype=torch.float16
- `assignments_view`: shape=(1, 8, 8192), stride=(262144, 32768, 1), storage_offset=0, dtype=torch.int32
- `alpha`: shape=(32, 1, 128), stride=(128, 128, 1), storage_offset=0, dtype=torch.float16
- `centroids`: shape=(8, 16, 128), stride=(2048, 128, 1), storage_offset=0, dtype=torch.float16
