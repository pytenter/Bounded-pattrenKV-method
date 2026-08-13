# Contiguous Reader Compatibility

- `logical_view()` uses `narrow` and never materializes.
- Current PatternKV CUDA wrappers require contiguous compact tensors and call `.contiguous()` internally.
- With slack capacity, current `[B,H,T,D]` / `[B,H,D,T]` layouts produce non-contiguous logical views.
- Therefore Stage B production integration would reintroduce historical materialization unless the ABI changes or CUDA VMM provides virtual-contiguous storage.

| Stream | Logical View Contiguous | Implicit Copy Needed | Stride |
|---|---:|---:|---|
| packed_k | False | True | `(33554432, 4194304, 32768, 1)` |
| packed_k_scale | False | True | `(262144, 32768, 256, 1)` |
| packed_k_zero | False | True | `(262144, 32768, 256, 1)` |
| packed_v | False | True | `(2097152, 262144, 8, 1)` |
| packed_v_scale | False | True | `(262144, 32768, 1, 1)` |
| packed_v_zero | False | True | `(262144, 32768, 1, 1)` |
| packed_v4 | False | True | `(4194304, 524288, 16, 1)` |
| packed_v4_scale | False | True | `(262144, 32768, 1, 1)` |
| packed_v4_zero | False | True | `(262144, 32768, 1, 1)` |
| v_precision_mask | True | False | `(32768, 1)` |
| k_assignments | False | True | `(262144, 32768, 1)` |
| v_assignment_idx | False | True | `(262144, 32768, 1)` |
| v_pattern_mask | False | True | `(262144, 32768, 1)` |
