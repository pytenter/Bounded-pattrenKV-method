# Stream Layout Audit

| Stream | Shape | Token dim | Dtype | Grows? | torch.cat? | Can preallocate? |
|---|---|---:|---|---:|---:|---:|
| packed_k | `(1, 8, 128, 'T')` | 3 | `torch.int32` | yes | yes | yes, but logical view with slack is non-contiguous for current CUDA layouts |
| packed_k_scale | `(1, 8, 128, 'T')` | 3 | `torch.float16` | yes | yes | yes, but logical view with slack is non-contiguous for current CUDA layouts |
| packed_k_zero | `(1, 8, 128, 'T')` | 3 | `torch.float16` | yes | yes | yes, but logical view with slack is non-contiguous for current CUDA layouts |
| packed_v | `(1, 8, 'T', 8)` | 2 | `torch.int32` | yes | yes | yes, but logical view with slack is non-contiguous for current CUDA layouts |
| packed_v_scale | `(1, 8, 'T', 1)` | 2 | `torch.float16` | yes | yes | yes, but logical view with slack is non-contiguous for current CUDA layouts |
| packed_v_zero | `(1, 8, 'T', 1)` | 2 | `torch.float16` | yes | yes | yes, but logical view with slack is non-contiguous for current CUDA layouts |
| packed_v4 | `(1, 8, 'T', 16)` | 2 | `torch.int32` | yes | yes | yes, but logical view with slack is non-contiguous for current CUDA layouts |
| packed_v4_scale | `(1, 8, 'T', 1)` | 2 | `torch.float16` | yes | yes | yes, but logical view with slack is non-contiguous for current CUDA layouts |
| packed_v4_zero | `(1, 8, 'T', 1)` | 2 | `torch.float16` | yes | yes | yes, but logical view with slack is non-contiguous for current CUDA layouts |
| v_precision_mask | `(1, 'T')` | 1 | `torch.uint8` | yes | yes | yes; compatible in current 2D layout |
| k_assignments | `(1, 8, 'T')` | 2 | `torch.int64` | yes | yes | yes, but logical view with slack is non-contiguous for current CUDA layouts |
| v_assignment_idx | `(1, 8, 'T')` | 2 | `torch.int64` | yes | yes | yes, but logical view with slack is non-contiguous for current CUDA layouts |
| v_pattern_mask | `(1, 8, 'T')` | 2 | `torch.uint8` | yes | yes | yes, but logical view with slack is non-contiguous for current CUDA layouts |
