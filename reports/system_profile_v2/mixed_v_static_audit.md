# Mixed-V Static Audit

## Call Path

`models/llama_patternkv.py::patternkv_mixed_value_attention` dispatches to
`quant/matmul.py::cuda_attn_v_mixed_fused_with_base` when
`PATTERNKV_MIXED_V_BACKEND=fused`.

The current implementation is a two-lane compact execution strategy:

1. Python reads the logical `v_precision_mask`.
2. It builds boolean masks for V2 (`~mask`) and V4 (`mask`).
3. It gathers logical-order attention weights, Pattern masks, and assignment
   indices into compact V2 and V4 order using boolean indexing plus
   `.contiguous()`.
4. It calls `cuda_attn_v_fused_with_base` once for V2 tokens when present.
5. It calls `cuda_attn_v_fused_with_base` once for V4 tokens when present.
6. It sums the two `[B, H, 1, D]` outputs.

## CUDA Binding

`cuda_attn_v_fused_with_base` prepares C++ extension inputs:

- `attn_q.to(torch.float16).contiguous()`
- `v_centroids.to(torch.float16).contiguous()`
- `v_scale.to(torch.float16).contiguous()`
- `v_zero.to(torch.float16).contiguous()`
- `v_mask_q.to(torch.uint8).contiguous()`
- `v_idx_q` dtype narrowing when needed, then `.contiguous()`
- views/reshapes/transposes for alpha, packed V, scale, and zero.

It calls `patternkv_gemv.attn_v_forward_cuda_outer_dim_with_base`, exported from
`quant/csrc/pybind.cpp`.

## CUDA Kernel

`quant/csrc/gemv_cuda.cu::attn_v_forward_cuda_outer_dim_with_base` launches
`battn_v_kernel_with_base<2>` or `<4>`.

Inside one kernel launch, the CUDA kernel performs:

- low-bit residual Value dequantized accumulation
- scale/zero loads
- Pattern centroid restore through shared-memory `Sacc`
- assignment/mask handling
- optional FP16 tail contribution
- output writeback

## Launch Count

For the frozen 25% mixed case with both V2 and V4 tokens present, one mixed-V
logical call launches two CUDA kernels: one 2-bit lane and one 4-bit lane.
