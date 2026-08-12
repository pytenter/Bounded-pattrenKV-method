# Phase S2B-1 V2 Kernel Audit

## Call Chain

`models/llama_patternkv.py::patternkv_mixed_value_attention`
calls `quant/matmul.py::cuda_attn_v_mixed_fused_with_base`.

For mixed V2/V4, the Python wrapper splits logical attention by
`v_precision_mask`:

- V2 lane: `attn_q[..., low_mask].contiguous()`, V Pattern mask gather, V
  assignment-index gather, then `cuda_attn_v_fused_with_base(..., bits=2, ...)`.
- V4 lane: same structure with `bits=4`.

`cuda_attn_v_fused_with_base` reshapes tensors and calls
`patternkv_gemv.attn_v_forward_cuda_outer_dim_with_base`, exported in
`quant/csrc/pybind.cpp`.

The CUDA entry in `quant/csrc/gemv_cuda.cu` dispatches:

- `battn_v_kernel_with_base<2>` for V2
- `battn_v_kernel_with_base<4>` for V4

## Grid / Block

Current host launch uses:

- `threads = dim3(32, 4, 1)`
- `blocks = dim3(BSnh, (OC / PACK + threads.y - 1) / threads.y, 1)`
- shared memory: `Mcent * sizeof(float)`

For DeepSeek-R1-Distill-Llama-8B:

- `num_attention_heads = 32`
- `num_key_value_heads = 8`
- `head_dim / OC = 128`
- GQA ratio = `4`

For V2:

- `BIT = 2`
- `PACK = 32 / BIT = 16`
- `OC / PACK = 8`
- With `threads.y = 4`, each query head uses 2 CUDA blocks along the output-channel packed dimension.

## Warp Responsibility

Within one block:

- `threadIdx.x` is the warp lane, 0..31.
- `threadIdx.y` selects one output-channel tile within the block.
- Each warp computes one packed output-channel tile.
- For V2 each tile covers 16 output channels.

## Loads

Attention weights:

- `alpha_q` is `[B*nh, K]`.
- Each lane loads up to 4 attention values per 128-token tile.

Packed V2:

- `_vq` is viewed as `[B*nh_kv, OC/PACK, K]`.
- Each V2 warp loads one `uint32_t` packed word per lane item, containing 16 2-bit values.

Scale / zero:

- `_vscale` and `_vzero` are `[B*nh_kv, OC/group, K]`.
- With `group_size=128` and `OC=128`, there is one OC group, so the V2 lane repeatedly loads scale/zero for the same output group along K.

Pattern mask / assignment:

- `_mask_q` is `[B, nh_kv, K]`.
- `_idx_q` is `[B, nh_kv, K]` with 1/2/4-byte index width.
- Only `wy == 0` accumulates the shared-memory centroid histogram `Sacc`.

Centroids:

- `_centroids` is `[nh_kv, Mcent, OC]`.
- After K-loop histogram accumulation, each warp loops over all centroids and loads centroid rows for its output-channel tile.

FP16 recent contribution:

- Optional `_alpha_f` and `_v_full` are handled in the same kernel when the mixed wrapper attaches a full-precision tail to the first nonempty lane.

## Reductions / Synchronization

- `s_Sacc[Mcent]` lives in dynamic shared memory.
- One `__syncthreads()` after shared-memory initialization.
- One `__syncthreads()` after the K-loop before centroid compensation.
- Warp reductions are used for `add_full` and residual `psum`.

## GQA Mapping

The kernel maps query head to KV head:

```text
ratio = nh / nh_kv = 4
hq = bnh % nh
hk = hq / ratio
```

Thus Q heads 0..3 share KV head 0, 4..7 share KV head 1, etc.

## GQA_V2_GLOBAL_LOAD_REUSE_OPPORTUNITY

`YES`, but nontrivial.

The same packed V2 payload, scale/zero, Pattern mask, assignment indices, and
centroids are read separately by the four query heads sharing one KV head.
However, attention weights are query-head-specific, and the current grid assigns
`blockIdx.x = B*nh`, so the four query heads execute as separate block groups.
Reusing KV data across query heads would require changing block ownership,
shared-memory staging, or multi-query-head computation per block. That is beyond
the minimal one-change V2-only candidate and risks larger register/shared-memory
pressure.

## Main Local Optimization Hypothesis

For V2, `OC/PACK = 8`. The current `threads.y=4` requires two block groups per
query head for the 8 V2 output tiles. A V2-only launch with `threads.y=8` can
cover all V2 output-channel tiles for a query head in one block along `blockIdx.y`.
This preserves semantics and leaves V4 unchanged.
