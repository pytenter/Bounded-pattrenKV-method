# S2B-2A Current Histogram Audit

## Scope

This audit covers the production Value-attention CUDA path in `quant/csrc/gemv_cuda.cu`, specifically `battn_v_kernel_with_base<BIT, MODE>` as called by `attn_v_forward_cuda_outer_dim_with_base` with `MODE=ABLATION_FULL`. The benchmark-only debug entry added in S2B-2 is not used by production unless explicitly called by tests/benchmarks.

## Histogram Threads

Only the `wy == 0` warp in each CTA participates in centroid histogram construction. The CTA has `dim3 threads(32, 4, 1)`, so there are four warps per block, but one warp builds `s_Sacc` and all four output-tile warps later consume it.

## Tokens Per Thread

The K loop uses `TILE=128`. Each lane in the histogram warp processes four token positions per tile: `t_base = kt * 128 + lane * 4`, then `i = 0..3`. Therefore each warp covers up to 128 tokens per tile and each participating thread handles up to four token checks per tile.

## Warp Count And BlockDim

- `blockDim.x = 32`
- `blockDim.y = 4`
- warps per block = 4
- histogram producer warps per block = 1
- output tile warps per block = 4

## Mcent And Shared Memory

`Mcent = _centroids.size(1)`. Frozen experiments use 16 or 32 Pattern centroids depending on model/config path; the S2B-2 synthetic benchmark used `Mcent=16`. Shared memory is `Mcent * sizeof(float)` for `s_Sacc` in the current production path.

## Per-token AtomicAdd

For each token handled by `wy == 0`, if `mask_row[t] != 0` and `idx` is in range, the current code performs one shared-memory atomic add:

`atomicAdd(&s_Sacc[idx], alpha_q[t])`

So the logical atomic count equals the number of active Pattern-mask tokens processed by that CTA.

## Atomic Scope

The atomic is shared-memory atomic, not global-memory atomic. The contention point is `s_Sacc[idx]`, especially when many lanes map to the same centroid bucket.

## Synchronization

There are two `__syncthreads()` protecting the histogram:

1. after all warps cooperatively zero `s_Sacc`
2. after the histogram warp finishes `atomicAdd` updates and before all warps read `s_Sacc` for centroid-table contribution

## Overlap With Residual Path

The histogram and residual paths are in the same K-loop. The histogram work is restricted to `wy == 0`, while all warps perform residual packed-V/scale/zero accumulation for their own output tile. They are not separate kernels; their latency can overlap partially inside the CTA but still contributes to the same kernel critical path before the second synchronization.

## Assignment Distribution And Contention

Assignment distribution directly controls contention. Uniform assignments over 16 centroids distribute writes across buckets. Skewed assignments, e.g. 50% of active tokens assigned to centroid 0, force many lanes to update the same `s_Sacc[0]` bucket and increase shared atomic serialization.

## GQA Repetition

The grid maps `blockIdx.x` over `B * nh`; `hk = hq / (nh / nh_kv)`. With `nh=32` and `nh_kv=8`, four query heads share one KV head. The same KV-side mask, assignment, and centroid table are processed separately for each query head because each query head has distinct attention weights. S2B-2A does not implement GQA-aware CTA redesign.
