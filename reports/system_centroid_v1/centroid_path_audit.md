# S2B-2 Centroid Path Audit

## Call Chain

`models/llama_patternkv.py` builds segmented PatternKV cache tensors and calls `cuda_attn_v_mixed_fused_with_base` through the Value-attention path. `quant/matmul.py` splits frozen mixed25 Value tokens into V2 and V4 compact lanes, gathers matching `attn`, `v_pattern_mask`, and `v_idx`, then calls `cuda_attn_v_fused_with_base`. That wrapper reshapes tensors and calls `patternkv_gemv.attn_v_forward_cuda_outer_dim_with_base`, bound in `quant/csrc/pybind.cpp` and declared in `quant/csrc/gemv_cuda.h`.

## Current Kernel

The active CUDA implementation is `battn_v_kernel_with_base<BIT, MODE>` in `quant/csrc/gemv_cuda.cu`. Production dispatch uses `MODE=ABLATION_FULL`; other modes are benchmark-only and reachable only through `attn_v_forward_cuda_outer_dim_with_base_debug`.

## Required Path Details

A. Residual path: inside the K-loop, each warp loads `alpha_q`, packed low-bit `vq`, `vscale`, and `vzero`, then accumulates `psum[p] += (scale * code + zero) * alpha`.

B. Centroid path: the same K-loop aggregates attention mass into shared-memory `s_Sacc[idx]`, then the post-loop centroid-table pass computes `add_base[p] += s_Sacc[c] * C[c, oc]`.

C. Pattern mask read: `mask_row[t]` is loaded with `__ldg(mask_row + t)` in the `wy == 0` histogram warp.

D. Assignment/index read: `idx_row` is read as uint8, uint16, or int32 according to `idx_bytes`, also only in the histogram warp.

E. Centroid attention mass: for active mask entries, `atomicAdd(&s_Sacc[idx], alpha)` accumulates per-centroid attention mass.

F. Shared memory: `extern __shared__ float s_Sacc[]`, sized to `Mcent * sizeof(float)`, stores the per-block centroid histogram.

G. AtomicAdd: yes, shared-memory `atomicAdd` is used on `s_Sacc[idx]`.

H. Synchronization: two `__syncthreads()` calls are used, one after zeroing `s_Sacc`, one after histogram accumulation before centroid table use.

I. Centroid table reads: `C = _centroids + hk * Mcent * OC`; after histogram, each output tile reads `C[c, oc_start+p]` from global memory with `__ldg`.

J. GQA repetition: grid x is `B*nh`; `hk = hq / (nh / nh_kv)`. With `nh=32`, `nh_kv=8`, ratio 4, the same KV head centroid table, mask, and assignment are processed separately for four query heads because each query head has distinct `alpha_q`.

K. FP16 recent contribution: recent FP16 contribution uses `_alpha_f` and `_v_full` in a separate post-histogram loop. It is not mathematically coupled to centroid aggregation, but it shares the same output tile and final writeback in FULL mode.
