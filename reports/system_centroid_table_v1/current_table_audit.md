# Current Centroid Table Audit

- Kernel path: `quant/csrc/gemv_cuda.cu::battn_v_kernel_with_base`.
- Python path: `quant/matmul.py::cuda_attn_v_fused_with_base`; debug path exposes `FULL`, `RESIDUAL_ONLY`, `NO_TABLE_CONTRIBUTION`, `PER_WARP_HIST_FULL`, and `LANE0_TABLE_FULL`.
- Centroid physical layout: `[nh_kv, Mcent, OC]`, contiguous row-major.
- Runtime shape in benchmark: `nh_kv=8`, `Mcent=16`, `head_dim/OC=128`.
- Dtype: fp16 (`half`).
- Bytes per KV-head table: `Mcent * head_dim * sizeof(fp16) = 16 * 128 * 2 = 4096` bytes.
- CUDA mapping: grid x covers `B * nh` query heads; grid y covers packed output-channel tiles. `threadIdx.y` selects one warp/output tile, `threadIdx.x` is the warp lane.
- GQA mapping: `nh=32`, `nh_kv=8`, ratio `4`; each four Q heads map to one KV head.
- Histogram state: S2B-2A per-warp private histogram is preserved as `s_Sacc[threadIdx.y][Mcent]`.
- Baseline table contribution before this phase: every lane executed the centroid row loop and loaded `C[c, oc_start+p]` using scalar `__ldg`, but final output consumed `add_base[p]` only under `lane == 0`.
- Candidate B table contribution: only `lane == 0` executes the centroid table loop; accumulation order and final `lane0` write semantics are unchanged.
- Vectorization/coalescing: the baseline scalar loads were contiguous within a row for each lane but redundantly repeated across lanes; this phase removes redundant lane work instead of adding a wider vector type.
- Shared memory possibility: staging a full KV-head centroid table costs 4096 bytes per KV head for `Mcent=16, head_dim=128`; useful mainly for GQA reuse, but it needs CTA/grid reorganization.
- Register possibility: `add_base[PACK]` remains in registers; the table itself is too large for registers at the CTA scope.
- SAcc zero behavior: current code guards centroid-row contribution with `if (s != 0.f)`, so rows with zero histogram mass do not load centroid data.
