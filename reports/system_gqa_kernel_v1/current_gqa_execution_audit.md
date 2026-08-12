# Current GQA Execution Audit

Preflight:

- `REPO_ROOT=/data/zypan/Bounded-pattrenKV-pseudodecode-3090`
- `CURRENT_BRANCH=sys/causal-v4-25-kernel-v1`
- `START_HEAD=cf100959a960982c610a157845715febbc6f8869`
- `WORKTREE_CLEAN=true` at phase start
- `BOUNDED_REMOTE=git@github.com:pytenter/Bounded-pattrenKV-method.git`
- `ORIGIN_REMOTE=https://github.com/HCOOOH/PatternKV.git`

Current production path:

- Python dispatch: `models/llama_patternkv.py::patternkv_mixed_value_attention` -> `quant/matmul.py::cuda_attn_v_mixed_fused_with_base`.
- Mixed-V wrapper compacts logical tokens into V2 and V4 lanes, then launches `cuda_attn_v_fused_with_base` once per non-empty lane. Current kernel launch count for mixed25 is 2.
- C++ entry: `attn_v_forward_cuda_outer_dim_with_base`.
- CUDA kernel: `battn_v_kernel_with_base<BIT, ABLATION_LANE0_TABLE_FULL>`.

Current CTA/warp organization:

- `grid.x = B * nh`; each CTA is one query head row.
- `grid.y = ceil((OC / PACK) / 4)`; each CTA covers four packed output tiles through `threadIdx.y`.
- `grid.z = 1`.
- `blockDim = (32, 4, 1)`, 128 threads, 4 warps.
- One CTA corresponds to one Q head and four output tiles, not one KV head.
- One warp corresponds to one output tile (`threadIdx.y`) for that Q head.
- Q->KV mapping is in CUDA: `ratio = nh / nh_kv`, `hk = hq / ratio`.

Duplicate reads for Q0/Q1/Q2/Q3 sharing KV0:

- packed V: repeated per Q head.
- scale: repeated per Q head and per packed output tile even when `group_size=128`.
- zero: repeated similarly to scale.
- Pattern mask: repeated per Q head and per output-tile CTA.
- assignment: repeated per Q head and per output-tile CTA.
- centroid table: repeated per Q head.
- recent FP16 V: repeated per Q head when recent is attached.

Q-head-specific data:

- attention alpha, histogram `SAcc`, final output. These cannot be shared.

KV-head-shared data:

- packed V2/V4 payload, scale, zero, Pattern mask, assignment, centroid table, and recent V rows.

Production preserved optimizations:

- Per-warp private histogram: `s_Sacc[threadIdx.y][Mcent]`, shared memory `4 * Mcent * sizeof(float) = 256 bytes` for Mcent=16.
- Lane0 table contribution: only lane0 computes the final centroid table contribution consumed by the output write.

Resource notes:

- Production threads/block: 128.
- Production shared memory/block: 256 bytes at Mcent=16.
- Register pressure: moderate; per warp keeps `psum[PACK]`, `add_base[PACK]`, `add_full[PACK]`.
