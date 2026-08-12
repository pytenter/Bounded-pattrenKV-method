#pragma once
#include <torch/extension.h>

torch::Tensor gemv_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    const int bit,
    const int group_size);


torch::Tensor gemv_forward_cuda_outer_dim(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    const int bit,
    const int group_size,
    const int nh,
    const int nh_kv);


// // === 新增：融合基向量（centroids + assignments）的外维GEMV ===
// torch::Tensor gemv_forward_cuda_outer_dim_with_base(
//     torch::Tensor _in_feats,          // [B*nh, 1, K] (q_len=1)
//     torch::Tensor _kernel,            // [B*nh_kv, N/pack, K]
//     torch::Tensor _scaling_factors,   // [B*nh_kv, N/group_size, K]
//     torch::Tensor _zeros,             // [B*nh_kv, N/group_size, K]
//     const int bit,                    // 2 or 4
//     const int group_size,
//     const int nh,                     // query heads
//     const int nh_kv,                  // kv heads
//     torch::Tensor _centroids,         // [nh_kv, M, K]
//     torch::Tensor _assignments        // [B, nh_kv, N] (uint8/uint16/int32)
// );

// === 新增：融合基向量（centroids + assignments）的外维 GEMV ===
torch::Tensor gemv_forward_cuda_outer_dim_with_base(
    torch::Tensor _in_feats,        // [B*nh, 1, K]
    torch::Tensor _kernel,          // [B*nh_kv, N/pack, K]
    torch::Tensor _scaling_factors, // [B*nh_kv, N/group, K]
    torch::Tensor _zeros,           // [B*nh_kv, N/group, K]
    const int bit,                  // 2 or 4
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,       // [nh_kv, M, K]
    torch::Tensor _assignments      // [B, nh_kv, N] (u8/u16/i32)
);

// 融合: attn_q @ V_quant(outer-dim dequant) + 基向量补偿 (+ 可选的 attn_f @ V_full)
torch::Tensor attn_v_forward_cuda_outer_dim_with_base(
    torch::Tensor _alpha_q,
    torch::Tensor _vq,
    torch::Tensor _vscale,
    torch::Tensor _vzero,
    const int bit,
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,
    torch::Tensor _mask_q,
    torch::Tensor _idx_q,
    torch::Tensor _alpha_f,
    torch::Tensor _v_full
);

// Benchmark-only centroid ablation entry. Production code must keep using
// attn_v_forward_cuda_outer_dim_with_base.
torch::Tensor attn_v_forward_cuda_outer_dim_with_base_debug(
    torch::Tensor _alpha_q,
    torch::Tensor _vq,
    torch::Tensor _vscale,
    torch::Tensor _vzero,
    const int bit,
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,
    torch::Tensor _mask_q,
    torch::Tensor _idx_q,
    torch::Tensor _alpha_f,
    torch::Tensor _v_full,
    const int debug_mode
);
