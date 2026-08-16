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

// Experimental S5A-3 stride-aware Pattern K/QK reader. It preserves the
// production Pattern K math and consumes historical K tensors through their
// physical tensor strides instead of requiring a tight transposed copy.
torch::Tensor gemv_forward_cuda_outer_dim_with_base_strided_k(
    torch::Tensor _in_feats,        // [B*nh, 1, K]
    torch::Tensor _kernel,          // [B, nh_kv, K, ceil(N/pack)], may be strided
    torch::Tensor _scaling_factors, // [B, nh_kv, K, ceil(N/group)], may be strided
    torch::Tensor _zeros,           // [B, nh_kv, K, ceil(N/group)], may be strided
    const int bit,                  // 2 or 4
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,       // [nh_kv, M, K]
    torch::Tensor _assignments      // [B, nh_kv, N] (u8/i16/i32/i64), may be strided
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

// Experimental S3-2 V2 page-native Value attention reader. It consumes fixed
// page pointer tables and preserves the production V2 math path.
torch::Tensor attn_v_forward_cuda_outer_dim_with_base_paged_v2(
    torch::Tensor _alpha_q,
    torch::Tensor _vq_page_ptrs,
    torch::Tensor _vscale_page_ptrs,
    torch::Tensor _vzero_page_ptrs,
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,
    torch::Tensor _mask_page_ptrs,
    torch::Tensor _idx_page_ptrs,
    torch::Tensor _alpha_f,
    torch::Tensor _v_full,
    const int K,
    const int page_size,
    const int idx_bytes
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

// Experimental S2B-3 V2-only GQA backend. Production remains
// attn_v_forward_cuda_outer_dim_with_base unless explicitly selected by the
// Python benchmark/backend switch.
torch::Tensor attn_v_forward_cuda_outer_dim_with_base_gqa_v2(
    torch::Tensor _alpha_q,
    torch::Tensor _vq,
    torch::Tensor _vscale,
    torch::Tensor _vzero,
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,
    torch::Tensor _mask_q,
    torch::Tensor _idx_q,
    torch::Tensor _alpha_f,
    torch::Tensor _v_full
);

// Experimental S5A-1 V2-only strided-capacity Value reader. It preserves the
// production fused V2 math path while reading historical cache tensors through
// their physical strides instead of requiring tight contiguous K strides.
torch::Tensor attn_v_forward_cuda_outer_dim_with_base_strided_v2(
    torch::Tensor _alpha_q,
    torch::Tensor _vq,
    torch::Tensor _vscale,
    torch::Tensor _vzero,
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,
    torch::Tensor _mask_q,
    torch::Tensor _idx_q,
    torch::Tensor _alpha_f,
    torch::Tensor _v_full
);

torch::Tensor attn_v_forward_cuda_outer_dim_with_base_strided_v4(
    torch::Tensor _alpha_q,
    torch::Tensor _vq,
    torch::Tensor _vscale,
    torch::Tensor _vzero,
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,
    torch::Tensor _mask_q,
    torch::Tensor _idx_q,
    torch::Tensor _alpha_f,
    torch::Tensor _v_full
);

torch::Tensor attn_v_forward_cuda_page_mixed_pool(
    torch::Tensor _alpha_q,
    torch::Tensor _v2_payload,
    torch::Tensor _v4_payload,
    torch::Tensor _v2_scale,
    torch::Tensor _v2_zero,
    torch::Tensor _v4_scale,
    torch::Tensor _v4_zero,
    torch::Tensor _v2_pattern,
    torch::Tensor _v4_pattern,
    torch::Tensor _v2_assignment,
    torch::Tensor _v4_assignment,
    torch::Tensor _centroids,
    torch::Tensor _v2_page_offsets,
    torch::Tensor _v4_page_offsets,
    torch::Tensor _v2_page_table,
    torch::Tensor _v4_page_table,
    torch::Tensor _metadata_page_table,
    torch::Tensor _v4_prefix_counts,
    torch::Tensor _seq_lens,
    const int group_size,
    const int nh,
    const int nh_kv,
    const int page_size
);

torch::Tensor request_invariant_fixed_split_softmax_cuda(
    torch::Tensor _scores,
    torch::Tensor _total_lens,
    torch::Tensor _sink_lens,
    torch::Tensor _packed_lens,
    torch::Tensor _pending_lens,
    torch::Tensor _recent_lens,
    const int sink_physical,
    const int packed_physical,
    const int pending_physical,
    const int recent_physical,
    const int split_size
);

torch::Tensor fp16_tail_value_forward_cuda(
    torch::Tensor _probs,
    torch::Tensor _sink_v,
    torch::Tensor _pending_v,
    torch::Tensor _recent_v,
    torch::Tensor _sink_lens,
    torch::Tensor _pending_lens,
    torch::Tensor _recent_lens,
    const int sink_offset,
    const int pending_offset,
    const int recent_offset,
    const int num_key_value_groups
);
