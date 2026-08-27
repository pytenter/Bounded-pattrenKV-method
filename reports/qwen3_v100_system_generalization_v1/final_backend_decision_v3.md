# Final Backend Decision V3

GO for fixed-length Qwen3-8B compressed true-batch backend readiness on V100.

Closed blocker: `QWEN_COMPRESSED_TRUE_BATCH_B2_FAIL` caused by request-local centroid handling. CUDA K/QK and V readers now accept request-local centroid layouts, the extension was rebuilt for V100 `sm_70`, and B2/B4 model smoke passed without serial request dispatch.

Evidence: `k_request_local_centroid_oracle.*`, `v_request_local_centroid_oracle.*`, `centroid_swap_isolation.*`, `b1_post_cuda_kv_fix_regression.*`, `true_batch_b2_kv_fix.*`, `true_batch_b4_kv_fix.*`, `request_local_state_isolation.*`, and `timed_window_kv_fix_closure.*`.

Limits: formal timing matrix was intentionally not run; ragged true-batch smoke was not run in this closure.
