# Final Backend Readiness V3

```json
{
  "b1_post_cuda_kv_fix_regression": "PASS",
  "branch": "sys/qwen3-8b-v100-system-generalization-v1",
  "centroid_swap_isolation": "PASS",
  "classification": "QWEN_COMPRESSED_TRUE_BATCH_REQUEST_LOCAL_CUDA_KV_CLOSED",
  "cuda_arch": "sm_70",
  "cuda_extension_rebuilt": true,
  "formal_timing_allowed": false,
  "formal_timing_run": false,
  "gpu0_3_touched": false,
  "gpu_used": "physical GPU4 via CUDA_VISIBLE_DEVICES=4",
  "k_request_local_centroid_oracle": "PASS",
  "ragged_true_batch_smoke": "NOT_RUN",
  "request_local_state_isolation": "PASS",
  "start_head": "5dd03dcf2dec1e91aed52498299fa1d6a73329e0",
  "status": "PASS",
  "timed_window_kv_fix_closure": "PASS",
  "true_batch_b2_kv_fix": "PASS",
  "true_batch_b4_kv_fix": "PASS",
  "v_request_local_centroid_oracle": "PASS",
  "verification": {
    "b1_regression": "PASS",
    "b2_model_smoke": "PASS",
    "b4_model_smoke": "PASS",
    "cuda_operator_oracles": "PASS",
    "git_diff_check": "PASS",
    "py_compile": "PASS",
    "pytest_tests/test_qwen3_compressed_backend.py": "PASS_14",
    "timed_window_smoke": "PASS"
  }
}
```
