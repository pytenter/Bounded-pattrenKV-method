# Formal Readiness Freeze

```json
{
  "CLASSIFICATION": "QWEN3_V100_COMPRESSED_BACKEND_READY_FOR_FIXED_BATCH_FORMAL",
  "FORMAL_FIXED_BATCH_TIMING_ALLOWED": "YES",
  "checks": {
    "B1_SEMANTIC_REGRESSION": "PASS",
    "CENTROID_SWAP_ISOLATION": "PASS",
    "K_REQUEST_LOCAL_CENTROID_GATE": "PASS",
    "RAGGED_TRUE_BATCH_SMOKE": "NOT_RUN",
    "REQUEST_LOCAL_STATE_ISOLATION": "PASS",
    "TIMED_WINDOW_PURITY": "PASS",
    "TRUE_BATCH_B2": "PASS",
    "TRUE_BATCH_B4": "PASS",
    "V_REQUEST_LOCAL_CENTROID_GATE": "PASS"
  },
  "cuda_arch": "sm_70",
  "formal_branch": "exp/qwen3-8b-v100-system-formal-v1",
  "formal_head": "322059bd8952065ec29bebc5f73e1472447b4c08",
  "ragged_true_batch_smoke_classification": "NON_BLOCKING_FOR_THIS_FIXED_BATCH_FORMAL_PROTOCOL",
  "source_backend_head": "322059bd8952065ec29bebc5f73e1472447b4c08",
  "status": "PASS",
  "timestamp": "2026-08-27T23:56:56+0800",
  "zero_counter_checks": {
    "fallback_count": 0,
    "historical_fp16_k_materialization_calls": 0,
    "historical_fp16_k_materialized_bytes": 0,
    "historical_fp16_v_materialization_calls": 0,
    "historical_fp16_v_materialized_bytes": 0,
    "serial_attention_dispatches": 0,
    "serial_request_forward_dispatches": 0
  }
}
```
