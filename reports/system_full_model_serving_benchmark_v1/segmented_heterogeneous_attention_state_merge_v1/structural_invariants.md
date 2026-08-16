# Structural Invariants

- `true_batch_preserved = true` in old and new C2048 B1 workers.
- `serial_request_forward_dispatches = 0`.
- `serial_attention_dispatches = 0`.
- `serial_mlp_request_dispatches = 0`.
- `serial_rmsnorm_request_dispatches = 0`.
- `fallback_count = 0`.
- `historical_fp16_k_materialization = 0`.
- `historical_fp16_v_materialization = 0`.
- `page_batch_pack_calls_decode8 = 0` in diagnostic profile.
- Frozen CAUSAL-V4@25% algorithm parameters were not changed.

The new path removed global score concat and global full probability materialization counters, but it increased segment-level calls and end-to-end TPOT.

