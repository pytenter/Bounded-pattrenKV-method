# Model Mutable State Audit

- `v_causal_importance`: `CACHE_LOCAL`
- `v_oracle_importance`: `CACHE_LOCAL`
- `selector_task_key`: `DEBUG_ONLY`
- `attention_layer_k_base`: `REQUEST_LOCAL_MODEL_STATE_RESET_ON_PREFILL`
- `attention_layer_v_centroids`: `REQUEST_LOCAL_MODEL_STATE_RESET_ON_PREFILL`
- `centroid_state_pool`: `CACHE_LOCAL`
- `operator_ready_page_pools`: `CACHE_LOCAL`
- `profiling_counters`: `DEBUG_ONLY`
- `ragged_decode_counters`: `DEBUG_ONLY`
- `model_global_math_state_detected`: `False`
