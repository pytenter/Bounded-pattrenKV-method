# Cache Mutation Static Audit

Relevant dynamic mutation sites are in `models/segmented_cache.py`.

## Categories

- `packed_k_payload`, `packed_k_scale`, `packed_k_zero`: `_cat_packed_k`.
- `packed_v2_payload`, `packed_v2_scale`, `packed_v2_zero`: `_cat_packed_v` and mixed V2 `_cat_v_payload`.
- `packed_v4_payload`, `packed_v4_scale`, `packed_v4_zero`: mixed V4 `_cat_v_payload`.
- `precision_mask`: `_cat_mixed_packed_v` appends `v_precision_mask`.
- `assignments`: `_cat_assignment` for K assignments and V assignment indices.
- `pattern_mask`: `_cat_assignment` for V Pattern mask.
- `causal_importance`: `update_value_causal_importance` grows the importance tensor by allocating a new state and copying old values.
- `recent_pending`: `_cat_token` appends recent and pending K/V during decode.
- `sink`: `_cat_token` can fill sink early, but the frozen profiled contexts already start with a full sink.
- `centroids`: `_append_dynamic_centroids` uses `torch.cat` for centroid banks; in the profiled synthetic decode this is not a recurring dominant mutation.

All byte counts in `cache_mutation.csv` are estimated as old input bytes plus
appended input bytes. They are not hardware DRAM traffic measurements.
