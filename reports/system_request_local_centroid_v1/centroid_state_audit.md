# Centroid State Audit

| symbol | current centroid shape | batch semantics | state owner | shared/request-local | required change | K affected? | V affected? | risk |
|---|---|---|---|---|---|---|---|---|
| _append_dynamic_centroids | [B,H,T,D] window -> pool[slot,H,M,D] | was batch-global | request slot | request-local | reduce T only, append per slot | YES | YES | high |
| pattern_gather_centroids | [H,M,D] | shared bank | static/shared legacy | shared | add request-aware gather | YES | YES | medium |
| pattern_gather_request_centroids | [B,H,M,D] | per active request | request slot view | request-local | new vectorized gather | YES | YES | medium |
| _assign_minmax_hnk | [H,M,D] | shared bank | legacy | shared | keep for B1/static | YES | YES | low |
| _assign_minmax_bhnk | [B,H,M,D] | per active request | request slot view | request-local | new masked assignment | YES | YES | medium |
| pack_mixed_v_pages | [H,M,D] or [B,H,M,D] | page pools | active request views | request-local aware | accept 4D centroids | NO | YES | medium |
| page_mixed_pool_value_kernel | [H,M,D] or [B,H,M,D] | centroid add addressing | active request row | request-local aware | address centroid by b when Bcent>1 | NO | YES | medium |
| serialize_cache/deserialize_cache | tuple | persistent cache | cache/request state | request-local | persist pool and slot indices | YES | YES | medium |
| validate_cache | 3D or 4D centroid | cache invariant | cache | request-local aware | allow 4D centroid views | YES | YES | low |
