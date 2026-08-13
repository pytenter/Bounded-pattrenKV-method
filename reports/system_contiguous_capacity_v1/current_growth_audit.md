# Current Growth Audit

- Growing historical streams are appended in `models/segmented_cache.py` through `_cat_packed_k`, `_cat_v_payload`, `_cat_assignment`, and precision-mask concatenation.
- `append_decode_rolling` also grows/rolls `recent_k`, `recent_v`, `pending_k`, and `pending_v` with `_cat_token`.

| Stream | Growth Site | Notes |
|---|---|---|
| packed_k | torch.cat along token dim 3 | dtype `torch.int32` |
| packed_k_scale | torch.cat along token dim 3 | dtype `torch.float16` |
| packed_k_zero | torch.cat along token dim 3 | dtype `torch.float16` |
| packed_v | torch.cat along token dim 2 | dtype `torch.int32` |
| packed_v_scale | torch.cat along token dim 2 | dtype `torch.float16` |
| packed_v_zero | torch.cat along token dim 2 | dtype `torch.float16` |
| packed_v4 | torch.cat along token dim 2 | dtype `torch.int32` |
| packed_v4_scale | torch.cat along token dim 2 | dtype `torch.float16` |
| packed_v4_zero | torch.cat along token dim 2 | dtype `torch.float16` |
| v_precision_mask | torch.cat along token dim 1 | dtype `torch.uint8` |
| k_assignments | torch.cat along token dim 2 | dtype `torch.int64` |
| v_assignment_idx | torch.cat along token dim 2 | dtype `torch.int64` |
| v_pattern_mask | torch.cat along token dim 2 | dtype `torch.uint8` |
