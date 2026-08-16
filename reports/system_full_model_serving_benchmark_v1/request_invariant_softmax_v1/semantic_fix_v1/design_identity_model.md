# Identity Model

- `request_id`: persistent logical request identity used by tests and reports.
- `slot_id`: persistent runtime/storage owner for centroid/page state during a request lifetime.
- `active_row`: ephemeral dense row for the current decode iteration only.
- `flat_split_idx`: ephemeral iteration-local identity for fixed-split softmax workspace.
- `request_local_split_idx`: stable only inside one request logical topology with split size 128.
- `page_id`: physical compressed KV page identity.

The semantic fix keeps partial softmax workspace iteration-local and leaves `PATTERNKV_FIXED_SPLIT_SOFTMAX` default off. The corrected ownership bug was in centroid state used by ragged per-row flush: a sliced single-request row cache was using batch-level scalar centroid update counts, which could expose stale centroid tail capacity from another slot. The fix uses slot-local pool counts for the single-slot active centroid view, and pack-time centroid tensors are trimmed to valid counts before request-local assignment/selection.
