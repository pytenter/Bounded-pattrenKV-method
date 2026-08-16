# Row Identity Audit

Core contract introduced in `models/request_lifecycle.py`:

- `request_id`: external logical request identity.
- `slot_id`: persistent runtime state owner.
- `row_idx`: current active batch row for one iteration.
- `page_id`: cache storage/page ownership unit.

Findings:

- `models/segmented_cache.py` uses batch row indices when assembling or slicing ragged active caches. These rows are transient and must be interpreted through request-local length vectors.
- `request_total_tokens`, `request_packed_k_tokens`, `request_packed_v_tokens`, and `request_packed_v4_tokens` are persistent slot/request metadata but appear as row-indexed tensors inside an assembled active cache.
- `v_causal_importance` is persistent request-local logical history. Ragged fixes already map physical attention segments back to logical request indices before mutation.
- `PatternKVCentroidStatePool` already has slot-indexed active state. The lifecycle manager keeps a stable `slot_id` across row reorder/removal and extracts committed active rows back into their owning slot.
- `PatternKVBatchMetadata.seq_lens` and page tables are active-row metadata after page-pool merge. They must be rebuilt from slot-owned caches rather than reused as request identity.
- Hidden row assumption risk remains highest in any direct `tensor[row]` access outside active row mapping. The new tests force `[A,B,C,D] -> [A,C,D]` so C/D row indices change while slot IDs remain stable.

No frozen algorithm settings were modified.

