# Multi-Request Iteration Plan Design

Persistent state remains owned by request/slot/cache lifetime, while `active_row` is only the current dense row. The serving harness now keeps an `ActiveBatchState` keyed by the ordered request ids for the current decode iteration. If membership is unchanged, the next iteration consumes the batched compressed cache returned by the previous model decode.

Shared per-iteration identity is `active_row -> request_id` plus the cache-resident page/ragged/fixed-split metadata already carried by each layer cache. Layer-local numerical state remains in each layer's compressed cache object. The harness no longer reconstructs all layer metadata or extracts every active row between steady decode iterations.

On membership changes, the harness falls back to explicit row extraction for surviving requests before rebuilding a new active batch. That preserves dynamic membership semantics without paying the copy cost during steady true-batch decode. Empty-active prefill uses `prefill_active_batch` to start directly from the batched cache and avoid the initial split/assemble round trip.

B8 support required removing a stale `pack_mixed_v_pages` MVP guard. The page packing implementation already iterated over arbitrary positive batch size and generated request indptr/page tables from `bsz`.
