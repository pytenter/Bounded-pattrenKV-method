# Pool Update Design

The runtime does not call `build_operator_ready_page_pools()` per decode token. Packing still occurs on 128-token chunk boundaries; each flush builds pools only for the new chunk and appends them to cache-resident pools. `operator_ready_pool_full_rebuilds` remains zero.
