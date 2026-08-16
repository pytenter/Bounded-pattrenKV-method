# Lifecycle Architecture

`RequestLifecycleManager` owns a fixed pool of persistent slots.

- `request_to_slot` maps logical request id to persistent slot id.
- `slots[slot_id].cache` stores request-owned `PatternQuantizedKVCache` state.
- `active_request_ids` stores active request ids only; active rows are rebuilt per iteration.
- `build_active_cache(order)` assembles a true ragged batch from slot caches and returns `ActiveRowMapping` records.
- `commit_active_cache(cache, mappings)` extracts each active row after decode and writes it back to the original slot.

This preserves true batched execution: requests are not forwarded one by one by the lifecycle manager.

