# Persistent Slot Contract

Persistent state is owned by `slot_id`, not `row_idx`.

During an active iteration:

1. The manager builds `row_idx -> slot_id -> request_id` mappings.
2. Existing ragged assembly creates a temporary active batch in the requested row order.
3. Decode mutates the active batch.
4. The manager commits each row back into the mapped slot.

Middle-row removal and reorder therefore change only `row_idx`; the persistent slot remains stable.

