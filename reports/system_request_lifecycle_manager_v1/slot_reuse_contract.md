# Slot Reuse Contract

Slot reuse is allocation from the free-list after release.

The reused slot:

- keeps stable `slot_id`
- increments generation
- receives the new `request_id`
- receives a freshly initialized cache
- rejects stale active row mappings through generation checks

Poison-before-reuse tests mutate old recent/pending/packed/importance/precision/length/centroid state before release and verify the next request starts clean.

