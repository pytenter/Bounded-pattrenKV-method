# Request Slot Row Mapping

The runtime distinguishes:

- request identity: logical input sequence
- active batch row: current dense model row
- centroid slot: storage slot in `PatternKVCentroidStatePool`

The B2 semantic oracle reports `centroid_state_indices = [0, 1]`; reorder `[1,0]` preserves generated top-1 sequences relative to independent B1 execution. This validates that centroid banks and assignments follow the request row rather than being collapsed to a shared bank.
