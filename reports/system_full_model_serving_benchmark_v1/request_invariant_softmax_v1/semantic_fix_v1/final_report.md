# Final Report

Fixed-split CUDA softmax semantics are now closed for the requested ragged gates. The production default remains off; opt-in is still `PATTERNKV_FIXED_SPLIT_SOFTMAX=1`.

Root cause: `ACTIVE_ROW_AS_PERSISTENT_IDENTITY` was not the observed cause. The first failing state was boundary flush packing: a single-slot row cache used batch-level scalar centroid update counts, exposing stale centroid tail capacity from another slot. The fix makes active centroid views use slot-local pool counts and trims pack-time centroid tensors to valid counts.

Correctness after fix: B2 multi-step PASS, B2 reorder PASS, B4 PASS, independent flush PASS with D=13 C=14 B=15 A=16, dynamic/lifecycle/continuous regressions PASS, full pytest PASS (`1007 passed`).

Performance sanity: B1 context=2048 decode=4 opt-in attention softmax is `14.485` ms/token and fixed-split kernel is `2.435` ms/token, preserving the previous major softmax speedup direction. Full serving benchmark remains not closed.
