# Root Cause Analysis

- Dominant bottleneck: `mixed host sync, page materialization, tiny launches, and fragmented matmul`
- Classification: `PAGE_BATCH_MIXED_OVERHEAD`
- HOST_SYNC_SIGNIFICANT: `True`
- PAGE_MATERIALIZATION_SIGNIFICANT: `True`
- KERNEL_LAUNCH_SIGNIFICANT: `True`
- MATMUL_FRAGMENTATION_SIGNIFICANT: `True`
- TEMP_ALLOCATION_SIGNIFICANT: `True`
