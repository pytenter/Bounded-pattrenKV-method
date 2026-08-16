# CTA Scheduling Audit

## Current Work Units
- QK: approximately request/head times packed-token tile. For B1/context2048, CTA count is roughly `B*nh*ceil((T/16)/8) = 512`.
- Page mixed Value: request/query-head/output-channel, so B1 uses `B*nh*head_dim = 4096` CTAs and each CTA loops over context tokens.
- Fixed split softmax: request/head, so B1 uses only 32 CTAs and loops over logical split IDs internally.

## Low-B Utilization
- Page Value has enough CTAs at B1; low-B wave quantization is unlikely to dominate that kernel.
- Fixed-split softmax has low CTA count at B1, but the kernel is only about 2.7 ms/iteration in profile ranges and is not the primary issue.
- QK has moderate CTA count and more potential for wave/tail effects, but measured compressed QK is only about 3.8 ms/iteration.

## Future Scheduling Feasibility
`DYNAMIC_PHYSICAL_TILE_SCHEDULING_FEASIBLE`: logical split IDs can remain deterministic while physical tiles are scheduled independently, as long as partial states are keyed by request/head/logical split and merged left-to-right.

`PERSISTENT_CTA_WORK_QUEUE_FEASIBLE`: unclear but plausible. Interface work would be needed to expose the IterationPlan as a compact device work queue for QK/value/softmax tiles and preserve deterministic output mapping.
