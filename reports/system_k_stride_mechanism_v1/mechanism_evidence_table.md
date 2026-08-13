# Mechanism Evidence Table

| Hypothesis | Evidence | Result |
| --- | --- | --- |
| Physical capacity scan | MEASURED: 8K logical with 8K/16K/32K capacity stays around 95-97 us. | REJECTED |
| Poor coalescing | STATIC: tight lane/channel offsets contiguous; strided offsets spaced by cap_packs. No NCU sectors measured. | INCONCLUSIVE |
| Extra memory transactions | STATIC supports possible extra sectors, but no NCU transaction counters. | INCONCLUSIVE |
| Extra L2 traffic | No NCU L2 sector/byte counters available. | INCONCLUSIVE |
| Extra address arithmetic | STATIC+SASS: K IMAD rises 486 -> 918 and IADD rises 173 -> 421; V2 changes are much smaller. | SUPPORTED |
| Register pressure | cuobjdump resource: tight K 69 regs/thread; strided K 63; no spills. | REJECTED |
| Occupancy loss | Same launch geometry; lower register count for strided K; no spill/local memory. | REJECTED |
| Vector-load loss | STATIC: generic strides prevent tight linear channel-tile addressing; SASS load count increases modestly, but vector transaction counters unavailable. | INCONCLUSIVE |
| Kernel launch difference | Launch geometry identical for tight and strided K: 32x8 threads, same blocks for 32K, same dynamic shared memory. | REJECTED |
