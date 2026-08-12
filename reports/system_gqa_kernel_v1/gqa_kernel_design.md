# GQA Kernel Design

## Design A: 4-Q-Head CTA

- Threads/block: `dim3(32, 4, 4)` = 512 threads, 16 warps.
- CTA ownership: one KV head group and four Q heads sharing it.
- Warps: `threadIdx.z` selects Q slot, `threadIdx.y` selects output tile.
- Shared memory/block at current shape: about 8192 bytes for histogram, staged VQ, scale, zero, mask, assignment, and centroid tile.
- Synchronization: per token tile after staging, per token tile before reuse, plus centroid staging synchronization.
- Global load reuse: attempts 4-way reuse for V2 packed payload, scale, zero, mask, assignment, and centroid tile.
- Histogram compatibility: preserved as private `SAcc[qslot][warp][centroid]`; no return to contended shared histogram.
- Centroid compatibility: lane0-only table contribution is preserved.
- Recent-window handling: computed per Q head; not staged in S2B-3 candidate.
- Expected speedup source: fewer logical duplicate KV-owned loads across Q heads.
- Main risk: 16 warps/block, shared-memory staging, and synchronizations dominate the small per-token work.

Implemented: YES.

## Design B: 2-Q-Head Partial Reuse

- Threads/block: projected `dim3(32, 4, 2)` = 256 threads, 8 warps.
- Reuse factor: 2-way instead of 4-way.
- Shared memory: lower than Design A, but still requires staging and sync.
- Expected benefit: less occupancy pressure than Design A.
- Risk: after Design A regressed strongly, partial reuse still has the same staging/sync structure with less theoretical byte reduction.

Implemented: NO. It was not pursued after the 4-Q-head candidate was a stable regression.

`DESIGN_SELECTED=Design A for experimental implementation only`

Reason: it directly tests the maximum available GQA reuse while preserving the frozen algorithm and existing histogram/table optimizations. It is not selected for production.
