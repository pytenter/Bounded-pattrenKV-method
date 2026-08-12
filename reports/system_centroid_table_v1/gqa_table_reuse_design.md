# GQA Table Reuse Design

Candidate C is `NOT_IMPLEMENTED` in S2B-2B and deferred to a dedicated GQA-aware redesign.

Design sketch:

- Launch one CTA around a KV-head group rather than one independent Q head.
- Assign four warp groups to the four Q heads sharing that KV head.
- Keep separate per-Q-head `SAcc_q[c]` values because attention alpha differs per Q head.
- Stage the shared centroid table `C_kv[c, oc]` once per KV-head/output tile and reuse it for the four Q heads.
- Preserve the S2B-2A per-warp private histogram; do not fall back to one contended shared histogram.

Resource estimate:

- Centroid staging for one full KV-head table at current config: 4096 bytes.
- Current per-warp histogram storage: `4 warps * 16 centroids * 4 bytes = 256` bytes per CTA.
- If one CTA handles four Q heads while retaining private histograms, histogram storage could grow to about `4 Q heads * 4 warps * 16 * 4 = 1024` bytes plus staged table.
- Expected extra synchronization: at least one CTA sync after table staging and one after histogram completion.
- Risks: changed grid mapping, increased shared memory, register pressure from multiple Q-head accumulators, and occupancy shifts.

Decision: defer. Candidate B already gives a same-run mixed-V win with exact correctness; GQA reuse should be isolated in the next phase.
