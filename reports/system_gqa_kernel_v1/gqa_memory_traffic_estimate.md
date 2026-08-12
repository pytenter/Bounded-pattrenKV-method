# GQA Memory Traffic Estimate

This is `ESTIMATED_LOGICAL_BYTES`, not hardware DRAM transaction accounting. The estimate uses the frozen V2 shape: `head_dim=128`, `PACK=16`, `group_size=128`, `GQA ratio=4`, `Mcent=16`, and two output-tile CTAs per full head.

For one KV group at 32K context, V2 compact tokens are `24384` in the mixed25 V2 lane. Per token baseline duplicate bytes include packed V, scale/zero, mask, and assignment repeated across four Q heads and output-tile CTAs. Candidate A stages these once per KV-head CTA group for the four Q heads.

- Baseline bytes / KV group: `6648832`
- Candidate bytes / KV group: `1662208`
- Estimated byte reduction: `75.00%`
- Theoretical duplicate factor: `4x`
- Actual reuse factor attempted: `4-way staged per CTA`

Detailed rows are in `memory_traffic_rows.csv`. Despite large logical byte reduction, measured latency regressed because the candidate uses 512 threads/block plus shared-memory staging and synchronization.
