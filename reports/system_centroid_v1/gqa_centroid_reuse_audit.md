# GQA Centroid Reuse Audit

## Model Shape

- Query heads: 32
- KV heads: 8
- GQA ratio: 4

## Current Behavior

The Value kernel grid maps `blockIdx.x` over `B*nh`, not `B*nh_kv`. The KV head is derived by `hk = hq / ratio`. Therefore four query heads sharing the same KV head each launch their own blocks and independently process the same KV-side compressed V payload, `v_pattern_mask`, `v_idx`, and centroid table.

## Reuse Opportunity

`GQA_CENTROID_REUSE_OPPORTUNITY = YES`

Data that can potentially be shared across the four query heads for a KV head:

- centroid table `C[hk]`
- Pattern assignment/index `v_idx[b, hk, :]`
- Pattern mask `v_pattern_mask[b, hk, :]`
- compressed V payload, scale, and zero

Data that cannot be shared directly:

- query-head-specific `alpha_q[b, hq, :]`
- per-query-head centroid attention mass `Sacc`, unless a redesigned kernel computes multiple Q heads cooperatively

## Scope Decision

No GQA redesign is implemented in S2B-2. The measured centroid fractions are already high enough to prioritize direct centroid-path optimization first; a later GQA-aware kernel can reuse this audit.
