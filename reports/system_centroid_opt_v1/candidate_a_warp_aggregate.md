# Candidate A: Warp-local Aggregation Before Atomic

## Hypothesis

Use warp-level grouping to combine lanes with the same centroid assignment before updating `s_Sacc[idx]`. This should reduce logical shared-memory `atomicAdd` calls, especially under skewed assignment distributions.

## Implementation

Candidate A was implemented as benchmark-only debug mode `WARP_AGG_FULL`. Production dispatch remained unchanged during evaluation.

For each token slot handled by the histogram warp:

1. compute `valid`, `idx`, and `attention mass`
2. use `__ballot_sync` and `__match_any_sync` to identify lanes in the warp with the same `idx`
3. use `__shfl_sync` to sum peer contributions
4. only the peer-group leader performs `atomicAdd(&s_Sacc[idx], sum)`

## Result

The logical atomic count decreased, but the warp-level matching/shuffle overhead dominated. Candidate A regressed normal and contention-heavy workloads and is rejected.

## Decision

`REGRESSION`
