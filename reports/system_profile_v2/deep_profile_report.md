# Phase S2A Deep Systems Profile

## Frozen algorithm status

- Algorithm changed: `NO`
- Frozen tag: `causal-v4-25-aime24-v1`

## S1 fused-kernel status

- Kernel speedup @16K: `2.699x`

## S1.5 E2E status

- S1.5 fused TPOT @16K: `113.052 ms/token`
- S1.5 fused TPOT @32K: `118.019 ms/token`

## Methodology

- Profile-off decode measures real TPOT at 8K/16K/32K.
- Profile-on decode collects cache mutation categories.
- Standalone same-shape mixed-V calls with synchronization decompose wrapper, layout, V2/V4 lanes, and output reduce.

## Profiling overhead caveat

- Component decomposition remains diagnostic; profile-off TPOT is the performance source of truth.

## Mixed-V call path

See `mixed_v_static_audit.md`.

## Mixed-V host vs CUDA breakdown

| T | wrapper us/call | CUDA us/call | host/dispatch us/call | launches/call |
|---:|---:|---:|---:|---:|
| 8192 | 1105.788 | 1068.411 | 37.377 | 2.00 |
| 16384 | 1197.887 | 1161.021 | 36.866 | 2.00 |
| 32768 | 1523.521 | 1493.893 | 29.628 | 2.00 |

## V2 vs V4 compute

Detailed rows are in `mixed_v_component_breakdown.csv`; both lanes launch separately.

@32K:

- V2 tokens/call: `24384`
- V4 tokens/call: `8128`
- V2 compute: `559.442 us/call`, `0.02294 us/token`
- V4 compute: `248.975 us/call`, `0.03063 us/token`
- V4 is more expensive per token, but V2 dominates total time because it owns 75% of tokens.

## Mapping/layout overhead

- Dominant mixed-V subcomponent @32K: `mixed_v_v2_compute`
- Mapping prepare @32K: `127.692 us/call`
- V2 layout prepare @32K: `210.325 us/call`
- V4 layout prepare @32K: `193.004 us/call`
- Host/dispatch estimate @32K: `29.628 us/call`

## Temporary allocations

See `mixed_v_temp_allocations.csv`.

## Cache mutation categories

See `cache_mutation.csv` and `cache_mutation_static_audit.md`.

@32K:

- Total mutation events: `20832`
- Top category by calls: `recent_pending`
- Top category by estimated bytes: `recent_pending`
- `recent_pending` estimated copy bytes/token: `25361408`
- `causal_importance` estimated copy bytes/token: `4202432`

## Cache copy volume

- 16384 bytes/token: `30624416`
- 32768 bytes/token: `35904160`
- 16K->32K ratio: `1.172`
- Classification: `constant`

## Context scaling

See `context_scaling.csv`.

## nsys findings

- NSYS_AVAILABLE=`false`

## ncu findings

- NCU_AVAILABLE=`false`

## Approximate Amdahl analysis

See `amdahl_estimate.md`.

## System-design interpretation

Mixed-V CUDA execution is the largest remaining root cause; host/layout overhead is small, and cache copy bytes/token growth from 16K to 32K is 1.172 (constant).

## Final decision

`S2B_MIXED_V_KERNEL_OPTIMIZATION`
