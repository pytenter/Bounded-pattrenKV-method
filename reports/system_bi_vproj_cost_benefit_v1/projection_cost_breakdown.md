# Projection Cost Breakdown

Kernel dispatch counters confirm P2 uses the existing V2 BI projection kernel for both K and V during prefill:

- P1 B4: 32 BI projection calls, all K prefill; V prefill remains normal.
- P2 B4: 64 BI projection calls, split as 32 K prefill and 32 V prefill.
- P2 decode-one: 0 BI K/V projection calls, 32 normal K decode calls, 32 normal V decode calls.

Whole-prefill median overhead at ctx512 was 0.24% for B2 and -0.02% for B4. A separate profiler-level K/V/other timing decomposition was not collected in this run.
