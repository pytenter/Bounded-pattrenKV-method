# S2B-1 V2 Kernel Optimization Scorecard

## Decision

`V2_KERNEL_LOCAL_OPTIMIZATION_SATURATED`

No CUDA optimization is kept. The source and rebuilt extension were restored to the pre-candidate V2/V4 launch semantics after Candidate A failed the stability gate.

## Candidate Table

| Candidate | V2 kernel change | Mixed-V change | 32K TPOT change | Correctness | Decision |
|---|---:|---:|---:|---|---|
| A memory access, fixed 8 warps | 32K V2 +8.571%, but 8K -4.425% and 16K -4.400% | 32K mixed-V -3.327% | +2.456% diagnostic only | PASS | REGRESSION |
| A memory access, K-gated rerun | 32K V2 -4.286% | 32K mixed-V -1.799% | not run | PASS | REGRESSION |
| B GQA reuse | not implemented | not measured | not measured | not measured | NOT_IMPLEMENTED |
| C metadata reuse | not implemented | not measured | not measured | not measured | NOT_IMPLEMENTED |

## Candidate A: V2 Memory/Launch Access

Hypothesis: for 2-bit V, `PACK=16` leaves `OC/PACK=8` output tiles. Increasing V2 `threads.y` from 4 to 8 can cover all output tiles in a single block-y group and reduce block scheduling overhead at long context.

Result:

- Fixed 8-warps V2 launch passed correctness, but regressed 8K and 16K V2 medians and mixed wrapper latency.
- 32K fixed 8-warps showed one microbench win: V2 speedup `8.571%`.
- A safer K-gated version did not reproduce the 32K win: first gated 32K speedup `-3.980%`; rerun speedup `-4.286%`.
- E2E diagnostic run showed median TPOT improvement, but because the microkernel win was not stable and the fixed variant regressed shorter contexts, the CUDA change was rejected.

## Candidate B: GQA Reuse

Documented in `gqa_reuse_design.md`. Not implemented in S2B-1 because real reuse needs non-local kernel restructuring, cooperative query-head tiling, and new synchronization/resource tradeoffs.

## Candidate C: Metadata Reuse

Not implemented. Useful metadata reuse would require changing mask/index layout or selector/cache contracts; this violates the local V2-only optimization scope.

## Correctness

All measured Candidate A variants passed the numerical gate:

- max_abs <= 3.0517578125e-05
- cosine >= 0.9999998807907104
- no NaN/Inf

## Final Classification

The current V2 CUDA kernel has no stable local launch/memory-access optimization under the S2B-1 constraints. Future work should target `CENTROID_PATH_OR_FIXED_PAGE_REDESIGN`, not additional random block-size tuning.
