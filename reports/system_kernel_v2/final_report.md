# S2B-1 Final Report: Optimize V2 CUDA Path Only

## Outcome

`V2_KERNEL_LOCAL_OPTIMIZATION_SATURATED`

S2B-1 evaluated V2-only local CUDA launch/memory-access optimization candidates and kept no CUDA code change. The V2/V4 algorithm, selector behavior, packing/cache layout, scale/zero, centroid semantics, sink/recent/residual/group-size semantics, and framework code paths remain unchanged.

## Baseline

Immutable baseline was captured in `baseline.csv` and `baseline_e2e.csv` before any CUDA optimization attempt.

Key baseline V2 medians:

- 8K context: 115.712 us
- 16K context: 256.000 us
- 32K context: 501.760 us

## Candidate A

Candidate A tested V2 `threads.y=8` launch shape, motivated by 2-bit `PACK=16` and `OC/PACK=8` output tiles.

The fixed variant passed correctness but was not acceptable:

- 8K V2 speedup: -4.425%
- 16K V2 speedup: -4.400%
- 32K V2 speedup: 8.571%

A K-gated variant was also tested, but the 32K win did not reproduce under rerun. Final 32K gated rerun V2 speedup: -4.286%.

## E2E Diagnostic

Candidate A fixed 8-warps E2E diagnostic showed lower median TPOT at 16K/32K, but it was not kept because the microkernel gate requires stable local V2 speedup without shorter-context regressions.

See `e2e_summary.csv`.

## Candidate B/C

Candidate B GQA reuse and Candidate C metadata reuse were audited but not implemented. Both require non-local restructuring or layout/contract changes, outside S2B-1 constraints.

## Final Gate

- Correctness: PASS for measured candidates
- Stable V2 >=3% speedup: FAIL
- V4 unchanged by retained code: PASS, because no CUDA optimization is retained
- CUDA source restored: PASS
- Final classification: `V2_KERNEL_LOCAL_OPTIMIZATION_SATURATED`
