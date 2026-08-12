# Phase S2B-2B Final Report

Classification: `CENTROID_TABLE_LOCAL_OPTIMIZATION_SUPPORTED`

Candidate B is retained in production. The frozen algorithm is unchanged: selector, V4 token identities/fraction, quantization formula, centroids, sink/recent/residual, group size, and cache layout are untouched. The S2B-2A per-warp private histogram remains the production histogram path.

## Decomposition

The post-histogram table component remains material:

- V2 16K: `225.280 us FULL`, `150.528 us NO_TABLE`, table estimate `74.752 us` = `33.18%`.
- V2 32K: `448.512 us FULL`, `295.936 us NO_TABLE`, table estimate `152.576 us` = `34.02%`.

All decomposition CVs were <= 5%. `TABLE_ONLY` was not implemented; table cost is marked `APPROXIMATE_ABLATION_COST_FULL_MINUS_NO_TABLE`.

## Candidate Decisions

- Candidate A active centroid skip: not implemented. Active centroid audit showed 16/16 active centroids across all measured workloads.
- Candidate B lane0 table contribution: implemented and retained. Baseline non-lane0 lanes redundantly computed centroid table contribution even though only lane0 output used it; candidate keeps the same lane0 accumulation and skips the unused non-lane0 work.
- Candidate C GQA reuse: not implemented in this phase. Audit confirms 4x duplicate table-read opportunity across Q heads sharing a KV head, but exploiting it requires CTA/grid redesign while preserving private histograms.

## Performance

- V2 32K: `448.512 -> 307.200 us`, `1.460x`.
- Mixed-V 32K same-run: `1072.128 -> 1017.856 us`, `1.053x`, candidate CV `3.49%`.
- E2E 32K median TPOT same-run: `115.434 -> 115.335 ms/token`, `1.001x`.

## Correctness

`LANE0_TABLE_FULL` matched `PER_WARP_HIST_FULL` exactly in decomposition (`max_abs=0`, `relative_L2=0`, `cosine=1`). Existing reference-based correctness also passed for mixed layouts, all-same centroid, all-centroids-active, mask-zero/full, random/skewed assignment, GQA ratio 4, and V4 guard.

NEXT_TASK: `GQA_AWARE_KERNEL_REDESIGN`
