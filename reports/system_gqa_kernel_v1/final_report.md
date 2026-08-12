# Phase S2B-3 Final Report

Classification: `GQA_AWARE_KERNEL_REDESIGN_NOT_SUPPORTED`

The experimental 4-Q-head GQA V2 kernel is numerically correct but not useful for production. The default backend remains the current baseline (`PATTERNKV_GQA_V_BACKEND=baseline` by default). Frozen algorithm semantics are unchanged, and the production per-warp histogram plus lane0 centroid table optimizations are preserved.

## Baseline

- 16K V2: `377.856 us`; V4: `286.720 us`; mixed-V: `846.848 us`.
- 32K V2: `509.952 us`; V4: `322.560 us`; mixed-V: `999.424 us`.

## Candidate A: 4-Q-Head CTA

- 16K V2: `723.968 us`, speedup `0.522x`; mixed-V `1216.512 us`, speedup `0.696x`.
- 32K V2: `1223.680 us`, speedup `0.417x`; mixed-V `1718.272 us`, speedup `0.582x`.

The candidate regressed because 512 threads/block, shared staging, and tile synchronizations outweighed the logical GQA byte reuse.

## Correctness

- Correctness cases: `48`
- All passed: `True`
- max_abs_error: `3.0517578125e-05`
- mean_abs_error max: `7.596099749207497e-09`
- relative_L2 max: `2.7791562388301827e-05`
- cosine min: `0.9999998807907104`
- NaN/Inf: `0/0`

Q-head mapping checks passed for Q0-Q31, including Q3->KV0, Q4->KV1, Q7->KV1, and Q8->KV2 boundaries.

## Decision

Do not switch production to the GQA-aware backend. Keep the experimental entry for benchmark/debug only, protected by `PATTERNKV_GQA_V_BACKEND=gqa` with safe fallback.

NEXT_TASK: `FIXED_PAGE_ABI`
