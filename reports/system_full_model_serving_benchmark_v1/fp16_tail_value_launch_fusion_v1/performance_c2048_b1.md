# Performance C2048 B1

## Formal Status

`FORMAL_PERFORMANCE_COMPLETE`

Physical GPU2 was clean and used for both OLD and FUSED in the same session.

## Required Protocol

- `context = 2048`
- `batch = 1`
- `decode = 8`
- `repeats = 3 old + 3 fused`
- `physical GPU = 1`

## Formal Metrics

- `OLD_CAUSAL_tpot_ms_median = 242.4437877489254`
- `FUSED_CAUSAL_tpot_ms_median = 199.60261625237763`
- `absolute_ms_saved = 42.84117149654776`
- `speedup = 1.214632314450124`
- `tok/s = 5.009954372219248`
- `ratio_vs_FP16_reference = 7.003600570258865` using `28.5 ms/token`
