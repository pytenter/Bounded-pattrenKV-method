# Optimization Scorecard

| Candidate | V2@32K | Mixed-V@32K | TPOT@32K | Correctness | Shared Mem | Decision |
|---|---:|---:|---:|---|---:|---|
| Baseline | 509.952 us | 999.424 us | NOT_RUN | PASS | 256 B | KEEP_DEFAULT |
| 4-Q-head CTA | 1223.680 us (0.417x) | 1718.272 us (0.582x) | NOT_RUN | PASS | ~8192 B | REGRESSION, experimental only |
| 2-Q-head partial reuse | NOT_RUN | NOT_RUN | NOT_RUN | N/A | projected lower | NOT_IMPLEMENTED |

CV was <= 5% for all measured 8K/16K/32K baseline and candidate rows.
