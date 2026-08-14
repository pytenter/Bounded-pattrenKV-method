# Cost Benefit Matrix

| Case | P1 median ms | P2 median ms | P2 overhead | Hidden drift ratio P1/P2 | Logit drift ratio P1/P2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ctx512 B2 | 2243.34 | 2248.62 | 0.24% | 1.02x | 1.11x |
| ctx512 B4 | 4141.40 | 4140.66 | -0.02% | 1.06x | 1.10x |

Decision rule outcome: low cost but small numerical benefit. P2 does not justify becoming the default production prefill path from these measurements, but it is cheap enough to preserve as an explicit strict mode.
