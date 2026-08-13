| Metric | Tight K Reader | Strided Capacity K | Change |
| --- | ---: | ---: | ---: |
| QK @8K | 74.752 us | 99.328 us | 32.88% |
| QK @16K | 124.928 us | 182.272 us | 45.90% |
| QK @24K | 186.368 us | 271.360 us | 45.60% |
| QK @32K | 247.808 us | 331.264 us | 33.68% |
| max abs | 0 | 0 | PASS |
| cosine min | 1 | 1 | PASS |
| historical materialized bytes | 0 | 0 | 0 |
| logical K tokens | benchmark context | benchmark context | logical-only |
| capacity tokens | tight logical | explicit slack | no page lookup |
| reader default status | default | experimental only | unchanged |
