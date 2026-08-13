# Optimization Scorecard

| Metric | Tight contiguous | Strided capacity | Change |
| --- | ---: | ---: | ---: |
| V2 @8K median us | 115.8880 | 118.7840 | 2.50% |
| V2 @16K median us | 213.7600 | 223.2320 | 4.43% |
| V2 @24K median us | 311.2960 | 325.6320 | 4.61% |
| V2 @32K median us | 410.6240 | 431.1040 | 4.99% |
| max abs | 0 reference | 0 | PASS |
| cosine min | 1 reference | 1 | PASS |
| historical materialized bytes | 0 | 0 | 0 |
| torch.cat calls | 0 | 0 | 0 |
| logical tokens processed | logical K | 61056 correctness tokens | logical only |
| capacity tokens | tight K | up to 33792 | stride only |
| reader default status | production default | experimental nondefault | unchanged |
