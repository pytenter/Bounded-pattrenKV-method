# Optimization Scorecard

| Metric | Baseline | Fixed Capacity | Chunked Capacity |
|---|---:|---:|---:|
| 32K old bytes copied/token | 1172718.0 | 0.0 | 0.0 |
| 32K torch.cat events/token | 0.1015625 | 0.0 | 0.0 |
| 32K realloc events/token | 0.1015625 | 0.0 | 0.0 |
| 32K mutation us/token | 15.7228914758889 | 4.238172550685704 | 4.012086719740182 |
| 32K TPOT | not run | blocked | blocked |
| peak allocated | 303110656 | 268435456 | 268435456 |
| unused capacity | 0 | 16319616 | 590976 |
| capacity utilization | 1.0 | 0.9022885704826369 | 0.99609375 |
| correctness | PASS | PASS | PASS |
