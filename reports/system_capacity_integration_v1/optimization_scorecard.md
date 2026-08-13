| Metric | Baseline | Fixed | Chunked |
| --- | ---: | ---: | ---: |
| 32K mutation us/token | 90.370 | 75.434 | 75.355 |
| 32K old bytes copied/token | 25429905.0 | 0.0 | 212372.0 |
| 32K historical torch.cat/token | baseline path | 7.96875 | 7.96875 |
| 32K mixed-V us | 1054.208 | 777.728 | 779.264 |
| 32K TPOT ms | 122.484 | 113.518 | 113.411 |
| 32K tokens/s | 8.164 | 8.809 | 8.817 |
| historical materialized bytes | 0 | 0 | 0 |
| correctness | PASS | PASS | PASS |
