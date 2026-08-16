# Long Decode

| Method | Context | Batch | Decode | Median TPOT ms | Throughput tok/s | Valid runs |
|---|---:|---:|---:|---:|---:|---:|
| CAUSAL_V4_25_FULL_MODEL | 4096 | 1 | 256 | 372.118 | 2.687 | 3 |
| FP16_FULL_MODEL | 4096 | 1 | 256 | 29.916 | 33.427 | 3 |
| KIVI_PAPER_G128_FULL_MODEL | 4096 | 1 | 256 | 60.004 | 16.665 | 3 |
| PATTERNKV_PAPER_FULL_MODEL | 4096 | 1 | 256 | 329.994 | 3.030 | 3 |

## CAUSAL Boundary Accounting

- `page_batch_pack_calls`: [64, 64, 64]
- Median page-pack calls per generated token: 0.2500.
- Page-pack kernel time is unavailable because hot-path profiling is disabled during formal timing.
