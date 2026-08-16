# Long Decode

| Method | Context | Batch | Decode | Median TPOT ms | Throughput tok/s | Valid runs |
|---|---:|---:|---:|---:|---:|---:|
| CAUSAL_V4_25_FULL_MODEL | 4096 | 1 | 256 | 159.530 | 6.268 | 3 |
| FP16_FULL_MODEL | 4096 | 1 | 256 | 28.663 | 34.888 | 3 |
| KIVI_PAPER_G128_FULL_MODEL | 4096 | 1 | 256 | 56.832 | 17.596 | 3 |
| PATTERNKV_PAPER_FULL_MODEL | 4096 | 1 | 256 | 154.646 | 6.466 | 3 |

## CAUSAL Boundary Accounting

- `page_batch_pack_calls`: [64, 64, 64]
- Median page-pack calls per generated token: 0.2500.
- Page-pack kernel time is unavailable because hot-path profiling is disabled during formal timing.
