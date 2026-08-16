# Context Scaling

| Method | Context | Batch | Decode | Median TPOT ms | Throughput tok/s | Valid runs |
|---|---:|---:|---:|---:|---:|---:|
| CAUSAL_V4_25_FULL_MODEL | 2048 | 1 | 8 | 303.882 | 3.291 | 3 |
| CAUSAL_V4_25_FULL_MODEL | 4096 | 1 | 8 | 359.284 | 2.783 | 3 |
| CAUSAL_V4_25_FULL_MODEL | 8192 | 1 | 8 | 487.901 | 2.050 | 3 |
| FP16_FULL_MODEL | 2048 | 1 | 8 | 29.791 | 33.562 | 3 |
| FP16_FULL_MODEL | 4096 | 1 | 8 | 30.802 | 32.461 | 3 |
| FP16_FULL_MODEL | 8192 | 1 | 8 | 31.976 | 31.269 | 3 |
| KIVI_PAPER_G128_FULL_MODEL | 2048 | 1 | 8 | 76.693 | 13.038 | 3 |
| KIVI_PAPER_G128_FULL_MODEL | 4096 | 1 | 8 | 76.172 | 13.127 | 3 |
| KIVI_PAPER_G128_FULL_MODEL | 8192 | 1 | 8 | 59.269 | 16.871 | 3 |
| PATTERNKV_PAPER_FULL_MODEL | 2048 | 1 | 8 | 272.372 | 3.671 | 3 |
| PATTERNKV_PAPER_FULL_MODEL | 4096 | 1 | 8 | 469.446 | 2.130 | 3 |
| PATTERNKV_PAPER_FULL_MODEL | 8192 | 1 | 8 | 457.698 | 2.185 | 3 |
