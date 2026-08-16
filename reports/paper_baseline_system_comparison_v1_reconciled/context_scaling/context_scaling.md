# Context Scaling

| Method | Context | Batch | Decode | Median TPOT ms | Throughput tok/s | Valid runs |
|---|---:|---:|---:|---:|---:|---:|
| CAUSAL_V4_25_FULL_MODEL | 2048 | 1 | 8 | 158.325 | 6.316 | 3 |
| CAUSAL_V4_25_FULL_MODEL | 4096 | 1 | 8 | 158.739 | 6.299 | 3 |
| CAUSAL_V4_25_FULL_MODEL | 8192 | 1 | 8 | 157.457 | 6.351 | 3 |
| FP16_FULL_MODEL | 2048 | 1 | 8 | 29.033 | 34.436 | 3 |
| FP16_FULL_MODEL | 4096 | 1 | 8 | 29.150 | 34.300 | 3 |
| FP16_FULL_MODEL | 8192 | 1 | 8 | 30.584 | 32.692 | 3 |
| KIVI_PAPER_G128_FULL_MODEL | 2048 | 1 | 8 | 56.432 | 17.719 | 3 |
| KIVI_PAPER_G128_FULL_MODEL | 4096 | 1 | 8 | 57.033 | 17.532 | 3 |
| KIVI_PAPER_G128_FULL_MODEL | 8192 | 1 | 8 | 56.754 | 17.618 | 3 |
| PATTERNKV_PAPER_FULL_MODEL | 2048 | 1 | 8 | 153.094 | 6.532 | 3 |
| PATTERNKV_PAPER_FULL_MODEL | 4096 | 1 | 8 | 156.361 | 6.395 | 3 |
| PATTERNKV_PAPER_FULL_MODEL | 8192 | 1 | 8 | 153.611 | 6.510 | 3 |
