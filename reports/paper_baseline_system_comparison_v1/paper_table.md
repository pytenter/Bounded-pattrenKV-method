# Paper Table

| Method | Effective KV bits | C2048 B1 TPOT | C2048 B4 TPOT | C2048 B4 throughput | C4096 max B | Capacity vs FP16 | C4096 peak memory | True batch | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| FP16_FULL_MODEL | 16 | 28.038 | 31.590 | 126.599 | 4 | 1.00 | 17153242112 | True | same-harness |
| KIVI_PAPER_G128_FULL_MODEL | 2.25 quantized region | 54.221 | 59.455 | 67.272 | 8 | 2.00 | 16679165952 | True | same-harness |
| PATTERNKV_PAPER_FULL_MODEL | 2.25 quantized region | 249.705 | 272.890 | 14.658 | 4 | 1.00 | 16908754944 | True | same-harness |
| CAUSAL_V4_25_FULL_MODEL | ~2.50 | 274.967 | 315.989 | 12.658 | 4 | 1.00 | 16936968192 | True | same-harness |
