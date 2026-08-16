# Paper Trade-off Table

| Method | Effective KV bits | C2048 B4 throughput tok/s | C4096 max concurrency | Quality evidence | Primary interpretation |
|---|---:|---:|---:|---|---|
| FP16_FULL_MODEL | 16 | 126.599 | 4 | reference | fastest matched-B decode in this harness |
| KIVI_PAPER_G128_FULL_MODEL | 2.25 quantized region | 67.272 | 8 | baseline quality context | lower memory and highest observed capacity, but slower decode than FP16 |
| PATTERNKV_PAPER_FULL_MODEL | 2.25 quantized region | 14.658 | 4 | baseline quality context | true-batch baseline; slower full-model decode in this harness |
| CAUSAL_V4_25_FULL_MODEL | ~2.50 | 12.658 | 4 | `docs/PAPER_EVIDENCE_MAP.md` | trades decode throughput for the frozen selective-precision quality evidence |
