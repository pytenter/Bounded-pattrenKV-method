# Paper Table

| Method | Effective KV bits | C2048 B1 TPOT | C2048 B4 TPOT | C2048 B4 throughput | C4096 B4 peak allocated | C4096 B4 peak reserved | C4096 max B | Capacity vs FP16 | Long Decode D256 TPOT | True batch | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| FP16_FULL_MODEL | 16 | 30.604 | 31.554 | 126.747 | 20405783552 | 20786970624 | 4 | 1.00 | 28.663 | True | reconciled allocator protocol |
| KIVI_PAPER_G128_FULL_MODEL | 2.25 quantized region | 68.967 | 62.328 | 64.171 | 18408726528 | 18886950912 | 8 | 2.00 | 56.832 | True | reconciled allocator protocol |
| PATTERNKV_PAPER_FULL_MODEL | 2.25 quantized region | 162.753 | 158.926 | 25.168 | 19181068288 | 19941818368 | 8 | 2.00 | 154.646 | True | reconciled allocator protocol |
| CAUSAL_V4_25_FULL_MODEL | ~2.50 | 165.168 | 165.709 | 24.138 | 19267788800 | 19700645888 | 8 | 2.00 | 159.530 | True | reconciled allocator protocol |
