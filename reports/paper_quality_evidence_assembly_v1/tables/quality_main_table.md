# Provisional Quality Main Table

| Method | Effective KV Bits | AIME24 | GSM8K | LongBench | Notes |
| --- | --- | --- | --- | --- | --- |
| FP16 | 16-bit KV | 45/90 (50.00%) | 1029/1319 (78.0136%) | 43.2862 | Full precision reference. |
| KIVI | 2.25-bit quantized-region | not run in canonical AIME24 four-method task-quality table | 909/1319 (68.9158%) | 41.2143 | Canonical baseline for GSM8K/LongBench. |
| PatternKV | 2.25-bit quantized-region | 32/90 (35.56%) | 973/1319 (73.7680%) | 41.6119 | AIME24 row is Pattern Base. |
| Random-25% | ~2.500488 bit/KV element | 36/90 (40.00%) | not run | not run | Same-budget AIME24 control. |
| CAUSAL-V4@25% | ~2.500488 bit/KV element | 45/90 (50.00%) | 1041/1319 (78.9234%) | 42.4657 | Selective heterogeneous V2/V4 method. |
