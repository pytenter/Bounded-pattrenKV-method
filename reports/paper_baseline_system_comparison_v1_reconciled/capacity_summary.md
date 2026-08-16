# Capacity

| Method | max_success_B | first_OOM_B | capacity_ratio_vs_FP16 | TPOT at max B | throughput at max B | peak allocated at max B |
|---|---:|---:|---:|---:|---:|---:|
| FP16_FULL_MODEL | 4 | 8 | 1.00 | 40.627 | 98.445 | 20405783552 |
| KIVI_PAPER_G128_FULL_MODEL | 8 | 16 | 2.00 | 68.958 | 115.998 | 20714807296 |
| PATTERNKV_PAPER_FULL_MODEL | 8 | 16 | 2.00 | 202.869 | 39.433 | 22259261440 |
| CAUSAL_V4_25_FULL_MODEL | 8 | 16 | 2.00 | 207.490 | 38.554 | 22432702464 |
