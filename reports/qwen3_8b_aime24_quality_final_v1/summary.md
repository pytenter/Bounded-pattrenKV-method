# Qwen3-8B AIME24 Quality Summary

| Method | Correct | Accuracy | Length stop | Runtime errors | Latest |
|---|---:|---:|---:|---:|---|
| FP16 | 71/90 | 78.89% | 3 | 0 | 2026-08-29 00:47:30 +0800 |
| CAUSAL_V4_25 | 63/90 | 70.00% | 9 | 0 | 2026-08-27 23:05:32 +0800 |
| PATTERNKV_PAPER | 55/90 | 61.11% | 8 | 0 | 2026-08-28 19:55:12 +0800 |
| KIVI_PAPER_G128 | 53/90 | 58.89% | 11 | 0 | 2026-08-29 18:53:36 +0800 |

## By Seed

### FP16
| Seed | Correct | Accuracy | Length stop | Runtime errors |
|---|---:|---:|---:|---:|
| seed42 | 24/30 | 80.00% | 1 | 0 |
| seed43 | 25/30 | 83.33% | 0 | 0 |
| seed44 | 22/30 | 73.33% | 2 | 0 |

### CAUSAL_V4_25
| Seed | Correct | Accuracy | Length stop | Runtime errors |
|---|---:|---:|---:|---:|
| seed42 | 20/30 | 66.67% | 3 | 0 |
| seed43 | 22/30 | 73.33% | 4 | 0 |
| seed44 | 21/30 | 70.00% | 2 | 0 |

### PATTERNKV_PAPER
| Seed | Correct | Accuracy | Length stop | Runtime errors |
|---|---:|---:|---:|---:|
| seed42 | 17/30 | 56.67% | 2 | 0 |
| seed43 | 20/30 | 66.67% | 4 | 0 |
| seed44 | 18/30 | 60.00% | 2 | 0 |

### KIVI_PAPER_G128
| Seed | Correct | Accuracy | Length stop | Runtime errors |
|---|---:|---:|---:|---:|
| seed42 | 17/30 | 56.67% | 3 | 0 |
| seed43 | 16/30 | 53.33% | 4 | 0 |
| seed44 | 20/30 | 66.67% | 4 | 0 |
