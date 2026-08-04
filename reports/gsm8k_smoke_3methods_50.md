# GSM8K smoke Summary

- created_at: `2026-08-03T13:25:04.782264+00:00`
- expected_samples_per_method: `50`
- status: `PARTIAL`

| method | rows | correct | accuracy | retention_vs_fp16 | delta_vs_fp16 | parser_fail | truncated | errors | peak_reserved_gb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 50 | 41 | 82.0 | 100.0 | 0.0 | 0 | 1 | 0 | 15.318 |
| kivi | 50 | 18 | 36.0 | 43.902 | -46.0 | 0 | 20 | 0 | 15.32 |
| patternkv | 30 | 28 | 93.333 | 113.821 | 11.333 | 0 | 0 | 0 | 15.299 |

## PatternKV vs KIVI Paired

- common_samples: `30`
- both_correct: `11`
- both_wrong: `2`
- only_patternkv_correct: `17`
- only_kivi_correct: `0`
- mcnemar_exact_p: `1.5e-05`
- bootstrap_delta_accuracy_points_ci95: `{'p2_5': 40.0, 'p50': 56.6667, 'p97_5': 73.3333}`

## Issues

- `missing_sample_index:patternkv:count=20 first=[7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24, 25, 35, 36, 37, 46, 47, 48, 49]`
- `wrong_total_rows:patternkv:30!=expected:50`
