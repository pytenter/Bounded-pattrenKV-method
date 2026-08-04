# GSM8K smoke Summary

- created_at: `2026-08-03T12:55:47.794465+00:00`
- expected_samples_per_method: `50`
- status: `PASS`

| method | rows | correct | accuracy | retention_vs_fp16 | delta_vs_fp16 | parser_fail | truncated | errors | peak_reserved_gb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 50 | 41 | 82.0 | 100.0 | 0.0 | 0 | 1 | 0 | 15.316 |
| kivi | 50 | 18 | 36.0 | 43.902 | -46.0 | 0 | 20 | 0 | 15.322 |
| patternkv | 50 | 42 | 84.0 | 102.439 | 2.0 | 0 | 1 | 0 | 15.299 |

## PatternKV vs KIVI Paired

- common_samples: `50`
- both_correct: `18`
- both_wrong: `8`
- only_patternkv_correct: `24`
- only_kivi_correct: `0`
- mcnemar_exact_p: `0.0`
- bootstrap_delta_accuracy_points_ci95: `{'p2_5': 34.0, 'p50': 48.0, 'p97_5': 62.0}`

No integrity issues detected.
