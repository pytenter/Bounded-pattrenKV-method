# GSM8K Selector Component Pilot

| method | completed | correct | acc % | avg tok/s | aggregate tok/s | avg gen toks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| importance_only_v4_25 | 50 | 28 | 56.0 | 10.722158 | 9.318894 | 631.54 |
| error_only_v4_25 | 50 | 31 | 62.0 | 10.291486 | 9.191434 | 521.22 |
| causal_v4_25 | 50 | 31 | 62.0 | 10.814774 | 9.355312 | 739.84 |

## Paired Vs CAUSAL

```json
[
  {
    "comparison": "importance_only_v4_25_vs_causal_v4_25",
    "paired_n": 50,
    "both_correct": 24,
    "both_wrong": 15,
    "left_correct_right_wrong": 4,
    "left_wrong_right_correct": 7,
    "paired_accuracy_delta": -0.06
  },
  {
    "comparison": "error_only_v4_25_vs_causal_v4_25",
    "paired_n": 50,
    "both_correct": 27,
    "both_wrong": 15,
    "left_correct_right_wrong": 4,
    "left_wrong_right_correct": 4,
    "paired_accuracy_delta": 0.0
  }
]
```
