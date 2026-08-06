# AIME24 Results Summary

results_dir: `results/paper_repro_v2/aime24_budget_n2`

| method | completed | valid | correct | Avg@N | strict_avg | oom | length | eos | avg tokens | p95 wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 60 | 59 | 29 | 49.1525 | 48.3333 | 0 | 3 | 57 | 13220.93 | 2473.3396 |
| kivi_paper_g128 | 60 | 56 | 12 | 21.4286 | 20.0 | 0 | 10 | 50 | 16430.63 | 2035.3173 |
| patternkv_paper | 60 | 54 | 13 | 24.0741 | 21.6667 | 0 | 8 | 52 | 14769.17 | 2179.8863 |

## Paired

```json
[
  {
    "comparison": "patternkv_paper-kivi_paper_g128",
    "paired_n": 60,
    "paired_accuracy_difference": 0.016666666666666666,
    "both_correct": 8,
    "both_wrong": 43,
    "patternkv_paper_correct_kivi_paper_g128_wrong": 5,
    "patternkv_paper_wrong_kivi_paper_g128_correct": 4
  },
  {
    "comparison": "patternkv_paper-fp16",
    "paired_n": 60,
    "paired_accuracy_difference": -0.26666666666666666,
    "both_correct": 11,
    "both_wrong": 29,
    "patternkv_paper_correct_fp16_wrong": 2,
    "patternkv_paper_wrong_fp16_correct": 18
  },
  {
    "comparison": "kivi_paper_g128-fp16",
    "paired_n": 60,
    "paired_accuracy_difference": -0.2833333333333333,
    "both_correct": 9,
    "both_wrong": 28,
    "kivi_paper_g128_correct_fp16_wrong": 3,
    "kivi_paper_g128_wrong_fp16_correct": 20
  }
]
```
