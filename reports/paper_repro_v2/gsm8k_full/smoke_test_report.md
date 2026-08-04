# GSM8K Paper Full Summary

| method | planned | completed | correct | acc completed | strict acc | parse | eos | length | oom | error | avg gen | p95 wall | peak mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 1319 | 3 | 2 | 66.6667 | 0.1516 | 100.0 | 2 | 1 | 0 | 0 | 466.67 | 29.1748 | 16387145728 |
| kivi_paper_g128 | 1319 | 3 | 0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 3 | 0 | None | None |
| patternkv_paper | 1319 | 3 | 3 | 100.0 | 0.2274 | 100.0 | 3 | 0 | 0 | 0 | 182 | 14.0657 | 16345202688 |

## Paired

```json
[
  {
    "comparison": "patternkv_paper-kivi_paper_g128",
    "paired_n": 3,
    "both_correct": 0,
    "both_wrong": 0,
    "left_correct_right_wrong": 3,
    "left_wrong_right_correct": 0,
    "paired_accuracy_difference": 1.0
  },
  {
    "comparison": "patternkv_paper-fp16",
    "paired_n": 3,
    "both_correct": 2,
    "both_wrong": 0,
    "left_correct_right_wrong": 1,
    "left_wrong_right_correct": 0,
    "paired_accuracy_difference": 0.333333
  },
  {
    "comparison": "kivi_paper_g128-fp16",
    "paired_n": 3,
    "both_correct": 0,
    "both_wrong": 1,
    "left_correct_right_wrong": 0,
    "left_wrong_right_correct": 2,
    "paired_accuracy_difference": -0.666667
  }
]
```
