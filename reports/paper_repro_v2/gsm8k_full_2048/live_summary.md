# GSM8K Paper Full Summary

| method | planned | completed | correct | acc completed | strict acc | parse | eos | length | oom | error | avg gen | p95 wall | peak mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 1319 | 174 | 140 | 80.4598 | 10.6141 | 100.0 | 172 | 2 | 0 | 0 | 243.22 | 12.9443 | 16668164096 |
| kivi_paper_g128 | 1319 | 0 | 0 | None | 0.0 | None | 0 | 0 | 0 | 0 | None | None | None |
| patternkv_paper | 1319 | 0 | 0 | None | 0.0 | None | 0 | 0 | 0 | 0 | None | None | None |

## Paired

```json
[
  {
    "comparison": "patternkv_paper-kivi_paper_g128",
    "paired_n": 0,
    "both_correct": 0,
    "both_wrong": 0,
    "left_correct_right_wrong": 0,
    "left_wrong_right_correct": 0,
    "paired_accuracy_difference": null
  },
  {
    "comparison": "patternkv_paper-fp16",
    "paired_n": 0,
    "both_correct": 0,
    "both_wrong": 0,
    "left_correct_right_wrong": 0,
    "left_wrong_right_correct": 0,
    "paired_accuracy_difference": null
  },
  {
    "comparison": "kivi_paper_g128-fp16",
    "paired_n": 0,
    "both_correct": 0,
    "both_wrong": 0,
    "left_correct_right_wrong": 0,
    "left_wrong_right_correct": 0,
    "paired_accuracy_difference": null
  }
]
```
