# GSM8K Paper Full Summary

| method | planned | completed | correct | acc completed | strict acc | parse | eos | length | oom | error | avg gen | p95 wall | peak mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 1319 | 1319 | 1029 | 78.0136 | 78.0136 | 100.0 | 1306 | 13 | 0 | 0 | 235.79 | 11.6035 | 16492003328 |
| kivi_paper_g128 | 1319 | 1287 | 888 | 68.9977 | 67.3237 | 100.0 | 1203 | 84 | 0 | 0 | 286.73 | 63.3286 | 16496197632 |
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
    "paired_n": 1287,
    "both_correct": 817,
    "both_wrong": 215,
    "left_correct_right_wrong": 71,
    "left_wrong_right_correct": 184,
    "paired_accuracy_difference": -0.087801
  }
]
```
