# GSM8K Paper Full Summary

| method | planned | completed | correct | acc completed | strict acc | parse | eos | length | oom | error | avg gen | p95 wall | peak mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 1319 | 1319 | 1029 | 78.0136 | 78.0136 | 100.0 | 1307 | 12 | 0 | 0 | 245.27 | 11.6182 | 16682844160 |
| kivi_paper_g128 | 1319 | 1319 | 909 | 68.9158 | 68.9158 | 100.0 | 1232 | 87 | 0 | 0 | 355.5 | 127.0603 | 16531849216 |
| patternkv_paper | 1319 | 1319 | 973 | 73.768 | 73.768 | 100.0 | 1281 | 38 | 0 | 0 | 283.4 | 22.0609 | 16544432128 |
| causal_v4_25 | 1319 | 1319 | 1041 | 78.9234 | 78.9234 | 99.9242 | 1308 | 11 | 0 | 0 | 245.9 | 37.4157 | 17741905920 |

## Paired

```json
[
  {
    "comparison": "patternkv_paper-kivi_paper_g128",
    "paired_n": 1319,
    "both_correct": 810,
    "both_wrong": 247,
    "left_correct_right_wrong": 163,
    "left_wrong_right_correct": 99,
    "paired_accuracy_difference": 0.048522
  },
  {
    "comparison": "patternkv_paper-fp16",
    "paired_n": 1319,
    "both_correct": 906,
    "both_wrong": 223,
    "left_correct_right_wrong": 67,
    "left_wrong_right_correct": 123,
    "paired_accuracy_difference": -0.042456
  },
  {
    "comparison": "kivi_paper_g128-fp16",
    "paired_n": 1319,
    "both_correct": 836,
    "both_wrong": 217,
    "left_correct_right_wrong": 73,
    "left_wrong_right_correct": 193,
    "paired_accuracy_difference": -0.090978
  },
  {
    "comparison": "causal_v4_25-fp16",
    "paired_n": 1319,
    "both_correct": 1009,
    "both_wrong": 258,
    "left_correct_right_wrong": 32,
    "left_wrong_right_correct": 20,
    "paired_accuracy_difference": 0.009098
  },
  {
    "comparison": "causal_v4_25-patternkv_paper",
    "paired_n": 1319,
    "both_correct": 913,
    "both_wrong": 218,
    "left_correct_right_wrong": 128,
    "left_wrong_right_correct": 60,
    "paired_accuracy_difference": 0.051554
  },
  {
    "comparison": "causal_v4_25-kivi_paper_g128",
    "paired_n": 1319,
    "both_correct": 840,
    "both_wrong": 209,
    "left_correct_right_wrong": 201,
    "left_wrong_right_correct": 69,
    "paired_accuracy_difference": 0.100076
  }
]
```
