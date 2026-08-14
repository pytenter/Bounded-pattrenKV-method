# LongBench 8K 4090 Summary

Planned total: `4200`
Completed total: `4200`
Success total: `4200`
OOM total: `0`
Error total: `0`

## Macro Average Complete Case

```json
{
  "fp16": 43.28619047619048,
  "kivi_paper_g128": 41.214285714285715,
  "patternkv_paper": 41.61190476190476,
  "causal_v4_25": 42.46619047619048
}
```

## Category Average Complete Case

```json
{
  "fp16": {
    "SQA": 36.583333333333336,
    "MQA": 32.574,
    "Summ.": 26.73,
    "Few-shot": 61.21,
    "Synth.": 56.70333333333333,
    "Code": 57.26
  },
  "kivi_paper_g128": {
    "SQA": 34.086666666666666,
    "MQA": 30.972,
    "Summ.": 26.53,
    "Few-shot": 60.552499999999995,
    "Synth.": 52.483333333333334,
    "Code": 51.3
  },
  "patternkv_paper": {
    "SQA": 34.39,
    "MQA": 31.006,
    "Summ.": 26.3575,
    "Few-shot": 60.305,
    "Synth.": 54.68666666666667,
    "Code": 52.47
  },
  "causal_v4_25": {
    "SQA": 36.373333333333335,
    "MQA": 30.126,
    "Summ.": 26.2875,
    "Few-shot": 61.3975,
    "Synth.": 56.0,
    "Code": 56.650000000000006
  }
}
```

## Paper Table 1 Comparison

Source: `https://arxiv.org/html/2510.05176v1`

| method | MQA | SQA | Summ. | Few-shot | Synth. | Code | Avg | Paper Avg | Delta Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 32.574 | 36.583333333333336 | 26.73 | 61.21 | 56.70333333333333 | 57.26 | 43.28619047619048 | 46.59 | -3.3038095238095266 |
| KIVI | 30.972 | 34.086666666666666 | 26.53 | 60.552499999999995 | 52.483333333333334 | 51.3 | 41.214285714285715 | 44.33 | -3.115714285714283 |
| PatternKV | 31.006 | 34.39 | 26.3575 | 60.305 | 54.68666666666667 | 52.47 | 41.61190476190476 | 45.33 | -3.7180952380952377 |
| CAUSAL_V4_25 | 30.126 | 36.373333333333335 | 26.2875 | 61.3975 | 56.0 | 56.650000000000006 | 42.46619047619048 | None | None |

## Task Scores

| method | task | category | planned | completed | success | OOM | error | score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | narrativeqa | SQA | 50 | 50 | 50 | 0 | 0 | 17.3 |
| FP16 | qasper | SQA | 50 | 50 | 50 | 0 | 0 | 41.09 |
| FP16 | multifieldqa_en | SQA | 50 | 50 | 50 | 0 | 0 | 51.36 |
| FP16 | multifieldqa_zh | MQA | 50 | 50 | 50 | 0 | 0 | 56.18 |
| FP16 | hotpotqa | MQA | 50 | 50 | 50 | 0 | 0 | 22.61 |
| FP16 | 2wikimqa | MQA | 50 | 50 | 50 | 0 | 0 | 40.13 |
| FP16 | musique | MQA | 50 | 50 | 50 | 0 | 0 | 10.12 |
| FP16 | dureader | MQA | 50 | 50 | 50 | 0 | 0 | 33.83 |
| FP16 | gov_report | Summ. | 50 | 50 | 50 | 0 | 0 | 35.49 |
| FP16 | qmsum | Summ. | 50 | 50 | 50 | 0 | 0 | 23.37 |
| FP16 | multi_news | Summ. | 50 | 50 | 50 | 0 | 0 | 26.47 |
| FP16 | vcsum | Summ. | 50 | 50 | 50 | 0 | 0 | 21.59 |
| FP16 | trec | Few-shot | 50 | 50 | 50 | 0 | 0 | 72.0 |
| FP16 | triviaqa | Few-shot | 50 | 50 | 50 | 0 | 0 | 89.77 |
| FP16 | samsum | Few-shot | 50 | 50 | 50 | 0 | 0 | 45.07 |
| FP16 | lsht | Few-shot | 50 | 50 | 50 | 0 | 0 | 38.0 |
| FP16 | passage_count | Synth. | 50 | 50 | 50 | 0 | 0 | 5.11 |
| FP16 | passage_retrieval_en | Synth. | 50 | 50 | 50 | 0 | 0 | 76.0 |
| FP16 | passage_retrieval_zh | Synth. | 50 | 50 | 50 | 0 | 0 | 89.0 |
| FP16 | lcc | Code | 50 | 50 | 50 | 0 | 0 | 63.98 |
| FP16 | repobench-p | Code | 50 | 50 | 50 | 0 | 0 | 50.54 |
| KIVI | narrativeqa | SQA | 50 | 50 | 50 | 0 | 0 | 16.64 |
| KIVI | qasper | SQA | 50 | 50 | 50 | 0 | 0 | 38.88 |
| KIVI | multifieldqa_en | SQA | 50 | 50 | 50 | 0 | 0 | 46.74 |
| KIVI | multifieldqa_zh | MQA | 50 | 50 | 50 | 0 | 0 | 56.28 |
| KIVI | hotpotqa | MQA | 50 | 50 | 50 | 0 | 0 | 17.85 |
| KIVI | 2wikimqa | MQA | 50 | 50 | 50 | 0 | 0 | 39.83 |
| KIVI | musique | MQA | 50 | 50 | 50 | 0 | 0 | 8.99 |
| KIVI | dureader | MQA | 50 | 50 | 50 | 0 | 0 | 31.91 |
| KIVI | gov_report | Summ. | 50 | 50 | 50 | 0 | 0 | 34.22 |
| KIVI | qmsum | Summ. | 50 | 50 | 50 | 0 | 0 | 24.52 |
| KIVI | multi_news | Summ. | 50 | 50 | 50 | 0 | 0 | 26.15 |
| KIVI | vcsum | Summ. | 50 | 50 | 50 | 0 | 0 | 21.23 |
| KIVI | trec | Few-shot | 50 | 50 | 50 | 0 | 0 | 70.0 |
| KIVI | triviaqa | Few-shot | 50 | 50 | 50 | 0 | 0 | 90.27 |
| KIVI | samsum | Few-shot | 50 | 50 | 50 | 0 | 0 | 46.94 |
| KIVI | lsht | Few-shot | 50 | 50 | 50 | 0 | 0 | 35.0 |
| KIVI | passage_count | Synth. | 50 | 50 | 50 | 0 | 0 | 6.75 |
| KIVI | passage_retrieval_en | Synth. | 50 | 50 | 50 | 0 | 0 | 67.87 |
| KIVI | passage_retrieval_zh | Synth. | 50 | 50 | 50 | 0 | 0 | 82.83 |
| KIVI | lcc | Code | 50 | 50 | 50 | 0 | 0 | 60.18 |
| KIVI | repobench-p | Code | 50 | 50 | 50 | 0 | 0 | 42.42 |
| PatternKV | narrativeqa | SQA | 50 | 50 | 50 | 0 | 0 | 16.02 |
| PatternKV | qasper | SQA | 50 | 50 | 50 | 0 | 0 | 40.01 |
| PatternKV | multifieldqa_en | SQA | 50 | 50 | 50 | 0 | 0 | 47.14 |
| PatternKV | multifieldqa_zh | MQA | 50 | 50 | 50 | 0 | 0 | 55.26 |
| PatternKV | hotpotqa | MQA | 50 | 50 | 50 | 0 | 0 | 21.96 |
| PatternKV | 2wikimqa | MQA | 50 | 50 | 50 | 0 | 0 | 38.66 |
| PatternKV | musique | MQA | 50 | 50 | 50 | 0 | 0 | 9.7 |
| PatternKV | dureader | MQA | 50 | 50 | 50 | 0 | 0 | 29.45 |
| PatternKV | gov_report | Summ. | 50 | 50 | 50 | 0 | 0 | 34.6 |
| PatternKV | qmsum | Summ. | 50 | 50 | 50 | 0 | 0 | 24.5 |
| PatternKV | multi_news | Summ. | 50 | 50 | 50 | 0 | 0 | 25.68 |
| PatternKV | vcsum | Summ. | 50 | 50 | 50 | 0 | 0 | 20.65 |
| PatternKV | trec | Few-shot | 50 | 50 | 50 | 0 | 0 | 72.0 |
| PatternKV | triviaqa | Few-shot | 50 | 50 | 50 | 0 | 0 | 89.78 |
| PatternKV | samsum | Few-shot | 50 | 50 | 50 | 0 | 0 | 43.44 |
| PatternKV | lsht | Few-shot | 50 | 50 | 50 | 0 | 0 | 36.0 |
| PatternKV | passage_count | Synth. | 50 | 50 | 50 | 0 | 0 | 5.33 |
| PatternKV | passage_retrieval_en | Synth. | 50 | 50 | 50 | 0 | 0 | 71.0 |
| PatternKV | passage_retrieval_zh | Synth. | 50 | 50 | 50 | 0 | 0 | 87.73 |
| PatternKV | lcc | Code | 50 | 50 | 50 | 0 | 0 | 60.9 |
| PatternKV | repobench-p | Code | 50 | 50 | 50 | 0 | 0 | 44.04 |
| CAUSAL_V4_25 | narrativeqa | SQA | 50 | 50 | 50 | 0 | 0 | 17.59 |
| CAUSAL_V4_25 | qasper | SQA | 50 | 50 | 50 | 0 | 0 | 41.45 |
| CAUSAL_V4_25 | multifieldqa_en | SQA | 50 | 50 | 50 | 0 | 0 | 50.08 |
| CAUSAL_V4_25 | multifieldqa_zh | MQA | 50 | 50 | 50 | 0 | 0 | 55.35 |
| CAUSAL_V4_25 | hotpotqa | MQA | 50 | 50 | 50 | 0 | 0 | 18.87 |
| CAUSAL_V4_25 | 2wikimqa | MQA | 50 | 50 | 50 | 0 | 0 | 38.38 |
| CAUSAL_V4_25 | musique | MQA | 50 | 50 | 50 | 0 | 0 | 7.19 |
| CAUSAL_V4_25 | dureader | MQA | 50 | 50 | 50 | 0 | 0 | 30.84 |
| CAUSAL_V4_25 | gov_report | Summ. | 50 | 50 | 50 | 0 | 0 | 35.13 |
| CAUSAL_V4_25 | qmsum | Summ. | 50 | 50 | 50 | 0 | 0 | 23.15 |
| CAUSAL_V4_25 | multi_news | Summ. | 50 | 50 | 50 | 0 | 0 | 25.48 |
| CAUSAL_V4_25 | vcsum | Summ. | 50 | 50 | 50 | 0 | 0 | 21.39 |
| CAUSAL_V4_25 | trec | Few-shot | 50 | 50 | 50 | 0 | 0 | 72.0 |
| CAUSAL_V4_25 | triviaqa | Few-shot | 50 | 50 | 50 | 0 | 0 | 89.77 |
| CAUSAL_V4_25 | samsum | Few-shot | 50 | 50 | 50 | 0 | 0 | 45.82 |
| CAUSAL_V4_25 | lsht | Few-shot | 50 | 50 | 50 | 0 | 0 | 38.0 |
| CAUSAL_V4_25 | passage_count | Synth. | 50 | 50 | 50 | 0 | 0 | 6.33 |
| CAUSAL_V4_25 | passage_retrieval_en | Synth. | 50 | 50 | 50 | 0 | 0 | 73.0 |
| CAUSAL_V4_25 | passage_retrieval_zh | Synth. | 50 | 50 | 50 | 0 | 0 | 88.67 |
| CAUSAL_V4_25 | lcc | Code | 50 | 50 | 50 | 0 | 0 | 63.52 |
| CAUSAL_V4_25 | repobench-p | Code | 50 | 50 | 50 | 0 | 0 | 49.78 |
