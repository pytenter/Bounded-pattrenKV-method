# PatternKV LongBench 21x50 8K Paper Comparison

Scope: 21 LongBench tasks x 50 samples per task x 3 methods, MAX_INPUT_LENGTH=8192, Llama-3.1-8B-Instruct, single RTX 4090 D. This is an 8K-capped subset reproduction, not the paper strict full LongBench setting.

Paper reference: PatternKV Table 1, LLaMA-3.1-8B-Instruct, 2-bit LongBench. Source: https://arxiv.org/html/2510.05176v1

## Completion

- Planned: `3150`
- Completed: `3150`
- Success: `3150`
- OOM: `0`
- Error: `0`

## Local Three-Method Result

| method | MQA | SQA | Summ. | Few-shot | Synth. | Code | Macro Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 32.57 | 36.58 | 26.73 | 61.21 | 56.70 | 57.26 | 43.29 |
| KIVI | 30.97 | 34.09 | 26.53 | 60.55 | 52.48 | 51.30 | 41.21 |
| PatternKV | 31.01 | 34.39 | 26.36 | 60.30 | 54.69 | 52.47 | 41.61 |

## Local Paired Deltas

| comparison | macro delta | interpretation |
| --- | ---: | --- |
| PatternKV - KIVI | 0.40 | higher is better for the first method |
| PatternKV - FP16 | -1.67 | higher is better for the first method |
| KIVI - FP16 | -2.07 | higher is better for the first method |

## Paper Table 1

| method | MQA | SQA | Summ. | Few-shot | Synth. | Code | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 36.63 | 46.56 | 25.54 | 61.16 | 59.99 | 59.42 | 46.59 |
| KIVI | 34.86 | 43.96 | 24.98 | 60.35 | 54.43 | 55.53 | 44.33 |
| PatternKV | 35.49 | 45.08 | 25.12 | 60.58 | 57.89 | 56.55 | 45.33 |

## Local Minus Paper

| method | MQA | SQA | Summ. | Few-shot | Synth. | Code | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | -4.06 | -9.98 | 1.19 | 0.05 | -3.29 | -2.16 | -3.30 |
| KIVI | -3.89 | -9.87 | 1.55 | 0.20 | -1.95 | -4.23 | -3.12 |
| PatternKV | -4.48 | -10.69 | 1.24 | -0.27 | -3.20 | -4.08 | -3.72 |

## Notes

- Absolute comparison to the paper is not strict because this run uses 50 samples per task and an 8192-token input cap.
- The most reliable local comparison is paired within this run: PatternKV beats KIVI by +0.40 macro points, while both quantized methods trail FP16.
- Task scores are aggregated from each row's stored `score`, preserving classification task scoring for `trec` and `lsht`.
