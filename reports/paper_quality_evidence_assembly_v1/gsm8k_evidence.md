# GSM8K Evidence

| Method | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| FP16 | 1029 | 1319 | 78.0136% |
| KIVI | 909 | 1319 | 68.9158% |
| PatternKV | 973 | 1319 | 73.7680% |
| CAUSAL-V4@25% | 1041 | 1319 | 78.9234% |

Status: CANONICAL_WITH_CAUSAL_RAW_AGGREGATION.

Canonical baseline source: `reports/paper_repro_v2/gsm8k_full_2048/summary_4gpu_sequence.json`.
Canonical CAUSAL source: `results/causal_v4_25_generalization_v1/gsm8k_full/causal_v4_25/`.

Deltas:

- CAUSAL - FP16: `0.9098` percentage points.
- CAUSAL - PatternKV: `5.1554` percentage points.
- CAUSAL - KIVI: `10.0076` percentage points.

Offline paired counts were computed from committed per-sample outputs only:

```json
{
  "CAUSAL_vs_FP16": {
    "both_correct": 1009,
    "both_wrong": 258,
    "left_correct_right_wrong": 32,
    "left_wrong_right_correct": 20,
    "mcnemar_exact_p": 0.12634707581392135,
    "paired_accuracy_difference": 0.009097801364670205,
    "paired_n": 1319
  },
  "CAUSAL_vs_KIVI": {
    "both_correct": 840,
    "both_wrong": 209,
    "left_correct_right_wrong": 201,
    "left_wrong_right_correct": 69,
    "mcnemar_exact_p": 3.9126454090944465e-16,
    "paired_accuracy_difference": 0.10007581501137225,
    "paired_n": 1319
  },
  "CAUSAL_vs_PatternKV": {
    "both_correct": 913,
    "both_wrong": 218,
    "left_correct_right_wrong": 128,
    "left_wrong_right_correct": 60,
    "mcnemar_exact_p": 7.874054664908622e-07,
    "paired_accuracy_difference": 0.05155420773313116,
    "paired_n": 1319
  }
}
```

Do not claim significance beyond these offline paired statistics without a predeclared statistical protocol.
