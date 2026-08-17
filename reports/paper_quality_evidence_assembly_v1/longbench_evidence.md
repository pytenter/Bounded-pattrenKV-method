# LongBench Evidence

| Method | Macro Average |
| --- | --- |
| FP16 | 43.2862 |
| KIVI | 41.2143 |
| PatternKV | 41.6119 |
| CAUSAL-V4@25% | 42.4657 |

Status: CANONICAL_FINAL_CAUSAL_21TASK for the 8K-capped 21 x 50 setup.

Baseline source: `reports/paper_repro_v2/longbench_21x50_8k_4090/summary.json`.
CAUSAL source: `results/causal_v4_25_generalization_v1/longbench_full/causal_v4_25/`.

Scope: `21` tasks, `50` samples per task, `1050` total samples per method for CAUSAL, errors `0`, OOM `0`.

Deltas:

- CAUSAL - FP16: `-0.8205`.
- CAUSAL - PatternKV: `0.8538`.
- CAUSAL - KIVI: `1.2514`.

FULL_OFFICIAL_LONGBENCH_EXISTS: `false`.

This is not a strict full official LongBench split run; it is an 8K-capped 21-task x 50-sample reproduction.
