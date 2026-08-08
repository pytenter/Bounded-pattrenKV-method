# AIME24 Wave 1A.3 Sink Length Sweep Report

## 1. Executive Summary

- Wave 1A.3 ran the planned Sink sweep for S16, S32, and S128 while reusing validated S0 and S64 records.
- S16 and S32 completed for both PatternKV and KIVI. S128 hit a repeatable `sink token count mismatch` validation error in both methods and is excluded from quality/Pareto decisions.
- On valid data, PatternKV improves from S0 7/12 to S16 9/12 and S32 9/12; KIVI improves from S0 2/12 to S16 6/12 and S32 5/12.
- The current Pareto candidate is S16/R128: it reaches the best observed PatternKV accuracy and most of the KIVI Sink benefit with lower bit cost than S32/S64.

## 2. Motivation

Wave 1A.2 showed that Sink64, not Recent256, was the main source of the S64/R256 gain. This sweep asks how much early-token FP16 protection is needed.

## 3. Experimental Design

- Task manifest hash: `ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e`
- Generation config hash: `a7d6b2f8bab37893b6331c66b3e5eb6a`
- Fixed: segmented rolling, recent_length=128, K2V2, group_size=128.
- Variable: sink_length in S0, S16, S32, S64, S128.

## 4. Reuse Validation

- Reuse validation status: `passed`.
- Reused records: `48`.
- Planned new records: `72`.

## 5. Runtime Validity

| method | sink | records | valid records | runtime errors | valid for quality | first error |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| PatternKV | 0 | 12 | 12 | 0 | True |  |
| PatternKV | 16 | 12 | 12 | 0 | True |  |
| PatternKV | 32 | 12 | 12 | 0 | True |  |
| PatternKV | 64 | 12 | 12 | 0 | True |  |
| PatternKV | 128 | 12 | 1 | 11 | False | "ValueError('sink token count mismatch: 117 != 118')" |
| KIVI | 0 | 12 | 12 | 0 | True |  |
| KIVI | 16 | 12 | 12 | 0 | True |  |
| KIVI | 32 | 12 | 12 | 0 | True |  |
| KIVI | 64 | 12 | 12 | 0 | True |  |
| KIVI | 128 | 12 | 1 | 11 | False | "ValueError('sink token count mismatch: 117 != 118')" |

## 6. PatternKV Sink Sweep

| sink | correct/valid | accuracy | length stops | mean tokens | median tokens | P90 tokens | theoretical bits | actual bits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7/12 | 58.3% | 1 | 12082.7 | 9276.5 | 23163 | 2.6673 | 3.3267 |
| 16 | 9/12 | 75.0% | 0 | 11223.2 | 7741.5 | 23728 | 2.7288 | 3.4095 |
| 32 | 9/12 | 75.0% | 0 | 7900.2 | 7406.0 | 11251 | 2.7971 | 3.4807 |
| 64 | 8/12 | 66.7% | 0 | 9219.6 | 8062.0 | 17003 | 2.8603 | 3.5262 |
| 128 | 1/1 | 100.0% | 0 | 4580 | 4580 | 4580 | 3.3035 | 4.1757 |

## 7. KIVI Sink Sweep

| sink | correct/valid | accuracy | length stops | mean tokens | median tokens | P90 tokens | theoretical bits | actual bits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2/12 | 16.7% | 1 | 15401.4 | 12425.0 | 30129 | 2.4069 | 2.4730 |
| 16 | 6/12 | 50.0% | 0 | 11149.3 | 11719.5 | 23262 | 2.5322 | 2.6634 |
| 32 | 5/12 | 41.7% | 0 | 7575.2 | 7410.0 | 10945 | 2.5847 | 2.6991 |
| 64 | 7/12 | 58.3% | 0 | 12155.8 | 11436.5 | 17650 | 2.5040 | 2.6045 |
| 128 | 1/1 | 100.0% | 0 | 3218 | 3218 | 3218 | 3.2939 | 3.4686 |

## 8. Paired Rescues and Regressions

| method | comparison | paired n | valid | rescues | regressions | ties | net gain |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| PatternKV | S16 vs S0 | 12 | True | 4 | 2 | 6 | 2 |
| PatternKV | S32 vs S0 | 12 | True | 3 | 1 | 8 | 2 |
| PatternKV | S64 vs S0 | 12 | True | 2 | 1 | 9 | 1 |
| PatternKV | S128 vs S0 | 1 | False | 0 | 0 | 1 | 0 |
| PatternKV | S0 -> S16 | 12 | True | 4 | 2 | 6 | 2 |
| PatternKV | S16 -> S32 | 12 | True | 2 | 2 | 8 | 0 |
| PatternKV | S32 -> S64 | 12 | True | 1 | 2 | 9 | -1 |
| PatternKV | S64 -> S128 | 1 | False | 0 | 0 | 1 | 0 |
| KIVI | S16 vs S0 | 12 | True | 4 | 0 | 8 | 4 |
| KIVI | S32 vs S0 | 12 | True | 3 | 0 | 9 | 3 |
| KIVI | S64 vs S0 | 12 | True | 7 | 2 | 3 | 5 |
| KIVI | S128 vs S0 | 1 | False | 0 | 0 | 1 | 0 |
| KIVI | S0 -> S16 | 12 | True | 4 | 0 | 8 | 4 |
| KIVI | S16 -> S32 | 12 | True | 2 | 3 | 7 | -1 |
| KIVI | S32 -> S64 | 12 | True | 5 | 3 | 4 | 2 |
| KIVI | S64 -> S128 | 1 | False | 1 | 0 | 0 | 1 |

## 9. Sink Saturation Analysis

- PatternKV saturation point: `16`.
- KIVI saturation point: `64`.
- S128 cannot be used to determine whether saturation continues beyond S64 because the current implementation/validation path rejects most S128 tasks.

## 10. Minimum Effective Sink Length

- PatternKV minimum effective Sink: `16`.
- KIVI minimum effective Sink: `16`.

## 11. Quality-Bitwidth Pareto

- PatternKV best Pareto Sink from valid data: `16`.
- KIVI best Pareto Sink from valid data: `64`.
- Cross-method recommended Sink: `16`.
- See `reports/aime24_int2_wave1_v100_8gpu/wave1a3_sink_sweep/wave1a3_sink_quality_bitwidth_tradeoff.csv`.

## 12. Long-CoT Stability

See `reports/aime24_int2_wave1_v100_8gpu/wave1a3_sink_sweep/wave1a3_sink_cot_stability.csv`.

## 13. Task-Level Sink Thresholds

See `reports/aime24_int2_wave1_v100_8gpu/wave1a3_sink_sweep/wave1a3_sink_task_thresholds.csv`.

## 14. Cross-Method Consistency

- `CROSS_METHOD_SINK_EFFECT_SUPPORTED=true`.
- `CROSS_METHOD_SINK_SCALE_CONSISTENT=false`.
- Both methods improve at S16 relative to S0, supporting cross-method early-token protection.

## 15. Hypothesis Decisions

- `PATTERN_SINK_EFFECT_SUPPORTED=True`
- `KIVI_SINK_EFFECT_SUPPORTED=True`
- `CROSS_METHOD_SINK_EFFECT_SUPPORTED=True`
- `PATTERN_SINK_SATURATION_POINT=16`
- `KIVI_SINK_SATURATION_POINT=64`
- `PATTERN_MINIMUM_EFFECTIVE_SINK_LENGTH=16`
- `KIVI_MINIMUM_EFFECTIVE_SINK_LENGTH=16`
- `PATTERN_BEST_PARETO_SINK_LENGTH=16`
- `KIVI_BEST_PARETO_SINK_LENGTH=64`
- `CROSS_METHOD_RECOMMENDED_SINK_LENGTH=16`
- `FULL_AIME24_VALIDATION_RECOMMENDED=False`
- `ATTENTION_MASS_DIAGNOSTIC_RECOMMENDED=True`
- `NEXT_PRIORITY=Fix or formally define S128 sink validation before any full AIME24 expansion; use S16 as current Pareto candidate from valid data.`

## 16. Limitations

- n=12 paired diagnostic cohort, not full AIME24 accuracy.
- S128 is runtime-invalid under current validation, so conclusions are limited to S0/S16/S32/S64.
- This experiment manipulates protected token positions; it does not directly measure attention mass on early tokens.

## 17. Recommended Next Experiment

- First resolve whether S128 should include decode-time early tokens in sink semantics or whether validation should reflect prefill-only sink behavior.
- After that, rerun S128 or proceed with S16/S32-focused validation depending on the clarified semantics.
- Do not start Wave 1B, Wave 2, full AIME24, AIME25, VarN, mixed-Key, or query-aware work from this script.

## 18. Reproducibility

- Branch: `exp/aime-int2-wave1-v100-8gpu`
- HEAD: `20bc5b25c4677ea98ed0ebc6bbb1e67751c01ea7`
- Python: `/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python`
- Torch: `2.4.1+cu118`
- CUDA runtime: `11.8`
- Result dir: `results/aime24_int2_wave1_v100_8gpu_wave1a3_sink_sweep/wave1a3`
- Report dir: `reports/aime24_int2_wave1_v100_8gpu/wave1a3_sink_sweep`
