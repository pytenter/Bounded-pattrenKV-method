# AIME24 Wave 1A.3 Sink Length Sweep Report

## 1. Executive Summary

- Wave 1A.3b resolved the S128 boundary blocker without weakening cache validation.
- Canonical Sink semantics are `absolute_sequence_prefix`: `sink_length=N` protects the first N logical sequence tokens, including early decode tokens when prompt length is shorter than N.
- PatternKV S128 rerun is valid at 7/12, so S128 does not improve over S16/S32 and does not improve over S0 on this cohort.
- KIVI S128 rerun is valid at 8/12, one task above S64 and six paired net gains above S0.
- Cross-method evidence still supports early-token protection, but the Pareto Sink length differs by method.

## 2. Motivation

Wave 1A.3 found S16/S32/S64 valid but S128 invalid because decode append and validator used inconsistent Sink semantics. Wave 1A.3b fixes that state-machine boundary and reruns only S128.

## 3. Experimental Design

- Task manifest hash: `ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e`
- Generation config hash: `a7d6b2f8bab37893b6331c66b3e5eb6a`
- Fixed: segmented rolling, recent_length=128, K2V2, group_size=128.
- Variable: sink_length in S0, S16, S32, S64, S128.
- S0/S16/S32/S64 use previously validated records; S128 uses Wave 1A.3b rerun records.

## 4. Reuse Validation

- Original reuse validation status: `passed`.
- S128 rerun source: `results/aime24_int2_wave1_v100_8gpu_wave1a3b_s128_resolution/wave1a3b`.
- Original S128 invalid records are preserved in `results/aime24_int2_wave1_v100_8gpu_wave1a3_sink_sweep/wave1a3`.

## 5. Runtime Validity

| method | sink | records | runtime errors | length stops | valid for quality |
| --- | ---: | ---: | ---: | ---: | --- |
| PatternKV | 0 | 12 | 0 | 1 | True |
| PatternKV | 16 | 12 | 0 | 0 | True |
| PatternKV | 32 | 12 | 0 | 0 | True |
| PatternKV | 64 | 12 | 0 | 0 | True |
| PatternKV | 128 | 12 | 0 | 0 | True |
| KIVI | 0 | 12 | 0 | 1 | True |
| KIVI | 16 | 12 | 0 | 0 | True |
| KIVI | 32 | 12 | 0 | 0 | True |
| KIVI | 64 | 12 | 0 | 0 | True |
| KIVI | 128 | 12 | 0 | 0 | True |

## 6. PatternKV Sink Sweep

| sink | correct/12 | accuracy | length stops | mean tokens | theoretical bits | actual bits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7/12 | 58.3% | 1 | 12082.7 | 2.6673 | 3.3267 |
| 16 | 9/12 | 75.0% | 0 | 11223.2 | 2.7288 | 3.4095 |
| 32 | 9/12 | 75.0% | 0 | 7900.2 | 2.7971 | 3.4807 |
| 64 | 8/12 | 66.7% | 0 | 9219.6 | 2.8603 | 3.5262 |
| 128 | 7/12 | 58.3% | 0 | 9867.4 | 2.9092 | 3.5495 |

## 7. KIVI Sink Sweep

| sink | correct/12 | accuracy | length stops | mean tokens | theoretical bits | actual bits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2/12 | 16.7% | 1 | 15401.4 | 2.4069 | 2.4730 |
| 16 | 6/12 | 50.0% | 0 | 11149.3 | 2.5322 | 2.6634 |
| 32 | 5/12 | 41.7% | 0 | 7575.2 | 2.5847 | 2.6991 |
| 64 | 7/12 | 58.3% | 0 | 12155.8 | 2.5040 | 2.6045 |
| 128 | 8/12 | 66.7% | 0 | 9267.8 | 2.8507 | 3.0349 |

## 8. Paired Rescues and Regressions

| method | comparison | rescues | regressions | ties | net gain |
| --- | --- | ---: | ---: | ---: | ---: |
| PatternKV | S16 vs S0 | 4 | 2 | 6 | 2 |
| PatternKV | S32 vs S0 | 3 | 1 | 8 | 2 |
| PatternKV | S64 vs S0 | 2 | 1 | 9 | 1 |
| PatternKV | S128 vs S0 | 1 | 1 | 10 | 0 |
| PatternKV | S0 -> S16 | 4 | 2 | 6 | 2 |
| PatternKV | S16 -> S32 | 2 | 2 | 8 | 0 |
| PatternKV | S32 -> S64 | 1 | 2 | 9 | -1 |
| PatternKV | S64 -> S128 | 1 | 2 | 9 | -1 |
| KIVI | S16 vs S0 | 4 | 0 | 8 | 4 |
| KIVI | S32 vs S0 | 3 | 0 | 9 | 3 |
| KIVI | S64 vs S0 | 7 | 2 | 3 | 5 |
| KIVI | S128 vs S0 | 6 | 0 | 6 | 6 |
| KIVI | S0 -> S16 | 4 | 0 | 8 | 4 |
| KIVI | S16 -> S32 | 2 | 3 | 7 | -1 |
| KIVI | S32 -> S64 | 5 | 3 | 4 | 2 |
| KIVI | S64 -> S128 | 3 | 2 | 7 | 1 |

## 9. Sink Saturation Analysis

- PatternKV saturation point: `16`. S16 and S32 are tied at 9/12, then S64 drops to 8/12 and S128 to 7/12.
- KIVI saturation point: `not_reached`. S128 reaches 8/12 and S64->S128 has net paired gain 1.

## 10. Minimum Effective Sink Length

- PatternKV minimum effective Sink: `16`.
- KIVI minimum effective Sink: `16`.

## 11. Quality-Bitwidth Pareto

- PatternKV best Pareto Sink: `16`.
- KIVI best Pareto Sink: `128`.
- Cross-method recommended Sink: `16`.
- S128 has higher bit cost and is not PatternKV-Pareto on this cohort.

## 12. Long-CoT Stability

- All valid S128 records stopped by EOS; S128 length stops are zero for both methods.
- See `reports/aime24_int2_wave1_v100_8gpu/wave1a3_sink_sweep/wave1a3_sink_cot_stability.csv`.

## 13. Task-Level Sink Thresholds

- See `reports/aime24_int2_wave1_v100_8gpu/wave1a3_sink_sweep/wave1a3_sink_task_thresholds.csv`.

## 14. Cross-Method Consistency

- `CROSS_METHOD_SINK_EFFECT_SUPPORTED=true`.
- `CROSS_METHOD_SINK_SCALE_CONSISTENT=false`.
- Both methods benefit from adding Sink, but PatternKV peaks at S16/S32 while KIVI peaks at S128.

## 15. Hypothesis Decisions

- `PATTERN_SINK_EFFECT_SUPPORTED=True`
- `PATTERN_MINIMUM_EFFECTIVE_SINK_LENGTH=16`
- `PATTERN_BEST_PARETO_SINK_LENGTH=16`
- `PATTERN_SINK_SATURATION_POINT=16`
- `PATTERN_SINK_SWEEP_MONOTONIC_ACCURACY=False`
- `KIVI_SINK_EFFECT_SUPPORTED=True`
- `KIVI_MINIMUM_EFFECTIVE_SINK_LENGTH=16`
- `KIVI_BEST_PARETO_SINK_LENGTH=128`
- `KIVI_SINK_SATURATION_POINT=not_reached`
- `KIVI_SINK_SWEEP_MONOTONIC_ACCURACY=False`
- `CROSS_METHOD_SINK_EFFECT_SUPPORTED=True`
- `CROSS_METHOD_SINK_SCALE_CONSISTENT=False`
- `CROSS_METHOD_RECOMMENDED_SINK_LENGTH=16`
- `FULL_AIME24_VALIDATION_RECOMMENDED=True`
- `ATTENTION_MASS_DIAGNOSTIC_RECOMMENDED=True`
- `NEXT_PRIORITY=Run attention-mass / early-token mechanism diagnostics before broadening methods; use Pattern S16/S32 and KIVI S64/S128 as validation candidates.`

## 16. Limitations

- n=12 diagnostic cohort, not full AIME24 headline accuracy.
- S128 is an absolute early-sequence Sink and may include early decode tokens for prompt lengths below 128.
- This experiment manipulates protection positions; it does not directly prove attention mass on those positions.

## 17. Recommended Next Experiment

- Run attention-mass / early-token mechanism diagnostics before launching new method families.
- For full AIME24 validation, use a small candidate set rather than all sweep points: Pattern S0/R128, Pattern S16/R128 or S32/R128, KIVI S0/R128, and KIVI S64/R128 or S128/R128.
- Do not start Wave 1B, Wave 2, AIME25, VarN, mixed-Key, Hadamard, query-aware, Pattern-MSE, or pseudo-decode from this run.

## 18. Reproducibility

- Branch: `exp/aime-int2-wave1-v100-8gpu`
- HEAD at report generation: `fdacdc668434c4ace1602a54ff1b88fa0ff78d6c`
- Starting HEAD for 1A.3b: `fdacdc668434c4ace1602a54ff1b88fa0ff78d6c`
- Python: `/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python`
- Torch: `2.4.1+cu118`
- CUDA runtime: `11.8`
- Model: `/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B`
