# AIME24 Wave 1A.2 Sink x Recent Causal Report

## 1. Executive Summary

- Wave 1A.2 completed the planned Sink x Recent 2x2 decomposition on the fixed 12-task paired diagnostic cohort.
- Four configurations were reused from the approved Wave 1A run and four missing configurations were newly run under the same manifest, model, generation config, and code HEAD.
- PatternKV results: S0/R128 `7/12`, S64/R128 `8/12`, S0/R256 `4/12`, S64/R256 `8/12`.
- KIVI results: S0/R128 `2/12`, S64/R128 `7/12`, S0/R256 `3/12`, S64/R256 `7/12`.
- The directional signal favors Sink64 protection over Recent256 alone on this cohort; Recent256 without Sink does not recover PatternKV and only weakly moves KIVI.

## 2. Motivation from Wave 1A

Wave 1A found that S64/R256 improved over S0/R128, but that comparison changed Sink and Recent simultaneously. Wave 1A.2 isolates the two factors with S0/R128, S64/R128, S0/R256, and S64/R256 for both PatternKV and KIVI.

## 3. Experimental Design

- Fixed task manifest: `configs/aime24_wave1_selected_tasks.json`
- Task manifest hash: `ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e`
- Generation config hash: `a7d6b2f8bab37893b6331c66b3e5eb6a`
- Model: `/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B`
- Factor A: Sink 0 vs 64.
- Factor B: Recent 128 vs 256.

## 4. Reused vs Newly Run Results

| GPU | config | method | sink | recent | source |
| ---: | --- | --- | ---: | ---: | --- |
| 0 | `pattern_rolling_k2v2_s0_r128` | PatternKV | 0 | 128 | reused_wave1a |
| 1 | `pattern_rolling_k2v2_s64_r128` | PatternKV | 64 | 128 | newly_run_wave1a2 |
| 2 | `pattern_rolling_k2v2_s0_r256` | PatternKV | 0 | 256 | newly_run_wave1a2 |
| 3 | `pattern_rolling_k2v2_s64_r256` | PatternKV | 64 | 256 | reused_wave1a |
| 4 | `kivi_rolling_k2v2_s0_r128` | KIVI | 0 | 128 | reused_wave1a |
| 5 | `kivi_rolling_k2v2_s64_r128` | KIVI | 64 | 128 | newly_run_wave1a2 |
| 6 | `kivi_rolling_k2v2_s0_r256` | KIVI | 0 | 256 | newly_run_wave1a2 |
| 7 | `kivi_rolling_k2v2_s64_r256` | KIVI | 64 | 256 | reused_wave1a |

## 5. Runtime Validity

| config | expected | actual | runtime errors | parser failures | length truncations | missing | duplicates |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `pattern_rolling_k2v2_s0_r128` | 12 | 12 | 0 | 1 | 1 |  |  |
| `pattern_rolling_k2v2_s64_r128` | 12 | 12 | 0 | 0 | 0 |  |  |
| `pattern_rolling_k2v2_s0_r256` | 12 | 12 | 0 | 0 | 1 |  |  |
| `pattern_rolling_k2v2_s64_r256` | 12 | 12 | 0 | 0 | 0 |  |  |
| `kivi_rolling_k2v2_s0_r128` | 12 | 12 | 0 | 0 | 1 |  |  |
| `kivi_rolling_k2v2_s64_r128` | 12 | 12 | 0 | 0 | 0 |  |  |
| `kivi_rolling_k2v2_s0_r256` | 12 | 12 | 0 | 0 | 2 |  |  |
| `kivi_rolling_k2v2_s64_r256` | 12 | 12 | 0 | 0 | 0 |  |  |

## 6. PatternKV 2x2 Results

| config | correct/12 | accuracy | length stops | mean gen tokens | theoretical bits | actual bits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `S0/R128` | 7/12 | 58.3% | 1 | 12082.7 | 2.6673 | 3.3267 |
| `S64/R128` | 8/12 | 66.7% | 0 | 9219.6 | 2.8603 | 3.5262 |
| `S0/R256` | 4/12 | 33.3% | 1 | 13839.6 | 2.8653 | 3.4995 |
| `S64/R256` | 8/12 | 66.7% | 0 | 9578.3 | 3.0746 | 3.7230 |

## 7. KIVI 2x2 Results

| config | correct/12 | accuracy | length stops | mean gen tokens | theoretical bits | actual bits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `S0/R128` | 2/12 | 16.7% | 1 | 15401.4 | 2.4069 | 2.4730 |
| `S64/R128` | 7/12 | 58.3% | 0 | 12155.8 | 2.5040 | 2.6045 |
| `S0/R256` | 3/12 | 25.0% | 2 | 16671.1 | 2.5441 | 2.6033 |
| `S64/R256` | 7/12 | 58.3% | 0 | 10014.8 | 2.8262 | 2.9558 |

## 8. Sink Main Effect

| method | contrast | rescues | regressions | ties | net gain | accuracy delta | length-stop delta | bit delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PatternKV | Sink effect @ R128 | 2 | 1 | 9 | 1 | 0.083 | -1 | 0.1930 |
| PatternKV | Sink effect @ R256 | 5 | 1 | 6 | 4 | 0.333 | -1 | 0.2093 |
| KIVI | Sink effect @ R128 | 7 | 2 | 3 | 5 | 0.417 | -1 | 0.0971 |
| KIVI | Sink effect @ R256 | 4 | 0 | 8 | 4 | 0.333 | -2 | 0.2821 |

## 9. Recent Main Effect

| method | contrast | rescues | regressions | ties | net gain | accuracy delta | length-stop delta | bit delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PatternKV | Recent effect @ S0 | 2 | 5 | 5 | -3 | -0.250 | 0 | 0.1980 |
| PatternKV | Recent effect @ S64 | 2 | 2 | 8 | 0 | 0.000 | 0 | 0.2143 |
| KIVI | Recent effect @ S0 | 2 | 1 | 9 | 1 | 0.083 | 1 | 0.1372 |
| KIVI | Recent effect @ S64 | 1 | 1 | 10 | 0 | 0.000 | 0 | 0.3222 |

## 10. Sink x Recent Interaction

- PatternKV accuracy interaction effect: `0.250`.
- KIVI accuracy interaction effect: `-0.083`.
- Positive interaction here means the measured S64/R256 outcome exceeds the additive expectation from individual Sink64 and Recent256 changes. Because n=12, this is descriptive rather than a statistical proof.

## 11. Task-Level Rescue Matrix

See `reports/aime24_int2_wave1_v100_8gpu/wave1a2_sink_recent/sink_recent_task_causal_matrix.csv` for per-task categories including SINK_ONLY_RESCUE, RECENT_ONLY_RESCUE, COMBINATION_ONLY_RESCUE, COMBINATION_REGRESSION, ALWAYS_CORRECT, ALWAYS_WRONG, and MIXED_NONMONOTONIC.

## 12. Long-CoT Stability

See `reports/aime24_int2_wave1_v100_8gpu/wave1a2_sink_recent/wave1a2_cot_stability_events.csv`. Length truncations are concentrated in S0/R128 or S0/R256; S64/R128 and S64/R256 finish normally in both methods on this cohort.

## 13. Effective Bitwidth Tradeoff

See `reports/aime24_int2_wave1_v100_8gpu/wave1a2_sink_recent/wave1a2_quality_bitwidth_tradeoff.csv`. S64/R128 adds less FP16-token overhead than S64/R256 while matching S64/R256 strict accuracy for both PatternKV and KIVI in this run.

## 14. Cross-Method Comparison

- PatternKV and KIVI both show a positive Sink64 main-effect signal at R128.
- Recent256 alone is not supported as the main driver: PatternKV drops from 7/12 to 4/12 at S0, while KIVI moves from 2/12 to 3/12.
- The cross-method commonality supports token-position protection, specifically early-token Sink protection, as the next immediate axis to sweep.

## 15. Hypothesis Decisions

- `PATTERN_SINK_MAIN_EFFECT_SUPPORTED=true`
- `PATTERN_RECENT_MAIN_EFFECT_SUPPORTED=false`
- `PATTERN_SINK_RECENT_INTERACTION_SUPPORTED=false`
- `KIVI_SINK_MAIN_EFFECT_SUPPORTED=true`
- `KIVI_RECENT_MAIN_EFFECT_SUPPORTED=false`
- `KIVI_SINK_RECENT_INTERACTION_SUPPORTED=false`
- `TOKEN_PROTECTION_CROSS_METHOD=true`
- `PATTERN_SPECIFIC_INTERACTION=false`
- `NEXT_PRIORITY=Sink length sweep: S0 / S16 / S32 / S64 / S128`

## 16. Limitations

- This is a 12-task paired diagnostic cohort, not a full AIME benchmark.
- The analysis observes outcome changes from protected token positions; it does not directly measure attention mass on sink or recent tokens.
- No core cache, quantization, assignment, centroid, V gate, or fused-kernel semantics were changed in this round.

## 17. Recommended Next Experiment

- Sink length sweep: S0 / S16 / S32 / S64 / S128.
- Do not start Wave 1B, Wave 2, VarN, mixed-Key, query-aware, pseudo-decode, or AIME25 from this result alone.

## 18. Reproducibility

- Branch: `exp/aime-int2-wave1-v100-8gpu`
- HEAD: `242a3a1b7e789a505006d74450f51a45ccfb055c`
- Python: `/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python`
- Torch: `2.4.1+cu118`
- CUDA runtime: `11.8`
- New result dir: `results/aime24_int2_wave1_v100_8gpu_wave1a2_sink_recent/wave1a2`
- Report dir: `reports/aime24_int2_wave1_v100_8gpu/wave1a2_sink_recent`
