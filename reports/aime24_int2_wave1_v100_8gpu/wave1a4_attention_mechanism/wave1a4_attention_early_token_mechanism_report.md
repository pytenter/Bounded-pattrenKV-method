# Wave 1A.4 Attention-Mass / Early-Token Mechanism Diagnostic

## 1. Executive Summary

- Wave 1A.4 has started from local HEAD `bc1ff8c483fc92ba0161c446398c427df0616607`.
- Pre-experiment push succeeded before this run; later push can be skipped if GitHub network is unavailable.
- Observer unit tests and smoke are complete; teacher-forcing and free-running phases are complete.
- FP16 reference trajectory generation status: `12/12` tasks available.
- Teacher-forcing valid traces: `84/84`.
- Free-running runs: `50/50` across `10` selected tasks.

## 2. Motivation

Wave 1A.3b showed that early Sink protection improves INT2 long-CoT quality, especially Pattern S16/S32 and KIVI S64/S128. Wave 1A.4 tests whether that quality change is explained by early-token attention, routing error, value-content error, or accumulated hidden-state drift.

## 3. Prior Sink Findings

- PatternKV: S0 `7/12`, S16 `9/12`, S32 `9/12`, S64 `8/12`, S128 `7/12`.
- KIVI: S0 `2/12`, S16 `6/12`, S32 `5/12`, S64 `7/12`, S128 `8/12`.

## 4. Experimental Design

- Main mode: common-trajectory teacher forcing from FP16 reference token IDs.
- Secondary mode: limited free-running observational trace.
- Current completed mode: observer smoke only.

## 5. Common-Trajectory Teacher Forcing

The offline driver uses saved FP16 token IDs as the only teacher tokens. Quantized paths do not sample or choose next tokens during mechanism collection.

## 6. Absolute Early-Window Attention Mass

- FP16 median E16 mass: `0.5611302256584167`.
- FP16 median E32 mass: `0.5630920231342316`.
- FP16 median E64 mass: `0.5650606155395508`.
- FP16 median E128 mass: `0.5801813006401062`.

## 7. Attention Enrichment

- FP16 median E16 enrichment: `39.53356170654297`.
- FP16 median E32 enrichment: `19.82754898071289`.
- FP16 median E64 enrichment: `9.989213943481445`.
- FP16 median E128 enrichment: `5.066664457321167`.

## 8. Head/Layer Localization

Top layer-head pairs by E16 mass:

- `pattern_rolling_k2v2_s0_r128` layer `23` head `10`: median E16 mass `0.965047`
- `kivi_rolling_k2v2_s0_r128` layer `23` head `10`: median E16 mass `0.963639`
- `kivi_rolling_k2v2_s128_r128` layer `23` head `10`: median E16 mass `0.959703`
- `pattern_rolling_k2v2_s16_r128` layer `23` head `10`: median E16 mass `0.957964`
- `pattern_rolling_k2v2_s128_r128` layer `23` head `10`: median E16 mass `0.957468`
- `pattern_rolling_k2v2_s0_r128` layer `23` head `26`: median E16 mass `0.957057`
- `kivi_rolling_k2v2_s16_r128` layer `23` head `10`: median E16 mass `0.955845`
- `kivi_rolling_k2v2_s0_r128` layer `23` head `26`: median E16 mass `0.955767`
- `fp16_reference` layer `23` head `10`: median E16 mass `0.955712`
- `fp16_reference` layer `23` head `26`: median E16 mass `0.952579`

## 9. Early K Reconstruction Error

Smoke K reconstruction CSV generated. Formal quantized-cache reconstruction comparisons are pending.

## 10. Early V Reconstruction Error

Smoke V reconstruction CSV generated. Formal quantized-cache reconstruction comparisons are pending.

## 11. Routing Error

| Metric | Pattern S0 | Pattern S16 | KIVI S0 | KIVI S128 |
|---|---:|---:|---:|---:|
| routing-only relative L2 | 0.29803861677646637 | 0.07333797588944435 | 0.32632026076316833 | 0.0686911977827549 |
| value-only relative L2 | 0.4004945755004883 | 0.09866815060377121 | 0.4434923678636551 | 0.10385147854685783 |
| attention-output relative L2 | 0.5134918689727783 | 0.14039704948663712 | 0.5685262680053711 | 0.13411666452884674 |

## 12. Region Contribution Error

Smoke region contribution CSV generated. Formal task-level comparisons are pending.

## 13. Routing-vs-Value Decomposition

- Pattern classification: `MIXED`.
- KIVI classification: `MIXED`.

## 14. Attention Output Error

Smoke output proxy rows exist; formal conclusions remain null.

## 15. Hidden-State Drift

Hidden-state drift formal instrumentation remains pending.

## 16. Rescue-vs-Nonrescue Analysis

Formal task-level summary is written to `reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism/wave1a4_task_mechanism_summary.csv`.

## 17. Pattern S16 vs S128

Pending formal teacher-forcing traces.

## 18. KIVI S16 vs S128

Pending formal teacher-forcing traces.

## 19. Free-Running Observational Traces

Phase B is observational and trajectory-confounded by design; the controlled mechanism evidence remains the teacher-forcing phase.

- Selected unique tasks: `10`.
- Expected free-running runs: `50`.
- Actual free-running runs: `50`.
- Runtime errors: `0`.
- NaN/Inf rows: `0`.
- Free-running support classification: `True`.

Artifacts:

- `reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism/wave1a4_free_running_selected_tasks.json`
- `reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism/wave1a4_free_running_attention_events.csv`
- `reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism/wave1a4_free_running_divergence.csv`
- `reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism/wave1a4_divergence_neighborhood_metrics.csv`
- `reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism/wave1a4_free_running_task_summary.csv`

## 20. Mechanism Decision

- `EARLY_TOKEN_ATTENTION_PRESENT=True`
- `EARLY_TOKEN_ATTENTION_ENRICHED=True`
- `PATTERN_RESCUE_MECHANISM_SUPPORTED=True`
- `KIVI_RESCUE_MECHANISM_SUPPORTED=True`
- `FREE_RUNNING_SUPPORTS_MECHANISM=True`
- `WAVE1A4_TEACHER_FORCING_COMPLETED=True`
- `WAVE1A4_FREE_RUNNING_COMPLETED=True`
- `WAVE1A4_COMPLETED=True`

## 21. Limitations

- Diagnostic n is 12 paired AIME24 tasks, not a full benchmark.
- Free-running observational traces are still separate from teacher-forcing causal comparisons.
- The observer stores sparse checkpoint reductions and reference KV captures; it does not store full attention matrices.

## 22. Recommended Next Experiment

full AIME24 validation

## 23. Reproducibility

- Branch: `exp/aime-int2-wave1-v100-8gpu`
- HEAD: `bc1ff8c483fc92ba0161c446398c427df0616607`
- Task manifest hash: `ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e`
- Generation config hash: `a7d6b2f8bab37893b6331c66b3e5eb6a`
