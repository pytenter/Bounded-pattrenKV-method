# AIME24 Full30 Three-Seed Formal Validation

## 1. Executive Summary

- Full run complete: `True`
- Expected records: `630`
- Actual records: `630`
- Runtime errors: `0`
- Parser failures: `35`
- Length truncations: `65`
- Pattern Sink16 supported: `True`
- KIVI Sink16 supported: `True`
- Cross-method Sink effect supported: `True`

## 2. Experimental Question

This formal validation tests whether rolling+S16 improves AIME24 long-CoT quality over paper and rolling-only baselines across all 30 AIME24 problems and three fixed seeds.

## 3. Dataset

- Full AIME24 source: `datasets/aime/aime24.jsonl`
- Full30 task hash: `07ec3f0c489406676be9d6057e2f97c9c32bc18e856d13df1d05c76724cbb08f`
- Problems: `30` unique problem_id values `0..29`.

## 4. Generation Configuration

- Generation config hash: `a7d6b2f8bab37893b6331c66b3e5eb6a`
- Seeds: `42`, `43`, `44`
- `temperature=0.6`, `top_p=0.95`, `do_sample=true`, `max_new_tokens=32768`, `dtype=float16`.

## 5. Seven Configurations

| config | method | cache | sink | recent | K/V bits |
| --- | --- | --- | ---: | ---: | --- |
| fp16 | fp16 | fp16 | 0 | 0 | 16/16 |
| patternkv_paper | patternkv_paper | legacy_tuple_chunked | 0 | 128 | 2/2 |
| pattern_rolling_s0_r128 | patternkv | segmented_rolling | 0 | 128 | 2/2 |
| pattern_rolling_s16_r128 | patternkv | segmented_rolling | 16 | 128 | 2/2 |
| kivi_paper | kivi_paper_g128 | legacy_tuple_chunked | 0 | 128 | 2/2 |
| kivi_rolling_s0_r128 | kivi_official | segmented_rolling | 0 | 128 | 2/2 |
| kivi_rolling_s16_r128 | kivi_official | segmented_rolling | 16 | 128 | 2/2 |

## 6. Paper Configuration Audit

- PatternKV paper audit: `reports/aime24_full30_3seed/patternkv_paper_config_audit.md`
- KIVI paper audit: `reports/aime24_full30_3seed/kivi_paper_config_audit.md`

## 7. Runtime Completeness

- Missing records: `0`
- Duplicate records: `0`

## 8. Main Three-Seed Accuracy Results

| config | seed42 | seed43 | seed44 | mean | std | length truncations | actual bits mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 0.4333 | 0.3667 | 0.4000 | 0.4000 | 0.0272 | 4 | 16.0000 |
| patternkv_paper | 0.2000 | 0.3000 | 0.2000 | 0.2333 | 0.0471 | 18 | 3.2593 |
| pattern_rolling_s0_r128 | 0.3000 | 0.2667 | 0.3000 | 0.2889 | 0.0157 | 10 | 3.2617 |
| pattern_rolling_s16_r128 | 0.3667 | 0.4667 | 0.5333 | 0.4556 | 0.0685 | 0 | 3.3149 |
| kivi_paper | 0.1333 | 0.2000 | 0.2000 | 0.1778 | 0.0314 | 15 | 2.4826 |
| kivi_rolling_s0_r128 | 0.1333 | 0.2000 | 0.2000 | 0.1778 | 0.0314 | 15 | 2.4826 |
| kivi_rolling_s16_r128 | 0.4667 | 0.4000 | 0.4667 | 0.4444 | 0.0314 | 3 | 2.5456 |

## 9. PatternKV Paper vs Rolling vs Sink16

| seed | rescues | regressions | ties | net |
| --- | ---: | ---: | ---: | ---: |
| 42 | 4 | 1 | 25 | 3 |
| 43 | 2 | 3 | 25 | -1 |
| 44 | 5 | 2 | 23 | 3 |
| aggregate | 11 | 6 | 73 | 5 |

| seed | rescues | regressions | ties | net |
| --- | ---: | ---: | ---: | ---: |
| 42 | 6 | 1 | 23 | 5 |
| 43 | 7 | 2 | 21 | 5 |
| 44 | 10 | 0 | 20 | 10 |
| aggregate | 23 | 3 | 64 | 20 |

## 10. Pattern S0 vs S16

| seed | rescues | regressions | ties | net |
| --- | ---: | ---: | ---: | ---: |
| 42 | 4 | 2 | 24 | 2 |
| 43 | 7 | 1 | 22 | 6 |
| 44 | 9 | 2 | 19 | 7 |
| aggregate | 20 | 5 | 65 | 15 |

## 11. KIVI S0 vs S16

| seed | rescues | regressions | ties | net |
| --- | ---: | ---: | ---: | ---: |
| 42 | 10 | 0 | 20 | 10 |
| 43 | 8 | 2 | 20 | 6 |
| 44 | 9 | 1 | 20 | 8 |
| aggregate | 27 | 3 | 60 | 24 |

## 12. Seed Stability

- `fp16` seed std: `0.0272`; min/max: `0.3667` / `0.4333`.
- `patternkv_paper` seed std: `0.0471`; min/max: `0.2000` / `0.3000`.
- `pattern_rolling_s0_r128` seed std: `0.0157`; min/max: `0.2667` / `0.3000`.
- `pattern_rolling_s16_r128` seed std: `0.0685`; min/max: `0.3667` / `0.5333`.
- `kivi_paper` seed std: `0.0314`; min/max: `0.1333` / `0.2000`.
- `kivi_rolling_s0_r128` seed std: `0.0314`; min/max: `0.1333` / `0.2000`.
- `kivi_rolling_s16_r128` seed std: `0.0314`; min/max: `0.4000` / `0.4667`.

## 13. Task-Level Stability

- Task-level classifications across pre-registered comparisons: `{'SEED_SENSITIVE': 16, 'NO_CHANGE': 65, 'STABLE_RESCUE': 29, 'PARTIAL_RESCUE': 38, 'STABLE_REGRESSION': 2}`
- Full table: `aime24_full30_task_seed_consistency.csv`.

## 14. Generation-Length Behavior

- `fp16` mean/median/P90/P95/max generated tokens: `14778.7` / `14046.0` / `24424` / `30582` / `32768`.
- `patternkv_paper` mean/median/P90/P95/max generated tokens: `15812.6` / `13082.0` / `32768` / `32768` / `32768`.
- `pattern_rolling_s0_r128` mean/median/P90/P95/max generated tokens: `14376.3` / `13387.5` / `32768` / `32768` / `32768`.
- `pattern_rolling_s16_r128` mean/median/P90/P95/max generated tokens: `12385.5` / `11822.5` / `20754` / `24415` / `28577`.
- `kivi_paper` mean/median/P90/P95/max generated tokens: `16276.4` / `14208.0` / `32768` / `32768` / `32768`.
- `kivi_rolling_s0_r128` mean/median/P90/P95/max generated tokens: `16276.4` / `14208.0` / `32768` / `32768` / `32768`.
- `kivi_rolling_s16_r128` mean/median/P90/P95/max generated tokens: `13151.8` / `12126.5` / `20940` / `29320` / `32768`.

## 15. Effective Bitwidth

- `fp16` theoretical bits mean `16.0000`, actual implementation bits mean `16.0000`.
- `patternkv_paper` theoretical bits mean `2.5051`, actual implementation bits mean `3.2593`.
- `pattern_rolling_s0_r128` theoretical bits mean `2.5224`, actual implementation bits mean `3.2617`.
- `pattern_rolling_s16_r128` theoretical bits mean `2.5706`, actual implementation bits mean `3.3149`.
- `kivi_paper` theoretical bits mean `2.4052`, actual implementation bits mean `2.4826`.
- `kivi_rolling_s0_r128` theoretical bits mean `2.4052`, actual implementation bits mean `2.4826`.
- `kivi_rolling_s16_r128` theoretical bits mean `2.4619`, actual implementation bits mean `2.5456`.

## 16. Quality-vs-Bitwidth Tradeoff

Quality/bitwidth tradeoffs should be read from `aime24_full30_config_summary.csv` and `aime24_full30_bitwidth_summary.csv`; actual storage includes Python tensor metadata and PatternKV assignment/gate/centroid storage.

## 17. Relation to Wave 1A Mechanism Findings

Wave 1A.4 found early-token attention present/enriched and classified both PatternKV and KIVI mechanisms as mixed routing plus value-content protection. This run is benchmark validation only; no observer traces were collected.

## 18. Hypothesis Decisions

- `FULL_AIME24_PATTERN_SINK_SUPPORTED=True`
- `PATTERN_SINK_SEED_CONSISTENT=True`
- `FULL_AIME24_KIVI_SINK_SUPPORTED=True`
- `CROSS_METHOD_SINK_EFFECT_SUPPORTED=True`
- `PATTERN_FINAL_METHOD_BEATS_PAPER_BASELINE=True`

## 19. Limitations

- Three seeds provide stronger validation than Wave 1A diagnostics but still represent task-seed samples, not 90 independent problems.
- Paper baselines use the repository's available paper-aligned reproduction path for DeepSeek-R1-Distill-Llama-8B.

## 20. Next Experiment

- `AIME25_VALIDATION_RECOMMENDED=True`
- `PSEUDO_DECODE_RECOMMENDED=False`
- `NEXT_PRIORITY=AIME25 full30 validation`

## 21. Reproducibility

- Result dir: `results/aime24_full30_3seed`
- Report dir: `reports/aime24_full30_3seed`
- Config manifest: `configs/aime24_full30_formal_validation.json`
