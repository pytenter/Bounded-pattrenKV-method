# Revised AIME24 Wave 1A Full Diagnostic Report

## 1. Executive Summary

- Runtime valid: `True`; completed `96/96` primary records.
- PatternKV legacy chunked: `4/12`; Pattern rolling S0/R128: `7/12`; Pattern S64/R256: `8/12`.
- K/V sensitivity: K4V2 `7/12`, K2V4 `4/12`, baseline K2V2 rolling `7/12`.
- This is a 12-task paired diagnostic cohort, not a final AIME accuracy headline benchmark.

## 2. Experimental Question

The run tests whether stable rolling recent tokens, combined Sink+Recent protection, and asymmetric K/V bitwidth changes directionally improve INT2 long-CoT fidelity.

## 3. Fixed Diagnostic Cohort

- Manifest: `configs/aime24_wave1_selected_tasks.json`
- Manifest SHA256: `ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e`
- Task count: `12` paired diagnostic task keys.

## 4. Methods and Cache Semantics

- GPU0 `pattern_legacy_chunked_k2v2_r128`: method=patternkv, cache=legacy_tuple_chunked, sink=0, recent=0, residual=128, K=2, V=2, role=PatternKV legacy baseline.
- GPU1 `pattern_rolling_k2v2_s0_r128`: method=patternkv, cache=segmented_rolling, sink=0, recent=128, residual=128, K=2, V=2, role=rolling-recent intervention.
- GPU2 `pattern_rolling_k2v2_s64_r256`: method=patternkv, cache=segmented_rolling, sink=64, recent=256, residual=128, K=2, V=2, role=Sink+Recent combined protection.
- GPU3 `pattern_rolling_k4v2_s0_r128`: method=patternkv, cache=segmented_rolling, sink=0, recent=128, residual=128, K=4, V=2, role=Key precision intervention.
- GPU4 `pattern_rolling_k2v4_s0_r128`: method=patternkv, cache=segmented_rolling, sink=0, recent=128, residual=128, K=2, V=4, role=Value precision intervention.
- GPU5 `kivi_legacy_chunked_k2v2_r128`: method=kivi_official, cache=legacy_tuple_chunked, sink=0, recent=0, residual=128, K=2, V=2, role=KIVI legacy baseline.
- GPU6 `kivi_rolling_k2v2_s0_r128`: method=kivi_official, cache=segmented_rolling, sink=0, recent=128, residual=128, K=2, V=2, role=KIVI rolling control.
- GPU7 `kivi_rolling_k2v2_s64_r256`: method=kivi_official, cache=segmented_rolling, sink=64, recent=256, residual=128, K=2, V=2, role=KIVI Sink+Recent control.

## 5. Effective Bitwidth

| config | theoretical compact bits | actual storage bits | strict accuracy | length stop rate |
| --- | ---: | ---: | ---: | ---: |
| `pattern_legacy_chunked_k2v2_r128` | 2.4449 | 3.0451 | 33.3% | 8.3% |
| `pattern_rolling_k2v2_s0_r128` | 2.6673 | 3.3267 | 58.3% | 8.3% |
| `pattern_rolling_k2v2_s64_r256` | 3.0746 | 3.7230 | 66.7% | 0.0% |
| `pattern_rolling_k4v2_s0_r128` | 3.7648 | 4.4307 | 58.3% | 0.0% |
| `pattern_rolling_k2v4_s0_r128` | 3.6152 | 4.2462 | 33.3% | 0.0% |
| `kivi_legacy_chunked_k2v2_r128` | 2.2500 | 2.3350 | 16.7% | 8.3% |
| `kivi_rolling_k2v2_s0_r128` | 2.4069 | 2.4730 | 16.7% | 8.3% |
| `kivi_rolling_k2v2_s64_r256` | 2.8262 | 2.9558 | 58.3% | 0.0% |

## 6. Runtime Validity

- Expected records: `96`; actual records: `96`.
- Runtime errors: `0`; parser failures: `2`; length truncations: `4`.
- Paired task set identical: `True`.

## 7. Strict Accuracy Results

| config | correct/total | strict accuracy | length stops | mean generated tokens | parser success |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pattern_legacy_chunked_k2v2_r128` | 4/12 | 33.3% | 1 | 11918.2 | 91.7% |
| `pattern_rolling_k2v2_s0_r128` | 7/12 | 58.3% | 1 | 12082.7 | 91.7% |
| `pattern_rolling_k2v2_s64_r256` | 8/12 | 66.7% | 0 | 9578.3 | 100.0% |
| `pattern_rolling_k4v2_s0_r128` | 7/12 | 58.3% | 0 | 8488.9 | 100.0% |
| `pattern_rolling_k2v4_s0_r128` | 4/12 | 33.3% | 0 | 11874.9 | 100.0% |
| `kivi_legacy_chunked_k2v2_r128` | 2/12 | 16.7% | 1 | 13984.5 | 100.0% |
| `kivi_rolling_k2v2_s0_r128` | 2/12 | 16.7% | 1 | 15401.4 | 100.0% |
| `kivi_rolling_k2v2_s64_r256` | 7/12 | 58.3% | 0 | 10014.8 | 100.0% |

## 8. Paired Task Outcomes

See `paired_task_outcomes.csv` for task-level correctness, stop reason, parsed answer, and generation length by config.

## 9. Rolling Recent Analysis

- Pattern legacy -> rolling: rescues=4, regressions=1, ties=7, net paired gain=3.
- Pattern rolling -> S64/R256: rescues=3, regressions=2, ties=7, net paired gain=1.
- Pattern K2V2 -> K4V2: rescues=2, regressions=2, ties=8, net paired gain=0.
- Pattern K2V2 -> K2V4: rescues=1, regressions=4, ties=7, net paired gain=-3.
- KIVI legacy -> rolling: rescues=2, regressions=2, ties=8, net paired gain=0.
- KIVI rolling -> S64/R256: rescues=6, regressions=1, ties=5, net paired gain=5.

## 10. Sink+Recent Analysis

- Pattern S64/R256 improves from `7/12` to `8/12` with net paired gain `1`.
- This is a Sink+Recent combined protection effect; it does not isolate Sink from Recent256.

## 11. Key vs Value Analysis

- K4V2 matches rolling K2V2 at `7/12` vs `7/12`; K2V4 is lower at `4/12`.
- Classification: `INCONCLUSIVE` for a positive Key/Value causal claim on this cohort; increasing Value precision alone does not recover quality here.

## 12. PatternKV vs KIVI Cross-Method Analysis

- PatternKV: legacy `4/12` -> rolling `7/12` -> S64/R256 `8/12`.
- KIVI: legacy `2/12` -> rolling `2/12` -> S64/R256 `7/12`.
- Rolling improves PatternKV more clearly than KIVI; Sink+Recent improves both in this diagnostic cohort.

## 13. Long-CoT Stability

See `cot_stability_events.csv`. Length stops remain in Pattern/KIVI legacy and S0/R128, while both S64/R256 configs have zero length stops.

## 14. Pattern Dynamic Statistics

See `pattern_dynamic_statistics.csv`. These are auxiliary dynamic-bank diagnostics only, not a formal Pattern bank drift experiment.

## 15. Quality-Bitwidth Tradeoff

See `quality_bitwidth_tradeoff.csv`. K4V2 costs more theoretical bits than K2V2 but does not improve strict accuracy on this cohort; S64/R256 improves quality while adding FP16 protected-token overhead.

## 16. Hypothesis Decisions

- `ROLLING_RECENT_HYPOTHESIS_SUPPORTED=true`
- `SINK_RECENT_PROTECTION_SUPPORTED=true`
- `KEY_SENSITIVITY_SUPPORTED=false`
- `VALUE_SENSITIVITY_SUPPORTED=false`
- `TOKEN_PROTECTION_ONLY_INSUFFICIENT=false`
- `FOLLOWUP_2X2_SINK_RECENT_RECOMMENDED=true`

## 17. Limitations

- n=12, so findings are directional paired diagnostic evidence, not statistically stable benchmark claims.
- S64/R256 changes Sink and Recent simultaneously; causal decomposition requires a 2x2 follow-up.

## 18. Recommended Wave 1A.2 / Wave 2

- Run S0/R128, S64/R128, S0/R256, S64/R256 for Sink vs Recent decomposition.
- Do not start mixed-Key, Query-aware, VarN, Pattern-MSE, pseudo-decode, or full AIME30x2 from this script.

## Research Plan Mapping

- Experiment 1 Sink-Recent: partial evidence; rolling recent is supported for PatternKV, and Sink+Recent combined protection is supported but not causally decomposed.
- Experiment 2 Key/Value asymmetry: partial evidence; K4V2 does not beat K2V2 and K2V4 regresses, so a positive asymmetry claim is inconclusive on this cohort.
- Experiment 3 assignment objective: not started.
- Experiment 4 Pattern + token-scale normalization: not started.
- Experiment 5 pseudo-decode accumulation: not started.
- Experiment 6 Pattern bank drift: auxiliary only; dynamic bank statistics were collected but this is not a formal drift experiment.

## 19. Reproducibility Information

- Branch: `exp/aime-int2-wave1-v100-8gpu`
- HEAD: `1caeff1237877e2bf1be283d57657f28ecc872db`
- Python: `/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python`
- Torch: `2.4.1+cu118`
- CUDA runtime: `11.8`
- Model: `/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B`
- Generation config hash: `a7d6b2f8bab37893b6331c66b3e5eb6a`
