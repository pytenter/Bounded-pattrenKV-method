# AIME24 Pseudo-Decode Accumulation Report

## 1. Executive Summary

Finding A: accumulated quantization error exists in the matched-path core experiment: `True`. Pattern S0 and KIVI S0 both show positive long-checkpoint median gaps and positive hidden/attention/KL accumulation AUC on the frozen 12-task cohort.

Finding B: Sink16 reduces accumulated error: Pattern `True`, KIVI `True`, cross-method `True`. This is a cohort-level diagnostic result, not a claim that Sink16 is universally optimal.

Finding C: after Sink16, residual Pattern error is classified as `ACCUMULATION_DOMINATED`. Static one-step degradation remains small relative to pseudo degradation in the long core checkpoints, but token-norm tail evidence is `insufficient_data`.

## 2. Experiment Scope

This report uses only the completed matched-path formal artifacts for 12 AIME24 task trajectories and 6 quantized configs. No new generation, pseudo-decode, static replay, or GPU long-run is used.

## 3. Matched-Path Definition

`static_degradation = D(Q_static, FP16_static)`, `pseudo_degradation = D(Q_pseudo, FP16_pseudo)`, and `accumulation_gap = pseudo_degradation - static_degradation`. The FP16 execution-path baseline is not double-subtracted.

## 4. Core Completion vs Extended Hardware Limit

`CORE_MATCHED_EXPERIMENT_COMPLETE=True` for checkpoints `128, 512, 1024, 2048, 4096`.
`EXTENDED_LONG_MATCHED_EXPERIMENT_COMPLETE=False` because static full-prefix replay at `8192/16384` exceeds 24GB RTX3090 memory. The 42 failed rows are unavailable extended matched static rows, not model failures.

## 5. Paper-vs-S0 Equality Audit

Pattern equality fraction: `1.0` over `1592` compared values.
KIVI equality fraction: `1.0` over `1592` compared values.
The paper-labelled and S0-labelled configurations are runtime-equivalent in this pseudo-decode harness; they therefore do not constitute an independent paper-vs-rolling comparison in this experiment.

## 6. Config Provenance Audit

`patternkv_paper` and `pattern_rolling_k2v2_s0_r128` both resolve to PatternKV segmented rolling, sink 0, recent 128, residual 128, K2/V2, group 128. `kivi_paper_g128` and `kivi_rolling_k2v2_s0_r128` both resolve to KIVI official segmented sink/recent cache semantics, sink 0, recent 128, residual 128, K2/V2, group 128.

## 7. Sink-Pair Provenance Validation

`SINK_PAIR_RESULT_PROVENANCE_VALID=True`. S0 and S16 use distinct config labels, distinct shard files, and distinct sink lengths for both PatternKV and KIVI.

## 8. Static Degradation Curves

For Pattern S16, long-core median static degradation stays near the one-step quantized representation floor for hidden/attention/KL metrics.

## 9. Pseudo Degradation Curves

For Pattern S16, pseudo degradation becomes much larger than static degradation after 512 tokens, consistent with recursive cache feedback amplifying the initial perturbation.

## 10. Accumulation Gap Curves

Pattern S0 and KIVI S0 have positive median accumulation gaps on hidden/attention/KL metrics at the 1024, 2048, and 4096 core checkpoints. Pattern S16 still has positive gaps, but they are materially smaller.

## 11. Accumulation AUC

Core AUC integrates accumulation gap over `x = log2(checkpoint)` using only the five matched checkpoints. Pseudo-only 8192/16384 rows are excluded.
Pattern S0 hidden L2 median AUC `1.4843093160598073`; Pattern S16 `0.5465083790186327`.
KIVI S0 hidden L2 median AUC `1.7127258986001834`; KIVI S16 `0.6169279780006036`.

## 12. Pattern S0 vs S16

hidden_relative_L2: median_delta `-0.9615185058210045`, improved `12/12`
attention_output_relative_L2: median_delta `-0.9615185058210045`, improved `12/12`
next_token_KL: median_delta `-0.17447546871147068`, improved `12/12`
target_token_NLL_delta: median_delta `-0.5772457927396317`, improved `9/12`

## 13. KIVI S0 vs S16

hidden_relative_L2: median_delta `-1.0693700152332895`, improved `12/12`
attention_output_relative_L2: median_delta `-1.0693700152332895`, improved `12/12`
next_token_KL: median_delta `-0.3425638193266636`, improved `12/12`
target_token_NLL_delta: median_delta `-0.07048986811423674`, improved `9/12`

## 14. Multi-Metric Sink Consistency

Hidden L2, attention-output L2, hidden cosine loss, and KL all have negative S16-S0 median AUC deltas with a majority of tasks improved for both methods. NLL is directionally improved but noisier. Top1 disagreement is mostly tied at zero and is not sensitive in this cohort.

## 15. Cross-Method Sink Mechanism

Together with Wave1A.4, the result supports a mechanism in which quantization errors on early, highly attended tokens act as an initial perturbation source whose influence propagates through later hidden states and Q/K/V computations. This supports, but does not mathematically prove, the early-error-as-accumulation-seed hypothesis.

## 16. Pattern S16 Residual Error Anatomy

Pattern S16 long-core median accumulation fraction across hidden/attention/KL families is `0.992932413742805`. The residual classification is `ACCUMULATION_DOMINATED`.

## 17. Static vs Accumulation Dominance

`REMAINING_ERROR_ACCUMULATION_DOMINATED=True` and `SINGLE_STEP_REPRESENTATION_ERROR_DOMINANT=False`. The descriptive fraction is not an orthogonal causal decomposition; it only reports the matched-path ratio `A/P` where `P > epsilon` and `A >= 0`.

## 18. Optional Norm Evidence

`TOKEN_NORM_ACCUMULATION_SUPPORTED=insufficient_data`. The formal run does not contain non-empty norm-tail formal metrics, so VarN is plausible but not confirmed by this audit.

## 19. Extended 8K/16K Pseudo-Only Observations

Pseudo rows at 8192/16384 are present for some tasks/configs, but they are not used for matched accumulation decisions because the paired static rows are unavailable under the 24GB hardware limit.

## 20. Hypothesis Decisions

`PSEUDODECODE_ACCUMULATION_SUPPORTED=True`.
`PATTERN_SINK_REDUCES_ACCUMULATION=True`.
`KIVI_SINK_REDUCES_ACCUMULATION=True`.
`EARLY_ERROR_AS_ACCUMULATION_SEED_SUPPORTED=True`.

## 21. Implication for Next Experiment

`NEXT_PRIORITY=norm-tail instrumentation plus small VarN diagnostic before assignment-objective work`. The data do not point to static representation error as the dominant remaining bottleneck after Sink16; however, norm-tail evidence is insufficient, so a full VarN commitment should be preceded by explicit norm-tail instrumentation or a small diagnostic.

## 22. Limitations

The cohort has 12 frozen tasks. The extended 8K/16K matched static experiment is hardware-limited on RTX3090 24GB. Paper-vs-S0 labels are aliases in this formal harness and should not be interpreted as a separate paper-vs-rolling ablation.

## 23. Reproducibility

Run `scripts/finalize_aime24_pseudodecode_formal.py` from commit `08e8334` or later on branch `exp/aime-pseudodecode-3090-8gpu`. The script reads only existing CSV/JSON formal artifacts and writes deterministic audit/decision tables.
