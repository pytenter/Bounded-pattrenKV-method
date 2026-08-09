# QK / Routing / V-Direction Propagation Diagnostic

## 1. Executive Summary

- Q1 Q direction accumulation supported: `True`; K direction accumulation supported: `True`.
- Q2 attention routing accumulation supported: `True`.
- Q3 V direction/content accumulation supported: `True`.
- Q4 local oracle classification: `VALUE_DOMINATED`.

## 2. Motivation After VarN Intervention

- VarN reduced norm drift but did not reduce hidden/attention accumulation, so this diagnostic separates routing from value/content channels.

## 3. Frozen Cohort / Provenance

- Parent commit: `f7f6ca9954daa76cb702941f1b018ae294c0e378`.
- Source VarN result commit: `2f63bddef151df0f32a40d51c73f125a8089800e`.
- Reused exact VarN six-task subset.

## 4. Matched Static-vs-Pseudo Protocol

- Static and pseudo are each compared only to their matched FP16 execution path.
- Accumulation is pseudo degradation minus static degradation.

## 5. Observer Non-Invasiveness

- Routing observer non-invasive: `True`.
- Oracle diagnostic non-invasive: `True`.

## 6. Q Direction Drift

- Median Q direction ACC_AUC: `0.00867817169947216`; positive tasks `6/6`.

## 7. K Direction Drift

- Median K direction ACC_AUC: `0.008906321989982757`; positive tasks `6/6`.

## 8. V Direction Drift

- Median V source direction ACC_AUC: `0.07337281958848507`; positive tasks `6/6`.

## 9. QK Attention-Logit Drift

- Median QK logit relative-L2 ACC_AUC: `0.049613011127803475`; positive tasks `6/6`.

## 10. Attention Ranking Drift

- Top-k agreement/overlap rows are in `qk_logit_metrics.csv.gz` and matched-path accumulation rows are in `recursive_channel_gap.csv.gz`.

## 11. Softmax Routing Drift

- Median attention JS ACC_AUC: `0.008618724828167501`; positive tasks `6/6`.
- Median attention TV ACC_AUC: `0.06432966065767687`; positive tasks `6/6`.

## 12. Early/Recent Attention-Mass Drift

- E16/E32/E64/E128 and Recent128 mass rows are included in attention routing metrics.

## 13. Attention-Weighted V Error

- Median FP-attention-weighted V direction ACC_AUC: `0.019521914300668186`; positive tasks `6/6`.

## 14. Routing-vs-Value Oracle Decomposition

- Routing-only output ACC_AUC median (`A_Q @ V_FP`): `0.11617226718226448`.
- Value-only output ACC_AUC median (`A_FP @ V_Q`): `0.28754451929125935`.
- Dominance ratio: `0.4040148894828608`.

## 15. Static vs Pseudo Channel Accumulation

- All channel accumulation rows use matched static/pseudo deltas only.

## 16. Channel AUC

- AUC uses trapezoidal integration over log2(checkpoint) for 128, 512, 1024, 2048, 4096.

## 17. Layerwise Propagation

- Layerwise rows: `60`.

## 18. Per-Task Dominance

- Routing-dominant tasks: `0/6`.
- Value-dominant tasks: `6/6`.
- Task rows: `6`.

## 19. Recursive Propagation Classification

- `RECURSIVE_PROPAGATION_CLASSIFICATION=VALUE_DOMINATED`.

## 20. Implications for Next Algorithm

- `NEXT_PRIORITY=attention-weighted / direction-preserving Value quantization`.

## 21. Limitations

- Oracle substitution is checkpoint-local and supports channel attribution; it is not a full future-trajectory causal intervention.

## 22. Reproducibility

- No new generation, seed, prompt, sampling, quantizer, Hadamard, VarN, Sink, Recent, or assignment objective was used.
