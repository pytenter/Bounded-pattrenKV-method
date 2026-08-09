# AIME24 Pseudo-Decode Formal Result Audit

## 1. Formal Data Integrity

Metric rows `6096`, gap rows `2880`, completeness rows `804`.

## 2. Core Completeness

`core_matched_experiment_complete=True` for checkpoints `128,512,1024,2048,4096`.

## 3. Hardware-Limited Extended Rows

`extended_long_matched_experiment_complete=False`; reason `static_full_prefix_oom_24gb`; failed rows verified hardware-limited static `True`.

## 4. Paper-vs-S0 Equality

Pattern exact fraction `1.0`. KIVI exact fraction `1.0`.

## 5. Runtime Config Resolution

Pattern paper/S0 runtime equivalent `True`. KIVI paper/S0 runtime equivalent `True`.

## 6. Result Provenance

Paper and S0 labels use distinct shard files; equality is explained by resolved runtime semantics, not by a result file collision.

## 7. S0-vs-S16 Provenance

`sink_pair_result_provenance_valid=True`.

## 8. AUC Definition Audit

`core_auc_definition_valid=True`. All core AUC rows have `n_available=5`; 8192/16384 are excluded from matched AUC.

## 9. Decision-Layer Inputs

Decision tables are emitted for checkpoint medians, task growth, multi-metric sink AUC, and Pattern S16 residual anatomy.

## 10. Audit Verdict

`formal_sink_conclusion_valid=True`. `paper_vs_s0_comparison_informative=False`.
