# AIME24 Pseudo-Decode Accumulation Report

## Executive Summary

The formal matched-path accumulated-error run completed for the core checkpoints `128, 512, 1024, 2048, 4096` across 12 frozen AIME24 tasks and 6 quantized configs.

- `FORMAL_RUN_APPROVED=True`
- `formal_core_matched_checkpoints_complete=True`
- `formal_run_complete=False`
- Metric rows: `6096`
- Matched accumulation gap rows: `2880`
- Completeness rows: `804`
- Failed rows: `42`

## Matched-Path Definition

`static_degradation = D(Q_static, FP16_static)` and `pseudo_degradation = D(Q_pseudo, FP16_pseudo)`. The reported accumulation gap is `pseudo_degradation - static_degradation`; no FP16 execution-path baseline is double-subtracted.

## Completion

The core matched checkpoints are complete. The unavailable formal rows are static full-prefix jobs at checkpoint `8192` or `16384`; these OOM on 24GB RTX3090 and are recorded in `formal_completeness_audit.csv`. Pseudo rows at those long checkpoints are retained, but accumulation gaps require matched static+pseudo pairs and therefore summarize only paired checkpoints.

## Sink Pair AUC

Primary sink-pair comparisons use final-layer `hidden_relative_L2` accumulation AUC.

- Pattern S16 vs S0: paired_n `12`, median_delta `-0.9615185058210045`, improved `12`, regressed `0`, ties `0`
- KIVI S16 vs S0: paired_n `12`, median_delta `-1.0693700152332895`, improved `12`, regressed `0`, ties `0`

## Artifacts

- `static_vs_pseudo_metrics.csv`
- `accumulation_gap.csv`
- `accumulation_auc.csv`
- `task_level_summary.csv`
- `formal_completeness_audit.csv`
- `formal_run_summary.json`
