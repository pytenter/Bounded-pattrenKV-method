# AIME24 Norm-Tail Accumulation Diagnostic

## Executive Summary

`TOKEN_NORM_ACCUMULATION_CLASSIFICATION=STRONG` and `TOKEN_NORM_ACCUMULATION_SUPPORTED=True`.
`VARN_MECHANISM_GATE=True`. Stage B still requires a canonical VarN source audit.

## Observer Validity

`NORM_OBSERVER_NONINVASIVE=True`.
`FP16_REGION_SANITY_PASS=True`.

## Source K/V Magnitude Drift

Pattern norm accumulation supported: `True`.
KIVI norm accumulation supported: `True`.

## Sink16 Norm Effect

Pattern K P95: median_delta `-0.16853652182253426`, improved `12/12`.
Pattern K P99: median_delta `-0.25992325892875673`, improved `12/12`.
Pattern V P95: median_delta `-0.26289608653169116`, improved `12/12`.
Pattern V P99: median_delta `-0.3771148046766757`, improved `12/12`.
KIVI K P95: median_delta `-0.18621445605967885`, improved `12/12`.
KIVI V P95: median_delta `-0.2839301186439115`, improved `12/12`.

## Correlation With Existing Accumulation

k_source p95 vs hidden_relative_L2: `0.8955076048053376`; k_source p99 vs hidden_relative_L2: `0.905187552790049`; v_source p95 vs hidden_relative_L2: `0.893466751275425`; v_source p99 vs hidden_relative_L2: `0.9059705599716659`; k_source p95 vs attention_output_relative_L2: `0.8955076048053376`; k_source p99 vs attention_output_relative_L2: `0.905187552790049`; v_source p95 vs attention_output_relative_L2: `0.893466751275425`; v_source p99 vs attention_output_relative_L2: `0.9059705599716659`

## Scientific Decision

The norm-tail diagnostic is read as association and mechanism consistency, not causal proof.

## Required Questions

1. Source K/V magnitude drift grows under pseudo decode: `True`.
2. Static norm distortion is the matched clean-path control and is lower than pseudo on the primary supported criteria.
3. Pseudo norm-tail error exceeds static on the primary criteria: `True`.
4. Norm drift increases across the core checkpoint AUC: `True`.
5. Sink16 reduces Pattern norm accumulation: `True`.
6. The same sink/norm pattern appears in KIVI: `True`.
7. Norm accumulation is positively associated with hidden/attention accumulation: `True`.
8. `TOKEN_NORM_ACCUMULATION_SUPPORTED=True`.
9. VarN is mechanistically justified by Stage A: `True`; execution still requires the separate canonical source gate.

## Artifact Storage

The full raw CSV artifacts remain available locally as `norm_tail_metrics.csv` and `norm_tail_accumulation_gap.csv`. Because each raw CSV exceeds GitHub's practical single-file limit, the versioned copies are `norm_tail_metrics.csv.gz` and `norm_tail_accumulation_gap.csv.gz`.
