# Claim Audit

| Claim | Status | Reason |
| --- | --- | --- |
| PatternKV evaluated AMC24 | SUPPORTED | PatternKV Table 2 includes AMC24 columns. |
| PatternKV used eight responses per problem | SUPPORTED | Paper states eight independent responses per problem. |
| Avg@8 definition recovered | SUPPORTED | Paper defines per-sample accuracy averaged over eight responses. |
| Maj@8 high-level definition recovered | PARTIALLY_SUPPORTED | Paper defines majority voting across eight responses, but not tie policy. |
| PatternKV AMC citation resolves to NuminaMath | SUPPORTED | Li et al. 2024a bibliography entry is NuminaMath. |
| Exact PatternKV AMC24 rows recovered | NOT_SUPPORTED | No row-selection evidence found. |
| Exact PatternKV AMC24 ground truth recovered | NOT_SUPPORTED | NuminaMath-CoT has no separate answer field or AMC24 labels. |
| AMC24 parser can be implemented now | NOT_SUPPORTED | answer space unresolved. |
| AMC24 GPU run is ready | NOT_SUPPORTED | dataset and scoring identity unresolved. |
| CAUSAL generalizes to AMC24 | NOT_SUPPORTED | no valid AMC24 run exists. |

## Scientific Interpretation

The audit strengthens the citation chain but does not recover the benchmark identity. Running AMC24 now would require inventing a dataset/protocol from incomplete evidence.
