# Optimization Scorecard

Post-histogram table fraction: V2 16K `33.18%`, V2 32K `34.02%`.

GQA duplicate load: `YES`, theoretical duplicate factor `4x`.

| Candidate | Table Component | V2@32K | Mixed-V@32K | TPOT@32K | Correctness | Decision |
|---|---:|---:|---:|---:|---|---|
| A active centroid skip | active count 16/16, no useful rows to skip | NOT_IMPLEMENTED | NOT_IMPLEMENTED | NOT_RUN | N/A | NOT_IMPLEMENTED |
| B lane0 table contribution | removes redundant non-lane0 table contribution | 448.512 -> 307.200 us (1.460x) | 1072.128 -> 1017.856 us (1.053x) | 115.434 -> 115.335 ms (1.001x) | PASS | WIN, retained |
| C GQA table reuse | real 4x table reuse opportunity, broader CTA redesign | NOT_IMPLEMENTED | NOT_IMPLEMENTED | NOT_RUN | N/A | DEFERRED |
