# Final Experiment Gap Audit

| Priority | Gap | Status | Reason |
| --- | --- | --- | --- |
| P0 | AIME24 selector component ablation: Importance-only and Error-reduction-only | MISSING | Random and CAUSAL exist; separate components do not. |
| P0 | AIME25 Llama four-method quality | MISSING | No smoke, partial, or full canonical AIME25 results found. |
| P0 | Second-backbone CAUSAL quality evidence | MISSING | No Qwen/second-backbone CAUSAL quality run found. |
| P1 | Full official-split LongBench | MISSING_BUT_OPTIONAL | Current evidence is 21x50 with 8K cap, not full official split. |
| P1 | AMC24 | MISSING | Useful long-reasoning external validation after P0. |
| P1 | AIME24 Avg@8/Maj@8 | MISSING | Would improve sampling robustness but current AIME24 is already canonical aggregate. |
| P2 | AMC23/GPQA/MATH-like | OPTIONAL | No current CAUSAL evidence; add only if narrative needs broader reasoning. |
| STOP | System optimization or system reruns | DO_NOT_RUN | System track is frozen. |
