# Claim Audit

| Claim | Status | Paper-Safe Wording |
| --- | --- | --- |
| AIME24 CAUSAL = FP16 aggregate | SUPPORTED_WITH_SCOPE | Matches FP16 aggregate accuracy in tested three-seed AIME24. |
| CAUSAL > Pattern Base AIME24 | SUPPORTED | Supported by canonical paired bootstrap CI. |
| CAUSAL > Random AIME24 | SUPPORTED_WITH_SCOPE | Aggregate advantage; do not call significant at 95%. |
| CAUSAL > FP16 GSM8K | PARTIAL | Aggregate numerical result only unless further stats are used. |
| CAUSAL > Pattern/KIVI GSM8K | SUPPORTED_WITH_SCOPE | Aggregate full-test result and offline paired counts support direction. |
| CAUSAL > Pattern/KIVI LongBench | SUPPORTED_WITH_SCOPE | Aggregate over tested 21x50 8K setup. |
| Value-path universally dominates | NOT_SUPPORTED | Only tested Long-CoT forensic regime supports dominance. |
| CAUSAL universally matches FP16 | NOT_SUPPORTED | AIME24 aggregate only; LongBench is below FP16. |
| 25% universally optimal | NOT_SUPPORTED | 25% is a supported operating point, not universal optimum. |
