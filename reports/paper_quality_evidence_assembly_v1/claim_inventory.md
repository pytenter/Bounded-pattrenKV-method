# Claim Inventory

| ID | Claim | Classification | Evidence |
| --- | --- | --- | --- |
| Q1 | CAUSAL-V4@25% preserves Long-CoT quality better than Pattern Base. | PRIMARY_SUPPORTED | AIME24 45/90 vs 32/90; CAUSAL-BASE bootstrap CI positive. |
| Q2 | CAUSAL-V4@25% preserves Long-CoT quality better than same-budget Random. | SUPPORTED_WITH_SCOPE | AIME24 aggregate 45/90 vs 36/90; bootstrap CI crosses zero. |
| Q3 | CAUSAL-V4@25% can match FP16 aggregate AIME24 accuracy in the tested protocol. | SUPPORTED_WITH_SCOPE | Both 45/90; no significance claim. |
| Q4 | CAUSAL improves GSM8K quality over PatternKV and KIVI. | SUPPORTED_WITH_SCOPE | Aggregate full split: +5.1554 pp vs Pattern, +10.0076 pp vs KIVI. |
| Q5 | CAUSAL improves LongBench average over PatternKV and KIVI while remaining close to FP16. | SUPPORTED_WITH_SCOPE | 21x50 8K macro: +0.8538 vs Pattern, +1.2514 vs KIVI, -0.8205 vs FP16. |
| Q6 | The 25% V4 budget is a useful quality/bit-efficiency operating point. | PARTIAL | Forensic budget curve shows saturation/utility at 25%; not a full task-quality sweep. |
| Q7 | Long-CoT quantization error accumulates recursively through persistent KV state. | SUPPLEMENTARY | Matched pseudo-decode accumulation supports this mechanism over tested AIME24 cohort. |
| Q8 | Value-path propagation dominates routing-only propagation in the tested forensic regime. | SUPPLEMENTARY | 6/6 tasks value-dominant; scoped to this regime. |
| Q9 | CAUSAL benefit is not merely same number of INT4 values randomly allocated. | SUPPORTED_WITH_SCOPE | AIME24 same-budget Random lags aggregate; CI for CAUSAL-Random crosses zero. |
| Q10 | CAUSAL operates at approximately ~2.500488 effective KV bits under project accounting. | PRIMARY_SUPPORTED | Bit accounting release and AIME24 bit_cost agree. |
