# Statistics Inventory

| Benchmark | Status | Available Statistics | Additional Offline Stats Needed |
| --- | --- | --- | --- |
| AIME24 | STATISTICS_COMPLETE_FOR_PRIMARY_CLAIMS | Question-level paired bootstrap for CAUSAL-Random and CAUSAL-Base. | None for current claims. |
| GSM8K | PARTIAL_WITH_NEW_OFFLINE_COUNTS | Baseline paired counts; CAUSAL-vs-baseline McNemar counts computed from committed raw outputs. | Predeclare if using p-values in paper. |
| LongBench | PARTIAL | Task-level aggregates and raw per-sample scores exist. | Optional bootstrap over tasks/samples if paper needs intervals. |
| Budget sweep | PARTIAL | Forensic bootstrap CIs for selector advantage in budget summary. | No task-quality budget sweep stats. |
| Random | STATISTICS_COMPLETE_FOR_AIME24_SCOPE | Same-budget AIME24 paired bootstrap vs CAUSAL. | None unless expanding to AIME25. |
