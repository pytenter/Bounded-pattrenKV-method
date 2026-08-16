# Superseded Results

The earlier `reports/paper_baseline_system_comparison_v1/` CAUSAL rows are retained for provenance but are superseded for the primary paper system table: C2048/B1 ~274.967 ms/token, C2048/B4 ~315.989 ms/token, C4096/B1 ~359.284 ms/token, C4096/B1/D256 ~372.118 ms/token, and C4096 capacity B4. They were not reproduced after `CAUSAL_FROZEN_VS_RESUMED_PROVENANCE_RECONCILIATION_V1`; the capacity loss was traced to allocator protocol drift.
