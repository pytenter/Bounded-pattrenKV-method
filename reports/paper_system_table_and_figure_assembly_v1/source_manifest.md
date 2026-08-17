# Source Manifest

Primary system numbers are derived only from the reconciled canonical report. Superseded pre-reconciliation data in `reports/paper_baseline_system_comparison_v1/` is excluded from primary tables and figures.

## Canonical Inputs

- `reports/paper_baseline_system_comparison_v1_reconciled/final_gate.json`: `FINAL_PAPER_SYSTEM_NUMBERS_V1`, classification, allocator protocol.
- `reports/paper_baseline_system_comparison_v1_reconciled/paper_table.md`: rounded paper-facing cross-check of final system rows.
- `reports/paper_baseline_system_comparison_v1_reconciled/batch_scaling/`: C2048/B1 and C2048/B4 TPOT and throughput provenance.
- `reports/paper_baseline_system_comparison_v1_reconciled/context_scaling/`: B1/D8 context-scaling summary used by `system_context_scaling`.
- `reports/paper_baseline_system_comparison_v1_reconciled/capacity/`: capacity raw stop probes and capacity report provenance.
- `reports/paper_baseline_system_comparison_v1_reconciled/capacity_summary.csv`: max successful batch, first OOM batch, and capacity ratios.
- `reports/paper_baseline_system_comparison_v1_reconciled/long_decode/`: C4096/B1/D256 long-decode summary.
- `reports/paper_baseline_system_comparison_v1_reconciled/matched_memory_c4096_b4.csv`: matched C4096/B4 full-model peak allocated and reserved memory.

## Asset Mapping

- Main system table: `final_gate.json` plus `matched_memory_c4096_b4.csv`.
- Matched memory table and figure: `matched_memory_c4096_b4.csv`.
- Capacity table and figure: `final_gate.json` and `capacity_summary.csv`.
- Context-scaling table and figure: `context_scaling/context_scaling_summary.csv`.
- Long-decode table and figure: `final_gate.json` and `long_decode/long_decode_summary.csv`.
- Pairwise metrics and narrative: `FINAL_PAPER_SYSTEM_NUMBERS_V1` in `final_gate.json`.
