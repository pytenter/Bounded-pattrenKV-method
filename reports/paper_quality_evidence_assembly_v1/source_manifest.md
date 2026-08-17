# Source Manifest

| Evidence | Source Path | Status | Scope |
| --- | --- | --- | --- |
| AIME24 accuracy | reports/aime24_full_causal25_quality_4gpu/full_aime24_quality_summary.json | CANONICAL | DeepSeek-R1-Distill-Llama-8B; 30 questions x 3 seeds x 4 methods |
| AIME24 paired bootstrap | reports/aime24_full_causal25_quality_4gpu/paired_bootstrap.json | CANONICAL | Question-level bootstrap, 10000 resamples |
| AIME24 protocol | reports/aime24_full_causal25_quality_4gpu/generation_config.json | CANONICAL | DeepSeek-R1 prompt, temperature 0.6, top_p 0.95, max_new_tokens 32768 |
| AIME24 raw provenance | reports/aime24_full_causal25_quality_4gpu/raw_generation_manifest.json | REFERENCE | Raw generations are local/ignored; compact committed records exist |
| GSM8K baselines | reports/paper_repro_v2/gsm8k_full_2048/summary_4gpu_sequence.json | CANONICAL | Llama-3.1-8B-Instruct, full 1319-test split |
| GSM8K CAUSAL | results/causal_v4_25_generalization_v1/gsm8k_full/causal_v4_25/ | CANONICAL_RAW | Committed per-sample JSON, full 1319-test split |
| LongBench baselines | reports/paper_repro_v2/longbench_21x50_8k_4090/summary.json | CANONICAL_FINAL_BASELINE | 21 tasks x 50, 8K cap |
| LongBench CAUSAL | results/causal_v4_25_generalization_v1/longbench_full/causal_v4_25/ | CANONICAL_RAW | 21 tasks x 50, 8K cap |
| Budget/effective bits | reports/aime24_value_capacity_budget_3090/budget_response_summary.json | CANONICAL_FORENSIC | Budgets 0, 12.5%, 25%, 50%, 100% |
| Bit accounting | releases/causal_v4_25_aime24_v1/bit_accounting.json | CANONICAL | Formal payload-and-metadata project metric |
| Error accumulation | reports/aime24_pseudodecode_3090_8gpu/pseudodecode_accumulation_report.md | CANONICAL_FORENSIC | Matched checkpoints 128-4096 |
| Value-path forensic | reports/aime24_routing_vdirection_3090/routing_vdirection_summary.json | CANONICAL_FORENSIC | 6/6 tasks value-dominant |
| System evidence reference | reports/paper_system_table_and_figure_assembly_v1/ | FROZEN_REFERENCE_ONLY | No system regeneration in this task |
