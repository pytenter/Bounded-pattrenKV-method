# Insight V1 Repository Audit

branch: `insight/patternkv-diagnostics-v1`
HEAD: `c7324746d7447be532bd6bdbe0c8d58dd5e30c67`
standard_baseline_config: `configs/standard_baselines.paper_v2.yaml`
standard_baseline_config_hash: `6e86c85cb82116d4c6403c0977d103ff73ac8978e8a3b02f44f17e1db3f40e21`

Canonical methods:

```text
fp16
kivi_paper_g128
patternkv_paper
```

Non-canonical legacy method names:

```text
kivi
kivi_official
kivi_original_g32
patternkv
```

git status --short:

```text
A  bench/analyze_existing_pattern_results.py
A  bench/bench_pattern_insight.py
A  bench/summarize_pattern_insight.py
A  insight/__init__.py
A  insight/attention_metrics.py
A  insight/collector.py
A  insight/config.py
A  insight/dynamic_metrics.py
A  insight/gate_metrics.py
AM insight/io.py
A  insight/oracle_metrics.py
A  insight/pattern_metrics.py
A  insight/quant_reference.py
M  models/llama_patternkv.py
A  reports/insight_v1/final/attention_error.csv
A  reports/insight_v1/final/decision_matrix.md
A  reports/insight_v1/final/dynamic_pattern_utility.csv
A  reports/insight_v1/final/implementation_report.json
A  reports/insight_v1/final/implementation_report.md
A  reports/insight_v1/final/matching_oracle_gap.csv
A  reports/insight_v1/final/pattern_gain_map.csv
A  reports/insight_v1/final/summary.md
A  reports/insight_v1/final/v_gate_confusion.csv
A  reports/insight_v1/observer_overhead.md
A  reports/insight_v1/v0/baseline_integrity.md
A  reports/insight_v1/v0/gsm8k_outcome_groups.csv
A  reports/insight_v1/v0/gsm8k_paired.csv
A  reports/insight_v1/v0/length_analysis.csv
A  reports/insight_v1/v0/longbench_paired.csv
A  reports/insight_v1/v0/longbench_task_summary.csv
A  reports/insight_v1/v0/repository_audit.md
A  reports/insight_v1/v0/selected_samples.json
A  reports/insight_v1/v0/selected_samples.md
M  scripts/run_longbench_paper_8k_single4090.py
A  tests/test_insight_collector.py
A  tests/test_insight_config.py
A  tests/test_insight_dynamic_pattern.py
A  tests/test_insight_gain_metrics.py
A  tests/test_insight_matching_oracle.py
A  tests/test_insight_quant_reference.py
A  tests/test_insight_result_pairing.py
A  tests/test_insight_resume.py
A  tests/test_insight_v_gate.py
M  tests/test_longbench_single_gpu_runner.py
?? reports/paper_repro_v2/aime24/smoke_deepseek_report.json
?? reports/paper_repro_v2/aime24/smoke_deepseek_report.md
```

git log -8 --oneline:

```text
c732474 document standard paper baseline methods
b406875 Merge 4090 LongBench 21x50 8K results
2fc9666 Add 4090 LongBench 21x50 8K results
8df9b0d add GSM8K full reproduction results
dbe3827 feat: add GSM8K paper reproduction workflow
044be34 fix: support Llama GQA in official KIVI backend
4c6ac3b feat: add resource-bounded PatternKV AIME24 reproduction
ddd5a00 fix: align PatternKV reproduction with paper v2 settings
```

Conclusion: standard baseline semantics match `configs/standard_baselines.paper_v2.yaml`.
