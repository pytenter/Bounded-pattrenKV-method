# Baseline Inventory

| Baseline | Classification | Paper Role |
| --- | --- | --- |
| FP16 | DIRECT_PRIMARY_BASELINE | Reference for AIME24/GSM8K/LongBench. |
| KIVI | DIRECT_PRIMARY_BASELINE | Canonical GSM8K/LongBench baseline; system baseline. |
| PatternKV / Pattern Base | DIRECT_PRIMARY_BASELINE | Compressed baseline and AIME24 base row. |
| Random-25% | DIRECT_PRIMARY_BASELINE | Same-budget AIME24 control. |
| ZipCache | MISSING | No current CAUSAL protocol result found. |
| SKVQ | MISSING | No current CAUSAL protocol result found. |
| OTT | MISSING | No current CAUSAL protocol result found. |
| KVQuant | MISSING | No current CAUSAL protocol result found. |
| AQUA-KV | MISSING | No current CAUSAL protocol result found. |
| KVarN | RELATED_HISTORICAL_ONLY | Do not count as CAUSAL evidence without current protocol participation. |

Baseline search hits sampled:

- `bench/__init__.py`
- `bench/aime24_int2_wave1.py`
- `bench/aime_utils.py`
- `bench/analyze_existing_pattern_results.py`
- `bench/bench_aime24_patternkv.py`
- `bench/bench_asymmetric_kv_final.py`
- `bench/bench_asymmetric_kv_serving.py`
- `bench/bench_capacity_integration.py`
- `bench/bench_centroid_ablation.py`
- `bench/bench_centroid_histogram_opt.py`
- `bench/bench_centroid_table_opt.py`
- `bench/bench_contiguous_capacity_cache.py`
- `bench/bench_gqa_kernel_redesign.py`
- `bench/bench_gsm8k_paper.py`
- `bench/bench_gsm8k_patternkv.py`
- `bench/bench_longbench_patternkv.py`
- `bench/bench_mixed_v_kernel_perf.py`
- `bench/bench_page_native_value_reader.py`
- `bench/bench_pattern_insight.py`
- `bench/bench_strided_capacity_reader.py`
- `bench/bench_strided_k_reader.py`
- `bench/bench_v2_kernel_candidates.py`
- `bench/deep_profile_mixed_v_cache.py`
- `bench/diagnose_k_stride_mechanism.py`
- `bench/full_model_serving_benchmark.py`
- `bench/gsm8k_paper_utils.py`
- `bench/longbench_config/__init__.py`
- `bench/paper_baseline_system_comparison.py`
- `bench/paper_config.py`
- `bench/patternkv_equivalence_reference.py`
- `bench/patternkv_page_batch_mvp.py`
- `bench/patternkv_paper_true_batch_runtime_support_audit.py`
- `bench/postopt_system_reprofile.py`
- `bench/prefill_v_trace_utils.py`
- `bench/profile_page_batch_operator.py`
- `bench/profile_post_fusion_decode.py`
- `bench/ragged_batch_decode_utils.py`
- `bench/reference_varn.py`
- `bench/run_actual_model_bi_prefill_runtime.py`
- `bench/run_actual_model_fixed_batch_smoke.py`
