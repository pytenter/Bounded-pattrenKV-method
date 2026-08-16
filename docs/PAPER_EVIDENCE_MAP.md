# Paper Evidence Map

This file points future paper assembly to the canonical evidence sources for each claim.

| Claim area | Canonical source |
|---|---|
| Quality / generalization results | `reports/aime24_full_causal25_quality_4gpu/full_aime24_quality_report.md` and `reports/aime24_full_causal25_quality_4gpu/full_aime24_quality_summary.json` |
| AIME24 provenance and protocol | `reports/aime24_pseudodecode_3090_8gpu/formal_result_audit.md` and `reports/aime24_pseudodecode_3090_8gpu/pseudodecode_accumulation_report.md` |
| GSM8K evidence | `reports/gsm8k_smoke_1024_protocol_config.md`, `reports/gsm8k_official_kivi_vs_patternkv_50.md`, and `reports/gsm8k_smoke_512_vs_1024.md` |
| LongBench evidence | `reports/longbench_fp16_patternkv_kivi_8x50.md` and the LongBench benchmark reports under `reports/` |
| Effective KV budget | `reports/aime24_value_capacity_budget_3090/capacity_ceiling_report.md` and `reports/system_full_model_serving_benchmark_v1/final_serving_benchmark_freeze_v1/kv_runtime_provenance.md` |
| Algorithm freeze / frozen semantics | `reports/system_full_model_serving_benchmark_v1/final_serving_benchmark_freeze_v1/final_system_claims.md` and `reports/system_full_model_serving_benchmark_v1/final_serving_benchmark_freeze_v1/final_decision.md` |
| Ragged correctness | `reports/system_ragged_batch_decode_mvp_v1/final_recommendation.md` and `reports/system_ragged_decode1_semantic_gate_v1/final_gate.json` |
| Continuous batching | `reports/system_iteration_level_continuous_batching_v1/ragged_regression.txt` and `reports/system_request_lifecycle_manager_v1/ragged_regression.txt` |
| KV-runtime memory and capacity | `reports/system_full_model_serving_benchmark_v1/final_serving_benchmark_freeze_v1/memory_scaling_repaired.csv` and `reports/system_full_model_serving_benchmark_v1/final_serving_benchmark_freeze_v1/capacity.csv` |
| Full-model batch scaling | `reports/system_full_model_serving_benchmark_v1/final_serving_benchmark_freeze_v1/batch_scaling.csv` |
| Context scaling | `reports/system_full_model_serving_benchmark_v1/final_serving_benchmark_freeze_v1/context_scaling.csv` and `reports/system_full_model_serving_benchmark_v1/context_scaling.csv` |
| Capacity | `reports/system_full_model_serving_benchmark_v1/final_serving_benchmark_freeze_v1/capacity.md` and `reports/system_full_model_serving_benchmark_v1/final_serving_benchmark_freeze_v1/capacity.csv` |
| Long decode | `reports/system_full_model_serving_benchmark_v1/final_serving_benchmark_freeze_v1/long_decode.md` and `reports/system_full_model_serving_benchmark_v1/final_serving_benchmark_freeze_v1/long_decode.csv` |
| FP16 tail value fusion | `reports/system_full_model_serving_benchmark_v1/causal_attention_kernel_launch_forensic_v1/fp16_tail_decomposition.md` and `reports/system_full_model_serving_benchmark_v1/final_serving_benchmark_freeze_v1/final_system_claims.md` |
| Negative state-merge result | `reports/system_full_model_serving_benchmark_v1/segmented_heterogeneous_attention_state_merge_v1/summary.md` and `reports/segmented_heterogeneous_attention_state_merge_v1/current_path_audit.md` |
| Negative CUDA graph result | `reports/system_full_model_serving_benchmark_v1/causal_decode_cudagraph_replay_v1/summary.md` and `reports/system_full_model_serving_benchmark_v1/causal_decode_cudagraph_replay_v1/decision.md` |
| Final limitations | `reports/system_full_model_serving_benchmark_v1/final_serving_benchmark_freeze_v1/limitations.md` |
| Final-serving trace provenance | `reports/system_full_model_serving_benchmark_v1/causal_attention_kernel_launch_forensic_v1/summary.md` |

When a claim needs raw data, prefer the markdown/CSV summary above first, then the paired JSON or trace artifact in the same directory.
