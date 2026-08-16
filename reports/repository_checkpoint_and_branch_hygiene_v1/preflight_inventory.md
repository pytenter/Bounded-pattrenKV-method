## pwd

```text
/data/zypan/Bounded-pattrenKV-pseudodecode-3090
```

## branch

```text
sys/causal-v4-25-kernel-v1
```

## head

```text
cc50fdc513181d2137438cc6a7c0dd8322ccf767
```

## status_short

```text
 M bench/run_actual_model_bi_prefill_runtime.py
 M bench/run_bi_vproj_cost_benefit.py
 M bench/run_prefill_projection_mode_policy.py
 M bench/run_ragged_multistep_correctness.py
 M models/llama_patternkv.py
 M models/segmented_cache.py
 M quant/batch_invariant_kproj.py
 M quant/csrc/gemv_cuda.cu
 M quant/csrc/gemv_cuda.h
 M quant/csrc/pybind.cpp
 M quant/matmul.py
 M quant/page_batch.py
 M reports/system_ragged_multistep_correctness_v1/b2_16step.md
 M reports/system_ragged_multistep_correctness_v1/b2_flush_events.json
 M reports/system_ragged_multistep_correctness_v1/b2_flush_schedule.md
 M reports/system_ragged_multistep_correctness_v1/b2_reorder.md
 M reports/system_ragged_multistep_correctness_v1/b2_reorder_steps.json
 M reports/system_ragged_multistep_correctness_v1/b2_steps.json
 M reports/system_ragged_multistep_correctness_v1/b4_16step.md
 M reports/system_ragged_multistep_correctness_v1/b4_flush_events.json
 M reports/system_ragged_multistep_correctness_v1/b4_flush_schedule.md
 M reports/system_ragged_multistep_correctness_v1/b4_steps.json
 M reports/system_ragged_multistep_correctness_v1/centroid_events.json
 M reports/system_ragged_multistep_correctness_v1/environment.md
 M reports/system_ragged_multistep_correctness_v1/final_gate.json
 M reports/system_ragged_multistep_correctness_v1/free_run.json
 M reports/system_ragged_multistep_correctness_v1/page_events.json
 M reports/system_ragged_multistep_correctness_v1/pytest.md
 M reports/system_ragged_multistep_correctness_v1/runtime_counters.json
 M reports/system_ragged_multistep_correctness_v1/semantic_metrics.json
 M tests/test_bi_kproj_prefill_runtime.py
 M tests/test_bi_mlp_oracle.py
 M tests/test_fused_page_batch_operator.py
 M tests/test_ragged_cache_assembly.py
 M tests/test_ragged_k_valid_lengths.py
 M tests/test_value_direction_screen.py
?? bench/full_model_serving_benchmark.py
?? bench/serving_benchmark_v1.py
?? forensics/
?? models/request_lifecycle.py
?? reports/centroid_determinism_causal_forensic.md
?? reports/system_attention_qk_online_softmax_forensic_v1/
?? reports/system_attention_value_reduction_forensic_v1/
?? reports/system_b4_request_count_kernel_geometry_fix_v1/
?? reports/system_b4_request_count_ragged_divergence_v1/
?? reports/system_bi_kproj_ragged_decode_fix_v1/
?? reports/system_dynamic_add_remove_batching_v1/
?? reports/system_first_late_step_persistent_divergence_v1/
?? reports/system_full_decode_batch_invariance_oracle_v1/
?? reports/system_full_model_serving_benchmark_v1/
?? reports/system_full_model_serving_benchmark_v1_probe_256/
?? reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/
?? reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/
?? reports/system_full_model_serving_benchmark_v1_smoke/
?? reports/system_full_model_serving_benchmark_v1_smoke_causal/
?? reports/system_iteration_level_continuous_batching_v1/
?? reports/system_late_step_post_attention_rmsnorm_v1/
?? reports/system_ragged_active_state_forensic_v1/
?? reports/system_recent_k_ownership_forensic_v1/
?? reports/system_request_invariant_attention_softmax_fix_v1/
?? reports/system_request_invariant_attention_value_fix_v1/
?? reports/system_request_lifecycle_manager_v1/
?? reports/system_secondary_mlp_batch_invariance_v1/
?? reports/system_serving_benchmark_v1/
?? reports/system_step1_layer0_kpath_forensic_v1/
?? reports/system_v_causal_importance_forensic_v1/
?? reports/system_v_causal_importance_ragged_mapping_fix_v1/
?? scripts/attention_qk_online_softmax_forensic.py
?? scripts/attention_value_reduction_forensic.py
?? scripts/b4_attention_microtrace.py
?? scripts/b4_request_count_ragged_divergence.py
?? scripts/centroid_determinism_causal_forensic.py
?? scripts/first_late_step_persistent_divergence.py
?? scripts/full_decode_batch_invariance_oracle.py
?? scripts/full_model_post_scaling_bottleneck_forensic.py
?? scripts/full_model_scaling_decode_only_protocol_repair.py
?? scripts/late_step_post_attention_rmsnorm_gate.py
?? scripts/ragged_active_state_forensic.py
?? scripts/recent_k_ownership_forensic.py
?? scripts/reconcile_scaling_path_attention_roofline.py
?? scripts/request_invariant_attention_softmax_fix_gate.py
?? scripts/secondary_mlp_batch_invariance_gate.py
?? scripts/step1_layer0_k_path_forensic.py
?? scripts/v_causal_importance_forensic.py
?? tests/test_b4_request_count_ragged_divergence.py
?? tests/test_dynamic_add_remove_batching.py
?? tests/test_first_late_step_persistent_divergence.py
?? tests/test_full_model_post_scaling_bottleneck_forensic.py
?? tests/test_full_model_scaling_decode_only_protocol_repair.py
?? tests/test_full_model_serving_benchmark.py
?? tests/test_iteration_level_continuous_batching.py
?? tests/test_request_invariant_rmsnorm.py
?? tests/test_request_lifecycle_manager.py
?? tests/test_selective_prefill_logits_projection.py
?? tests/test_serving_benchmark_harness.py
```

## status_branch

```text
On branch sys/causal-v4-25-kernel-v1
Your branch is up to date with 'bounded/sys/causal-v4-25-kernel-v1'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   bench/run_actual_model_bi_prefill_runtime.py
	modified:   bench/run_bi_vproj_cost_benefit.py
	modified:   bench/run_prefill_projection_mode_policy.py
	modified:   bench/run_ragged_multistep_correctness.py
	modified:   models/llama_patternkv.py
	modified:   models/segmented_cache.py
	modified:   quant/batch_invariant_kproj.py
	modified:   quant/csrc/gemv_cuda.cu
	modified:   quant/csrc/gemv_cuda.h
	modified:   quant/csrc/pybind.cpp
	modified:   quant/matmul.py
	modified:   quant/page_batch.py
	modified:   reports/system_ragged_multistep_correctness_v1/b2_16step.md
	modified:   reports/system_ragged_multistep_correctness_v1/b2_flush_events.json
	modified:   reports/system_ragged_multistep_correctness_v1/b2_flush_schedule.md
	modified:   reports/system_ragged_multistep_correctness_v1/b2_reorder.md
	modified:   reports/system_ragged_multistep_correctness_v1/b2_reorder_steps.json
	modified:   reports/system_ragged_multistep_correctness_v1/b2_steps.json
	modified:   reports/system_ragged_multistep_correctness_v1/b4_16step.md
	modified:   reports/system_ragged_multistep_correctness_v1/b4_flush_events.json
	modified:   reports/system_ragged_multistep_correctness_v1/b4_flush_schedule.md
	modified:   reports/system_ragged_multistep_correctness_v1/b4_steps.json
	modified:   reports/system_ragged_multistep_correctness_v1/centroid_events.json
	modified:   reports/system_ragged_multistep_correctness_v1/environment.md
	modified:   reports/system_ragged_multistep_correctness_v1/final_gate.json
	modified:   reports/system_ragged_multistep_correctness_v1/free_run.json
	modified:   reports/system_ragged_multistep_correctness_v1/page_events.json
	modified:   reports/system_ragged_multistep_correctness_v1/pytest.md
	modified:   reports/system_ragged_multistep_correctness_v1/runtime_counters.json
	modified:   reports/system_ragged_multistep_correctness_v1/semantic_metrics.json
	modified:   tests/test_bi_kproj_prefill_runtime.py
	modified:   tests/test_bi_mlp_oracle.py
	modified:   tests/test_fused_page_batch_operator.py
	modified:   tests/test_ragged_cache_assembly.py
	modified:   tests/test_ragged_k_valid_lengths.py
	modified:   tests/test_value_direction_screen.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	bench/full_model_serving_benchmark.py
	bench/serving_benchmark_v1.py
	forensics/
	models/request_lifecycle.py
	reports/centroid_determinism_causal_forensic.md
	reports/system_attention_qk_online_softmax_forensic_v1/
	reports/system_attention_value_reduction_forensic_v1/
	reports/system_b4_request_count_kernel_geometry_fix_v1/
	reports/system_b4_request_count_ragged_divergence_v1/
	reports/system_bi_kproj_ragged_decode_fix_v1/
	reports/system_dynamic_add_remove_batching_v1/
	reports/system_first_late_step_persistent_divergence_v1/
	reports/system_full_decode_batch_invariance_oracle_v1/
	reports/system_full_model_serving_benchmark_v1/
	reports/system_full_model_serving_benchmark_v1_probe_256/
	reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/
	reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/
	reports/system_full_model_serving_benchmark_v1_smoke/
	reports/system_full_model_serving_benchmark_v1_smoke_causal/
	reports/system_iteration_level_continuous_batching_v1/
	reports/system_late_step_post_attention_rmsnorm_v1/
	reports/system_ragged_active_state_forensic_v1/
	reports/system_recent_k_ownership_forensic_v1/
	reports/system_request_invariant_attention_softmax_fix_v1/
	reports/system_request_invariant_attention_value_fix_v1/
	reports/system_request_lifecycle_manager_v1/
	reports/system_secondary_mlp_batch_invariance_v1/
	reports/system_serving_benchmark_v1/
	reports/system_step1_layer0_kpath_forensic_v1/
	reports/system_v_causal_importance_forensic_v1/
	reports/system_v_causal_importance_ragged_mapping_fix_v1/
	scripts/attention_qk_online_softmax_forensic.py
	scripts/attention_value_reduction_forensic.py
	scripts/b4_attention_microtrace.py
	scripts/b4_request_count_ragged_divergence.py
	scripts/centroid_determinism_causal_forensic.py
	scripts/first_late_step_persistent_divergence.py
	scripts/full_decode_batch_invariance_oracle.py
	scripts/full_model_post_scaling_bottleneck_forensic.py
	scripts/full_model_scaling_decode_only_protocol_repair.py
	scripts/late_step_post_attention_rmsnorm_gate.py
	scripts/ragged_active_state_forensic.py
	scripts/recent_k_ownership_forensic.py
	scripts/reconcile_scaling_path_attention_roofline.py
	scripts/request_invariant_attention_softmax_fix_gate.py
	scripts/secondary_mlp_batch_invariance_gate.py
	scripts/step1_layer0_k_path_forensic.py
	scripts/v_causal_importance_forensic.py
	tests/test_b4_request_count_ragged_divergence.py
	tests/test_dynamic_add_remove_batching.py
	tests/test_first_late_step_persistent_divergence.py
	tests/test_full_model_post_scaling_bottleneck_forensic.py
	tests/test_full_model_scaling_decode_only_protocol_repair.py
	tests/test_full_model_serving_benchmark.py
	tests/test_iteration_level_continuous_batching.py
	tests/test_request_invariant_rmsnorm.py
	tests/test_request_lifecycle_manager.py
	tests/test_selective_prefill_logits_projection.py
	tests/test_serving_benchmark_harness.py

no changes added to commit (use "git add" and/or "git commit -a")
```

## remote_v

```text
bounded	git@github.com:pytenter/Bounded-pattrenKV-method.git (fetch)
bounded	git@github.com:pytenter/Bounded-pattrenKV-method.git (push)
origin	https://github.com/HCOOOH/PatternKV.git (fetch)
origin	https://github.com/HCOOOH/PatternKV.git (push)
```

## log_20

```text
cc50fdc (HEAD -> sys/causal-v4-25-kernel-v1, bounded/sys/causal-v4-25-kernel-v1) test: diagnose PatternKV ragged multistep drift
7099633 fix: preserve PatternKV ragged multistep state
1f418f9 test: record request lifecycle prerequisite block
6b9f32d fix: preserve PatternKV ragged decode semantics
ea34144 feat: support PatternKV ragged K valid lengths
1544904 (repair/split-generalization-from-system-20260814) docs: record generalization and system branch split
b1c0ee0 Revert "Add CAUSAL V4 25 generalization results"
3dcedb4 (bounded/exp/causal-v4-25-generalization-v1) Add CAUSAL V4 25 generalization results
3a9fa06 feat: assemble PatternKV ragged request caches
c66676c feat: add PatternKV ragged batch decode MVP audit
27d9982 test: close final PatternKV fixed-batch semantic gate
9c86f32 test: validate batch-invariant MLP causal oracle
57716a5 test: trace P2 production first divergence
82d43a1 feat: define PatternKV prefill projection mode policy
b7f7103 Evaluate BI V prefill projection mode
9b49829 test: audit PatternKV V centroid semantic impact
17fad41 test: trace actual-model prefill V centroid divergence
7cada5a feat: integrate batch-invariant K projection into PatternKV prefill
ca0463c perf: optimize batch-invariant K projection with persistent Triton GEMM
9616629 feat: add batch-invariant PatternKV K projection prototype
```

## diff_stat

```text
 bench/run_actual_model_bi_prefill_runtime.py       |    5 +-
 bench/run_bi_vproj_cost_benefit.py                 |    8 +-
 bench/run_prefill_projection_mode_policy.py        |   12 +-
 bench/run_ragged_multistep_correctness.py          |   68 +-
 models/llama_patternkv.py                          |  335 +-
 models/segmented_cache.py                          |  346 +-
 quant/batch_invariant_kproj.py                     |   10 +-
 quant/csrc/gemv_cuda.cu                            |  196 +-
 quant/csrc/gemv_cuda.h                             |   15 +
 quant/csrc/pybind.cpp                              |    1 +
 quant/matmul.py                                    |   31 +
 quant/page_batch.py                                |   54 +-
 .../b2_16step.md                                   |   16 +-
 .../b2_flush_events.json                           |   33 +-
 .../b2_flush_schedule.md                           |    5 +-
 .../b2_reorder.md                                  |   16 +-
 .../b2_reorder_steps.json                          | 1172 +++-
 .../b2_steps.json                                  | 1172 +++-
 .../b4_16step.md                                   |   16 +-
 .../b4_flush_events.json                           |   63 +-
 .../b4_flush_schedule.md                           |    7 +-
 .../b4_steps.json                                  | 2244 ++++++-
 .../centroid_events.json                           |   96 +-
 .../environment.md                                 |   22 +-
 .../final_gate.json                                |   43 +-
 .../free_run.json                                  | 1734 +-----
 .../page_events.json                               |   96 +-
 .../pytest.md                                      |    8 +-
 .../runtime_counters.json                          |   86 +-
 .../semantic_metrics.json                          | 6280 ++++++++++++++------
 tests/test_bi_kproj_prefill_runtime.py             |   23 +-
 tests/test_bi_mlp_oracle.py                        |   85 +-
 tests/test_fused_page_batch_operator.py            |   30 +-
 tests/test_ragged_cache_assembly.py                |   42 +
 tests/test_ragged_k_valid_lengths.py               |  341 ++
 tests/test_value_direction_screen.py               |   11 +
 36 files changed, 10768 insertions(+), 3954 deletions(-)
```

## diff_name_status

```text
M	bench/run_actual_model_bi_prefill_runtime.py
M	bench/run_bi_vproj_cost_benefit.py
M	bench/run_prefill_projection_mode_policy.py
M	bench/run_ragged_multistep_correctness.py
M	models/llama_patternkv.py
M	models/segmented_cache.py
M	quant/batch_invariant_kproj.py
M	quant/csrc/gemv_cuda.cu
M	quant/csrc/gemv_cuda.h
M	quant/csrc/pybind.cpp
M	quant/matmul.py
M	quant/page_batch.py
M	reports/system_ragged_multistep_correctness_v1/b2_16step.md
M	reports/system_ragged_multistep_correctness_v1/b2_flush_events.json
M	reports/system_ragged_multistep_correctness_v1/b2_flush_schedule.md
M	reports/system_ragged_multistep_correctness_v1/b2_reorder.md
M	reports/system_ragged_multistep_correctness_v1/b2_reorder_steps.json
M	reports/system_ragged_multistep_correctness_v1/b2_steps.json
M	reports/system_ragged_multistep_correctness_v1/b4_16step.md
M	reports/system_ragged_multistep_correctness_v1/b4_flush_events.json
M	reports/system_ragged_multistep_correctness_v1/b4_flush_schedule.md
M	reports/system_ragged_multistep_correctness_v1/b4_steps.json
M	reports/system_ragged_multistep_correctness_v1/centroid_events.json
M	reports/system_ragged_multistep_correctness_v1/environment.md
M	reports/system_ragged_multistep_correctness_v1/final_gate.json
M	reports/system_ragged_multistep_correctness_v1/free_run.json
M	reports/system_ragged_multistep_correctness_v1/page_events.json
M	reports/system_ragged_multistep_correctness_v1/pytest.md
M	reports/system_ragged_multistep_correctness_v1/runtime_counters.json
M	reports/system_ragged_multistep_correctness_v1/semantic_metrics.json
M	tests/test_bi_kproj_prefill_runtime.py
M	tests/test_bi_mlp_oracle.py
M	tests/test_fused_page_batch_operator.py
M	tests/test_ragged_cache_assembly.py
M	tests/test_ragged_k_valid_lengths.py
M	tests/test_value_direction_screen.py
```

## diff_check

```text

```

## others

```text
bench/full_model_serving_benchmark.py
bench/serving_benchmark_v1.py
forensics/centroid_determinism/fixed_layer0_k.pt
forensics/centroid_determinism/payload.json
forensics/step1_layer0_k_path/fixed_layer0_step1_norm_A.pt
models/request_lifecycle.py
reports/centroid_determinism_causal_forensic.md
reports/system_attention_qk_online_softmax_forensic_v1/attention_probability_call_graph.md
reports/system_attention_qk_online_softmax_forensic_v1/attention_split_planner.md
reports/system_attention_qk_online_softmax_forensic_v1/dynamic_split_boundaries.json
reports/system_attention_qk_online_softmax_forensic_v1/environment.md
reports/system_attention_qk_online_softmax_forensic_v1/final_gate.json
reports/system_attention_qk_online_softmax_forensic_v1/final_probability_reconstruction.json
reports/system_attention_qk_online_softmax_forensic_v1/fixed_split_b2_16step.json
reports/system_attention_qk_online_softmax_forensic_v1/fixed_split_boundary_oracle.json
reports/system_attention_qk_online_softmax_forensic_v1/fixed_split_layer_propagation.json
reports/system_attention_qk_online_softmax_forensic_v1/fixed_split_merge_state_oracle.json
reports/system_attention_qk_online_softmax_forensic_v1/fixed_split_probability_oracle.json
reports/system_attention_qk_online_softmax_forensic_v1/fixed_split_size_design.md
reports/system_attention_qk_online_softmax_forensic_v1/mask_semantics.md
reports/system_attention_qk_online_softmax_forensic_v1/mask_value_audit.json
reports/system_attention_qk_online_softmax_forensic_v1/masked_valid_logits_comparison.json
reports/system_attention_qk_online_softmax_forensic_v1/merge_state_audit.md
reports/system_attention_qk_online_softmax_forensic_v1/merged_lse_comparison.json
reports/system_attention_qk_online_softmax_forensic_v1/no_split_attention_oracle.json
reports/system_attention_qk_online_softmax_forensic_v1/no_split_attention_oracle.md
reports/system_attention_qk_online_softmax_forensic_v1/online_softmax_state_audit.md
reports/system_attention_qk_online_softmax_forensic_v1/per_split_lse_comparison.json
reports/system_attention_qk_online_softmax_forensic_v1/per_split_max_comparison.json
reports/system_attention_qk_online_softmax_forensic_v1/preflight.json
reports/system_attention_qk_online_softmax_forensic_v1/qk_dot_reduction_oracle.json
reports/system_attention_qk_online_softmax_forensic_v1/qk_input_comparison.json
reports/system_attention_qk_online_softmax_forensic_v1/raw_qk_logits_comparison.json
reports/system_attention_qk_online_softmax_forensic_v1/root_cause_evidence.md
reports/system_attention_qk_online_softmax_forensic_v1/scaled_logits_comparison.json
reports/system_attention_qk_online_softmax_forensic_v1/secondary_value_reduction_after_softmax_fix.json
reports/system_attention_value_reduction_forensic_v1/atomic_audit.md
reports/system_attention_value_reduction_forensic_v1/attention_probs_comparison.json
reports/system_attention_value_reduction_forensic_v1/attention_value_call_graph.md
reports/system_attention_value_reduction_forensic_v1/b2_16step_fixed_reduction.json
reports/system_attention_value_reduction_forensic_v1/b2_16step_fixed_reduction.md
reports/system_attention_value_reduction_forensic_v1/canonical_attention_contract.md
reports/system_attention_value_reduction_forensic_v1/effective_value_comparison.json
reports/system_attention_value_reduction_forensic_v1/environment.md
reports/system_attention_value_reduction_forensic_v1/final_gate.json
reports/system_attention_value_reduction_forensic_v1/fixed_reduction_mode.md
reports/system_attention_value_reduction_forensic_v1/fixed_reduction_oracle.json
reports/system_attention_value_reduction_forensic_v1/frozen_pv_production_oracle.json
reports/system_attention_value_reduction_forensic_v1/golden_value_reduction.json
reports/system_attention_value_reduction_forensic_v1/per_token_pv_contribution.json
reports/system_attention_value_reduction_forensic_v1/preflight.json
reports/system_attention_value_reduction_forensic_v1/production_reduction_topology_b1.json
reports/system_attention_value_reduction_forensic_v1/production_reduction_topology_ragged.json
reports/system_attention_value_reduction_forensic_v1/reduction_topology_audit.md
reports/system_attention_value_reduction_forensic_v1/root_cause_evidence.md
reports/system_attention_value_reduction_forensic_v1/secondary_divergence_after_fixed_reduction.json
reports/system_attention_value_reduction_forensic_v1/step1_layer0_fixed_reduction.json
reports/system_attention_value_reduction_forensic_v1/step1_layer1_propagation.json
reports/system_attention_value_reduction_forensic_v1/v_precision_mapping_comparison.json
reports/system_b4_request_count_kernel_geometry_fix_v1/REPORT.md
reports/system_b4_request_count_kernel_geometry_fix_v1/attention_microtrace_b_request_step1_layer0.json
reports/system_b4_request_count_kernel_geometry_fix_v1/attention_microtrace_summary.md
reports/system_b4_request_count_kernel_geometry_fix_v1/b1b_vs_b4b_step6_layer_trace.json
reports/system_b4_request_count_ragged_divergence_v1/b2_known_good_control.json
reports/system_b4_request_count_ragged_divergence_v1/b2_regression_postfix.json
reports/system_b4_request_count_ragged_divergence_v1/b4_16step_postfix.json
reports/system_b4_request_count_ragged_divergence_v1/b4_16step_postfix.md
reports/system_b4_request_count_ragged_divergence_v1/b4_initial_failure.json
reports/system_b4_request_count_ragged_divergence_v1/b4_runtime_geometry_timeline.json
reports/system_b4_request_count_ragged_divergence_v1/b4_runtime_geometry_timeline.md
reports/system_b4_request_count_ragged_divergence_v1/environment.md
reports/system_b4_request_count_ragged_divergence_v1/final_gate.json
reports/system_b4_request_count_ragged_divergence_v1/first_bad_b4_component.json
reports/system_b4_request_count_ragged_divergence_v1/first_bad_b4_layer.json
reports/system_b4_request_count_ragged_divergence_v1/first_bad_b4_step.json
reports/system_b4_request_count_ragged_divergence_v1/first_bad_b4_step_input_output.json
reports/system_b4_request_count_ragged_divergence_v1/independent_flush_postfix.json
reports/system_b4_request_count_ragged_divergence_v1/page_chunk_offset_audit.json
reports/system_b4_request_count_ragged_divergence_v1/peer_content_control.json
reports/system_b4_request_count_ragged_divergence_v1/peer_identity_control.json
reports/system_b4_request_count_ragged_divergence_v1/peer_length_control.json
reports/system_b4_request_count_ragged_divergence_v1/preflight.json
reports/system_b4_request_count_ragged_divergence_v1/prior_fix_state.md
reports/system_b4_request_count_ragged_divergence_v1/regression_summary.md
reports/system_b4_request_count_ragged_divergence_v1/request_b_b2_vs_b4_timeline.json
reports/system_b4_request_count_ragged_divergence_v1/request_b_b2_vs_b4_timeline.md
reports/system_b4_request_count_ragged_divergence_v1/request_count_ladder.json
reports/system_b4_request_count_ragged_divergence_v1/row_ownership_audit.md
reports/system_b4_request_count_ragged_divergence_v1/seq_len_mapping_audit.json
reports/system_b4_request_count_ragged_divergence_v1/softmax_value_split_audit.json
reports/system_b4_request_count_ragged_divergence_v1/system_invariants.json
reports/system_b4_request_count_ragged_divergence_v1/workspace_ownership_audit.json
reports/system_bi_kproj_ragged_decode_fix_v1/after_fix_dispatch.md
reports/system_bi_kproj_ragged_decode_fix_v1/b2_16step_postfix.json
reports/system_bi_kproj_ragged_decode_fix_v1/b2_16step_postfix.md
reports/system_bi_kproj_ragged_decode_fix_v1/b2_reorder_postfix.json
reports/system_bi_kproj_ragged_decode_fix_v1/b4_ragged_multistep_postfix.json
reports/system_bi_kproj_ragged_decode_fix_v1/b4_ragged_multistep_postfix.md
reports/system_bi_kproj_ragged_decode_fix_v1/before_fix_dispatch.md
reports/system_bi_kproj_ragged_decode_fix_v1/bi_kproj_contract_test.json
reports/system_bi_kproj_ragged_decode_fix_v1/current_k_postfix.json
reports/system_bi_kproj_ragged_decode_fix_v1/environment.md
reports/system_bi_kproj_ragged_decode_fix_v1/final_gate.json
reports/system_bi_kproj_ragged_decode_fix_v1/independent_flush_postfix.json
reports/system_bi_kproj_ragged_decode_fix_v1/preflight.json
reports/system_bi_kproj_ragged_decode_fix_v1/production_fix.md
reports/system_bi_kproj_ragged_decode_fix_v1/recent_k_postfix.json
reports/system_bi_kproj_ragged_decode_fix_v1/regression_summary.md
reports/system_bi_kproj_ragged_decode_fix_v1/root_cause_evidence.md
reports/system_bi_kproj_ragged_decode_fix_v1/step1_layer0_postfix.json
reports/system_bi_kproj_ragged_decode_fix_v1/system_invariants.json
reports/system_dynamic_add_remove_batching_v1/dynamic_sequence.json
reports/system_dynamic_add_remove_batching_v1/final_gate.json
reports/system_dynamic_add_remove_batching_v1/full_pytest.txt
reports/system_dynamic_add_remove_batching_v1/git_state.txt
reports/system_dynamic_add_remove_batching_v1/lifecycle_regression.txt
reports/system_dynamic_add_remove_batching_v1/ragged_regression.txt
reports/system_dynamic_add_remove_batching_v1/runtime_env.txt
reports/system_dynamic_add_remove_batching_v1/survivor_trajectory.json
reports/system_dynamic_add_remove_batching_v1/targeted_pytest.txt
reports/system_first_late_step_persistent_divergence_v1/b2_16step_postfix.json
reports/system_first_late_step_persistent_divergence_v1/b2_16step_postfix.md
reports/system_first_late_step_persistent_divergence_v1/b2_reorder_postfix.json
reports/system_first_late_step_persistent_divergence_v1/b2_stepwise_semantic_timeline.json
reports/system_first_late_step_persistent_divergence_v1/b2_stepwise_semantic_timeline.md
reports/system_first_late_step_persistent_divergence_v1/b2_transition_timeline.json
reports/system_first_late_step_persistent_divergence_v1/b2_transition_timeline.md
reports/system_first_late_step_persistent_divergence_v1/b4_postfix.json
reports/system_first_late_step_persistent_divergence_v1/environment.md
reports/system_first_late_step_persistent_divergence_v1/final_gate.json
reports/system_first_late_step_persistent_divergence_v1/first_bad_component.json
reports/system_first_late_step_persistent_divergence_v1/first_bad_layer.json
reports/system_first_late_step_persistent_divergence_v1/first_bad_step.json
reports/system_first_late_step_persistent_divergence_v1/first_bad_step_input_output.json
reports/system_first_late_step_persistent_divergence_v1/independent_flush_postfix.json
reports/system_first_late_step_persistent_divergence_v1/peer_content_control.json
reports/system_first_late_step_persistent_divergence_v1/peer_length_control.json
reports/system_first_late_step_persistent_divergence_v1/persistent_state_breakdown.json
reports/system_first_late_step_persistent_divergence_v1/preflight.json
reports/system_first_late_step_persistent_divergence_v1/production_fix_state.md
reports/system_first_late_step_persistent_divergence_v1/regression_summary.md
reports/system_first_late_step_persistent_divergence_v1/reorder_control.json
reports/system_first_late_step_persistent_divergence_v1/system_invariants.json
reports/system_full_decode_batch_invariance_oracle_v1/attention_batch_invariance_audit.md
reports/system_full_decode_batch_invariance_oracle_v1/attention_batch_shape_oracle.json
reports/system_full_decode_batch_invariance_oracle_v1/b2_16step_full_bi.json
reports/system_full_decode_batch_invariance_oracle_v1/b2_16step_full_bi.md
reports/system_full_decode_batch_invariance_oracle_v1/current_worktree_fix_state.md
reports/system_full_decode_batch_invariance_oracle_v1/decode_operator_bi_coverage_manifest.json
reports/system_full_decode_batch_invariance_oracle_v1/decode_operator_bi_coverage_manifest.md
reports/system_full_decode_batch_invariance_oracle_v1/environment.md
reports/system_full_decode_batch_invariance_oracle_v1/final_gate.json
reports/system_full_decode_batch_invariance_oracle_v1/full_bi_coverage.json
reports/system_full_decode_batch_invariance_oracle_v1/full_bi_mode_implementation.md
reports/system_full_decode_batch_invariance_oracle_v1/full_bi_secondary_divergence.json
reports/system_full_decode_batch_invariance_oracle_v1/layer1_recent_k_pretrace.json
reports/system_full_decode_batch_invariance_oracle_v1/linear_batch_invariance_matrix.json
reports/system_full_decode_batch_invariance_oracle_v1/preflight.json
reports/system_full_decode_batch_invariance_oracle_v1/rmsnorm_batch_shape_oracle.json
reports/system_full_decode_batch_invariance_oracle_v1/selective_bi_ablation.json
reports/system_full_decode_batch_invariance_oracle_v1/selective_bi_ablation.md
reports/system_full_decode_batch_invariance_oracle_v1/step1_layer0_full_bi_trace.json
reports/system_full_decode_batch_invariance_oracle_v1/step1_layer0_full_bi_trace.md
reports/system_full_decode_batch_invariance_oracle_v1/step1_layer1_full_bi_trace.json
reports/system_full_decode_batch_invariance_oracle_v1/system_invariants.json
reports/system_full_model_serving_benchmark_v1/baseline_audit.json
reports/system_full_model_serving_benchmark_v1/benchmark_config.json
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/cache_mutation_sites.csv
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/call_graph.md
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/causal_value_softmax_timing.csv
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/coarse_timing.csv
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/copy_allocation_sites.csv
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/copy_allocation_sites.json
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/final_profile_gate.json
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/gpu_state_before_profile.txt
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/layer_timing.csv
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/memory_breakdown.json
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/memory_checkpoints.csv
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/metadata_rebuild_audit.json
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/page_pool_audit.json
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/path_diff_isolated_vs_full_model.md
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/profile_config.json
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/profile_range_timing.csv
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/root_cause_report.md
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/row_slice_audit.json
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/runtime_counters.json
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/sync_sites.json
reports/system_full_model_serving_benchmark_v1/bottleneck_profile/temp_allocation_sites.csv
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/component_schema.md
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/current_heterogeneous_attention_dataflow.md
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/dequantization_path.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/exclusive_attention_breakdown.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/final_gate.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/kernel_launch_accounting.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/long_decode_boundary_probe.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/points/long_decode_boundary__causal_v4_25_full_model__c2048__b1__d136.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/points/profile__causal_v4_25_full_model__c2048__b4__d8.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/points/profile__causal_v4_25_full_model__c8192__b1__d8.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_c2048_b1.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_c2048_b2.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_c2048_b4.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_context_scaling.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_fp16_reference.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/cache_mutation_forensic.csv
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/cache_mutation_forensic.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/decode_component_profile.csv
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/decode_component_profile.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/environment.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/final_gate.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/memory_breakdown.csv
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/memory_breakdown.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/memory_lifecycle.csv
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/memory_lifecycle.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/oom_forensic.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/points/profile__causal_v4_25_full_model__c2048__b1__d8.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/points/profile__causal_v4_25_full_model__c2048__b2__d8.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/points/profile__causal_v4_25_full_model__c4096__b1__d8.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/points/profile__fp16_full_model__c2048__b1__d8.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/points/profile__fp16_full_model__c2048__b2__d8.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/points/profile__fp16_full_model__c4096__b1__d8.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/scaling_component_analysis.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/summary.md
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/temp_allocation_forensic.csv
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/temp_allocation_forensic.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/worker_summaries.csv
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/profile_matrix/worker_summaries.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/state_merge_feasibility.json
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/summary.md
reports/system_full_model_serving_benchmark_v1/causal_heterogeneous_attention_forensic_v1/temporary_allocation_accounting.json
reports/system_full_model_serving_benchmark_v1/causal_v4_25_full_model_concurrency_sweep.json
reports/system_full_model_serving_benchmark_v1/context_scaling.csv
reports/system_full_model_serving_benchmark_v1/context_scaling_comparison.csv
reports/system_full_model_serving_benchmark_v1/context_scaling_comparison.json
reports/system_full_model_serving_benchmark_v1/context_scaling_fresh_decode8_runs/causal_v4_25_full_model_ctx2048.json
reports/system_full_model_serving_benchmark_v1/context_scaling_fresh_decode8_runs/causal_v4_25_full_model_ctx256.json
reports/system_full_model_serving_benchmark_v1/context_scaling_fresh_decode8_runs/causal_v4_25_full_model_ctx4096.json
reports/system_full_model_serving_benchmark_v1/context_scaling_fresh_decode8_runs/fp16_full_model_ctx2048.json
reports/system_full_model_serving_benchmark_v1/context_scaling_fresh_decode8_runs/fp16_full_model_ctx256.json
reports/system_full_model_serving_benchmark_v1/context_scaling_fresh_decode8_runs/fp16_full_model_ctx4096.json
reports/system_full_model_serving_benchmark_v1/context_scaling_fresh_decode8_runs/fp16_full_model_ctx8192.json
reports/system_full_model_serving_benchmark_v1/context_scaling_fresh_runs/causal_v4_25_full_model_ctx256.json
reports/system_full_model_serving_benchmark_v1/context_scaling_probe.json
reports/system_full_model_serving_benchmark_v1/context_scaling_probe_report.md
reports/system_full_model_serving_benchmark_v1/context_scaling_probe_runs/causal_v4_25_full_model_ctx256.json
reports/system_full_model_serving_benchmark_v1/context_scaling_probe_runs/causal_v4_25_full_model_ctx4096.json
reports/system_full_model_serving_benchmark_v1/context_scaling_probe_runs/causal_v4_25_full_model_ctx8192.json
reports/system_full_model_serving_benchmark_v1/context_scaling_probe_runs/fp16_full_model_ctx16384.json
reports/system_full_model_serving_benchmark_v1/context_scaling_probe_runs/fp16_full_model_ctx256.json
reports/system_full_model_serving_benchmark_v1/context_scaling_probe_runs/fp16_full_model_ctx4096.json
reports/system_full_model_serving_benchmark_v1/context_scaling_probe_runs/fp16_full_model_ctx8192.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/context_scaling.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/copy_accounting.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/copy_accounting_after.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/copy_accounting_before.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/current_dataflow.md
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/final_gate.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/README.md
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/capacity_scaling.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/capacity_scaling.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/context_scaling.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/context_scaling.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/decode_only_b_scaling_raw.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/decode_only_b_scaling_raw.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/decode_only_b_scaling_summary.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/decode_only_b_scaling_summary.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/decode_only_capacity_raw.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/decode_only_capacity_raw.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/decode_only_capacity_summary.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/decode_only_capacity_summary.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/decode_only_context_scaling_raw.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/decode_only_context_scaling_raw.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/decode_only_context_scaling_summary.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/decode_only_context_scaling_summary.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/environment.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/exact_commands.md
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/final_gate.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/master_run_summaries.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/master_run_summaries.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/matched_b_scaling.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/matched_b_scaling.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/memory_scaling_repaired.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/memory_scaling_repaired.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/old_vs_new_comparison.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/old_vs_new_comparison.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/point_summaries.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/point_summaries.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/points/smoke__causal_v4_25_full_model__c2048__b1__d8__ac1__tr1__w0__m1.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/points/smoke__fp16_full_model__c2048__b1__d8__ac1__tr1__w0__m1.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/points/smoke_repeat__fp16_full_model__c2048__b1__d8__ac1__tr1__w0__m1.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/protocol_definition.md
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/protocol_validation.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/structural_counters.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/formal_smoke_no_optimization/summary.md
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/matched_b_scaling.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/performance_b1.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/points/capacity_check__causal_v4_25_full_model__c4096__b8__d8.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_after.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_before.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/cache_mutation_forensic.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/cache_mutation_forensic.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/decode_component_profile.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/decode_component_profile.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/environment.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/final_gate.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/memory_breakdown.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/memory_breakdown.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/memory_lifecycle.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/memory_lifecycle.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/oom_forensic.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/points/profile__causal_v4_25_full_model__c2048__b1__d8.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/points/profile__causal_v4_25_full_model__c2048__b2__d8.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/points/profile__causal_v4_25_full_model__c4096__b1__d8.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/points/profile__fp16_full_model__c2048__b1__d8.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/points/profile__fp16_full_model__c2048__b2__d8.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/points/profile__fp16_full_model__c4096__b1__d8.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/scaling_component_analysis.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/summary.md
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/temp_allocation_forensic.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/temp_allocation_forensic.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/worker_summaries.csv
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/profile_decode_only_window/worker_summaries.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/semantic_gate.json
reports/system_full_model_serving_benchmark_v1/direct_compressed_page_append_v1/summary.md
reports/system_full_model_serving_benchmark_v1/final_gate.json
reports/system_full_model_serving_benchmark_v1/final_report.md
reports/system_full_model_serving_benchmark_v1/fp16_full_model_concurrency_sweep.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/README.md
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/capacity_scaling_raw.csv
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/capacity_scaling_raw.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/capacity_scaling_summary.csv
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/capacity_scaling_summary.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/capacity_structural.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/capacity_structural_B1.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/capacity_structural_B2.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/capacity_summary.csv
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/capacity_summary.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/environment.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/exact_commands.md
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/final_gate.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/gpu_state_after_capacity.txt
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/gpu_state_after_matched_b.txt
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/matched_b_paired.csv
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/matched_b_paired.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/matched_b_raw.csv
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/matched_b_raw.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/matched_b_structural.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/matched_b_summary.csv
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/matched_b_summary.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/matched_structural_B1.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/matched_structural_B2.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/matched_structural_B4.json
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/memory_scaling_by_B.csv
reports/system_full_model_serving_benchmark_v1/full_model_b_concurrency_scaling_v1/regressions.txt
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/README.md
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/config_audit.json
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/context_scaling_comparison.csv
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/context_scaling_comparison.json
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/context_scaling_raw.csv
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/context_scaling_raw.json
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/context_scaling_summary.csv
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/context_scaling_summary.json
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/environment.json
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/exact_commands.md
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/final_gate.json
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/gpu_state_after_grid.txt
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/memory_scaling.csv
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/memory_scaling.json
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/old_vs_new_context_scaling.csv
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/old_vs_new_context_scaling.json
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/regressions.txt
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/structural_context_2048.json
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/structural_context_256.json
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/structural_context_4096.json
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/structural_counters.csv
reports/system_full_model_serving_benchmark_v1/full_model_context_scaling_v2/structural_counters.json
reports/system_full_model_serving_benchmark_v1/full_model_path_audit.json
reports/system_full_model_serving_benchmark_v1/full_model_path_parity.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/README.md
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/b1_attention_subbreakdown.csv
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/b1_component_profile.csv
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/b1_component_profile.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/b1_profile_off.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/b1_profile_on.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/b4_attention_subbreakdown.csv
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/b4_component_profile.csv
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/b4_component_profile.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/b4_profile_off.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/b4_profile_on.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/component_comparison.csv
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/component_comparison.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/config_audit.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/exact_commands.md
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/final_b1_profile_off.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/final_b1_profile_on.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/final_gate.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/formal_b1_profile_off_0.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/formal_b1_profile_off_1.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/formal_b1_profile_on_0.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/formal_b4_profile_off_0.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/formal_b4_profile_off_1.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/formal_b4_profile_on_0.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/formal_profile_run_summary.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/gpu5_state_after.txt
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/gpu5_state_after_phase_profile.txt
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/gpu5_state_after_projection_profile.txt
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/gpu5_state_before.txt
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/gpu5_state_before_phase_profile.txt
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/gpu5_state_before_projection_profile.txt
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/invalid_gpu1_attempt.txt
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/memory_profile.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/phase_b1_profile_on.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/phase_b4_profile_on.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/preflight.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/profile_accounting.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/projection_b1_profile_on.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/projection_b4_profile_on.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/regressions.txt
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/repeat_0_b1_profile_off.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/repeat_1_b1_profile_on.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/repeat_2_b4_profile_off.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/repeat_3_b4_profile_on.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/repeat_4_b1_profile_off.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/repeat_5_b4_profile_off.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/repeat_summary.json
reports/system_full_model_serving_benchmark_v1/full_model_post_optimization_bottleneck_profile_v2/structural_counters.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/cache_mutation_forensic.csv
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/cache_mutation_forensic.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/decode_component_profile.csv
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/decode_component_profile.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/environment.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/final_gate.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/memory_breakdown.csv
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/memory_breakdown.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/memory_lifecycle.csv
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/memory_lifecycle.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/oom_forensic.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/points/memory__causal_v4_25_full_model__c4096__b1__d8.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/points/memory__causal_v4_25_full_model__c4096__b2__d8.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/points/memory__causal_v4_25_full_model__c4096__b4__d8.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/points/memory__fp16_full_model__c4096__b1__d8.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/points/memory__fp16_full_model__c4096__b2__d8.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/points/memory__fp16_full_model__c4096__b4__d8.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/points/profile__causal_v4_25_full_model__c2048__b1__d8.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/points/profile__causal_v4_25_full_model__c2048__b2__d8.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/points/profile__causal_v4_25_full_model__c4096__b1__d8.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/points/profile__fp16_full_model__c2048__b1__d8.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/points/profile__fp16_full_model__c2048__b2__d8.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/points/profile__fp16_full_model__c4096__b1__d8.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/scaling_component_analysis.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/summary.md
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/temp_allocation_forensic.csv
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/temp_allocation_forensic.json
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/worker_summaries.csv
reports/system_full_model_serving_benchmark_v1/full_model_post_scaling_bottleneck_forensic_v1/worker_summaries.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/README.md
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/capacity_scaling.csv
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/capacity_scaling.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/context_scaling.csv
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/context_scaling.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/decode_only_b_scaling_raw.csv
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/decode_only_b_scaling_raw.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/decode_only_b_scaling_summary.csv
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/decode_only_b_scaling_summary.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/decode_only_capacity_raw.csv
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/decode_only_capacity_raw.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/decode_only_capacity_summary.csv
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/decode_only_capacity_summary.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/decode_only_context_scaling_raw.csv
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/decode_only_context_scaling_raw.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/decode_only_context_scaling_summary.csv
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/decode_only_context_scaling_summary.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/environment.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/exact_commands.md
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/final_gate.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/master_run_summaries.csv
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/master_run_summaries.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/matched_b_scaling.csv
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/matched_b_scaling.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/memory_scaling_repaired.csv
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/memory_scaling_repaired.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/old_vs_new_comparison.csv
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/old_vs_new_comparison.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/planned_points.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/point_summaries.csv
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/point_summaries.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/capacity__causal_v4_25_full_model__c4096__b1__d8__ac1__tr1__w0__m1.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/capacity__causal_v4_25_full_model__c4096__b2__d8__ac2__tr2__w0__m1.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/capacity__causal_v4_25_full_model__c4096__b4__d8__ac4__tr4__w0__m1.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/capacity__fp16_full_model__c4096__b1__d8__ac1__tr1__w0__m1.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/capacity__fp16_full_model__c4096__b2__d8__ac2__tr2__w0__m1.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/capacity__fp16_full_model__c4096__b4__d8__ac4__tr4__w0__m1.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/context__causal_v4_25_full_model__c2048__b1__d8__ac1__tr1__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/context__causal_v4_25_full_model__c256__b1__d8__ac1__tr1__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/context__causal_v4_25_full_model__c4096__b1__d8__ac1__tr1__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/context__causal_v4_25_full_model__c8192__b1__d8__ac1__tr1__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/context__fp16_full_model__c2048__b1__d8__ac1__tr1__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/context__fp16_full_model__c256__b1__d8__ac1__tr1__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/context__fp16_full_model__c4096__b1__d8__ac1__tr1__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/context__fp16_full_model__c8192__b1__d8__ac1__tr1__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/matched_b__causal_v4_25_full_model__c2048__b1__d8__ac1__tr1__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/matched_b__causal_v4_25_full_model__c2048__b2__d8__ac2__tr2__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/matched_b__causal_v4_25_full_model__c2048__b4__d8__ac4__tr4__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/matched_b__causal_v4_25_full_model__c2048__b8__d8__ac8__tr8__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/matched_b__fp16_full_model__c2048__b1__d8__ac1__tr1__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/matched_b__fp16_full_model__c2048__b2__d8__ac2__tr2__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/matched_b__fp16_full_model__c2048__b4__d8__ac4__tr4__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/matched_b__fp16_full_model__c2048__b8__d8__ac8__tr8__w1__m3.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/smoke__causal_v4_25_full_model__c2048__b1__d8__ac1__tr1__w0__m1.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/smoke__fp16_full_model__c2048__b1__d8__ac1__tr1__w0__m1.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/points/smoke_repeat__fp16_full_model__c2048__b1__d8__ac1__tr1__w0__m1.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/protocol_definition.md
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/protocol_validation.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/structural_counters.json
reports/system_full_model_serving_benchmark_v1/full_model_scaling_decode_only_protocol_repair_v1/summary.md
reports/system_full_model_serving_benchmark_v1/gpu_state.txt
reports/system_full_model_serving_benchmark_v1/integration_optimization_v1/after_profile.json
reports/system_full_model_serving_benchmark_v1/integration_optimization_v1/before_after_summary.csv
reports/system_full_model_serving_benchmark_v1/integration_optimization_v1/centroid_capacity_before.json
reports/system_full_model_serving_benchmark_v1/integration_optimization_v1/correctness_regressions.txt
reports/system_full_model_serving_benchmark_v1/integration_optimization_v1/design_before.md
reports/system_full_model_serving_benchmark_v1/integration_optimization_v1/final_gate.json
reports/system_full_model_serving_benchmark_v1/integration_optimization_v1/final_report.md
reports/system_full_model_serving_benchmark_v1/integration_optimization_v1/iteration_plan_design.md
reports/system_full_model_serving_benchmark_v1/integration_optimization_v1/preflight.txt
reports/system_full_model_serving_benchmark_v1/integration_optimization_v1/row_slice_sites_before.json
reports/system_full_model_serving_benchmark_v1/integration_optimization_v1/stage1_metadata_ab.json
reports/system_full_model_serving_benchmark_v1/integration_optimization_v1/stage2_zero_copy_ab.json
reports/system_full_model_serving_benchmark_v1/integration_optimization_v1/stage3_centroid_ab.json
reports/system_full_model_serving_benchmark_v1/matched_concurrency_summary.csv
reports/system_full_model_serving_benchmark_v1/max_concurrency.json
reports/system_full_model_serving_benchmark_v1/memory_breakdown_probe.json
reports/system_full_model_serving_benchmark_v1/memory_scaling.csv
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/b2_after.json
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/b2_before.json
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/b4_after.json
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/b4_before.json
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/b8_sanity.json
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/copy_site_audit.json
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/dynamic_membership_audit.json
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/final_gate.json
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/final_report.md
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/iteration_plan_multi_request_design.md
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/memory_ab.csv
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/metadata_rebuild_audit.json
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/path_diff_b1_vs_b2_b4.md
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/performance_ab.csv
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/preflight.txt
reports/system_full_model_serving_benchmark_v1/multi_request_low_copy_v1/regressions.txt
reports/system_full_model_serving_benchmark_v1/preflight.txt
reports/system_full_model_serving_benchmark_v1/probe_final_gate.json
reports/system_full_model_serving_benchmark_v1/probe_gpu_state.txt
reports/system_full_model_serving_benchmark_v1/probe_preflight.json
reports/system_full_model_serving_benchmark_v1/raw_runs.json
reports/system_full_model_serving_benchmark_v1/raw_runs.jsonl
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/README.md
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/attention_component_profile.csv
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/attention_component_profile.json
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/canonical_profile_raw.json
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/cta_scheduling_audit.md
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/cuda_graph_headroom.md
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/diagnostic_summary.json
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/environment.json
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/exact_commands.md
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/final_gate.json
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/int2_unpack_audit.md
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/kernel_resource_summary.csv
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/kernel_resource_summary.json
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/kernel_roofline_summary.csv
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/kernel_roofline_summary.json
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/metadata_layout_audit.md
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/minimal_repro_matrix.csv
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/minimal_repro_matrix.json
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/minimal_repro_summary.csv
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/minimal_repro_summary.json
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/path_config_diff.csv
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/per_iteration_counters.csv
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/per_iteration_counters.json
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/per_iteration_counters_profile.csv
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/per_iteration_counters_profile.json
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/refill_events.csv
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/refill_events.json
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/refill_events_profile.csv
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/refill_events_profile.json
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/torch_profiler_kernel_summary.csv
reports/system_full_model_serving_benchmark_v1/reconcile_scaling_path_attention_roofline_v1/torch_profiler_summary.json
reports/system_full_model_serving_benchmark_v1/regression_validation.txt
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/b1_ab.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/b2_ab.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/b4_ab.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/b_greater_than_1_low_copy_audit.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/before_after_summary.csv
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/design.md
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/final_gate.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/final_report.md
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/memory_workspace.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/preflight.txt
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/ragged_gate.txt
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/regressions.txt
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/b2_reorder_identity_dump.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/b4_identity_dump.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/design_identity_model.md
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/final_gate.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/final_report.md
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/first_divergence.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/flush_lifecycle_dump.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/invalid_split_audit.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/pack_window_final_step_comparison.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/pack_window_input_comparison.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/partial_state_comparison.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/performance_sanity_ab.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/preflight.txt
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/ragged_gate.txt
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/regressions.txt
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/semantic_before_after.md
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/semantic_fix_v1/workspace_freshness_audit.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/softmax_after_profile.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/softmax_before_profile.json
reports/system_full_model_serving_benchmark_v1/request_invariant_softmax_v1/split_topology_tests.json
reports/system_full_model_serving_benchmark_v1/scheduler_overhead.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/capacity_after.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/capacity_after.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/correctness.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/decode_regression.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/final_gate.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/cache_mutation_forensic.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/cache_mutation_forensic.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/decode_component_profile.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/decode_component_profile.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/environment.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/final_gate.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/memory_breakdown.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/memory_breakdown.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/memory_lifecycle.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/memory_lifecycle.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/oom_forensic.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/points/memory__causal_v4_25_full_model__c4096__b1__d8.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/points/memory__causal_v4_25_full_model__c4096__b2__d8.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/points/memory__causal_v4_25_full_model__c4096__b4__d8.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/points/memory__fp16_full_model__c4096__b1__d8.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/points/memory__fp16_full_model__c4096__b2__d8.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/points/memory__fp16_full_model__c4096__b4__d8.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/scaling_component_analysis.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/summary.md
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/temp_allocation_forensic.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/temp_allocation_forensic.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/worker_summaries.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/forensic_after/worker_summaries.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/lm_head_shapes.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/memory_before_after.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/memory_before_after.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/README.md
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/capacity_scaling.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/capacity_scaling.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/context_scaling.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/context_scaling.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/decode_only_b_scaling_raw.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/decode_only_b_scaling_raw.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/decode_only_b_scaling_summary.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/decode_only_b_scaling_summary.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/decode_only_capacity_raw.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/decode_only_capacity_raw.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/decode_only_capacity_summary.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/decode_only_capacity_summary.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/decode_only_context_scaling_raw.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/decode_only_context_scaling_raw.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/decode_only_context_scaling_summary.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/decode_only_context_scaling_summary.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/environment.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/exact_commands.md
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/final_gate.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/master_run_summaries.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/master_run_summaries.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/matched_b_scaling.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/matched_b_scaling.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/memory_scaling_repaired.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/memory_scaling_repaired.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/old_vs_new_comparison.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/old_vs_new_comparison.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/point_summaries.csv
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/point_summaries.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/capacity__causal_v4_25_full_model__c4096__b16__d8__ac16__tr16__w0__m1.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/capacity__causal_v4_25_full_model__c4096__b1__d8__ac1__tr1__w0__m1.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/capacity__causal_v4_25_full_model__c4096__b2__d8__ac2__tr2__w0__m1.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/capacity__causal_v4_25_full_model__c4096__b4__d8__ac4__tr4__w0__m1.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/capacity__causal_v4_25_full_model__c4096__b8__d8__ac8__tr8__w0__m1.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/capacity__fp16_full_model__c4096__b1__d8__ac1__tr1__w0__m1.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/capacity__fp16_full_model__c4096__b2__d8__ac2__tr2__w0__m1.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/capacity__fp16_full_model__c4096__b4__d8__ac4__tr4__w0__m1.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/capacity__fp16_full_model__c4096__b8__d8__ac8__tr8__w0__m1.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/matched_b__causal_v4_25_full_model__c2048__b1__d8__ac1__tr1__w1__m3.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/matched_b__causal_v4_25_full_model__c2048__b2__d8__ac2__tr2__w1__m3.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/matched_b__causal_v4_25_full_model__c2048__b4__d8__ac4__tr4__w1__m3.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/matched_b__causal_v4_25_full_model__c2048__b8__d8__ac8__tr8__w1__m3.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/matched_b__fp16_full_model__c2048__b1__d8__ac1__tr1__w1__m3.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/matched_b__fp16_full_model__c2048__b2__d8__ac2__tr2__w1__m3.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/matched_b__fp16_full_model__c2048__b4__d8__ac4__tr4__w1__m3.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/matched_b__fp16_full_model__c2048__b8__d8__ac8__tr8__w1__m3.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/smoke__causal_v4_25_full_model__c2048__b1__d8__ac1__tr1__w0__m1.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/smoke__fp16_full_model__c2048__b1__d8__ac1__tr1__w0__m1.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/points/smoke_repeat__fp16_full_model__c2048__b1__d8__ac1__tr1__w0__m1.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/protocol_definition.md
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/protocol_validation.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/structural_counters.json
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/repaired_after/summary.md
reports/system_full_model_serving_benchmark_v1/selective_prefill_logits_projection_v1/summary.md
reports/system_full_model_serving_benchmark_v1/smoke_results.json
reports/system_full_model_serving_benchmark_v1/summary.json
reports/system_full_model_serving_benchmark_v1/throughput_scaling.csv
reports/system_full_model_serving_benchmark_v1/workload.json
reports/system_full_model_serving_benchmark_v1_probe_256/baseline_audit.json
reports/system_full_model_serving_benchmark_v1_probe_256/benchmark_config.json
reports/system_full_model_serving_benchmark_v1_probe_256/causal_v4_25_full_model_concurrency_sweep.json
reports/system_full_model_serving_benchmark_v1_probe_256/fp16_full_model_concurrency_sweep.json
reports/system_full_model_serving_benchmark_v1_probe_256/full_model_path_audit.json
reports/system_full_model_serving_benchmark_v1_probe_256/gpu_state.txt
reports/system_full_model_serving_benchmark_v1_probe_256/matched_concurrency_summary.csv
reports/system_full_model_serving_benchmark_v1_probe_256/max_concurrency.json
reports/system_full_model_serving_benchmark_v1_probe_256/memory_scaling.csv
reports/system_full_model_serving_benchmark_v1_probe_256/preflight.txt
reports/system_full_model_serving_benchmark_v1_probe_256/raw_runs.json
reports/system_full_model_serving_benchmark_v1_probe_256/raw_runs.jsonl
reports/system_full_model_serving_benchmark_v1_probe_256/scheduler_overhead.json
reports/system_full_model_serving_benchmark_v1_probe_256/smoke_results.json
reports/system_full_model_serving_benchmark_v1_probe_256/summary.json
reports/system_full_model_serving_benchmark_v1_probe_256/throughput_scaling.csv
reports/system_full_model_serving_benchmark_v1_probe_256/workload.json
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/baseline_audit.json
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/benchmark_config.json
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/causal_v4_25_full_model_concurrency_sweep.json
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/fp16_full_model_concurrency_sweep.json
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/full_model_path_audit.json
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/gpu_state.txt
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/matched_concurrency_summary.csv
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/max_concurrency.json
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/memory_scaling.csv
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/preflight.txt
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/raw_runs.json
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/raw_runs.jsonl
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/scheduler_overhead.json
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/smoke_results.json
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/summary.json
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/throughput_scaling.csv
reports/system_full_model_serving_benchmark_v1_probe_ctx256_decode16/workload.json
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/baseline_audit.json
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/benchmark_config.json
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/causal_v4_25_full_model_concurrency_sweep.json
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/fp16_full_model_concurrency_sweep.json
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/full_model_path_audit.json
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/gpu_state.txt
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/matched_concurrency_summary.csv
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/max_concurrency.json
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/memory_scaling.csv
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/preflight.txt
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/raw_runs.json
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/raw_runs.jsonl
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/scheduler_overhead.json
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/smoke_results.json
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/summary.json
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/throughput_scaling.csv
reports/system_full_model_serving_benchmark_v1_probe_ctx4096_decode16/workload.json
reports/system_full_model_serving_benchmark_v1_smoke/baseline_audit.json
reports/system_full_model_serving_benchmark_v1_smoke/fp16_full_model_concurrency_sweep.json
reports/system_full_model_serving_benchmark_v1_smoke/full_model_path_audit.json
reports/system_full_model_serving_benchmark_v1_smoke/gpu_state.txt
reports/system_full_model_serving_benchmark_v1_smoke/preflight.txt
reports/system_full_model_serving_benchmark_v1_smoke_causal/baseline_audit.json
reports/system_full_model_serving_benchmark_v1_smoke_causal/benchmark_config.json
reports/system_full_model_serving_benchmark_v1_smoke_causal/causal_v4_25_full_model_concurrency_sweep.json
reports/system_full_model_serving_benchmark_v1_smoke_causal/full_model_path_audit.json
reports/system_full_model_serving_benchmark_v1_smoke_causal/gpu_state.txt
reports/system_full_model_serving_benchmark_v1_smoke_causal/matched_concurrency_summary.csv
reports/system_full_model_serving_benchmark_v1_smoke_causal/max_concurrency.json
reports/system_full_model_serving_benchmark_v1_smoke_causal/memory_scaling.csv
reports/system_full_model_serving_benchmark_v1_smoke_causal/preflight.txt
reports/system_full_model_serving_benchmark_v1_smoke_causal/raw_runs.json
reports/system_full_model_serving_benchmark_v1_smoke_causal/raw_runs.jsonl
reports/system_full_model_serving_benchmark_v1_smoke_causal/scheduler_overhead.json
reports/system_full_model_serving_benchmark_v1_smoke_causal/smoke_results.json
reports/system_full_model_serving_benchmark_v1_smoke_causal/summary.json
reports/system_full_model_serving_benchmark_v1_smoke_causal/throughput_scaling.csv
reports/system_full_model_serving_benchmark_v1_smoke_causal/workload.json
reports/system_iteration_level_continuous_batching_v1/continuous_batch_sequence.json
reports/system_iteration_level_continuous_batching_v1/dynamic_add_remove_regression.txt
reports/system_iteration_level_continuous_batching_v1/final_gate.json
reports/system_iteration_level_continuous_batching_v1/full_pytest.txt
reports/system_iteration_level_continuous_batching_v1/git_diff_check.txt
reports/system_iteration_level_continuous_batching_v1/git_state.txt
reports/system_iteration_level_continuous_batching_v1/lifecycle_regression.txt
reports/system_iteration_level_continuous_batching_v1/ragged_regression.txt
reports/system_iteration_level_continuous_batching_v1/runtime_env.txt
reports/system_iteration_level_continuous_batching_v1/scheduler_trace.json
reports/system_iteration_level_continuous_batching_v1/survivor_trajectory.json
reports/system_iteration_level_continuous_batching_v1/targeted_pytest.txt
reports/system_late_step_post_attention_rmsnorm_v1/b2_16step_postfix.json
reports/system_late_step_post_attention_rmsnorm_v1/b2_16step_postfix.md
reports/system_late_step_post_attention_rmsnorm_v1/b2_reorder_postfix.json
reports/system_late_step_post_attention_rmsnorm_v1/b4_postfix.json
reports/system_late_step_post_attention_rmsnorm_v1/environment.md
reports/system_late_step_post_attention_rmsnorm_v1/final_gate.json
reports/system_late_step_post_attention_rmsnorm_v1/fixed_reduction_rmsnorm_oracle.json
reports/system_late_step_post_attention_rmsnorm_v1/independent_flush_postfix.json
reports/system_late_step_post_attention_rmsnorm_v1/layer9_persistent_postfix.json
reports/system_late_step_post_attention_rmsnorm_v1/layout_rmsnorm_oracle.json
reports/system_late_step_post_attention_rmsnorm_v1/peer_content_rmsnorm_oracle.json
reports/system_late_step_post_attention_rmsnorm_v1/peer_length_rmsnorm_oracle.json
reports/system_late_step_post_attention_rmsnorm_v1/preflight.json
reports/system_late_step_post_attention_rmsnorm_v1/prior_fix_state.md
reports/system_late_step_post_attention_rmsnorm_v1/production_fix.md
reports/system_late_step_post_attention_rmsnorm_v1/real_input_normal_rmsnorm_oracle.json
reports/system_late_step_post_attention_rmsnorm_v1/reference_rmsnorm_oracle.json
reports/system_late_step_post_attention_rmsnorm_v1/regression_summary.md
reports/system_late_step_post_attention_rmsnorm_v1/reorder_rmsnorm_oracle.json
reports/system_late_step_post_attention_rmsnorm_v1/rmsnorm_boundary_call_graph.md
reports/system_late_step_post_attention_rmsnorm_v1/rmsnorm_implementation.md
reports/system_late_step_post_attention_rmsnorm_v1/rmsnorm_input_layout_comparison.json
reports/system_late_step_post_attention_rmsnorm_v1/rmsnorm_internal_reduction_trace.json
reports/system_late_step_post_attention_rmsnorm_v1/secondary_late_step_divergence.json
reports/system_late_step_post_attention_rmsnorm_v1/step5_layer8_boundary_trace.json
reports/system_late_step_post_attention_rmsnorm_v1/step5_layer8_postfix.json
reports/system_late_step_post_attention_rmsnorm_v1/system_invariants.json
reports/system_late_step_post_attention_rmsnorm_v1/temporal_timeline_postfix.json
reports/system_late_step_post_attention_rmsnorm_v1/temporal_timeline_postfix.md
reports/system_ragged_active_state_forensic_v1/active_state_timeline.json
reports/system_ragged_active_state_forensic_v1/active_state_timeline.md
reports/system_ragged_active_state_forensic_v1/control_matrix.json
reports/system_ragged_active_state_forensic_v1/environment.md
reports/system_ragged_active_state_forensic_v1/final_gate.json
reports/system_ragged_active_state_forensic_v1/inactive_capacity_false_positive_regression.json
reports/system_ragged_active_state_forensic_v1/original_failure_reproduction.json
reports/system_ragged_active_state_forensic_v1/padding_poison_oracle.json
reports/system_ragged_active_state_forensic_v1/pre_step5_snapshot.json
reports/system_ragged_active_state_forensic_v1/prefill_active_state_comparison.json
reports/system_ragged_active_state_forensic_v1/preflight.json
reports/system_ragged_active_state_forensic_v1/reference_oracle_audit.md
reports/system_ragged_active_state_forensic_v1/request_isolation_oracle.json
reports/system_ragged_active_state_forensic_v1/root_cause_status.md
reports/system_ragged_active_state_forensic_v1/semantic_extraction_rules.md
reports/system_ragged_active_state_forensic_v1/semantic_state_manifest.json
reports/system_ragged_active_state_forensic_v1/semantic_state_manifest.md
reports/system_ragged_active_state_forensic_v1/step1_active_state_comparison.json
reports/system_ragged_active_state_forensic_v1/step2_active_state_comparison.json
reports/system_ragged_active_state_forensic_v1/step3_active_state_comparison.json
reports/system_ragged_active_state_forensic_v1/step4_active_state_comparison.json
reports/system_ragged_active_state_forensic_v1/step5_layerwise_trace.json
reports/system_ragged_active_state_forensic_v1/step5_layerwise_trace.md
reports/system_recent_k_ownership_forensic_v1/batch_row_reorder_oracle.json
reports/system_recent_k_ownership_forensic_v1/environment.md
reports/system_recent_k_ownership_forensic_v1/final_gate.json
reports/system_recent_k_ownership_forensic_v1/golden_recent_transition_oracle.json
reports/system_recent_k_ownership_forensic_v1/peer_content_independence_oracle.json
reports/system_recent_k_ownership_forensic_v1/peer_length_oracle.json
reports/system_recent_k_ownership_forensic_v1/preflight.json
reports/system_recent_k_ownership_forensic_v1/read_write_identity_audit.md
reports/system_recent_k_ownership_forensic_v1/recent_k_call_graph.md
reports/system_recent_k_ownership_forensic_v1/recent_k_state_ownership_manifest.json
reports/system_recent_k_ownership_forensic_v1/recent_k_transition_contract.md
reports/system_recent_k_ownership_forensic_v1/root_cause_evidence.md
reports/system_recent_k_ownership_forensic_v1/slot_reuse_poison_oracle.json
reports/system_recent_k_ownership_forensic_v1/step1_layer0_transition_b1.json
reports/system_recent_k_ownership_forensic_v1/step1_layer0_transition_ragged.json
reports/system_recent_k_ownership_forensic_v1/transition_address_comparison.json
reports/system_recent_k_ownership_forensic_v1/transition_input_comparison.json
reports/system_recent_k_ownership_forensic_v1/valid_length_audit.md
reports/system_request_invariant_attention_softmax_fix_v1/attention_pre_o_postfix.json
reports/system_request_invariant_attention_softmax_fix_v1/attention_probability_postfix.json
reports/system_request_invariant_attention_softmax_fix_v1/b2_16step_postfix.json
reports/system_request_invariant_attention_softmax_fix_v1/b2_16step_postfix.md
reports/system_request_invariant_attention_softmax_fix_v1/b2_reorder_postfix.json
reports/system_request_invariant_attention_softmax_fix_v1/b4_postfix.json
reports/system_request_invariant_attention_softmax_fix_v1/b4_split_test.json
reports/system_request_invariant_attention_softmax_fix_v1/before_fix_planner.md
reports/system_request_invariant_attention_softmax_fix_v1/environment.md
reports/system_request_invariant_attention_softmax_fix_v1/final_gate.json
reports/system_request_invariant_attention_softmax_fix_v1/fixed_split_size_selection.md
reports/system_request_invariant_attention_softmax_fix_v1/independent_flush_postfix.json
reports/system_request_invariant_attention_softmax_fix_v1/layer0_propagation_postfix.json
reports/system_request_invariant_attention_softmax_fix_v1/layer1_propagation_postfix.json
reports/system_request_invariant_attention_softmax_fix_v1/peer_content_split_test.json
reports/system_request_invariant_attention_softmax_fix_v1/peer_length_split_test.json
reports/system_request_invariant_attention_softmax_fix_v1/preflight.json
reports/system_request_invariant_attention_softmax_fix_v1/production_fix.md
reports/system_request_invariant_attention_softmax_fix_v1/regression_summary.md
reports/system_request_invariant_attention_softmax_fix_v1/reorder_split_test.json
reports/system_request_invariant_attention_softmax_fix_v1/request_invariant_softmax_contract.md
reports/system_request_invariant_attention_softmax_fix_v1/secondary_value_reduction.json
reports/system_request_invariant_attention_softmax_fix_v1/softmax_internal_state_postfix.json
reports/system_request_invariant_attention_softmax_fix_v1/split_boundary_unit_tests.json
reports/system_request_invariant_attention_softmax_fix_v1/step1_layer0_softmax_postfix.json
reports/system_request_invariant_attention_softmax_fix_v1/system_invariants.json
reports/system_request_invariant_attention_value_fix_v1/attention_pre_o_postfix.json
reports/system_request_invariant_attention_value_fix_v1/attention_probability_regression.json
reports/system_request_invariant_attention_value_fix_v1/attention_value_reduction_call_graph.md
reports/system_request_invariant_attention_value_fix_v1/b2_16step_postfix.json
reports/system_request_invariant_attention_value_fix_v1/b2_16step_postfix.md
reports/system_request_invariant_attention_value_fix_v1/b2_reorder_postfix.json
reports/system_request_invariant_attention_value_fix_v1/b4_postfix.json
reports/system_request_invariant_attention_value_fix_v1/b4_value_test.json
reports/system_request_invariant_attention_value_fix_v1/before_fix_value_planner.md
reports/system_request_invariant_attention_value_fix_v1/current_worktree_fix_state.md
reports/system_request_invariant_attention_value_fix_v1/environment.md
reports/system_request_invariant_attention_value_fix_v1/final_gate.json
reports/system_request_invariant_attention_value_fix_v1/fixed_value_split_size_selection.md
reports/system_request_invariant_attention_value_fix_v1/golden_value_reduction_postfix.json
reports/system_request_invariant_attention_value_fix_v1/independent_flush_postfix.json
reports/system_request_invariant_attention_value_fix_v1/layer0_propagation_postfix.json
reports/system_request_invariant_attention_value_fix_v1/layer1_propagation_postfix.json
reports/system_request_invariant_attention_value_fix_v1/peer_content_value_test.json
reports/system_request_invariant_attention_value_fix_v1/peer_length_value_test.json
reports/system_request_invariant_attention_value_fix_v1/per_token_pv_postfix.json
reports/system_request_invariant_attention_value_fix_v1/preflight.json
reports/system_request_invariant_attention_value_fix_v1/production_fix.md
reports/system_request_invariant_attention_value_fix_v1/regression_summary.md
reports/system_request_invariant_attention_value_fix_v1/reorder_value_test.json
reports/system_request_invariant_attention_value_fix_v1/request_invariant_value_contract.md
reports/system_request_invariant_attention_value_fix_v1/secondary_divergence_after_value_fix.json
reports/system_request_invariant_attention_value_fix_v1/system_invariants.json
reports/system_request_invariant_attention_value_fix_v1/value_split_boundary_tests.json
reports/system_request_lifecycle_manager_v1/allocation_peer_isolation.json
reports/system_request_lifecycle_manager_v1/allocation_reset_contract.md
reports/system_request_lifecycle_manager_v1/environment.md
reports/system_request_lifecycle_manager_v1/final_gate.json
reports/system_request_lifecycle_manager_v1/full_pytest.txt
reports/system_request_lifecycle_manager_v1/git_state.txt
reports/system_request_lifecycle_manager_v1/lifecycle_architecture.md
reports/system_request_lifecycle_manager_v1/lifecycle_state_machine.md
reports/system_request_lifecycle_manager_v1/manual_lifecycle_sequence.json
reports/system_request_lifecycle_manager_v1/manual_lifecycle_sequence.md
reports/system_request_lifecycle_manager_v1/middle_row_removal.json
reports/system_request_lifecycle_manager_v1/page_ownership_contract.md
reports/system_request_lifecycle_manager_v1/page_ownership_release.json
reports/system_request_lifecycle_manager_v1/persistent_slot_contract.md
reports/system_request_lifecycle_manager_v1/persistent_state_reset_matrix.json
reports/system_request_lifecycle_manager_v1/preflight.json
reports/system_request_lifecycle_manager_v1/ragged_gate_regression.json
reports/system_request_lifecycle_manager_v1/ragged_regression.txt
reports/system_request_lifecycle_manager_v1/regression_summary.md
reports/system_request_lifecycle_manager_v1/release_contract.md
reports/system_request_lifecycle_manager_v1/release_peer_isolation.json
reports/system_request_lifecycle_manager_v1/request_state_inventory.md
reports/system_request_lifecycle_manager_v1/row_identity_audit.md
reports/system_request_lifecycle_manager_v1/row_remap_state_comparison.json
reports/system_request_lifecycle_manager_v1/runtime_env.txt
reports/system_request_lifecycle_manager_v1/slot_reuse_contract.md
reports/system_request_lifecycle_manager_v1/slot_reuse_poison_test.json
reports/system_request_lifecycle_manager_v1/system_invariants.json
reports/system_request_lifecycle_manager_v1/targeted_pytest.txt
reports/system_secondary_mlp_batch_invariance_v1/activation_comparison.json
reports/system_secondary_mlp_batch_invariance_v1/attention_regression_after_mlp_fix.json
reports/system_secondary_mlp_batch_invariance_v1/b2_16step_postfix.json
reports/system_secondary_mlp_batch_invariance_v1/b2_16step_postfix.md
reports/system_secondary_mlp_batch_invariance_v1/b2_reorder_postfix.json
reports/system_secondary_mlp_batch_invariance_v1/b4_postfix.json
reports/system_secondary_mlp_batch_invariance_v1/bi_mlp_forward_oracle.json
reports/system_secondary_mlp_batch_invariance_v1/bi_mlp_frozen_oracle.json
reports/system_secondary_mlp_batch_invariance_v1/down_proj_comparison.json
reports/system_secondary_mlp_batch_invariance_v1/environment.md
reports/system_secondary_mlp_batch_invariance_v1/final_gate.json
reports/system_secondary_mlp_batch_invariance_v1/gate_proj_comparison.json
reports/system_secondary_mlp_batch_invariance_v1/gated_product_comparison.json
reports/system_secondary_mlp_batch_invariance_v1/independent_flush_postfix.json
reports/system_secondary_mlp_batch_invariance_v1/layer0_propagation_postfix.json
reports/system_secondary_mlp_batch_invariance_v1/layer1_propagation_postfix.json
reports/system_secondary_mlp_batch_invariance_v1/minimal_bi_mlp_ablation.json
reports/system_secondary_mlp_batch_invariance_v1/mlp_boundary_trace.json
reports/system_secondary_mlp_batch_invariance_v1/mlp_call_graph.md
reports/system_secondary_mlp_batch_invariance_v1/mlp_input_comparison.json
reports/system_secondary_mlp_batch_invariance_v1/mlp_norm_comparison.json
reports/system_secondary_mlp_batch_invariance_v1/preflight.json
reports/system_secondary_mlp_batch_invariance_v1/production_mlp_fix.md
reports/system_secondary_mlp_batch_invariance_v1/production_mlp_postfix.json
reports/system_secondary_mlp_batch_invariance_v1/real_input_mlp_frozen_oracle.json
reports/system_secondary_mlp_batch_invariance_v1/regression_summary.md
reports/system_secondary_mlp_batch_invariance_v1/secondary_divergence_after_mlp_fix.json
reports/system_secondary_mlp_batch_invariance_v1/system_invariants.json
reports/system_secondary_mlp_batch_invariance_v1/up_proj_comparison.json
reports/system_serving_benchmark_v1/baseline_audit.json
reports/system_serving_benchmark_v1/benchmark_config.json
reports/system_serving_benchmark_v1/environment.json
reports/system_serving_benchmark_v1/final_gate.json
reports/system_serving_benchmark_v1/final_report.md
reports/system_serving_benchmark_v1/full_pytest.txt
reports/system_serving_benchmark_v1/git_diff_check.txt
reports/system_serving_benchmark_v1/git_state.txt
reports/system_serving_benchmark_v1/gpu_state_before.txt
reports/system_serving_benchmark_v1/max_concurrency.json
reports/system_serving_benchmark_v1/memory_results.json
reports/system_serving_benchmark_v1/preflight.txt
reports/system_serving_benchmark_v1/raw_runs.json
reports/system_serving_benchmark_v1/raw_runs.jsonl
reports/system_serving_benchmark_v1/regressions.txt
reports/system_serving_benchmark_v1/scheduler_overhead.json
reports/system_serving_benchmark_v1/smoke/benchmark_config.json
reports/system_serving_benchmark_v1/smoke/max_concurrency.json
reports/system_serving_benchmark_v1/smoke/raw_runs.json
reports/system_serving_benchmark_v1/smoke/raw_runs.jsonl
reports/system_serving_benchmark_v1/smoke/summary.csv
reports/system_serving_benchmark_v1/smoke/summary.json
reports/system_serving_benchmark_v1/summary.csv
reports/system_serving_benchmark_v1/summary.json
reports/system_step1_layer0_kpath_forensic_v1/environment.md
reports/system_step1_layer0_kpath_forensic_v1/existing_bi_kproj_audit.md
reports/system_step1_layer0_kpath_forensic_v1/existing_bi_kproj_oracle.json
reports/system_step1_layer0_kpath_forensic_v1/final_gate.json
reports/system_step1_layer0_kpath_forensic_v1/fixed_shape_padding_oracle.json
reports/system_step1_layer0_kpath_forensic_v1/fp32_diagnostic_oracle.json
reports/system_step1_layer0_kpath_forensic_v1/frozen_kproj_batch_shape_oracle.json
reports/system_step1_layer0_kpath_forensic_v1/kpath_call_graph.md
reports/system_step1_layer0_kpath_forensic_v1/kproj_kernel_audit.md
reports/system_step1_layer0_kpath_forensic_v1/kproj_runtime_dispatch.json
reports/system_step1_layer0_kpath_forensic_v1/model_input_comparison.json
reports/system_step1_layer0_kpath_forensic_v1/operator_boundary_comparison.json
reports/system_step1_layer0_kpath_forensic_v1/operator_boundary_comparison.md
reports/system_step1_layer0_kpath_forensic_v1/peer_content_fixed_shape_oracle.json
reports/system_step1_layer0_kpath_forensic_v1/preflight.json
reports/system_step1_layer0_kpath_forensic_v1/raw_projection_comparison.json
reports/system_step1_layer0_kpath_forensic_v1/rmsnorm_batch_shape_oracle.json
reports/system_step1_layer0_kpath_forensic_v1/root_cause_evidence.md
reports/system_v_causal_importance_forensic_v1/attention_golden_oracle.json
reports/system_v_causal_importance_forensic_v1/attention_input_comparison.json
reports/system_v_causal_importance_forensic_v1/attention_logits_comparison.json
reports/system_v_causal_importance_forensic_v1/attention_probs_comparison.json
reports/system_v_causal_importance_forensic_v1/batch_row_reorder_oracle.json
reports/system_v_causal_importance_forensic_v1/current_worktree_fix_state.md
reports/system_v_causal_importance_forensic_v1/direct_input_manifest.json
reports/system_v_causal_importance_forensic_v1/environment.md
reports/system_v_causal_importance_forensic_v1/existing_bi_qproj_oracle.json
reports/system_v_causal_importance_forensic_v1/final_gate.json
reports/system_v_causal_importance_forensic_v1/golden_importance_update_oracle.json
reports/system_v_causal_importance_forensic_v1/importance_addressing_audit.md
reports/system_v_causal_importance_forensic_v1/importance_metadata_comparison.json
reports/system_v_causal_importance_forensic_v1/instant_importance_signal_comparison.json
reports/system_v_causal_importance_forensic_v1/peer_content_oracle.json
reports/system_v_causal_importance_forensic_v1/peer_length_oracle.json
reports/system_v_causal_importance_forensic_v1/preflight.json
reports/system_v_causal_importance_forensic_v1/previous_importance_comparison.json
reports/system_v_causal_importance_forensic_v1/pytest.md
reports/system_v_causal_importance_forensic_v1/q_path_comparison.json
reports/system_v_causal_importance_forensic_v1/qproj_batch_shape_oracle.json
reports/system_v_causal_importance_forensic_v1/root_cause_evidence.md
reports/system_v_causal_importance_forensic_v1/v_causal_importance_call_graph.md
reports/system_v_causal_importance_forensic_v1/v_causal_importance_update_contract.md
reports/system_v_causal_importance_ragged_mapping_fix_v1/b2_16step_postfix.json
reports/system_v_causal_importance_ragged_mapping_fix_v1/b2_16step_postfix.md
reports/system_v_causal_importance_ragged_mapping_fix_v1/b2_reorder_postfix.json
reports/system_v_causal_importance_ragged_mapping_fix_v1/b4_postfix.json
reports/system_v_causal_importance_ragged_mapping_fix_v1/b4_postfix.md
reports/system_v_causal_importance_ragged_mapping_fix_v1/before_fix_mapping.md
reports/system_v_causal_importance_ragged_mapping_fix_v1/environment.md
reports/system_v_causal_importance_ragged_mapping_fix_v1/final_gate.json
reports/system_v_causal_importance_ragged_mapping_fix_v1/golden_update_postfix.json
reports/system_v_causal_importance_ragged_mapping_fix_v1/independent_flush_postfix.json
reports/system_v_causal_importance_ragged_mapping_fix_v1/logical_index_contract.md
reports/system_v_causal_importance_ragged_mapping_fix_v1/mapping_unit_tests.json
reports/system_v_causal_importance_ragged_mapping_fix_v1/peer_content_postfix.json
reports/system_v_causal_importance_ragged_mapping_fix_v1/peer_length_postfix.json
reports/system_v_causal_importance_ragged_mapping_fix_v1/physical_segment_layout.md
reports/system_v_causal_importance_ragged_mapping_fix_v1/preflight.json
reports/system_v_causal_importance_ragged_mapping_fix_v1/production_fix.md
reports/system_v_causal_importance_ragged_mapping_fix_v1/regression_summary.md
reports/system_v_causal_importance_ragged_mapping_fix_v1/reorder_postfix.json
reports/system_v_causal_importance_ragged_mapping_fix_v1/segment_mapping_tests.json
reports/system_v_causal_importance_ragged_mapping_fix_v1/step1_layer0_postfix.json
reports/system_v_causal_importance_ragged_mapping_fix_v1/system_invariants.json
scripts/attention_qk_online_softmax_forensic.py
scripts/attention_value_reduction_forensic.py
scripts/b4_attention_microtrace.py
scripts/b4_request_count_ragged_divergence.py
scripts/centroid_determinism_causal_forensic.py
scripts/first_late_step_persistent_divergence.py
scripts/full_decode_batch_invariance_oracle.py
scripts/full_model_post_scaling_bottleneck_forensic.py
scripts/full_model_scaling_decode_only_protocol_repair.py
scripts/late_step_post_attention_rmsnorm_gate.py
scripts/ragged_active_state_forensic.py
scripts/recent_k_ownership_forensic.py
scripts/reconcile_scaling_path_attention_roofline.py
scripts/request_invariant_attention_softmax_fix_gate.py
scripts/secondary_mlp_batch_invariance_gate.py
scripts/step1_layer0_k_path_forensic.py
scripts/v_causal_importance_forensic.py
tests/test_b4_request_count_ragged_divergence.py
tests/test_dynamic_add_remove_batching.py
tests/test_first_late_step_persistent_divergence.py
tests/test_full_model_post_scaling_bottleneck_forensic.py
tests/test_full_model_scaling_decode_only_protocol_repair.py
tests/test_full_model_serving_benchmark.py
tests/test_iteration_level_continuous_batching.py
tests/test_request_invariant_rmsnorm.py
tests/test_request_lifecycle_manager.py
tests/test_selective_prefill_logits_projection.py
tests/test_serving_benchmark_harness.py
```

## modified

```text
bench/run_actual_model_bi_prefill_runtime.py
bench/run_bi_vproj_cost_benefit.py
bench/run_prefill_projection_mode_policy.py
bench/run_ragged_multistep_correctness.py
models/llama_patternkv.py
models/segmented_cache.py
quant/batch_invariant_kproj.py
quant/csrc/gemv_cuda.cu
quant/csrc/gemv_cuda.h
quant/csrc/pybind.cpp
quant/matmul.py
quant/page_batch.py
reports/system_ragged_multistep_correctness_v1/b2_16step.md
reports/system_ragged_multistep_correctness_v1/b2_flush_events.json
reports/system_ragged_multistep_correctness_v1/b2_flush_schedule.md
reports/system_ragged_multistep_correctness_v1/b2_reorder.md
reports/system_ragged_multistep_correctness_v1/b2_reorder_steps.json
reports/system_ragged_multistep_correctness_v1/b2_steps.json
reports/system_ragged_multistep_correctness_v1/b4_16step.md
reports/system_ragged_multistep_correctness_v1/b4_flush_events.json
reports/system_ragged_multistep_correctness_v1/b4_flush_schedule.md
reports/system_ragged_multistep_correctness_v1/b4_steps.json
reports/system_ragged_multistep_correctness_v1/centroid_events.json
reports/system_ragged_multistep_correctness_v1/environment.md
reports/system_ragged_multistep_correctness_v1/final_gate.json
reports/system_ragged_multistep_correctness_v1/free_run.json
reports/system_ragged_multistep_correctness_v1/page_events.json
reports/system_ragged_multistep_correctness_v1/pytest.md
reports/system_ragged_multistep_correctness_v1/runtime_counters.json
reports/system_ragged_multistep_correctness_v1/semantic_metrics.json
tests/test_bi_kproj_prefill_runtime.py
tests/test_bi_mlp_oracle.py
tests/test_fused_page_batch_operator.py
tests/test_ragged_cache_assembly.py
tests/test_ragged_k_valid_lengths.py
tests/test_value_direction_screen.py
```
