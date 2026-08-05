# Decode Trigger Code Audit

| file | line | function | condition | interpretation |
| --- | --- | --- | --- | --- |
| models/llama_patternkv.py | 717 | LlamaAttention forward decode K path | if key_states_full.shape[-2] == self.residual_length | Decode K window metrics fire only when the residual decode buffer reaches exactly residual_length tokens. |
| models/llama_patternkv.py | 737 | LlamaAttention forward decode K path | window_idx=int(assignments.shape[-1] // self.residual_length) if assignments is not None else 0 | The first decode event after prefill uses the number of already-quantized windows as the decode window index. |
| models/llama_patternkv.py | 845 | LlamaAttention forward decode V path | if value_full_length == self.residual_length | Decode V window metrics use the same exact-full-window trigger as decode K. |
| insight/hook_metrics.py | 420 | record_decode_k_window_metrics | observer.add_scalar for old_mse/new_mse/relative_mse_gain/relative_range_gain/candidate_assignment_fraction | Each decode K event emits five aggregate metrics per layer-head, but the published pattern_gain_map.csv keeps only relative_mse_gain and relative_range_gain. |
| insight/hook_metrics.py | 489 | record_decode_v_window_metrics | observer.add_scalar for old/new assignment/actual MSE, candidate_assignment_fraction, candidate_gate_accepted_fraction | Each decode V event emits six aggregate metrics per layer-head, but the published dynamic_pattern_utility.csv keeps only candidate_gate_accepted_fraction. |
| scripts/summarize_insight_wave_a_8gpu.py | 227 | main | pattern_gain_rows metric filter | The published pattern_gain_map.csv keeps relative_benefit, relative_mse_gain, relative_candidate_benefit, range_contraction, and relative_range_gain only. |
| scripts/summarize_insight_wave_a_8gpu.py | 229 | main | dynamic_rows metric filter | The published dynamic_pattern_utility.csv keeps candidate_gate_accepted_fraction and a small fixed metric allowlist only. |
| bench/paper_config.py | 162 | pattern_boundary_events | step % residual_length == 0 | The helper encodes the same 1-based residual-length boundary rule used by the runtime audit model. |
