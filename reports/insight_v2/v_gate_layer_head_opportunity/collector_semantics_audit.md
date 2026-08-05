# Collector Semantics Audit

Status: `aggregate_only`

| file | line | function | evidence | interpretation |
| --- | ---: | --- | --- | --- |
| insight/hook_metrics.py | 275 | record_prefill_v_metrics | oracle = pattern_mse_vec < raw_mse_vec; gate is the current mask. | Positive class means pattern wins over raw; gate acceptance is predicted positive. |
| insight/hook_metrics.py | 277 | record_prefill_v_metrics | tp = gate & oracle; tn = ~gate & ~oracle; fp = gate & ~oracle; fn = ~gate & oracle. | TP/TN/FP/FN are token-level confusion counts, not per-sample labels. |
| insight/hook_metrics.py | 282 | record_prefill_v_metrics | observer.add_confusion(f"{prefix}.gate_vs_mse_oracle", ...). | Confusion counters are accumulated directly into the collector. |
| insight/hook_metrics.py | 283 | record_prefill_v_metrics | raw_mse, pattern_candidate_mse, actual_selected_path_mse, relative_candidate_benefit are recorded as scalars. | The raw/pattern MSE signals exist in raw observer output, but not in the summary confusion CSV. |
| insight/hook_metrics.py | 299 | record_prefill_v_metrics | sample record contains gate_current, gate_oracle, rho, false_positive_penalty, false_negative_opportunity. | Candidate-level sweep fields exist only in sample records, not in v_gate_confusion.csv. |
| scripts/summarize_insight_wave_a_8gpu.py | 209 | main | confusion_store accumulates counters from observer payloads. | CSV rows are built from complete confusion counters, not from truncated sample records. |
| scripts/summarize_insight_wave_a_8gpu.py | 230 | main | v_gate_rows are generated from aggregated confusion_store totals. | The published CSV reflects aggregate counts at task/layer/head/bucket granularity. |
