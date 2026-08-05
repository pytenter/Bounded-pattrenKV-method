# Schema Audit

| root | csv | rows | duplicate primary keys | header |
| --- | --- | ---: | ---: | --- |
| v100 | pattern_gain_map.csv | 15360 | 0 | task, phase, kv_type, layer, kv_head, bucket, metric, count, mean, min, max, std |
| v100 | matching_oracle_gap.csv | 2160 | 0 | task, phase, kv_type, layer, kv_head, bucket, metric, count, mean, min, max, std |
| v100 | v_gate_confusion.csv | 4608 | 0 | task, phase, kv_type, layer, kv_head, bucket, metric, true_positive, true_negative, false_positive, false_negative, total, false_positive_rate, false_negative_rate |
| v100 | dynamic_pattern_utility.csv | 1280 | 0 | task, phase, kv_type, layer, kv_head, bucket, metric, count, mean, min, max, std |
| gpu4090 | pattern_gain_map.csv | 15872 | 0 | task, phase, kv_type, layer, kv_head, bucket, metric, count, mean, std |
| gpu4090 | matching_oracle_gap.csv | 2160 | 0 | task, phase, kv_type, layer, kv_head, bucket, metric, count, mean, std |
| gpu4090 | v_gate_confusion.csv | 4608 | 0 | task, phase, kv_type, layer, kv_head, bucket, metric, true_positive, true_negative, false_positive, false_negative, total, false_positive_rate, false_negative_rate |
| gpu4090 | dynamic_pattern_utility.csv | 1536 | 0 | task, phase, kv_type, layer, kv_head, bucket, metric, count, mean, std |
