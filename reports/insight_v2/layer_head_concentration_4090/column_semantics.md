# Column Semantics

| column | inferred type | notes |
| --- | --- | --- |
| task | categorical | dataset/task id |
| phase | categorical | `prefill` or `decode` |
| kv_type | categorical | `k` or `v` |
| layer | integer | transformer layer index |
| kv_head | integer | KV head index |
| bucket | categorical | bucket partition label |
| metric | categorical | metric name |
| count | integer | sample count for the row |
| mean | float | aggregate mean value |
| std | float | aggregate standard deviation |

Main analysis phase: `prefill`
Main K metric: `relative_benefit`
V main metric: `relative_benefit`
V proxy metric: `relative_candidate_benefit`
