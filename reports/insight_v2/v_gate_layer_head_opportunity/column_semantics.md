# Column Semantics

| column | meaning | status | evidence |
| --- | --- | --- | --- |
| task | LongBench/GSM8K task label; used as the primary task dimension. |  | parsed directly from CSV. |
| phase | Model phase; summary CSV is prefill-only for this confusion table. |  | parsed directly from CSV. |
| kv_type | V gate only; this CSV records V-side gate-vs-oracle confusion. |  | parsed directly from CSV. |
| layer | Transformer layer index. |  | parsed directly from CSV. |
| kv_head | Key/value head index. |  | parsed directly from CSV. |
| bucket | Position bucket derived from sampled token position (first/middle/last). |  | present in CSV; populated by _bucket(token, total) in insight/hook_metrics.py. |
| metric | Fixed summary metric name; gate_vs_mse_oracle. |  | fixed by summarize_insight_wave_a_8gpu.py gate_vs_mse_oracle filter. |
| true_positive | Count of sampled tokens where gate accepted and oracle said pattern was better. |  | record_prefill_v_metrics() adds confusion counts via observer.add_confusion(). |
| true_negative | Count of sampled tokens where gate rejected and oracle said pattern was worse. |  | record_prefill_v_metrics() adds confusion counts via observer.add_confusion(). |
| false_positive | Count of sampled tokens where gate accepted but oracle said raw was better. |  | record_prefill_v_metrics() adds confusion counts via observer.add_confusion(). |
| false_negative | Count of sampled tokens where gate rejected but oracle said pattern was better. |  | record_prefill_v_metrics() adds confusion counts via observer.add_confusion(). |
| total | TP+TN+FP+FN; row-level candidate support for the sampled tokens. |  | summarizer reconstructs this from TP/TN/FP/FN counters. |
| false_positive_rate | FP / (FP + TN), blank when denominator is zero. |  | summarizer computes fp / (fp + tn). |
| false_negative_rate | FN / (FN + TP), blank when denominator is zero. |  | summarizer computes fn / (fn + tp). |
