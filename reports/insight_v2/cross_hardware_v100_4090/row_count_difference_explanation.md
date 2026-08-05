# Row Count Difference Explanation

The extra 4090 rows are not random noise.

- `pattern_gain_map.csv`: `512` 4090-only rows, all in `passage_retrieval_zh` / `decode` / `kv_type=k`.
- `dynamic_pattern_utility.csv`: `256` 4090-only rows, all in `passage_retrieval_zh` / `decode` / `kv_type=v`.
- `matching_oracle_gap.csv`: exact key-set match across hardware.
- `v_gate_confusion.csv`: exact key-set match across hardware.

The summary CSVs do not contain sample_id/problem_id fields, so a true per-sample attribution table cannot be reconstructed from these files alone.
