# Implementation Semantics Audit

- `models/llama_patternkv.py:_nearest_v_centroid`
  - Current behavior: V prefill runtime assignment is min-max nearest centroid over head-dim token vectors.
  - Required change: none for inference; targeted run adds observer-only L2-vs-min-max aggregates.

- `models/llama_patternkv.py:forward`
  - Current behavior: K prefill bank is L2-mined and K runtime prefill assignment is tokenwise L2 nearest centroid.
  - Required change: diagnostics keep that runtime path and aggregate onto real K quantization units: post-RoPE `[B,H,T,D]`, transpose to `[B,H,D,T]`, 128-token groups, per-channel residual range.

- `insight/hook_metrics.py`
  - Current behavior: range-aware evidence was mostly recoverable from bounded sample records, which truncated on 4090 Wave A.
  - Required change: add complete aggregate-only path for targeted collection.

- `insight/collector.py`
  - Current behavior: sample records were always allowed when observer was enabled.
  - Required change: allow `sample_records_enabled=false` while keeping aggregate collection complete.

- `insight/range_aware_metrics.py`
  - Added pure chunk-stable helpers for V per-token diagnostics and K grouped per-channel diagnostics.

