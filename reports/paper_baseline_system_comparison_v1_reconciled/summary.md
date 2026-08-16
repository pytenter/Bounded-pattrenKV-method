# Summary

PAPER_BASELINE_SYSTEM_COMPARISON_V1_RERUN_WITH_RECONCILED_ALLOCATOR_PROTOCOL is supported. The same-GPU formal comparison contains 119 valid rows across FP16, KIVI-paper-g128, PatternKV-paper, and CAUSAL-V4@25%. All valid rows preserve the reconciled allocator protocol, true batch, zero serial dispatch, zero fallback, decode-only timing, and subprocess isolation. The highest observed C4096 capacity is `KIVI_PAPER_G128_FULL_MODEL` at B8. CAUSAL C2048/B2 includes one elevated protocol-valid repeat; raw rows and the median-based primary statistic are retained in `anomaly_audit.md`.
