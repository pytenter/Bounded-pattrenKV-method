# Final Report

Classification: `FULL_MODEL_INTEGRATION_OVERHEAD_OPTIMIZATION_V1_SUPPORTED`. The matched context=2048, B=1, decode=4 after-profile reduced decode wall TPOT from `788.231` ms/token to `316.277` ms/token. Harness assemble fell from `119.335` to `0.182` ms/token and split fell from `162.619` to `0.017` ms/token. Row-slice copy bytes fell from `221769792` bytes/token to `0`.

Centroid physical capacity now sizes to the required active slot-id range and context-derived dynamic budget with geometric growth. For the matched B=1 profile, formula capacity falls from `1140850688` bytes to `7471104` bytes while preserving centroid counts, values, reset, and grow semantics.

Remaining top blocker is model decode at `304.606` ms/token, so the next task is `OPTIMIZE_REQUEST_INVARIANT_FIXED_SPLIT_SOFTMAX`. The full model serving benchmark remains `PATTERNKV_FULL_MODEL_SERVING_BENCHMARK_V1_NOT_CLOSED`.

Derived metrics: `{"ASSEMBLE_REDUCTION": 0.9984736749406992, "CENTROID_CAPACITY_REDUCTION_FORMULA": 0.9934512867647058, "PEAK_ALLOCATED_REDUCTION": 0.07668455938100438, "ROW_SLICE_BYTE_REDUCTION": 1.0, "SPLIT_REDUCTION": 0.999892952236176, "TOTAL_RUNTIME_SPEEDUP": 2.4922162565926134}`
