# Final Report

The fixed split CUDA softmax path was implemented and measured, but this round is **not closed** because the opt-in path failed the ragged hard gate. It is disabled by default to preserve production correctness.

Opt-in B1 A/B at context=2048, B=1, decode=4: total `317.907` -> `232.278` ms/token, model decode `306.253` -> `220.533` ms/token, softmax `102.694` -> `15.806` ms/token. Derived opt-in speedups: `{"CAUSAL_VS_FP16_GAP_AFTER_OPT_IN": 4.020533992931963, "MODEL_DECODE_SPEEDUP_OPT_IN": 1.3812288473706698, "SOFTMAX_REDUCTION_SPEEDUP_OPT_IN": 6.496971369673773, "TOTAL_SPEEDUP_OPT_IN": 1.3616295016577993}`.

Semantic blocker: opt-in ragged hard gate failed B2 reorder, B4, and independent flush. The disabled-kernel control also failed B4/flush in the current dirty tree, but the opt-in path worsened B2 reorder and max logit drift, so it cannot be production-enabled.

B>1 low-copy pre-gate: B2/B4 full-model serving still shows layer metadata rebuilds and row-slice bytes, so `MULTI_REQUEST_LOW_COPY_GENERALIZATION_PENDING` remains true for the serving harness.

Next task: `FIX_FIXED_SPLIT_DETERMINISTIC_MERGE_SEMANTICS`.
