# Segmented Heterogeneous Attention State Merge V1

## Classification

`SEGMENTED_HETEROGENEOUS_ATTENTION_STATE_MERGE_V1_NOT_SUPPORTED_AS_RUNTIME_OPTIMIZATION`

## Required Answers

1. Eliminated global intermediates: global score concat and global full normalized probability tensor on the rolling segmented-state path.
2. Production path still concatenates all logits: no, when `PATTERNKV_SEGMENTED_STATE_MERGE=1` and cache mode is rolling.
3. Production path still materializes one full global normalized probability tensor: no.
4. State ABI: `SegmentedAttentionState(o:[B,Hq,Q,D] fp32, m:[B,Hq,Q] fp32, l:[B,Hq,Q] fp32)`.
5. Merge order: sink -> packed/history -> pending -> recent.
6. Merge exactness: mathematically exact online-softmax state merge.
7. Floating-point differences: reduction order changes; bitwise equality is not required. FP32 oracle tests pass at `rtol=1e-5, atol=1e-5`.
8. B1/B2/B4/ragged correctness: PASS in targeted tests.
9. Request lifecycle/add-remove/continuous batching: PASS in targeted tests.
10. Historical FP16 K/V materialization: zero.
11. Kernel/range count decrease: global concat/probability calls drop from 256/256 to 0/0, but segment-level calls rise to 1024 state/value/softmax calls and 768 merges.
12. Temporary tensor traffic: global tensors decrease, but segment-local probability/state traffic remains and causal-importance work increases.
13. Old vs new formal C2048 B1 TPOT: old 191.697 ms/token; best new repeat 281.952 ms/token, with another new run at 384.408 ms/token.
14. Measured speedup: best observed 0.680x vs old.
15. C4096 B8 capacity: not run; primary B1 performance regressed, so scaling/capacity expansion stopped.
16. Largest post-change bottleneck: state-path integration overhead, especially segment-local softmax/state generation and per-segment causal-importance updates.
17. Next task: `ATTENTION_KERNEL_LAUNCH_FUSION_V1` or `STOP_STATE_MERGE_OPTIMIZATION`; evidence does not support continuing Python-level state merge integration.

## Stop/Go

`STOP` for this V1 runtime path. The abstraction is semantically valid, but the V1 implementation does not materially reduce full-model decode TPOT.

