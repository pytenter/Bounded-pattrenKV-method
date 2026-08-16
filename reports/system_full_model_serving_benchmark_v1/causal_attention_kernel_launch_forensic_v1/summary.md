# Summary

1. Profiler backend: PYTORCH_PROFILER.
2. Actual CUDA kernels per CAUSAL output token: 12599.000.
3. Attention kernels per layer: 149.000.
4. Tiny kernels: <5us=97320, <10us=98216, <20us=98728, <50us=99665.
5. Tiny launch fraction <10us: 0.974442.
6. Tiny GPU time fraction <10us: 0.322601.
7. Top kernels by GPU time: see `kernel_inventory.md`; #1 is `_bi_linear_persistent_kernel`.
8. GPU idle/launch gap per token: 324.546 ms approximate same-stream positive gaps.
9. Explicit synchronizations in decode hot path: benchmark-only `torch.cuda.synchronize()` after each decode step; no production semantic sync removal attempted.
10. FP16 Value tail: launch/orchestration-dominated.
11. FP16 QK tail: launch/orchestration-dominated.
12. Cache append: orchestration-dominated.
13. Fixed-split softmax: wrapper dominated.
14. Explained CAUSAL TPOT fraction: 1.000000.
15. Best launch-fusion Amdahl bound: 2.314x, TPOT 82.833 ms/token.
16. ATTENTION_KERNEL_LAUNCH_FUSION_V1 supported: yes.
17. Exact first fusion target: FP16_TAIL_VALUE_LAUNCH_FUSION.
18. New root cause if no: NOT_APPLICABLE.
19. Further throughput optimization scientifically worthwhile: yes, one targeted task.
20. Project decision: ONE_FINAL_TARGETED_OPTIMIZATION.
