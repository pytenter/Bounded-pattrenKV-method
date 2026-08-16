# Summary

1. Can the current CAUSAL decode path be CUDA-Graph captured? blocked
2. Full decode graph or piecewise graph? FULL_DECODE_SEQUENCE, one full decode graph per generated token in the fixed horizon.
3. What prevented full capture, if anything? Host `.item()` reads in softmax and Value tail initially blocked capture; fixed-step dynamic cache shapes prevent one reusable single graph.
4. Static addresses: token buffers and graph-owned cache/output tensors.
5. Metadata updated between replays: initial cache tensor values restored before replay; output cache tensors are produced by prior graph steps.
6. Position IDs updated correctly: captured per fixed step; not general for arbitrary longer decode without more captured steps.
7. KV valid lengths updated correctly: True.
8. Cache write indices updated correctly: True.
9. Centroid counts updated correctly: True.
10. Multi-step graph replay matches eager semantics: False.
11. Historical FP16 K/V materialization still zero: yes.
12. Graph capture latency: 2615.918575087562 ms.
13. Eager TPOT: 188.357206992805.
14. Graph replay TPOT: 55.18938950262964.
15. Formal speedup: 3.473439400717888 measured but invalid because correctness failed.
16. GPU idle gaps: NOT_AVAILABLE post-graph on clean GPU1.
17. CPU launch API overhead: graph replay submissions/token = 1, eager launch API calls/token = 12503 from prior forensic.
18. Physical GPU kernel count changed: no expected kernel fusion; count not formally reprofiled here.
19. Graph memory overhead: 31457280 reserved bytes.
20. C4096 B8 capacity threat: not_run.
21. New dominant bottleneck: CUDA Graph replay correctness, then prior Value-tail launch fragmentation if continuing optimization.
22. Next step: FP16_TAIL_VALUE_LAUNCH_FUSION_V1.
