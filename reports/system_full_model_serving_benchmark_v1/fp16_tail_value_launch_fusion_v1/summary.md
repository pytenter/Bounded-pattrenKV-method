# Summary

1. Old FP16 tail Value path sliced the global probability tensor for `sink`, `pending`, and `recent`, ran `request_invariant_full_value_attention` once per non-empty segment, then added the segment outputs.
2. With all tail segments present, old logical tail execution had three separately dispatched Value operators plus two output adds.
3. Fused ABI: `fp16_tail_value_forward_cuda(probs, sink_v, pending_v, recent_v, sink_lens, pending_lens, recent_lens, offsets, num_key_value_groups) -> [B,Hq,1,D]`.
4. Sink, pending, and recent are accumulated directly into one output.
5. Three intermediate output tensors are eliminated in the fused path.
6. No new probability or Value concatenation is materialized.
7. GQA mapping is preserved as `kv_head = query_head // num_key_value_groups`.
8. Ragged lengths are supported through request-local `[B]` valid-length tensors.
9. True batching is preserved; the kernel grid covers `[B,Hq]` and has no Python per-request production loop.
10. Historical mixed-V is untouched.
11. Fixed-split global softmax is untouched.
12. Old prior forensic FP16-tail kernel count/token was approximately `2560`; direct non-formal synthetic old-tail probe was `1472.0`.
13. Fused direct non-formal synthetic FP16-tail kernel count/token was `32.0`.
14. Old direct non-formal FP16-tail GPU compute was `3.3549025000002954 ms/token`; fused direct was `0.4791142499999999 ms/token`.
15. Old vs fused total full-model kernels/token is `12887.0` vs `10647.0`.
16. Old vs fused formal TPOT median is `242.4437877489254` vs `199.60261625237763 ms/token`.
17. Actual formal ms/token saved is `42.84117149654776`.
18. Capacity passed at C4096 B8 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; allocated memory did not regress in C2048 B1 formal runs.
19. Post-fusion runtime remains diffuse across self-attention, MLP, RMSNorm, cache append, softmax, and QK regions.
20. One more targeted optimization is not scientifically justified after formal closure.
21. Project decision: `STOP_THROUGHPUT_ENGINEERING_AND_FREEZE` until clean formal evidence is available.

## Classification

`FP16_TAIL_VALUE_LAUNCH_FUSION_V1_SUPPORTED`
