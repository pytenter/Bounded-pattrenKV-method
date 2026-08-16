# Formal Closure

## Answers

1. Physical RTX 3090 used: GPU2.
2. GPU clean: yes, GPU2 had 20 MiB used, 0% utilization, and no compute processes before formal runs.
3. OLD and FUSED measured on same GPU: yes, physical GPU2.
4. Runs interleaved: yes, OLD/FUSED pairs for three measured pairs.
5. Timed-window gates zero: yes for OLD and FUSED: prefill calls/tokens, refill calls, membership changes, and page-pack calls were all `0`.
6. Full-model top1 matched every tested decode step: yes, 8/8 steps.
7. Logits max_abs and relL2: max over steps `0.5859375` and `0.022798627614974976`.
8. OLD median TPOT: `242.4437877489254 ms/token`.
9. FUSED median TPOT: `199.60261625237763 ms/token`.
10. Absolute saved: `42.84117149654776 ms/token`.
11. Speedup: `1.214632314450124x`.
12. Percent TPOT reduction: `17.67055856300762%`.
13. FP16-tail kernels/token: old inferred matched decode-only CUDA profile `2272.0`; fused named kernel count `32.0`.
14. Total kernels/token: old `12887.0`; fused `10647.0`.
15. Attention kernels/token: old prior attention forensic `4768`; fused estimated `2528` after subtracting the matched tail kernel reduction from the same attention envelope.
16. FP16-tail GPU time: old profile-range median `42.69132802262902 ms/token`; fused profile-range median `2.154880004003644 ms/token`.
17. Peak allocated/reserved memory regression: no allocated regression; old/fused max peak allocated both `16513622016`, reserved old `16758341632`, fused `16756244480`.
18. C4096 B8 still passed: yes with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
19. New largest runtime component: diffuse; fused profile still has self-attention aggregate around `110.9 ms/token`, MLP around `40.8 ms/token`, post-attention RMSNorm around `33.2 ms/token`, cache append around `18.8 ms/token`, and softmax around `14.5 ms/token`.
20. Throughput engineering should now stop: yes.

## Classification

`FP16_TAIL_VALUE_LAUNCH_FUSION_V1_SUPPORTED`

## Notes

Formal TPOT is the non-profiler decode-only harness result. CUDA profiler results are reported separately and are not used as serving TPOT.
