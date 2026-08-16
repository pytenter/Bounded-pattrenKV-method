# Full-Model Post-Optimization Bottleneck Profile V2

## Workload
- CAUSAL-V4@25% PatternKV full-model decode
- Context 2048, decode 4, B1 and B4
- Fixed-split softmax enabled, logical split 128
- Formal measurements rerun on physical GPU 5 after GPU 1 contamination

## Headline Timing
- B1: 217.182 ms/iteration, 4.606 output tok/s, peak allocated 17.780 GB
- B4: 268.629 ms/iteration, 67.157 ms/output-token, 14.892 output tok/s, peak allocated 22.863 GB
- Instrumentation overhead: B1 34.5%, B4 11.6%
- Profile accounting coverage: B1 98.4%, B4 98.3%

## Ranked Decode Components
| component | B1 ms/iter | B1 % | B4 ms/iter | B4 % | B4 ms/output-token | scaling |
|---|---:|---:|---:|---:|---:|---|
| Attention aggregate | 185.639 | 66.2% | 165.282 | 64.3% | 41.320 | NEAR_CONSTANT_PER_ITERATION |
| RMSNorm aggregate | 46.133 | 16.5% | 48.179 | 18.7% | 12.045 | NEAR_CONSTANT_PER_ITERATION |
| MLP aggregate | 41.579 | 14.8% | 36.774 | 14.3% | 9.193 | NEAR_CONSTANT_PER_ITERATION |
| LM head | 1.192 | 0.4% | 1.214 | 0.5% | 0.304 | NEAR_CONSTANT_PER_ITERATION |
| Residual adds | 1.184 | 0.4% | 1.106 | 0.4% | 0.276 | NEAR_CONSTANT_PER_ITERATION |
| Harness assemble/control | 0.247 | 0.1% | 0.143 | 0.1% | 0.036 | NEAR_CONSTANT_PER_ITERATION |

## Attention Sub-Breakdown
| component | B4 ms/iter | B4 ms/output-token | % attention | notes |
|---|---:|---:|---:|---|
| Mixed V FP16 tail | 37.609 | 9.402 | 22.8% | inclusive nested timer |
| Cache append/update wrapper | 23.529 | 5.882 | 14.2% | inclusive nested timer |
| RoPE/position | 22.469 | 5.617 | 13.6% | inclusive nested timer |
| Fixed-split softmax wrapper | 14.747 | 3.687 | 8.9% | inclusive nested timer |
| FP16 sink/pending/recent QK | 8.026 | 2.007 | 4.9% | inclusive nested timer |
| Compressed historical QK | 4.954 | 1.238 | 3.0% | inclusive nested timer |
| Importance update | 4.328 | 1.082 | 2.6% | inclusive nested timer |
| Output projection | 3.414 | 0.853 | 2.1% | inclusive nested timer |
| Attention score concat | 3.282 | 0.821 | 2.0% | inclusive nested timer |
| Q projection | 3.178 | 0.794 | 1.9% | inclusive nested timer |
| Fixed-split CUDA kernel | 2.486 | 0.622 | 1.5% | inclusive nested timer; fixed_split_cuda_kernel is included in softmax wrapper |
| K projection | 2.018 | 0.504 | 1.2% | inclusive nested timer |

## Diagnosis
The #1 bottleneck after low-copy integration is the compressed attention datapath as a whole, not fixed-split softmax alone. Within exposed attention children, the largest B4 decode items are mixed value FP16 tail aggregation, V/K/Q projections, cache append/update, and fixed-split softmax wrapper. The fixed-split CUDA kernel itself is only about 2.486 ms/iteration, so it is not materially dominant.

The prior integration pathologies remain eliminated: B1/B4 have one iteration-plan build per decode iteration, zero layer metadata rebuilds, zero row-slice bytes, zero fallback, zero historical FP16 K/V materialization, and true batched dispatch. B4 iteration latency is close to B1 for this workload, while output-token latency is much lower.

Centroid/importance/update counters are visible but not dominant in steady decode. Peak memory is high for B4, but no new large temporary allocation or allocator pathology was isolated beyond expected batch/cache footprint.

## Next Task
`FULL_MODEL_CONTEXT_SCALING_V2`

## Final Scientific Answers

1. The #1 post-optimization bottleneck is the compressed attention datapath aggregate.
2. It is mostly per-iteration and only partially increases from B1 to B4 in this workload.
3. It is expected data-plane work, not recurrence of the old serving integration tax.
4. The B>1 low-copy gains are still present: metadata rebuilds and row-slice bytes remain zero.
5. B4 preserves near-B1 iteration latency for a full-model path: 268.629 ms vs 217.182 ms, while output-token latency drops to 67.157 ms/token.
6. Fixed-split softmax is not materially dominant: B4 softmax wrapper is 14.747 ms/iteration and the CUDA kernel is 2.486 ms/iteration.
7. Importance update is not significant at 4.328 ms/iteration on B4; centroid update did not appear as a steady decode top component.
8. MLP is third, not dominant; LM head is negligible.
9. No new large temporary/workspace allocator problem was isolated; B4 peak allocation is high but consistent with batch/cache footprint.
10. Next task should be `FULL_MODEL_CONTEXT_SCALING_V2`, not another immediate targeted optimization.
