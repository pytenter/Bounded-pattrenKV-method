# Post Fusion Bottleneck

Formal post-fusion attribution shows the fusion translated into end-to-end TPOT improvement, but the remaining runtime is diffuse rather than dominated by one narrow launch-fragmented tail.

Largest fused components from profile-range CUDA events include:

- self-attention aggregate: about `110.9 ms/token`
- MLP: about `40.8 ms/token`
- post-attention RMSNorm: about `33.2 ms/token`
- cache append: about `18.8 ms/token`
- softmax: about `14.5 ms/token`
- FP16 QK regions: about `12.7 ms/token`

No single next target satisfies the high-Amdahl, low-risk, narrow-fix rule. Throughput engineering should stop and the project should freeze for final serving benchmarks.
