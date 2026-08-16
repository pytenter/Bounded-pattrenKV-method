# Full Model Kernel Before After

## Decode-Only CUDA Profiler

The CUDA profiler was run with C2048 B1 decode=8 after prefill was completed outside the profiler. This keeps profiler attribution separate from formal serving TPOT.

| metric | old | fused |
|---|---:|---:|
| total CUDA kernels/token | 12887.0 | 10647.0 |
| kernel busy ms/token | 69.82833312499655 | 69.69408700000533 |
| tiny kernels/token | 12752.625 | 10507.75 |
| tiny-kernel fraction | 0.9895728253278497 | 0.98692119845966 |
| named fused FP16-tail kernels/token | 0.0 | 32.0 |

## Tail Attribution

The fused tail kernel is directly named by the CUDA profiler and appears `32.0` times/token, matching `32 layers * 1 fused call/layer/token`.

The old tail path is implemented through generic PyTorch elementwise/reduction kernels, so it is not directly name-attributed as one old-tail kernel family. Matched full-model total kernel delta plus the fused named kernel count gives:

- inferred old FP16-tail kernels/token: `2272.0`
- fused FP16-tail kernels/token: `32.0`
- tail kernel reduction: `98.59154929577466%`

This is consistent with the prior full forensic old estimate of about `2560` tail kernels/token and the direct-tail probe reduction from `1472.0` to `32.0`.

## Component CUDA-Event Timing

Profile-range CUDA event timing from the formal harness:

- old FP16-tail median: `42.69132802262902 ms/token`
- fused FP16-tail median: `2.154880004003644 ms/token`
- tail GPU/event speedup: `19.810072545899127x`

## Attention Kernel Estimate

Prior old attention kernels/token: `4768`.

Subtracting the matched tail kernel reduction from that same attention envelope gives fused estimated attention kernels/token: `2528`.

This estimate is reported separately from directly measured total CUDA kernels/token because the decode-only CUDA profiler did not attach old generic PyTorch tail kernels to attention ranges.
