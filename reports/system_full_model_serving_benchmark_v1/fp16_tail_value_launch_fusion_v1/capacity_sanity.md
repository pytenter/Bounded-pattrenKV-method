# Capacity Sanity

`C4096_B8 = PASS`

The first attempt without allocator configuration OOMed due allocator fragmentation. The retry used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, matching prior capacity-sensitive scripts, and passed.

## Result

- context: `4096`
- batch: `8`
- decode: `8`
- output tokens: `64`
- run valid: `true`
- page batch pack calls in timed window: `0`
- prefill calls/tokens in timed window: `0`
- refill calls in timed window: `0`
- membership changes in timed window: `0`
- peak allocated: `22432702464`
- peak reserved: `23551016960`
- fused fallback count: `0`

No capacity regression is observed under the allocator configuration used for the prior capacity path.
