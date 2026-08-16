# Reconcile Full-Model Scaling Path + Attention Roofline V1

## Execution Discrepancy
The Profile V2 path and scaling path do not time the same request protocol. Profile V2 B1 uses `total_requests=1`, so the initial 2048-token prefill is outside the measured wall interval and no replacement request is admitted. Context/B scaling B1 uses `total_requests=2` with saturated FIFO replacement, so the second request prefill occurs inside the measured wall interval.

The controlled matrix reproduced decode-only latency near Profile V2: profile protocol decode4 averaged `208.835` ms/output-token, while profile protocol decode8 averaged `201.847` ms/output-token. Scaling protocol decode8 averaged `722.711` ms/output-token, decomposed into `199.136` ms decode-only plus `523.125` ms measured refill prefill per output token.

Execution discrepancy classification: `FULL_MODEL_EXECUTION_DISCREPANCY_CLOSED`. Explained percent: `100.0%`.

## Attention Diagnosis
The canonical production scaling path is `context=2048`, `B=1`, `decode=8`, `total_requests=2`, FIFO saturated replacement, fixed split enabled, active batch cache enabled. For decode-only attention under this path, the largest exposed attention component is FP16 sink/pending/recent Value tail, not compressed historical QK or fused page mixed Value.

Top attention components from the diagnostic profile:

| component | ms/iteration | % attention | % full decode |
|---|---:|---:|---:|
| FP16 Value tail | 42.198 | 31.2% | 15.8% |
| cache append + K/V quant/pack | 20.831 | 15.4% | 7.8% |
| fixed-split wrapper/merge | 17.895 | 13.2% | 6.7% |
| RoPE / position | 13.994 | 10.3% | 5.2% |
| full-precision sink/recent/pending QK | 9.351 | 6.9% | 3.5% |
| mixed V2/V4 Value | 5.992 | 4.4% | 2.2% |


Compressed QK is about 3.762 ms/iteration and fused page mixed Value is about 5.992 ms/iteration in the diagnostic profile. Fixed-split CUDA kernel time remains small at about 2.679 ms/iteration, though the wrapper is larger.

## Classification
- Task: `RECONCILE_FULL_MODEL_SCALING_PATH_AND_ATTENTION_ROOFLINE_V1_SUPPORTED`
- Attention diagnosis: `MULTI_COMPONENT`
- Primary correction: prior scaling throughput included measured refill prefill, so decode-only scaling must be repaired before selecting a kernel optimization based on those throughput curves.
- Next task: `FULL_MODEL_SCALING_DECODE_ONLY_PROTOCOL_REPAIR_V1`
