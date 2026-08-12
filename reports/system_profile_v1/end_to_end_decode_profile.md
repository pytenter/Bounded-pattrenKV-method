# Phase S1.5 End-to-End Decode Profiling

## Frozen algorithm

- Frozen commit: `c73aeed3247c136859f695d5b238eeb357434b17`
- Frozen tag: `causal-v4-25-aime24-v1`
- Algorithm changed in this phase: `NO`

## S1 kernel status

- S1 correctness: `56/56`
- Kernel speedup @16K: `2.699x`
- E2E speedups below are separate from kernel microbenchmark speedups.

## Profiling methodology

- Decode-focused synthetic cache workload using the real DeepSeek-R1-Distill-Llama-8B PatternKV model.
- Each case runs `q_len=1` decode with seeded legal PatternKV cache tensors matching the frozen cache ABI.
- E2E TPOT is measured with `PATTERNKV_PROFILE=0`; component breakdown is measured in a separate `PATTERNKV_PROFILE=1` pass.
- CUDA events aggregate component timing; ranges are nested, so component percentages are diagnostic shares, not an exclusive flamegraph.
- Short real-model smoke is kept separate and is not used as a throughput benchmark.
- `prefill_ms` in `e2e_summary.csv` is synthetic cache construction/setup time, not full dense prefill latency; this phase is decode-focused.

## Reference vs fused E2E results

| T | reference TPOT ms | fused TPOT ms | E2E speedup |
|---:|---:|---:|---:|
| 2048 | 124.068 | 105.174 | 1.180 |
| 4096 | 111.644 | 106.089 | 1.052 |
| 8192 | 116.930 | 108.063 | 1.082 |
| 16384 | 136.194 | 113.052 | 1.205 |
| 32768 | 193.148 | 118.019 | 1.637 |

## Component breakdown

Fused backend @T=32768:

| Component group | Share of decode wall time |
|---|---:|
| mixed_v | 33.22% |
| cache_mutation | 20.46% |
| QK | 15.50% |
| importance_update | 3.78% |
| output_projection | 2.65% |
| packing | 1.78% |
| softmax | 1.30% |
| selector | 1.10% |
| lm_head | 1.01% |

## Context-length scaling

Scaling ratios are available in `scaling_analysis.csv` and grouped in `scaling_summary.csv`.

Fused grouped scaling:

| Component | 2K->4K | 4K->8K | 8K->16K | 16K->32K | Classification |
|---|---:|---:|---:|---:|---|
| QK | 1.026 | 1.033 | 0.999 | 1.058 | constant-like |
| softmax | 0.998 | 0.971 | 0.979 | 1.049 | constant-like |
| mixed V | 1.011 | 1.107 | 1.154 | 1.251 | constant-like to sublinear |
| importance update | 0.992 | 0.993 | 1.001 | 1.007 | constant-like |
| selector | 1.019 | 0.979 | 0.990 | 1.022 | constant-like |
| packing | 1.008 | 0.986 | 0.999 | 1.016 | constant-like |
| cache mutation | 1.019 | 0.986 | 1.022 | 0.986 | constant-like |

## Selector cost

- Selector share @T=32768: `1.10%`

## Causal-importance-update cost

- Importance update share @T=32768: `3.78%`

## Packing cost

- Packing share @T=32768: `1.78%`

## Cache mutation / torch.cat cost

- Cache mutation share @T=32768: `20.46%`
- Runtime cache concat events @T=32768: `16544`
- Approximate bytes copied @T=32768: `4057821184`
- Largest single concat input footprint @T=32768: `9400320`
- Static audit: `cache_mutation_audit.md`

## Mixed fused kernel share

- Mixed fused/reference Value share @T=32768: `33.22%`

## Real model smoke

- AIME-style prompt count: `1`
- max_new_tokens: `512`
- Runtime error: `none`
- NaN/Inf: `false`
- mixed_v_fused_calls: `12608`
- mixed_v_reference_calls: `0`
- Profile records generated: `true`

## GPU kernel profile

- NSYS_AVAILABLE=`false`
- NCU_AVAILABLE=`false`

## Memory behavior

Runtime memory peaks are recorded in `e2e_summary.csv`.

## Profiling overhead

- Profile off TPOT: `105.074 ms`
- Profile on TPOT: `135.746 ms`
- Overhead: `29.19%`

## Dominant bottleneck

1. mixed_v: 33.22%
2. cache_mutation: 20.46%
3. QK: 15.50%
4. importance_update: 3.78%
5. output_projection: 2.65%

## Recommended next systems phase

- `PROFILE_INCONCLUSIVE`
- Reason: mixed_v dominates, but this phase did not authorize a matching optimization path
