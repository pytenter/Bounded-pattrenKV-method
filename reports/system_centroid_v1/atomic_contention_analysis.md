# Atomic Contention Analysis

## Setup

Representative V2 workload: context 16K, all quantized V2 tokens, mask density 100%, `Mcent=16`, `nh=32`, `nh_kv=8`, GQA ratio 4. Timings use CUDA events with 5 rounds, 30 warmup iterations, and 200 timed iterations per round.

## Assignment Distribution

| Distribution | Active assignments | Top centroid | Top count | Max fraction | Entropy | Mean tokens/centroid |
|---|---:|---:|---:|---:|---:|---:|
| RANDOM_UNIFORM | 129024 | 10 | 8188 | 0.0635 | 3.9999 | 8064.0 |
| SKEWED | 129024 | 0 | 64494 | 0.4999 | 2.9539 | 8064.0 |

## Timing

| Distribution | FULL median us | Slowdown vs uniform |
|---|---:|---:|
| RANDOM_UNIFORM | 335.872 | 1.000x |
| SKEWED | 598.016 | 1.780x |

## Interpretation

`atomicAdd(&s_Sacc[idx], alpha)` is structurally sensitive to assignment concentration. The skewed workload sends about 50.0% of active assignments to one centroid and slows the V2 FULL kernel by 78.0% versus uniform. This supports `atomic_contention_supported=YES` for centroid-path optimization planning.
