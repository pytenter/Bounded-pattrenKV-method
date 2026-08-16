# Full-Model Context Scaling V2

## Protocol
- Physical GPU: 5, NVIDIA GeForce RTX 3090, isolated before and after run.
- Workload: B=1, decode=8, total_requests=2, FIFO saturated steady state.
- Contexts: 256, 2048, 4096, 8192.
- Methods: FP16_FULL_MODEL and optimized CAUSAL_V4_25_FULL_MODEL.
- Headline timing: profiling off, 1 warmup and 3 measured runs per valid point.

## Throughput
| context | FP16 tok/s | CAUSAL tok/s | CAUSAL/FP16 | FP16 status | CAUSAL status |
|---:|---:|---:|---:|---|---|
| 256 | 28.559 | 3.326 | 0.116 | PASS | PASS |
| 2048 | 10.471 | 1.494 | 0.143 | PASS | PASS |
| 4096 | 6.815 | 0.870 | 0.128 | PASS | PASS |
| 8192 |  |  |  | OOM | OOM |

## Memory
| context | FP16 allocated GB | CAUSAL allocated GB | FP16-CAUSAL allocated GB | FP16 reserved GB | CAUSAL reserved GB |
|---:|---:|---:|---:|---:|---:|
| 256 | 16.372 | 16.395 | -0.023 | 16.609 | 16.597 |
| 2048 | 18.469 | 17.927 | 0.542 | 19.319 | 18.438 |
| 4096 | 20.867 | 19.678 | 1.189 | 22.664 | 20.793 |
| 8192 | 23.514 | 22.814 | 0.700 | 23.763 | 23.385 |

## Findings
- Throughput ratio trend across valid matched points: `NON_MONOTONIC`.
- Allocated-memory crossover observed: `True`; first observed allocated crossover context: `2048`.
- Reserved-memory crossover observed: `True`; first observed reserved crossover context: `256`.
- Both FP16 and CAUSAL OOM at 8192 under the recovered decode=8,total_requests=2 protocol; 16K was not attempted.
- CAUSAL structural gates are clean for every valid CAUSAL context: one plan per iteration, no layer metadata rebuild, no row-slice bytes, no fallback, no historical FP16 KV materialization.

## Historical Directional Comparison
| context | old CAUSAL tok/s | new CAUSAL tok/s | new/old | old ratio | new ratio |
|---:|---:|---:|---:|---:|---:|
| 256 | 2.968 | 3.326 | 1.12x | 0.105 | 0.116 |
| 2048 | 1.002 | 1.494 | 1.49x | 0.091 | 0.143 |
| 4096 | 0.612 | 0.870 | 1.42x | 0.091 | 0.128 |

## Classification
`FULL_MODEL_CONTEXT_SCALING_V2_SUPPORTED`. Next task: `FULL_MODEL_B_CONCURRENCY_SCALING_V1`.

## Regression Closure

- compileall modified Python files: PASS
- targeted semantic/regression gates: 117 passed
- full pytest: 1008 passed
- git diff --check: PASS

## STOP / GO

`STOP_CONTEXT_SCALING_COMPLETE`: the matched B=1 context-scaling curve is measured through the valid range, 8192 OOMs are recorded honestly, CAUSAL structural invariants remain clean, and the next task is `FULL_MODEL_B_CONCURRENCY_SCALING_V1`.
