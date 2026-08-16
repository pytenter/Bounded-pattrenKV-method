# Full-Model B / Concurrency Scaling V1

## Matched-B Scaling, Context 2048
| B | FP16 tok/s | CAUSAL tok/s | CAUSAL/FP16 | FP16 peak GB | CAUSAL peak GB | status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 11.371 | 1.486 | 0.131 | 18.469 | 17.927 | FP16 PASS, CAUSAL PASS |
| 2 | 13.921 | 2.343 | 0.168 | 20.869 | 19.706 | FP16 PASS, CAUSAL PASS |
| 4 |  | 3.237 |  | 23.514 | 23.381 | FP16 OOM, CAUSAL PASS |
| 8 |  |  |  | 22.554 | 21.216 | FP16 OOM, CAUSAL OOM |

## Capacity Scaling, Context 4096
| method | max successful B | first OOM B | tok/s at max B | iteration ms at max B | ms/output-token at max B | peak allocated GB | peak reserved GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16_FULL_MODEL | 1 | 2 | 6.894 | 145.051 | 145.051 | 20.867 | 22.664 |
| CAUSAL_V4_25_FULL_MODEL | 2 | 4 | 1.293 | 1547.122 | 773.561 | 23.143 | 24.709 |

## Findings
- Matched common max B: 2; CAUSAL/FP16 throughput at that B: 0.168x.
- FP16 max B at context 4096: 1; CAUSAL max B: 2; capacity ratio: 2.00x.
- Own-max throughput ratio CAUSAL/FP16: 0.188x.
- Memory advantage translates to concurrency: SUPPORTED; concurrency translates to throughput: NOT_SUPPORTED.
- CAUSAL structural gates remain clean at every successful CAUSAL point.

## Regression Closure
- Compileall: PASS for modified benchmark/instrumentation files.
- Targeted regressions: PASS, 117 passed.
- Full pytest: PASS, 1008 passed.
- Git diff check: PASS.

## Classification
`FULL_MODEL_B_CONCURRENCY_SCALING_V1_SUPPORTED`. Next task: `COMPRESSED_ATTENTION_DATAPATH_OPTIMIZATION_V1`.
