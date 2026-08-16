# Full Model Context Scaling Probe

## Executive Summary

The full-model benchmark remains not closed. The closed correctness gates were preserved, but the context-scaling probe does not support moving to a formal full-model capacity sweep yet.

The requested `decode=16` probe failed immediately for `CAUSAL_V4_25_FULL_MODEL` at `context=256`, `B=1`, `total_requests=2` with CUDA OOM. The probe therefore followed the fallback rule and used `decode=8` for both FP16 and CAUSAL.

At the highest matched valid context, `4096`, CAUSAL remains much slower and still uses more peak GPU memory than FP16:

- Throughput ratio: `0.091x` CAUSAL/FP16.
- Peak memory delta: `2.381 GB` CAUSAL minus FP16.
- Classification: `THROUGHPUT_AND_MEMORY_BOTTLENECK`.
- Next task: `FULL_MODEL_BOTTLENECK_PROFILE`.

## Scope

This is full-model decode serving under the decode-ready / prefill-completed benchmark harness. Prefill is used to prepare cache state but is not the target serving metric. This is not end-to-end request serving and not traditional TTFT.

## Regression Preservation

- Compileall: pass.
- Full-model harness tests: 4 passed.
- Continuous batching regression: 15 passed.
- Dynamic add/remove regression: 10 passed.
- Request lifecycle regression: 22 passed.
- Ragged targeted regression: 42 passed.
- Full pytest: 999 passed.
- `git diff --check`: pass before report generation; rerun pending in final terminal summary.

## Context Scaling Results

| Context | FP16 tok/s | CAUSAL tok/s | Ratio | FP16 peak GB | CAUSAL peak GB | Delta GB | Matched Valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 256 | 28.397 | 2.968 | 0.105 | 16.337 | 20.334 | 3.997 | yes |
| 2048 | 10.974 | 1.002 | 0.091 | 18.200 | 20.918 | 2.717 | yes |
| 4096 | 6.740 | 0.612 | 0.091 | 20.330 | 22.711 | 2.381 | yes |
| 8192 | not_measured | not_measured | not_measured | 23.514 | not_measured | not_measured | no |
| 16384 | not_measured | not_measured | not_measured | not_measured | not_measured | not_measured | no |

## Interpretation

The tiny-context anomaly is not simply fixed overhead that cleanly amortizes by 2K or 4K. CAUSAL peak memory remains higher than FP16 at 256, 2048, and 4096, while throughput stays near `0.09x-0.10x` of FP16. `FP16_FULL_MODEL` OOMs during full prefill at 8192 on the current harness, so 4096 is the highest matched valid context in this probe.

The strongest current diagnosis is a full-model runtime/integration bottleneck, likely in the CAUSAL prefill/cache construction path and compressed decode control path rather than the frozen KV algorithm itself. No selector, bitwidth, V4 ratio, sink/recent/residual, group size, or quantization semantics were changed.

## Claim Boundary

This result blocks the formal full-model capacity matrix. It does not invalidate the previous KV-runtime benchmark; it shows that the current full-model integration does not yet convert the KV-runtime memory advantage into full-model serving advantage.

## Artifacts

- `context_scaling.csv`
- `context_scaling_comparison.csv`
- `context_scaling_probe.json`
- `memory_breakdown_probe.json`
- `probe_final_gate.json`
- `regression_validation.txt`
