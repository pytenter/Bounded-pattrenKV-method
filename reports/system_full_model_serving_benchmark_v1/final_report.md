# Full Model Serving Benchmark V1

## Executive Summary

This phase is not closed as a formal 16K/decode128 full-model serving benchmark.

What is complete:

- Added a full-model decode serving harness for `FP16_FULL_MODEL` and `CAUSAL_V4_25_FULL_MODEL`.
- Verified the path executes embeddings, transformer layers, attention, MLP projections, RMSNorm, LM head, token selection, and the FIFO scheduler.
- Produced same-GPU tiny smoke results for both methods at `context=256`, `decode=4`, `B=1`, `total_requests=2`.
- Preserved the decode-ready / prefill-completed scope. Prefill is excluded from timing.

What is not complete:

- No formal `context=16384`, `decode=128`, `B=1/2/4/8`, `measured_runs>=3` matrix was completed.
- No formal maximum-concurrency sweep was completed.
- No regression suite or full pytest was rerun after the benchmark harness changes.

## Scope / Claim Boundary

The measured result is a full-model decode smoke/probe, not an end-to-end request-serving benchmark and not traditional TTFT. The first-token metric is decode admission to first token only.

## Hardware / Environment

- GPU: NVIDIA GeForce RTX 3090, physical GPU 1, 24576 MiB
- Python: `/data/zypan/.local/share/mamba/envs/patternkv/bin/python`
- PyTorch: `2.4.1+cu124`
- CUDA runtime in PyTorch: `12.4`
- pytest: `9.1.1`

## Full-Model Path Audit

`full_model_path_audit.json` shows both FP16 and CAUSAL execute:

- embedding
- transformer layers
- attention
- MLP projections
- RMSNorm
- LM head
- token selection
- scheduler

## Baseline Audit

- `FP16_FULL_MODEL`: available for same-harness smoke comparison.
- `CAUSAL_V4_25_FULL_MODEL`: available for same-harness smoke comparison.
- `ORIGINAL_PATTERNKV_FULL_MODEL`: not currently serving-comparable in this harness.
- `KIVI_FULL_MODEL`: not currently serving-comparable in this harness.

## Tiny Smoke Results

Workload:

- context: 256
- decode: 4
- B: 1
- total requests: 2
- warmup runs: 1
- measured runs: 1
- scheduler: FIFO

Summary:

| Method | Throughput tok/s | Mean TPOT ms | P95 TPOT ms | Peak allocated GB |
| --- | ---: | ---: | ---: | ---: |
| FP16_FULL_MODEL | 22.618 | 45.643 | 46.562 | 16.438 |
| CAUSAL_V4_25_FULL_MODEL | 1.583 | 685.554 | 694.036 | 20.859 |

These values are smoke/probe evidence only. They must not be reported as the requested formal full-model 16K/decode128 serving benchmark.

## Interpretation

The tiny B1 smoke does not support a full-model memory or throughput advantage for CAUSAL. It instead indicates the current full-model CAUSAL path has substantial overhead at low concurrency and small context. This is not enough to answer whether the KV-runtime concurrency advantage survives in full-model 16K serving.

## KV-Runtime Comparison Boundary

Prior KV-runtime benchmark results remain separate:

- KV-runtime memory reduction at B8: 84.37%
- KV-runtime concurrency gain: 8x
- KV-runtime matched-B8 throughput: 1.02x

Those results are not full-model throughput results.

## Validation

- `py_compile`: pass for modified benchmark/runtime files
- Targeted harness tests: `10 passed`
- `git diff --check`: pass
- Full pytest: not run
- Continuous/dynamic/lifecycle/ragged regressions: not run in this phase

## Final Classification

`PATTERNKV_FULL_MODEL_SERVING_BENCHMARK_V1_NOT_CLOSED`

Next task: `FULL_MODEL_FORMAL_BENCHMARK_CAPACITY_PROBE`
