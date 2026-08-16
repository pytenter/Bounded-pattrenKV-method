# Serving Benchmark v1 Final Report

## Executive Summary

This run measures decode-ready, prefill-completed KV-runtime continuous batching on one physical RTX 3090 GPU. It is not a full end-to-end model-serving TTFT benchmark.

At 16K context, decode128, B=8, CAUSAL_V4_25 reached 78520.32 output tokens/s versus 76981.08 for the FP16 KV-runtime baseline. Peak memory fell by 84.37%.

Maximum successful concurrency in the sweep was 64 for CAUSAL_V4_25 and 8 for FP16_KV_RUNTIME.

## Hardware / Environment

- GPU: physical GPU 2, NVIDIA GeForce RTX 3090, 24576 MiB
- Python: `/data/zypan/.local/share/mamba/envs/patternkv/bin/python`
- Torch: 2.4.1+cu124

## Methods

- `CAUSAL_V4_25`: frozen CAUSAL-V4@25% theoretical KV budget, measured with compressed KV-runtime allocation.
- `FP16_KV_RUNTIME`: same workload/scheduler/model geometry, FP16 KV-runtime allocation baseline.

## Fairness Protocol

- Same physical GPU, model geometry, context length, decode length, total requests, FIFO scheduler, saturated steady-state arrival protocol, warmup count, measured run count, and workload hashes for comparable rows.

## Baseline Availability Audit

- FP16: AVAILABLE_AS_FP16_KV_RUNTIME_BASELINE - Same FIFO decode-ready scheduler/workload/model geometry; KV runtime only, not full model forward.

- Original PatternKV: BASELINE_NOT_YET_SERVING_COMPARABLE - No existing same-policy continuous serving harness found.

- KIVI: BASELINE_NOT_YET_SERVING_COMPARABLE - Model class exists, but no same-policy continuous serving harness found.

- CAUSAL_V4_25: AVAILABLE - Measured as compressed-domain KV-runtime under same FIFO scheduler and workload.

## Results

- CAUSAL_V4_25 B=1: throughput=9865.59 tok/s, mean_tpot=0.1013 ms/token, p95_tpot=0.1049 ms/token, peak_mem=0.336 GB

- CAUSAL_V4_25 B=2: throughput=19895.64 tok/s, mean_tpot=0.1005 ms/token, p95_tpot=0.1035 ms/token, peak_mem=0.671 GB

- CAUSAL_V4_25 B=4: throughput=39029.34 tok/s, mean_tpot=0.1025 ms/token, p95_tpot=0.1081 ms/token, peak_mem=1.342 GB

- CAUSAL_V4_25 B=8: throughput=78520.32 tok/s, mean_tpot=0.1018 ms/token, p95_tpot=0.1030 ms/token, peak_mem=2.685 GB

- FP16_KV_RUNTIME B=1: throughput=9943.35 tok/s, mean_tpot=0.1005 ms/token, p95_tpot=0.1050 ms/token, peak_mem=2.147 GB

- FP16_KV_RUNTIME B=2: throughput=17931.19 tok/s, mean_tpot=0.1119 ms/token, p95_tpot=0.1945 ms/token, peak_mem=4.295 GB

- FP16_KV_RUNTIME B=4: throughput=39258.66 tok/s, mean_tpot=0.1018 ms/token, p95_tpot=0.1062 ms/token, peak_mem=8.590 GB

- FP16_KV_RUNTIME B=8: throughput=76981.08 tok/s, mean_tpot=0.1038 ms/token, p95_tpot=0.1066 ms/token, peak_mem=17.180 GB

## Maximum Concurrency

- FP16_KV_RUNTIME: max_success=8, first_oom=16, peak_at_max=17.180 GB

- CAUSAL_V4_25: max_success=64, first_oom=128, peak_at_max=21.479 GB

## First Token Scope

The first-token metric is `decode_admission_to_first_token_ms`. Traditional end-to-end TTFT is not claimed because prefill scheduling is outside the current supported scope.

## Bottleneck Analysis

In this KV-runtime benchmark, memory capacity is the clear differentiator: CAUSAL reaches 8x the successful concurrency of FP16 in the 16K sweep. At shared B=8, throughput is modestly higher because the synthetic batched decode step is not memory-bandwidth dominated enough to fully expose the memory reduction as per-token speed. Full model profiling is required before attributing end-to-end bottlenecks.

## Correctness Invariants

- Serial request forward dispatches: 0
- Historical FP16 K/V materialization: 0 / 0
- Fallback count: 0
- Continuous/dynamic/lifecycle/ragged/full pytest regressions: pass

## Claim Boundaries

This is a supported Benchmark v1 for decode-ready KV-runtime continuous batching. It does not claim a full vLLM-equivalent serving frontend, mixed prefill/decode scheduling, or full-model FP16/PatternKV/KIVI serving superiority.

## Final Classification

`PATTERNKV_SERVING_BENCHMARK_V1_SUPPORTED_KV_RUNTIME_SCOPE`
