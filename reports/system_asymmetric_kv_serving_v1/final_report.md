# Final Report

## Result

- Final classification: `CONCURRENCY_RUNTIME_BLOCKED`.
- Recommended next phase: `BATCH_SAFE_RUNTIME_FEASIBILITY`.
- Serving throughput gain: `BLOCKED`.

## Why

The S6-B definition requires true shared-model batched serving. The frozen causal_v4 mixed-V runtime stores selected V4 tokens in compact V2/V4 streams selected by `precision_mask[0]`, and the fused mixed-V attention wrapper raises on `B != 1`. Supporting B>1 while preserving independent per-request V4 identities requires a batch-safe mixed-V cache ABI and runtime plumbing, not a benchmark-only change.

## What Was Not Run

- No capacity sweep.
- No throughput sweep.
- No long-decode serving stress.
- No vLLM/SGLang/AIME24/AIME25/GPQA/CUDA VMM.
