# MVP Implementation Plan

## Scope

First implementation phase after this study:

- B=2 and B=4
- fixed equal sequence length
- decode only
- standalone, not formal SGLang/vLLM serving
- no CUDA graph
- no ragged/continuous batching yet

## Goals

Prove:

```text
batch-safe cache ABI
+
batched compressed-domain decode operator
```

## Steps

1. Implement `PatternKVBatchMetadata` test builder for synthetic B=2/B=4 fixed-length pages.
2. Implement `REFERENCE_BATCH` that runs independent B=1 references and compares output.
3. Implement metadata-aware materialized reference that reconstructs logical V from page metadata, only in tests.
4. Implement standalone batched compressed-domain decode operator.
5. Add correctness tests for independent V4 positions per request.
6. Add counters proving historical V materialization is zero in production path.

## Correctness Gate

Record:

- max abs error;
- relative L2;
- cosine;
- cache isolation PASS;
- selector isolation PASS;
- NaN/Inf count 0;
- historical V materialization 0;
- K tight;
- page-native old reader OFF;
- experimental GQA OFF.

## Performance Gate

No performance in this phase. Future early positive signal:

```text
batched_operator_runtime <= 0.95 * sum(serial_B1_dispatch_runtime)
```

This is not a formal serving gate.

## Next Task

`PATTERNKV_PAGE_CENTRIC_BATCH_ABI_MVP`
