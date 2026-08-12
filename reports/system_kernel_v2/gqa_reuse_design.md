# Candidate B: GQA Reuse Design

## Current Global-Memory Behavior

The current V2 value-attention kernel uses `blockIdx.x = B * num_attention_heads`.
For the real model, `num_attention_heads=32`, `num_key_value_heads=8`, and the
GQA ratio is `4`.

Each group of four query heads maps to the same KV head:

```text
Q0 Q1 Q2 Q3 -> KV0
Q4 Q5 Q6 Q7 -> KV1
...
```

The packed V2 payload, scale, zero, Pattern mask, assignment indices, and
centroid bank are therefore logically reusable across four query heads.

## Candidate Shared Data

Potentially shareable:

- packed V2 payload rows
- scale and zero rows
- Pattern mask
- assignment indices
- centroid rows

Not shareable:

- attention weights `alpha_q`, because each query head has different attention.
- output accumulators, because each query head produces a different result.

## Expected Bytes Saved

In theory, staging one KV head's compressed V2 metadata for four query heads
could reduce repeated global reads of V payload and metadata. But the current
kernel streams over K and computes one query head per block group, so sharing
would require blocks that cover multiple query heads simultaneously.

## Expected Synchronization / Storage Cost

A GQA-aware block would need either:

- more warps per block, one or more per query head and output tile, or
- shared-memory staging of packed V2/scale/zero/mask/assignment across query
  heads.

For 32K context, staging K-long V payload or metadata is too large for shared
memory. Tile-level staging is possible, but it introduces additional
synchronization and a larger block schedule. It also risks increasing register
pressure because each query head has separate attention and accumulators.

## Decision

`NOT_IMPLEMENTED` in S2B-1.

Reason: the reuse opportunity is real, but the implementation is not a local
V2-only one-change optimization. It is closer to a future execution-strategy
experiment (`S2B-3_V2_V4_EXECUTION_STRATEGY`) than a safe candidate for this
phase.
