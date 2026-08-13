# Final Recommendation

## Core Answer

Choose B:

```text
Current global V2/V4 streams are fundamentally not serving-native.
A page-centric redesign is required for true B>1 serving.
```

The current ABI is safe for B=1 because one precision vector defines one V2 stream and one V4 stream. In serving, each request has independent V4 positions and potentially different stream counts, so the physical layout must carry request/page-local offsets or tables.

## Recommended ABI

`PAGE_CENTRIC_DUAL_STREAM`, with framework-compatible page tables.

Properties:

- page size: `128`
- independent V2/V4 affine streams preserved
- asymmetric K/V runtime preserved
- K remains tight and compute-oriented
- V becomes page/capacity-friendly
- metadata uses request/page tables, precision bitmap, counts, and compact Pattern metadata

## Operator Recommendation

A custom batched decode operator is required.

Production work decomposition should prioritize:

```text
(request, kv_head, page/split)
```

Reference batch may use serial B=1 only for correctness.

## Framework Recommendation

Preferred order:

1. standalone page-centric ABI + batched decode MVP;
2. SGLang decode backend adapter;
3. vLLM native backend/cache spec.

SGLang is the preferred first framework adapter because its backend boundary naturally allows existing prefill plus custom decode.

## Classification

`BATCH_ABI_PAGE_REDESIGN_REQUIRED`

## Next Task

`PATTERNKV_PAGE_CENTRIC_BATCH_ABI_MVP`
