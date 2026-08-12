# Fixed Page ABI Design

## Logical Layout

The logical cache order remains unchanged: `sink -> packed historical -> pending -> recent`. Sink remains a small contiguous FP16 tensor. Recent is represented by `RecentRingBuffer`. Historical compressed streams are represented by `FixedPageBuffer` instances grouped under `FixedPageCacheStorage`.

## Physical Layout

Each stream owns pages shaped like its current tensor layout, replacing only the token dimension with `PAGE_SIZE=128`. Examples:

- FP16 K/V or V payload-like streams: `[B, KVH, 128, ...]`.
- Packed K streams: token/page dimension can be configured as dim=3.
- `v_precision_mask`: `[B, 128]`.
- Pattern metadata: `[B, KVH, 128]`.

## Page Descriptor

`PageDescriptor` records:

- `stream`
- `page_id`
- `logical_start_token`
- `valid_tokens`
- `page_size`
- `shape`
- `dtype`
- `device`

## Append

`FixedPageBuffer.append_block(x)` writes into the current page, allocates a new fixed page only when needed, and automatically splits across page boundaries. It never recopies old pages during append.

## Recent Ring

`RecentRingBuffer` stores a fixed 128-token window with `start` and `length`. Appending returns overflow in logical oldest-to-newest order and keeps materialized recent order stable for debug/correctness.

## Read / Debug

`materialize_contiguous()` exists only for correctness/debug and for bridge code while page-native CUDA is not implemented. It must not be counted as the final performance path.

## Future CUDA Contract

A page-native kernel should consume a compact descriptor array or parallel arrays:

- page pointers
- page valid lengths
- logical start token
- page size
- stream/lane identifiers

This phase does not implement page-native attention math.

`PAGE_ABI_READY_FOR_PAGE_NATIVE_CUDA = PARTIAL`

`PAGE_ABI_READY_FOR_VLLM_INTEGRATION = PARTIAL`
