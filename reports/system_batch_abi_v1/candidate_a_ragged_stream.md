# Candidate A: Ragged Global Stream

## ABI

```text
global_k_stream
global_v2_stream
global_v4_stream
global_metadata_stream

per request:
  k_offset, k_length
  v2_offset, v2_length
  v4_offset, v4_length
  seq_len
  precision_bitmap_offset
  selector_state_offset
```

## Pros

- Lowest conceptual distance from the current B=1 compact streams.
- Good for a standalone reference/MVP.
- Explicitly represents variable V2/V4 stream lengths.
- Does not require immediate SGLang/vLLM allocator integration.

## Cons

- Continuous batching append/evict is allocator-heavy.
- Coalescing depends on allocation history.
- Request-local stream ranges fragment over time unless compacted.
- Kernel still needs logical-to-physical rank lookup.

## Assessment

`MODERATE` as an MVP reference ABI, but not the recommended serving target. It solves the row-0 precision mask bug but does not naturally become a page allocator.
