# Serving-Native PatternKV Batch ABI

Status: design recommendation. No runtime code was changed.

## Design Goals

- Preserve `ASYMMETRIC_KV_RUNTIME`.
- Keep K INT2 in a compute-optimized tight layout.
- Keep V as independent affine V2 and V4 streams.
- Support true B>1 serving with ragged request lengths and continuous batching.
- Avoid Python `for b in range(B)` as production decode.
- Fit naturally into SGLang/vLLM-style request/page metadata.

## Candidate A: Ragged Global Stream

Layout:

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
  precision bitmap
  logical->physical rank metadata
  selector state pointer/ref
```

Pros:

- minimal conceptual change from current compact streams;
- easy to explain and validate;
- good for standalone prototype.

Cons:

- allocator must compact/grow ragged streams;
- continuous batching append and eviction are awkward;
- coalescing is fragile if requests append unevenly;
- kernel needs offsets and prefix/rank metadata for every request.

Assessment: useful bridge design, not ideal serving ABI.

## Candidate B: Page-Centric Dual Stream

Preferred page size: `128`.

Each logical page owns:

```text
K payload reference
V2 payload page(s)
V4 payload page(s)
V2 scale/zero pages
V4 scale/zero pages
precision bitmap[128]
V2 count, V4 count
V2 compact pattern metadata
V4 compact pattern metadata
optional page-local logical->physical rank table
logical token start
valid tokens
```

Per request:

```text
page_table
num_pages
seq_len
selector_state_ref
```

Pros:

- natural ragged support;
- compatible with continuous batching/page allocation;
- page-local V2/V4 counts solve variable compact lengths;
- coalescing can be maintained inside each stream page;
- future VMM/page allocator compatibility is high.

Cons:

- requires new append/write path;
- operator must read page-local precision/rank metadata;
- metadata format must be carefully packed to keep overhead bounded.

Assessment: best standalone PatternKV ABI and best conceptual fit for serving.

## Candidate C: Framework-Native Pool ABI

Layout:

```text
K compressed pool
V2 compressed pool
V4 compressed pool
metadata pool

per request:
  k_page_table
  v2_page_table
  v4_page_table
  metadata_page_table
  seq_len
  selector_state_ref
```

Pros:

- closest to vLLM block tables, SGLang KV pools, and FlashInfer paged metadata;
- clear ownership split between scheduler, allocator, backend, and kernel;
- reduces future integration cost.

Cons:

- needs serving framework adaptation;
- more initial plumbing than standalone global streams;
- K/V asymmetry must be represented explicitly, not hidden under ordinary KV slots.

Assessment: target integration ABI after standalone operator proof.

## Recommended ABI: Page-Centric Dual Stream With Framework-Compatible Tables

`PatternKVBatchMetadata`:

```text
request_indptr:      int32[B+1]       logical page table CSR offsets
seq_lens:            int32[B]
num_pages:           int32[B]
k_page_table:        int32[total_pages]
v2_page_table:       int32[total_pages]
v4_page_table:       int32[total_pages]
metadata_page_table: int32[total_pages]
precision_bitmap:    uint32[metadata_pages, 4]       # 128 bits/page
v2_counts:           uint16[metadata_pages]
v4_counts:           uint16[metadata_pages]
v2_prefix_counts:    uint16[metadata_pages, 129] or computed page-local
v4_prefix_counts:    uint16[metadata_pages, 129] or computed as t - v2_prefix
selector_state_ids:  int32[B] or opaque refs
out_cache_loc:       int32[B] for append/write
```

Payload pools:

```text
k_pool:       tight INT2 K residual pages/blocks
k_scale_zero: affine metadata for K
v2_pool:      independent affine INT2 V pages
v2_scale_zero
v4_pool:      independent affine INT4 V pages
v4_scale_zero
v2_pattern:   compact V2 gate + assignment metadata
v4_pattern:   compact V4 gate + assignment metadata
centroids:    [layers,Hkv,M,D]
```

## K Serving Layout

Do not unify K and V layouts. K should remain compute-optimized.

Candidates:

- per-request tight K segment: easiest to preserve current QK path, poor allocator behavior;
- global K pool + offsets: reasonable standalone bridge, still ragged;
- tight packed K blocks/pages: recommended long-term, because it preserves tight per-page QK math while fitting serving allocation.

Avoid reintroducing the failed generic strided K reader as the primary path. The K page reader should operate on tight packed pages/blocks and keep the current fast QK assumptions inside a page.

## Precision Metadata

Recommended lookup:

```text
logical token t in page p
  bit = precision_bitmap[p][t]
  rank2 = popcount(~precision_bitmap[p] before t) + page_v2_base
  rank4 = popcount( precision_bitmap[p] before t) + page_v4_base
```

Options:

- bitmap + popcount: lowest metadata, slightly more GPU arithmetic;
- page-local prefix counts every token: faster lookup, more metadata;
- rank LUT: simplest kernel, `128 bytes/page`, acceptable only if profiling requires it;
- hash table/global search: prohibited for production ABI.

## Final Recommendation

Build a standalone batched decode operator against Candidate B, but define the metadata names and table shapes so Candidate C integration is a mechanical adapter. Do not start by adapting full SGLang/vLLM serving. The minimum useful artifact is a page-centric PatternKV batch ABI plus a standalone operator simulator/kernel contract.
