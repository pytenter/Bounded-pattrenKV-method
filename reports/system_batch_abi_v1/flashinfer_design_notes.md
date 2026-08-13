# FlashInfer Design Notes

Sources:

- FlashInfer attention API: https://docs.flashinfer.ai/api/attention.html
- FlashInfer append paged KV API: https://docs.flashinfer.ai/generated/flashinfer.page.append_paged_kv_cache.html
- FlashInfer GitHub: https://github.com/flashinfer-ai/flashinfer

## Observations

- `BatchDecodeWithPagedKVCacheWrapper.plan` uses paged KV metadata:
  - `indptr: [batch_size + 1] int32`
  - `indices: [indptr[-1]] int32`
  - `last_page_len: [batch_size] int32`
  - `num_qo_heads`, `num_kv_heads`, `head_dim`, and `page_size`.
- The attention docs also expose packed/ragged prefill paths with `qo_indptr` and `kv_indptr`.
- Plan/run separation allows metadata planning once and repeated launches under compatible shape/capacity constraints.
- Page metadata is enough for batch decode; there is no requirement that KV be a dense `[B,H,T,D]` tensor.

## Why Batch Decode Does Not Need `[B,H,T,D]`

The kernel work item can be derived as:

```text
request b
  logical page range = indices[indptr[b] : indptr[b+1]]
  valid tokens in final page = last_page_len[b]
  seq_len = (num_pages - 1) * page_size + last_page_len[b]
  work = request/head/page-or-split tiles
```

Physical pages can be arbitrary. The page table is the request-local logical view. This is exactly the property PatternKV needs for ragged V2/V4 streams.

## PatternKV Reuse

Directly reusable:

- `indptr/indices/last_page_len` page table shape;
- page-size-aware decode metadata;
- split-KV planning and per-request sequence lengths.

Conceptually reusable:

- separate append path and decode read path;
- page-aligned physical pools;
- graph-stable metadata buffers.

Not directly reusable:

- FlashInfer's dense FP/FP8 page payload kernels; PatternKV's independent affine V2/V4 streams and centroid restoration require custom kernels.

## Design Lesson

PatternKV should expose V2/V4 as paged streams with request-local tables, not as `[B,H,T_v2,D]` and `[B,H,T_v4,D]`. The operator should build work over `(request, head, logical page, split)` and use page-local precision/rank metadata.
