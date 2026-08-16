# B1 vs B2/B4 Path Diff

| stage | B1 behavior | B2/B4 old behavior | fixed behavior | copy | metadata rebuild | candidate/root fix |
|---|---|---|---|---|---|---|
| Prefill | `PatternKVAdapter.split_batch(..., 1)` returns the cache object directly | `prefill_batch` split the batched prefill cache into per-request caches | empty-active PatternKV prefill keeps one batched cache via `prefill_active_batch` | old B>1 row extraction copied compressed page pools | old B>1 split deserialized every layer | keep active batch cache when membership starts empty |
| Decode assemble | singleton `assemble_batch` returns the request cache directly | `assemble_batch(len(caches)>1)` called `assemble_ragged_patternkv_cache` for every layer | active membership match reuses the current batched cache | old B>1 rebuilt packed active rows | old B>1 recorded 32 layer rebuilds/iteration | `ActiveBatchState` keyed by request ids |
| Model decode | one true batched model dispatch | one true batched model dispatch | unchanged true batched model dispatch | none introduced | none introduced | preserve compressed-domain model path |
| Decode split | singleton split returns the cache object directly | `split_batch(batch_size>1)` extracted every row from every layer every token | split only when membership changes and only if survivors need cache state | old steady B>1 copied 12.8 MiB/iteration for B2 and 51.4 MiB/iteration for B4 at context 512 | old split work repeated every token | make partial cache ownership active-batch local, not per-request per-token |
| B8 page pack | not applicable | hard-coded guard rejected B8 in `pack_mixed_v_pages` | guard now accepts any positive batch size; loop/metadata were already generic | none | none | remove stale MVP batch limit |

First divergence from the B1 optimized path was `PatternKVAdapter.assemble_batch(len(caches) > 1)` and `PatternKVAdapter.split_batch(batch_size > 1)`, which forced per-layer active cache assembly and per-token row extraction.
