# Implementation Summary

- Added experimental `PatternKVPageBatchCache` and `PatternKVBatchMetadata` in `bench/patternkv_page_batch_mvp.py`.
- Added `pack_mixed_v_pages(...)` using request-local precision masks; it never uses `precision_mask[0]` as a batch-global layout.
- Added `patternkv_page_batched_v_decode(...)`, a single batched API that accumulates compact V2/V4 page contributions without calling serial B=1 kernels.
- Added `reference_batch_mixed_v(...)` as golden serial B=1 reference only.
