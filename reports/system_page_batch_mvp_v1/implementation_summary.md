# Implementation Summary

- Added production-facing `PatternKVPageBatchCache` and `PatternKVBatchMetadata` in `quant/page_batch.py`.
- Added `pack_mixed_v_pages(...)` using request-local precision masks; it never uses `precision_mask[0]` as a batch-global layout.
- Added `patternkv_page_batch_decode(...)`, a single batched API that accumulates compact V2/V4 page contributions without calling serial B=1 kernels.
- Added `reference_batch_mixed_v(...)` as golden serial B=1 reference only.
