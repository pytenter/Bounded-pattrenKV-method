# Multi-Request Low-Copy V1 Final Report

## Root Cause

B1 stayed on an optimized singleton path where assemble/split returned the cache object directly. B2/B4 diverged at `PatternKVAdapter.assemble_batch(len(caches) > 1)` and `PatternKVAdapter.split_batch(batch_size > 1)`, causing every decode token to rebuild all 32 layer metadata objects and extract every active request row from compressed page pools.

Classification: `MULTI_COMPONENT`: `MULTI_REQUEST_ACTIVE_ROW_MATERIALIZATION`, `SEGMENTED_CACHE_ROW_SLICE_DEPENDENCY`, `MULTI_REQUEST_METADATA_REBUILD_LEGACY_PATH`, plus stale B8 `PAGE_BATCH_MVP_LIMIT`.

## Fix

Added `ActiveBatchState` to keep the current PatternKV batched compressed cache while active membership is unchanged. Empty-active prefill now starts directly from a batched cache via `prefill_active_batch`. Membership changes still rebuild explicitly through the existing safe extraction path. The B8 page-pack guard now accepts any positive batch size.

## Measured Structural Result

- B2 after: plan builds/iteration=1.0, layer rebuilds/iteration=0.0, row-slice bytes/iteration=0.0
- B4 after: plan builds/iteration=1.0, layer rebuilds/iteration=0.0, row-slice bytes/iteration=0.0
- B8 sanity: plan builds/iteration=1.0, layer rebuilds/iteration=0.0, row-slice bytes/iteration=0.0

## Validation

- Focused semantic/low-copy tests: 117 passed
- Full pytest: 1008 passed
- `git diff --check`: PASS

Status: `FULL_MODEL_MULTI_REQUEST_LOW_COPY_INTEGRATION_SUPPORTED`. Formal serving benchmark remains not closed.
