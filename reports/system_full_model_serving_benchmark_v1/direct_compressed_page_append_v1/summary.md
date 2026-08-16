# DIRECT_COMPRESSED_PAGE_APPEND_V1

## Status

`DIRECT_COMPRESSED_PAGE_APPEND_V1_NOT_SUPPORTED_AS_RUNTIME_OPTIMIZATION`. I did not implement a direct page append kernel/path because corrected decode-only profiling shows `page_batch_pack` is not in the C2048 decode=8 timing window. Optimizing it now would target prefill or 128-token boundary flush, not the current formal TPOT blocker.

## Key Finding

The old `page_batch_pack ~46%` attribution came from profile counters being reset before initial prefill. After resetting profile counters at the decode timing boundary, C2048 B1 and B2 decode-only profiles contain no `page_batch_pack` calls. The measured decode cache mutation remains `recent_pending`: B1 `257.4 MB` over `1024` calls, B2 `514.9 MB` over `1024` calls.

## Root Cause of Page Batch Pack

`page_batch_pack` exists because mixed V2/V4 pages require page-local precision partitioning, compact V2/V4 payload streams, page tables, per-page counts, precision bitmaps, prefix counts, and metadata aligned to the fused page Value operator ABI. It is an ABI/layout artifact for compressed-window commit. In formal C2048 decode=8 it is not executed after the timing boundary.

## Root Cause of Recent Pending Copy

`recent_pending` comes from `_cat_token` in `append_decode_rolling`: the full recent tail and pending tensors are represented as contiguous tensors, so each decode token performs `torch.cat(...).contiguous()` to append new FP16 K/V and move recent overflow into pending. Bytes scale almost linearly with B: B1 `257.4 MB`, B2 `514.9 MB`.

## Performance

No production direct-append optimization was applied. C2048 B1 formal smoke after instrumentation: CAUSAL `191.73 ms/token`, `5.22 tok/s`; P0 baseline was `195.65 ms/token`, `5.11 tok/s`. This is normal run variance, not a direct append speedup.

## Decision

Do not write direct compressed page append as P1 for the current decode=8 gate. The next actionable runtime target should be the largest decode-only components: self-attention total, FP16 tail Value path, RMSNorm/MLP contribution, or a recent/pending ring-buffer design if the goal is specifically to remove the B-scaling copy artifact.
