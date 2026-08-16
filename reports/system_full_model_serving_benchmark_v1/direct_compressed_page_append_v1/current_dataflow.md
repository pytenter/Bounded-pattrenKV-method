# DIRECT_COMPRESSED_PAGE_APPEND_V1 Current Dataflow

## Decode Token Path

1. `q_proj/k_proj/v_proj` produce new FP16 K/V for one decode token per active request.
2. `append_decode_rolling` receives `[B,Hkv,1,D]` K/V. With C2048 after prefill, sink is fixed and recent is already full at 128 tokens.
3. The new token is appended to `recent_k/recent_v`; the oldest recent token overflows into `pending_k/pending_v`. Current implementation uses `_cat_token`, which calls `torch.cat(...).contiguous()` and records category `recent_pending`.
4. `flush_pending` only compresses when pending reaches `group_size=128`. For formal decode=8, no boundary flush happens in the measured window.
5. Attention consumes sink, packed history, pending, and recent. Mixed compressed Value uses existing `operator_ready_page_pools` built during prefill or boundary flush.

## Page Batch Pack Path

`page_batch_pack` is entered by `_cat_mixed_packed_v` when a 128-token window is actually compressed. It partitions the window by V2/V4 precision per request/page, quantizes V2 and V4 compact streams, builds page metadata, builds operator-ready pools, and appends those pools.

In C2048/B1/B2 decode=8, corrected decode-only profiling shows this does not execute inside the timed decode window. The old post-scaling profile reset before prefill, so `page_batch_pack` from initial prefill was attributed against decode wall time.

## Classification

- `page_batch_pack`: `ABI_LAYOUT_ARTIFACT` for compressed-window commit/prefill, but not a current decode=8 timed-window bottleneck.
- `recent_pending`: runtime layout artifact from contiguous FP16 recent/pending representation and `_cat_token` append/overflow handling. It scales with B and layers, but measured CUDA event time is small relative to total CAUSAL TPOT in decode=8.
