# Current Cache Layout Audit

Preflight:

- `REPO_ROOT=/data/zypan/Bounded-pattrenKV-pseudodecode-3090`
- `CURRENT_BRANCH=sys/causal-v4-25-kernel-v1`
- `START_HEAD=54c417272559ed0977279365b7d638dab15349f7`
- `WORKTREE_CLEAN=true` at phase start
- `BOUNDED_REMOTE=git@github.com:pytenter/Bounded-pattrenKV-method.git`
- `ORIGIN_REMOTE=https://github.com/HCOOOH/PatternKV.git`

Current cache object: `models.segmented_cache.PatternQuantizedKVCache`, extending `QuantizedKVCache`.

## Current Tensors

- K FP16 sink: `sink_k` as `[B, KVH, T_sink, D]`.
- V FP16 sink: `sink_v` as `[B, KVH, T_sink, D]`.
- K FP16 recent: `recent_k` as `[B, KVH, T_recent, D]`.
- V FP16 recent: `recent_v` as `[B, KVH, T_recent, D]`.
- Residual/pending K/V: `pending_k`, `pending_v` as `[B, KVH, T_pending, D]`.
- Packed historical K: `packed_k` as `[B, KVH, D, packed_token_words]`; `packed_k_tokens` tracks logical tokens.
- K scale/zero: `packed_k_scale`, `packed_k_zero`, concatenated on packed token-group dimension.
- Packed historical V2: `packed_v` as `[B, KVH, T_v2_compact, D / pack]`.
- V2 scale/zero: `packed_v_scale`, `packed_v_zero`.
- Packed historical V4: `packed_v4`, `packed_v4_scale`, `packed_v4_zero`, with `packed_v4_tokens`.
- V precision mask: `v_precision_mask` as `[B, packed_v_tokens]`, preserving logical V2/V4 identities.
- Pattern K assignment: `k_assignments` as `[B, KVH, packed_k_tokens]`.
- Pattern V assignment/index: `v_assignment_idx` as `[B, KVH, packed_v_tokens]`.
- Pattern mask: `v_pattern_mask` / `v_assignments` as `[B, KVH, packed_v_tokens]`.
- Centroids: `k_centroids`, `v_centroids` as `[KVH, Mcent, D]`; small centroid append can still use contiguous tensors.

## Mutation Sites

- `_cat_token`: appends sink/recent/pending FP16 tokens with `torch.cat(..., dim=2).contiguous()`.
- `_cat_packed_k`: appends packed K, K scale, K zero with `torch.cat`.
- `_cat_packed_v` and `_cat_v_payload`: append V2/V4 payload, scale, and zero with `torch.cat`.
- `_cat_assignment`: appends Pattern assignments, V masks, and metadata with `torch.cat`.
- `_cat_mixed_packed_v`: appends `v_precision_mask` with `torch.cat`.
- `append_decode_rolling`: appends decode tokens to recent via `_cat_token`; overflow is sliced into pending, then `flush_pending` packs 128-token groups.

## Why recent_pending is largest

`recent_k/recent_v` are fixed-window concepts but currently grow by `torch.cat` before slicing overflow. That means each single decode token can copy the existing 128-token recent K and V tensors. The synthetic mutation benchmark reports `524288` old recent bytes copied/token at 16K and 32K for the FP16 K/V recent window.

## Which concat can be eliminated

- Historical packed K/V payload appends: fixed pages can eliminate old-history recopy.
- V2/V4 compact payload appends: fixed pages preserve compact order without recopy.
- scale/zero and Pattern metadata: fixed pages eliminate old metadata recopy.
- `v_precision_mask`: fixed pages preserve logical identity without recopy.
- Recent append: fixed-capacity ring buffer removes unbounded recent `torch.cat`; logical order is exposed as at most two segments or debug materialization.

## Logical order

Current logical order is `sink -> packed historical -> pending -> recent`. Mixed V keeps a logical precision mask plus separate compact V2/V4 streams. The fixed-page ABI preserves the same logical order by descriptors: each page has `logical_start_token`, `valid_tokens`, and page-local offset.
