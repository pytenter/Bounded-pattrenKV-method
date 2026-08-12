# Phase S1.5 Cache Mutation / torch.cat Audit

## Static Audit

The decode cache still grows through dynamic tensor concatenation in `models/segmented_cache.py`.

Primary hot-path helpers:

- `_cat_token`: appends sink/recent/pending K/V tensors with `torch.cat(..., dim=2)`.
- `_cat_packed_k`: appends packed K payload plus scale/zero tensors.
- `_cat_packed_v`: appends non-mixed packed V payload plus scale/zero tensors.
- `_cat_v_payload`: appends mixed V2 and V4 payload plus scale/zero tensors.
- `_cat_assignment`: appends K assignments, V assignment indices, and V Pattern masks.
- `_cat_mixed_packed_v`: appends `v_precision_mask` through `torch.cat(..., dim=1)`.
- `_append_dynamic_centroids`: appends K/V centroid banks when dynamic centroid updates occur.

Decode entry points:

- `append_decode_rolling` appends each decode token into the rolling recent region, moves overflow into pending, and calls `flush_pending`.
- `flush_pending` packs a window when pending reaches `group_size`.
- `flush_chunked_buffer` is also instrumented, but the frozen CAUSAL_V4_25 formal config uses the rolling segmented cache path.

Reconstruction helpers also contain `torch.cat` calls, but they are not part of the S1 fused decode hot path.

## Runtime Counter Method

Phase S1.5 instrumentation counts cache concatenation events, approximate bytes copied, and largest single concatenation by summing input tensor sizes at each dynamic append. This is not allocator tracing; it is a conservative hot-path copy audit.

## Runtime Findings

For the fused backend at `T=32768`, `decode_tokens=128`:

- Cache concat events: `16544`
- Approximate bytes copied: `4057821184`
- Largest single concat input footprint: `9400320`
- Cache mutation share of decode wall time: `20.46%`

For the fused backend at `T=16384`, `decode_tokens=128`:

- Cache concat events: `16544`
- Approximate bytes copied: `3650449408`
- Largest single concat input footprint: `4681728`
- Cache mutation share of decode wall time: `21.65%`

## Interpretation

Dynamic cache mutation is not the largest measured bottleneck after S1; mixed-V fused attention remains larger. But cache mutation is the largest non-kernel / ABI-shaped systems cost in the fused decode path, and it is large enough to make a fixed-page ABI a plausible follow-up after the remaining mixed-V contribution is understood.
