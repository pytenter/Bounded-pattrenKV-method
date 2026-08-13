# Decode Operator Spec

Canonical interface:

```python
output = patternkv_batch_decode(
    q,
    k_pool,
    v2_pool,
    v4_pool,
    k_metadata,
    v_metadata,
    batch_metadata,
    centroids,
)
```

Required support:

- `B >= 1`
- fixed equal sequence length for MVP
- ragged sequence lengths after MVP
- independent V4 positions per request
- independent selector state per request
- GQA `32Q / 8KV`

Detailed operator contract: [patternkv_batched_decode_operator_spec.md](patternkv_batched_decode_operator_spec.md).

## Work Decomposition Comparison

| Decomposition | Coalescing | Load balance | GQA reuse | Long context | Ragged requests | Branch divergence | Assessment |
|---|---|---|---|---|---|---|---|
| `(request,q_head)` | simple Q ownership; poor page locality | weak for long contexts | weak, repeats KV work per Q head | needs inner split loop | ok with metadata | precision branches inner-loop | good reference, poor production |
| `(request,kv_head,split)` | good KV locality | good split-KV balance | strong for GQA if Q heads share KV work | strong | good | manageable | recommended production direction |
| `(request,kv_head,page)` | best page locality | many small tasks | strong | good | excellent | page-local precision branches | recommended for page-centric V stage |
| flattened dynamic work queue | best balance potential | best | configurable | best | best | metadata-heavy | later optimization, not MVP |

Do not recreate the previously failed 512-thread GQA CTA design. Prefer KV-head/page/split-oriented grids and reduce across Q heads/GQA groups deliberately.

## Reference vs Production

`REFERENCE_BATCH`:

- may dispatch independent B=1 paths in a Python loop;
- used only for correctness comparison.

`PRODUCTION_BATCH`:

- must be a single batched operator launch or a fixed small number of batched stage launches;
- must not use Python B-times dispatch as serving implementation.
