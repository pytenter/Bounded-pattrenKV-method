# PatternKV Batched Decode Operator Spec

Status: proposed operator contract for S6-B.1.

## Inputs

```text
q: [B,Hq,1,D] fp16/bf16

K:
  k_pool
  k_scale
  k_zero
  k_assignment
  k_centroids [Hkv,Mk,D]

V2:
  v2_pool
  v2_scale
  v2_zero
  v2_pattern_gate
  v2_assignment

V4:
  v4_pool
  v4_scale
  v4_zero
  v4_pattern_gate
  v4_assignment

metadata:
  PatternKVBatchMetadata

V shared:
  v_centroids [Hkv,Mv,D]
```

Output:

```text
out: [B,Hq,1,D]
optional lse/debug counters
```

## Work Decomposition

Recommended decode structure:

1. QK stage:
   - work items over `(request, query_head, K page/split)`;
   - read request-local K pages from `k_page_table`;
   - preserve tight K page layout;
   - produce partial logits/LSE state.

2. Softmax/reduction:
   - reduce split-KV partials per `(request, head)`;
   - maintain exact causal length from `seq_lens`.

3. Mixed V accumulation:
   - iterate logical pages/tokens;
   - use precision bitmap to route each token to V2 or V4 physical page/rank;
   - dequantize V2/V4 using its own affine scale/zero;
   - restore centroid contribution using compact Pattern gate and assignment;
   - accumulate weighted Value.

The first standalone implementation can separate QK and V stages. A later fused design can share page iteration and softmax state.

## Required Invariants

- `seq_lens[b] <= num_pages[b] * page_size`.
- last page valid length is `seq_lens[b] % page_size`, with `page_size` when remainder is zero.
- `v2_counts[p] + v4_counts[p] == valid_tokens[p]`.
- V2/V4 compact pattern metadata lengths match V2/V4 payload counts.
- K/V request identity is carried only through page tables and request ids, not through implicit dense B layout.
- V2 and V4 affine streams are independent. No shared bitplane or residual precision boost is allowed.

## Append/Write Contract

For each decode step:

```text
for each active request b:
  scheduler/allocator provides out_cache_loc or target logical page
  backend computes selector decision for new eligible window/page
  append K to tight K page/block
  append adjusted V to V2 or V4 stream page
  append precision bit and page-local count metadata
  append compact Pattern metadata to the same stream
```

The selector can remain semantically unchanged, but its output must be committed through page-local metadata, not row-0 dense compaction.

## Standalone vs SGLang/vLLM

Recommended minimum path:

1. Define `PatternKVBatchMetadata` as CPU/Python dataclass plus torch tensors.
2. Implement a reference batched reader that uses metadata for correctness on synthetic B>1 ragged masks.
3. Implement standalone CUDA/Triton decode for the proposed ABI.
4. Add an adapter that maps SGLang/vLLM request/page tables to `PatternKVBatchMetadata`.
5. Only then wire a `PatternKVAttentionBackend` for decode.

Do not begin by attaching formal SGLang/vLLM serving; the current missing piece is the operator/cache ABI.

## Success Criteria

- B=2 synthetic requests with different precision patterns produce correct outputs against a materialized reference.
- No Python per-request loop in production decode.
- No full historical V materialization.
- K path remains tight and asymmetric.
- Metadata overhead is bounded and quantified.
