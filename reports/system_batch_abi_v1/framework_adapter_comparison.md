# Framework Adapter Comparison

| Option | Batch support | Ragged support | Continuous batching | Scheduler reuse | KV allocator reuse | Custom KV layout flexibility | RTX3090 compatibility | Effort | Debug complexity | Long-term value |
|---|---|---|---|---|---|---|---|---|---|---|
| Standalone | controlled | initially no, later yes | no | none | custom only | highest | highest | medium | lowest | proves ABI/operator |
| SGLang | strong | strong | strong | high | medium-high with custom pool | high | good if custom kernels target 3090 | high | medium | best next adapter |
| vLLM | strong | strong | strong | high | high with cache spec | medium-high but cache spec work is deeper | good but TurboQuant path may be newer/complex | very high | high | strong paper/system value |

## SGLang Feasibility

`PatternKVAttentionBackend` is feasible.

Ideal structure:

```text
Prefill: existing FlashInfer/Triton/FA backend
Decode: PatternKV custom backend/operator
```

Likely components:

- new attention backend class;
- metadata init method that builds `PatternKVBatchMetadata`;
- PatternKV KV pool or side pools for K/V2/V4/metadata;
- decode call path for `patternkv_batch_decode`;
- adapter from `ForwardBatch.req_pool_indices`, `seq_lens`, and `out_cache_loc`.

## vLLM Feasibility

Feasible but likely later.

Likely components:

- `PatternKVAttentionBackend`;
- `PatternKVCacheSpec` or quant mode;
- cache writer/slot mapping support for K/V2/V4/metadata pools;
- decode op integration using block tables and sequence lengths.

PatternKV must not be forced into ordinary FP16 KV layout; the vLLM TurboQuant design supports the argument for a backend-specific cache spec.

## FlashInfer Direct Reuse

- `flashinfer_direct_kernel_reuse=false`
- `flashinfer_metadata_design_reusable=true`

FlashInfer's paged metadata is reusable as design, not its current kernels.

## Preferred Order

1. Serving-native batch ABI.
2. PatternKV custom batched decode operator.
3. SGLang decode backend adapter.
4. vLLM native backend.
