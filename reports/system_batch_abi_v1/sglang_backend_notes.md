# SGLang Backend Notes

Sources:

- SGLang `base_attn_backend.py`: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/base_attn_backend.py
- SGLang `flashinfer_backend.py`: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py
- SGLang `memory_pool.py`: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py
- SGLang `forward_batch_info.py`: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py

## Observations

- `AttentionBackend` exposes metadata initialization hooks and separates graph-safe in-graph metadata from host-side out-graph metadata.
- The FlashInfer backend comments state that attention backends support extend/prefill and decode operators; FlashInfer is faster, Triton easier to customize.
- `ForwardBatch` carries core serving metadata: `forward_mode`, `batch_size`, `input_ids`, `req_pool_indices`, `seq_lens`, and `out_cache_loc`.
- Decode metadata is prepared from request pool indices and sequence lengths.
- KV pools and output cache locations are the allocator boundary: scheduler decides which requests are live and where new tokens are written.

## Minimal New Backend Surface

A `PatternKVAttentionBackend` would minimally need:

- `init_forward_metadata_out_graph(forward_batch)` to build/request PatternKV batch metadata from `req_pool_indices`, `seq_lens`, `out_cache_loc`, and PatternKV pool state;
- optional `init_forward_metadata_in_graph` only after CUDA graph constraints are understood;
- `forward_decode(q, k, v, layer, forward_batch, save_kv_cache=True)` or equivalent backend decode path to call the PatternKV batched decode operator;
- `forward_extend` behavior. The least risky design is to continue using an existing prefill/extend backend initially, then only switch decode to PatternKV.

## Scheduler vs Backend vs Allocator

- Scheduler: chooses live requests, batching, prefix reuse, decode/extend mode, and produces request ids, sequence lengths, and output cache locations.
- KV allocator/pool: owns physical slots/pages and maps request/token positions to physical storage.
- Backend: turns scheduler and allocator metadata into kernel-ready tensors and launches attention operators.
- PatternKV custom operator: consumes `PatternKVBatchMetadata` and quantized pools; it should not own scheduling.

## Can Decode Use a Custom Backend?

Yes in design. SGLang's backend boundary is exactly where a custom decode operator belongs. The clean split is:

```text
Prefill/extend: existing backend
Decode: PatternKVAttentionBackend -> PatternKVBatchedDecodeOperator
```

This avoids rewriting scheduling and avoids forcing PatternKV prefill into a custom kernel before the decode ABI is proven.

## Ragged Requests

SGLang represents ragged serving batches through request pool indices, sequence lengths, and cache locations, then each backend derives page/block indices. PatternKV should follow that boundary: no dense `[B,H,T,D]` assumption should be required at the backend/operator interface.

## PatternKV Reuse

Directly reusable:

- backend split between metadata initialization and operator calls;
- request pool indices, sequence length arrays, and output cache locations;
- option to use existing prefill and custom decode.

Conceptually reusable:

- CUDA graph metadata discipline;
- KV pool ownership by serving runtime.

Not directly reusable:

- existing FP/FP8 KV pool layout as-is, because PatternKV needs K INT2 plus dual independent V2/V4 streams and precision metadata.
