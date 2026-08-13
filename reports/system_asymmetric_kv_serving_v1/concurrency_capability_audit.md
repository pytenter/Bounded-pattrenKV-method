# Concurrency Capability Audit

## Verdict

- `CONCURRENCY_RUNTIME_BLOCKED`.
- The current frozen mixed V2/V4 runtime is single-request only for true batched serving.
- S6-B performance stages were not run because serial pseudo-concurrency is explicitly disallowed.

## Questions

| Question | Answer | Evidence |
|---|---|---|
| cache object supports B>1? | Partial | Base tensor fields carry batch dim, but mixed-V packing rejects B>1. |
| packed K/V carries real B dim? | Partial | K/V pack tensors include B dim; mixed split assumes request 0 mask. |
| assignments/masks support B>1? | Partial | Shapes include B, but compact V2/V4 split is generated from `precision_mask[0]`. |
| capacity buffer allows batch>1? | Likely yes for raw tensor storage | It stores shape_except_token including B, but cannot solve variable per-request mixed split. |
| mixed-V kernel batch-aware? | No | `cuda_attn_v_mixed_fused_with_base` raises for `B != 1`. |
| QK kernel batch-aware? | Likely yes | QK path consumes tensors with B dim and has no B=1 guard in the production tight reader. |
| selector request-isolated? | Scoring is batch-shaped; downstream storage is not | Selector can return `[B,T]`, but packer consumes `precision_mask[0]`. |
| append_decode handles batch? | Partial | Generic append can carry B tensors, but mixed packing raises before/at flush. |
| hard-coded B=1 exists? | Yes | See source evidence below. |
| model forward can advance multiple requests? | No for frozen causal_v4 mixed-V path | Batch prefill/decode hits B=1 mixed-V limitations. |

## Source Evidence

| File | Line | Evidence |
|---|---:|---|
| `models/segmented_cache.py` | 1150 | `raise ValueError("mixed Value precision currently requires batch size 1")` |
| `models/segmented_cache.py` | 1206 | `raise ValueError("mixed Value precision currently requires batch size 1")` |
| `quant/matmul.py` | 1191 | `raise RuntimeError("Phase S1 mixed fused Value attention currently supports B=1, matching frozen mixed cache packing")` |
| `models/segmented_cache.py` | 1151 | `mask = precision_mask[0].bool()` |
| `quant/matmul.py` | 1202 | `mask = precision_mask[0].bool()` |
| `models/segmented_cache.py` | 1152 | `low = v_adjusted[:, :, ~mask, :].contiguous()` |
| `quant/matmul.py` | 1226 | `attn2 = attn_q[..., low_mask].contiguous()` |

## Runtime Probes

| Probe | Status | Exception | Message |
|---|---|---|---|
| build_cache_from_prefill_batch2_mixed | EXPECTED_BLOCKED | ValueError | mixed Value precision currently requires batch size 1 |
| mixed_v_fused_kernel_entry_batch2 | EXPECTED_BLOCKED | RuntimeError | Phase S1 mixed fused Value attention currently supports B=1, matching frozen mixed cache packing |
