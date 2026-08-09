# Mixed Precision Value Cache Audit

## Existing V2 Layout

- Production segmented PatternKV stores packed Value history on `PatternQuantizedKVCache.packed_v`.
- `packed_v` shape is `[batch, kv_heads, packed_tokens, packed_head_dim]`.
- For INT2, `packed_head_dim = head_dim / 16` because 16 two-bit elements fit in one int32 lane.
- `packed_v_scale` and `packed_v_zero` are stored separately with shape `[batch, kv_heads, packed_tokens, head_dim / group_size]`.
- `quantize_pack_v_reference(v, group_size, bits)` uses affine quantization along head_dim: per token and KV head, it groups the last dimension, stores min as `zero`, stores `(max-min)/(2**bits-1)` as `scale`, rounds/clamps to unsigned integer levels, and packs along the last dimension.
- `dequantize_v_reference(...)` reverses that layout, producing `[batch, kv_heads, packed_tokens, head_dim]`.

## Pattern Value Semantics

- Pattern Value assignment is stored separately from the payload:
  - `v_assignment_idx`: centroid index per `[batch, kv_head, token]`.
  - `v_pattern_mask` / `v_assignments`: whether the selected centroid was subtracted before quantization and must be restored after dequantization.
- The packed Value payload is the adjusted Value:
  `v_adjusted = v - mask * centroid`.
- Read reconstruction is:
  `dequant(v_adjusted_payload) + mask * centroid`.
- Experiment 8 freezes this centroid assignment to the BASE minmax objective for every config.

## Pack Window

- Rolling segmented mode flushes one pack window whenever `pending_k` reaches `group_size`; under the frozen baseline this is 128 tokens.
- Static prefill can flush multiple full windows from the prompt+prefix history.
- Sink and Recent remain FP16. Selective V2/V4 applies only to packed Value history.

## Mixed-Bit Replacement Feasibility

Clean replacement is implementable without storing duplicate V2+V4 copies:

- Store a token-level `v_precision_mask` aligned to logical packed history.
- For selected logical tokens, store adjusted Value only in a V4 payload.
- For unselected logical tokens, store adjusted Value only in the existing V2 payload.
- Store separate scale/zero metadata for V2 and V4 payloads.
- Reconstruct packed history by dequantizing each payload and scattering back into original token order using `v_precision_mask`.

This requires the packed Value token count to remain the logical packed history count, while physical V2/V4 payload token counts are derived from the mask. The current `packed_v_tokens` can remain logical and still match `packed_k_tokens`.

## Production Read Hook

The current CUDA fused Value matmul accepts only one payload and one bitwidth. It cannot directly consume mixed V2/V4 payloads. The clean path for this screen is:

1. Reconstruct packed Value history in original token order from V2/V4 replacement payloads.
2. Restore PatternKV centroids after dequantization.
3. Use dense matmul for the packed Value segment.

This changes runtime speed, not storage semantics. The stored cache remains true selective replacement.

## Storage Accounting

The implementation must count:

- Value payload bits: V2 payload tokens use 2 bits/element; V4 payload tokens use 4 bits/element.
- Value affine metadata: scale and zero per payload token, KV head, and head_dim group.
- Pattern centroid metadata: unchanged across BASE/RANDOM/CAUSAL/ORACLE.
- Precision metadata: token-level bitmap or equivalent indices.

At exactly 12.5% selected tokens, the Value payload sanity check is:
`0.875 * 2 + 0.125 * 4 = 2.25` bits/value element.

## Gate

`MIXED_VALUE_PRECISION_IMPLEMENTABLE=true`.

No duplicate V2+V4 override sidecar is needed or allowed.
