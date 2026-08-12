# Phase S1 Reference Path Audit

## Scope

This audit covers the frozen CAUSAL_V4_25 mixed Value decode path on branch
`sys/causal-v4-25-kernel-v1`. It records the current reference semantics before
adding a compressed-domain fused Value attention path. The frozen algorithm tag
`causal-v4-25-aime24-v1` points to
`c73aeed3247c136859f695d5b238eeb357434b17`; this systems branch starts from
archive commit `d59c349b7ef9d02a45ea12870d3e9e717e96ec8e`.

## 1. PATTERN_BASE Fast Value Path

For non-mixed PatternKV Value precision, `models/llama_patternkv.py` calls
`cuda_attn_v_fused_with_base()` from `quant/matmul.py`. That wrapper reshapes
the Python tensors into the C++ extension layout and calls
`patternkv_gemv.attn_v_forward_cuda_outer_dim_with_base()`, implemented in
`quant/csrc/gemv_cuda.cu`.

The CUDA path computes, in compressed domain:

```text
attention weights
+ packed V residual
+ per-token/per-group scale and zero
+ Pattern centroid selected by assignment
+ Pattern mask gate
+ optional FP16 tail
-> [B, H, 1, D]
```

The fused base path does not materialize `[B, H, T, D]` historical FP16 Value.

## 2. CAUSAL Mixed Fallback

When `value_precision_is_mixed(cache.v_precision_selector)` is true, the current
decode path falls back to:

```text
reconstruct_packed_v(cache)
-> restore Pattern centroid with v_centroids/v_assignment_idx/v_pattern_mask
-> repeat_kv(...)
-> torch.matmul(attention weights, restored V)
```

The fallback appears in both rolling and chunked decode branches in
`models/llama_patternkv.py`. It is algorithmically authoritative but allocates
the complete historical FP16 Value tensor for the quantized region.

## 3. V2 Payload Layout

`packed_v` stores only low-precision V2 tokens in compact order. It is produced
by `_cat_mixed_packed_v()` through `quantize_pack_v_reference(low, group_size,
2)`.

Python shape before the matmul wrapper:

```text
packed_v:       [B, H_kv, T_v2, D / 16]
packed_v_scale: [B, H_kv, T_v2, D / group_size]
packed_v_zero:  [B, H_kv, T_v2, D / group_size]
```

The wrapper transposes to the extension layout:

```text
vq_:      [B * H_kv, D / 16, T_v2]
scale_:   [B * H_kv, D / group_size, T_v2]
zero_:    [B * H_kv, D / group_size, T_v2]
```

The frozen dequantization formula is:

```text
value = q * scale + zero
```

Here `zero` is the floating minimum/offset, not an integer zero point.

## 4. V4 Payload Layout

`packed_v4` stores only selected high-precision V4 tokens in compact order. It
is produced by `_cat_mixed_packed_v()` through `quantize_pack_v_reference(high,
group_size, 4)`.

Python shape before the matmul wrapper:

```text
packed_v4:       [B, H_kv, T_v4, D / 8]
packed_v4_scale: [B, H_kv, T_v4, D / group_size]
packed_v4_zero:  [B, H_kv, T_v4, D / group_size]
```

The wrapper transposes to:

```text
vq4_:     [B * H_kv, D / 8, T_v4]
scale4_:  [B * H_kv, D / group_size, T_v4]
zero4_:   [B * H_kv, D / group_size, T_v4]
```

V2 and V4 use independent payloads and independent scale/zero tensors. Phase S1
must keep this representation.

## 5. Scale/Zero Layout

For Value quantization, scale and zero are grouped along the head dimension.
With the frozen DeepSeek-R1-Distill-Llama-8B config, `head_dim=128` and
`group_size=128`, so there is one scale/zero group per token and KV head. The
code still supports the general `[D / group_size]` layout as long as `D` is
divisible by `group_size`.

## 6. Precision Mask Layout

`v_precision_mask` is a persistent logical-order metadata tensor:

```text
v_precision_mask: [B, T_quantized_history], uint8/bool semantics
```

`0` means the logical token is stored in the compact V2 payload. `1` means it is
stored in the compact V4 payload. The attention weights remain in logical token
order, so mixed fused execution must map:

```text
logical token index -> V2 compact ordinal or V4 compact ordinal
```

The current implementation uses the mask only during reconstruction, where
logical slots are filled from compact V2/V4 payloads.

## 7. Centroid Layout

Pattern centroids are stored as:

```text
v_centroids: [H_kv, M_centroids, D]
```

The CUDA base kernel computes centroid restoration inside the Value attention
kernel by accumulating attention mass per centroid assignment and adding
`sum_t alpha[t] * mask[t] * centroid[assignment[t]]` to the residual attention
result.

## 8. Assignment And Residual Mask Layout

Pattern metadata for quantized Value history is logical-order:

```text
v_assignment_idx: [B, H_kv, T_quantized_history]
v_pattern_mask:   [B, H_kv, T_quantized_history]
```

The frozen restored Value semantics are:

```text
V_restored = V_dequant + v_pattern_mask * v_centroids[v_assignment_idx]
```

For mixed execution, these metadata tensors must be gathered into the same
compact V2/V4 order as the payload being processed.

## 9. Sink / Pending / Recent Participation

Rolling decode splits attention into logical segments listed in
`value_parts`: sink, packed quantized history, pending, and recent. Sink,
pending, and recent are stored as FP16 tensors and are currently handled by
`torch.matmul(weights, repeat_kv(source, groups))`.

Chunked decode can pass `attn_f` and `v_full` to the existing fused low-bit
Value kernel for an FP16 pending tail. Phase S1 can reuse this philosophy:
fuse the quantized mixed history and keep small FP16 regions separate unless the
existing fused interface safely handles them.

## 10. Batch-Size And GQA

The frozen mixed packing code currently enforces batch size 1 in
`_cat_mixed_packed_v()` and `reconstruct_packed_v()`. Phase S1 therefore treats
`B=1` as the required support level and records `PHASE_S1_BATCH_SUPPORT=1`.

The DeepSeek-R1-Distill-Llama-8B config used here has:

```text
num_attention_heads = 32
num_key_value_heads = 8
head_dim = 128
```

So GQA is active. Any fused path must pass `nh=32` and `nh_kv=8` through the
existing head mapping semantics rather than assuming `H == H_kv`.

## Phase S1 Design Implication

The first implementation should preserve the current cache representation and
avoid full Value reconstruction by computing:

```text
output =
  Attn[V2 logical positions] @ V2_compact_restored
+ Attn[V4 logical positions] @ V4_compact_restored
+ optional FP16 segment contribution
```

This compressed-domain decomposition is mathematically equivalent to the
reference reconstruction path while avoiding the full `[B, H, T, D]` historical
FP16 Value tensor.
