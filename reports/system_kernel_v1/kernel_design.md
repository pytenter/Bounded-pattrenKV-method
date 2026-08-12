# Phase S1 Mixed V2/V4 Value Attention Kernel Design

## Goal

The frozen CAUSAL_V4_25 algorithm stores quantized historical Value tokens in
two compact payloads:

- V2 residual payload: `packed_v`, `packed_v_scale`, `packed_v_zero`
- V4 residual payload: `packed_v4`, `packed_v4_scale`, `packed_v4_zero`

The frozen reference path reconstructs a full logical-order FP16 Value tensor
before `torch.matmul`. Phase S1 adds `cuda_attn_v_mixed_fused_with_base()` to
avoid that reconstruction in fused mode.

## Chosen Decomposition

The implementation uses compressed-domain two-pass decomposition:

```text
output =
  Attn[V2 logical positions] @ V2_compact_restored
+ Attn[V4 logical positions] @ V4_compact_restored
+ optional FP16 full segment
```

Each compact pass reuses the existing CUDA fused Value kernel
`cuda_attn_v_fused_with_base()`, which already fuses:

- packed residual unpack
- `q * scale + zero` dequantization
- Pattern centroid restoration
- GQA head mapping
- attention-weighted accumulation

This is a fused compressed-domain Value attention path. It does not materialize
the full `[B, H, T, D]` historical FP16 Value tensor.

## Mapping Strategy

Phase S1 keeps the current cache ABI and uses the existing
`v_precision_mask` to gather logical attention weights and Pattern metadata into
compact V2/V4 order:

```text
logical attention weights -> V2 compact attention
logical attention weights -> V4 compact attention
logical v_pattern_mask    -> V2/V4 compact masks
logical v_assignment_idx  -> V2/V4 compact assignments
```

No persistent rank/index metadata is introduced in this phase. The temporary
gather cost is counted in `fused_temp_bytes`.

## Backend Switch

`PATTERNKV_MIXED_V_BACKEND` controls integration:

- `fused`: use `cuda_attn_v_mixed_fused_with_base()`
- `reference`: use the frozen `reconstruct_packed_v() + torch.matmul` oracle

The default is `fused` on the systems branch. The reference path is preserved
for regression testing.

## Unsupported Conditions

The fused path raises a clear error rather than silently falling back when:

- batch size is not 1
- Pattern centroid/mask/assignment metadata is missing
- `v_precision_mask` is missing
- compact payload token counts disagree with the precision mask

`PHASE_S1_BATCH_SUPPORT = 1`, matching the frozen mixed cache packing
restriction.

## Algorithm Semantics

The implementation does not change:

- K/V bits
- V4 budget
- selector semantics
- importance accumulation
- local V2/V4 gain
- centroid selection
- scale/zero calculation
- quantization formula
- packing format

The formal 2.500488 bit/element algorithm-budget metric remains unchanged.
