# CAUSAL-V4@25% Systems Handoff

## Frozen Semantics

The following semantics are frozen and must not be changed by systems optimization:

- K2
- V2/V4
- 25% V4 budget
- Sink16
- Recent128
- group128
- causal selector
- importance score
- local V2-to-V4 gain
- centroid semantics
- scale/zero semantics

## Current Reference Mixed-V Storage

The current code uses:

- `packed_v`
- `packed_v_scale`
- `packed_v_zero`
- `packed_v4`
- `packed_v4_scale`
- `packed_v4_zero`
- `v_precision_mask`

## Current Primary Decode Bottleneck

Current mixed V2/V4 path:

```text
mixed V2/V4 path
    ->
reconstruct_packed_v()
    ->
materialize full FP16 historical V
    ->
torch.matmul()
```

This is the first systems optimization target.

## Existing Fast Baseline Infrastructure

The repository already contains:

- `cuda_bmm_fA_qB_outer`
- `cuda_bmm_fA_qB_outer_with_base`
- `cuda_attn_v_fused_with_base`
- `quant/csrc/gemv_cuda.cu`

PATTERN_BASE already has fused low-bit Value attention.

CAUSAL_V4_25 currently falls back to explicit V reconstruction for the mixed V2/V4 path.

## Systems Roadmap

### Phase S1 - Standalone Mixed-V Kernel

Target interface:

```text
cuda_attn_v_mixed_fused_with_base
```

Requirements:

- V2 unpack
- V4 unpack
- scale/zero
- Pattern centroid restore
- attention-weighted accumulation
- all fused

Do not materialize a `[B, H, T, D]` full historical FP16 V tensor.

### Phase S2 - GPU Selector

Eliminate:

- Python per-token loop
- `.item()`
- Python sort

First implement a GPU top-k equivalent version with bit-exact selected-token identity against the frozen reference.

### Phase S3 - Fixed Tile ABI

For each complete 128-token tile:

- 96 V2 tokens
- 32 V4 tokens

The selected token identities are dynamic; the memory budget per page/tile should be fixed.

### Phase S4 - vLLM

Future work:

- custom AttentionBackend
- custom KV storage
- store kernel
- decode kernel
- PagedAttention-compatible layout

### Phase S5 - SGLang

Reuse as much of the same packed layout and CUDA/Triton primitives as possible.

## Not Started In This Freeze

- Kernel implementation: NO
- vLLM integration: NO
- SGLang integration: NO
