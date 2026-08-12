# PatternKV Systems Optimization Plan

## Frozen Algorithm Boundary

The frozen algorithm checkpoint is `causal-v4-25-aime24-v1`, pointing to
`c73aeed3247c136859f695d5b238eeb357434b17`. Systems work may optimize
implementation but must not change CAUSAL_V4_25 semantics:

- K2
- V2/V4
- 25% V4 budget
- Sink16
- Recent128
- Residual128
- group128
- causal_v4 selector
- importance score
- local V2-to-V4 gain
- centroid and scale/zero semantics

## Phase S1

Implement a compressed-domain mixed V2/V4 Value attention path that avoids full
historical FP16 Value reconstruction. Keep the current split payload cache ABI:

- `packed_v`
- `packed_v_scale`
- `packed_v_zero`
- `packed_v4`
- `packed_v4_scale`
- `packed_v4_zero`
- `v_precision_mask`

## Phase S2

Move selector top-k work to GPU while preserving bit-exact selected-token
identity against the frozen reference.

## Phase S1.5 Profile Finding

Post-S1 decode profiling is observation-only. The controlled profile found that
the mixed-V fused attention path remains the largest measured fused-backend
component at 32K context, followed by dynamic cache mutation and QK. Selector,
causal-importance update, and packing were not dominant. The recorded next
phase is `PROFILE_INCONCLUSIVE` because the listed S2/S3 options do not directly
target the largest remaining component.

## Phase S2A Deep Profile Finding

S2A split the two remaining candidates. Mixed-V is dominated by CUDA execution,
especially the V2 lane, while host/layout overhead is small. Cache mutation is
visible but copy bytes per token grew only `1.172x` from 16K to 32K in the
controlled profile. The next phase is therefore
`S2B_MIXED_V_KERNEL_OPTIMIZATION`; fixed-page ABI remains relevant later, but it
is not the immediate root-cause target.

## Phase S3

Design a fixed tile ABI for full 128-token tiles with 96 V2 tokens and 32 V4
tokens. The selected token identities may vary; the physical budget per tile
should become fixed.

## Phase S4

Prototype vLLM integration with custom KV storage and a decode kernel compatible
with PagedAttention-style scheduling.

## Phase S5

Reuse the packed layout and CUDA/Triton primitives for SGLang integration where
possible.
