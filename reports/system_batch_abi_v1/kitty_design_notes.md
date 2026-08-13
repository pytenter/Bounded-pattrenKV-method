# Kitty Design Notes

Sources:

- Kitty arXiv entry: https://arxiv.org/abs/2511.18643
- Kitty OpenReview PDF: https://openreview.net/pdf?id=r3mQiuYKIN
- Kitty paper HTML mirror: https://arxiv.org/html/2511.18643v1
- Kitty implementation link from paper: https://github.com/Summer-Summer/Kitty

## Relevant Ideas

Kitty targets mixed-precision KV cache serving with a system design based on:

- page-centric KV cache layout;
- Triton-compatible dequantization kernels;
- runtime pipeline that avoids scattered access and warp divergence.

The paper summary says the main systems challenge is dynamic higher-precision boosts while preserving coalesced page layout and uniform dequantization.

## Compatibility Classification

| Idea | Classification | Reason |
|---|---|---|
| Page-centric layout | CONCEPTUALLY REUSABLE | PatternKV also needs ragged serving and page-local metadata. |
| Avoid scattered reads | CONCEPTUALLY REUSABLE | PatternKV should avoid global searches or per-token hash lookups. |
| Triton/custom dequant kernel for pages | CONCEPTUALLY REUSABLE | PatternKV decode will need custom readers for V2/V4 plus centroid restoration. |
| Dynamic mixed precision metadata | CONCEPTUALLY REUSABLE | PatternKV needs precision bitmap/rank metadata, but semantics differ. |
| 2-bit base plus precision boost / enhancement design | NOT COMPATIBLE | PatternKV frozen algorithm requires independent affine V2 and V4 streams, not residual/enhancement bits. |
| Channel-wise precision boost | NOT COMPATIBLE as algorithm | PatternKV selects tokens for V4, not channels, and must keep V2/V4 independent. |

## PatternKV Constraint

Kitty can inspire the serving layout but not the quantization semantics. PatternKV must not collapse V2 and V4 into shared bitplanes or a residual precision boost. The right reuse is page-level organization and kernel scheduling discipline, not the bit representation.
