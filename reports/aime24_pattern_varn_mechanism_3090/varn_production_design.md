# VarN-only Production Design

- Intervention: Pattern S16 packed-history tile magnitude balancing plus matching inverse metadata.
- Hadamard: disabled.
- K axis: post-RoPE residual tile [D, group], s_col per token, s_row per channel.
- V axis: post-projection adjusted residual tile [group, D], s_col per channel, s_row per token.
- Pattern centroids, assignments, gates, grouping, sink, recent, and pending semantics are unchanged.
- Decode: dequant balanced tile, apply inverse VarN scales, then restore Pattern centroid/base and attend.
- FP16 Sink/Recent/Pending regions bypass VarN.
