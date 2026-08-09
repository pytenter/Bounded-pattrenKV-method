# VarN Hadamard Dependency

`VARN_REQUIRES_HADAMARD_MATHEMATICALLY=False`.

Reason: The audited formula is invertible as X = balanced * s_col * s_row for any finite tile. It is defined on the tile it receives; canonical KVarN feeds it Hadamard-rotated K/V, but the scaling algebra itself does not require H.

`VARN_FUSED_WITH_HADAMARD_IMPLEMENTATION=True`.

Canonical KVarN uses Hadamard before VarN and its deployed attention backend is
written around rotated K/V. A clean Pattern+VarN intervention therefore still
needs a pinned VarN-only implementation path, even though the scaling formula
can be expressed independently.
