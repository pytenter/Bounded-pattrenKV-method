# Canonical Attention Contract

For request r, pre-O output is `sum_j P[r,j] * V[r,j]` over request-local logical valid KV positions. Logical order is sink, packed historical V, pending, then recent. Ragged physical padding and peer-driven packed width are outside the semantic index space.
