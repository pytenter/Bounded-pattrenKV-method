# Request-Invariant Softmax Contract

Split boundaries are defined over request-local logical valid KV indices in canonical sink, packed, pending, recent order. `SplitBoundaries(r) = F(valid_logical_kv_length, fixed_split_size=128)`. Physical segment offsets are only a storage mapping.
