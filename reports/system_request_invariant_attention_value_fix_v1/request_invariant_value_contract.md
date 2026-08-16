# Request-Invariant Value Contract

For request r, Value split boundaries are defined over request-local logical valid KV positions using the same 128-token split contract as softmax. Physical pages/segments are storage mappings only. Packed fused-page reduction uses `metadata.seq_lens[b]`; full precision segments use explicit valid lengths from `k_segment_valid_lengths(cache)`.
