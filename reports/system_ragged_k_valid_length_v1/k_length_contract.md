# K Length Contract

Physical K workspace length is rectangular and equals the batch maximum. Logical validity is request-local: `request_packed_k_tokens` is authoritative for compressed K, while sink/pending/recent valid lengths are derived from `request_total_tokens`, sink/recent config, and packed K lengths.
