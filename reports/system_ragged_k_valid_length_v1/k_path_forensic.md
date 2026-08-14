# K Path Forensic

The old blocker came from passing padded ragged assignments into the compressed K reader while packed K payloads had been padded by logical tokens instead of compressed columns. `OC` is the logical compressed K token count implied by `packed_k.shape[-1] * (32 / bits)`. The repaired path pads only compressed payload columns, keeps assignment shape `[B, nh_kv, OC_max]`, and masks per-request invalid tails before softmax.
