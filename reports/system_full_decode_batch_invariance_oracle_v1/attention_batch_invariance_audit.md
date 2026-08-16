# Attention Batch-Invariance Audit

Static audit: segmented decode attention uses the same semantic sink/packed/pending/recent order, with ragged valid-length masks. Under FULL_BI linear coverage, layer0 Q/K/V and RoPE outputs for request A are exact, but `ATTENTION_PRE_O_PROJ` is not exact. This localizes the first remaining full-BI divergence to attention/value reduction before O projection.
