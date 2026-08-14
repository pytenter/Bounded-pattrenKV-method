# Strict Mode Scope

Historical `normal` has no batch-invariance guarantee. Recommended serving mode is `bi_k`. `bi_kv` means strict prefill K/V projection invariance under the same hidden input; it does not claim whole-model bitwise determinism. `PATTERNKV_BI_MLP_ORACLE` is diagnostic-only and default-off.
