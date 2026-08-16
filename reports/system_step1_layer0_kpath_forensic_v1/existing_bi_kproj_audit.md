# Existing BI KProj Audit

Static dispatch shows existing BI KProj is guarded by `patternkv_use_bi_prefill_kproj(past_key_value)`, which requires `past_key_value is None`; step1 decode has non-None segmented cache for both independent B1 and ragged B2, so both runtime paths bypass BI KProj and call `self.k_proj(hidden_states)`. B1 normal decode KProj calls: 0; ragged normal decode KProj calls: 0.
