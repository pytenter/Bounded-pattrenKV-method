# Integration Callsite

`models/llama_patternkv.py` dispatches BI KProj from `LlamaFlashAttention_PatternKV.forward` only when `past_key_value is None` and `PATTERNKV_BATCH_INVARIANT_KPROJ=1`. Decode keeps `self.k_proj`.
