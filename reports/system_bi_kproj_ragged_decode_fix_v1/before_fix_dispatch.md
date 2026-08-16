# Before Fix Dispatch

Before this round, `models/llama_patternkv.py` called `batch_invariant_k_projection` only under `patternkv_use_bi_prefill_kproj(past_key_value)`, so decode with non-None cache used `self.k_proj(hidden_states)`. S6-B.3.4H recorded B1/Ragged decode `used_bi_kproj=false`, normal decode KProj calls=32, and shapes `[1,1,4096]` vs `[2,1,4096]`.
