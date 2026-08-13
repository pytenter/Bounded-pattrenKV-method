# Runtime Integration Map

| file | symbol | current call path | B semantics | current Value backend | required integration change | risk |
| --- | --- | --- | --- | --- | --- | --- |
| `models/llama_patternkv.py` | `patternkv_mixed_value_attention` | attention softmax -> mixed Value | B1 legacy, B>1 fused_page | `cuda_attn_v_mixed_fused_with_base` or `patternkv_fused_page_batch_decode` | select `fused_page` backend from cache pools | low |
| `models/segmented_cache.py` | `_cat_mixed_packed_v` | selector -> V2/V4 pack | request-local B | legacy global compact for B1, page pools for B>1 | append operator-ready pools per packed chunk | medium |
| `models/llama_patternkv.py` | segmented attention forward | QK -> softmax -> Value | fixed-length B | fused page Value only replaces Value point | no QK/softmax/selector changes | low |
