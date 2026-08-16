# Attention Value Reduction Call Graph

`models/llama_patternkv.py` segmented decode consumes physical-layout `attn_weights` after request-invariant softmax. Packed mixed V calls `patternkv_mixed_value_attention` -> `quant/page_batch.py:patternkv_fused_page_batch_decode` -> `quant/csrc/gemv_cuda.cu:page_mixed_pool_value_kernel`. Full precision sink/pending/recent call `request_invariant_full_value_attention` over request-local valid lengths. Outputs are merged in sink, packed, pending, recent order before `ATTENTION_PRE_O_PROJ`.
