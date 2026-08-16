# Attention Value Call Graph

`LlamaFlashAttention_PatternKV.forward` builds softmax `attn_weights`, updates causal importance, then in rolling segmented mode iterates `value_parts` in sink/packed/pending/recent order. Packed mixed V uses `patternkv_mixed_value_attention`; with `PATTERNKV_MIXED_V_BACKEND=fused_page` this calls `patternkv_fused_page_batch_decode`, which launches `patternkv_gemv.attn_v_forward_cuda_page_mixed_pool`. Full precision sink/pending/recent tails use `torch.matmul`; segment outputs are added into pre-O output.
