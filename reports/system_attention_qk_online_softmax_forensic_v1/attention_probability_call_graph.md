# Attention Probability Call Graph

`LlamaFlashAttention_PatternKV.forward` projects Q/K/V, applies RoPE, appends decode K/V into the segmented cache, builds QK score parts in sink/packed/pending/recent order, concatenates them into a physical attention axis, divides by `sqrt(head_dim)`, applies `build_k_segment_validity_mask` with `torch.finfo(dtype).min` on invalid physical padding, then calls `torch.nn.functional.softmax(..., dtype=torch.float32).to(fp16)`.
