# K Path Call Graph

- `LlamaForCausalLM_PatternKV.forward` calls `LlamaModel_PatternKV.forward`.
- `LlamaModel_PatternKV.forward` creates `position_ids` from segmented cache total tokens for decode and runs `embed_tokens(input_ids)`.
- `LlamaDecoderLayer_PatternKV.forward` records layer input, then applies `input_layernorm`.
- `LlamaFlashAttention_PatternKV.forward` receives RMSNorm output, runs Q/K/V projections, reshapes K to `[B, Hkv, T, D]`, applies RoPE, then appends current K to segmented cache.
