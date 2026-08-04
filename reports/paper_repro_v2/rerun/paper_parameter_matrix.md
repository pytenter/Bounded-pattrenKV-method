# Paper Parameter Matrix

| 参数 | 论文明确 | 官方代码 | 当前设置 | 来源类别 | 是否一致 |
|---|---|---|---|---|---|
| 模型 | Llama-family 8B experiments | local Llama runner | `/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct` | D | yes for current reproduction |
| k_bits | INT2 | config.k_bits | 2 | A/B | yes |
| v_bits | INT2 | config.v_bits | 2 | A/B | yes |
| group_size | G128 for paper KIVI baseline | quantizer receives `group_size` | 128 | A/B | yes |
| residual_length | 128 | model cache residual window | 128 | B | yes |
| initial patterns | 32 | `num_k_base`, `num_v_base` | 32 K + 32 V | A/B | yes |
| G_pattern | 128 | decode update on residual window | 128 | A/B | yes |
| K axis | per-channel | K transposed before pack | per-channel | B | yes |
| V axis | per-token | V packed over head_dim | per-token | B | yes |
| selection position | post-RoPE | PatternKV code after RoPE | post-RoPE | B | yes |
| GSM8K prompt | zero-shot CoT, exact string not verified | no official local GSM8K runner | `{question}\\n\\nLet's think step by step.` | C/D | labeled reproduction choice |
| GSM8K decoding | not fully public | current runner | greedy, max_new_tokens=1024 | D | labeled reproduction choice |
| LongBench tasks | LongBench full | config has 21 tasks | 21 tasks | C | yes |
| LongBench max input | expected 31500 | `DEFAULT_INPUT_CAP=31500` | 31500 strict | B/D | yes, runtime gated |
| LongBench prompt | official LongBench | frozen JSON | task-specific | C | yes |
| LongBench max_gen | official LongBench | frozen JSON | task-specific | C | yes |
