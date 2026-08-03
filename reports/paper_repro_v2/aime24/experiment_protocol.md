# AIME24 Resource-Bounded Protocol

## A. PatternKV Paper Explicit Settings

- Task family: AIME long CoT quality experiment.
- Methods compared: FP16, KIVI INT2, PatternKV INT2.
- This reproduction is resource-bounded: default `NUM_SAMPLES=2`, not paper Avg@8/Maj@8.

## B. Confirmed From Current Official-Code Reproduction

- `kivi_paper_g128`: `k_bits=2`, `v_bits=2`, `group_size=128`, `residual_length=128`, K per-channel, V per-token.
- `patternkv_paper`: `k_bits=2`, `v_bits=2`, `group_size=128`, `num_k_base=32`, `num_v_base=32`, `G_pattern=128`, post-RoPE pattern selection.
- No tensor parallel; each GPU loads one independent model.

## C. DeepSeek-R1 Evaluation Protocol Used Here

PatternKV paper does not publish an exact AIME prompt string. This framework uses the DeepSeek-R1 recommended math prompt:

```text
{problem}

Please reason step by step, and put your final answer within \boxed{}.
```

- system prompt: none
- chat template: tokenizer chat template
- `force_think_prefix=true` by default
- generation: `temperature=0.6`, `top_p=0.95`, `max_new_tokens=32768`, `do_sample=true`

## D. Resource-Limited Modification

- Default samples per problem: 2.
- Planned tasks: `30 problems × 2 samples × 3 methods = 180`.
- Future Avg@8 is supported by setting `NUM_SAMPLES=8`; existing `sample_id=0,1` results with the same config hash are skipped.
- Avg@2 must not be labeled as paper Avg@8.
