# GSM8K Paper Rerun Protocol

- dataset: `openai/gsm8k`, config `main`, split `test`, 1319 examples.
- model: `/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct`
- methods: `fp16`, `kivi_paper_g128`, `patternkv_paper`
- prompt protocol: zero-shot CoT
- prompt source: reproduction choice, because exact PatternKV GSM8K prompt string was not found in local official code.
- prompt: `{question}\n\nLet's think step by step.`
- decoding: greedy, `do_sample=false`, `max_new_tokens=1024`, `batch_size=1`, `num_return_sequences=1`
- output: one atomic JSON per sample under `results/paper_repro_v2/gsm8k_full`
