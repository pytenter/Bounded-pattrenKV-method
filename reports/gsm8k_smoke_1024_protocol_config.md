# GSM8K Smoke 1024 Protocol Config

- model_path: `/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct`
- tokenizer_path: `/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct`
- chat_template_present: `True`
- prompt_template: `{Question}Please reason step by step, and put your final answer within \boxed{}.`
- dtype: `float16`
- do_sample: `False`
- num_beams: `1`
- temperature: `None`
- top_p: `None`
- seed: `0`
- max_new_tokens: `1024`
- eos_token_ids: `[128001, 128008, 128009]`
- pad_token_id: `128009`
- use_cache: `True`
- data_path: `data/gsm8k/test.jsonl`

## Model Config
```json
{
  "model_type": "llama",
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 128256
}
```

## Quantization
```json
{
  "k_bits": 2,
  "v_bits": 2,
  "group_size": 128,
  "residual_length": 128,
  "num_k_base": 32,
  "num_v_base": 32,
  "kivi_axis_key": 1,
  "kivi_axis_value": 0,
  "kivi_asym": true,
  "kivi_compute_dtype": "torch.float16"
}
```
