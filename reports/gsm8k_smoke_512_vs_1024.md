# GSM8K Smoke Comparison

- A: `archive/gsm8k_20260803_195442_max512_post_eot_fix/results/gsm8k/smoke`
- B: `results/gsm8k/smoke_1024_existing_mixed_gpu`

| method | A rows | A acc | A trunc | B rows | B acc | B trunc |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 50 | 82.0 | 2 | 50 | 82.0 | 1 |
| kivi | 0 | None | 0 | 50 | 36.0 | 20 |
| patternkv | 0 | None | 0 | 50 | 84.0 | 1 |
