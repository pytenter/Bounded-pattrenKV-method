# AIME24 Evidence

Status: CANONICAL_PRIMARY.

Source: `reports/aime24_full_causal25_quality_4gpu/full_aime24_quality_summary.json`.

Protocol: `deepseek_r1_recommended`, `do_sample=True`, temperature `0.6`, top_p `0.95`, max_new_tokens `32768`, seeds `42, 43, 44`, 30 questions, paired by problem and seed.

| Method | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| FP16 | 45 | 90 | 50.00% |
| Pattern Base | 32 | 90 | 35.56% |
| Random-25% | 36 | 90 | 40.00% |
| CAUSAL-V4@25% | 45 | 90 | 50.00% |

Paired bootstrap:

- CAUSAL - Random: mean `0.10012444444444443`, CI95 `[-0.011111111111111108, 0.21111111111111105]`; aggregate advantage, CI crosses zero.
- CAUSAL - Base: mean `0.14438888888888887`, CI95 `[0.04444444444444444, 0.24444444444444444]`; CI is positive.

Paper-safe language: CAUSAL matches FP16 aggregate accuracy in the tested three-seed AIME24 evaluation. Do not infer statistical equivalence from equal aggregate accuracy.
