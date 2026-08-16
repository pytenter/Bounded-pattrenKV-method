# Profile Before/After

Diagnostic profile: C2048 B1 decode=8, `PATTERNKV_SYSTEM_PROFILE=1`, GPU1, same loaded model process.

| Component | Old Calls | Old ms/run | New Calls | New ms/run |
| --- | ---: | ---: | ---: | ---: |
| model_decode | 8 | 1991.126 | 8 | 2812.120 |
| attention_score_concat | 256 | 14.693 | 0 | 0.000 |
| attention_softmax | 256 | 145.315 | 1024 | 59.849 |
| fixed_split_softmax_kernel | 256 | 19.999 | 0 | 0.000 |
| qk_int2_history | 256 | 40.642 | 256 | 39.251 |
| qk_fp16_regions | 768 | 96.701 | 768 | 99.152 |
| mixed_historical_value | 0 | 0.000 | 256 | 52.956 |
| value_fp16_tail | 768 | 337.181 | 768 | 256.140 |
| importance_update | 256 | 36.333 | 1024 | 213.118 |
| cache_append | 256 | 149.797 | 256 | 150.321 |
| state_generation_calls | 0 | 0.000 | 1024 | 0.000 |
| state_merge_calls | 0 | 0.000 | 768 | 0.000 |
| global_score_concat_calls | 256 | 0.000 | 0 | 0.000 |
| global_probability_materialization_calls | 256 | 0.000 | 0 | 0.000 |

Interpretation: the intended global intermediates are removed, but V1 pays more in segment-local softmax/state generation, mixed historical Value state construction, and per-segment causal-importance updates. QK duplication was found in an intermediate draft and fixed before the final profile.

