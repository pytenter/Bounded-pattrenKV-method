# GSM8K Official KIVI vs PatternKV 50-Sample Comparison

- created_at: `2026-08-03T15:24:40Z`
- sample set: first 50 GSM8K test samples, `gsm8k:0` through `gsm8k:49`
- model: `/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct`
- prompt: local zero-shot CoT prompt used by the current GSM8K runner
- max_new_tokens: `1024`
- official KIVI config: `k_bits=2`, `v_bits=2`, `group_size=32`, `residual_length=32`

## Summary

| method | rows | correct | accuracy | truncated | normal EOS | errors |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 50 | 41 | 82.0% | 1 | 49/50 | 0 |
| patternkv | 50 | 42 | 84.0% | 1 | 49/50 | 0 |
| kivi_flexible_cache_old | 50 | 18 | 36.0% | 20 | 30/50 | 0 |
| kivi_official | 50 | 43 | 86.0% | 3 | 47/50 | 0 |

The old flexible-cache KIVI result is not a useful baseline for PatternKV. It collapses into repeated long generations on many GSM8K samples. After replacing it with the official KIVI model path, KIVI recovers to normal task quality.

## Paired Comparison

PatternKV vs official KIVI on the same 50 samples:

- both correct: `39`
- only PatternKV correct: `3`
- only official KIVI correct: `4`
- both wrong: `4`

PatternKV vs old flexible-cache KIVI:

- both correct: `18`
- only PatternKV correct: `24`
- only old flexible-cache KIVI correct: `0`
- both wrong: `8`

## Interpretation

The current evidence supports this conclusion:

1. PatternKV should be compared against `kivi_official`, not the old `kivi` flexible-cache path.
2. The severe 36% KIVI result was caused by implementation/protocol mismatch, not by official KIVI quality.
3. On this 50-sample GSM8K smoke, PatternKV and official KIVI are close: `84.0%` vs `86.0%`.
4. The official KIVI path still has slightly more length truncation than PatternKV here, `3` vs `1`, but it no longer shows runaway degeneration.

## Files

- official KIVI result: `results/gsm8k/kivi_official_smoke_1024_g32r32/smoke/kivi_official/all.jsonl`
- official KIVI summary: `reports/gsm8k_kivi_official_smoke_1024_g32r32.md`
- old three-method summary: `reports/gsm8k_smoke_1024.md`
- KIVI implementation audit: `reports/kivi_gsm8k_implementation_audit.md`

