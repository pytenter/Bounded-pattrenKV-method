# Paper v2 Rerun Config Audit Before

Generated: 2026-08-04

Initial state:

- branch before rerun branch: `repro/patternkv-gsm8k-4x3090`
- expected HEAD: `4c6ac3b5df971f5d8e1ec636dffd3793aa78e6bd`
- actual HEAD: `4c6ac3b5df971f5d8e1ec636dffd3793aa78e6bd`
- worktree: dirty with legacy untracked GSM8K/LongBench/AIME files; no old result directories were deleted or overwritten.

## Source Classes

- A. Paper explicit: INT2 K/V, PatternKV vs KIVI/FP16, LongBench/GSM8K task families.
- B. Official/current code: KIVI G128/R128, PatternKV 32 K patterns + 32 V patterns, G_pattern=128, post-RoPE selection, LongBench 21 task config.
- C. Official eval protocol: GSM8K test split, LongBench official prompts/max_gen/scorer.
- D. Reproduction choice: exact GSM8K zero-shot CoT prompt string, GSM8K greedy `max_new_tokens=1024`, runner logging/schema, 8GPU staging.

## Findings

- LongBench runner already supports `fp16`, `kivi_paper_g128`, `patternkv_paper`; default max input is `31500`.
- LongBench 31500 previously OOMed on RTX 3090 24GB for long tasks; strict full must be gated by memory qualification.
- Existing legacy GSM8K runner was untracked and mixed old method names; this rerun adds a clean `bench/bench_gsm8k_paper.py` runner and isolated directories.
- GSM8K local data exists and was normalized to `datasets/gsm8k/gsm8k_test.jsonl`.
- Old `kivi_original_g32` remains available but is excluded from paper method lists.
