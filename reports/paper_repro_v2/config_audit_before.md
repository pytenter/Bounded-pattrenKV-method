# PatternKV Paper v2 Config Audit Before

Generated: 2026-08-04

## Repository State

- cwd: `/data/zypan/PatternKV-repro`
- current branch: `repro/patternkv-gsm8k-4x3090`
- actual HEAD: `358c633009d96d147d27fbc90e5207daa5f5aac7`
- requested reference commit in prompt: `5ba536a5c80366d6d384d9dca1e656325d36c436`
- status: dirty worktree with prior GSM8K, KIVI official, LongBench subset scripts/results; existing results were not moved or overwritten.

## Before Findings

- `bench/bench_longbench_patternkv.py` accepted only 8 LongBench tasks and defaulted `--max-input-length 8192`.
- `bench/longbench_config/__init__.py` exposed only the prior 8-task subset as `SUBTASKS`.
- `dataset2prompt.json` and `dataset2maxlen.json` contained 13 tasks, missing 8 of the 21 paper-panel tasks: `narrativeqa`, `multifieldqa_zh`, `musique`, `dureader`, `vcsum`, `lsht`, `passage_count`, `passage_retrieval_zh`.
- `kivi_official` used the official KIVI model class but did not distinguish `group_size=32` from the PatternKV paper baseline `group_size=128`.
- The completed directory `results/longbench_official_kivi_8x50/kivi_official` is a historical 8-task, 50-sample, `group_size=32` subset and must be treated as `longbench_subset_8x50`, not paper v2.
- PatternKV code path applies RoPE before pattern selection in `models/llama_patternkv.py`, and KIVI official code applies RoPE before K/V quantization in `models/llama_kivi.py`.
- Bit accounting in `bench/summarize_longbench.py` mixed paper-theoretical payload/metadata with current Python tensor storage and assumed a fixed residual window of 128 for averages.
