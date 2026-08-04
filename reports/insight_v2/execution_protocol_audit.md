# Insight V2 Execution Protocol Audit

- generated_at: `2026-08-04T14:05:23.512539+00:00`
- branch: `exp/patternkv-parity-microsmoke-wave-a`
- runtime_commit: `18e2f788ba47225e251a94cb2606d53e2203294a`
- working_tree_dirty: `True`

## Environment

- python: `/data/zypan/.local/share/mamba/envs/patternkv/bin/python`
- model_path: `/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct`
- model_config_exists: `True`
- tokenizer_config_exists: `True`
- safetensors_count: `4 `
- gsm8k_data: `datasets/gsm8k/gsm8k_test.jsonl`, exists=`True`
- historical_longbench_data_dir: `/root/Block-kvcache-experiment/data/LongBench`
- historical_longbench_hotpotqa_readable_by_current_user: `permission_denied`

## LongBench Protocol

- `bench/bench_pattern_insight.py` delegates generation to `scripts/run_longbench_paper_8k_single4090.py`.
- LongBench model is `Llama-3.1-8B-Instruct` through `--model-path`.
- LongBench max input defaults to 8192 in the insight runner.
- Prompt rendering and middle truncation are reused from the official LongBench reproduction path via `encode_prompt` and `build_prompt`.
- Generation uses greedy decoding (`do_sample=False`, `num_beams=1`).
- Batch size is one sample per process.
- Task-specific `MAX_NEW_TOKENS[task]` is used by `lb_runner.run_one`; the generic `--max-new-tokens` flag is not passed into LongBench generation.
- Scoring uses `bench._longbench_scorer.score_example` through `lb_runner.run_one`.

## GSM8K Protocol

- `bench/bench_pattern_insight.py` delegates generation to `bench/bench_gsm8k_paper.py`.
- Prompt rendering reuses `bench.gsm8k_paper_utils.build_prompt`: `{question}

Let's think step by step.`
- Generation uses greedy decoding (`do_sample=False`, `num_beams=1`).
- EOS handling and parser are reused from `bench/bench_gsm8k_paper.py` and `bench/gsm8k_paper_utils.py`.
- GSM8K `max_new_tokens` is controlled by the insight runner argument; for parity/micro-smoke it must be run with `--max-new-tokens 2048`.

## PatternKV Canonical Config

- Method is restricted to `patternkv_paper`.
- `bench.paper_config.apply_method_defaults` enforces K2/V2, group_size=128, residual_length=128, num_k_base=32, num_v_base=32.
- PatternKV implementation uses post-RoPE key/value states in `models/llama_patternkv.py`.
- K reference quantization is per-channel along token groups after transpose; V reference quantization is per-token along head dimension.

## Blocking Findings

- Current user cannot read `/root/Block-kvcache-experiment/data/LongBench/data/*.jsonl` (`Permission denied` observed during environment audit).
- No alternate readable LongBench source JSONL or ZIP was found under `/data/zypan` or `/data` during this run.
- Therefore LongBench parity cannot be executed without either granting read access to the existing LongBench data directory or providing a readable local LongBench data copy. Re-downloading data was not attempted because the task forbids it.
