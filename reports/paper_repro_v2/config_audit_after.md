# PatternKV Paper v2 Config Audit After

Generated: 2026-08-04

## Status

PARTIAL PASS.

The paper v2 code/config alignment is in place and the 8192-token compatibility smoke passes. The strict `max_input_length=31500` smoke is partial on RTX 3090 24GB because `passage_retrieval_en` and `lcc` hit CUDA OOM for all three methods.

## Updated Configuration

- LongBench task panel is now 21 tasks: `narrativeqa`, `qasper`, `multifieldqa_en`, `multifieldqa_zh`, `hotpotqa`, `2wikimqa`, `musique`, `dureader`, `gov_report`, `qmsum`, `multi_news`, `vcsum`, `trec`, `triviaqa`, `samsum`, `lsht`, `passage_count`, `passage_retrieval_en`, `passage_retrieval_zh`, `lcc`, `repobench-p`.
- Official prompt/max-gen config was refreshed from `THUDM/LongBench/main/LongBench/config`.
- `longbench_subset_8x50` is preserved as the historical 8-task subset and is not treated as paper v2.
- Default paper input cap is `31500`; runner records raw prompt tokens, truncated tokens, generated tokens, and truncation flag per sample.
- `configs/longbench_paper_v2.yaml` and `configs/longbench_paper_v2_smoke.yaml` were added.

## Method Names

- `fp16`: unquantized baseline, single GPU, no tensor parallel.
- `kivi_paper_g128`: official KIVI model class with `k_bits=2`, `v_bits=2`, `group_size=128`, `residual_length=128`, K per-channel, V per-token, asymmetric affine metadata.
- `kivi_original_g32`: preserved compatibility alias for the original `group_size=32` setting; it is not the PatternKV paper baseline.
- `patternkv_paper`: PatternKV model class with `k_bits=2`, `v_bits=2`, `group_size=128`, `residual_length=128`, `num_k_base=32`, `num_v_base=32`, `G_pattern=128`, post-RoPE pattern selection.

## Runtime Checks

- Runner prints one `[PaperConfigCheck]` JSON line per worker.
- Result rows include `paper_config_snapshot` and `cache_bitwidth_stats`.
- KIVI G128 quantized-region theoretical bitwidth is asserted as `2.25`.
- PatternKV boundary unit test asserts no dynamic pattern before 128 tokens and a boundary event at token 128/129.

## Smoke

- Strict paper smoke at `max_input_length=31500`: PARTIAL due 24GB OOM on `passage_retrieval_en` and `lcc`.
- Compatibility smoke at `max_input_length=8192`: PASS for `fp16`, `kivi_paper_g128`, `patternkv_paper` on `qasper`, `passage_retrieval_en`, `lcc`, 1 sample each.
- Smoke report: `reports/paper_repro_v2/smoke_test_report.md`.
- Summary report: `reports/paper_repro_v2/smoke_8192_summary.md`.

## Tests

- `/data/zypan/.local/share/mamba/envs/patternkv/bin/python -m pytest tests/test_paper_v2_config.py -q`: 3 passed.
- `/data/zypan/.local/share/mamba/envs/patternkv/bin/python -m pytest tests/test_paper_v2_config.py tests/test_gsm8k_parser.py tests/test_gsm8k_stop_reason.py -q`: 17 passed.

## Full Run Script

Created `scripts/run_longbench_paper_v2_6gpu.sh`. It uses 6 workers, no tensor parallel, one process per visible GPU, 21 tasks split `4+4+4+3+3+3`, and each GPU serially runs `fp16`, `kivi_paper_g128`, then `patternkv_paper`.

Default full launch command:

```bash
bash scripts/run_longbench_paper_v2_6gpu.sh
```

On RTX 3090 24GB, expect OOM unless `MAX_INPUT_LENGTH` is lowered or the implementation is further optimized for long-context prefill.
