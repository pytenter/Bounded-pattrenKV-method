
# Unit Test Report

Commands:

```bash
/data/zypan/.local/share/mamba/envs/patternkv/bin/python -m pytest tests/test_kivi_gqa.py tests/test_kivi_gqa_cache.py tests/test_kivi_gqa_attention_parity.py -q
/data/zypan/.local/share/mamba/envs/patternkv/bin/python -m pytest tests/test_paper_v2_config.py tests/test_gsm8k_parser.py tests/test_gsm8k_stop_reason.py tests/test_gsm8k_paper_config.py tests/test_gsm8k_full_manifest.py tests/test_gsm8k_resume.py tests/test_gsm8k_summary.py -q
```

Results:

- KIVI GQA tests: `8 passed`
- Existing paper/GSM8K tests: `22 passed`
- `py_compile`: PASS for `models/llama_kivi.py`, `bench/paper_config.py`, `bench/bench_gsm8k_paper.py`, `scripts/summarize_gsm8k_paper_results.py`
